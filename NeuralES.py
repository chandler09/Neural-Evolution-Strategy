import math

import torch
import numpy as np
from model import Model
from torch import sum as t_sum
from ES_Template import ParametricES

np.set_printoptions(precision=4)
_type = torch.float
eps = 1e-16


class NeuralES(ParametricES):
    def __init__(self, p, n_part=16, n_path=1, k=None, opt='Adam', lr=1e-3, dec='TCH', n_hid=2, d_hid=1024,
                 verbose=True, seed=1, model=None, gpu=True):
        """
        p: test problem
        n_part: number of partition, should be a factor of the problem dimension; recommendation & default: 16
        n_path: rank of the low-rank covariance matrix of the Gaussian; recommendation & default: 1
        k: number of sampled preferences; recommendation & default: 5 * p.D // 16
        opt: optimizer of the set model; default: adam
        lr: learning rate of the optimizer; default: 1e-3
        dec: decomposition method: support TCH or PBI; default: TCH (smooth-TCH recommended for future development)
        n_hid: number of the hidden layers of the set model; default: 2
        d_hid: dimension of the hidden layers of the set model; default: 1024
        verbose: whether to output detailed information during optimization
        seed: random seed
        model: pretrained model; default: None
        gpu: whether to use GPU; only support single-GPU so far
        """
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        d_gauss = p.D // n_part

        nn = Model(p.n_var, d_gauss, p.xl, p.xu, n_path, p.n_obj, n_hid, d_hid,
                   torch.device("cuda" if gpu else "cpu"), _type, seed) if model is None else model

        super().__init__(p, nn, k, opt, lr, dec, verbose, seed, gpu)

        self._dn = d_gauss  # dimension of the decomposed normal distributions
        self._gauss_idx = np.arange(self._dn)
        self.m = np.zeros(self.D, float)
        self.p = np.zeros(self.D, float)
        self._update_mean()
        self._c = 2. / (self.D + 7.)
        self._c2 = np.sqrt(self._c * (2. - self._c))
        self._c = 1. - self._c

    def _update_mean(self):
        x = torch.zeros((1, self.nn.din), dtype=_type, device=self._dv)
        for idx, j in enumerate(range(0, self.D, self._dn)):  # np.random.permutation(np.arange(self.D)):  #
            k = j + self._gauss_idx
            theta = self.nn.forward(x.clone().detach().requires_grad_(False))
            x[:, k] = theta[0].detach()
            x[:, self.D + k] = 1.
        self.m = x[:, :self.D].cpu().numpy().flatten()

    def _sample(self, w=None):
        w = self._sample_pref() if w is None else w
        wx = np.tile(np.repeat(w, self.N, axis=0) if self.nn.training else w, math.ceil(self.D / self.M))
        n = len(wx)
        x = np.zeros((n, self.nn.din), float)
        if self.M > 1:
            x[:, :self.D] = wx[:, :self.D]
            x[:, self.D * 2:] = wx[:, self.D:]
        # upload once; all per-Gaussian-block work stays on the device
        x_ten = torch.from_numpy(x).to(device=self._dv, dtype=_type)
        zs = np.random.randn(n, self.D)
        zs_ten = torch.from_numpy(zs).to(device=self._dv, dtype=_type)
        thetas = []
        ctx = torch.enable_grad() if self.nn.training else torch.no_grad()
        with ctx:
            for i, j in enumerate(range(0, self.D, self._dn)):  # np.random.permutation(np.arange(self.D)):  #
                k = j + self._gauss_idx
                theta = self.nn.forward(x_ten.clone())  # clone: forward saves the input for backward
                with torch.no_grad():  # the noise mixing is not part of the loss; keep it off the autograd graph
                    m, s, zi = theta[0], theta[1], zs_ten[:, k]
                    y = zi
                    for v, v2 in zip(theta[2], theta[3]):
                        y = y + (torch.sqrt(1. + v2) - 1.) / torch.clamp(v2, min=eps) * t_sum(zi * v, 1, keepdim=True) * v
                    x_ten[:, k] = m + s * y
                    x_ten[:, self.D + k] = 1.
                thetas.append(theta)
        x = x_ten[:, :self.D].cpu().numpy()
        self._pop_np, self._pop_ten = x, x_ten[:, :self.D]  # let _train reuse the device copy
        return (x, thetas, zs, w) if self.nn.training else x

    def _train(self, pop, zs, w, theta):
        w = torch.as_tensor(w, dtype=_type, device=self._dv)
        if getattr(self, '_pop_np', None) is pop:  # same array _sample just produced: reuse its device copy
            x_ten = self._pop_ten
        else:
            x_ten = torch.from_numpy(np.ascontiguousarray(pop)).to(device=self._dv, dtype=_type)
        n_blocks = len(theta)
        if self.D % self._dn == 0:
            # fast path: the Gaussian blocks partition [0, D), so everything batches across blocks
            m = torch.stack([theta[k][0] for k in range(n_blocks)], 1)                    # (n, b, dn)
            s2 = torch.stack([theta[k][1].squeeze(-1) ** 2 for k in range(n_blocks)], 1)  # (n, b)
            dif = x_ten.view(x_ten.shape[0], n_blocks, self._dn) - m.detach()
            dm = -(w[:, None] * t_sum(dif * m, 2)).sum()
            v = torch.stack([torch.stack(theta[k][2], 1) for k in range(n_blocks)], 2)    # (n, p, b, dn)
            u2 = torch.stack([torch.stack(theta[k][3], 1) for k in range(n_blocks)], 2)   # (n, p, b, 1)
            tra = 2. * t_sum(u2, (1, 3)) + t_sum(u2 ** 2, (1, 3))                         # (n, b)
            rank = t_sum(torch.einsum('nbd,npbd->npb', dif, v).pow(2), 1)                 # (n, b)
            dc = w[:, None] * ((self._dn + tra) / 2 * s2 ** 2 - (t_sum(dif ** 2, 2) + rank) * s2)
            self._bp(dm + t_sum(dc))
        else:
            loss = []
            for k, j in enumerate(range(0, self.D, self._dn)):
                nid = j + self._gauss_idx  # normal distribution idx
                m, s2, vs, vvs = theta[k][0], theta[k][1].squeeze() ** 2, theta[k][2], theta[k][3]
                dif = x_ten[:, nid] - m.detach()
                dm = -w * t_sum(dif * m, 1)
                tra = t_sum(torch.hstack([vtv for vtv in vvs]), 1) * 2 + t_sum(torch.hstack([vtv ** 2 for vtv in vvs]), 1)
                rank_m = t_sum(torch.hstack([t_sum(dif * v, 1, keepdim=True) ** 2 for v in vs]), dim=1)
                dc = w * ((self._dn + tra) / 2 * s2 ** 2 - (t_sum(dif ** 2, 1) + rank_m) * s2)
                loss.append(dm + dc)
            self._bp(t_sum(torch.stack(loss)))


if __name__ == '__main__':
    from Problems import P1, P5, MO_UAV
    p = P1(1024)
    nes = NeuralES(p)
    nes.evolve(1e7, valid=False)  # whether validate the model at each epoch
