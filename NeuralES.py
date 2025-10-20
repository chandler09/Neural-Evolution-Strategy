import math

import torch
import numpy as np
from model import Model1, ModelR1
from model import Model
from torch import sum as t_sum
from torch import tensor
from ES_Template import ParametricES

np.set_printoptions(precision=4)
_type = torch.float
eps = 1e-16


class NeuralESR1(ParametricES):
    def __init__(self, p, d_gauss, n_path=0, k=None, opt='Adam', lr=1e-3, dec='TCH',
                 detail=False, verbose=True, seed=1, model=None, gpu=True):
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        n_path = n_path if n_path else 1  # p.n_obj - 1  # max(p.n_obj - 1, math.floor(np.log10(d_gauss)))

        nn = ModelR1(p.n_var, d_gauss, p.xl, p.xu, n_path, p.n_obj, torch.device("cuda" if gpu else "cpu"), _type,
                     seed=seed) if model is None else model

        super().__init__(p, nn, k, opt, lr, dec, detail, verbose, seed, gpu)

        self._dn = d_gauss  # dimension of the decomposed normal distributions
        self._gauss_idx = np.arange(self._dn)
        self.m = np.zeros(self.D, float)

    def _update_mean(self):
        x = torch.zeros((1, self.nn.din), dtype=_type, device=self._dv)
        for idx, j in enumerate(range(0, self.D, self._dn)):  # np.random.permutation(np.arange(self.D)):  #
            k = j + self._gauss_idx
            theta = self.nn.forward(x.clone().detach().requires_grad_(False), idx)
            x[:, k] = theta[0].detach()
            x[:, self.D + k] = 1.
        self.m = x[:, :self.D].cpu().numpy().flatten()

    def _sample(self, w=None):
        w = self._sample_pref() if w is None else w
        wx = np.tile(np.repeat(w, self.N, axis=0) if self.nn.training else w, math.ceil(self.D / self.M))
        n = len(wx)
        x, zs, thetas = np.zeros((n, self.nn.din), float), np.zeros((n, self.D), float), []
        if self.M > 1:
            x[:, :self.D] = wx[:, :self.D]
            x[:, self.D * 2:] = wx[:, self.D:]
        for i, j in enumerate(range(0, self.D, self._dn)):  # np.random.permutation(np.arange(self.D)):  #
            k = j + self._gauss_idx
            theta = self.nn.forward(tensor(x, dtype=_type, device=self._dv, requires_grad=self.nn.training), i)
            m, s, v = theta[0].cpu().detach().numpy(), theta[1].cpu().detach().numpy(), theta[2].cpu().detach().numpy()
            zs[:, k] = np.random.randn(n, self._dn) + np.random.randn(n, 1) * v
            x[:, k] = m + np.exp(s) * zs[:, k]
            x[:, self.D + k] = 1.
            thetas.append(theta)
        return (x[:, :self.D], thetas, zs, w) if self.nn.training else x[:, :self.D]

    def _train(self, pop, ws, f, theta):
        loss = []
        x_ten, f = tensor(pop, dtype=_type, device=self._dv), tensor(f, dtype=_type, device=self._dv)
        w_ten = tensor(ws, dtype=_type, device='cpu')
        for k, j in enumerate(range(0, self.D, self._dn)):
            nid = j + self._gauss_idx  # normal distribution idx
            m, s, v, d = theta[k][0], theta[k][1], theta[k][2], self._dn
            m0, v0, w = m.detach(), v.detach().cpu(), w_ten[:, nid]
            r = torch.sqrt(t_sum(v0 ** 2, 1, keepdim=True))
            z, r2 = v0 / r, r ** 2
            dm = t_sum((x_ten[:, nid] - m0) * m, 1, keepdim=True)
            w2 = t_sum(w ** 2, 1, keepdim=True)
            wz = t_sum(w * z, 1, keepdim=True)
            wz2 = wz ** 2
            ds = ((w2 - d + 1 - wz2) / 2 / (d - 1)).cuda() * s
            dv = ((r2 - d + 2) * wz2 - (r2 + 1) * w2) / 2 / r / (d - 1) * z + wz / r * w
            dv = t_sum(dv.cuda() * v, 1, keepdim=True)
            loss.append(-f * (dm + ds + dv).squeeze())
        self._bp(t_sum(torch.stack(loss)))


class NeuralES1(ParametricES):
    def __init__(self, p, d_gauss, n_path=0, k=None, opt='Adam', lr=1e-3, dec='TCH',
                 detail=False, verbose=True, seed=1, model=None, gpu=True, d_hid=1024):
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        n_path = n_path if n_path else 1  # p.n_obj - 1  # max(p.n_obj - 1, math.floor(np.log10(d_gauss)))

        nn = Model1(p.n_var, d_gauss, p.xl, p.xu, n_path, p.n_obj, torch.device("cuda" if gpu else "cpu"), _type,
                    seed=seed, d_hid=d_hid) if model is None else model

        super().__init__(p, nn, k, opt, lr, dec, detail, verbose, seed, gpu)

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
            theta = self.nn.forward(x.clone().detach().requires_grad_(False), idx)
            x[:, k] = theta[0].detach()
            x[:, self.D + k] = 1.
        self.m = x[:, :self.D].cpu().numpy().flatten()

    def _sample(self, w=None):
        w = self._sample_pref() if w is None else w
        wx = np.tile(np.repeat(w, self.N, axis=0) if self.nn.training else w, math.ceil(self.D / self.M))
        x, zs, thetas = np.zeros((len(wx), self.nn.din), float), np.random.randn(len(wx), self.D), []
        if self.M > 1:
            x[:, :self.D] = wx[:, :self.D]
            x[:, self.D * 2:] = wx[:, self.D:]
        for i, j in enumerate(range(0, self.D, self._dn)):  # np.random.permutation(np.arange(self.D)):  #
            k = j + self._gauss_idx
            theta = self.nn.forward(tensor(x, dtype=_type, device=self._dv, requires_grad=self.nn.training), i)
            m, s, zi = theta[0].cpu().detach().numpy(), theta[1].cpu().detach().numpy(), zs[:, k]
            y = zi
            for v, v2 in zip(theta[2], theta[3]):
                v, v2 = v.cpu().detach().numpy(), v2.cpu().detach().numpy()
                y += (np.sqrt(1 + v2) - 1) / np.maximum(v2, eps) * np.sum(zi * v, 1, keepdims=True) * v
            x[:, k] = m + s * y
            x[:, self.D + k] = 1.
            thetas.append(theta)
        return (x[:, :self.D], thetas, zs, w) if self.nn.training else x[:, :self.D]

    '''def _train(self, pop, zs, w, theta):
        loss = []
        x_ten, w = tensor(pop, dtype=_type, device=self._dv), tensor(w, dtype=_type, device=self._dv)
        z_ten = tensor(zs, dtype=_type, device=self._dv)
        for k, j in enumerate(range(0, self.D, self._dn)):
            nid = j + self._gauss_idx  # normal distribution idx
            m, s2, vs, vvs = theta[k][0], theta[k][1].detach().squeeze() ** 2, theta[k][2], theta[k][3]
            m0 = m.detach()
            dm = w * t_sum((m0 - x_ten[:, nid]) * m, 1)
            tra = t_sum(torch.hstack([vtv for vtv in vvs]), 1) + t_sum(torch.hstack([vtv ** 2 for vtv in vvs]), 1) / 2
            rank_m = t_sum(torch.hstack([t_sum((x_ten[:, nid] - m0) * v, 1, keepdim=True) ** 2 for v in vs]), dim=1)
            dc = w * (tra * s2 ** 2 - rank_m * s2)
            s = theta[k][1].squeeze()
            ds = w * (self._dn - t_sum(z_ten[:, nid] ** 2, 1)) / 2. / self._dn * s.detach() * s
            loss.append(dm + dc + ds)
        self._bp(t_sum(torch.stack(loss)))'''

    def _update_path(self):
        if self.M == 1:
            m0 = self.m
            self._update_mean()
            self.p = self._c * self.p + self._c2 * (self.m - m0)

    '''def _train(self, pop, zs, w, theta):
        loss = []
        x_ten, w = tensor(pop, dtype=_type, device=self._dv), tensor(w, dtype=_type, device=self._dv)
        for k, j in enumerate(range(0, self.D, self._dn)):
            nid = j + self._gauss_idx  # normal distribution idx
            m, s2, vs, vvs = theta[k][0], theta[k][1].squeeze() ** 2, theta[k][2], theta[k][3]
            m0 = m.detach()
            # dm = w * t_sum((m0 - x_ten[:, nid]) * m, 1)
            dm = w * t_sum(m * m / 2. - x_ten[:, nid] * m, 1)
            tra = t_sum(torch.hstack([vtv for vtv in vvs]), 1) + t_sum(torch.hstack([vtv ** 2 for vtv in vvs]), 1) / 2
            dif = x_ten[:, nid] - m0
            rank_m = t_sum(torch.hstack([t_sum(dif * v, 1, keepdim=True) ** 2 for v in vs]), dim=1)
            dc = w * ((self._dn / 2 + tra) * s2 ** 2 - (t_sum(dif ** 2, 1) + rank_m) * s2)
            loss.append(dm + dc)
        self._bp(t_sum(torch.stack(loss)))'''

    def _train(self, pop, zs, w, theta):
        loss = []
        x_ten, w = tensor(pop, dtype=_type, device=self._dv), tensor(w, dtype=_type, device=self._dv)
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


class NeuralES(ParametricES):
    def __init__(self, p, d_gauss, k=None, opt='Adam', lr=1e-3, dec='TCH',
                 detail=False, verbose=True, seed=1, model=None, gpu=True):
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        nn = Model(p.n_var, d_gauss, p.xl, p.xu, p.n_obj, torch.device("cuda" if gpu else "cpu"), _type,
                   seed=seed) if model is None else torch.load(model, weights_only=False)

        super().__init__(p, nn, k, opt, lr, dec, detail, verbose, seed, gpu)

        self._dn = d_gauss  # dimension of the decomposed normal distributions
        self._gauss_idx = np.arange(self._dn)

    def _sample(self, w=None):
        w = self._sample_pref() if w is None else w
        wx = np.tile(np.repeat(w, self.N, axis=0) if self.nn.training else w, math.ceil(self.D / self.M))
        x, thetas = np.zeros((len(wx), self.nn.din), float), []
        if self.M > 1:
            x[:, :self.D] = wx[:, :self.D]
            x[:, self.D * 2:] = wx[:, self.D:]
        zs = torch.tensor(np.random.randn(len(wx), self.D), dtype=_type, device=self._dv)
        for i, j in enumerate(range(0, self.D, self._dn)):  # np.random.permutation(np.arange(self.D)):  #
            k = j + self._gauss_idx
            theta = self.nn.forward(tensor(x, dtype=_type, device=self._dv, requires_grad=self.nn.training), i)
            m, s, A = theta[0].cpu().detach().numpy(), theta[1].cpu().detach().numpy(), theta[-1].detach()
            x[:, k] = m + s * (A @ zs[:, k].unsqueeze(-1)).cpu().detach().squeeze().numpy()
            x[:, self.D + k] = 1.
            thetas.append(theta)
        return (x[:, :self.D], thetas, zs, w) if self.nn.training else x[:, :self.D]

    def _train(self, pop, zs, w, theta):
        loss = []
        x_ten, w = tensor(pop, dtype=_type, device=self._dv), tensor(w, dtype=_type, device=self._dv)
        for k, j in enumerate(range(0, self.D, self._dn)):
            nid = j + self._gauss_idx  # normal distribution idx
            m, s2, C, A = theta[k][0], theta[k][1].detach().squeeze() ** 2, theta[k][2], theta[k][3]
            m0 = m.detach()
            dm = w * t_sum((m0 - x_ten[:, nid]) * m, 1)
            d = x_ten[:, nid] - m0
            dc = w * (t_sum(C ** 2, [1, 2]) / 2 * s2 ** 2 - t_sum((A @ d.unsqueeze(-1)).squeeze() ** 2, 1) * s2)
            s = theta[k][1].squeeze()
            ds = w * (self._dn - t_sum(zs[:, nid] ** 2, 1)) / 2. / self._dn * s.detach() * s
            loss.append(dm + dc + ds)
        self._bp(t_sum(torch.stack(loss)))


if __name__ == '__main__':
    from pymoo.problems.single.rosenbrock import Rosenbrock
    # from Problems import P1
    from Problems import MO_UAV, DeepSeaTreasure, FishWood, MountainCar
    # from Problems import TraPlan
    # from Problems import MuJoCo

    torch.autograd.set_detect_anomaly(True)
    # nes = NeuralES1(TraPlan(), 6)
    # nes = NeuralES1(MO_UAV(1), 6)
    p = MountainCar(1)
    nes = NeuralES1(p, p.d)
    nes.evolve(1000000, valid=False)
