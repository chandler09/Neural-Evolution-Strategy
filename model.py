import numpy as np
import torch
from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import leaky_relu
import math


class Model(Module):
    def __init__(self, n, dn, xl, xu, n_path, m, n_layers=2, d_hid=1024, device="cuda", _type=torch.float, seed=1):
        torch.manual_seed(seed)
        super().__init__()
        self.din = n + math.ceil(n / m) * m
        self.dn = dn
        self._dv = device
        self._k = n_path

        n_layers = 1 if n_layers is None else n_layers - 1
        d_hid = 1024 if d_hid is None else max(self.dn, d_hid)
        self._in = Linear(self.din, d_hid, dtype=_type, device=device)
        self._hidden = ModuleList([Linear(d_hid, d_hid, dtype=_type, device=device) for _ in range(n_layers)])
        self._mu = Linear(d_hid, dn, dtype=_type, device=device)
        self._u = ModuleList([Linear(d_hid, dn, dtype=_type, device=device) for _ in range(n_path)])
        self._s = Linear(d_hid, 1, dtype=_type, device=device)

        self._range, self._center = [], []
        for j in range(0, n, dn):
            k = j + np.arange(dn)
            self._range.append(torch.tensor((xu[k] - xl[k]), device=self._dv, dtype=_type))
            self._center.append(torch.tensor((xl[k] + xu[k]) / 2., device=self._dv, dtype=_type))

    @staticmethod
    def _proj(u, v, u2):
        return torch.sum(v * u, dim=1, keepdim=True) / torch.clip(u2, 1e-12) * u

    def forward(self, x, _type=torch.float):  # output the mean and Cholesky matrix , center, scale
        mu, x = self.get_mean(x, _type, True)
        s = torch.abs(self._s(x))  # elu(self._s(x)) + 1.  #

        us, u2s = [], []
        for i in range(self._k):
            v = self._u[i](x)  # leaky_relu()  # tanhshrink(self._u[i](x))  # )  # self._u[i](x)  #
            u = v
            for j in range(i):
                u = u - self._proj(us[j], v, u2s[j])
            us.append(u)
            u2s.append(torch.sum(u ** 2, dim=1, keepdim=True))
        return mu, s, us, u2s

    def get_mean(self, x, _type=torch.float, return_x=False):
        x = leaky_relu(self._in(x))
        for hid in self._hidden:
            x = leaky_relu(hid(x))
        mu = self._mu(x)
        return mu if not return_x else (mu, x)

    def to(self, dv):
        self._dv = dv
        module = super(Model, self).to(dv)  # registered submodules (incl. _hidden/_u) move automatically

        if str(dv) == 'cpu':  # plain tensor lists are not modules: move them by hand
            module._range = [x_range.cpu() for x_range in self._range]
            module._center = [center.cpu() for center in self._center]
        return module
