"""Abstract class for Parametric ES"""
import copy
import math

import torch
import numpy as np
import time
from pymoo.decomposition.asf import ASF
from my_utils import PBI
from pymoo.util.ref_dirs import get_reference_directions as grd  # preferences
from pymoo.util.reference_direction import get_partition_closest_to_points as pcp
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV

np.set_printoptions(precision=4)
_type = torch.float
eps = 1e-16


class ParametricES:
    def __init__(self, p, nn, k=None, opt='Adam', lr=1e-3, dec='TCH', detail=False, verbose=True, seed=1, gpu=True):
        self._seed = seed
        self._dv = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        self._p = p
        self.M = p.n_obj
        self.D = p.n_var
        self._xl, self._xu = p.xl, p.xu
        self._verbose = verbose
        self._detail = detail
        self._t = 0

        # self.K = (2 ** (1 + self.M) * 5) if k is None else k  # 2 * round(self.M ** 2)
        self.K = 5 * self.D // 16 if k is None else k  # 2 * round(self.M ** 2)  # number of sampled preferences
        if self.M > 1:
            self.N = 2 + round(1.5 * np.log(self.D))  # number of sampled solutions per preference
            # self.N = 4 + math.floor(3 * np.log(self.D))  # number of sampled solutions per preference
        else:
            self.N = 4 + math.floor(3 * np.log(self.D))
        self._w = np.log(self.N // 2. + .5) - np.log(1. + np.arange(self.N))

        self.nn = nn if not isinstance(nn, str) else torch.load(nn, weights_only=False)
        self.nn.train()
        self._opt = torch.optim.Adam(self.nn.parameters(), lr=lr, amsgrad=True) if opt == 'Adam' else \
            torch.optim.SGD(self.nn.parameters(), lr=lr, momentum=.9, nesterov=True)  # )
        if torch.cuda.is_available() and gpu:
            self.nn.to(self._dv)
            self.nn.cuda(self._dv)

        if self.M > 1:
            if dec == '':
                self._dec = PBI(theta=5.) if p.n_obj > 2 else ASF(eps=1e-16)
            else:
                self._dec = PBI(theta=5.) if dec == 'PBI' else ASF(eps=1e-16)
            # self._z = np.array(self._p.zl) if problem.pareto_front() is not None else np.full(self.M, np.inf, float) #
            self._z = self._p.zl if self._p.zl is not None else np.full(self.M, np.inf, float)  #
            self._r = grd("uniform", n_dim=self.M, n_partitions=pcp(100 if self.M < 3 else 300, self.M))
            self._x, self._f = np.zeros((len(self._r), self.D), float), np.full((len(self._r), self.M), 1e32, float)

            pf = p.pareto_front(300 if self.M == 2 else 990)
            self._igd = IGD(pf) if pf is not None else (HV(ref_point=p.zu * 1.1) if p.zu is not None else None)
            self._hist = {'igd': [], 'igd0': [], 'igd1': [], 'T': -1, 'nnv': None}  # 0 for opt, 1 for eval
        else:
            self._x, self._f = np.zeros(self.D, float), np.inf
            self._hist = {'f_opt': [], 'x_opt': [], 'f_cub': [], 'm': [], 'T': -1}

    @classmethod
    def name(cls):
        return cls.__name__

    def _evaluate(self, pop, w):
        pop_r = np.clip(pop, self._xl, self._xu)
        fr = self._p.evaluate(pop_r)  # .flatten()
        g = 1e20 * (np.any(np.logical_or(pop < self._xl, pop > self._xu), axis=1) + np.linalg.norm(pop_r - pop, axis=1))
        ranks = np.zeros_like(g, float)
        if self.M > 1:
            self._z = np.minimum(self._z, np.min(fr, axis=0))
            for i in range(len(w)):
                idx = np.arange(i * self.N, (i + 1) * self.N)
                g[idx] += self._dec.do(fr[idx], w[i], utopian_point=self._z)
                ranks[idx[np.argsort(g[idx])]] = self._w
        else:
            fr = fr.flatten()
            ranks[np.argsort(g + fr)] = self._w
        return fr, ranks, pop_r

    def _sample(self, w=None):
        raise NotImplementedError

    def sample(self, w):
        self.nn.eval()
        xs = self._sample(w)
        self.nn.train()
        return xs

    def _sample_pref(self):  # dup: duplicate
        if self.M == 1:
            return None
        w = np.random.rand(self.K, self.M - 1)  # np.random.dirichlet(np.ones(self.M, float), n_pre)
        return np.hstack((w, 1. - np.sum(w, axis=1, keepdims=True)))  # sample preferences

    def _train(self, pop, zs, w, theta):
        raise NotImplementedError

    def _bp(self, loss):
        self._opt.zero_grad()
        loss.backward()
        if isinstance(self._opt, torch.optim.SGD):
            torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 100.)
        self._opt.step()

    def evolve(self, n_eval, igd_tar=None, f_tar=None, valid=True, skip=10):
        self._hist['T'] = n_eval
        tik, evl = time.time(), 0
        igd_val_best = np.inf if hasattr(self, '_igd') and isinstance(self._igd, IGD) else 0.
        while evl < n_eval:

            pop, theta, zs, pref = self._sample()
            fr, ranks, pop_r = self._evaluate(pop, pref)

            # if not evl:
            # print(self._igd.ref_point)
            if self._igd is None:
                self._igd = HV(ref_point=np.clip(np.max(fr, axis=0), 1., 10.) * 2.)
                print(self._igd.ref_point)

            self._train(pop, zs, ranks, theta)
            evl += len(pop_r)

            if self.M > 1:
                g = self._dec.do(fr, self._r, utopian_point=self._z, _type="many_to_many")
                elite = np.argmin(g, axis=0)
                pick = np.where(np.diag(g[elite]) < self._dec.do(self._f, self._r, utopian_point=self._z,
                                                                 _type="one_to_one"))[0]
                self._x[pick], self._f[pick] = pop_r[elite[pick]], fr[elite[pick]]

                if not self._t % skip:
                    igd, igd0 = self._igd(fr), self._igd(self._f)
                    self._hist['igd'].append(igd)
                    self._hist['igd0'].append(igd0)
                    igd1 = 0.
                    if valid:
                        igd1 = self._igd(self._p.evaluate(np.clip(self.sample(self._r), self._xl, self._xu)))
                        self._hist['igd1'].append(igd1)
                        if (igd1 < igd_val_best and isinstance(self._igd, IGD)) or \
                                (igd1 > igd_val_best and isinstance(self._igd, HV)):
                            self._hist['nnv'] = copy.deepcopy(self.nn)
                            self._hist['nnv'].to('cpu'), self._hist['nnv'].eval()
                            igd_val_best = igd1
                    if self._verbose:
                        tok = time.time() - tik
                        print(f'{self._t}, EVA:{evl:.1e}, IGD: CUR {igd:.4f} | OPT {igd0:.4f} | EVL {igd1:.4f}, '
                              f'MIN:{self._z if self._p.zl is None else np.min(fr, axis=0)}, {tok:.2f}s')
                    tik = time.time()
            else:
                pick = np.argmin(fr)
                if fr[pick] < self._f:
                    self._x, self._f = pop_r[pick], fr[pick]
                f_cub = fr[pick]
                self._hist['x_opt'].append(self._x)
                self._hist['f_opt'].append(self._f)
                self._hist['f_cub'].append(f_cub)
                if self._verbose and not self._t % (skip if self.D < 100000 else 1):
                    f_med, tok = np.median(fr), time.time() - tik
                    if np.abs(f_med) > 1e6 or np.abs(self._f) < 1e-4:
                        print(f'{self._t}, EVA:{evl}, OPT:{self._f:.2e}, CUB:{f_cub:.1e}, MED:{f_med:.1e}, '
                              f'STD:{np.std(pop):.4f}, {tok:.2f}s')
                    else:
                        print(f'{self._t}, EVA:{evl}, OPT:{self._f:.4f}, CUB:{f_cub:.4f}, MED:{f_med:.4f}, '
                              f'STD:{np.std(pop):.4f}, {tok:.2f}s')
                    tik = time.time()

            self._t += 1

            if self.M > 1 and self._t and igd_tar is not None:
                if self._hist['igd0'][-1] < igd_tar:
                    self._hist['T'] = evl
                    break
            elif self._t and f_tar is not None:
                if self._hist['f_opt'][-1] < f_tar:
                    self._hist['T'] = evl
                    break

        self.nn.eval()
        self.nn = self.nn.to('cpu')
        torch.cuda.empty_cache()

        return self._hist, self._x, self._f, self.nn
