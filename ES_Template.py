"""Abstract class for Parametric ES"""
import copy
import math

import torch
import numpy as np
import time
from pymoo.decomposition.asf import ASF
from pymoo.core.decomposition import Decomposition
from pymoo.util.ref_dirs import get_reference_directions as grd  # preferences
from pymoo.util.reference_direction import get_partition_closest_to_points as pcp
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV

np.set_printoptions(precision=4)
eps = 1e-16


# Currently, the algorithm and the Problem templates only support bi- and tri-objective problems, 
# though the mathematical formulation of Neural-ES is general. 
class ParametricES:
    def __init__(self, p, nn, k=None, opt='Adam', lr=1e-3, dec='TCH', verbose=True, seed=1, gpu=True):
        self._seed = seed
        self._dv = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        self._p = p
        self.M = p.n_obj
        self.D = p.n_var
        self._xl, self._xu = p.xl, p.xu
        self._verbose = verbose
        self._t = 0

        self.K = 5 * self.D // 16 if k is None else k  # 2 * round(self.M ** 2)  # number of sampled preferences
        self.N = 2 + round(1.5 * np.log(self.D))  # number of sampled solutions per preference
        self._w = np.log(self.N // 2. + .5) - np.log(1. + np.arange(self.N))

        self.nn = nn if not isinstance(nn, str) else torch.load(nn, weights_only=False)
        self.nn.train()
        self._opt = torch.optim.Adam(self.nn.parameters(), lr=lr, amsgrad=True) if opt == 'Adam' else \
            torch.optim.SGD(self.nn.parameters(), lr=lr, momentum=.9, nesterov=True)  # )
        if torch.cuda.is_available() and gpu:
            self.nn.to(self._dv)

        if dec == '':
            self._dec = PBI(theta=5.) if p.n_obj > 2 else ASF(eps=1e-16)
        else:
            self._dec = PBI(theta=5.) if dec == 'PBI' else ASF(eps=1e-16)
        self._z = self._p.zl if self._p.zl is not None else np.full(self.M, np.inf, float)  # utopian point
        self._r = grd("uniform", n_dim=self.M, n_partitions=pcp(100 if self.M < 3 else 300, self.M))  # preferences
        # solution archive
        self._x, self._f = np.zeros((len(self._r), self.D), float), np.full((len(self._r), self.M), 1e32, float)
        
        pf = p.pareto_front(300 if self.M == 2 else 990)
        self._igd = IGD(pf) if pf is not None else (HV(ref_point=p.zu * 1.1) if p.zu is not None else None)
        self._hist = {'igd': [],  # IGD curve of every-epoch population
                      'igd0': [],  # IGD curve of the solution archive
                      'igd1': [],  # curve of the validation IGD
                      'T': -1,  # number of evaluations consumed 
                      'nnv': None  # model with the lowest validation IGD so-far
                      }

    @classmethod
    def name(cls):
        return cls.__name__

    def _evaluate(self, pop, w):
        pop_r = np.clip(pop, self._xl, self._xu)
        fr = self._p.evaluate(pop_r)  # .flatten()
        g = 1e20 * (np.any(np.logical_or(pop < self._xl, pop > self._xu), axis=1) + np.linalg.norm(pop_r - pop, axis=1))
        ranks = np.zeros_like(g, float)
        self._z = np.minimum(self._z, np.min(fr, axis=0))
        # one vectorized one-to-one decomposition with per-row weights instead of len(w) small calls
        gg = self._dec.do(fr, np.repeat(w, self.N, axis=0), utopian_point=self._z) + g
        gg = gg.reshape(len(w), self.N)
        order = np.argsort(gg, axis=1)                            # (len(w), N)
        ranks = np.empty_like(gg)
        ranks[np.arange(len(w))[:, None], order] = self._w        # rank weights per group
        ranks = ranks.reshape(-1)
        return fr, ranks, pop_r

    def _sample(self, w=None):
        raise NotImplementedError

    def sample(self, w):
        self.nn.eval()
        xs = self._sample(w)
        self.nn.train()
        return xs

    def _sample_pref(self):  # dup: duplicate
        return np.random.dirichlet(np.ones(self.M, float), self.K)

    def _train(self, pop, zs, w, theta):
        raise NotImplementedError

    def _bp(self, loss):
        self._opt.zero_grad()
        loss.backward()
        if isinstance(self._opt, torch.optim.SGD):
            torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 100.)
        self._opt.step()

    def evolve(self, n_eval, valid=False, igd_tar=None, skip=10):
        """
        n_eval: evaluation budget
        valid: whether to evaluate the validation IGD curve, slower the training if True
        igd_tar: early stop if the validation IGD is below this threshold
        skip: evaluation frequency for logging
        """
        self._hist['T'] = n_eval
        tik, evl = time.time(), 0
        igd_val_best = np.inf if hasattr(self, '_igd') and isinstance(self._igd, IGD) else 0.
        while evl < n_eval:

            pop, theta, zs, pref = self._sample()
            fr, ranks, pop_r = self._evaluate(pop, pref)

            if self._igd is None:
                self._igd = HV(ref_point=np.clip(np.max(fr, axis=0), 1., 10.) * 2.)
                print(self._igd.ref_point)

            self._train(pop, zs, ranks, theta)
            evl += len(pop_r)

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
                        self._hist['nnv'].to('cpu')
                        self._hist['nnv'].eval()
                        igd_val_best = igd1
                if self._verbose:
                    print(f'{self._t}, {evl / n_eval * 100:.1f}%, IGD: CUR {igd:.4f} | OPT {igd0:.4f} | VAL {igd1:.4f}, '
                          f'F_MIN:{self._z if self._p.zl is None else np.min(fr, axis=0)}, '
                          f'{(time.time() - tik) / 60:.2f}min')
            self._t += 1
            if self._t and igd_tar is not None:
                if self._hist['igd0'][-1] < igd_tar:
                    self._hist['T'] = evl
                    break

        self.nn.eval()
        self.nn = self.nn.to('cpu')
        torch.cuda.empty_cache()

        return self._hist, self._x, self._f, self.nn


class PBI(Decomposition):

    def __init__(self, theta=5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.theta = theta

    def _do(self, F, weights, **kwargs):
        norm = np.linalg.norm(weights, axis=1)
        F = F - self.utopian_point
        d1 = (F * weights).sum(axis=1) / norm
        d2 = np.linalg.norm(F - (d1[:, None] * weights / norm[:, None]), ord=1, axis=1)
        return d1 + self.theta * d2
