import numpy as np
import torch
from torch import sum as t_sum
from pymoo.core.problem import Problem
from pymoo.util.reference_direction import get_partition_closest_to_points as pcp
from pymoo.util.ref_dirs import get_reference_directions as grd
from pymoo.problems.many.dtlz import generic_sphere
from numpy import transpose

condition = 2.
STEPS = 200


def elliptical(x, cd=condition, w=None):  # cond: condition number
    if w is None:
        w = cd ** (np.arange(x.shape[1]) / (x.shape[1] - 1.))
    return np.sum(w * x ** 2, axis=1)


def pf_sphere_tri(n_points):
    ref_dirs = grd("uniform", n_dim=3, n_partitions=pcp(n_points, 3))
    return generic_sphere(ref_dirs)


def f3_sphere(x):
    pix = .5 * np.pi * x[:, :2]
    sin1, cos1, sin2, cos2 = np.sin(pix[:, 0]), np.cos(pix[:, 0]), np.sin(pix[:, 1]), np.cos(pix[:, 1])
    return [cos1 * cos2, cos1 * sin2, sin1]


class UF(Problem):
    def __init__(self, D, M, seed=1):
        """
        D: problem dimension
        M: objective numbers
        """
        self.zl, self.zu = np.zeros(M, float), np.ones(M, float)
        super().__init__(n_var=D, n_obj=M, n_ieq_constr=0, xl=np.zeros(D, float), xu=np.ones(D, float))

        np.random.seed(seed)
        self.D, self.M = D, M

        self._rot = []
        for _ in range(M):
            self._rot.append(np.linalg.qr(np.random.randn(D - M + 1, D - M + 1))[0])
            while np.linalg.matrix_rank(self._rot[-1]) < D - M + 1:
                self._rot[-1] = np.linalg.qr(np.random.randn(D - M + 1, D - M + 1))[0]

        self._dv = torch.device("cuda") if torch.cuda.is_available() else None
        self._kernel_dim = -1        # dimension the cached kernel weights were built for
        self._kweights = None        # cached elliptical weights (CPU)
        self._rot_t = None           # cached rotation matrices (GPU)

    def _calc_pareto_front(self, n_pareto_points=100):
        raise NotImplementedError

    def _kernel(self, x):  # the difficulty of optimization
        raise NotImplementedError

    def _pf(self, x):
        raise NotImplementedError

    def _ps(self, x):  # the shape of the ps
        raise NotImplementedError

    def _kernel_w(self, dim):
        # cache the elliptical weights cd ** (j / (dim - 1)); probing with the identity
        # recovers them from _kernel regardless of the subclass's condition number
        if self._kernel_dim != dim or self._kweights is None:
            self._kernel_dim = dim
            self._kweights = self._kernel(np.eye(dim))
        return self._kweights

    def _evaluate(self, x, out, *args, **kwargs):
        d = x[:, self.n_obj - 1:] - self._ps(x)
        pf = self._pf(x)
        n, dm = d.shape
        if self._dv is not None and n >= 256:
            # GPU path: one upload, cached rotation matrices, fused elliptical kernel
            if self._rot_t is None:
                self._rot_t = [torch.from_numpy(np.ascontiguousarray(r)).to(self._dv, torch.float32)
                               for r in self._rot]
                self._ew_t = torch.from_numpy(self._kernel_w(dm).astype(np.float32)).to(self._dv)
            dt = torch.from_numpy(np.ascontiguousarray(d, dtype=np.float32)).to(self._dv, non_blocking=True)
            ks = [t_sum(self._ew_t * torch.square(dt @ r), dim=1, keepdim=True) for r in self._rot_t]
            k = (torch.stack(ks, 1) / dm).to("cpu", torch.float64).numpy().reshape(n, -1)
            out['F'] = np.vstack(pf).T + k
        else:
            w = self._kernel_w(dm)
            out['F'] = np.vstack([f + self._kernel(d.dot(self._rot[i]), w=w) / dm
                                  for i, f in enumerate(pf)]).T

    def ps(self, n_points=100):
        raise NotImplementedError


class UFBi(UF):
    def __init__(self, D, M=2):
        super().__init__(D, M)
        self.xl[M-1:], self.xu[M-1:] = -1., 1.

    def _calc_pareto_front(self, n_pareto_points=100):  # default implementation is sphere (i.e., UF8)
        f0 = np.linspace(0., 1., n_pareto_points, True)
        return np.vstack((f0, 1. - np.sqrt(f0))).T

    def _pf(self, x):  # default implementation is sphere (i.e., UF8)
        return x[:, 0], 1. - np.sqrt(x[:, 0])

    def _kernel(self, x, w=None):  # the difficulty of optimization
        return elliptical(x, 100., w=w)

    def _ps(self, x):
        raise NotImplementedError

    def ps(self, n_points=100):
        x = np.zeros((n_points, self.n_var), float)
        x[:, 0] = np.linspace(0., 1., n_points, True)
        x[:, self.n_obj - 1:] = self._ps(x)  # .dot(np.linalg.inv(self._rot))
        return x

    def _merge(self, even, odd):
        y = np.zeros((len(even), self.n_var - 1), float)
        y[:, np.arange(0, self.n_var - 1, 2)] = even
        y[:, np.arange(1, self.n_var - 1, 2)] = odd
        return y

    def _split(self, i=2):
        return np.arange(self.n_var - 1) / (self.n_var - i)

    @property
    def _flip(self):
        return np.full(self.n_var - 1, -1, float) ** np.arange(self.n_var - 1)


class UFTri(UF):
    def __init__(self, D, M=3):
        super().__init__(D, M)
        self.xl[M-1:], self.xu[M-1:] = -1., 1.

    def _calc_pareto_front(self, n_pareto_points=100):  # default implementation is sphere (i.e., UF8)
        return pf_sphere_tri(n_pareto_points)

    def _kernel(self, x, w=None):  # the difficulty of optimization
        return elliptical(x, 2., w=w)

    def _pf(self, x):  # default implementation is sphere (i.e., UF8)
        return f3_sphere(x)

    def _ps(self, x):
        raise NotImplementedError

    def ps(self, n_points=100):
        n = round(np.sqrt(n_points))
        x = np.zeros((n * n, self.n_var), float)
        X, Y = np.meshgrid(np.linspace(0., 1., n, True), np.linspace(0., 1., n, True))
        x[:, :2] = np.vstack([X.ravel(), Y.ravel()]).T
        x[:, self.n_obj - 1:] = self._ps(x)  # .dot(np.linalg.inv(self._rot))
        return x

    @property
    def _idx(self):
        # return np.cos(np.pi * np.arange(0, self.n_var - 2) / (self.n_var - 3))
        return 1. - np.arange(0, self.n_var - 2) / (self.n_var - 2)


class P4(UFBi):
    def __init__(self, D=30):
        UFBi.__init__(self, D)

    def _ps(self, x):
        return 2. * transpose([x[:, 0]]) ** (.5 + 1.5 * self._split()) - 1.


class P3(UFBi):
    def __init__(self, D=30):
        UFBi.__init__(self, D)

    def _ps(self, x):
        idx = self._split()
        x1 = 2. * transpose([x[:, 0]]) - 1.
        return (1. - idx) * x1 ** 3. + idx * x1  # () * self._flip


class P1(UFBi):
    def __init__(self, D=30):
        UFBi.__init__(self, D)

    def _ps(self, x):
        x1 = transpose([x[:, 0]])
        return x1 * self._merge(np.cos(2. * np.pi * x1), np.sin(2. * np.pi * x1)) * .75 * (1. - self._split())


class P2(UFBi):
    def __init__(self, D=30):
        UFBi.__init__(self, D)

    def _ps(self, x):
        x1 = transpose([x[:, 0]])
        idx = self._split()
        y = np.zeros((len(x), self.n_var - 1), float)
        even, odd = np.arange(0, self.n_var - 1, 2), np.arange(1, self.n_var - 1, 2)
        y[:, even] = (.5 - (2. * x1 - 1.) ** 2.) * (1. - idx[even])
        y[:, odd] = x1 ** (1. + idx[odd]) - .5
        return y


class P5(UFTri):
    def __init__(self, D=30):
        UFTri.__init__(self, D)

    def _ps(self, x):
        x1, x2 = np.transpose([x[:, 0]]), np.transpose([x[:, 1]])
        return np.cos(np.pi * (x1 + x2) / 2) * np.cos(np.pi * np.abs(x1 - x2)) * self._idx / 2.


class P6(UFTri):
    def __init__(self, D=30):
        UFTri.__init__(self, D)

    def _ps(self, x):
        x1, x2 = np.transpose([x[:, 0]]), np.transpose([x[:, 1]])
        return (x1 ** 2. * (1. - x2) + (1. - x1 ** 2.) * x2 - .5) * self._idx


class P7(UFTri):
    def __init__(self, D=30):
        UFTri.__init__(self, D)

    def _ps(self, x):
        x1, x2 = (2. * np.transpose([x[:, 0]]) - 1.), np.transpose([x[:, 1]])
        return (x2 * (1. - x1 ** 2.) + (1. - x2) * x1 ** 2. / 2. - .5) * self._idx


class P8(UFTri):
    def __init__(self, D=30):
        UFTri.__init__(self, D)

    def _ps(self, x):
        x1, x2 = (2. * np.transpose([x[:, 0]]) - 1.), (2. * np.transpose([x[:, 1]]) - 1.)
        return (x1 ** 2 - x2 ** 2) / 2. * self._idx


class P9(UFTri):
    def __init__(self, D=30):
        UFTri.__init__(self, D)

    def _ps(self, x):
        x1, x2 = np.transpose([x[:, 0]]), np.transpose([x[:, 1]])
        return (x2 * np.cos(np.pi * x1) + (1. - x2) * np.cos(np.pi * (1. - x1)) / 2.) * self._idx


class MO_UAV(Problem):
    def __init__(self, idx=1, r=100, rr=10, n_threat=10, n=30, zu=np.array([1., 1., 1.])):
        from others.UAV import createmodel, Terrain
        terrain = createmodel(r=r, rr=rr, num_threats=n_threat,
                              rng=np.random.RandomState(idx))
        terrain['n'] = n  # 48
        terrain['J_pen'] = 1e4
        self._p = Terrain(terrain, idx)
        self.M = 3
        self.D = self._p.dim
        super().__init__(n_var=self.D, n_obj=self.M, n_ieq_constr=0, xl=self._p.lb, xu=self._p.ub)
        self.zl = np.zeros(self.M, float)
        self.zu = zu  # np.ones(self.M, float)  # None
        # print(self.xl, self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = self._p.func(x)
        # print(np.min(out['F'], axis=0), np.max(out['F'], axis=0))

    def name(self):
        return f'MO-UAV{self._p.problem_id}'


class TraPlan(Problem):
    def __init__(self, D=60, seed=1):
        self.D, self.M = D, 3  # no points in a trajectory
        from others.Rover import get_rover_fn
        f, bounds = get_rover_fn(dim=D)
        self._f = f
        super().__init__(n_var=D, n_obj=self.M, n_ieq_constr=0, xl=bounds[0], xu=bounds[1])
        self.zl, self.zu = np.zeros(self.n_obj, float), np.array((5, 2, 1.5), float)
        # np.full(self.n_obj, np.inf, float)

        np.random.seed(seed)
        self._rot = np.linalg.qr(np.random.randn(D, D))[0]
        while np.linalg.matrix_rank(self._rot) < D:
            self._rot = np.linalg.qr(np.random.randn(D, D))[0]

    def _evaluate(self, x, out, *args, **kwargs):
        f = self._f(x)  # .dot(self._rot)
        out['F'] = f
