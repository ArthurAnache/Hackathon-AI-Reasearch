import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'iterations': (105, 50, 160), 'jitter': (0.018, 0.002, 0.06), 'portfolio_fraction': (0.72, 0.45, 0.9)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    portfolio_end = start + 0.70 * budget_s * params['portfolio_fraction']
    rng = np.random.default_rng(seed)
    ai, aj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    saved = None

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def array(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((obj['centers'], obj['radii']))

    def perimeter(a):
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        if perimeter(a) > 2:
            return -float('inf')
        d = a[ai, :2]-a[aj, :2]
        if np.any(np.sqrt(d[:, 0]**2+d[:, 1]**2) < a[ai, 2]+a[aj, 2]):
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            best_score = exact(array(init))
        except (ValueError, TypeError, KeyError):
            pass

    def retain(a, model=None, z=None):
        nonlocal best, best_score, saved
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = a[ai, :2]-a[aj, :2]
        factor = min(1., float(np.min(np.sqrt(d[:, 0]**2+d[:, 1]**2)/(a[ai, 2]+a[aj, 2]))))
        if factor <= 0:
            return
        a[:, 2] *= factor*(1-3e-14)
        pf = np.sum(np.max(a[:, :2]+a[:, 2, None], axis=0)-np.min(a[:, :2]-a[:, 2, None], axis=0))
        if pf <= 0 or 2*np.sum(a[:, 2])/pf <= best_score:
            return
        a *= float(Fraction(2)/perimeter(a))*(1-3e-14)
        score = exact(a)
        if score > best_score:
            best_score = score
            best = {'centers': a[:, :2].tolist(), 'radii': a[:, 2].tolist()}
            if model is not None:
                saved = (model, z.copy())

    class Model:
        def __init__(self, counts):
            check()
            self.s = s = counts[0]
            self.m = m = sum(counts[1:])
            self.n = n = s+m
            self.ir = ir = n+m
            self.iw = iw = ir+n
            self.size = iw+1
            I, J = np.triu_indices(n, 1)
            U, V = np.triu_indices(m, 1)
            self.pi = pi = np.r_[I, U+s, np.arange(s, n)]
            self.pj = pj = np.r_[J, V+s, np.arange(s, n)]
            self.sign = np.r_[np.ones(len(I)), -np.ones(len(U)+m)]
            nr = len(pi)
            self.rr = np.arange(nr)
            self.maski = pi >= s
            self.maskj = pj >= s
            ids = np.arange(n)
            self.grad = np.zeros(iw+1)
            self.grad[ir:iw] = -np.r_[np.ones(s), np.full(m, 2.)]
            self.linear = np.zeros((nr+3*n, iw+1))
            A = self.linear
            A[self.rr, ir+pi] -= 1
            A[self.rr, ir+pj] -= 1
            off = nr
            A[off+ids, ids] = 1
            A[off+ids, ir+ids] = -1
            off += n
            A[off+ids, ids] = -1
            A[off+ids, ir+ids] = -1
            A[off+ids, iw] = 1
            off += n
            A[off+ids, ir+ids] = -1
            A[off+ids, iw] = -.5
            A[off+np.arange(s,n), np.arange(n,ir)] = -1
            self.bounds = [(None,None)]*n + [(0.,1.)]*m + [(1e-8,1.)]*n + [(.03,1.97)]
            z = np.zeros(iw+1)
            row = 0
            for level, count in enumerate(counts):
                check()
                for col in range(count):
                    check()
                    z[row] = (col-(count-1)/2)*.2
                    if level == 0 and s == 3:
                        z[row] *= 1.45
                    if level:
                        z[n+row-s] = level*.177
                    z[ir+row] = .095
                    row += 1
            y = np.r_[np.zeros(s), z[n:ir]]
            lo = np.min(z[:n]-z[ir:iw])
            hi = np.max(z[:n]+z[ir:iw])
            hh = np.max(y+z[ir:iw])
            scale = 2/(hi-lo+2*hh)
            z[:iw] *= scale
            z[:n] -= lo*scale
            z[iw] = (hi-lo)*scale
            self.initial = z
            self.cache = None

        def expand(self, z):
            a = np.column_stack((z[:self.n], np.r_[np.zeros(self.s), z[self.n:self.ir]], z[self.ir:self.iw]))
            b = a[self.s:].copy()
            b[:, 1] *= -1
            return np.concatenate((a, b))

        def geometry(self, z):
            check()
            if self.cache is not None and np.array_equal(z, self.cache[0]):
                return self.cache[1:]
            y = np.r_[np.zeros(self.s), z[self.n:self.ir]]
            dx = z[self.pi]-z[self.pj]
            dy = y[self.pi]-self.sign*y[self.pj]
            d = np.sqrt(dx*dx+dy*dy)
            self.cache = (z.copy(), dx, dy, d)
            return dx, dy, d

        def fun(self, z):
            dx, dy, d = self.geometry(z)
            v = self.linear@z
            v[:len(d)] += d
            v[-self.n:] += 1
            return v

        def jac(self, z):
            dx, dy, d = self.geometry(z)
            den = np.maximum(d, 1e-15)
            ux, uy = dx/den, dy/den
            A = self.linear.copy()
            rr = self.rr
            A[rr, self.pi] += ux
            A[rr, self.pj] -= ux
            mask = self.maski
            A[rr[mask], self.n+self.pi[mask]-self.s] += uy[mask]
            mask = self.maskj
            A[rr[mask], self.n+self.pj[mask]-self.s] -= self.sign[mask]*uy[mask]
            return A

        def solve(self, z, iterations):
            check()
            last = [z.copy()]
            count = [0]
            def callback(t):
                check()
                last[0] = t.copy()
                count[0] += 1
                if count[0] % 20 == 0:
                    retain(self.expand(t), self, t)
            try:
                result = minimize(lambda t: float(self.grad@t), z,
                                  jac=lambda t: self.grad, method='SLSQP',
                                  bounds=self.bounds,
                                  constraints={'type':'ineq', 'fun':self.fun, 'jac':self.jac},
                                  callback=callback,
                                  options={'maxiter':int(iterations), 'ftol':1e-12, 'disp':False})
                last[0] = result.x
            except Stop:
                retain(self.expand(last[0]), self, last[0])
                raise
            retain(self.expand(last[0]), self, last[0])
            return last[0]

    try:
        retain(np.array([[(i%7)*.2, (i//7)*.2, .0999999999999] for i in range(21)]))
        for counts in ((3,5,4), (5,4,4), (3,4,5), (5,5,3), (3,3,4,2)):
            check()
            if time.monotonic() > portfolio_end and saved is not None:
                break
            model = Model(counts)
            retain(model.expand(model.initial), model, model.initial)
            model.solve(model.initial, params['iterations'])
        attempt = 0
        while saved is not None and time.monotonic() < deadline-.045:
            check()
            model, z = saved
            z = z.copy()
            movable = np.arange(model.ir)
            choose = rng.choice(movable, min(7,len(movable)), replace=False)
            z[choose] += rng.normal(0., params['jitter']*(1+attempt%3), len(choose))
            z[model.n:model.ir] = np.maximum(z[model.n:model.ir], 1e-6)
            model.solve(z, 70)
            attempt += 1
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    return best
