import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import qr

PARAMS = {
    'proposals': (80, 20, 200),
    'screens': (5, 2, 10),
    'step': (0.22, 0.08, 0.4),
    'soft_mu': (65.0, 10.0, 300.0, 'log'),
    'time_fraction': (0.70, 0.5, 0.75)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
R = np.arange(M)

class Stop(Exception):
    pass


def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    xs = [Fraction(float(v)) for v in a[:, 0]]
    ys = [Fraction(float(v)) for v in a[:, 1]]
    rs = [Fraction(float(v)) for v in a[:, 2]]
    width = max(x+r for x, r in zip(xs, rs))-min(x-r for x, r in zip(xs, rs))
    height = max(y+r for y, r in zip(ys, rs))-min(y-r for y, r in zip(ys, rs))
    if width+height > 2:
        return -np.inf
    dist = np.sqrt((a[I, 0]-a[J, 0])**2+(a[I, 1]-a[J, 1])**2)
    if np.any(dist < a[I, 2]+a[J, 2]):
        return -np.inf
    return float(np.sum(a[:, 2]))


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started+params['time_fraction']*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(a)
    else:
        a = np.column_stack((np.tile(np.arange(7), 3)*0.14,
                             np.repeat(np.arange(3), 7)*0.14,
                             np.full(21, 0.069)))
        best = {'circles': a.tolist()}
        best_score = exact_score(a)

    def check(end=None):
        if time.monotonic() >= (deadline if end is None else min(end, deadline)):
            raise Stop()

    p = a[:, :2].copy()
    radii = a[:, 2].copy()
    lo = np.min(p-radii[:, None], axis=0)
    sides = np.max(p+radii[:, None], axis=0)-lo
    scale = 2.0/float(sides.sum())
    p = (p-lo)*scale
    z0 = np.r_[p[:, 0], p[:, 1], radii*scale, sides[0]*scale]
    objective = np.r_[np.zeros(42), -np.ones(21), 0.0]
    base = np.zeros((M+84, 64))
    base[R, 42+I] = -1
    base[R, 42+J] = -1
    base[M+K, K] = 1
    base[M+K, 42+K] = -1
    base[M+21+K, K] = -1
    base[M+21+K, 42+K] = -1
    base[M+21+K, 63] = 1
    base[M+42+K, 21+K] = 1
    base[M+42+K, 42+K] = -1
    base[M+63+K, 21+K] = -1
    base[M+63+K, 42+K] = -1
    base[M+63+K, 63] = -1
    cached = [None, None, None]

    def geometry(z):
        check()
        if cached[0] is not None and np.array_equal(z, cached[0]):
            return cached[1], cached[2]
        x, y, r, w = z[:21], z[21:42], z[42:63], z[63]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy)
        safe = np.maximum(d, 1e-25)
        g = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
        jac = base.copy()
        jac[R, I], jac[R, J] = dx/safe, -dx/safe
        jac[R, 21+I], jac[R, 21+J] = dy/safe, -dy/safe
        cached[:] = [z.copy(), g, jac]
        return g, jac

    def repair(z):
        if not np.isfinite(z).all() or np.min(z[42:63]) <= 0:
            return None
        q = np.column_stack((z[:21], z[21:42], z[42:63])).copy()
        dist = np.sqrt((q[I, 0]-q[J, 0])**2+(q[I, 1]-q[J, 1])**2)
        factor = min(1.0, float(np.min(dist/(q[I, 2]+q[J, 2]))))
        if factor <= 0:
            return None
        q[:, 2] *= factor*(1-2e-14)
        low = np.min(q[:, :2]-q[:, 2, None], axis=0)
        high = np.max(q[:, :2]+q[:, 2, None], axis=0)
        s = 2.0/float(np.sum(high-low))*(1-2e-14)
        q[:, :2] = (q[:, :2]-low)*s
        q[:, 2] *= s
        return q

    def consider(z):
        nonlocal best, best_score
        check()
        q = repair(z)
        if q is None:
            return -np.inf
        estimated = float(np.sum(q[:, 2]))
        if estimated > best_score:
            score = exact_score(q)
            if score > best_score:
                best, best_score = {'circles': q.tolist()}, score
        return estimated

    bounds = [(None, None)]*42+[(1e-6, 0.5)]*21+[(0.3, 1.7)]

    def polish(z, end):
        last = z.copy()
        def fun(v):
            check(end)
            return float(objective @ v), objective
        def callback(v):
            nonlocal last
            check(end)
            last = v.copy()
            consider(v)
        try:
            result = minimize(fun, z, jac=True, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': lambda v: geometry(v)[0],
                                           'jac': lambda v: geometry(v)[1]},
                              callback=callback,
                              options={'maxiter': 100, 'ftol': 5e-14})
            last = result.x
        except Stop:
            pass
        if time.monotonic() < deadline:
            consider(last)
        return last

    try:
        g, jac = geometry(z0)
        tight = np.flatnonzero(g < 1e-6)
        if len(tight) < 64:
            z0 = polish(z0, started+0.18*budget_s)
            g, jac = geometry(z0)
            tight = np.argsort(g)[:64]
        _, _, piv = qr(jac[tight].T, mode='economic', pivoting=True)
        active = tight[piv[:64]]
        for _ in range(4):
            check()
            g, jac = geometry(z0)
            delta = np.linalg.solve(jac[active], -g[active])
            if np.max(np.abs(delta)) > 0.025:
                break
            z0 = z0+delta
            consider(z0)
            if np.max(np.abs(delta)) < 1e-14:
                break
        g, jac = geometry(z0)
        inverse = np.linalg.solve(jac[active], np.eye(64))
        stress = objective @ inverse
        probability = 1/np.sqrt(np.maximum(stress, 0)+0.04)
        probability /= probability.sum()
        proposals = []
        for trial in range(int(params['proposals'])):
            check()
            selected = rng.choice(64, size=4+trial % 6, replace=False, p=probability)
            direction = inverse[:, selected] @ rng.uniform(0.4, 1.6, len(selected))
            size = np.max(np.sqrt(direction[:21]**2+direction[21:42]**2))
            if size < 1e-12:
                continue
            direction *= params['step']*(0.7+0.15*(trial % 5))/size
            target = z0+direction
            target[42:63] = np.clip(target[42:63], 0.025, 0.25)
            target[63] = np.clip(target[63], 0.4, 1.6)
            gg, _ = geometry(target)
            v = np.minimum(gg, 0)
            rank = float(objective @ target+params['soft_mu']*(v @ v))
            proposals.append((rank, target, direction))
        proposals.sort(key=lambda item: item[0])
        screened = []
        used = []
        screen_end = started+0.37*budget_s
        for _, target, direction in proposals:
            check()
            if len(screened) >= params['screens'] or time.monotonic() >= screen_end:
                break
            if used:
                similarity = np.abs(np.asarray(used) @ direction)
                norm = np.linalg.norm(np.asarray(used), axis=1)*np.linalg.norm(direction)
                if np.any(similarity > 0.92*norm):
                    continue
            used.append(direction)
            local_end = min(screen_end, time.monotonic()+0.065*budget_s)
            last = target.copy()
            def soft(v):
                check(local_end)
                gg, jj = geometry(v)
                neg = np.minimum(gg, 0)
                mu = params['soft_mu']
                return float(objective @ v+mu*(neg @ neg)), objective+2*mu*(jj.T @ neg)
            def soft_callback(v):
                nonlocal last
                check(local_end)
                last = v.copy()
            try:
                result = minimize(soft, target, jac=True, method='L-BFGS-B', bounds=bounds,
                                  callback=soft_callback,
                                  options={'maxiter': 85, 'ftol': 1e-11, 'maxls': 12})
                last = result.x
            except Stop:
                pass
            check()
            value = consider(last)
            screened.append((value, last.copy()))
        screened.sort(key=lambda item: item[0], reverse=True)
        if not screened:
            screened = [(0.0, z0)]
        for index, (_, target) in enumerate(screened[:2]):
            check()
            remaining = deadline-time.monotonic()
            end = time.monotonic()+remaining/(min(2, len(screened))-index)
            polish(target, end)
    except (Stop, np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass
    return best
