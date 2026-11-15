import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq, qr

PARAMS = {
    "release_groups": (3, 3, 5),
    "path_length": (0.19, 0.04, 0.35),
    "trials": (5, 1, 12),
    "fraction": (0.70, 0.5, 0.75),
}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['fraction']) * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    ids = np.arange(n)
    best = init
    best_score = -float('inf')

    def exact_score(data):
        try:
            c = np.asarray(data['circles'], dtype=np.float64)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            xs = list(map(lambda x: Fraction(float(x)), c[:, 0]))
            ys = list(map(lambda x: Fraction(float(x)), c[:, 1]))
            rs = list(map(lambda x: Fraction(float(x)), c[:, 2]))
            w = max(x+r for x, r in zip(xs, rs)) - min(x-r for x, r in zip(xs, rs))
            h = max(y+r for y, r in zip(ys, rs)) - min(y-r for y, r in zip(ys, rs))
            if w+h > 2:
                return -float('inf')
            for a, b in itertools.combinations(c, 2):
                time.monotonic()
                d = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
                if d < a[2]+b[2]:
                    return -float('inf')
            return float(np.sum(c[:, 2]))
        except (ValueError, KeyError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)
        c0 = np.asarray(init['circles'], dtype=float).copy()
    else:
        c0 = np.zeros((n, 3))
        c0[:, 0] = (ids % 5) * 0.195 + 0.09
        c0[:, 1] = (ids // 5) * 0.195 + 0.09
        c0[:, 2] = 0.089

    def check(end=None):
        if time.monotonic() >= (deadline if end is None else min(end, deadline)):
            raise StopSearch

    def constraints(z):
        c = z[:63].reshape(n, 3)
        dx = c[ii, 0]-c[jj, 0]
        dy = c[ii, 1]-c[jj, 1]
        d = np.sqrt(dx*dx+dy*dy)
        return np.concatenate((d-c[ii, 2]-c[jj, 2],
            c[:, 0]-c[:, 2], z[63]-c[:, 0]-c[:, 2],
            c[:, 1]-c[:, 2], 2-z[63]-c[:, 1]-c[:, 2]))

    def jacobian(z):
        c = z[:63].reshape(n, 3)
        dx = c[ii, 0]-c[jj, 0]
        dy = c[ii, 1]-c[jj, 1]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-100)
        a = np.zeros((m+84, 64))
        a[rows, 3*ii] = dx/d
        a[rows, 3*jj] = -dx/d
        a[rows, 3*ii+1] = dy/d
        a[rows, 3*jj+1] = -dy/d
        a[rows, 3*ii+2] = -1
        a[rows, 3*jj+2] = -1
        a[m+ids, 3*ids] = 1
        a[m+ids, 3*ids+2] = -1
        a[m+n+ids, 3*ids] = -1
        a[m+n+ids, 3*ids+2] = -1
        a[m+n+ids, 63] = 1
        a[m+2*n+ids, 3*ids+1] = 1
        a[m+2*n+ids, 3*ids+2] = -1
        a[m+3*n+ids, 3*ids+1] = -1
        a[m+3*n+ids, 3*ids+2] = -1
        a[m+3*n+ids, 63] = -1
        return a

    def consider(z):
        nonlocal best, best_score
        c = np.asarray(z[:63]).reshape(n, 3).copy()
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        if float(c[:, 2].sum()) < best_score-1e-9:
            return
        d = np.sqrt((c[ii, 0]-c[jj, 0])**2 + (c[ii, 1]-c[jj, 1])**2)
        f = min(1.0, float(np.min(d/(c[ii, 2]+c[jj, 2]))))
        c[:, 2] *= f*(1-3e-15)
        c[:, 0] -= np.min(c[:, 0]-c[:, 2])
        c[:, 1] -= np.min(c[:, 1]-c[:, 2])
        p = (np.max(c[:, 0]+c[:, 2])-np.min(c[:, 0]-c[:, 2]) +
             np.max(c[:, 1]+c[:, 2])-np.min(c[:, 1]-c[:, 2]))
        c *= min(1.0, 2.0/p)*(1-3e-15)
        data = {'circles': c.tolist()}
        score = exact_score(data)
        if score > best_score:
            best, best_score = data, score

    c0[:, 0] -= np.min(c0[:, 0]-c0[:, 2])
    c0[:, 1] -= np.min(c0[:, 1]-c0[:, 2])
    width = np.max(c0[:, 0]+c0[:, 2])
    z0 = np.r_[c0.ravel(), width]
    grad = np.zeros(64)
    grad[2:63:3] = -1
    bounds = [(None, None)]*64
    for k in range(n):
        time.monotonic()
        bounds[3*k+2] = (1e-6, 0.5)
    bounds[63] = (0.1, 1.9)

    def polish(z, seconds, forced=None):
        end = min(deadline, time.monotonic()+seconds)
        last = z.copy()
        def fun(v):
            check(end)
            return float(grad @ v)
        def callback(v):
            nonlocal last
            check(end)
            last = v.copy()
            if -float(grad @ v) > best_score and np.min(constraints(v)) > -2e-7:
                consider(v)
        cons = [{'type':'ineq', 'fun':constraints, 'jac':jacobian}]
        if forced is not None:
            cons.append({'type':'eq',
                         'fun':lambda v: constraints(v)[forced:forced+1],
                         'jac':lambda v: jacobian(v)[forced:forced+1]})
        try:
            result = minimize(fun, z, jac=lambda v: grad, method='SLSQP',
                bounds=bounds, constraints=cons, callback=callback,
                options={'maxiter':90, 'ftol':2e-13, 'disp':False})
            last = result.x
        except StopSearch:
            pass
        if time.monotonic() < deadline:
            consider(last)
        return last

    try:
        check()
        z0 = polish(z0, min(0.12, budget_s*0.07))
        check()
        g0 = constraints(z0)
        a0 = jacobian(z0)
        active_all = np.flatnonzero(g0 < 2e-7)
        if len(active_all) < 60:
            z0 = polish(z0, deadline-time.monotonic())
            return best if best is not None else {'circles': c0.tolist()}
        _, rr, piv = qr(a0[active_all].T, mode='economic', pivoting=True)
        rank = int(np.sum(np.abs(np.diag(rr)) > 1e-9))
        active = active_all[piv[:rank]]
        aa = a0[active]
        stress = lstsq(aa.T, -grad, lapack_driver='gelsy')[0]
        # Mirror each contact, including top/bottom wall contacts.
        cc = z0[:63].reshape(n, 3)
        mirrored = cc.copy()
        mirrored[:, 1] = 2-z0[63]-mirrored[:, 1]
        reflection = np.argmin(np.sum((mirrored[:, None, :]-cc[None, :, :])**2, axis=2), axis=1)
        lookup = np.full((n,n), -1, dtype=int)
        lookup[ii,jj] = rows
        lookup[jj,ii] = rows
        location = {int(v): k for k,v in enumerate(active)}
        groups = []
        used = set()
        for k, edge in enumerate(active):
            check()
            if k in used:
                continue
            if edge < m:
                partner = lookup[reflection[ii[edge]], reflection[jj[edge]]]
            else:
                wall, circle = divmod(int(edge)-m,n)
                wall = {0:0,1:1,2:3,3:2}[wall]
                partner = m+wall*n+int(reflection[circle])
            group = [k]
            if int(partner) in location and location[int(partner)] != k:
                group.append(location[int(partner)])
            used.update(group)
            groups.append(group)
        costs = np.array([sum(abs(stress[k]) for k in gr) for gr in groups])
        weights = 1.0/(0.05+costs)
        weights /= weights.sum()
        inactive = np.ones(m+84, dtype=bool)
        inactive[active_all] = False
        other = np.flatnonzero(inactive)
        for trial in range(int(params['trials'])):
            check()
            if deadline-time.monotonic() < 0.13:
                break
            chosen = rng.choice(len(groups), size=min(int(params['release_groups']),len(groups)), replace=False, p=weights)
            target_direction = np.zeros(len(active))
            for gi in chosen:
                check()
                target_direction[groups[int(gi)]] = rng.uniform(0.6,1.4)
            tangent = lstsq(aa, target_direction, lapack_driver='gelsy')[0]
            speed = np.max(np.linalg.norm(tangent[:63].reshape(n,3)[:,:2],axis=1))
            target_direction /= max(speed,1e-8)
            current = z0.copy()
            previous = current.copy()
            event = None
            length = float(params['path_length'])
            # Nonlinear continuation keeps the unreleased contacts closed.
            for step in range(1,9):
                check()
                t = length*step/8
                target = g0[active]+t*target_direction
                v = current.copy()
                for newton in range(5):
                    check()
                    residual = constraints(v)[active]-target
                    if np.max(np.abs(residual)) < 2e-9:
                        break
                    delta = lstsq(jacobian(v)[active], -residual, lapack_driver='gelsy')[0]
                    v += delta
                if not np.isfinite(v).all() or np.min(v[2:63:3]) < 0.015:
                    break
                gaps = constraints(v)
                if len(other) and np.min(gaps[other]) <= 0:
                    event = int(other[np.argmin(gaps[other])])
                    previous = current
                    current = v
                    break
                if np.min(gaps) < -2e-6:
                    break
                previous, current = current, v
            check()
            left = deadline-time.monotonic()
            if event is not None:
                # Optimize on the new contact first, preventing immediate reversal.
                current = polish(current, min(0.17, left*0.47), event)
                check()
                current = polish(current, min(0.15, (deadline-time.monotonic())*0.7))
            else:
                current = polish(current, min(0.21, left*0.75))
    except (StopSearch, np.linalg.LinAlgError, ValueError):
        pass
    if best is None:
        # Conservative, always feasible cold-start fallback.
        c = np.zeros((21,3))
        c[:,0] = (np.arange(21)%5)*0.2+0.1
        c[:,1] = (np.arange(21)//5)*0.2+0.1
        c[:,2] = 0.099999999999
        return {'circles':c.tolist()}
    return best
