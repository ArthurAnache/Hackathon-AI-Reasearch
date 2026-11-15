import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'attempts': (12, 2, 40), 'amplitude': (0.022, 0.002, 0.09), 'iterations': (65, 20, 160)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    best = None
    best_score = -np.inf

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -np.inf
        x, y, r = [[Fraction(float(v)) for v in a[:, j]] for j in range(3)]
        p = max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r))
        return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

    original = None
    if init is not None:
        original = np.array(init['circles'], float) if 'circles' in init else np.array(init['state'], float).reshape(3, 21).T
        best_score = exact(original)
        best = original.copy()

    def repair(a):
        a = np.array(a, float, copy=True)
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return None
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a[:, :2] -= (lo+hi)/2
        d = a[ii, :2]-a[jj, :2]
        ratio = np.min(np.sqrt(d[:, 0]**2+d[:, 1]**2)/(a[ii, 2]+a[jj, 2]))
        a[:, 2] *= min(1., ratio)*(1.-4e-15)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        a *= (2./np.sum(hi-lo))*(1.-4e-15)
        return a

    def consider(a):
        nonlocal best, best_score
        a = repair(a)
        if a is None:
            return
        s = exact(a)
        if s > best_score:
            best, best_score = a.copy(), s

    if original is None:
        a = []
        for row, count in enumerate([4, 3, 4, 3, 4, 3]):
            if time.monotonic() >= deadline:
                break
            a.extend([[2*j-(count-1), row*np.sqrt(3.), 1.] for j in range(count)])
        if len(a) != 21:
            return {'circles': [[(i%3)*.2, (i//3)*.2, .09] for i in range(21)]}
        original = repair(a)
        consider(original)

    # Discover which coordinate is reversed by the incumbent's reflection.
    base = original.copy()
    lo = np.min(base[:, :2]-base[:, 2, None], axis=0)
    hi = np.max(base[:, :2]+base[:, 2, None], axis=0)
    base[:, :2] -= (lo+hi)/2
    choices = []
    for axis in range(2):
        if time.monotonic() >= deadline:
            return init if init is not None else {'circles': best.tolist()}
        b = base.copy()
        if axis:
            b[:, [0, 1]] = b[:, [1, 0]]
        reflected = b.copy()
        reflected[:, 0] *= -1
        cost = np.sum((b[:, None, :]-reflected[None, :, :])**2, axis=2)
        _, assignment = linear_sum_assignment(cost)
        choices.append((float(cost[np.arange(n), assignment].sum()), axis, b, assignment))
    _, swapped, base, assignment = min(choices, key=lambda v: v[0])
    singles = [i for i in range(n) if assignment[i] == i]
    pairs = [(i, int(assignment[i])) for i in range(n) if i < assignment[i] and assignment[assignment[i]] == i]
    if len(singles) != 3 or len(pairs) != 9:
        singles = list(np.argsort(np.abs(base[:, 0]))[:3])
        rest = [i for i in range(n) if i not in singles]
        pos = sorted(rest, key=lambda i: base[i, 0])
        left, right = pos[:9], pos[9:]
        c = np.array([[np.sum((base[i, 1:]-base[j, 1:])**2) for j in right] for i in left])
        _, match = linear_sum_assignment(c)
        pairs = [(left[k], right[match[k]]) for k in range(9)]
    pairs = [(i,j) if base[i,0] >= base[j,0] else (j,i) for i,j in pairs]
    order = singles + [i for i,j in pairs] + [j for i,j in pairs]
    base = base[order]

    T = np.zeros((64,34))
    for j in range(9):
        if time.monotonic() >= deadline:
            return init if init is not None else {'circles': best.tolist()}
        T[3+j,j] = 1
        T[12+j,j] = -1
        T[24+j,12+j] = T[33+j,12+j] = 1
        T[45+j,24+j] = T[54+j,24+j] = 1
    for j in range(3):
        if time.monotonic() >= deadline:
            return init if init is not None else {'circles': best.tolist()}
        T[21+j,9+j] = 1
        T[42+j,21+j] = 1
    T[63,33] = 1
    mirror = np.r_[np.arange(3), np.arange(12,21), np.arange(3,12)]
    keep = []
    for i,j in zip(ii,jj):
        if time.monotonic() >= deadline:
            return init if init is not None else {'circles': best.tolist()}
        p = tuple(sorted((int(mirror[i]),int(mirror[j]))))
        if (int(i),int(j)) <= p:
            keep.append((i,j))
    pi = np.array([v[0] for v in keep], int)
    pj = np.array([v[1] for v in keep], int)
    m = len(pi)
    rows = np.arange(m)
    wall = np.zeros((84,64))
    k = np.arange(21)
    wall[k,k] = -1; wall[k,42+k] = -1; wall[k,63] = .5
    wall[21+k,k] = 1; wall[21+k,42+k] = -1; wall[21+k,63] = .5
    wall[42+k,21+k] = -1; wall[42+k,42+k] = -1; wall[42+k,63] = -.5
    wall[63+k,21+k] = 1; wall[63+k,42+k] = -1; wall[63+k,63] = -.5
    wallq = wall @ T
    wallq, indices = np.unique(wallq, axis=0, return_index=True)
    constants = (indices >= 42).astype(float)
    grad = -np.sum(T[42:63], axis=0)
    bounds = [(0.,1.)]*9+[(-1.,1.)]*12+[(1e-6,.5)]*12+[(.05,1.95)]
    lo = np.min(base[:,:2]-base[:,2,None],axis=0)
    hi = np.max(base[:,:2]+base[:,2,None],axis=0)
    z = np.r_[base.T.ravel(),hi[0]-lo[0]]
    anchor = (T.T @ z)/np.sum(T*T,axis=0)
    best_q = anchor.copy()
    best_raw = float(-grad@anchor)

    class Stop(Exception):
        pass

    def unpack(q):
        a = (T@q)[:63].reshape(3,21).T.copy()
        if swapped:
            a[:,[0,1]] = a[:,[1,0]]
        return a

    def constraints(q):
        z = T@q
        dx = z[pi]-z[pj]
        dy = z[21+pi]-z[21+pj]
        r = z[42+pi]+z[42+pj]
        return np.r_[4*(dx*dx+dy*dy-r*r), wallq@q+constants]

    def jacobian(q):
        z = T@q
        dx = 8*(z[pi]-z[pj]); dy = 8*(z[21+pi]-z[21+pj])
        rr = -8*(z[42+pi]+z[42+pj])
        # Direct quotient derivatives avoid a dense matrix multiplication.
        J = dx[:,None]*(T[pi]-T[pj]) + dy[:,None]*(T[21+pi]-T[21+pj]) + rr[:,None]*(T[42+pi]+T[42+pj])
        return np.vstack((J,wallq))

    for attempt in range(int(params['attempts'])):
        now = time.monotonic()
        if now >= deadline-.015:
            break
        end = min(deadline-.008, now + (deadline-now)/max(1, int(params['attempts'])-attempt))
        if attempt == 0:
            q0 = anchor.copy()
        else:
            q0 = best_q.copy()
            amplitude = float(params['amplitude'])*[.4,1.,2.,.7,1.5][(attempt-1)%5]
            # Release several contacts together rather than a single edge.
            c = constraints(q0)
            J = jacobian(q0)
            active = np.flatnonzero(c < 2e-6)
            drop = rng.choice(len(active), size=min(len(active), 5+attempt%7), replace=False)
            retain = np.delete(active, drop)
            if len(retain):
                _, s, vh = np.linalg.svd(J[retain], full_matrices=True)
                rank = int(np.sum(s > 1e-7))
                null = vh[rank:]
            else:
                null = np.eye(34)
            if len(null):
                d = rng.normal(size=len(null))@null
            else:
                d = rng.normal(size=34)
            d /= max(np.max(np.abs(d)),1e-12)
            q0 += amplitude*d
            q0[:9] = np.clip(q0[:9],.001,.9)
            q0[21:33] = np.clip(q0[21:33],.025,.22)
            q0[33] = np.clip(q0[33],.2,1.8)
        last = [q0.copy()]
        def check():
            if time.monotonic() >= end:
                raise Stop()
        def fun(q):
            check()
            return float(grad@q)
        def con(q):
            check()
            return constraints(q)
        def jac(q):
            check()
            return jacobian(q)
        def callback(q):
            last[0] = q.copy()
            check()
        try:
            result = minimize(fun,q0,jac=lambda q:grad,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                              options={'maxiter':int(params['iterations']),'ftol':3e-13,'disp':False})
            last[0] = result.x
        except (Stop,ValueError,FloatingPointError):
            pass
        q = last[0]
        a = repair(unpack(q))
        if a is not None:
            s = exact(a)
            if s > best_score:
                best, best_score = a.copy(),s
            if s > best_raw-2e-8 and np.min(constraints(q)) > -1e-7:
                best_raw = max(best_raw,s)
                best_q = q.copy()
    if init is not None and exact(original) >= best_score:
        return init
    return {'state':best.T.ravel().tolist()}
