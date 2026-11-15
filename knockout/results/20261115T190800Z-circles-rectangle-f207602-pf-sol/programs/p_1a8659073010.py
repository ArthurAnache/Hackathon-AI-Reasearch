import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_fraction': (0.31, 0.15, 0.45),
    'second_fraction': (0.25, 0.1, 0.4),
    'tolerance': (2e-11, 1e-13, 1e-7, 'log'),
}

class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    finish = started + 0.70 * budget_s
    best = None
    best_score = -float('inf')
    best_c = None
    ii, jj = np.triu_indices(21, 1)

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((c[ii, 0] - c[jj, 0])**2 + (c[ii, 1] - c[jj, 1])**2)
        if np.any(d < c[ii, 2] + c[jj, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        wh = max(a+b for a,b in zip(x,r)) - min(a-b for a,b in zip(x,r))
        wh += max(a+b for a,b in zip(y,r)) - min(a-b for a,b in zip(y,r))
        if wh > 2:
            return -float('inf')
        return float(np.sum(c[:, 2]))

    if init is not None:
        try:
            if 'circles' in init:
                c = np.array(init['circles'], dtype=float)
            else:
                z = np.asarray(init['state'], dtype=float)
                c = np.column_stack((z[:21], z[21:42], z[42:63]))
            s = exact(c)
            if s > best_score:
                best_score, best, best_c = s, init, c.copy()
        except (ValueError, TypeError, KeyError):
            pass

    def retain(c):
        nonlocal best, best_score, best_c
        if time.monotonic() >= finish:
            raise StopSearch()
        c = np.array(c, dtype=float, copy=True)
        if c.shape != (21, 3) or not np.isfinite(c).all():
            return
        c[:, 2] = np.maximum(c[:, 2], 1e-12)
        c[:, :2] -= (np.max(c[:, :2], axis=0) + np.min(c[:, :2], axis=0)) / 2
        d = np.sqrt((c[ii, 0]-c[jj, 0])**2 + (c[ii, 1]-c[jj, 1])**2)
        ratio = float(np.min(d/(c[ii, 2]+c[jj, 2])))
        c[:, 2] *= min(1.0, ratio) * (1.0-3e-14)
        if np.any(c[:, 2] <= 0):
            return
        wh = float(np.sum(np.max(c[:, :2]+c[:, 2,None], axis=0)-np.min(c[:, :2]-c[:, 2,None], axis=0)))
        c *= (2.0/wh)*(1.0-4e-14)
        s = float(np.sum(c[:, 2]))
        if s > best_score and exact(c) > best_score:
            best_score, best_c = s, c.copy()
            best = {'circles': c.tolist()}

    # A valid fallback independent of the supplied cold start.
    fallback = np.array([[0.22*(i % 7-3), 0.22*(i//7-1), 0.109] for i in range(21)])
    try:
        retain(fallback)
    except StopSearch:
        return best if best is not None else {'circles': fallback.tolist()}

    def quotient(counts):
        # Full coordinates equal T @ q; q comprises 12 x's, 9 y's,
        # 12 radii, and the rectangle width.
        T = np.zeros((64, 34))
        q = np.zeros(34)
        mirror = np.empty(21, dtype=int)
        single = 0
        pair = 0
        circle = 0
        for col, count in enumerate(counts):
            if time.monotonic() >= finish:
                raise StopSearch()
            xc = (col - (len(counts)-1)/2)*np.sqrt(3.0)
            if count % 2:
                k = single
                single += 1
                T[circle, k] = 1
                T[42+circle, 21+k] = 1
                q[k], q[21+k] = xc, 0.97
                mirror[circle] = circle
                circle += 1
            for level in range(count//2):
                if time.monotonic() >= finish:
                    raise StopSearch()
                k = 3+pair
                yy = 2*(level+1) if count % 2 else 2*level+1
                q[k], q[12+pair], q[21+k] = xc, yy, 0.97
                a, b = circle, circle+1
                T[a, k] = T[b, k] = 1
                T[21+a, 12+pair] = 1
                T[21+b, 12+pair] = -1
                T[42+a, 21+k] = T[42+b, 21+k] = 1
                mirror[a], mirror[b] = b, a
                pair += 1
                circle += 2
        T[63, 33] = 1
        v = T @ q
        c = np.column_stack((v[:21],v[21:42],v[42:63]))
        wx = np.ptp(c[:,0])+2*0.97
        hy = np.ptp(c[:,1])+2*0.97
        q[:33] *= 2/(wx+hy)
        q[33] = 2*wx/(wx+hy)
        return T, q, mirror

    def solve(T, q, mirror, deadline, maxiter=140):
        m = T.shape[1]
        tx, ty, tr = T[:21], T[21:42], T[42:63]
        inds = np.arange(len(ii))
        if mirror is not None:
            # Keep only one pair from each reflected orbit.
            ma = np.minimum(mirror[ii], mirror[jj])
            mb = np.maximum(mirror[ii], mirror[jj])
            inds = np.flatnonzero((ii*21+jj) <= (ma*21+mb))
        ia, ib = ii[inds], jj[inds]
        DX, DY, RR = tx[ia]-tx[ib], ty[ia]-ty[ib], tr[ia]+tr[ib]
        tw = T[63]
        walls = np.vstack((0.5*tw-tx-tr, 0.5*tw+tx-tr,
                           -0.5*tw-ty-tr, -0.5*tw+ty-tr))
        offsets = np.concatenate((np.zeros(42),np.ones(42)))
        if mirror is not None:
            unique = np.unique(np.column_stack((walls, offsets)), axis=0, return_index=True)[1]
            walls, offsets = walls[unique], offsets[unique]
        grad = -np.sum(tr, axis=0)
        cached_q = None
        cached_c = None
        cached_j = None

        def calc(v):
            nonlocal cached_q, cached_c, cached_j
            if time.monotonic() >= deadline:
                raise StopSearch()
            if cached_q is None or not np.array_equal(v, cached_q):
                dx, dy = DX@v, DY@v
                dd = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
                cached_c = np.concatenate((dd-RR@v, walls@v+offsets))
                cached_j = np.vstack(((dx/dd)[:,None]*DX+(dy/dd)[:,None]*DY-RR, walls))
                cached_q = v.copy()
            return cached_c, cached_j

        def callback(v):
            if time.monotonic() >= deadline:
                raise StopSearch()
            u = T@v
            retain(np.column_stack((u[:21],u[21:42],u[42:63])))

        if m == 34:
            bounds = [(None,None)]*12 + [(0,None)]*9 + [(1e-8,None)]*12 + [(0.1,1.9)]
        else:
            bounds = [(None,None)]*42 + [(1e-8,None)]*21 + [(0.1,1.9)]
        try:
            result = minimize(lambda v: float(grad@v), q, jac=lambda v: grad,
                method='SLSQP', bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0], 'jac':lambda v:calc(v)[1]},
                callback=callback, options={'maxiter':maxiter,'ftol':params['tolerance'],'disp':False})
            callback(result.x)
            return result.x
        except StopSearch:
            return None

    # Distinct compact strip families have different contact graphs.
    families = ([3,4,3,4,3,4], [3,4,5,4,5], [5,4,3,4,5])
    allocations = (params['first_fraction'], params['second_fraction'], 0.10)
    for counts, allocation in zip(families, allocations):
        if time.monotonic() >= finish-0.045:
            break
        try:
            T, q, mirror = quotient(counts)
            deadline = min(finish-0.025, time.monotonic()+allocation*budget_s)
            solve(T,q,mirror,deadline)
        except StopSearch:
            break
        if best_score > 2.3657:
            break

    if best_c is not None and time.monotonic() < finish-0.04:
        c = best_c.copy()
        c[:,:2] -= (np.max(c[:,:2]+c[:,2,None],axis=0)+np.min(c[:,:2]-c[:,2,None],axis=0))/2
        w = np.ptp(c[:,0])+2*np.max(c[:,2])
        w = np.max(c[:,0]+c[:,2])-np.min(c[:,0]-c[:,2])
        q = np.concatenate((c[:,0],c[:,1],c[:,2],[w]))
        solve(np.eye(64),q,None,finish-0.008,100)
    return best
