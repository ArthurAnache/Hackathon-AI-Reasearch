import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'time_fraction': (0.69, 0.4, 0.7),
    'stage_iterations': (24, 10, 60),
    'contact_margin': (0.13, 0.03, 0.4),
    'jitter': (0.018, 0.001, 0.08)
}


def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        extent = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        extent += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if extent > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                if np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2) < a[i,2]+a[j,2]:
                    return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic()+float(params['time_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    pi, pj = np.triu_indices(21, 1)
    ids = np.arange(21)
    best = init
    score = exact_score(init)
    best_a = None
    if score > -float('inf'):
        best_a = np.asarray(init['circles'], dtype=float).copy()

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def normalize(a):
        a = a.copy()
        a[:,:2] -= np.min(a[:,:2]-a[:,2,None], axis=0)
        wh = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = 2/float(np.sum(wh))
        return a*scale, float(wh[0]*scale)

    def offer(a):
        nonlocal best, score, best_a
        a = np.asarray(a, dtype=float).reshape(21,3).copy()
        if not np.isfinite(a).all():
            return
        a[:,2] = np.maximum(a[:,2], 1e-10)
        delta = a[pi,:2]-a[pj,:2]
        ratio = np.min(np.sqrt(np.sum(delta*delta, axis=1))/(a[pi,2]+a[pj,2]))
        if ratio <= 0:
            return
        a[:,2] *= min(1.0, float(ratio))*(1-2e-13)
        a, _ = normalize(a)
        a *= 1-4e-13
        if float(np.sum(a[:,2])) <= score:
            return
        candidate = {'circles': a.tolist()}
        value = exact_score(candidate)
        if value > score:
            best, score, best_a = candidate, value, a

    # A horizontally reflected triangular-lattice cluster.
    lattice = []
    for row in range(-2, 3):
        if row == 0:
            xx = np.arange(5)*2.0
        elif abs(row) == 1:
            xx = np.arange(4)*2.0+1.0
        else:
            xx = np.arange(4)*2.0
        for x in xx:
            if time.monotonic() >= deadline:
                break
            lattice.append([float(x), float(row*np.sqrt(3)), 1.0])
    if len(lattice) != 21:
        return best
    lattice, _ = normalize(np.array(lattice))
    offer(lattice)
    try:
        raw = np.asarray(init['circles'], dtype=float)
        if raw.shape == (21,3) and np.isfinite(raw).all():
            offer(raw)
    except Exception:
        pass

    grad = np.zeros(64)
    grad[2:63:3] = -1

    def objective(z):
        check()
        return -float(np.sum(z[2:63:3]))

    def callback(z):
        check()
        offer(z[:63])

    attempt = 0
    while time.monotonic() < deadline:
        a = best_a.copy()
        if attempt and attempt % 4 == 0 and score < 2.32:
            a = lattice.copy()
        if attempt:
            amplitude = float(params['jitter'])*(1+(attempt % 3))
            a[:,:2] += rng.normal(0, amplitude, (21,2))
            a[:,2] *= rng.uniform(0.95, 1.05, 21)
        a, width = normalize(a)
        z = np.r_[a.ravel(), width]
        previous = float('inf')
        try:
            for stage in range(5):
                check()
                a = z[:63].reshape(21,3)
                delta = a[pi,:2]-a[pj,:2]
                gap = np.sqrt(np.sum(delta*delta, axis=1))-a[pi,2]-a[pj,2]
                active = gap < float(params['contact_margin'])
                # Always retain several neighbours of every circle.
                for i in range(21):
                    check()
                    edges = np.flatnonzero((pi == i)|(pj == i))
                    active[edges[np.argsort(gap[edges])[:5]]] = True
                ii, jj = pi[active], pj[active]
                kk = np.arange(len(ii))+84
                base = np.zeros((84+len(ii),64))
                base[ids,3*ids] = 1
                base[ids,3*ids+2] = -1
                base[21+ids,3*ids] = -1
                base[21+ids,3*ids+2] = -1
                base[21+ids,63] = 1
                base[42+ids,3*ids+1] = 1
                base[42+ids,3*ids+2] = -1
                base[63+ids,3*ids+1] = -1
                base[63+ids,3*ids+2] = -1
                base[63+ids,63] = -1
                base[kk,3*ii+2] = -1
                base[kk,3*jj+2] = -1

                def constraints(v):
                    check()
                    x,y,r = v[:63].reshape(21,3).T
                    dx,dy = x[ii]-x[jj],y[ii]-y[jj]
                    return np.r_[x-r,v[63]-x-r,y-r,2-v[63]-y-r,np.sqrt(dx*dx+dy*dy+1e-30)-r[ii]-r[jj]]

                def jac(v):
                    check()
                    b = v[:63].reshape(21,3)
                    dx,dy = b[ii,0]-b[jj,0],b[ii,1]-b[jj,1]
                    d = np.sqrt(dx*dx+dy*dy+1e-30)
                    out = base.copy()
                    out[kk,3*ii] = dx/d
                    out[kk,3*jj] = -dx/d
                    out[kk,3*ii+1] = dy/d
                    out[kk,3*jj+1] = -dy/d
                    return out

                result = minimize(objective,z,jac=lambda v: grad,method='SLSQP',
                    bounds=[(0,2),(0,2),(1e-7,0.7)]*21+[(0.15,1.85)],
                    constraints={'type':'ineq','fun':constraints,'jac':jac},callback=callback,
                    options={'maxiter':int(params['stage_iterations']),'ftol':2e-11,'disp':False})
                z = result.x
                offer(z[:63])
                current = float(result.fun)
                if abs(current-previous) < 1e-10:
                    break
                previous = current
        except StopSearch:
            break
        except (ValueError, FloatingPointError):
            pass
        attempt += 1
    return best
