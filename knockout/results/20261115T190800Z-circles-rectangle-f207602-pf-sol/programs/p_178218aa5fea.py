import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (105, 50, 180), "perturbation": (0.022, 0.002, 0.08), "portfolio": (4, 2, 7)}


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    champion = None

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
        d = a[ii, :2]-a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2+d[:, 1]**2) < a[ii, 2]+a[jj, 2]):
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            best_score = exact(array(init))
        except (ValueError, TypeError, KeyError):
            pass

    def retain(model, z):
        nonlocal best, best_score, champion
        X, Y, R = model[:3]
        a = np.column_stack((X@z, Y@z, R@z))
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = a[ii, :2]-a[jj, :2]
        dist = np.sqrt(d[:, 0]**2+d[:, 1]**2)
        factor = min(1., float(np.min(dist/(a[ii, 2]+a[jj, 2]))))
        if factor <= 0:
            return
        a[:, 2] *= factor*(1-4e-14)
        p = perimeter(a)
        if p <= 0:
            return
        a *= float(Fraction(2)/p)*(1-4e-14)
        score = exact(a)
        if score > best_score:
            best_score = score
            best = {'centers': a[:, :2].tolist(), 'radii': a[:, 2].tolist()}
            # Preserve the uncontracted optimizer coordinates for restarts.
            champion = (model, z.copy())

    def make_model(counts):
        check()
        ns = counts[0]
        npair = sum(counts[1:])
        nvar = 2*ns+3*npair+1
        X = np.zeros((21, nvar))
        Y = np.zeros_like(X)
        R = np.zeros_like(X)
        z = np.zeros(nvar)
        mirror = np.arange(21)
        unique = []
        rcols = []
        ycols = []
        idx = 0
        col = 0
        for level, count in enumerate(counts):
            check()
            for k in range(count):
                check()
                x = (k-(count-1)/2)*0.20
                if level == 0 and count == 3:
                    x *= 1.45
                unique.append(idx)
                if level == 0:
                    X[idx, col] = 1
                    R[idx, col+1] = 1
                    z[col:col+2] = [x, .095]
                    rcols.append(col+1)
                    idx += 1
                    col += 2
                else:
                    X[idx:idx+2, col] = 1
                    Y[idx, col+1] = 1
                    Y[idx+1, col+1] = -1
                    R[idx:idx+2, col+2] = 1
                    mirror[idx], mirror[idx+1] = idx+1, idx
                    z[col:col+3] = [x, level*.177, .095]
                    ycols.append(col+1)
                    rcols.append(col+2)
                    idx += 2
                    col += 3
        a = np.column_stack((X@z, Y@z, R@z))
        low = np.min(a[:, :2]-a[:, 2, None], axis=0)
        high = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2/np.sum(high-low)
        z *= scale
        # All x coordinates have one independent column per reflection orbit.
        xcols = np.flatnonzero(np.any(X != 0, axis=0))
        z[xcols] -= low[0]*scale
        z[-1] = (high[0]-low[0])*scale
        selected = []
        for i,j in zip(ii,jj):
            check()
            p = tuple(sorted((int(mirror[i]), int(mirror[j]))))
            if (int(i), int(j)) <= p:
                selected.append((i,j))
        p = np.asarray(selected)
        DX = X[p[:, 0]]-X[p[:, 1]]
        DY = Y[p[:, 0]]-Y[p[:, 1]]
        RS = R[p[:, 0]]+R[p[:, 1]]
        u = np.array(unique)
        L = np.concatenate((X[u]-R[u], -X[u]-R[u], -Y[u]-R[u]), axis=0)
        m = len(u)
        L[m:2*m, -1] = 1
        L[2*m:, -1] = -.5
        c = np.zeros(3*m)
        c[2*m:] = 1
        bounds = [(None, None)]*nvar
        for k in rcols:
            check()
            bounds[k] = (1e-7, .5)
        for k in ycols:
            check()
            bounds[k] = (0., 1.)
        bounds[-1] = (.05, 1.95)
        grad = -np.sum(R, axis=0)
        model = (X,Y,R,DX,DY,RS,L,c,bounds,grad,xcols,np.array(ycols),np.array(rcols))
        return model,z

    def optimize(model, z, iterations):
        check()
        X,Y,R,DX,DY,RS,L,c,bounds,grad = model[:10]
        latest = [z.copy()]
        count = [0]
        cache = [None, None, None]
        def calc(t):
            check()
            if cache[0] is None or not np.array_equal(t, cache[0]):
                dx = DX@t
                dy = DY@t
                d = np.sqrt(dx*dx+dy*dy)
                den = np.maximum(d, 1e-15)
                cache[0] = t.copy()
                cache[1] = np.concatenate((d-RS@t, L@t+c))
                cache[2] = np.vstack((DX*(dx/den)[:, None]+DY*(dy/den)[:, None]-RS, L))
            return cache[1],cache[2]
        def cb(t):
            check()
            latest[0] = t.copy()
            count[0] += 1
            if count[0] % 15 == 0:
                retain(model,t)
        def fun(t):
            check()
            return float(grad@t)
        try:
            result = minimize(fun,z,jac=lambda t:grad,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':lambda t:calc(t)[0],
                                           'jac':lambda t:calc(t)[1]},callback=cb,
                              options={'maxiter':iterations,'ftol':8e-13,'disp':False})
            latest[0] = result.x
        except Stop:
            retain(model,latest[0])
            raise
        retain(model,latest[0])
        return latest[0]

    try:
        families = [(3,5,4),(5,4,4),(3,4,5),(5,5,3),(3,3,4,2),(7,4,3),(7,7)]
        for counts in families[:params['portfolio']]:
            check()
            model,z = make_model(counts)
            retain(model,z)
            optimize(model,z,params['iterations'])
        attempt = 0
        while time.monotonic() < deadline and champion is not None:
            check()
            model,z0 = champion
            z = z0.copy()
            xc,yc,rc = model[10:13]
            amplitude = params['perturbation']*(1.8 if attempt % 3 == 2 else 1.)
            chosen = rng.choice(len(xc), min(4,len(xc)), replace=False)
            z[xc[chosen]] += rng.normal(0, amplitude, len(chosen))
            if len(yc):
                z[yc] += rng.normal(0, amplitude*.5, len(yc))
                z[yc] = np.maximum(z[yc], 1e-5)
            optimize(model,z,70)
            attempt += 1
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    if best is None or best_score == -float('inf'):
        a = np.array([[(i%7)*.2,(i//7)*.2,.0999999999999] for i in range(21)])
        fallback = {'circles':a.tolist()}
        if exact(a) > best_score:
            return fallback
    return best
