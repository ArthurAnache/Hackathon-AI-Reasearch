import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (100, 40, 180), "perturbation": (0.018, 0.002, 0.06), "fraction": (0.70, 0.5, 0.72)}


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + params['fraction'] * budget_s
    rng = np.random.default_rng(seed)
    ai, aj = np.triu_indices(21, 1)
    best = init
    score = -float('inf')
    winning = None

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def array(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((obj['centers'], obj['radii']))

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        d = a[ai, :2] - a[aj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ai, 2] + a[aj, 2]):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        p = max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r))
        if p > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            score = exact(array(init))
        except (ValueError, KeyError, TypeError):
            pass

    def retain(a, model=None, q=None):
        nonlocal best, score, winning
        a = a.copy()
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = a[ai, :2] - a[aj, :2]
        ratio = np.min(np.sqrt(np.sum(d*d, axis=1)) / (a[ai, 2]+a[aj, 2]))
        if ratio <= 0:
            return
        a[:, 2] *= min(1., float(ratio)) * (1.-2e-14)
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        p = max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r))
        a *= float(Fraction(2)/p) * (1.-2e-14)
        val = exact(a)
        if val > score:
            score = val
            best = {'centers': a[:, :2].tolist(), 'radii': a[:, 2].tolist()}
            if model is not None:
                winning = (model, q.copy())

    def build(counts):
        check()
        xi, yi, ri, signs, mirror, reps, initial = [], [], [], [], [], [], []
        c = 0
        for level, count in enumerate(counts):
            check()
            for col in range(count):
                check()
                x = (col-(count-1)/2)*.20
                if level == 0 and count == 3:
                    x *= 1.45
                if level == 0:
                    k = len(xi)
                    xi.append(c); yi.append(0); ri.append(c+1); signs.append(0.)
                    mirror.append(k); reps.append(k)
                    initial.extend((x, .095)); c += 2
                else:
                    k = len(xi)
                    xi.extend((c,c)); yi.extend((c+1,c+1)); ri.extend((c+2,c+2))
                    signs.extend((1.,-1.)); mirror.extend((k+1,k)); reps.append(k)
                    initial.extend((x, level*.177, .095)); c += 3
        xi, yi, ri = map(lambda z: np.asarray(z, dtype=int), (xi,yi,ri))
        signs = np.asarray(signs)
        reps = np.asarray(reps)
        pairs = []
        for i,j in zip(ai,aj):
            check()
            if mirror[i] == j:
                continue
            reflected = tuple(sorted((mirror[i],mirror[j])))
            if (int(i),int(j)) <= reflected:
                pairs.append((i,j))
        pi,pj = np.asarray(pairs).T
        n = len(pi)
        rr = np.arange(n)
        bx, by, br, bs = xi[reps], yi[reps], ri[reps], signs[reps]
        m = len(reps)
        rows = np.arange(m)
        q = np.r_[initial, 1.]

        def expand(t):
            return np.column_stack((t[xi], signs*t[yi], t[ri]))

        a = expand(q)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2./np.sum(hi-lo)
        q[:-1] *= scale
        q[np.unique(xi)] -= lo[0]*scale
        q[-1] = (hi[0]-lo[0])*scale
        g = np.zeros(c+1)
        np.add.at(g,ri,-1.)
        bounds = [(0.,2.)]*(c+1)
        for k in np.unique(ri):
            check(); bounds[k] = (1e-7,.6)
        bounds[-1] = (.05,1.95)

        def constraints(t, jac=False):
            check()
            x = t[xi]; y = signs*t[yi]; r = t[ri]
            dx = x[pi]-x[pj]; dy = y[pi]-y[pj]
            dist = np.sqrt(dx*dx+dy*dy)
            if not jac:
                return np.r_[dist-r[pi]-r[pj], t[bx]-t[br], t[-1]-t[bx]-t[br],
                             1.-t[-1]/2-bs*t[by]-t[br],
                             t[by[bs>0]]-t[br[bs>0]]]
            npair = int(np.sum(bs>0))
            J = np.zeros((n+3*m+npair,c+1))
            den = np.maximum(dist,1e-16)
            J[rr,xi[pi]] += dx/den
            J[rr,xi[pj]] -= dx/den
            mask = signs[pi] != 0
            J[rr[mask],yi[pi[mask]]] += signs[pi[mask]]*dy[mask]/den[mask]
            mask = signs[pj] != 0
            J[rr[mask],yi[pj[mask]]] -= signs[pj[mask]]*dy[mask]/den[mask]
            J[rr,ri[pi]] -= 1.
            J[rr,ri[pj]] -= 1.
            J[n+rows,bx] = 1.; J[n+rows,br] = -1.
            J[n+m+rows,bx] = -1.; J[n+m+rows,br] = -1.; J[n+m+rows,-1] = 1.
            J[n+2*m+rows,br] = -1.; J[n+2*m+rows,-1] = -.5
            mask = bs>0
            J[n+2*m+rows[mask],by[mask]] = -1.
            end = n+3*m+np.arange(npair)
            J[end,by[mask]] = 1.; J[end,br[mask]] = -1.
            return J
        return (expand, constraints, g, bounds, xi, yi, ri, signs), q

    def optimize(model,q,iterations):
        check()
        expand,cons,g,bounds,xi,yi,ri,signs = model
        latest = [q.copy()]
        counter = [0]
        def fun(t):
            check()
            return float(g@t)
        def callback(t):
            check()
            latest[0] = t.copy()
            counter[0] += 1
            if counter[0] % 20 == 0:
                retain(expand(t),model,t)
        try:
            result = minimize(fun,q,jac=lambda t:g,method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':cons,'jac':lambda t:cons(t,True)},
                              callback=callback,options={'maxiter':iterations,'ftol':8e-12,'disp':False})
            latest[0] = result.x
        except Stop:
            retain(expand(latest[0]),model,latest[0])
            raise
        retain(expand(latest[0]),model,latest[0])
        return latest[0]

    try:
        fallback = np.array([[(i%7)*.2,(i//7)*.2,.0999999999999] for i in range(21)])
        retain(fallback)
        for counts in ((3,5,4),(5,4,4),(3,4,5),(5,5,3)):
            check()
            model,q = build(counts)
            optimize(model,q,params['iterations'])
        while time.monotonic() < deadline and winning is not None:
            check()
            model,q = winning
            q = q.copy()
            expand,cons,g,bounds,xi,yi,ri,signs = model
            inds = np.unique(np.r_[xi,yi[signs!=0]])
            q[inds] += rng.normal(0.,params['perturbation'],len(inds))
            q = np.maximum(q,np.asarray([v[0] for v in bounds]))
            q = np.minimum(q,np.asarray([v[1] for v in bounds]))
            optimize(model,q,65)
    except (Stop,ValueError,np.linalg.LinAlgError):
        pass
    if best is None:
        return {'circles': [[(i%7)*.2,(i//7)*.2,.0999999999999] for i in range(21)]}
    return best
