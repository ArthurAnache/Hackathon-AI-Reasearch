import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize
from scipy.linalg import qr, solve

PARAMS = {
    'release_size': (0.085, 0.02, 0.22),
    'weak_pool': (18, 6, 40),
    'restarts': (8, 1, 20),
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    ids = np.arange(n)
    best = init
    best_score = -np.inf

    class Expired(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Expired()

    def exact(c):
        check()
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -np.inf
        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        if np.any(np.sqrt(dx**2 + dy**2) < c[ii, 2] + c[jj, 2]):
            return -np.inf
        xs = [Fraction(float(x)) for x in c[:, 0]]
        ys = [Fraction(float(y)) for y in c[:, 1]]
        rs = [Fraction(float(r)) for r in c[:, 2]]
        w = max(x+r for x,r in zip(xs,rs)) - min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs)) - min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -np.inf
        return float(np.sum(c[:, 2]))

    def offer(z):
        nonlocal best, best_score
        check()
        c = np.column_stack((z[:n], z[n:2*n], z[2*n:3*n])).copy()
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        d = np.sqrt((c[ii, 0]-c[jj, 0])**2 + (c[ii, 1]-c[jj, 1])**2)
        ratio = min(1.0, float(np.min(d/(c[ii, 2]+c[jj, 2]))))
        c[:, 2] *= ratio * (1.0-8e-16)
        c[:, :2] -= np.mean(c[:, :2], axis=0)
        perimeter = float(np.ptp(c[:, 0]))
        w = np.max(c[:, 0]+c[:, 2])-np.min(c[:, 0]-c[:, 2])
        h = np.max(c[:, 1]+c[:, 2])-np.min(c[:, 1]-c[:, 2])
        c *= (2.0/(w+h))*(1.0-8e-16)
        for attempt in range(3):
            check()
            score = exact(c)
            if score > best_score:
                best_score = score
                best = {'circles': c.tolist()}
            if np.isfinite(score):
                break
            c[:, 2] *= 1.0-2e-15
            c *= 1.0-2e-15

    def encode(c):
        c = np.asarray(c, dtype=float).copy()
        left = np.min(c[:, 0]-c[:, 2])
        bottom = np.min(c[:, 1]-c[:, 2])
        width = np.max(c[:, 0]+c[:, 2])-left
        height = np.max(c[:, 1]+c[:, 2])-bottom
        c[:, 0] -= left
        c[:, 1] -= bottom
        scale = 2.0/(width+height)
        c *= scale
        return np.r_[c[:, 0], c[:, 1], c[:, 2], width*scale]

    def constraints(z):
        check()
        x,y,r,w = z[:n],z[n:2*n],z[2*n:3*n],z[-1]
        d = np.sqrt((x[ii]-x[jj])**2+(y[ii]-y[jj])**2)
        return np.r_[d-r[ii]-r[jj], x-r, w-x-r, y-r, 2-w-y-r]

    def jacobian(z):
        check()
        x,y = z[:n],z[n:2*n]
        dx,dy = x[ii]-x[jj],y[ii]-y[jj]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-100)
        j = np.zeros((m+4*n, 3*n+1))
        j[rows,ii] = dx/d
        j[rows,jj] = -dx/d
        j[rows,n+ii] = dy/d
        j[rows,n+jj] = -dy/d
        j[rows,2*n+ii] = -1
        j[rows,2*n+jj] = -1
        j[m+ids,ids] = 1
        j[m+ids,2*n+ids] = -1
        j[m+n+ids,ids] = -1
        j[m+n+ids,2*n+ids] = -1
        j[m+n+ids,-1] = 1
        j[m+2*n+ids,n+ids] = 1
        j[m+2*n+ids,2*n+ids] = -1
        j[m+3*n+ids,n+ids] = -1
        j[m+3*n+ids,2*n+ids] = -1
        j[m+3*n+ids,-1] = -1
        return j

    gradient = np.r_[np.zeros(2*n),-np.ones(n),0.0]
    bounds = [(None,None)]*(2*n)+[(1e-6,0.5)]*n+[(0.1,1.9)]
    def objective(z):
        check()
        return -float(np.sum(z[2*n:3*n]))

    def optimize(z, iterations):
        last = [z.copy()]
        counter = [0]
        def callback(q):
            check()
            last[0] = q.copy()
            counter[0] += 1
            if counter[0] % 4 == 0:
                offer(q)
        result = minimize(objective, z, jac=lambda z: gradient,
                          method='SLSQP', bounds=bounds,
                          constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                          callback=callback,
                          options={'maxiter':iterations,'ftol':2e-13,'disp':False})
        offer(result.x)
        return result.x

    def contact_polish(z):
        vals = constraints(z)
        selected = np.flatnonzero(vals < 3e-6)
        if len(selected) < 64:
            return z, None, None
        j = jacobian(z)
        _,_,piv = qr(j[selected].T, pivoting=True, mode='economic', check_finite=False)
        active = selected[piv[:64]]
        q = z.copy()
        for k in range(7):
            check()
            residual = constraints(q)[active]
            a = jacobian(q)[active]
            try:
                step = solve(a, -residual, check_finite=False)
            except Exception:
                return z, None, None
            if np.max(np.abs(step)) > 0.1:
                return z, None, None
            q += step
            if np.max(np.abs(step)) < 2e-15:
                break
        offer(q)
        if np.min(constraints(q)) < -1e-8:
            return z, None, None
        return q, active, jacobian(q)[active]

    try:
        if init is not None:
            original = np.asarray(init['circles'], dtype=float)
            best_score = exact(original)
            z = encode(original)
        else:
            grid = np.array([(i/5.0+0.1, j/5.0+0.1, 0.095)
                             for j in range(5) for i in range(5)], dtype=float)[:21]
            z = encode(grid)
            offer(z)
        base, active, matrix = contact_polish(z)
        if active is None:
            base = optimize(base, 70)
            base, active, matrix = contact_polish(base)
        else:
            offer(base)

        inverse = None
        pool = None
        if matrix is not None:
            check()
            try:
                dual = solve(matrix.T, gradient, check_finite=False)
                inverse = solve(matrix, np.eye(64), check_finite=False)
                contact_positions = np.flatnonzero(active < m)
                order = contact_positions[np.argsort(np.abs(dual[contact_positions]))]
                pool = order[:int(params['weak_pool'])]
            except Exception:
                inverse = None

        for restart in range(int(params['restarts'])):
            check()
            trial = base.copy()
            if inverse is not None and pool is not None and len(pool) >= 3:
                picks = rng.choice(pool, size=3, replace=False)
                direction = inverse[:, picks] @ rng.uniform(0.5,1.5,3)
                magnitude = np.max(np.sqrt(direction[:n]**2+direction[n:2*n]**2))
                amplitude = float(params['release_size']) * (0.75+0.5*(restart % 3))
                trial += direction * amplitude / max(magnitude, 1e-10)
                trial[2*n:3*n] = np.clip(trial[2*n:3*n],0.025,0.23)
                trial[-1] = np.clip(trial[-1],0.65,1.35)
            else:
                anchor = int(rng.integers(n))
                distances = (base[:n]-base[anchor])**2+(base[n:2*n]-base[n+anchor])**2
                group = np.argsort(distances)[:5]
                trial[group] += rng.normal(0,0.055,len(group))
                trial[n+group] += rng.normal(0,0.055,len(group))
            trial = optimize(trial, 65)
            contact_polish(trial)
    except Expired:
        pass
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass
    if best is None:
        best = {'circles': [[(i % 5)*0.2+0.1,(i // 5)*0.2+0.1,0.099] for i in range(21)]}
    return best
