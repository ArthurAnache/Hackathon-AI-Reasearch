import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'opening': (0.045, 0.008, 0.14),
    'groups': (3, 3, 4),
    'restart_seconds': (0.23, 0.12, 0.4),
    'preservation': (8.0, 2.0, 30.0),
    'time_fraction': (0.70, 0.55, 0.74)
}

class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    ids = np.arange(n)
    best = init
    best_score = -np.inf
    stage_end = deadline

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def stage_check():
        if time.monotonic() >= min(stage_end, deadline):
            raise StopSearch()

    def exact_score(c):
        check()
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -np.inf
        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        if np.any(np.sqrt(dx**2 + dy**2) < c[ii, 2] + c[jj, 2]):
            return -np.inf
        lx = []; ux = []; ly = []; uy = []
        for x, y, r in c:
            check()
            x = Fraction(float(x)); y = Fraction(float(y)); r = Fraction(float(r))
            lx.append(x-r); ux.append(x+r)
            ly.append(y-r); uy.append(y+r)
        if max(ux)-min(lx)+max(uy)-min(ly) > Fraction(2):
            return -np.inf
        return float(np.sum(c[:, 2]))

    def certify(z):
        nonlocal best, best_score
        check()
        c = np.column_stack((z[:n], z[n:2*n], z[2*n:3*n])).copy()
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        d = np.sqrt((c[ii, 0]-c[jj, 0])**2 + (c[ii, 1]-c[jj, 1])**2)
        shrink = min(1.0, float(np.min(d/(c[ii, 2]+c[jj, 2]))))
        if shrink <= 0:
            return
        c[:, 2] *= shrink * (1.0-4e-15)
        c[:, 0] -= np.min(c[:, 0]-c[:, 2])
        c[:, 1] -= np.min(c[:, 1]-c[:, 2])
        perimeter = (np.max(c[:, 0]+c[:, 2])-np.min(c[:, 0]-c[:, 2])
                     + np.max(c[:, 1]+c[:, 2])-np.min(c[:, 1]-c[:, 2]))
        c *= (2.0/perimeter)*(1.0-4e-15)
        if float(np.sum(c[:, 2])) <= best_score:
            return
        score = exact_score(c)
        if score > best_score:
            best_score = score
            best = {'circles': c.tolist()}

    def encode(c):
        c = np.asarray(c, dtype=float).copy()
        c[:, 0] -= np.min(c[:, 0]-c[:, 2])
        c[:, 1] -= np.min(c[:, 1]-c[:, 2])
        width = float(np.max(c[:, 0]+c[:, 2]))
        height = float(np.max(c[:, 1]+c[:, 2]))
        c *= 2.0/(width+height)
        return np.concatenate((c[:, 0], c[:, 1], c[:, 2], [2.0*width/(width+height)]))

    def geometry(z, jac=False):
        check()
        x = z[:n]; y = z[n:2*n]; r = z[2*n:3*n]; w = z[-1]
        dx = x[ii]-x[jj]; dy = y[ii]-y[jj]
        d = np.sqrt(dx*dx+dy*dy)
        g = np.concatenate((d-r[ii]-r[jj], x-r, w-x-r, y-r, 2.0-w-y-r))
        if not jac:
            return g
        a = np.zeros((m+4*n, 3*n+1))
        k = np.arange(m)
        den = np.maximum(d, 1e-100)
        a[k, ii] = dx/den; a[k, jj] = -dx/den
        a[k, n+ii] = dy/den; a[k, n+jj] = -dy/den
        a[k, 2*n+ii] = -1; a[k, 2*n+jj] = -1
        a[m+ids, ids] = 1; a[m+ids, 2*n+ids] = -1
        a[m+n+ids, ids] = -1; a[m+n+ids, 2*n+ids] = -1; a[m+n+ids, -1] = 1
        a[m+2*n+ids, n+ids] = 1; a[m+2*n+ids, 2*n+ids] = -1
        a[m+3*n+ids, n+ids] = -1; a[m+3*n+ids, 2*n+ids] = -1; a[m+3*n+ids, -1] = -1
        return g, a

    grad = np.zeros(64)
    grad[42:63] = -1
    bounds = [(None, None)]*42 + [(1e-6, 0.6)]*21 + [(0.3, 1.7)]

    def polish(z, seconds, iterations=90):
        nonlocal stage_end
        stage_end = min(deadline, time.monotonic()+seconds)
        last = z.copy()
        def fun(v):
            stage_check()
            return -float(np.sum(v[42:63]))
        def jac(v):
            stage_check()
            return grad
        def cons(v):
            stage_check()
            return geometry(v)
        def cjac(v):
            stage_check()
            return geometry(v, True)[1]
        def callback(v):
            nonlocal last
            stage_check()
            last = v.copy()
            certify(v)
        try:
            result = minimize(fun, z, jac=jac, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': cons, 'jac': cjac},
                              callback=callback,
                              options={'maxiter': iterations, 'ftol': 2e-13, 'disp': False})
            last = result.x
            certify(last)
        except StopSearch:
            pass
        return last

    try:
        if init is not None:
            best_score = exact_score(np.asarray(init['circles'], dtype=float))
            base = encode(init['circles'])
        else:
            k = np.arange(21)
            c = np.column_stack(((k % 5)*0.2+0.1, (k//5)*0.2+0.1, np.full(21, 0.1*(1-1e-14))))
            best = {'circles': c.tolist()}
            best_score = exact_score(c)
            base = encode(c)
        polish(base, min(0.24, budget_s*0.13), 45)
        check()
        if best is not None:
            base = encode(best['circles'])
        g0, a0 = geometry(base, True)
        active = np.flatnonzero(g0 < 2e-6)
        if len(active) < 12:
            polish(base, deadline-time.monotonic())
            return best

        # Identify the geometric reflection rather than relying on input order.
        x = base[:21]; y = base[21:42]; r = base[42:63]
        h = 2.0-base[-1]
        metric = ((x[:, None]-x[None, :])**2
                  +(y[:, None]+y[None, :]-h)**2
                  +(r[:, None]-r[None, :])**2)
        mirror = np.argmin(metric, axis=1)
        pair_lookup = np.full((21, 21), -1, dtype=int)
        pair_lookup[ii, jj] = np.arange(m)
        pair_lookup[jj, ii] = np.arange(m)
        position = {int(k): q for q, k in enumerate(active)}
        groups = []
        used = set()
        centres = []
        for row in active:
            check()
            row = int(row)
            if row in used:
                continue
            if row < m:
                reflected = int(pair_lookup[mirror[ii[row]], mirror[jj[row]]])
                cx = (x[ii[row]]+x[jj[row]])/2
                cy = abs((y[ii[row]]+y[jj[row]])/2-h/2)
            else:
                wall, circle = divmod(row-m, n)
                rw = (3 if wall == 2 else 2 if wall == 3 else wall)
                reflected = m+rw*n+int(mirror[circle])
                cx = x[circle]; cy = abs(y[circle]-h/2)
            group = [position[row]]
            used.add(row)
            if reflected in position and reflected not in used:
                group.append(position[reflected]); used.add(reflected)
            groups.append(np.asarray(group, dtype=int))
            centres.append((cx, cy))
        centres = np.asarray(centres)
        aa = a0[active]
        multipliers = lstsq(aa.T, grad, cond=1e-10, lapack_driver='gelsy')[0]
        costs = np.empty(len(groups))
        for q, group in enumerate(groups):
            check()
            costs[q] = max(0.015, float(np.sum(np.maximum(multipliers[group], 0))))
        weights = 1/np.sqrt(costs)
        weights /= np.sum(weights)
        attempt = 0
        while time.monotonic() < deadline-0.055:
            check()
            first = int(rng.choice(len(groups), p=weights))
            spatial = np.sum((centres-centres[first])**2, axis=1)
            probs = weights*np.exp(-spatial/(0.10 if attempt % 2 == 0 else 1.0))
            probs[first] = 0
            probs /= np.sum(probs)
            others = rng.choice(len(groups), size=min(int(params['groups'])-1, len(groups)-1), replace=False, p=probs)
            chosen_groups = np.concatenate(([first], others))
            selected = []
            for q in chosen_groups:
                check()
                selected.extend(groups[int(q)].tolist())
            selected = np.asarray(selected, dtype=int)
            target = np.zeros(len(active))
            amplitude = float(params['opening'])*(0.7, 1.3, 2.0, 1.0, 1.7)[attempt % 5]
            target[selected] = amplitude
            importance = np.full(len(active), float(params['preservation']))
            importance[selected] = 1.0
            z = base.copy()
            for step in range(5):
                check()
                values, matrix = geometry(z, True)
                residual = target-values[active]
                delta = lstsq(matrix[active]*importance[:, None], residual*importance,
                              cond=1e-9, lapack_driver='gelsy')[0]
                size = float(np.max(np.abs(delta)))
                if size > 0.14:
                    delta *= 0.14/size
                alpha = 1.0
                falling = delta[42:63] < 0
                if np.any(falling):
                    alpha = min(alpha, float(np.min((z[42:63][falling]-0.004)/(-delta[42:63][falling])))*0.9)
                if alpha <= 0:
                    break
                z += alpha*delta
                if np.linalg.norm(residual) < 1e-7:
                    break
            check()
            if np.all(z[42:63] > 0) and 0.3 < z[-1] < 1.7:
                certify(z)
                polish(z, float(params['restart_seconds']))
            attempt += 1
    except StopSearch:
        pass
    return best
