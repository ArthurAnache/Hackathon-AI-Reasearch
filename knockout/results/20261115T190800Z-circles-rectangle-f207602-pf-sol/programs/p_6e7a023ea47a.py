import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "screen_seconds": (0.22, 0.08, 0.4),
    "radius_floor": (0.00001, 0.000001, 0.001, "log"),
    "restart_noise": (0.008, 0.001, 0.04, "log")
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')

    def exact(data):
        try:
            c = np.asarray(data['circles'], dtype=float)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
                return -float('inf')
            d = np.sqrt((c[ii, 0] - c[jj, 0]) ** 2 + (c[ii, 1] - c[jj, 1]) ** 2)
            if np.any(d < c[ii, 2] + c[jj, 2]):
                return -float('inf')
            x = list(map(Fraction, map(float, c[:, 0])))
            y = list(map(Fraction, map(float, c[:, 1])))
            r = list(map(Fraction, map(float, c[:, 2])))
            w = max(a+b for a,b in zip(x,r)) - min(a-b for a,b in zip(x,r))
            h = max(a+b for a,b in zip(y,r)) - min(a-b for a,b in zip(y,r))
            if w+h > 2:
                return -float('inf')
            return float(np.sum(c[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        best_score = exact(init)

    def retain(c):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        c = np.array(c, dtype=float, copy=True)
        if not np.isfinite(c).all() or np.min(c[:, 2]) <= 0:
            return
        d = np.sqrt((c[ii, 0]-c[jj, 0])**2 + (c[ii, 1]-c[jj, 1])**2)
        ratio = float(np.min(d / (c[ii, 2]+c[jj, 2])))
        if ratio <= 0:
            return
        c[:, 2] *= min(1.0, ratio) * (1.0-3e-14)
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        c[:, :2] -= (lo+hi)/2
        c *= (2.0/perimeter)*(1.0-3e-14)
        if float(np.sum(c[:, 2])) <= best_score:
            return
        data = {'circles': c.tolist()}
        score = exact(data)
        if score > best_score:
            best_score, best = score, data

    # A certified fallback, also useful when the supplied start is malformed.
    fallback = np.array([[float(k % 3), float(k // 3), 0.49] for k in range(21)])
    retain(fallback)

    class StopSearch(Exception):
        pass

    def model(axis_count, upper_counts, offset=0.0):
        if time.monotonic() >= deadline:
            raise StopSearch()
        groups = []
        for row, count in enumerate([axis_count] + list(upper_counts)):
            if time.monotonic() >= deadline:
                raise StopSearch()
            for col in range(count):
                if time.monotonic() >= deadline:
                    raise StopSearch()
                groups.append((row, 2.0*col-(count-1)+ (offset if row % 2 else 0.0), row*np.sqrt(3.0)))
        nv = 2*axis_count + 3*(len(groups)-axis_count) + 1
        X = np.zeros((21, nv)); Y = X.copy(); R = X.copy()
        z = np.zeros(nv)
        mirror = np.arange(21)
        representatives = []
        circle = 0; var = 0
        for row, x, y in groups:
            if time.monotonic() >= deadline:
                raise StopSearch()
            representatives.append(circle)
            if row == 0:
                X[circle, var] = 1; R[circle, var+1] = 1
                z[var:var+2] = [x, 0.99]
                circle += 1; var += 2
            else:
                X[circle:circle+2, var] = 1
                Y[circle, var+1] = 1; Y[circle+1, var+1] = -1
                R[circle:circle+2, var+2] = 1
                mirror[circle] = circle+1; mirror[circle+1] = circle
                z[var:var+3] = [x, y, 0.99]
                circle += 2; var += 3
        c = np.column_stack((X@z, Y@z, R@z))
        lo = np.min(c[:, :2]-c[:, 2,None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2,None], axis=0)
        scale = 2/np.sum(hi-lo)
        z *= scale
        for k in representatives:
            if time.monotonic() >= deadline:
                raise StopSearch()
            ix = np.flatnonzero(X[k])[0]
            z[ix] -= lo[0]*scale
        z[-1] = (hi[0]-lo[0])*scale
        seen = set(); pairs = []
        for a,b in zip(ii,jj):
            if time.monotonic() >= deadline:
                raise StopSearch()
            key = tuple(sorted((int(mirror[a]),int(mirror[b]))))
            if key not in seen:
                pairs.append((a,b)); seen.add((int(a),int(b)))
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        DX = X[a]-X[b]; DY = Y[a]-Y[b]; RS = R[a]+R[b]
        rows = []; constants = []
        e = np.zeros(nv); e[-1] = 1
        bounds = [(None,None)]*nv
        bounds[-1] = (0.08,1.92)
        for k in representatives:
            if time.monotonic() >= deadline:
                raise StopSearch()
            rows.extend([X[k]-R[k], e-X[k]-R[k], -0.5*e-Y[k]-R[k]])
            constants.extend([0.,0.,1.])
            bounds[np.flatnonzero(R[k])[0]] = (params['radius_floor'],0.7)
            if np.any(Y[k]):
                bounds[np.flatnonzero(Y[k])[0]] = (0.,0.99)
        B = np.asarray(rows); bc = np.asarray(constants)
        obj = -np.sum(R,axis=0)
        cache_z = None; cache_val = None; cache_jac = None
        def calculate(v):
            nonlocal cache_z, cache_val, cache_jac
            if time.monotonic() >= local_end:
                raise StopSearch()
            if cache_z is None or not np.array_equal(v,cache_z):
                dx = DX@v; dy = DY@v
                dist = np.maximum(np.hypot(dx,dy),1e-16)
                cache_val = np.concatenate((dist-RS@v, B@v+bc))
                cache_jac = np.vstack(((dx/dist)[:,None]*DX+(dy/dist)[:,None]*DY-RS,B))
                cache_z = v.copy()
            return cache_val,cache_jac
        def unpack(v):
            return np.column_stack((X@v,Y@v,R@v))
        return z, obj, bounds, calculate, unpack

    local_end = deadline
    trajectories = []
    families = [(5,(4,4),0.0),(3,(4,5),0.0),(3,(5,4),0.0),(3,(3,3,3),0.0),(1,(5,5),0.0)]

    def run(m, z, duration):
        nonlocal local_end
        local_end = min(deadline-0.015, time.monotonic()+duration)
        _, obj, bounds, calc, unpack = m
        last = z.copy()
        peak = -float('inf'); peak_z = last.copy()
        def callback(v):
            nonlocal last, peak, peak_z
            if time.monotonic() >= local_end:
                raise StopSearch()
            last = v.copy()
            c = unpack(v)
            retain(c)
            values,_ = calc(v)
            merit = float(-obj@v) - 10*max(0.,-float(np.min(values)))
            if merit > peak:
                peak = merit; peak_z = v.copy()
        try:
            result = minimize(lambda v: float(obj@v), z, jac=lambda v: obj,
                method='SLSQP', bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0], 'jac':lambda v:calc(v)[1]},
                callback=callback, options={'maxiter':180,'ftol':2e-12,'disp':False})
            last = result.x
        except StopSearch:
            pass
        if time.monotonic() < deadline:
            retain(unpack(last))
        return peak, peak_z

    for ac, counts, offset in families:
        if time.monotonic() >= deadline-0.10:
            break
        try:
            m = model(ac, counts, offset)
            retain(m[4](m[0]))
            merit,z = run(m,m[0],params['screen_seconds'])
            trajectories.append((merit,m,z))
        except StopSearch:
            break
    trajectories.sort(key=lambda t:t[0],reverse=True)
    turn = 0
    while trajectories and time.monotonic() < deadline-0.035:
        if time.monotonic() >= deadline:
            break
        index = 0 if turn % 3 != 2 or len(trajectories)==1 else 1
        merit,m,z = trajectories[index]
        if turn > 0:
            z = z + rng.normal(0,params['restart_noise'],size=z.size)
            for k,(low,high) in enumerate(m[2]):
                if time.monotonic() >= deadline-0.02:
                    break
                if low is not None: z[k] = max(z[k],low)
                if high is not None: z[k] = min(z[k],high)
        new_merit,new_z = run(m,z,min(0.23,deadline-time.monotonic()-0.02))
        if new_merit >= merit:
            trajectories[index] = (new_merit,m,new_z)
        trajectories.sort(key=lambda t:t[0],reverse=True)
        turn += 1
    return best
