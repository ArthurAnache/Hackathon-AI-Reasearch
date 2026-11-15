import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize
from scipy.spatial import Delaunay

PARAMS = {
    'penalty_share': (0.23, 0.10, 0.35),
    'mu': (18.0, 4.0, 100.0, 'log'),
    'growth': (9.0, 2.0, 30.0),
    'hole_rank': (0, 0, 5),
    'ftol': (2e-13, 1e-14, 1e-8, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
T = np.arange(len(I))
K = np.arange(N)
G = np.r_[np.zeros(42), -np.ones(21), 0.0]

class Stop(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    best_a = None

    def guard(limit=end):
        if time.monotonic() >= limit:
            raise Stop()

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return -np.inf
        left, right, bottom, top = [], [], [], []
        for x, y, r in a:
            guard()
            x, y, r = Fraction(float(x)), Fraction(float(y)), Fraction(float(r))
            left.append(x-r)
            right.append(x+r)
            bottom.append(y-r)
            top.append(y+r)
        if max(right)-min(left)+max(top)-min(bottom) > Fraction(2):
            return -np.inf
        return float(np.sum(a[:, 2]))

    def offer(a, original=None):
        nonlocal best, best_score, best_a
        guard()
        if a is None:
            return
        score = exact(a)
        if score > best_score:
            best_score = score
            best_a = a.copy()
            best = original if original is not None else {'circles': a.tolist()}

    def repair(a):
        guard()
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return None
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        ratio = min(1.0, float(np.min(d/(a[I, 2]+a[J, 2]))))
        if ratio <= 0:
            return None
        a[:, 2] *= ratio * (1-1e-14)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        span = float(np.sum(hi-lo))
        if span <= 0:
            return None
        a *= (2/span)*(1-1e-14)
        return a

    def unpack(z):
        return np.column_stack((z[:21], z[21:42], z[42:63]))

    def pack(a):
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        p = a[:, :2]-lo
        return np.r_[p[:, 0], p[:, 1], a[:, 2], hi[0]-lo[0]]

    def constraints(z):
        x, y, r, w = z[:21], z[21:42], z[42:63], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy)
        return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

    def jacobian(z):
        dx, dy = z[I]-z[J], z[21+I]-z[21+J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy), 1e-15)
        a = np.zeros((294, 64))
        a[T, I], a[T, J] = dx/d, -dx/d
        a[T, 21+I], a[T, 21+J] = dy/d, -dy/d
        a[T, 42+I], a[T, 42+J] = -1, -1
        a[210+K, K], a[210+K, 42+K] = 1, -1
        a[231+K, K], a[231+K, 42+K], a[231+K, -1] = -1, -1, 1
        a[252+K, 21+K], a[252+K, 42+K] = 1, -1
        a[273+K, 21+K], a[273+K, 42+K], a[273+K, -1] = -1, -1, -1
        return a

    bounds = [(None, None)]*42 + [(1e-7, 0.6)]*21 + [(0.1, 1.9)]

    def polish(z, limit):
        latest = [z.copy()]
        count = [0]
        def obj(q):
            guard(limit)
            return -float(np.sum(q[42:63]))
        def cf(q):
            guard(limit)
            return constraints(q)
        def cj(q):
            guard(limit)
            return jacobian(q)
        def cb(q):
            guard(limit)
            latest[0] = q.copy()
            count[0] += 1
            if count[0] % 5 == 0:
                offer(repair(unpack(q)))
        try:
            result = minimize(obj, z, jac=lambda q: G, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': cf, 'jac': cj}, callback=cb,
                options={'maxiter': 200, 'ftol': float(params['ftol']), 'disp': False})
            latest[0] = result.x
        except Stop:
            pass
        if time.monotonic() < end-0.003:
            offer(repair(unpack(latest[0])))
        return latest[0]

    def penalty(z, mu):
        x, y, r, w = z[:21], z[21:42], z[42:63], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        v = np.maximum(r[I]+r[J]-d, 0)
        b0 = np.maximum(r-x, 0)
        b1 = np.maximum(r+x-w, 0)
        b2 = np.maximum(r-y, 0)
        b3 = np.maximum(r+y-(2-w), 0)
        c = 2*mu*v
        gx = np.bincount(I, weights=-c*dx/d, minlength=N)+np.bincount(J, weights=c*dx/d, minlength=N)
        gy = np.bincount(I, weights=-c*dy/d, minlength=N)+np.bincount(J, weights=c*dy/d, minlength=N)
        gr = -np.ones(N)+np.bincount(I, weights=c, minlength=N)+np.bincount(J, weights=c, minlength=N)
        gx += 2*mu*(-b0+b1)
        gy += 2*mu*(-b2+b3)
        gr += 2*mu*(b0+b1+b2+b3)
        value = -r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3)
        return value, np.r_[gx, gy, gr, 2*mu*(-b1.sum()+b3.sum())]

    def reconstruct(a):
        guard()
        z = pack(a)
        a = unpack(z)
        w, h = z[-1], 2-z[-1]
        points = a[:, :2]
        radii = a[:, 2]
        triangles = Delaunay(points).simplices
        holes = []
        for ids in triangles:
            guard()
            p = points[ids]
            r = radii[ids]
            matrix = 2*(p[1:]-p[0])
            if abs(np.linalg.det(matrix)) < 1e-12:
                continue
            rhs = np.sum(p[1:]**2, axis=1)-np.sum(p[0]**2)-r[1:]**2+r[0]**2
            uv = np.linalg.solve(matrix, np.column_stack((rhs, -2*(r[1:]-r[0]))))
            u, v = uv[:, 0], uv[:, 1]
            q = u-p[0]
            aa, bb, cc = v@v-1, 2*(q@v-r[0]), q@q-r[0]**2
            roots = []
            if abs(aa) < 1e-12:
                if abs(bb) > 1e-12:
                    roots = [-cc/bb]
            else:
                disc = bb*bb-4*aa*cc
                if disc >= 0:
                    root = np.sqrt(disc)
                    roots = [(-bb-root)/(2*aa), (-bb+root)/(2*aa)]
            for rr in roots:
                guard()
                if rr <= 1e-5 or rr > 0.3:
                    continue
                centre = u+v*rr
                wall = min(centre[0], w-centre[0], centre[1], h-centre[1])
                clearance = float(np.min(np.linalg.norm(points-centre, axis=1)-radii))
                radius = min(wall, clearance)
                if radius > 1e-5:
                    holes.append((radius, centre.copy()))
        if not holes:
            return z
        wallgap = np.minimum.reduce((points[:, 0]-radii, w-points[:, 0]-radii,
                                    points[:, 1]-radii, h-points[:, 1]-radii))
        boundary = np.flatnonzero(wallgap < 1e-5)
        if len(boundary) == 0:
            boundary = np.argsort(wallgap)[:8]
        choices = []
        for rr, centre in holes:
            guard()
            dist = np.linalg.norm(points[boundary]-centre, axis=1)
            eligible = boundary[dist > 3.0*np.mean(radii)]
            if len(eligible) == 0:
                continue
            costs = radii[eligible]
            remove = int(eligible[np.argmin(costs)])
            # Favor large interstices and a low-cost boundary vacancy.
            merit = rr-0.65*radii[remove]
            choices.append((merit+float(rng.uniform(0, 1e-8)), rr, centre, remove))
        if not choices:
            return z
        choices.sort(key=lambda t: t[0], reverse=True)
        choice = choices[min(int(params['hole_rank']), len(choices)-1)]
        _, rr, centre, remove = choice
        a[remove, :2] = centre
        a[remove, 2] = max(rr*0.95, 0.015)
        # The deleted boundary circle creates room for collective relaxation;
        # the inserted circle initially fits its interstice without scattering.
        return pack(a)

    try:
        if init is not None:
            offer(np.asarray(init['circles'], dtype=float), init)
        if best_a is None:
            circles = []
            for y in range(4):
                guard()
                for x in range(4):
                    guard()
                    circles.append([2*x+1, 2*y+1, 1.0])
            for x, y in ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)):
                guard()
                circles.append([2*x+2, 2*y+2, np.sqrt(2)-1])
            offer(repair(np.array(circles, dtype=float)))
        if best_a is None:
            return best
        initial = pack(best_a)
        polish(initial, min(end, start+0.08*budget_s))
        guard()
        z = reconstruct(best_a)
        smooth_end = min(end-0.03, start+(0.08+float(params['penalty_share']))*budget_s)
        mu = float(params['mu'])
        for stage in range(2):
            guard()
            if time.monotonic() >= smooth_end:
                break
            limit = time.monotonic()+(smooth_end-time.monotonic())/(2-stage)
            latest = [z.copy()]
            def fun(q):
                guard(limit)
                return penalty(q, mu)
            def cb(q):
                guard(limit)
                latest[0] = q.copy()
            try:
                result = minimize(fun, z, jac=True, method='L-BFGS-B', bounds=bounds,
                    callback=cb, options={'maxiter': 180, 'maxls': 18, 'ftol': 1e-11})
                z = result.x
            except Stop:
                z = latest[0]
            guard()
            offer(repair(unpack(z)))
            mu *= float(params['growth'])
        if time.monotonic() < end-0.015:
            polish(z, end-0.008)
    except Stop:
        pass
    return best
