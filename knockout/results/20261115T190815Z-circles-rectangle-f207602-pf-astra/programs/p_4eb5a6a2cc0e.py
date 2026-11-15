import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'loading': (0.65, 0.15, 1.2),
    'opening': (0.035, 0.005, 0.12),
    'groups': (3, 3, 6),
    'paths': (2, 1, 5),
    'preservation': (5.0, 1.0, 20.0),
    'loaded_fraction': (0.30, 0.15, 0.45)
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
K = np.arange(N)
R = np.arange(M)

class Stop(Exception):
    pass


def exact_score(c):
    if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((c[I, 0]-c[J, 0])**2 + (c[I, 1]-c[J, 1])**2)
    if np.any(d < c[I, 2]+c[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), c[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), c[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), c[:, 2]))
    width = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
    height = max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
    return float(np.sum(c[:, 2])) if width+height <= 2 else -np.inf


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    best_c = None

    def check(end=None):
        if time.monotonic() >= (deadline if end is None else min(end, deadline)):
            raise Stop()

    def offer(c, original=None):
        nonlocal best, best_score, best_c
        score = exact_score(c)
        if score > best_score:
            best_score = score
            best_c = c.copy()
            best = original if original is not None else {'circles': c.tolist()}

    if init is not None:
        offer(np.asarray(init['circles'], dtype=float), init)
    if best_c is None:
        k = np.arange(N)
        c = np.column_stack((0.1+0.2*(k % 7), 0.1+0.2*(k//7), np.full(N, 0.1-1e-14)))
        offer(c)

    def encode(c):
        c = c.copy()
        lo = np.min(c[:, :2]-c[:, 2, None], axis=0)
        hi = np.max(c[:, :2]+c[:, 2, None], axis=0)
        c[:, :2] -= lo
        factor = 2.0/float(np.sum(hi-lo))
        c *= factor
        return np.r_[c[:, 0], c[:, 1], c[:, 2], (hi[0]-lo[0])*factor]

    def certify(z):
        check()
        c = np.column_stack((z[:21], z[21:42], z[42:63]))
        if not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return
        c[:, :2] -= np.mean(c[:, :2], axis=0)
        d = np.sqrt((c[I, 0]-c[J, 0])**2 + (c[I, 1]-c[J, 1])**2)
        f = min(1.0, float(np.min(d/(c[I, 2]+c[J, 2]))))
        if f <= 0:
            return
        c[:, 2] *= f*(1-8e-15)
        ext = np.max(c[:, :2]+c[:, 2,None], axis=0)-np.min(c[:, :2]-c[:, 2,None], axis=0)
        c *= (2.0/float(np.sum(ext)))*(1-8e-15)
        if np.sum(c[:, 2]) > best_score:
            offer(c)

    boundary = np.zeros((84, 64))
    boundary[K,K] = 1
    boundary[K,42+K] = -1
    boundary[21+K,K] = -1
    boundary[21+K,42+K] = -1
    boundary[21+K,-1] = 1
    boundary[42+K,21+K] = 1
    boundary[42+K,42+K] = -1
    boundary[63+K,21+K] = -1
    boundary[63+K,42+K] = -1
    boundary[63+K,-1] = -1
    offset = np.r_[np.zeros(63), np.full(21, 2.0)]

    def geometry(z):
        check()
        dx = z[I]-z[J]
        dy = z[21+I]-z[21+J]
        d = np.sqrt(dx*dx+dy*dy)
        g = np.r_[d-z[42+I]-z[42+J], boundary@z+offset]
        den = np.maximum(d, 1e-30)
        a = np.zeros((M+84, 64))
        a[R,I] = dx/den
        a[R,J] = -dx/den
        a[R,21+I] = dy/den
        a[R,21+J] = -dy/den
        a[R,42+I] = -1
        a[R,42+J] = -1
        a[M:] = boundary
        return g,a

    bounds = [(None,None)]*42+[(1e-5,0.65)]*21+[(0.2,1.8)]

    def solve(z, weights, end):
        last = z.copy()
        grad = np.r_[np.zeros(42), -weights, 0.0]
        cache = [None, None]
        count = 0
        def calc(v):
            check(end)
            if cache[0] is None or not np.array_equal(cache[0],v):
                cache[:] = [v.copy(), geometry(v)]
            return cache[1]
        def fun(v):
            check(end)
            return float(grad@v)
        def callback(v):
            nonlocal last, count
            check(end)
            last = v.copy()
            count += 1
            if count % 5 == 0:
                certify(last)
        try:
            result = minimize(fun,z,jac=lambda v:grad,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0], 'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':100,'ftol':1e-13,'disp':False})
            last = result.x.copy()
        except Stop:
            pass
        if time.monotonic() < deadline-0.004:
            certify(last)
        return last

    try:
        check()
        ones = np.ones(N)
        solve(encode(best_c), ones, min(deadline, started+0.075*budget_s))
        base = encode(best_c)
        g,a = geometry(base)
        active = np.flatnonzero(g < 2e-6)
        if len(active) < 15:
            solve(base, ones, deadline-0.008)
            return best
        grad = np.r_[np.zeros(42), -ones, 0.0]
        stress = np.maximum(0.0, lstsq(a[active].T, -grad, cond=1e-10, lapack_driver='gelsy')[0])

        # Build reflection-related contact groups, but allow asymmetric loads.
        x,y,r = base[:21],base[21:42],base[42:63]
        h = 2-base[-1]
        metric = (x[:,None]-x[None,:])**2+(y[:,None]+y[None,:]-h)**2+(r[:,None]-r[None,:])**2
        mirror = np.argmin(metric,axis=1)
        reflection_ok = float(np.max(np.min(metric,axis=1))) < 1e-7
        lookup = np.full((21,21),-1,dtype=int)
        lookup[I,J] = R
        lookup[J,I] = R
        positions = {int(row): k for k,row in enumerate(active)}
        groups = []
        centres = []
        costs = []
        used = set()
        for row in active:
            check()
            row = int(row)
            if row in used:
                continue
            if row < M:
                reflected = int(lookup[mirror[I[row]],mirror[J[row]]])
                cx = (x[I[row]]+x[J[row]])/2
                cy = (y[I[row]]+y[J[row]])/2
            else:
                wall,circle = divmod(row-M,N)
                rw = 3 if wall == 2 else 2 if wall == 3 else wall
                reflected = M+rw*N+int(mirror[circle])
                cx,cy = x[circle],y[circle]
            group = [positions[row]]
            used.add(row)
            if reflection_ok and reflected in positions and reflected not in used:
                group.append(positions[reflected])
                used.add(reflected)
            group = np.asarray(group,dtype=int)
            groups.append(group)
            centres.append((cx,abs(cy-h/2)))
            costs.append(0.025+float(np.sum(stress[group])))
        centres = np.asarray(centres)
        prob = 1/np.sqrt(np.asarray(costs))
        prob /= prob.sum()
        paths = int(params['paths'])
        for attempt in range(paths):
            check()
            now = time.monotonic()
            if deadline-now < 0.09:
                break
            end = now+(deadline-now)/(paths-attempt)-0.006
            first = int(rng.choice(len(groups),p=prob))
            distance = np.sum((centres-centres[first])**2,axis=1)
            p = prob*np.exp(-distance/0.08)
            p[first] = 0
            p /= p.sum()
            others = rng.choice(len(groups),min(int(params['groups'])-1,len(groups)-1),replace=False,p=p)
            selected = np.concatenate([groups[int(q)] for q in np.r_[first,others]])
            target = np.zeros(len(active))
            target[selected] = float(params['opening'])*(1.0 if attempt%2 == 0 else -0.7)
            importance = np.full(len(active),float(params['preservation']))
            importance[selected] = 1.0
            z = base.copy()
            for step in range(3):
                check(end)
                values,matrix = geometry(z)
                delta = lstsq(matrix[active]*importance[:,None], (target-values[active])*importance,
                              cond=1e-9,lapack_driver='gelsy')[0]
                largest = np.max(np.abs(delta))
                if largest > 0.09:
                    delta *= 0.09/largest
                falling = delta[42:63] < 0
                alpha = 1.0
                if np.any(falling):
                    alpha = min(alpha,0.8*float(np.min((z[42:63][falling]-0.005)/(-delta[42:63][falling]))))
                if alpha <= 0:
                    break
                z += alpha*delta
            # Load exactly the circles touched by the released constraints.
            involvement = np.zeros(N)
            for index in selected:
                check(end)
                row = int(active[index])
                if row < M:
                    involvement[I[row]] += 1
                    involvement[J[row]] += 1
                else:
                    involvement[(row-M)%N] += 1
            involvement /= max(1.0,float(involvement.max()))
            weights = 1+float(params['loading'])*(1 if attempt%2 == 0 else -1)*involvement
            weights = np.maximum(weights,0.1)
            weights /= weights.mean()
            load_end = min(end,time.monotonic()+float(params['loaded_fraction'])*(end-time.monotonic()))
            loaded = solve(z,weights,load_end)
            check()
            solve(loaded,ones,end)
    except Stop:
        pass
    return best
