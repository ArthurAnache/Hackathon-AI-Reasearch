import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"first_share": (0.37, 0.2, 0.6), "jitter": (0.025, 0.0, 0.1)}

class StopSearch(Exception):
    pass

def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=float)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        if max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))+max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r)) > 2:
            return -float('inf')
        for i,j in itertools.combinations(range(21),2):
            if np.sqrt((c[i,0]-c[j,0])**2+(c[i,1]-c[j,1])**2) < c[i,2]+c[j,2]:
                return -float('inf')
        return float(np.sum(c[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best_data = init
    best_exact = exact_score(init)
    best_score = best_exact
    best_c = None
    best_z = None
    ii,jj = np.triu_indices(21,1)
    # Coordinates of twelve representatives: nine paired circles and three axis circles.
    X = np.zeros((21,34)); Y = np.zeros((21,34)); R = np.zeros((21,34))
    for i in range(9):
        X[i,i] = X[i+9,i] = 1
        Y[i,12+i] = 1; Y[i+9,12+i] = -1
        R[i,21+i] = R[i+9,21+i] = 1
    for i in range(3):
        X[18+i,9+i] = 1; R[18+i,30+i] = 1
    pairs = []
    for i in range(9):
        for j in range(i+1,9): pairs.append((i,j))
        for j in range(18,21): pairs.append((i,j))
        for j in range(i,9): pairs.append((i,j+9))
    pairs.extend([(18,19),(18,20),(19,20)])
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    DX = X[a]-X[b]; DY = Y[a]-Y[b]; SR = R[a]+R[b]
    walls = []; constants = []
    for i in list(range(9))+list(range(18,21)):
        for sign in (-1,1):
            row = sign*X[i]-R[i]
            row = row.copy(); row[33] += .5
            walls.append(row); constants.append(0.)
        row = -Y[i]-R[i]; row = row.copy(); row[33] -= .5
        walls.append(row); constants.append(1.)
    W = np.array(walls); wc = np.array(constants)
    objective = np.zeros(34); objective[21:30] = -2; objective[30:33] = -1
    bounds = [(None,None)]*12 + [(1e-7,None)]*9 + [(1e-7,None)]*12 + [(0.15,1.85)]

    def unpack(z):
        return np.column_stack((X@z,Y@z,R@z))

    def retain(z):
        nonlocal best_score,best_c,best_z
        if time.monotonic() >= deadline:
            raise StopSearch()
        c = unpack(z)
        if not np.isfinite(c).all() or np.min(c[:,2]) <= 0:
            return
        d = np.hypot(c[ii,0]-c[jj,0],c[ii,1]-c[jj,1])
        ratio = min(1.,float(np.min(d/(c[ii,2]+c[jj,2]))))
        if ratio <= 0:
            return
        c[:,2] *= ratio*(1.-2e-13)
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        total = float(np.sum(hi-lo))
        if total <= 0: return
        c[:,:2] -= (lo+hi)/2
        c *= (2./total)*(1.-2e-13)
        score = float(np.sum(c[:,2]))
        if score > best_score:
            best_score = score; best_c = c.copy(); best_z = z.copy()

    def lattice(counts, shift=0.):
        z = np.zeros(34)
        k = 0
        for row,count in enumerate(counts):
            if time.monotonic() >= deadline: raise StopSearch()
            xs = 2*np.arange(count)-(count-1)
            if count == 3: xs = xs + (1 if row%2 == 0 else 0) + shift
            z[k:k+count] = xs
            z[12+k:12+k+count] = (row+1)*np.sqrt(3.)
            k += count
        z[9:12] = [-2,0,2]
        z[21:33] = 1
        c = unpack(z)
        width = np.ptp(c[:,0])+2
        height = np.ptp(c[:,1])+2
        scale = 2/(width+height)
        z[:33] *= scale
        z[33] = width*scale
        return z

    local_end = deadline
    cache_z = None; cache_v = None; cache_j = None
    def calculate(z):
        nonlocal cache_z,cache_v,cache_j
        if time.monotonic() >= local_end: raise StopSearch()
        if cache_z is not None and np.array_equal(z,cache_z): return
        dx = DX@z; dy = DY@z
        d = np.maximum(np.hypot(dx,dy),1e-15)
        cache_v = np.concatenate((d-SR@z,W@z+wc))
        cache_j = np.vstack((DX*(dx/d)[:,None]+DY*(dy/d)[:,None]-SR,W))
        cache_z = z.copy()
    def con(z):
        calculate(z); return cache_v
    def jac(z):
        calculate(z); return cache_j
    def run(z, seconds, iterations=160):
        nonlocal local_end,cache_z
        if time.monotonic() >= deadline: return
        local_end = min(deadline,time.monotonic()+seconds)
        cache_z = None
        try:
            retain(z)
            result = minimize(lambda v: float(objective@v),z,jac=lambda v: objective,
                              method='SLSQP',bounds=bounds,
                              constraints={'type':'ineq','fun':con,'jac':jac},
                              callback=retain,
                              options={'maxiter':iterations,'ftol':2e-12,'disp':False})
            retain(result.x)
        except StopSearch:
            pass
        except (ValueError,FloatingPointError):
            pass

    try:
        available = max(0.,deadline-time.monotonic())
        run(lattice([4,5]),available*params['first_share'])
        if time.monotonic() < deadline:
            run(lattice([5,4]),available*.27)
        if time.monotonic() < deadline:
            run(lattice([3,3,3]),available*.18)
        attempt = 0
        while time.monotonic() < deadline and best_z is not None:
            z = best_z.copy()
            if attempt:
                z[:21] += rng.normal(0,params['jitter'],21)
            remaining = deadline-time.monotonic()
            if remaining < .025: break
            run(z,min(.24,remaining),100)
            attempt += 1
    except StopSearch:
        pass

    if best_c is not None:
        candidate = {'circles':best_c.tolist()}
        score = exact_score(candidate)
        if score > best_exact:
            best_data = candidate; best_exact = score
    if best_data is None or best_exact == -float('inf'):
        # A deterministic safe fallback, independent of the supplied candidate.
        c = []
        for i in range(21):
            if time.monotonic() > start + .94*budget_s:
                # The loop has only 21 iterations; complete the fixed-size fallback.
                pass
            c.append([(i%7)*.099,(i//7)*.099,.049])
        fallback = {'circles':c}
        if exact_score(fallback) > best_exact: best_data = fallback
    return best_data
