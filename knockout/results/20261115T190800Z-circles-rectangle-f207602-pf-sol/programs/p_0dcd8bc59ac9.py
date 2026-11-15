import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'time_fraction': (0.70, 0.4, 0.72),
    'iterations': (100, 30, 180),
    'perturbation': (0.025, 0.005, 0.10),
}
N = 21
II, JJ = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')
    best_array = None

    def check():
        if time.monotonic() >= deadline:
            raise Deadline()

    def score(a):
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
        if np.any(d < a[II, 2]+a[JJ, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                     + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
        if perimeter > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    def retain(a, original=None):
        nonlocal best, best_score, best_array
        s = score(a)
        if s > best_score:
            best_score = s
            best_array = a.copy()
            best = original if original is not None else {'circles': a.tolist()}

    def repair(a):
        if not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return
        a = a.copy()
        p = a[:, :2]
        r = a[:, 2]
        d = np.sqrt((p[II, 0]-p[JJ, 0])**2 + (p[II, 1]-p[JJ, 1])**2)
        factor = min(1.0, float(np.min(d/(r[II]+r[JJ]))))
        if factor <= 0:
            return
        r *= factor*(1-2e-14)
        lo = np.min(p-r[:, None], axis=0)
        hi = np.max(p+r[:, None], axis=0)
        scale = (2-8e-14)/float(np.sum(hi-lo))
        p -= (lo+hi)/2
        a *= scale
        if float(np.sum(a[:, 2])) > best_score:
            retain(a)

    if init is not None:
        try:
            retain(np.asarray(init['circles'], dtype=float), init)
        except (ValueError, KeyError, TypeError):
            pass

    def lattice(counts):
        rows = []
        mirror = []
        middle = len(counts)//2
        offsets = []
        running = 0
        for count in counts:
            check()
            offsets.append(running)
            running += count
        for k, count in enumerate(counts):
            check()
            for j in range(count):
                check()
                rows.append([2*j-(count-1), (k-middle)*np.sqrt(3.0), 0.999])
                mirror.append(offsets[len(counts)-1-k]+j)
        a = np.asarray(rows, dtype=float)
        lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
        a *= (2-1e-12)/np.sum(hi-lo)
        return a, np.asarray(mirror, dtype=int)

    def infer(a):
        a = a.copy()
        lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
        a[:, :2] -= (lo+hi)/2
        options = []
        for axis in range(2):
            check()
            reflected = a.copy()
            reflected[:, axis] *= -1
            cost = np.sum((a[:,None,:]-reflected[None,:,:])**2, axis=2)
            _, perm = linear_sum_assignment(cost)
            options.append((float(cost[np.arange(N),perm].sum()), axis, perm))
        error, axis, perm = min(options, key=lambda t:t[0])
        if error < 1e-8 and np.all(perm[perm] == np.arange(N)):
            if axis == 0:
                a[:, [0,1]] = a[:, [1,0]]
            return a, perm
        return a, None

    def optimize(a, mirror, perturb=False):
        check()
        columns = []
        if mirror is None:
            B = np.eye(64)
            pi, pj = II, JJ
        else:
            seen = set()
            for i in range(N):
                check()
                if i in seen:
                    continue
                j = int(mirror[i])
                seen.update((i,j))
                members = [i] if i == j else [i,j]
                c = np.zeros(64); c[members] = 1
                columns.append(c)
                if i != j:
                    c = np.zeros(64); c[N+i] = 1; c[N+j] = -1
                    columns.append(c)
                c = np.zeros(64); c[2*N+np.array(members)] = 1
                columns.append(c)
            c = np.zeros(64); c[-1] = 1
            columns.append(c)
            B = np.column_stack(columns)
            keep = []
            for k, (i,j) in enumerate(zip(II,JJ)):
                check()
                if (int(i),int(j)) <= tuple(sorted((int(mirror[i]),int(mirror[j])))):
                    keep.append(k)
            pi, pj = II[keep], JJ[keep]
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        a = a.copy()
        a[:,:2] -= (lo+hi)/2
        z = np.r_[a[:,0],a[:,1],a[:,2],hi[0]-lo[0]]
        q = (B.T@z)/np.sum(B*B, axis=0)
        if perturb:
            dz = np.zeros(64)
            dz[:42] = rng.normal(0, float(params['perturbation']), 42)
            dz[42:63] = rng.normal(0, 0.004, 21)
            q += (B.T@dz)/np.sum(B*B, axis=0)
        A = np.zeros((107,64)); b = np.zeros(107)
        for i in range(N):
            check()
            A[4*i,i] = 1; A[4*i,42+i] = -1; A[4*i,-1] = .5
            A[4*i+1,i] = -1; A[4*i+1,42+i] = -1; A[4*i+1,-1] = .5
            A[4*i+2,21+i] = 1; A[4*i+2,42+i] = -1; A[4*i+2,-1] = -.5; b[4*i+2] = 1
            A[4*i+3,21+i] = -1; A[4*i+3,42+i] = -1; A[4*i+3,-1] = -.5; b[4*i+3] = 1
            A[84+i,42+i] = 1; b[84+i] = -1e-7
        A[-2,-1] = 1; b[-2] = -.05
        A[-1,-1] = -1; b[-1] = 1.95
        unique = np.unique(np.column_stack((A@B,b)), axis=0)
        C, b = unique[:,:-1], unique[:,-1]
        X = B[pi]-B[pj]
        Y = B[N+pi]-B[N+pj]
        R = B[42+pi]+B[42+pj]
        grad = -np.sum(B[42:63], axis=0)
        last = [q.copy()]
        counter = [0]

        def objective(v):
            check()
            return float(grad@v)

        def constraints(v):
            check()
            dx, dy, rr = X@v, Y@v, R@v
            return np.r_[C@v+b, dx*dx+dy*dy-rr*rr]

        def jac(v):
            check()
            return np.vstack((C, 2*((X@v)[:,None]*X+(Y@v)[:,None]*Y-(R@v)[:,None]*R)))

        def save(v):
            z = B@v
            repair(np.column_stack((z[:21],z[21:42],z[42:63])))

        def callback(v):
            check()
            last[0] = v.copy()
            counter[0] += 1
            if counter[0] % 4 == 0 and -grad@v > best_score:
                save(v)

        try:
            result = minimize(objective, q, jac=lambda v:grad, method='SLSQP',
                constraints={'type':'ineq','fun':constraints,'jac':jac},
                callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':3e-13,'disp':False})
            save(result.x)
        except Deadline:
            save(last[0])
            raise

    try:
        if best_array is not None and best_score > 1.5:
            a, mirror = infer(best_array)
            optimize(a, mirror)
        families = [(4,5,3,5,4), (5,4,3,4,5), (4,4,5,4,4), (3,3,3,3,3,3,3)]
        for counts in families:
            check()
            a, mirror = lattice(counts)
            repair(a)
            optimize(a, mirror)
        while time.monotonic() < deadline:
            check()
            if best_array is not None:
                a, mirror = infer(best_array)
                optimize(a, mirror, True)
            else:
                a, mirror = lattice(families[0])
                optimize(a, mirror, True)
    except Deadline:
        pass
    return best
