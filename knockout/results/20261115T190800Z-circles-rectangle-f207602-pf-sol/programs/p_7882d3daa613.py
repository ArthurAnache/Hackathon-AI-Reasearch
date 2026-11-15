import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'search_fraction': (0.70, 0.4, 0.75),
    'iterations': (100, 20, 200),
    'noise': (0.045, 0.005, 0.15),
    'cluster_size': (5, 3, 12),
}
N = 21
II, JJ = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + float(params['search_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf

    def check():
        if time.monotonic() >= deadline:
            raise Deadline()

    def score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
        if np.any(d < a[II, 2]+a[JJ, 2]):
            return -np.inf
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                     + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
        if perimeter > 2:
            return -np.inf
        return float(np.sum(a[:, 2]))

    def retain(a, original=None):
        nonlocal best, best_score
        s = score(a)
        if s > best_score:
            best_score = s
            best = original if original is not None else {'circles': a.tolist()}

    if init is not None:
        a0 = np.asarray(init['circles'], dtype=float)
        retain(a0, init)
    else:
        xx, yy = np.meshgrid(np.arange(3), np.arange(7))
        p = np.column_stack((xx.ravel()+0.5*(yy.ravel()%2), yy.ravel()*np.sqrt(3)/2))
        a0 = np.column_stack((p, np.full(N, 0.499)))
        lo = np.min(a0[:, :2]-a0[:, 2,None], axis=0)
        hi = np.max(a0[:, :2]+a0[:, 2,None], axis=0)
        a0[:, :2] -= (lo+hi)/2
        a0 *= (2-1e-13)/np.sum(hi-lo)
        retain(a0)

    def centered(a):
        a = a.copy()
        lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
        a[:, :2] -= (lo+hi)/2
        return a, float(hi[0]-lo[0])

    a0, width = centered(a0)
    # Determine which coordinate is the reflection-normal direction.
    options = []
    for axis in range(2):
        check()
        reflected = a0.copy()
        reflected[:, axis] *= -1
        cost = np.sum((a0[:,None,:]-reflected[None,:,:])**2, axis=2)
        _, perm = linear_sum_assignment(cost)
        options.append((float(cost[np.arange(N),perm].sum()), axis, perm))
    error, axis, mirror = min(options, key=lambda t:t[0])
    if axis == 0:
        a0[:, [0,1]] = a0[:, [1,0]]
        width = 2-width
    use_symmetry = error < 1e-8 and np.all(mirror[mirror] == np.arange(N))

    # z = (x coordinates, y coordinates, radii, width), with a centered box.
    groups = []
    columns = []
    if use_symmetry:
        seen = set()
        for i in range(N):
            check()
            if i in seen:
                continue
            j = int(mirror[i])
            seen.add(i); seen.add(j)
            members = [i] if i == j else [i,j]
            group = []
            c = np.zeros(64); c[members] = 1
            group.append(len(columns)); columns.append(c)
            if i != j:
                c = np.zeros(64); c[N+i] = 1; c[N+j] = -1
                group.append(len(columns)); columns.append(c)
            c = np.zeros(64); c[2*N+np.array(members)] = 1
            group.append(len(columns)); columns.append(c)
            groups.append((members, group))
        c = np.zeros(64); c[-1] = 1; columns.append(c)
        B = np.column_stack(columns)
        keep = []
        for k,(i,j) in enumerate(zip(II,JJ)):
            check()
            reflected_pair = tuple(sorted((int(mirror[i]),int(mirror[j]))))
            if (int(i),int(j)) <= reflected_pair:
                keep.append(k)
        pi, pj = II[keep], JJ[keep]
    else:
        B = np.eye(64)
        pi, pj = II, JJ
        groups = [([i],[i,N+i,2*N+i]) for i in range(N)]
    dim = B.shape[1]
    norm = np.sum(B*B, axis=0)
    z0 = np.r_[a0[:,0],a0[:,1],a0[:,2],width]
    base = (B.T@z0)/norm

    # Box and positivity constraints remain linear in quotient coordinates.
    A = np.zeros((5*N+2,64)); b = np.zeros(5*N+2)
    for i in range(N):
        check()
        A[4*i,i] = 1; A[4*i,2*N+i] = -1; A[4*i,-1] = .5
        A[4*i+1,i] = -1; A[4*i+1,2*N+i] = -1; A[4*i+1,-1] = .5
        A[4*i+2,N+i] = 1; A[4*i+2,2*N+i] = -1; A[4*i+2,-1] = -.5; b[4*i+2] = 1
        A[4*i+3,N+i] = -1; A[4*i+3,2*N+i] = -1; A[4*i+3,-1] = -.5; b[4*i+3] = 1
        A[4*N+i,2*N+i] = 1; b[4*N+i] = -1e-7
    A[-2,-1] = 1; b[-2] = -.05
    A[-1,-1] = -1; b[-1] = 1.95
    C = A@B
    unique = np.unique(np.column_stack((C,b)),axis=0)
    C, b = unique[:,:-1], unique[:,-1]
    objective_jac = -np.sum(B[2*N:3*N],axis=0)
    rows = np.arange(len(pi))

    def objective(q):
        check()
        return float(objective_jac@q)

    def constraints(q):
        check()
        z = B@q
        dx = z[pi]-z[pj]; dy = z[N+pi]-z[N+pj]
        rr = z[2*N+pi]+z[2*N+pj]
        return dx*dx+dy*dy-rr*rr

    def jacobian(q):
        check()
        z = B@q
        dx = 2*(z[pi]-z[pj]); dy = 2*(z[N+pi]-z[N+pj])
        rr = -2*(z[2*N+pi]+z[2*N+pj])
        return (dx[:,None]*(B[pi]-B[pj]) + dy[:,None]*(B[N+pi]-B[N+pj])
                + rr[:,None]*(B[2*N+pi]+B[2*N+pj]))

    def repair(q):
        z = B@q
        p = np.column_stack((z[:N],z[N:2*N]))
        r = z[2*N:3*N].copy()
        if not np.isfinite(z).all() or np.any(r <= 0):
            return
        d = np.sqrt((p[II,0]-p[JJ,0])**2+(p[II,1]-p[JJ,1])**2)
        shrink = min(1.,float(np.min(d/(r[II]+r[JJ]))))
        if shrink <= 0:
            return
        r *= shrink*(1-6e-15)
        lo = np.min(p-r[:,None],axis=0); hi = np.max(p+r[:,None],axis=0)
        scale = (2-2e-14)/float(np.sum(hi-lo))
        p = (p-(lo+hi)/2)*scale
        r *= scale
        candidate = np.column_stack((p,r))
        if np.sum(r) > best_score:
            retain(candidate)

    latest = [base.copy()]
    count = [0]
    def callback(q):
        check()
        latest[0] = q.copy()
        count[0] += 1
        if count[0] % 5 == 0 and -objective_jac@q > best_score:
            repair(q)

    attempt = 0
    try:
        while time.monotonic() < deadline:
            check()
            q = base.copy()
            if attempt:
                centers = np.array([np.mean((B@base)[np.array(g[0])]) for g in groups])
                pivot = int(rng.integers(len(groups)))
                z = B@base
                gp = np.array([[np.mean(z[np.array(m)]),np.mean(np.abs(z[N+np.array(m)]))] for m,g in groups])
                dist = np.sum((gp-gp[pivot])**2,axis=1)
                size = min(len(groups),int(params['cluster_size']) + (attempt%3)-1)
                chosen = np.argsort(dist)[:size]
                amplitude = float(params['noise'])*(0.65+0.45*(attempt%4))
                for k in chosen:
                    check()
                    members, inds = groups[int(k)]
                    for col in inds[:-1]:
                        check()
                        q[col] += rng.normal(0,amplitude)
                    q[inds[-1]] *= float(rng.uniform(.80,1.06))
                q[-1] += rng.normal(0,amplitude*.5)
            latest[0] = q.copy()
            result = minimize(objective,q,jac=lambda q:objective_jac,method='SLSQP',
                constraints=[{'type':'ineq','fun':lambda q:C@q+b,'jac':lambda q:C},
                             {'type':'ineq','fun':constraints,'jac':jacobian}],
                callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-14,'disp':False})
            repair(result.x)
            if attempt == 0 and np.max(np.abs(constraints(result.x)[constraints(result.x)<0]),initial=0) < 1e-8:
                base = result.x.copy()
            attempt += 1
    except Deadline:
        repair(latest[0])
    return best
