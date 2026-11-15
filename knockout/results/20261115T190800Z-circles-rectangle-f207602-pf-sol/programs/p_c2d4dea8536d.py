import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.linalg import lstsq

PARAMS = {
    'restart_noise': (0.035, 0.001, 0.12),
    'lp_steps': (2, 1, 8),
    'polish_steps': (65, 10, 160),
    'contact_threshold': (2e-7, 1e-10, 1e-4, 'log'),
    'safety': (3e-15, 1e-16, 1e-11, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)
ROWS = np.arange(M)
C = np.zeros(64)
C[42:63] = -1
A = np.zeros((105, 64))
B = np.zeros(105)
for k in range(N):
    A[k, k] = -1
    A[k, 42+k] = 1
    A[21+k, k] = 1
    A[21+k, 42+k] = 1
    A[21+k, 63] = -1
    A[42+k, 21+k] = -1
    A[42+k, 42+k] = 1
    A[63+k, 21+k] = 1
    A[63+k, 42+k] = 1
    A[63+k, 63] = 1
    B[63+k] = 2
    A[84+k, 42+k] = -1
    B[84+k] = -1e-8

class Expired(Exception):
    pass

def perimeter(p, r):
    xx = [Fraction(float(v)) for v in p[:, 0]]
    yy = [Fraction(float(v)) for v in p[:, 1]]
    rr = [Fraction(float(v)) for v in r]
    return (max(x+t for x,t in zip(xx,rr))-min(x-t for x,t in zip(xx,rr))
            + max(y+t for y,t in zip(yy,rr))-min(y-t for y,t in zip(yy,rr)))

def valid(p, r):
    if p.shape != (21, 2) or r.shape != (21,):
        return False
    if not np.isfinite(p).all() or not np.isfinite(r).all() or np.min(r) <= 0:
        return False
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    return not np.any(d < r[I]+r[J]) and perimeter(p,r) <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    safety = float(params['safety'])

    def check():
        if time.monotonic() >= deadline:
            raise Expired()

    def save(z):
        nonlocal best, best_score
        check()
        p = np.column_stack((z[:21], z[21:42])).copy()
        r = z[42:63].copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all() or np.min(r) <= 0:
            return
        d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
        ratio = min(1.0, float(np.min(d/(r[I]+r[J]))))
        if ratio <= 0:
            return
        r *= ratio*(1-safety)
        p -= np.min(p, axis=0)
        per = perimeter(p,r)
        if per > 2:
            scale = (2.0/float(per))*(1-safety)
            p *= scale
            r *= scale
        score = float(np.sum(r))
        if score > best_score and valid(p,r):
            best_score = score
            best = {'circles': np.column_stack((p,r)).tolist()}

    if init is not None:
        arr = np.asarray(init['circles'], dtype=float)
        p = arr[:,:2].copy()
        r = arr[:,2].copy()
        if valid(p,r):
            best_score = float(np.sum(r))
    else:
        p = np.array([[0.12+0.25*(k%7),0.12+0.25*(k//7)] for k in range(21)])
        r = np.full(21,0.1)
    lo = np.min(p-r[:,None],axis=0)
    sides = np.max(p+r[:,None],axis=0)-lo
    scale = 2/float(np.sum(sides))
    p = (p-lo)*scale
    r *= scale
    sides *= scale

    try:
        check()
        # Choose the reflection direction with the smallest matching residual.
        errors = []
        for axis in (0,1):
            check()
            ref = p.copy()
            ref[:,axis] = sides[axis]-ref[:,axis]
            dd = np.sum((ref[:,None,:]-p[None,:,:])**2,axis=2)
            dd += (r[:,None]-r[None,:])**2
            errors.append(float(np.max(np.min(dd,axis=1))))
        if errors[0] < errors[1]:
            p = p[:,::-1].copy()
            sides = sides[::-1].copy()
        z0 = np.r_[p[:,0],p[:,1],r,sides[0]]
        groups = []
        used = set()
        symmetric = min(errors) < 1e-7
        for k in range(N):
            check()
            if k in used:
                continue
            if abs(p[k,1]-sides[1]/2) < 3e-4:
                groups.append((k,))
                used.add(k)
            else:
                target = np.array([p[k,0],sides[1]-p[k,1]])
                choices = [j for j in range(N) if j not in used and j != k]
                if not choices:
                    symmetric = False
                    break
                j = min(choices, key=lambda j: float(np.sum((p[j]-target)**2)+(r[j]-r[k])**2))
                if np.linalg.norm(p[j]-target) > 1e-3:
                    symmetric = False
                if p[k,1] < p[j,1]:
                    groups.append((k,j))
                else:
                    groups.append((j,k))
                used.update((k,j))
        if symmetric and len(used) == N:
            # q = x for each orbit, lower y for pairs, radius for each orbit, width.
            ng = len(groups)
            npair = sum(len(g)==2 for g in groups)
            dim = 2*ng+npair+1
            T = np.zeros((64,dim))
            offset = np.zeros(64)
            wi = dim-1
            yi = ng
            for a,g in enumerate(groups):
                check()
                for k in g:
                    T[k,a] = 1
                    T[42+k,ng+npair+a] = 1
                if len(g)==1:
                    offset[21+g[0]] = 1
                    T[21+g[0],wi] = -0.5
                else:
                    k,j = g
                    T[21+k,yi] = 1
                    offset[21+j] = 2
                    T[21+j,wi] = -1
                    T[21+j,yi] = -1
                    yi += 1
            T[63,wi] = 1
            q0 = lstsq(T,z0-offset,check_finite=False)[0]
        else:
            T = np.eye(64)
            offset = np.zeros(64)
            dim = 64
            q0 = z0.copy()
        ar = A@T
        br = B-A@offset
        cr = C@T

        def geom(q, derivative=False):
            check()
            z = offset+T@q
            dx = z[I]-z[J]
            dy = z[21+I]-z[21+J]
            dd = np.maximum(np.sqrt(dx*dx+dy*dy),1e-30)
            g = np.r_[dd-z[42+I]-z[42+J],br-ar@q]
            if not derivative:
                return g
            mat = np.zeros((M,64))
            mat[ROWS,I] = dx/dd
            mat[ROWS,J] = -dx/dd
            mat[ROWS,21+I] = dy/dd
            mat[ROWS,21+J] = -dy/dd
            mat[ROWS,42+I] = -1
            mat[ROWS,42+J] = -1
            return np.vstack((mat@T,-ar))

        def callback(q):
            check()
            z = offset+T@q
            if -float(C@z) > best_score:
                save(z)

        def refine(q):
            # An isostatic contact graph can be polished without an optimizer's
            # objective-based stopping test.
            g = geom(q)
            active = np.flatnonzero(g < float(params['contact_threshold']))
            if len(active) < dim-2:
                return q
            for iteration in range(5):
                check()
                g = geom(q)
                jac = geom(q,True)[active]
                step = lstsq(jac,-g[active],cond=1e-11,check_finite=False)[0]
                if np.max(np.abs(step)) > 0.002:
                    break
                candidate = q+step
                if np.linalg.norm(geom(candidate)[active]) > np.linalg.norm(g[active])*1.01+1e-15:
                    break
                q = candidate
                callback(q)
                if np.max(np.abs(step)) < 3e-15:
                    break
            return q

        q = q0.copy()
        callback(q)
        attempt = 0
        while time.monotonic() < deadline:
            check()
            if attempt:
                # Perturb several whole mirror orbits together, rather than
                # breaking the symmetry with independent circle noise.
                if best is not None:
                    arr = np.asarray(best['circles'],dtype=float)
                    pp,rr = arr[:,:2].copy(),arr[:,2].copy()
                    lo = np.min(pp-rr[:,None],axis=0)
                    ss = np.max(pp+rr[:,None],axis=0)-lo
                    if errors[0] < errors[1]:
                        pp = pp[:,::-1].copy()
                        lo = lo[::-1].copy()
                        ss = ss[::-1].copy()
                    pp -= lo
                    zz = np.r_[pp[:,0],pp[:,1],rr,ss[0]]
                    q = lstsq(T,zz-offset,check_finite=False)[0]
                else:
                    q = q0.copy()
                amplitude = float(params['restart_noise'])*(1+0.45*((attempt-1)%3))
                delta = rng.normal(size=dim)
                # Sparse coordinated surgery changes a handful of orbit contacts.
                mask = rng.random(dim) < 0.42
                q += amplitude*delta*mask
            for lp_iteration in range(int(params['lp_steps'])):
                check()
                if time.monotonic()+0.05 >= deadline:
                    break
                g = geom(q)
                jac = geom(q,True)
                result = linprog(cr,A_ub=-jac,b_ub=g-jac@q,
                                 bounds=[(None,None)]*dim,method='highs',
                                 options={'time_limit':max(0.01,min(0.12,deadline-time.monotonic()-0.02))})
                check()
                if not result.success:
                    break
                q = result.x
                callback(q)
            result = minimize(lambda v:(float(cr@v),cr),q,jac=True,
                              method='SLSQP',constraints={'type':'ineq','fun':geom,
                                                         'jac':lambda v:geom(v,True)},
                              callback=callback,
                              options={'maxiter':int(params['polish_steps']),
                                       'ftol':8e-15,'disp':False})
            check()
            q = result.x
            callback(q)
            refine(q)
            attempt += 1
    except Expired:
        pass
    if best is not None and best_score > -np.inf:
        return best
    return {'circles': [[0.1+0.2*(i%7),0.1+0.2*(i//7),0.01] for i in range(21)]}
