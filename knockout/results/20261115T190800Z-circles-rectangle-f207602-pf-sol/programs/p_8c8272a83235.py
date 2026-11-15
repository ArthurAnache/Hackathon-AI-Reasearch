import time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

PARAMS = {
    'samples': (220, 40, 800),
    'angles': (12, 4, 24),
    'iterations': (32, 8, 80),
    'trust': (0.16, 0.03, 0.5),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)


def perimeter(a):
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
            +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))


def score(a):
    if a.shape != (N,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
        return -np.inf
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    if np.any(d < a[I,2]+a[J,2]):
        return -np.inf
    if perimeter(a) > 2:
        return -np.inf
    return float(np.sum(a[:,2]))


def normalize(a):
    a = a.copy()
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    s = 2.0 / np.sum(hi-lo)
    a[:,:2] = (a[:,:2]-lo)*s
    a[:,2] *= s
    return a


def repair(a):
    a = normalize(a)
    d = np.sqrt((a[I,0]-a[J,0])**2 + (a[I,1]-a[J,1])**2)
    q = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
    a[:,2] *= q*(1.0-2e-14)
    p = perimeter(a)
    if p > 2:
        a *= (2.0/float(p))*(1.0-2e-14)
    return a


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.zeros((N,3))
        a[:,0] = np.arange(N)/11.0
        a[:,2] = 1.0/22.0
        a = repair(a)
        init = {'circles': a.tolist()}
    base = np.asarray(init['circles'],dtype=float)
    best = init
    best_score = score(base)
    base = normalize(base)
    p = base[:,:2].copy()
    r = base[:,2].copy()
    center = (np.min(p-r[:,None],axis=0)+np.max(p+r[:,None],axis=0))/2
    p -= center
    bins = [[], [], [], []]
    # Each bin keeps a different scale of conformal deformation.
    for k in range(int(params['samples'])):
        if time.monotonic() >= end - 0.18:
            break
        group = k % 4
        theta = rng.uniform(0,2*np.pi)
        if group == 0:
            pole = rng.uniform(-0.65,0.65,2)
        else:
            rho = (0.75, 1.25, 2.8)[group-1]*np.exp(rng.uniform(-0.28,0.28))
            pole = rho*np.array([np.cos(theta),np.sin(theta)])
        delta = p-pole
        den = np.sum(delta*delta,axis=1)-r*r
        if np.min(den) <= 1e-7:
            continue
        pp = delta/den[:,None]
        rr = r/den
        best_local = None
        local_score = -np.inf
        phase = rng.uniform(0,np.pi/(2*int(params['angles'])))
        for j in range(int(params['angles'])):
            if time.monotonic() >= end - 0.16:
                break
            ang = phase + j*np.pi/(2*int(params['angles']))
            c,s = np.cos(ang),np.sin(ang)
            rot = np.array([[c,-s],[s,c]])
            xy = pp @ rot.T
            lo = np.min(xy-rr[:,None],axis=0)
            extent = np.max(xy+rr[:,None],axis=0)-lo
            factor = 2.0/extent.sum()
            val = float(rr.sum()*factor)
            if val > local_score:
                local_score = val
                best_local = np.column_stack(((xy-lo)*factor,rr*factor))
        if best_local is not None:
            bins[group].append((local_score,best_local))
            bins[group].sort(key=lambda z:-z[0])
            bins[group] = bins[group][:2]

    # LP variables: all x, all y, all radii, and box width W; H=2-W.
    nv = 3*N+1
    A = np.zeros((M+4*N,nv))
    b = np.zeros(M+4*N)
    rows = np.arange(M)
    A[rows,2*N+I] = 1
    A[rows,2*N+J] = 1
    for i in range(N):
        z = M+4*i
        A[z,i] = -1; A[z,2*N+i] = 1
        A[z+1,i] = 1; A[z+1,2*N+i] = 1; A[z+1,-1] = -1
        A[z+2,N+i] = -1; A[z+2,2*N+i] = 1
        A[z+3,N+i] = 1; A[z+3,2*N+i] = 1; A[z+3,-1] = 1
        b[z+3] = 2
    objective = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    starts = []
    # Interleave distant and nearby poles rather than spending all time in one regime.
    for rank in range(2):
        if time.monotonic() >= end:
            break
        for g in (1,0,2,3):
            if time.monotonic() >= end:
                break
            if len(bins[g]) > rank:
                starts.append(bins[g][rank][1])
    if not starts:
        starts = [base]
    for a0 in starts:
        if time.monotonic() >= end - 0.025:
            break
        a = a0.copy()
        prev = -np.inf
        for it in range(int(params['iterations'])):
            if time.monotonic() >= end - 0.025:
                break
            xy = a[:,:2]
            dx = xy[I,0]-xy[J,0]
            dy = xy[I,1]-xy[J,1]
            dd = np.maximum(np.sqrt(dx*dx+dy*dy),1e-30)
            nx,ny = dx/dd,dy/dd
            A[rows,I] = -nx; A[rows,J] = nx
            A[rows,N+I] = -ny; A[rows,N+J] = ny
            trust = float(params['trust'])
            bounds = [(max(0.0,float(v)-trust),min(2.0,float(v)+trust)) for v in xy[:,0]]
            bounds += [(max(0.0,float(v)-trust),min(2.0,float(v)+trust)) for v in xy[:,1]]
            bounds += [(1e-7,1.0)]*N + [(0.05,1.95)]
            remaining = max(0.001,end-time.monotonic()-0.015)
            res = linprog(objective,A_ub=A,b_ub=b,bounds=bounds,method='highs',
                          options={'time_limit':remaining,'primal_feasibility_tolerance':1e-9,
                                   'dual_feasibility_tolerance':1e-9})
            if not res.success:
                break
            z = res.x
            a = np.column_stack((z[:N],z[N:2*N],z[2*N:3*N]))
            val = float(np.sum(a[:,2]))
            if it % 4 == 3 or val-prev < 2e-11:
                candidate = repair(a)
                exact = score(candidate)
                if exact > best_score:
                    best_score = exact
                    best = {'circles':candidate.tolist()}
            if val-prev < 2e-12:
                break
            prev = val
        if time.monotonic() < end:
            candidate = repair(a)
            exact = score(candidate)
            if exact > best_score:
                best_score = exact
                best = {'circles':candidate.tolist()}
    return best
