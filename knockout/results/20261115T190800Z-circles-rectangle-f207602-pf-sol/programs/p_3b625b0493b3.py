import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'time_fraction': (0.69, 0.4, 0.72),
    'stage_seconds': (0.29, 0.1, 0.5),
    'iterations': (140, 50, 240)
}
N = 21
II, JJ = np.triu_indices(N, 1)
RR = np.arange(210)
IX = 3*np.arange(N)
IR = IX+2


def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=np.float64)
        if c.shape != (21,3) or not np.isfinite(c).all() or np.any(c[:,2] <= 0):
            return -float('inf')
        x = [Fraction(float(t)) for t in c[:,0]]
        y = [Fraction(float(t)) for t in c[:,1]]
        r = [Fraction(float(t)) for t in c[:,2]]
        perimeter = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
        perimeter += max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
        if perimeter > 2:
            return -float('inf')
        for a,b in itertools.combinations(c,2):
            if np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2) < a[2]+b[2]:
                return -float('inf')
        return float(np.sum(c[:,2]))
    except Exception:
        return -float('inf')


def repair(c):
    c = np.asarray(c, dtype=float).reshape(21,3).copy()
    if not np.isfinite(c).all():
        return None
    c[:,2] = np.maximum(c[:,2],1e-10)
    d = c[II,:2]-c[JJ,:2]
    ratio = np.min(np.sqrt(np.sum(d*d,axis=1))/(c[II,2]+c[JJ,2]))
    if ratio <= 0:
        return None
    c[:,2] *= min(1.,float(ratio))*(1.-3e-13)
    lo = np.min(c[:,:2]-c[:,2,None],axis=0)
    hi = np.max(c[:,:2]+c[:,2,None],axis=0)
    c[:,:2] -= lo
    c *= (2.-2e-13)/np.sum(hi-lo)
    return {'circles':c.tolist()}


class Stop(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start+params['time_fraction']*budget_s
    best = init
    best_score = exact_score(init)
    pending = None
    pending_score = best_score
    rng = np.random.default_rng(seed)
    grad = np.zeros(64)
    grad[IR] = -1.

    def consider(c):
        nonlocal pending, pending_score
        candidate = repair(c)
        if candidate is not None:
            score = float(np.sum(np.asarray(candidate['circles'])[:,2]))
            if score > pending_score:
                pending_score = score
                pending = candidate

    def values(z):
        c = z[:63].reshape(21,3)
        x,y,r = c.T
        dx = x[II]-x[JJ]
        dy = y[II]-y[JJ]
        return np.concatenate((np.sqrt(dx*dx+dy*dy)-r[II]-r[JJ],x-r,z[63]-x-r,y-r,2.-z[63]-y-r))

    def jac(z):
        c = z[:63].reshape(21,3)
        dxy = c[II,:2]-c[JJ,:2]
        d = np.maximum(np.sqrt(np.sum(dxy*dxy,axis=1)),1e-20)
        j = np.zeros((294,64))
        j[RR,3*II] = dxy[:,0]/d
        j[RR,3*JJ] = -dxy[:,0]/d
        j[RR,3*II+1] = dxy[:,1]/d
        j[RR,3*JJ+1] = -dxy[:,1]/d
        j[RR,3*II+2] = -1
        j[RR,3*JJ+2] = -1
        t = 210+np.arange(21)
        j[t,IX] = 1
        j[t,IR] = -1
        j[t+21,IX] = -1
        j[t+21,IR] = -1
        j[t+21,63] = 1
        j[t+42,IX+1] = 1
        j[t+42,IR] = -1
        j[t+63,IX+1] = -1
        j[t+63,IR] = -1
        j[t+63,63] = -1
        return j

    def solve(z, A, q, selected, seconds):
        until = min(deadline,time.monotonic()+seconds)
        counter = 0
        g = A.T@grad
        def check():
            if time.monotonic() >= until:
                raise Stop()
        def objective(v):
            check()
            return float(g@v)
        def derivative(v):
            check()
            return g
        def cons(v):
            check()
            return values(A@v)[selected]
        def cj(v):
            check()
            return jac(A@v)[selected]@A
        def callback(v):
            nonlocal counter
            check()
            counter += 1
            if counter%3 == 0:
                consider((A@v)[:63].reshape(21,3))
        try:
            result = minimize(objective,q,jac=derivative,method='SLSQP',
                constraints={'type':'ineq','fun':cons,'jac':cj},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':3e-12,'disp':False})
            consider((A@result.x)[:63].reshape(21,3))
        except (Stop,ValueError,FloatingPointError):
            pass

    def normalize(c):
        c = c.copy()
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        f = 2./np.sum(hi-lo)
        c[:,:2] -= lo
        c *= f
        return np.r_[c.ravel(),(hi[0]-lo[0])*f]

    def unrestricted(c,seconds):
        z = normalize(c)
        solve(z,np.eye(64),z,np.arange(294),seconds)

    patterns = [
        [(1,0),(1,1),(1,2),(0,0),(2,0)],
        [(1,0),(1,1),(1,2),(0,1),(2,1)],
        [(0,0),(0,2),(1,1),(2,0),(2,2)]
    ]
    fallback = np.array([[1+2*x,1+2*y,1.] for y in range(4) for x in range(4)] +
        [[2+2*x,2+2*y,np.sqrt(2)-1] for x,y in patterns[0]],dtype=float)/8.
    consider(fallback)

    if best_score > 2.2 and time.monotonic() < deadline:
        unrestricted(np.asarray(init['circles'],dtype=float),0.42*budget_s)

    for pattern in patterns:
        if time.monotonic() >= deadline-0.08:
            break
        c = np.array([[1+2*x,1+2*y,1.] for y in range(4) for x in range(4)] +
            [[2+2*x,2+2*y,np.sqrt(2)-1] for x,y in pattern],dtype=float)/8.
        z = np.r_[c.ravel(),1.]
        permutation = []
        for i in range(21):
            if time.monotonic() >= deadline:
                break
            target = np.array([1.-c[i,0],c[i,1]])
            permutation.append(int(np.argmin(np.sum((c[:,:2]-target)**2,axis=1))))
        if len(permutation) != 21:
            break
        columns = []
        used = set()
        for i in range(21):
            if time.monotonic() >= deadline:
                break
            if i in used:
                continue
            k = permutation[i]
            used.update((i,k))
            if i != k:
                v = np.zeros(64); v[3*i]=1; v[3*k]=-1
                columns.append(v)
            for coord in (1,2):
                v = np.zeros(64); v[3*i+coord]=1; v[3*k+coord]=1
                columns.append(v)
        if len(used) != 21:
            break
        v = np.zeros(64); v[63]=1
        for i in range(21):
            if time.monotonic() >= deadline:
                break
            if permutation[i] == i:
                v[3*i]=0.5
            elif i > permutation[i]:
                v[3*i]=1.
        columns.append(v)
        A = np.array(columns).T
        q = np.linalg.lstsq(A,z,rcond=None)[0]
        pair_lookup = {(int(i),int(j)):k for k,(i,j) in enumerate(zip(II,JJ))}
        selection = []
        for k,(i,j) in enumerate(zip(II,JJ)):
            if time.monotonic() >= deadline:
                break
            partner = pair_lookup[tuple(sorted((permutation[i],permutation[j])))]
            if k <= partner:
                selection.append(k)
        for block in range(4):
            if time.monotonic() >= deadline:
                break
            for i in range(21):
                if time.monotonic() >= deadline:
                    break
                k = 210+21*block+i
                otherblock = 1-block if block < 2 else block
                partner = 210+21*otherblock+permutation[i]
                if k <= partner:
                    selection.append(k)
        if time.monotonic() < deadline:
            solve(z,A,q,np.array(selection,dtype=int),float(params['stage_seconds']))

    attempt = 0
    while time.monotonic() < deadline-0.06:
        anchor = pending if pending is not None else best
        if anchor is None:
            break
        c = np.asarray(anchor['circles'],dtype=float).copy()
        if attempt:
            c[:,:2] += rng.normal(0.,0.009,size=(21,2))
        unrestricted(c,min(0.3,deadline-time.monotonic()))
        attempt += 1
    if pending is not None:
        score = exact_score(pending)
        if score > best_score:
            best = pending
    return best
