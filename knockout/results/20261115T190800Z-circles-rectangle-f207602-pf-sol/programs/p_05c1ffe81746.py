import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'poles': (180, 20, 500),
    'rotations': (12, 4, 24),
    'iterations': (110, 20, 300),
    'distortion': (1.5, 1.05, 4.0),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    r = list(map(Fraction, map(float, a[:, 2])))
    perimeter = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                 +max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(np.sum(a[:, 2])) if perimeter <= 2 else -np.inf

def normalized(p, r):
    lo = np.min(p-r[:, None], axis=0)
    sides = np.max(p+r[:, None], axis=0)-lo
    s = 2.0 / sides.sum()
    return (p-lo)*s, r*s, sides[0]*s

def repaired(p, r, deadline):
    if time.monotonic() >= deadline or not np.isfinite(r).all() or np.min(r) <= 0:
        return None
    p = p.copy()
    r = r.copy()
    d = np.sqrt(np.sum((p[I]-p[J])**2, axis=1))
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    r *= factor*(1-2e-14)
    p, r, _ = normalized(p, r)
    a = np.column_stack((p,r))
    a *= 1-2e-14
    for k in range(5):
        if time.monotonic() >= deadline:
            return None
        if np.isfinite(exact_score(a)):
            return a
        a[:, 2] *= 1-1e-12
        a *= 1-1e-13
    return None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k%7+0.5)/7, (k//7+0.5)/3, 0.07] for k in range(N)])
        a = repaired(a[:,:2], a[:,2], deadline)
        if a is None:
            return {'circles': [[(k%7)*0.1,(k//7)*0.1,0.01] for k in range(N)]}
        init = {'circles': a.tolist()}
    a = np.asarray(init['circles'], dtype=float)
    best = init
    best_score = exact_score(a)
    p0, r0, _ = normalized(a[:,:2], a[:,2])
    center = (np.min(p0-r0[:,None],axis=0)+np.max(p0+r0[:,None],axis=0))/2
    pool = []
    nr = int(params['rotations'])
    angles = np.arange(nr)*np.pi/nr
    cosine, sine = np.cos(angles), np.sin(angles)
    screen_end = start + 0.12*budget_s
    for k in range(int(params['poles'])):
        if time.monotonic() >= min(screen_end, deadline):
            break
        theta = rng.uniform(0,2*np.pi)
        rho = np.exp(rng.uniform(np.log(0.12),np.log(2.0)))
        pole = center + rho*np.array([np.cos(theta),np.sin(theta)])
        v = p0-pole
        den = np.sum(v*v,axis=1)-r0*r0
        if np.min(den) <= 1e-7:
            continue
        if np.max(den)/np.min(den) < float(params['distortion']):
            continue
        p = v/den[:,None]
        r = r0/den
        xx = p[:,0,None]*cosine-p[:,1,None]*sine
        yy = p[:,0,None]*sine+p[:,1,None]*cosine
        widths = np.max(xx+r[:,None],axis=0)-np.min(xx-r[:,None],axis=0)
        heights = np.max(yy+r[:,None],axis=0)-np.min(yy-r[:,None],axis=0)
        scores = 2*r.sum()/(widths+heights)
        j = int(np.argmax(scores))
        pp, rr, w = normalized(np.column_stack((xx[:,j],yy[:,j])),r)
        pool.append((float(scores[j]), pp, rr, w))
    pool.sort(key=lambda t:t[0],reverse=True)
    if not pool:
        pool = [(float(r0.sum()),p0,r0,float(np.ptp(p0[:,0])+2*np.max(r0)))]
    objective = np.r_[np.zeros(42),-np.ones(21),0.0]
    cache_z = None
    cache_c = None
    cache_j = None
    def calculate(z):
        nonlocal cache_z, cache_c, cache_j
        if time.monotonic() >= deadline:
            raise StopSearch
        if cache_z is not None and np.array_equal(z,cache_z):
            return cache_c, cache_j
        x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
        dx,dy = x[I]-x[J],y[I]-y[J]
        rr = r[I]+r[J]
        c = np.r_[dx*dx+dy*dy-rr*rr,x-r,w-x-r,y-r,2-w-y-r]
        jac = np.zeros((M+84,64))
        rows = np.arange(M)
        jac[rows,I]=2*dx
        jac[rows,J]=-2*dx
        jac[rows,21+I]=2*dy
        jac[rows,21+J]=-2*dy
        jac[rows,42+I]=-2*rr
        jac[rows,42+J]=-2*rr
        q = np.arange(21)
        jac[M+q,q]=1
        jac[M+q,42+q]=-1
        jac[M+21+q,q]=-1
        jac[M+21+q,42+q]=-1
        jac[M+21+q,63]=1
        jac[M+42+q,21+q]=1
        jac[M+42+q,42+q]=-1
        jac[M+63+q,21+q]=-1
        jac[M+63+q,42+q]=-1
        jac[M+63+q,63]=-1
        cache_z,cache_c,cache_j=z.copy(),c,jac
        return c,jac
    selected = []
    for _,p,r,w in pool:
        if time.monotonic() >= deadline:
            break
        # Avoid spending the short budget on virtually identical inversions.
        signature = np.sort(r/r.sum())
        if any(np.linalg.norm(signature-s)<0.004 for s in selected):
            continue
        selected.append(signature)
        z = np.r_[p[:,0],p[:,1],r,w]
        last = [z.copy()]
        def callback(v):
            if time.monotonic() >= deadline:
                raise StopSearch
            last[0]=v.copy()
        try:
            res = minimize(lambda v: float(objective@v), z, jac=lambda v:objective,
                constraints={'type':'ineq','fun':lambda v:calculate(v)[0],
                             'jac':lambda v:calculate(v)[1]},
                bounds=[(None,None)]*42+[(1e-7,0.5)]*21+[(0.05,1.95)],
                method='SLSQP',callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            z=res.x
        except StopSearch:
            z=last[0]
        if time.monotonic() >= deadline:
            break
        aa = repaired(np.column_stack((z[:21],z[21:42])),z[42:63],deadline)
        if aa is not None:
            score=exact_score(aa)
            if score>best_score:
                best_score=score
                best={'circles':aa.tolist()}
    return best
