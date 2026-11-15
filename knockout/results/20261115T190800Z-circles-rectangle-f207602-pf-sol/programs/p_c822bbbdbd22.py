import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'samples': (1600, 200, 6000),
    'restarts': (4, 1, 12),
    'pole_scale': (0.8, 0.1, 3.0, 'log'),
}
N = 21
I, J = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    if w+h > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def normalized(a):
    a = a.copy()
    lo = np.min(a[:, :2]-a[:, 2,None], axis=0)
    hi = np.max(a[:, :2]+a[:, 2,None], axis=0)
    a[:, :2] -= lo
    a *= (2.0-3e-14)/np.sum(hi-lo)
    a[:, 2] *= 1.0-3e-14
    return a

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i%5)*0.18+0.06, (i//5)*0.18+0.06, 0.05] for i in range(N)])
        a = normalized(a)
        best = {'circles': a.tolist()}
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
        best = init
    bestscore = exact_score(a)
    base = normalized(a)
    center = (np.min(base[:, :2]-base[:, 2,None],axis=0)+np.max(base[:, :2]+base[:, 2,None],axis=0))/2
    p = base[:, :2]-center
    r = base[:, 2]
    pool = []
    def retain(b):
        nonlocal best, bestscore
        b = normalized(b)
        s = exact_score(b)
        if s > bestscore:
            bestscore = s
            best = {'circles': b.tolist()}
        return b
    # Sample inversion poles in batches, including poles in bounded interstices.
    sampled = 0
    while sampled < int(params['samples']) and time.monotonic() < start+0.19*budget_s:
        count = min(128, int(params['samples'])-sampled)
        poles = rng.normal(size=(count,2))*float(params['pole_scale'])
        angles = rng.uniform(0, np.pi/2, count)
        delta = p[None,:,:]-poles[:,None,:]
        den = np.sum(delta*delta,axis=2)-r[None,:]**2
        good = np.min(den,axis=1)>1e-7
        den = np.maximum(den,1e-15)
        q = delta/den[:,:,None]
        rr = r[None,:]/den
        cs, sn = np.cos(angles)[:,None], np.sin(angles)[:,None]
        xx = q[:,:,0]*cs-q[:,:,1]*sn
        yy = q[:,:,0]*sn+q[:,:,1]*cs
        wh = np.max(xx+rr,axis=1)-np.min(xx-rr,axis=1)+np.max(yy+rr,axis=1)-np.min(yy-rr,axis=1)
        scores = 2*np.sum(rr,axis=1)/wh
        scores[~good] = -np.inf
        for k in np.argsort(scores)[-5:]:
            if time.monotonic() >= end:
                break
            if np.isfinite(scores[k]):
                b = np.column_stack((xx[k],yy[k],rr[k]))
                pool.append((float(scores[k]), b))
        sampled += count
    pool.sort(key=lambda t:-t[0])
    starts = []
    signatures = []
    for score,b in pool:
        if time.monotonic() >= end:
            break
        b = normalized(b)
        sig = np.sort(b[:,2])
        if all(np.linalg.norm(sig-s)>0.008 for s in signatures):
            signatures.append(sig)
            starts.append(b)
        if len(starts)>=int(params['restarts']):
            break
    # Analytic constraints in x,y,r,W; separating circle inequalities use distances.
    rows = np.arange(len(I))
    def objective(z):
        if time.monotonic() >= end:
            raise Deadline()
        g = np.zeros(64)
        g[42:63] = -1
        return -float(np.sum(z[42:63])),g
    def constraints(z):
        if time.monotonic() >= end:
            raise Deadline()
        x,y,rr,w = z[:21],z[21:42],z[42:63],z[63]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.concatenate((d-rr[I]-rr[J], x-rr,w-x-rr,y-rr,2-w-y-rr))
    def jacobian(z):
        if time.monotonic() >= end:
            raise Deadline()
        x,y = z[:21],z[21:42]
        dx,dy=x[I]-x[J],y[I]-y[J]
        d=np.sqrt(dx*dx+dy*dy+1e-30)
        jac=np.zeros((294,64))
        jac[rows,I]=dx/d
        jac[rows,J]=-dx/d
        jac[rows,I+21]=dy/d
        jac[rows,J+21]=-dy/d
        jac[rows,I+42]=-1
        jac[rows,J+42]=-1
        ar=np.arange(21)
        for k,(col,sgn) in enumerate(((0,1),(0,-1),(21,1),(21,-1))):
            if time.monotonic() >= end:
                raise Deadline()
            ix=210+k*21+ar
            jac[ix,col+ar]=sgn
            jac[ix,42+ar]=-1
        jac[231:252,63]=1
        jac[273:294,63]=-1
        return jac
    for b in starts:
        if time.monotonic() >= end:
            break
        retain(b)
        w = np.max(b[:,0]+b[:,2])
        z = np.r_[b[:,0],b[:,1],b[:,2],w]
        last = [z.copy()]
        def callback(v):
            if time.monotonic() >= end:
                raise Deadline()
            last[0]=v.copy()
        try:
            res=minimize(objective,z,jac=True,method='SLSQP',
                constraints=[{'type':'ineq','fun':constraints,'jac':jacobian}],
                bounds=[(None,None)]*42+[(1e-7,None)]*21+[(0.01,1.99)],
                callback=callback,options={'maxiter':65,'ftol':2e-11,'disp':False})
            z=res.x
        except Deadline:
            z=last[0]
        candidate=np.column_stack((z[:21],z[21:42],z[42:63]))
        # Uniform radius repair conservatively removes remaining solver violations.
        if np.isfinite(candidate).all() and np.min(candidate[:,2])>0:
            d=np.sqrt((candidate[I,0]-candidate[J,0])**2+(candidate[I,1]-candidate[J,1])**2)
            ratio=np.min(d/(candidate[I,2]+candidate[J,2]))
            candidate[:,2]*=min(1.0,ratio)*(1-4e-14)
            retain(candidate)
    return best
