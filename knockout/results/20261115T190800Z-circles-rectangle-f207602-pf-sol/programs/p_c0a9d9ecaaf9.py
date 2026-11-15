import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'pressure': (2.60, 2.40, 2.90),
    'stages': (4, 2, 7),
    'iterations': (100, 30, 250),
    'jitter': (0.003, 0.0, 0.04)
}
N = 21
I, J = np.triu_indices(N, 1)

class StopSearch(Exception):
    pass

def valid(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    p, r = a[:, :2], a[:, 2]
    d = np.sqrt((p[I, 0]-p[J, 0])**2 + (p[I, 1]-p[J, 1])**2)
    if np.any(d < r[I]+r[J]):
        return False
    x, y, q = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    hp = max(u+t for u,t in zip(x,q))-min(u-t for u,t in zip(x,q))
    hp += max(u+t for u,t in zip(y,q))-min(u-t for u,t in zip(y,q))
    return hp <= 2

def repair(a):
    a = a.copy()
    p, r = a[:, :2], a[:, 2]
    if not np.isfinite(a).all() or np.any(r <= 0):
        return None
    d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    r *= factor * (1-2e-14)
    lo = np.min(p-r[:,None], axis=0)
    hi = np.max(p+r[:,None], axis=0)
    scale = min(1.0, 2/float(np.sum(hi-lo))) * (1-2e-14)
    a[:, :2] = (p-lo)*scale
    a[:, 2] *= scale
    return a if valid(a) else None

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    pressure_end = start + 0.44*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(i%5+0.5)/5, (i//5+0.5)/5, 0.07] for i in range(N)])
        best = {'circles': a.tolist()}
        bestscore = float(a[:,2].sum())
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
        best = init
        bestscore = float(a[:,2].sum()) if valid(a) else -np.inf
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    sides = np.max(a[:,:2]+a[:,2,None], axis=0)-lo
    scale = 2/sides.sum()
    p = (a[:,:2]-lo)*scale
    r = a[:,2]*scale
    w = sides[0]*scale
    p += rng.normal(0, params['jitter'], p.shape)
    u = np.r_[p[:,0], p[:,1], np.log(np.maximum(r,1e-8)), w]
    last = [u.copy()]

    def decode(z, total):
        t = np.exp(z[42:63]-np.max(z[42:63]))
        rr = total*t/t.sum()
        return z[:21], z[21:42], rr, z[63]

    def stress(z, total):
        if time.monotonic() >= pressure_end:
            raise StopSearch()
        last[0] = z.copy()
        x,y,rr,ww = decode(z,total)
        dx,dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-24)
        s = rr[I]+rr[J]
        v = np.maximum(1-d/s,0)
        gx,gy,gr = np.zeros(N),np.zeros(N),np.zeros(N)
        c = 2*v/(s*d)
        np.add.at(gx,I,-c*dx); np.add.at(gx,J,c*dx)
        np.add.at(gy,I,-c*dy); np.add.at(gy,J,c*dy)
        cr = 2*v*d/(s*s)
        np.add.at(gr,I,cr); np.add.at(gr,J,cr)
        distances = np.array([x,ww-x,y,2-ww-y])
        vb = np.maximum(1-distances/rr,0)
        bc = 2*vb/rr
        gx += -bc[0]+bc[1]
        gy += -bc[2]+bc[3]
        gr += np.sum(2*vb*distances/(rr*rr),axis=0)
        gw = -bc[1].sum()+bc[3].sum()
        gu = rr*(gr-np.dot(gr,rr)/total)
        return float(v@v+np.sum(vb*vb)), np.r_[gx,gy,gu,gw]

    total = float(params['pressure'])
    stages = int(params['stages'])
    for k in range(stages):
        if time.monotonic() >= pressure_end:
            break
        total = float(params['pressure']) + (2.369-float(params['pressure']))*k/max(1,stages-1)
        last[0] = u.copy()
        try:
            res = minimize(stress,u,args=(total,),jac=True,method='L-BFGS-B',
                           bounds=[(-1,3)]*42+[(-12,2)]*21+[(0.35,1.65)],
                           options={'maxiter':int(params['iterations']), 'ftol':1e-12,'gtol':1e-7,'maxls':15})
            u = res.x
        except StopSearch:
            u = last[0]
            break
    x,y,r,w = decode(u,total)
    z = np.r_[x,y,r,w]
    latest = [z.copy()]
    ident = np.eye(N)
    def constraints(z, jac=False):
        x,y,r,w = z[:21],z[21:42],z[42:63],z[63]
        dx,dy=x[I]-x[J],y[I]-y[J]
        d=np.sqrt(dx*dx+dy*dy+1e-30)
        if not jac:
            return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
        mat=np.zeros((len(I)+84,64))
        ix=np.arange(len(I))
        mat[ix,I]=dx/d; mat[ix,J]=-dx/d
        mat[ix,21+I]=dy/d; mat[ix,21+J]=-dy/d
        mat[ix,42+I]=-1; mat[ix,42+J]=-1
        k=len(I)
        mat[k:k+21,:21]=ident; mat[k:k+21,42:63]=-ident
        mat[k+21:k+42,:21]=-ident; mat[k+21:k+42,42:63]=-ident; mat[k+21:k+42,63]=1
        mat[k+42:k+63,21:42]=ident; mat[k+42:k+63,42:63]=-ident
        mat[k+63:k+84,21:42]=-ident; mat[k+63:k+84,42:63]=-ident; mat[k+63:k+84,63]=-1
        return mat
    grad=np.r_[np.zeros(42),-np.ones(21),0.]
    def objective(z):
        if time.monotonic() >= end:
            raise StopSearch()
        latest[0]=z.copy()
        return -float(z[42:63].sum()),grad
    if time.monotonic() < end:
        try:
            out=minimize(objective,z,jac=True,method='SLSQP',
                         constraints={'type':'ineq','fun':constraints,'jac':lambda q:constraints(q,True)},
                         bounds=[(None,None)]*42+[(1e-6,0.6)]*21+[(0.25,1.75)],
                         options={'maxiter':120,'ftol':2e-12})
            z=out.x
        except StopSearch:
            z=latest[0]
    candidate=repair(np.column_stack([z[:21],z[21:42],z[42:63]]))
    if candidate is not None and float(candidate[:,2].sum()) > bestscore:
        best={'circles':candidate.tolist()}
    return best
