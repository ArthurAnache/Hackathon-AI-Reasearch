import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "cluster_max": (5, 2, 8),
    "trials": (4, 1, 12),
    "iterations": (85, 20, 200),
}
N = 21
I, J = np.triu_indices(N, 1)

class Deadline(Exception):
    pass

def exact(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    extent = max(u+t for u,t in zip(x,r))-min(u-t for u,t in zip(x,r))
    extent += max(u+t for u,t in zip(y,r))-min(u-t for u,t in zip(y,r))
    return float(np.sum(a[:, 2])) if extent <= 2 else -np.inf

def repair(z, deadline):
    p = np.column_stack((z[:N], z[N:2*N])).copy()
    r = np.maximum(z[2*N:3*N], 1e-10).copy()
    d = np.sqrt(np.sum((p[I]-p[J])**2, axis=1))
    factor = min(1.0, float(np.min(d/(r[I]+r[J]))))
    r *= factor*(1-3e-14)
    lo = np.min(p-r[:, None], axis=0)
    hi = np.max(p+r[:, None], axis=0)
    scale = (2-2e-14)/np.sum(hi-lo)
    p = (p-lo)*scale
    r *= scale
    a = np.column_stack((p,r))
    for _ in range(5):
        if time.monotonic() >= deadline:
            break
        if np.isfinite(exact(a)):
            return a
        a[:,2] *= 1-2e-13
    return a

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k%5+0.5)*0.2, (k//5+0.5)*0.2, 0.04] for k in range(N)])
        init = {"circles": a.tolist()}
    else:
        a = np.asarray(init['circles'], dtype=float).copy()
    best = init
    best_score = exact(a)
    p0, r0 = a[:,:2].copy(), a[:,2].copy()
    lo = np.min(p0-r0[:,None],axis=0)
    sides = np.max(p0+r0[:,None],axis=0)-lo
    scale = 2/np.sum(sides)
    p0 = (p0-lo)*scale
    r0 *= scale
    w0 = sides[0]*scale
    distances = np.sqrt(np.sum((p0[:,None,:]-p0[None,:,:])**2,axis=2))
    gap = distances-r0[:,None]-r0[None,:]
    np.fill_diagonal(gap, np.inf)
    edges = [(int(i),int(j)) for i,j in zip(I,J) if gap[i,j] < 0.012]
    neighbors = [list(np.argsort(gap[k])[:5]) for k in range(N)]
    pool = []
    # Each hinge is an existing near-contact. Reflect a connected patch on one side.
    for ai, aj in edges:
        if time.monotonic() >= start+0.18*budget_s:
            break
        origin = p0[ai]
        axis = p0[aj]-origin
        length = np.linalg.norm(axis)
        if length < 1e-12:
            continue
        axis /= length
        normal = np.array([-axis[1],axis[0]])
        signed = (p0-origin)@normal
        midpoint = (p0[ai]+p0[aj])*0.5
        for side in (-1,1):
            if time.monotonic() >= start+0.18*budget_s:
                break
            eligible = [k for k in range(N) if k not in (ai,aj) and signed[k]*side > 1e-5]
            eligible.sort(key=lambda k: np.linalg.norm(p0[k]-midpoint))
            for root in eligible[:3]:
                if time.monotonic() >= start+0.18*budget_s:
                    break
                cluster = [root]
                for size in range(1,int(params['cluster_max'])+1):
                    if time.monotonic() >= start+0.18*budget_s:
                        break
                    if size > 1:
                        frontier = {int(v) for u in cluster for v in neighbors[u] if v in eligible and v not in cluster}
                        if not frontier:
                            break
                        nxt = min(frontier,key=lambda v: min(gap[v,u] for u in cluster))
                        cluster.append(nxt)
                    ids = np.array(cluster)
                    p = p0.copy()
                    p[ids] -= 2*signed[ids,None]*normal
                    movement = np.sum(np.linalg.norm(p-p0,axis=1)*r0)
                    if movement < 0.003:
                        continue
                    d = np.sqrt(np.sum((p[I]-p[J])**2,axis=1))
                    ov = np.maximum(r0[I]+r0[J]-d,0)
                    box = np.maximum(r0-p[:,0],0)**2 + np.maximum(p[:,0]+r0-w0,0)**2
                    box += np.maximum(r0-p[:,1],0)**2 + np.maximum(p[:,1]+r0-(2-w0),0)**2
                    damage = np.sum(ov**2)+np.sum(box)
                    pool.append((float(damage),float(rng.random()),p))
    pool.sort(key=lambda t:(t[0],t[1]))
    selected = []
    for item in pool:
        if time.monotonic() >= deadline:
            break
        p = item[2]
        if all(np.linalg.norm(p-q) > 0.025 for q in selected):
            selected.append(p)
        if len(selected) >= int(params['trials']):
            break
    if not selected:
        selected = [p0]
    bounds = [(None,None)]*(2*N)+[(1e-7,0.7)]*N+[(0.1,1.9)]
    objective_gradient = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    def check():
        if time.monotonic() >= deadline:
            raise Deadline()
    def objective(z):
        check()
        return -np.sum(z[2*N:3*N]), objective_gradient
    def constraints(z):
        check()
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy=x[I]-x[J],y[I]-y[J]
        d=np.sqrt(dx*dx+dy*dy+1e-28)
        return np.r_[d-r[I]-r[J],x-r,w-x-r,y-r,2-w-y-r]
    def jacobian(z):
        check()
        x,y=z[:N],z[N:2*N]
        dx,dy=x[I]-x[J],y[I]-y[J]
        d=np.sqrt(dx*dx+dy*dy+1e-28)
        m=len(I)
        g=np.zeros((m+4*N,3*N+1))
        rows=np.arange(m)
        g[rows,I]=dx/d; g[rows,J]=-dx/d
        g[rows,N+I]=dy/d; g[rows,N+J]=-dy/d
        g[rows,2*N+I]=-1; g[rows,2*N+J]=-1
        k=np.arange(N)
        for block in range(4):
            rr=m+block*N+k
            g[rr,2*N+k]=-1
            if block==0: g[rr,k]=1
            elif block==1:
                g[rr,k]=-1; g[rr,-1]=1
            elif block==2: g[rr,N+k]=1
            else:
                g[rr,N+k]=-1; g[rr,-1]=-1
        return g
    for p in selected:
        if time.monotonic() >= deadline-0.03:
            break
        z=np.r_[p[:,0],p[:,1],r0,w0]
        last=[z]
        def callback(zz):
            check()
            last[0]=zz.copy()
        try:
            result=minimize(objective,z,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            z=result.x
        except Deadline:
            z=last[0]
        if time.monotonic() > start+0.77*budget_s:
            break
        candidate=repair(z,start+0.79*budget_s)
        score=exact(candidate)
        if score > best_score:
            best_score=score
            best={'circles':candidate.tolist()}
    return best
