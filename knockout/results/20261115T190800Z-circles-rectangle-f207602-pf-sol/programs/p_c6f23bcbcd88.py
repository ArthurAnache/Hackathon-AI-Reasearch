import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'polish': (90, 30, 160), 'probe': (35, 15, 70), 'jitter': (0.008, 0.001, 0.04)}

class StopSearch(Exception):
    pass

def canonical(obj):
    if 'circles' in obj:
        return obj
    rows = []
    for x,y,r,m in obj['orbits']:
        rows.append([x,y,r])
        if m == 2:
            rows.append([x,-y,r])
    return {'circles': rows}

def score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -float('inf')
        xs,ys,rs = [[Fraction(float(t)) for t in a[:,k]] for k in range(3)]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = score(init) if init is not None else -float('inf')
    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()
    if best_score == -float('inf'):
        a = np.array([[2*(k%7),2*(k//7),1.] for k in range(21)]) / 10.000000000001
        best = {'circles':a.tolist()}
        best_score = score(best)
    try:
        a = np.asarray(canonical(init)['circles'],float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all():
            raise ValueError()
    except Exception:
        a = np.asarray(best['circles'],float)
    lo = np.min(a[:,:2]-a[:,2,None],axis=0)
    hi = np.max(a[:,:2]+a[:,2,None],axis=0)
    a[:,:2] -= (lo+hi)/2
    groups = np.array([0,1,2]+[g for g in range(3,12) for _ in range(2)])
    signs = np.array([0.,0.,0.]+[s for _ in range(9) for s in (1.,-1.)])
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    for k in range(21):
        tick()
        g = groups[k]
        X[k,g] = 1.
        R[k,21+g] = 1.
        if g >= 3:
            Y[k,9+g] = signs[k]
    all_i,all_j = np.triu_indices(21,1)
    pairs=[]; seen=set()
    for i,j in zip(all_i,all_j):
        tick()
        g,h = sorted((int(groups[i]),int(groups[j])))
        key = (g,h,int(signs[i]*signs[j]))
        if key not in seen:
            seen.add(key); pairs.append((i,j))
    pi,pj = np.array(pairs).T
    DX=X[pi]-X[pj]; DY=Y[pi]-Y[pj]; RS=R[pi]+R[pj]
    walls=np.zeros((36,34)); offsets=np.zeros(36)
    for g in range(12):
        tick()
        walls[g,g]=1.; walls[g,21+g]=-1.; walls[g,33]=.5
        walls[12+g,g]=-1.; walls[12+g,21+g]=-1.; walls[12+g,33]=.5
        walls[24+g,21+g]=-1.; walls[24+g,33]=-.5
        offsets[24+g]=1.
        if g>=3:
            walls[24+g,9+g]=-1.
    weights=np.array([1.]*3+[2.]*9)
    grad=np.zeros(34); grad[21:33]=-weights
    bounds=[(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.6)]*12+[(.15,1.85)]
    anchor=None; anchor_score=-float('inf')
    def retain(v):
        nonlocal best,best_score,anchor,anchor_score
        if not np.isfinite(v).all() or np.min(v[21:33])<=0:
            return
        xx=v[:12].copy(); yy=np.r_[np.zeros(3),v[12:21]]
        rr=v[21:33].copy()
        cx=xx[groups]; cy=yy[groups]*signs; cr=rr[groups]
        d=np.sqrt((cx[all_i]-cx[all_j])**2+(cy[all_i]-cy[all_j])**2)
        ratio=float(np.min(d/(cr[all_i]+cr[all_j])))
        rr*=min(1.,ratio)*(1.-3e-14)
        if np.min(rr)<=0:
            return
        width=np.max(xx+rr)-np.min(xx-rr)
        height=2*np.max(yy+rr)
        scale=2./(width+height)*(1.-3e-14)
        estimate=float(weights@rr)*scale
        if estimate>anchor_score:
            anchor_score=estimate
            anchor=v.copy()
            anchor[:12]=xx*scale
            anchor[12:21]=yy[3:]*scale
            anchor[21:33]=rr*scale
            anchor[33]=width*scale
        if estimate<=best_score:
            return
        obj={'orbits':[[float(xx[g]*scale),float(yy[g]*scale),float(rr[g]*scale),1 if g<3 else 2] for g in range(12)]}
        val=score(obj)
        if val>best_score:
            best,best_score=obj,val
    def fun(v):
        tick(); return float(grad@v)
    def jac(v):
        tick(); return grad
    def cons(v):
        tick()
        dx=DX@v; dy=DY@v
        return np.r_[np.sqrt(dx*dx+dy*dy)-RS@v,walls@v+offsets]
    def cjac(v):
        tick()
        dx=DX@v; dy=DY@v
        d=np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,walls))
    local_deadline=deadline
    def callback(v):
        tick()
        retain(v)
        if time.monotonic()>=local_deadline:
            raise StopSearch()
    def solve(v, n, seconds):
        nonlocal local_deadline
        tick()
        local_deadline=min(deadline,time.monotonic()+seconds)
        retain(v)
        try:
            res=minimize(fun,v,jac=jac,method='SLSQP',bounds=bounds,constraints={'type':'ineq','fun':cons,'jac':cjac},callback=callback,options={'maxiter':int(n),'ftol':2e-13,'disp':False})
            retain(res.x)
        except StopSearch:
            pass
    def vector(g):
        z=np.zeros(34)
        z[:12]=g[:,0]; z[12:21]=np.abs(g[3:,1]); z[21:33]=np.maximum(g[:,2],1e-5)
        z[33]=np.clip(np.ptp(g[:,0])+2*np.max(g[:,2]),.2,1.8)
        return z
    projections=[]
    try:
        for swap in (False,True):
            tick()
            b=a[:,[1,0,2]].copy() if swap else a.copy()
            order=np.argsort(b[:,1]); lower=order[:9]; singles=order[9:12]; upper=order[12:]
            reflected=b[lower].copy(); reflected[:,1]*=-1
            cost=np.sum((b[upper,None,:]-reflected[None,:,:])**2,axis=2)
            u,v=linear_sum_assignment(cost)
            g=[[b[k,0],0.,b[k,2]] for k in singles]
            for p,q in zip(u,v):
                tick(); g.append(((b[upper[p]]+reflected[q])/2).tolist())
            error=float(np.sum(cost[u,v])+np.sum(b[singles,1]**2))
            projections.append((error,vector(np.asarray(g))))
        projections.sort(key=lambda t:t[0])
        solve(projections[0][1],params['polish'],budget_s*.31)
        if best_score<2.34 or projections[1][0]<projections[0][0]*1.4:
            solve(projections[1][1],params['probe'],budget_s*.13)
        # A compact five-column triangular family, with three axis circles.
        if best_score<2.35:
            g=[[-np.sqrt(3.),0.,1.],[0.,0.,1.],[np.sqrt(3.),0.,1.]]
            # Columns with populations 4,5,3,5,4.
            g=[]
            for x in (-np.sqrt(3.),0.,np.sqrt(3.)):
                tick(); g.append([x,0.,1.])
            for x,ys in ((-2*np.sqrt(3.),(1.,3.)),(-np.sqrt(3.),(2.,4.)),(0.,(2.,)),(np.sqrt(3.),(2.,4.)),(2*np.sqrt(3.),(1.,3.))):
                tick()
                for y in ys:
                    tick(); g.append([x,y,1.])
            g=np.array(g); g*=2./(4*np.sqrt(3.)+12.)
            solve(vector(g),params['polish'],budget_s*.16)
        attempt=0
        while time.monotonic()<deadline:
            tick()
            if anchor is None:
                break
            z=anchor.copy()
            if attempt>0:
                amplitude=float(params['jitter'])*(1.+.7*(attempt%3))
                z[:21]+=rng.normal(0.,amplitude,21)
                z[12:21]=np.maximum(z[12:21],0.)
                z[21:33]*=rng.uniform(.97,1.01,12)
            solve(z,params['polish'] if attempt==0 else params['probe'],budget_s*.16)
            attempt+=1
    except (StopSearch,ValueError,FloatingPointError):
        pass
    return best
