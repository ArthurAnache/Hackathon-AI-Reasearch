import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {'primary_share': (0.66, 0.4, 0.85), 'ftol': (2e-12, 1e-14, 1e-8, 'log')}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = None
    best_score = -float('inf')
    best_z = None

    def exact(obj):
        try:
            c = np.asarray(obj['circles'], dtype=float)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            x = list(map(Fraction, map(float, c[:, 0])))
            y = list(map(Fraction, map(float, c[:, 1])))
            r = list(map(Fraction, map(float, c[:, 2])))
            p = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
            p += max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
            if p > 2:
                return -float('inf')
            for i in range(21):
                if time.monotonic() >= deadline:
                    return -float('inf')
                for j in range(i):
                    d = np.sqrt((c[i,0]-c[j,0])**2+(c[i,1]-c[j,1])**2)
                    if d < c[i,2]+c[j,2]:
                        return -float('inf')
            return float(np.sum(c[:,2]))
        except Exception:
            return -float('inf')

    if init is not None:
        if 'circles' not in init:
            cc = []
            for x,y,r,m in init['orbits']:
                cc.append([x,y,r])
                if m == 2:
                    cc.append([x,-y,r])
            init = {'circles': cc}
        val = exact(init)
        if val > best_score:
            best, best_score = init, val

    # Twelve orbits: three axis circles, then nine reflected pairs.
    orbit = np.array(list(range(3)) + [i for i in range(3,12) for _ in range(2)])
    signs = np.array([0]*3 + [s for _ in range(9) for s in (1,-1)])
    X = np.zeros((21,34)); Y = X.copy(); R = X.copy()
    for k in range(21):
        X[k,orbit[k]] = 1
        R[k,21+orbit[k]] = 1
        if orbit[k] >= 3:
            Y[k,12+orbit[k]-3] = signs[k]
    aa=[]; bb=[]; rr=[]; seen=set()
    for i,j in itertools.combinations(range(21),2):
        if time.monotonic() >= deadline:
            break
        a=X[i]-X[j]; b=Y[i]-Y[j]; r=R[i]+R[j]
        def canon(v):
            nz=np.flatnonzero(v)
            return tuple(v if not len(nz) or v[nz[0]]>0 else -v)
        key=(canon(a),canon(b),tuple(r))
        if key not in seen:
            seen.add(key); aa.append(a); bb.append(b); rr.append(r)
    A=np.array(aa); B=np.array(bb); C=np.array(rr)
    walls=[]; offsets=[]
    for i in range(12):
        v=np.zeros(34); v[i]=1; v[21+i]=-1
        walls.append(v); offsets.append(0.)
        v=np.zeros(34); v[33]=1; v[i]=-1; v[21+i]=-1
        walls.append(v); offsets.append(0.)
        v=np.zeros(34); v[33]=-.5; v[21+i]=-1
        if i>=3: v[12+i-3]=-1
        walls.append(v); offsets.append(1.)
    L=np.array(walls); off=np.array(offsets)
    objective=np.zeros(34); objective[21:24]=-1; objective[24:33]=-2
    bounds=[(0,2)]*12+[(0,1)]*9+[(1e-7,.5)]*12+[(.2,1.8)]

    def expand(z):
        return np.column_stack((X@z,Y@z,R@z))

    def consider(z):
        nonlocal best, best_score, best_z
        if time.monotonic() >= deadline-.006:
            return
        c=expand(z)
        if not np.isfinite(c).all() or np.any(c[:,2]<=0):
            return
        # Repair overlaps with a common radius contraction, preserving centers.
        factor=1.
        for i in range(21):
            if time.monotonic() >= deadline-.004:
                return
            if i:
                d=np.sqrt(np.sum((c[:i,:2]-c[i,:2])**2,axis=1))
                factor=min(factor,float(np.min(d/(c[:i,2]+c[i,2]))))
        c[:,2] *= max(0., min(1.,factor))*(1-3e-14)
        if np.any(c[:,2]<=0): return
        lo=np.min(c[:,:2]-c[:,2,None],axis=0)
        hi=np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,:2]-=(lo+hi)/2
        scale=min(1.,2/float(np.sum(hi-lo)))*(1-3e-14)
        c*=scale
        obj={'circles':c.tolist()}
        val=exact(obj)
        if val>best_score:
            best_score=val; best=obj; best_z=z.copy()

    seeds=[]
    if init is not None:
        raw=np.asarray(init['circles'],float)
        for swap in (False,True):
            if time.monotonic() >= deadline-.03: break
            c=raw.copy()
            if swap: c[:,[0,1]]=c[:,[1,0]]
            lo=np.min(c[:,:2]-c[:,2,None],axis=0)
            hi=np.max(c[:,:2]+c[:,2,None],axis=0)
            c[:,:2]-=(lo+hi)/2
            axis=np.argsort(np.abs(c[:,1]))[:3]
            others=np.array([i for i in range(21) if i not in axis])
            order=others[np.argsort(c[others,1])]
            neg=order[:9]; pos=order[9:]
            costs=(c[pos,None,0]-c[neg,0])**2+(c[pos,None,1]+c[neg,1])**2
            costs+=.3*(c[pos,None,2]-c[neg,2])**2
            ip,jn=linear_sum_assignment(costs)
            z=np.zeros(34)
            z[:3]=c[axis,0]; z[21:24]=c[axis,2]
            error=float(np.sum(c[axis,1]**2))
            for k,(a,b) in enumerate(zip(pos[ip],neg[jn])):
                z[3+k]=(c[a,0]+c[b,0])/2
                z[12+k]=abs(c[a,1]-c[b,1])/2
                z[24+k]=(c[a,2]+c[b,2])/2
                error+=float(costs[ip[k],jn[k]])
            q=expand(z)
            left=np.min(q[:,0]-q[:,2]); right=np.max(q[:,0]+q[:,2])
            z[:12]-=left
            width=right-left
            height=2*np.max(np.abs(q[:,1])+q[:,2])
            scale=2/(width+height)
            z[:33]*=scale; z[33]=width*scale
            seeds.append((error,z))
    seeds.sort(key=lambda t:t[0])
    # Compact five-row fallback: 4,5,3,5,4 circles.
    z=np.zeros(34)
    z[:3]=[-2,0,2]
    z[3:8]=[-4,-2,0,2,4]; z[8:12]=[-3,-1,1,3]
    z[12:17]=np.sqrt(3); z[17:21]=2*np.sqrt(3)
    z[21:33]=1
    z[:12]+=5
    scale=2/(10+4*np.sqrt(3)+2)
    z[:33]*=scale; z[33]=10*scale
    seeds.append((100.,z))

    def solve(z, until, iterations=220):
        last=[z.copy()]
        cache=[None,None,None]
        def calc(v):
            if time.monotonic() >= until:
                raise StopSearch()
            if cache[0] is None or not np.array_equal(v,cache[0]):
                dx=A@v; dy=B@v; sr=C@v
                g=np.concatenate((dx*dx+dy*dy-sr*sr,L@v+off))
                jac=np.vstack((2*(dx[:,None]*A+dy[:,None]*B-sr[:,None]*C),L))
                cache[:]=[v.copy(),g,jac]
            return cache[1],cache[2]
        def callback(v):
            last[0]=v.copy()
            if time.monotonic()>=until: raise StopSearch()
        try:
            res=minimize(lambda v:float(objective@v),z,jac=lambda v:objective,
                         method='SLSQP',bounds=bounds,
                         constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                                      'jac':lambda v:calc(v)[1]},
                         callback=callback,options={'maxiter':iterations,'ftol':params['ftol'],'disp':False})
            last[0]=res.x
        except StopSearch:
            pass
        except (ValueError,FloatingPointError):
            pass
        consider(last[0])
        return last[0]

    if seeds and time.monotonic()<deadline-.03:
        primary_end=min(deadline-.04,start+params['primary_share']*.70*budget_s)
        first=solve(seeds[0][1],primary_end)
        # Polish the recovered topology before trying unrelated basins.
        if time.monotonic()<deadline-.06:
            first=solve(first,min(deadline-.025,time.monotonic()+.16),100)
        if time.monotonic()<deadline-.08:
            fallback=seeds[-1][1]
            solve(fallback,deadline-.025,180)
    if best is None:
        # Always-available strictly separated fallback.
        c=[]
        for i in range(21):
            if time.monotonic()>=deadline: break
            c.append([.18*(i%7),.18*(i//7),.02])
        if len(c)<21:
            c=[[.18*(i%7),.18*(i//7),.02] for i in range(21)]
        return {'circles':c}
    return best
