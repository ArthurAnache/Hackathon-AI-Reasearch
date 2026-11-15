import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"time_fraction": (0.70, 0.5, 0.75), "iterations": (115, 40, 200), "tolerance": (2e-12, 1e-14, 1e-8, "log")}


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + float(params['time_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -float('inf')

    def exact_score(obj):
        try:
            if 'circles' in obj:
                c = np.asarray(obj['circles'], dtype=float)
            else:
                c = np.column_stack((obj['centers'], obj['radii']))
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            xx = [Fraction(float(v)) for v in c[:, 0]]
            yy = [Fraction(float(v)) for v in c[:, 1]]
            rr = [Fraction(float(v)) for v in c[:, 2]]
            per = max(x+r for x,r in zip(xx,rr))-min(x-r for x,r in zip(xx,rr))
            per += max(y+r for y,r in zip(yy,rr))-min(y-r for y,r in zip(yy,rr))
            if per > 2:
                return -float('inf')
            for i in range(21):
                if time.monotonic() > started + 0.95 * budget_s:
                    return -float('inf')
                for j in range(i):
                    dx = c[i,0]-c[j,0]
                    dy = c[i,1]-c[j,1]
                    if np.sqrt(dx*dx+dy*dy) < c[i,2]+c[j,2]:
                        return -float('inf')
            return float(np.sum(c[:,2]))
        except (ValueError, TypeError, KeyError, OverflowError):
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)

    # Coordinates: twelve x's, nine positive y's, twelve radii, width.
    nr = 12
    nv = 34
    weights = np.array([2.0]*9 + [1.0]*3)
    objective = np.zeros(nv)
    objective[21:33] = -weights
    A = []
    b = []
    for i in range(12):
        if time.monotonic() >= deadline:
            break
        row = np.zeros(nv); row[i] = 1; row[21+i] = -1
        A.append(row); b.append(0.)
        row = np.zeros(nv); row[33] = 1; row[i] = -1; row[21+i] = -1
        A.append(row); b.append(0.)
        row = np.zeros(nv); row[33] = -.5; row[21+i] = -1
        if i < 9:
            row[12+i] = -1
        A.append(row); b.append(1.)
        if i < 9:
            row = np.zeros(nv); row[12+i] = 1; row[21+i] = -1
            A.append(row); b.append(0.)
    if len(A) != 45:
        return best
    A = np.asarray(A); b = np.asarray(b)
    DX=[]; DY=[]; RS=[]
    for i in range(12):
        if time.monotonic() >= deadline:
            return best
        for j in range(i):
            x = np.zeros(nv); x[i]=1; x[j]=-1
            y = np.zeros(nv)
            if i < 9: y[12+i]=1
            if j < 9: y[12+j]=-1
            r = np.zeros(nv); r[21+i]=1; r[21+j]=1
            DX.append(x); DY.append(y); RS.append(r)
            if i < 9 and j < 9:
                y = np.zeros(nv); y[12+i]=1; y[12+j]=1
                DX.append(x); DY.append(y); RS.append(r)
    DX=np.asarray(DX); DY=np.asarray(DY); RS=np.asarray(RS)
    cache_z = None
    cache_v = None
    cache_j = None

    class Stop(Exception):
        pass

    def constraints(z):
        nonlocal cache_z, cache_v, cache_j
        if time.monotonic() >= deadline:
            raise Stop()
        if cache_z is None or not np.array_equal(z, cache_z):
            dx=DX@z; dy=DY@z
            d=np.sqrt(dx*dx+dy*dy)
            inv=1/np.maximum(d,1e-15)
            cache_v=np.concatenate((A@z+b,d-RS@z))
            cache_j=np.vstack((A, (dx*inv)[:,None]*DX+(dy*inv)[:,None]*DY-RS))
            cache_z=z.copy()
        return cache_v

    def jac(z):
        constraints(z)
        return cache_j

    def unpack(z):
        c=np.empty((21,3))
        c[:9,0]=z[:9]; c[:9,1]=z[12:21]; c[:9,2]=z[21:30]
        c[9:18]=c[:9]; c[9:18,1]*=-1
        c[18:,0]=z[9:12]; c[18:,1]=0.; c[18:,2]=z[30:33]
        return c

    def consider(z):
        nonlocal best,best_score
        if time.monotonic() >= deadline + .07*budget_s:
            return
        c=unpack(z)
        if not np.isfinite(c).all() or np.any(c[:,2] <= 0):
            return
        ii,jj=np.triu_indices(21,1)
        delta=c[ii,:2]-c[jj,:2]
        distances=np.sqrt(np.sum(delta*delta,axis=1))
        ratio=float(np.min(distances/(c[ii,2]+c[jj,2])))
        c[:,2]*=min(1.,ratio)*(1.-3e-13)
        if np.any(c[:,2]<=0): return
        # Center before normalization, then use exact binary-float extents.
        lo=np.min(c[:,:2]-c[:,2,None],axis=0)
        hi=np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,:2]-=(hi+lo)/2
        xx=[Fraction(float(v)) for v in c[:,0]]
        yy=[Fraction(float(v)) for v in c[:,1]]
        rr=[Fraction(float(v)) for v in c[:,2]]
        perimeter=max(x+r for x,r in zip(xx,rr))-min(x-r for x,r in zip(xx,rr))
        perimeter+=max(y+r for y,r in zip(yy,rr))-min(y-r for y,r in zip(yy,rr))
        c*=float(Fraction(2)/perimeter)*(1.-2e-14)
        obj={'centers':c[:,:2].tolist(),'radii':c[:,2].tolist()}
        s=exact_score(obj)
        if s>best_score:
            best_score=s; best=obj

    def layered(counts, width=1.):
        z=np.zeros(nv); z[33]=width
        k=0
        step=(1.-width/2)/(len(counts)+.6)
        for row,n in enumerate(counts):
            if time.monotonic() >= deadline: raise Stop()
            margin=width/(2*(n+.25))
            z[k:k+n]=np.linspace(margin,width-margin,n)
            z[12+k:12+k+n]=(row+1)*step
            z[21+k:21+k+n]=min(width/(2*n),step*.52)*.9
            k+=n
        z[9:12]=np.linspace(width*.17,width*.83,3)
        z[30:33]=min(width/6,step*.6)
        return z

    def interstitial():
        z=np.zeros(nv); z[33]=1.
        z[:4]=[.125,.375,.625,.875]
        z[4:8]=z[:4]
        z[8]=.5
        z[12:16]=.125; z[16:20]=.375; z[20]=.25
        z[21:29]=.124; z[29]=.050
        z[9:12]=[.25,.5,.75]; z[30:33]=.050
        return z

    bounds=[(0.,2.)]*12+[(0.,1.)]*9+[(1e-7,1.)]*12+[(.25,1.75)]
    solutions=[]
    latest=None
    def callback(z):
        nonlocal latest
        latest=z.copy()
        if time.monotonic() >= deadline: raise Stop()

    try:
        starts=[layered([4,5]),layered([5,4]),interstitial(),layered([3,3,3],.78),layered([2,4,3],.9)]
    except Stop:
        return best
    for z in starts:
        if time.monotonic() >= deadline-.04: break
        latest=z.copy()
        try:
            res=minimize(lambda v: float(objective@v),z,jac=lambda v:objective,
                constraints={'type':'ineq','fun':constraints,'jac':jac},bounds=bounds,
                method='SLSQP',callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':float(params['tolerance']),'disp':False})
            latest=res.x.copy()
        except Stop:
            pass
        consider(latest)
        solutions.append(latest.copy())
    # Small coordinated perturbations explore alternative contacts without breaking reflection.
    attempt=0
    while time.monotonic()<deadline-.06 and solutions:
        if time.monotonic() >= deadline: break
        base=max(solutions,key=lambda v:float(-objective@v))
        z=base.copy()
        magnitude=.012 if attempt%2==0 else .035
        z[:12]+=rng.normal(0,magnitude,12)
        z[12:21]+=rng.normal(0,magnitude,9)
        z[:12]=np.clip(z[:12],0,2)
        z[12:21]=np.clip(z[12:21],0,1)
        latest=z.copy()
        try:
            res=minimize(lambda v:float(objective@v),z,jac=lambda v:objective,
                constraints={'type':'ineq','fun':constraints,'jac':jac},bounds=bounds,
                method='SLSQP',callback=callback,
                options={'maxiter':85,'ftol':float(params['tolerance']),'disp':False})
            latest=res.x.copy()
        except Stop:
            pass
        consider(latest)
        solutions.append(latest.copy())
        attempt+=1
    return best
