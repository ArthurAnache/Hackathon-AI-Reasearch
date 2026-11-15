import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'search_fraction': (0.70, 0.5, 0.74),
    'iterations': (110, 40, 200),
    'polish_threshold': (2e-7, 1e-9, 1e-4, 'log'),
    'jitter': (0.009, 0.0, 0.05),
}
II, JJ = np.triu_indices(21, 1)

class StopSearch(Exception):
    pass

def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = np.sqrt((a[II, 0]-a[JJ, 0])**2 + (a[II, 1]-a[JJ, 1])**2)
        if np.any(d < a[II, 2]+a[JJ, 2]):
            return -float('inf')
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        wh = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        wh += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if wh > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))
    except (TypeError, KeyError, ValueError, OverflowError):
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + float(params['search_fraction'])*budget_s
    rng = np.random.default_rng(seed)
    best = init
    score = exact_score(init) if init is not None else -float('inf')

    def check(deadline=end):
        if time.monotonic() >= deadline:
            raise StopSearch()

    def offer(a):
        nonlocal best, score
        check()
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        a[:,:2] -= np.mean(a[:,:2], axis=0)
        d = np.sqrt((a[II,0]-a[JJ,0])**2+(a[II,1]-a[JJ,1])**2)
        ratio = min(1., float(np.min(d/(a[II,2]+a[JJ,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1.-1.5e-14)
        wh = np.ptp(np.concatenate((a[:,:2]-a[:,2,None], a[:,:2]+a[:,2,None])), axis=0)
        a *= (2./float(wh.sum()))*(1.-1.5e-14)
        if float(a[:,2].sum()) > score:
            candidate = {'circles': a.tolist()}
            s = exact_score(candidate)
            if s > score:
                best, score = candidate, s

    def make_model(singles):
        # Each entity is either an axis circle or a reflected pair.
        dim = 1 + sum(2 if s else 3 for s in singles)
        X = np.zeros((21,dim)); Y = X.copy(); R = X.copy()
        mirrors = []; reps = []; col = 1; row = 0
        for single in singles:
            check()
            reps.append(row)
            if single:
                Y[row,col] = 1.; R[row,col+1] = 1.
                mirrors.append(row); row += 1; col += 2
            else:
                X[row,col] = 1.; X[row+1,col] = -1.
                Y[row:row+2,col+1] = 1.; R[row:row+2,col+2] = 1.
                mirrors.extend([row+1,row]); row += 2; col += 3
        pairs = []
        for i,j in zip(II,JJ):
            check()
            if (int(i),int(j)) <= tuple(sorted((mirrors[i],mirrors[j]))):
                pairs.append((i,j))
        p,q = np.array(pairs).T
        DX = X[p]-X[q]; DY = Y[p]-Y[q]; SR = R[p]+R[q]
        W = np.zeros(dim); W[0] = 1.
        A=[]; b=[]
        for i in reps:
            check()
            A.extend([.5*W-X[i]-R[i], Y[i]-R[i], -W-Y[i]-R[i]])
            b.extend([0.,0.,2.])
        return X,Y,R,DX,DY,SR,np.asarray(A),np.asarray(b),-R.sum(axis=0)

    def decode(m,z):
        return np.column_stack((m[0]@z,m[1]@z,m[2]@z))

    def values(m,z):
        X,Y,R,DX,DY,SR,A,b,obj = m
        dx=DX@z; dy=DY@z
        d=np.maximum(np.sqrt(dx*dx+dy*dy),1e-25)
        c=np.r_[d-SR@z,A@z+b]
        J=np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-SR,A))
        return c,J

    def polish(m,z,deadline):
        z=z.copy()
        for iteration in range(7):
            check(deadline)
            c,J=values(m,z)
            active=c<float(params['polish_threshold'])
            if np.count_nonzero(active)<len(z):
                break
            step,_,rank,_=np.linalg.lstsq(J[active],-c[active],rcond=1e-11)
            if rank<len(z) or np.max(np.abs(step))>.02:
                break
            z+=step
            offer(decode(m,z))
            if np.max(np.abs(step))<2e-15:
                break
        return z

    def optimize(m,z,deadline):
        obj=m[-1]; last=z.copy(); cached=[None,None,None]
        def cj(v):
            nonlocal last
            check(deadline)
            if cached[0] is None or not np.array_equal(cached[0],v):
                c,J=values(m,v)
                cached[:]=[v.copy(),c,J]
                last=v.copy()
            return cached[1],cached[2]
        def objective(v):
            check(deadline)
            return float(obj@v),obj
        bounds=[(None,None)]*len(z)
        bounds[0]=(.25,1.75)
        ridx=np.flatnonzero(np.any(m[2]!=0,axis=0))
        for k in ridx:
            check(deadline)
            bounds[int(k)]=(1e-7,.6)
        try:
            res=minimize(objective,z,method='SLSQP',jac=True,bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:cj(v)[0],'jac':lambda v:cj(v)[1]},
                options={'maxiter':int(params['iterations']),'ftol':3e-14,'disp':False})
            last=res.x
        except StopSearch:
            pass
        if time.monotonic()<end:
            offer(decode(m,last))
        if time.monotonic()<deadline-.003:
            try:
                last=polish(m,last,deadline)
            except StopSearch:
                pass
        return last

    def infer(a):
        # Test both possible orientations of the reflection axis.
        for swap in (False,True):
            check()
            b=a.copy()
            if swap:
                b[:,[0,1]]=b[:,[1,0]]
            lo=(b[:,:2]-b[:,2,None]).min(0)
            hi=(b[:,:2]+b[:,2,None]).max(0)
            b[:,0]-=.5*(lo[0]+hi[0]); b[:,1]-=lo[1]
            axis=np.flatnonzero(np.abs(b[:,0])<1e-5)
            pos=np.flatnonzero(b[:,0]>1e-5)
            neg=np.flatnonzero(b[:,0]<-1e-5)
            if len(axis)!=3 or len(pos)!=9 or len(neg)!=9:
                continue
            err=0.
            for i in pos:
                check()
                target=b[i].copy(); target[0]*=-1
                err=max(err,float(np.min(np.linalg.norm(b[neg]-target,axis=1))))
            if err>2e-5:
                continue
            singles=[True]*3+[False]*9
            z=[hi[0]-lo[0]]
            for i in axis:
                check()
                z.extend([b[i,1],b[i,2]])
            for i in pos:
                check()
                z.extend(b[i].tolist())
            return make_model(singles),np.asarray(z)
        return None

    try:
        if init is not None:
            inferred=infer(np.asarray(init['circles'],dtype=float))
            if inferred is not None:
                m,z=inferred
                warm_end=min(end,start+.24*budget_s)
                z=polish(m,z,warm_end)
                if time.monotonic()<warm_end-.01:
                    optimize(m,z,warm_end)
        if best is None:
            offer(np.array([[.2*x,.2*y,.1] for y in range(3) for x in range(7)]))
        families=[(3,4,5,4,5),(5,4,3,4,5),(4,5,3,5,4),(4,4,5,4,4)]
        records=[]
        search_end=start+.60*budget_s
        for index,rows in enumerate(families):
            check(search_end)
            singles=[]; z=[1.05]
            for row,count in enumerate(rows):
                check(search_end)
                if count%2:
                    singles.append(True); z.extend([.105+row*np.sqrt(3)*.105,.105])
                for k in range(count//2):
                    check(search_end)
                    singles.append(False)
                    offset=2*k+(2 if count%2 else 1)
                    z.extend([offset*.105,.105+row*np.sqrt(3)*.105,.105])
            m=make_model(singles)
            remaining=search_end-time.monotonic()
            deadline=min(search_end,time.monotonic()+remaining/max(1,min(3,len(families)-index)))
            z=optimize(m,np.asarray(z),deadline)
            records.append((float(-m[-1]@z),m,z))
        records.sort(key=lambda t:t[0],reverse=True)
        if best is not None and time.monotonic()<end-.01:
            inferred=infer(np.asarray(best['circles'],dtype=float))
            if inferred is not None:
                m,z=inferred
                optimize(m,z,end-.004)
        while records and time.monotonic()<end-.015:
            check()
            _,m,z=records[0]
            trial=z+rng.normal(0,float(params['jitter']),len(z))
            out=optimize(m,trial,end-.003)
            value=float(-m[-1]@out)
            if value>records[0][0]:
                records[0]=(value,m,out)
    except StopSearch:
        pass
    return best
