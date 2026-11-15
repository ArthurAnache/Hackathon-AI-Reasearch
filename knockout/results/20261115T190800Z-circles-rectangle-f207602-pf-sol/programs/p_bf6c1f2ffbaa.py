import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'first_fraction': (0.38, 0.2, 0.6),
    'second_fraction': (0.28, 0.1, 0.4),
    'iterations': (130, 40, 250),
    'safety': (2e-14, 3e-15, 1e-10, 'log')
}
I, J = np.triu_indices(21, 1)

class Stop(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .70*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_a = None
    best_score = -np.inf
    eps = params['safety']
    if init is not None:
        try:
            a = np.asarray(init['circles'], float) if 'circles' in init else np.asarray(init['state'], float).reshape(3,21).T
            best_score = exact_score(a)
            if np.isfinite(best_score):
                best_a = a.copy()
        except (ValueError, KeyError):
            pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def offer(a):
        nonlocal best, best_a, best_score
        a = np.asarray(a, float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        q = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if q <= 0:
            return
        a[:,2] *= q*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        s = (2-16*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*s
        a[:,2] *= s
        if float(np.sum(a[:,2])) > best_score:
            score = exact_score(a)
            if score > best_score:
                best_score, best_a = score, a.copy()
                best = {'circles': a.tolist()}

    # Each independent site is either an axis circle or an upper/lower pair.
    ids = np.r_[np.arange(12), np.arange(3,12)]
    xi = ids
    yi = np.r_[np.zeros(3,dtype=int),np.arange(12,21),np.arange(12,21)]
    sg = np.r_[np.zeros(3),np.ones(9),-np.ones(9)]
    ri = 21+ids
    pairs = [(i,j) for i in range(12) for j in range(i+1,12)]
    pairs += [(i,12+j-3) for i in range(3,12) for j in range(i,12)]
    pi,pj = np.array(pairs).T

    def solve(a, symmetric, seconds):
        check()
        end = min(deadline, time.monotonic()+seconds)
        if symmetric:
            n = 34
            xx, yy, rr, signs = xi, yi, ri, sg
            ii,jj = pi,pj
            v = np.zeros(n)
            v[:12] = a[:12,0]
            v[12:21] = a[3:12,1]
            v[21:33] = a[:12,2]
            walls = np.arange(12)
        else:
            n = 64
            xx, yy, rr = np.arange(21),np.arange(21)+21,np.arange(21)+42
            signs = np.ones(21)
            ii,jj = I,J
            v = np.r_[a[:,0],a[:,1],a[:,2],1.]
            walls = np.arange(21)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        v[-1] = np.clip(hi[0]-lo[0],.15,1.85)
        nw, m = len(walls),len(ii)
        B = np.zeros((4*nw,n))
        k = np.arange(nw)
        B[k,xx[walls]]=1
        B[nw+k,xx[walls]]=-1
        B[:2*nw,-1]=.5
        B[k,rr[walls]]=-1
        B[nw+k,rr[walls]]=-1
        B[2*nw+k,yy[walls]]=signs[walls]
        B[3*nw+k,yy[walls]]=-signs[walls]
        B[2*nw+k,rr[walls]]-=1
        B[3*nw+k,rr[walls]]-=1
        B[2*nw:,-1]=-.5
        offset=np.r_[np.zeros(2*nw),np.ones(2*nw)]
        obj=np.zeros(n)
        if symmetric:
            obj[21:24]=-1
            obj[24:33]=-2
        else:
            obj[42:63]=-1
        bounds=[(None,None)]*n
        for j in np.unique(rr):
            bounds[j]=(1e-6,.7)
        if symmetric:
            for j in range(12,21):
                bounds[j]=(1e-6,None)
        bounds[-1]=(.15,1.85)
        rows=np.arange(m)
        cache=[None,None,None]
        def unpack(t):
            return np.column_stack((t[xx],signs*t[yy],t[rr]))
        def calc(t):
            if time.monotonic() >= end:
                raise Stop()
            if cache[0] is not None and np.array_equal(t,cache[0]):
                return cache[1],cache[2]
            x,y,r=t[xx],signs*t[yy],t[rr]
            dx,dy=x[ii]-x[jj],y[ii]-y[jj]
            d=np.sqrt(dx*dx+dy*dy+1e-30)
            c=np.r_[d-r[ii]-r[jj],B@t+offset]
            jac=np.zeros((m+4*nw,n))
            np.add.at(jac,(rows,xx[ii]),dx/d)
            np.add.at(jac,(rows,xx[jj]),-dx/d)
            np.add.at(jac,(rows,yy[ii]),dy/d*signs[ii])
            np.add.at(jac,(rows,yy[jj]),-dy/d*signs[jj])
            np.add.at(jac,(rows,rr[ii]),-1.)
            np.add.at(jac,(rows,rr[jj]),-1.)
            jac[m:]=B
            cache[:]=[t.copy(),c,jac]
            return c,jac
        last=[v.copy()]
        count=[0]
        def cb(t):
            last[0]=t.copy()
            count[0]+=1
            if count[0]%8==0:
                offer(unpack(t))
            if time.monotonic() >= end:
                raise Stop()
        try:
            res=minimize(lambda t:(float(obj@t),obj),v,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda t:calc(t)[0],'jac':lambda t:calc(t)[1]},
                callback=cb,options={'maxiter':params['iterations'],'ftol':1e-12,'disp':False})
            last[0]=res.x
        except Stop:
            pass
        out=unpack(last[0])
        offer(out)
        return out

    def lattice(kind):
        singles=[]
        pairs=[]
        if kind==0:
            for col in range(6):
                if time.monotonic() >= deadline:
                    raise Stop()
                x=(col-2.5)*np.sqrt(3.)
                if col%2==0:
                    singles.append([x,0.,1.])
                    pairs.append([x,2.,1.])
                else:
                    pairs.extend([[x,1.,1.],[x,3.,1.]])
        else:
            singles=[[-2.,0.,1.],[0.,0.,1.],[2.,0.,1.]]
            pairs=[[x,np.sqrt(3.),1.] for x in (-3.,-1.,1.,3.)]
            pairs += [[x,2*np.sqrt(3.),1.] for x in (-4.,-2.,0.,2.,4.)]
        a=np.array(singles+pairs+[[x,-y,r] for x,y,r in pairs],float)
        lo=np.min(a[:,:2]-a[:,2,None],axis=0)
        hi=np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2]-=(hi+lo)/2
        a*=2/np.sum(hi-lo)
        return a

    try:
        a0=lattice(0)
        offer(a0)
        if best_score > 2.32 and best_a is not None:
            solve(best_a,False,max(.05,deadline-time.monotonic()-.015))
        else:
            solve(a0,True,budget_s*params['first_fraction'])
            if time.monotonic() < deadline-.12:
                solve(lattice(1),True,min(budget_s*params['second_fraction'],deadline-time.monotonic()-.10))
            if best_a is not None and time.monotonic() < deadline-.025:
                solve(best_a,False,deadline-time.monotonic()-.008)
    except Stop:
        pass
    return best
