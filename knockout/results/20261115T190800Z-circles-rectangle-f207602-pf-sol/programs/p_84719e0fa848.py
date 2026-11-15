import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

PARAMS = {"family_iterations": (110, 40, 180), "repair_iterations": (65, 20, 120), "shrink": (2e-13, 1e-14, 1e-10, "log")}

class StopSearch(Exception):
    pass

def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=float)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -np.inf
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        p = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
        p += max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
        if p > 2:
            return -np.inf
        i,j = np.triu_indices(21, 1)
        d = c[i,:2]-c[j,:2]
        if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < c[i,2]+c[j,2]):
            return -np.inf
        return float(np.sum(c[:,2]))
    except Exception:
        return -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -np.inf
    shrink = params['shrink']
    archive = []
    graphs = set()
    ii,jj = np.triu_indices(21,1)

    def save(c):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        c = np.asarray(c, float).copy()
        if not np.isfinite(c).all() or np.min(c[:,2]) <= 0:
            return
        delta = c[ii,:2]-c[jj,:2]
        dist = np.sqrt(np.sum(delta*delta,axis=1))
        factor = min(1.0, float(np.min(dist/(c[ii,2]+c[jj,2]))))
        if factor <= 0:
            return
        c[:,2] *= factor*(1-shrink)
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        perimeter = float(np.sum(hi-lo))
        c[:,:2] -= (lo+hi)/2
        c *= (2.0/perimeter)*(1-shrink)
        score = float(np.sum(c[:,2]))
        if score <= best_score:
            return
        obj = {'circles': c.tolist()}
        score = exact_score(obj)
        if score > best_score:
            best_score, best = score, obj

    def setup(rows):
        a,b,c = rows
        k=a+b+c
        x=np.concatenate([np.arange(t)*2-(t-1) for t in rows]).astype(float)
        y=np.concatenate([np.zeros(a),np.full(b,np.sqrt(3)),np.full(c,2*np.sqrt(3))])
        r=np.ones(k)
        width=np.max(x+r)-np.min(x-r)
        height=2*np.max(y+r)
        scale=2/(width+height)
        return np.r_[x*scale,y[a:]*scale,r*scale,width/(width+height)]

    def search(rows, z0, iterations, local_end):
        a,b,c = rows
        k=a+b+c
        q=k-a
        nv=3*k-a+1
        ri=2*k-a
        widx=nv-1
        I,J=np.triu_indices(k,1)
        pairs=len(I)
        mult=np.r_[np.ones(a),np.full(q,2.)]
        grad=np.zeros(nv)
        grad[ri:ri+k]=-mult
        template=np.zeros((pairs+3*k+q,nv))
        ar=np.arange(k)
        t=pairs
        template[t+ar,ar]=1
        template[t+ar,ri+ar]=-1
        template[t+ar,widx]=1
        t+=k
        template[t+ar,ar]=-1
        template[t+ar,ri+ar]=-1
        template[t+ar,widx]=1
        t+=k
        template[t+ar,ri+ar]=-1
        template[t+ar,widx]=-1
        template[t+np.arange(a,k),k+np.arange(q)]=-1
        t+=k
        template[t+np.arange(q),k+np.arange(q)]=1
        template[t+np.arange(q),ri+a+np.arange(q)]=-1
        cache_z=None
        cache_val=None
        cache_jac=None
        last=[z0.copy()]

        def unpack(z):
            return z[:k],np.r_[np.zeros(a),z[k:ri]],z[ri:ri+k],z[-1]

        def expand(z):
            x,y,r,w=unpack(z)
            upper=np.column_stack((x,y,r))
            lower=upper[a:].copy()
            lower[:,1]*=-1
            return np.vstack((upper,lower))

        def calculate(z):
            nonlocal cache_z,cache_val,cache_jac
            if time.monotonic() >= local_end:
                raise StopSearch()
            if cache_z is not None and np.array_equal(cache_z,z):
                return
            x,y,r,w=unpack(z)
            dx=x[I]-x[J]
            dy=y[I]-y[J]
            rs=r[I]+r[J]
            val=np.r_[dx*dx+dy*dy-rs*rs,w+x-r,w-x-r,1-w-y-r,y[a:]-r[a:]]
            jac=template.copy()
            pr=np.arange(pairs)
            jac[pr,I]=2*dx
            jac[pr,J]=-2*dx
            jac[pr,ri+I]=-2*rs
            jac[pr,ri+J]=-2*rs
            mask=I>=a
            jac[pr[mask],k+I[mask]-a]=2*dy[mask]
            mask=J>=a
            jac[pr[mask],k+J[mask]-a]=-2*dy[mask]
            cache_z=z.copy()
            cache_val=val
            cache_jac=jac

        def fun(z):
            if time.monotonic() >= local_end:
                raise StopSearch()
            return float(np.dot(grad,z))

        count=[0]
        def callback(z):
            if time.monotonic() >= local_end:
                raise StopSearch()
            last[0]=z.copy()
            count[0]+=1
            if count[0]%7==0:
                save(expand(z))

        def con(z):
            calculate(z)
            return cache_val
        def jac(z):
            calculate(z)
            return cache_jac
        bounds=[(-1,1)]*k+[(0,1)]*q+[(0.001,0.48)]*k+[(0.15,0.85)]
        try:
            result=minimize(fun,z0,jac=lambda z:grad,method='SLSQP',bounds=bounds,
                            constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                            options={'maxiter':iterations,'ftol':2e-12,'disp':False})
            last[0]=result.x.copy()
        except StopSearch:
            pass
        z=last[0]
        save(expand(z))
        x,y,r,w=unpack(z)
        d=np.sqrt((x[I]-x[J])**2+(y[I]-y[J])**2)-r[I]-r[J]
        key=(rows,tuple(np.flatnonzero(d<1e-5)))
        score=float(np.dot(mult,r))
        if key not in graphs:
            graphs.add(key)
            archive.append((score,rows,z.copy()))
        return z,score

    families=[(3,4,5),(5,4,4),(3,5,4),(5,3,5)]
    for index,rows in enumerate(families):
        now=time.monotonic()
        if now >= deadline-0.09:
            break
        allocation=min(0.24*budget_s, max(0.08, (deadline-now)*0.52))
        search(rows,setup(rows),params['family_iterations'],min(deadline-0.025,now+allocation))

    turn=0
    while time.monotonic() < deadline-0.06 and archive:
        archive.sort(key=lambda item:item[0],reverse=True)
        _,rows,z=archive[min(turn%3,len(archive)-1)]
        z=z.copy()
        a,b,c=rows
        k=a+b+c
        ri=2*k-a
        if turn:
            x=z[:k]
            y=np.r_[np.zeros(a),z[k:ri]]
            r=z[ri:ri+k]
            first=int(rng.integers(a,k))
            distance=(x-x[first])**2+(y-y[first])**2
            distance[:a]=np.inf
            removed=np.argsort(distance)[:2]
            survivors=np.ones(k,dtype=bool)
            survivors[removed]=False
            w=z[-1]
            for idx in removed:
                if time.monotonic() >= deadline-0.04:
                    break
                px=rng.uniform(-w*0.92,w*0.92,100)
                py=rng.uniform(0.01,max(0.02,1-w-0.01),100)
                clearance=np.minimum.reduce([w-px,w+px,1-w-py,py])
                dd=np.sqrt((px[:,None]-x[survivors])**2+(py[:,None]-y[survivors])**2)-r[survivors]
                clearance=np.minimum(clearance,np.min(dd,axis=1))
                choice=int(np.argmax(clearance))
                z[idx]=px[choice]
                z[k+idx-a]=py[choice]
                z[ri+idx]=max(0.002,float(clearance[choice])*0.92)
                x=z[:k]
                y=np.r_[np.zeros(a),z[k:ri]]
                r=z[ri:ri+k]
                survivors[idx]=True
        now=time.monotonic()
        search(rows,z,params['repair_iterations'],min(deadline-0.01,now+0.23*budget_s))
        turn+=1
    if best is None:
        r=1/21.000000001
        fallback={'circles': [[(i-10)*2*r,0.,r] for i in range(21)]}
        if exact_score(fallback)>best_score:
            best=fallback
    return best
