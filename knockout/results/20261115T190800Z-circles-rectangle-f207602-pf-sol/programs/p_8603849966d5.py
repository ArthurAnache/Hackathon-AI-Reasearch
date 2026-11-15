import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'perturbation': (0.025, 0.003, 0.09, 'log'),
    'penalty': (180.0, 20.0, 3000.0, 'log'),
    'safety': (8e-15, 2e-15, 1e-11, 'log'),
    'iterations': (90, 20, 200)
}

class Halt(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    nr = len(ii)
    rows = np.arange(nr)
    eye = np.eye(n)
    safety = float(params['safety'])
    best = init
    best_score = -np.inf

    def valid(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        xs, ys, rs = ([Fraction(float(v)) for v in a[:, k]] for k in range(3))
        perimeter = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        perimeter += max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if perimeter > 2:
            return -np.inf
        for i,j in zip(ii,jj):
            if time.monotonic() > start + 0.94*budget_s:
                return -np.inf
            d = np.sqrt((a[i,0]-a[j,0])**2 + (a[i,1]-a[j,1])**2)
            if d < a[i,2]+a[j,2]:
                return -np.inf
        return float(np.sum(a[:,2]))

    if init is not None:
        a0 = np.array(init['circles'], dtype=float)
        best_score = valid(a0)
    else:
        a0 = np.array([[2*c+1.,2*r+1.,1.] for r in range(4) for c in range(5)] + [[4.,4.,np.sqrt(2.)-1.]])/9

    def check(end=deadline):
        if time.monotonic() >= end:
            raise Halt()

    def retain(a):
        nonlocal best,best_score
        check()
        a = a.copy()
        if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        dx = a[ii,0]-a[jj,0]
        dy = a[ii,1]-a[jj,1]
        d = np.sqrt(dx*dx+dy*dy)
        ratio = min(1.,float(np.min(d/(a[ii,2]+a[jj,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1-safety)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= lo
        a *= (2/float(np.sum(hi-lo)))*(1-safety)
        if float(np.sum(a[:,2])) > best_score:
            score = valid(a)
            if score > best_score:
                best_score = score
                best = {'circles':a.tolist()}

    def normalize(a):
        a = a.copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        wh = np.max(a[:,:2]+a[:,2,None],axis=0)-lo
        a[:,:2] -= lo
        scale = 2/np.sum(wh)
        a *= scale
        return np.r_[a[:,0],a[:,1],a[:,2],wh[0]*scale]

    def constraints(v, need_jac=True):
        x,y,r,w = v[:21],v[21:42],v[42:63],v[63]
        dx=x[ii]-x[jj]
        dy=y[ii]-y[jj]
        d=np.sqrt(dx*dx+dy*dy+1e-30)
        c=np.r_[d-r[ii]-r[jj],x-r,w-x-r,y-r,2-w-y-r]
        if not need_jac:
            return c
        j=np.zeros((294,64))
        j[rows,ii]=dx/d
        j[rows,jj]=-dx/d
        j[rows,21+ii]=dy/d
        j[rows,21+jj]=-dy/d
        j[rows,42+ii]=-1
        j[rows,42+jj]=-1
        j[210:231,:21]=eye
        j[210:231,42:63]=-eye
        j[231:252,:21]=-eye
        j[231:252,42:63]=-eye
        j[231:252,63]=1
        j[252:273,21:42]=eye
        j[252:273,42:63]=-eye
        j[273:294,21:42]=-eye
        j[273:294,42:63]=-eye
        j[273:294,63]=-1
        return c,j

    grad=np.r_[np.zeros(42),-np.ones(21),0.]

    def optimize(q, M, b, allowance, penalty=False):
        end=min(deadline,time.monotonic()+allowance)
        last=q.copy()
        def unpack(u):
            return M@u+b
        def callback(u):
            nonlocal last
            check(end)
            last=u.copy()
            v=unpack(u)
            retain(np.column_stack((v[:21],v[21:42],v[42:63])))
        def f(u):
            check(end)
            v=unpack(u)
            return float(grad@v)
        def c(u):
            check(end)
            return constraints(unpack(u),False)
        def j(u):
            check(end)
            return constraints(unpack(u))[1]@M
        try:
            if penalty:
                mu=float(params['penalty'])
                def soft(u):
                    check(end)
                    v=unpack(u)
                    cv,jv=constraints(v)
                    neg=np.minimum(cv,0.)
                    return float(grad@v+mu*(neg@neg)), (grad+2*mu*(neg@jv))@M
                res=minimize(soft,q,jac=True,method='L-BFGS-B',callback=lambda u: check(end),options={'maxiter':35,'ftol':1e-9,'maxls':12})
                last=res.x
            else:
                res=minimize(f,q,jac=lambda u:grad@M,method='SLSQP',constraints={'type':'ineq','fun':c,'jac':j},callback=callback,options={'maxiter':int(params['iterations']),'ftol':2e-13,'disp':False})
                last=res.x
                callback(last)
        except Halt:
            pass
        return last

    try:
        retain(a0)
        z=normalize(a0)
        optimize(z,np.eye(64),np.zeros(64),min(0.19*budget_s,0.28))
        check()
        anchor=np.array(best['circles'],float) if best is not None else a0
        # Choose the orientation whose horizontal reflection most closely matches.
        chosen=None
        for swap in (False,True):
            check()
            a=anchor.copy()
            if swap:
                a[:,[0,1]]=a[:,[1,0]]
            v=normalize(a)
            p=np.column_stack((v[:21],v[21:42]))
            reflected=p.copy()
            reflected[:,1]=2-v[63]-reflected[:,1]
            cost=np.sum((p[:,None,:]-reflected[None,:,:])**2,axis=2)+(a[:,2,None]-a[None,:,2])**2
            ri,match=linear_sum_assignment(cost)
            error=float(cost[ri,match].sum())
            if chosen is None or error < chosen[0]:
                chosen=(error,v,match,swap)
        _,v,match,swap=chosen
        groups=[]
        seen=set()
        for i in range(21):
            check()
            if i in seen:
                continue
            j=int(match[i])
            if int(match[j]) != i:
                j=i
            group=[i] if j==i or j in seen else [i,j]
            groups.append(group)
            seen.update(group)
        ng=len(groups)
        pairs=[g for g in groups if len(g)==2]
        npair=len(pairs)
        dim=2*ng+npair+1
        M=np.zeros((64,dim))
        b=np.zeros(64)
        q=np.zeros(dim)
        q[-1]=v[63]
        M[63,-1]=1
        k=0
        for gidx,g in enumerate(groups):
            check()
            q[gidx]=np.mean(v[g])
            q[ng+npair+gidx]=np.mean(v[42+np.array(g)])
            for i in g:
                check()
                M[i,gidx]=1
                M[42+i,ng+npair+gidx]=1
                b[21+i]=1
                M[21+i,-1]=-.5
            if len(g)==2:
                i,jj0=g
                q[ng+k]=(v[21+i]-v[21+jj0])/2
                M[21+i,ng+k]=1
                M[21+jj0,ng+k]=-1
                k+=1
        # Coordinates in this orientation are equivalent; no inverse transform is needed.
        q=optimize(q,M,b,0.15)
        center=q.copy()
        attempt=0
        while time.monotonic() < deadline-0.08:
            check()
            trial=center.copy()
            amp=float(params['perturbation'])*(1.0+0.6*(attempt%3))
            noise=rng.normal(size=dim)
            trial[:ng+npair]+=amp*noise[:ng+npair]
            trial[ng+npair:-1]+=amp*.35*noise[ng+npair:-1]
            trial[-1]+=amp*noise[-1]
            remaining=deadline-time.monotonic()
            trial=optimize(trial,M,b,min(.075,remaining*.24),True)
            optimize(trial,M,b,min(.29,deadline-time.monotonic()))
            attempt+=1
    except (Halt,ValueError,FloatingPointError):
        pass
    if best is not None:
        return best
    fallback=a0.copy()
    fallback[:,2]*=1-1e-9
    fallback*=1-1e-9
    return {'circles':fallback.tolist()}
