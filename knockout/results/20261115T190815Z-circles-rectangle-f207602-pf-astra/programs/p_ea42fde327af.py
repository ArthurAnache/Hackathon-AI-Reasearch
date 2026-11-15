import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

PARAMS = {"grid": (61, 31, 101), "restarts": (4, 1, 8), "iterations": (65, 20, 140), "separation": (2.3, 1.5, 4.0)}

class Deadline(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    n = 21
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)
    rows = np.arange(m)
    ids = np.arange(n)
    best = init
    best_score = -float('inf')

    def check():
        if time.monotonic() >= deadline:
            raise Deadline()

    def exact(c):
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        xs, ys, rs = [], [], []
        for x, y, r in c:
            check()
            xs.append(Fraction(float(x)))
            ys.append(Fraction(float(y)))
            rs.append(Fraction(float(r)))
        width = max(x+r for x,r in zip(xs,rs)) - min(x-r for x,r in zip(xs,rs))
        height = max(y+r for y,r in zip(ys,rs)) - min(y-r for y,r in zip(ys,rs))
        if width + height > 2:
            return -float('inf')
        for a, b in zip(ii, jj):
            check()
            dist = np.sqrt((c[a,0]-c[b,0])**2 + (c[a,1]-c[b,1])**2)
            if dist < c[a,2] + c[b,2]:
                return -float('inf')
        return float(np.sum(c[:,2]))

    def capture(z):
        nonlocal best, best_score
        check()
        c = np.column_stack((z[:n], z[n:2*n], z[2*n:3*n])).copy()
        if not np.isfinite(c).all() or np.min(c[:,2]) <= 0:
            return
        d = np.sqrt((c[ii,0]-c[jj,0])**2 + (c[ii,1]-c[jj,1])**2)
        factor = min(1.0, float(np.min(d/(c[ii,2]+c[jj,2]))))
        c[:,2] *= factor * (1.0-2e-14)
        ext = np.max(c[:,:2]+c[:,2,None],axis=0)-np.min(c[:,:2]-c[:,2,None],axis=0)
        c *= (2.0 / float(np.sum(ext))) * (1.0-2e-14)
        if float(np.sum(c[:,2])) <= best_score:
            return
        score = exact(c)
        if score > best_score:
            best_score = score
            best = {"circles": c.tolist()}

    def constraints(z):
        check()
        x,y,r,w = z[:n],z[n:2*n],z[2*n:3*n],z[-1]
        dx,dy = x[ii]-x[jj],y[ii]-y[jj]
        return np.concatenate((dx*dx+dy*dy-(r[ii]+r[jj])**2,x-r,w-x-r,y-r,2-w-y-r))

    wall = np.zeros((4*n,3*n+1))
    wall[ids,ids] = 1
    wall[ids,2*n+ids] = -1
    wall[n+ids,ids] = -1
    wall[n+ids,2*n+ids] = -1
    wall[n+ids,-1] = 1
    wall[2*n+ids,n+ids] = 1
    wall[2*n+ids,2*n+ids] = -1
    wall[3*n+ids,n+ids] = -1
    wall[3*n+ids,2*n+ids] = -1
    wall[3*n+ids,-1] = -1

    def jacobian(z):
        check()
        x,y,r = z[:n],z[n:2*n],z[2*n:3*n]
        out = np.zeros((m+4*n,3*n+1))
        dx,dy,s = 2*(x[ii]-x[jj]),2*(y[ii]-y[jj]),-2*(r[ii]+r[jj])
        out[rows,ii]=dx
        out[rows,jj]=-dx
        out[rows,n+ii]=dy
        out[rows,n+jj]=-dy
        out[rows,2*n+ii]=s
        out[rows,2*n+jj]=s
        out[m:]=wall
        return out

    grad = np.zeros(3*n+1)
    grad[2*n:3*n] = -1
    def objective(z):
        check()
        return -float(np.sum(z[2*n:3*n]))

    try:
        if init is not None:
            c = np.asarray(init['circles'],dtype=float).copy()
            best_score = exact(c)
        else:
            c = np.empty((n,3))
            for k in range(n):
                check()
                row,col=divmod(k,5)
                c[k]=[0.10+0.19*col,0.10+0.19*row,0.089]
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,:2] -= lo
        dims = hi-lo
        c *= 2.0/np.sum(dims)
        dims *= 2.0/np.sum(dims)
        w,h = dims
        base = np.r_[c[:,0],c[:,1],c[:,2],w]
        capture(base)

        g = int(params['grid'])
        gx,gy = np.meshgrid(np.linspace(0,w,g),np.linspace(0,h,g))
        points=np.column_stack((gx.ravel(),gy.ravel()))
        border=np.min(np.column_stack((points[:,0],w-points[:,0],points[:,1],h-points[:,1])),axis=1)
        distances=np.sqrt(np.sum((points[:,None,:]-c[None,:,:2])**2,axis=2))
        clearance=distances-c[None,:,2]
        nearest=np.argmin(clearance,axis=1)
        two=np.partition(clearance,1,axis=1)[:,:2]
        moves=[]
        for k in range(n):
            check()
            available=np.minimum(border,np.where(nearest==k,two[:,1],two[:,0]))
            available[distances[:,k] < float(params['separation'])*c[k,2]]=-1
            q=int(np.argmax(available))
            if available[q]>0:
                moves.append((float(available[q]-c[k,2]),k,q,float(available[q])))
        moves.sort(reverse=True)
        used=[]
        count=0
        for merit,k,q,rnew in moves:
            check()
            if count >= int(params['restarts']):
                break
            if any(np.linalg.norm(points[q]-p)<0.08 and k==j for j,p in used):
                continue
            used.append((k,points[q].copy()))
            z=base.copy()
            z[k],z[n+k]=points[q]
            z[2*n+k]=max(0.002,rnew*0.98)
            count+=1
            remaining=deadline-time.monotonic()
            if remaining<0.035:
                break
            local_deadline=min(deadline,time.monotonic()+remaining/max(1,int(params['restarts'])-count+1))
            last=[z.copy()]
            def callback(v):
                check()
                last[0]=v.copy()
                capture(v)
                if time.monotonic()>=local_deadline:
                    raise Deadline()
            bounds=[(0,2)]*(2*n)+[(1e-5,0.5)]*n+[(0.2,1.8)]
            try:
                result=minimize(objective,z,jac=lambda v:grad,method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':constraints,'jac':jacobian},callback=callback,
                    options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
                capture(result.x)
            except Deadline:
                if time.monotonic()>=deadline:
                    break
                capture(last[0])
    except Deadline:
        pass
    if best is None:
        fallback=[]
        for k in range(21):
            if time.monotonic() >= start+0.9*budget_s:
                return {'circles': [[0.1+0.19*(j%5),0.1+0.19*(j//5),0.08] for j in range(21)]}
            fallback.append([0.1+0.19*(k%5),0.1+0.19*(k//5),0.08])
        best={'circles':fallback}
    return best
