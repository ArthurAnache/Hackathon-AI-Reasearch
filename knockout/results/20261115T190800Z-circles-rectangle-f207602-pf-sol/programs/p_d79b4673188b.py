import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'search_fraction': (0.70, 0.4, 0.75),
    'polish_iterations': (100, 20, 220),
    'penalty_iterations': (35, 10, 100),
    'noise': (0.024, 0.001, 0.12),
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(len(I))

class Deadline(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    end = time.monotonic() + float(params['search_fraction']) * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf

    def check():
        if time.monotonic() >= end:
            raise Deadline()

    def valid(a):
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return False
        r = a[:, 2]
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < r[I]+r[J]):
            return False
        x, y, q = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        perimeter = (max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
                     + max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q)))
        return perimeter <= 2

    def retain(a, original=None):
        nonlocal best, best_score
        s = float(np.sum(a[:, 2]))
        if s > best_score and valid(a):
            best_score = s
            best = original if original is not None else {'circles': a.tolist()}

    def normalize(a):
        p, r = a[:, :2].copy(), a[:, 2].copy()
        lo = np.min(p-r[:, None], axis=0)
        side = np.max(p+r[:, None], axis=0)-lo
        scale = 2.0 / float(side.sum())
        p = (p-lo)*scale
        return np.r_[p[:, 0], p[:, 1], r*scale, side[0]*scale]

    def repair(z):
        p = np.column_stack((z[:N], z[N:2*N])).copy()
        r = np.maximum(z[2*N:3*N], 1e-12).copy()
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            return
        d = np.sqrt((p[I,0]-p[J,0])**2+(p[I,1]-p[J,1])**2)
        f = min(1.0, float(np.min(d/(r[I]+r[J]))))
        if f <= 0:
            return
        r *= f*(1-3e-14)
        lo = np.min(p-r[:, None], axis=0)
        side = np.max(p+r[:, None], axis=0)-lo
        scale = (2-5e-14)/float(side.sum())
        p = (p-lo)*scale
        r *= scale*(1-3e-14)
        retain(np.column_stack((p,r)))

    if init is not None:
        a0 = np.asarray(init['circles'], dtype=float)
        retain(a0, init)
    else:
        xx, yy = np.meshgrid(np.arange(5), np.arange(5))
        p = np.column_stack((xx.ravel(), yy.ravel()))[:N].astype(float)
        p[:,0] += 0.5*(p[:,1] % 2)
        p[:,1] *= np.sqrt(3)/2
        a0 = np.column_stack((p, np.full(N, 0.48)))
        repair(np.r_[p[:,0],p[:,1],a0[:,2],1.0])

    A = np.zeros((4*N, 3*N+1))
    b = np.zeros(4*N)
    k = np.arange(N)
    A[k,k] = 1; A[k,2*N+k] = -1
    A[N+k,k] = -1; A[N+k,2*N+k] = -1; A[N+k,-1] = 1
    A[2*N+k,N+k] = 1; A[2*N+k,2*N+k] = -1
    A[3*N+k,N+k] = -1; A[3*N+k,2*N+k] = -1; A[3*N+k,-1] = -1
    b[3*N:] = 2
    grad = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    bounds = [(None,None)]*(2*N)+[(1e-7,0.4)]*N+[(0.1,1.9)]

    def fun(z):
        check()
        return -float(z[2*N:3*N].sum())

    def con(z):
        check()
        dx, dy = z[I]-z[J], z[N+I]-z[N+J]
        r = z[2*N+I]+z[2*N+J]
        return dx*dx+dy*dy-r*r

    def jac(z):
        check()
        dx, dy = 2*(z[I]-z[J]), 2*(z[N+I]-z[N+J])
        r = -2*(z[2*N+I]+z[2*N+J])
        a = np.zeros((len(I),3*N+1))
        a[K,I] = dx; a[K,J] = -dx
        a[K,N+I] = dy; a[K,N+J] = -dy
        a[K,2*N+I] = r; a[K,2*N+J] = r
        return a

    def penalty(z, mu):
        check()
        dx, dy = z[I]-z[J], z[N+I]-z[N+J]
        d = np.sqrt(dx*dx+dy*dy+1e-26)
        v = np.maximum(z[2*N+I]+z[2*N+J]-d,0)
        wall = np.minimum(A@z+b,0)
        c = 2*mu*v
        gx = np.bincount(I,-c*dx/d,minlength=N)+np.bincount(J,c*dx/d,minlength=N)
        gy = np.bincount(I,-c*dy/d,minlength=N)+np.bincount(J,c*dy/d,minlength=N)
        gr = -np.ones(N)+np.bincount(I,c,minlength=N)+np.bincount(J,c,minlength=N)
        g = np.r_[gx,gy,gr,0.0]+2*mu*(A.T@wall)
        return -z[2*N:3*N].sum()+mu*(v@v+wall@wall), g

    last = [normalize(a0)]
    count = [0]
    def callback(z):
        check()
        last[0] = z.copy()
        count[0] += 1
        if count[0] % 7 == 0:
            repair(z)

    def polish(z, iterations):
        last[0] = z.copy()
        result = minimize(fun,z,jac=lambda z:grad,method='SLSQP',bounds=bounds,
            constraints=[{'type':'ineq','fun':lambda z:A@z+b,'jac':lambda z:A},
                         {'type':'ineq','fun':con,'jac':jac}],callback=callback,
            options={'maxiter':iterations,'ftol':2e-14})
        repair(result.x)
        return result.x

    try:
        polish(last[0], int(params['polish_iterations']))
        attempt = 0
        while time.monotonic() < end:
            check()
            a = np.asarray(best['circles'], dtype=float) if best is not None else a0.copy()
            z = normalize(a)
            p = np.column_stack((z[:N],z[N:2*N]))
            scale = float(params['noise'])*(1.0,2.0,0.5,3.5)[attempt % 4]
            displacement = rng.normal(0,scale,(N,2))
            dr = rng.normal(0,scale*0.15,N)
            if attempt % 3 != 2:
                # Detect the better reflection and coordinate complete mirror orbits.
                choices = []
                for axis in (0,1):
                    check()
                    center = 0.5*(np.min(p[:,axis]-z[2*N:3*N])+np.max(p[:,axis]+z[2*N:3*N]))
                    reflected = p.copy()
                    reflected[:,axis] = 2*center-reflected[:,axis]
                    distances = np.sum((reflected[:,None,:]-p[None,:,:])**2,axis=2)
                    mate = np.argmin(distances,axis=1)
                    choices.append((float(np.sum(np.min(distances,axis=1))),axis,mate))
                _, axis, mate = min(choices,key=lambda v:v[0])
                for i in range(N):
                    check()
                    j = int(mate[i])
                    if j > i and mate[j] == i:
                        displacement[j] = displacement[i]
                        displacement[j,axis] *= -1
                        dr[j] = dr[i]
                    elif j == i:
                        displacement[i,axis] = 0
            p += displacement
            z[:N], z[N:2*N] = p[:,0], p[:,1]
            z[2*N:3*N] = np.maximum(0.015,z[2*N:3*N]+dr)
            for mu in (80.0, 1600.0):
                check()
                result = minimize(penalty,z,args=(mu,),jac=True,method='L-BFGS-B',bounds=bounds,
                    options={'maxiter':int(params['penalty_iterations']),'ftol':1e-11,'maxls':15})
                z = result.x
                last[0] = z.copy()
            polish(z,int(params['polish_iterations']))
            attempt += 1
    except Deadline:
        repair(last[0])
    return best
