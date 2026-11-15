import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'block_size': (9, 5, 15),
    'mutation': (0.065, 0.01, 0.18),
    'local_iterations': (65, 20, 160),
    'global_iterations': (90, 30, 220),
    'polish_iterations': (65, 20, 180),
}
N = 21
I, J = np.triu_indices(N, 1)
M = len(I)

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -np.inf
        d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
        if np.any(d < a[I, 2]+a[J, 2]):
            return -np.inf
        f = [[Fraction(float(t)) for t in row] for row in a]
        per = (max(x+r for x,y,r in f)-min(x-r for x,y,r in f)
               +max(y+r for x,y,r in f)-min(y-r for x,y,r in f))
        if per > 2:
            return -np.inf
        return float(np.sum(a[:, 2]))

    if init is not None:
        original = np.array(init['circles'], dtype=float)
        best = init
        score = exact(original)
    else:
        original = np.array([[(i % 5)*0.19+0.1, (i//5)*0.19+0.1, 0.04] for i in range(21)])
        best = {'circles': original.tolist()}
        score = exact(original)

    lo = np.min(original[:, :2]-original[:, 2, None], axis=0)
    sides = np.max(original[:, :2]+original[:, 2, None], axis=0)-lo
    scale = 2.0 / sides.sum()
    a = original.copy()
    a[:, :2] = (a[:, :2]-lo)*scale
    a[:, 2] *= scale
    base = np.r_[a[:, 0], a[:, 1], a[:, 2], sides[0]*scale]
    elite = base.copy()
    elite_value = score

    def constraints(z, jac=False):
        check()
        x, y, r, w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx, dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        g = np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]
        if not jac:
            return g
        A = np.zeros((M+4*N, 3*N+1))
        rows = np.arange(M)
        A[rows, I] = dx/d
        A[rows, J] = -dx/d
        A[rows, N+I] = dy/d
        A[rows, N+J] = -dy/d
        A[rows, 2*N+I] = -1
        A[rows, 2*N+J] = -1
        q = np.arange(N)
        A[M+q, q] = 1
        A[M+N+q, q] = -1
        A[M+N+q, -1] = 1
        A[M+2*N+q, N+q] = 1
        A[M+3*N+q, N+q] = -1
        A[M+3*N+q, -1] = -1
        A[M+q, 2*N+q] = -1
        A[M+N+q, 2*N+q] = -1
        A[M+2*N+q, 2*N+q] = -1
        A[M+3*N+q, 2*N+q] = -1
        return A

    def penalty(z, mu):
        check()
        x,y,r,w = z[:N], z[N:2*N], z[2*N:3*N], z[-1]
        dx,dy = x[I]-x[J], y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        v = np.maximum(r[I]+r[J]-d, 0)
        b0 = np.maximum(r-x, 0)
        b1 = np.maximum(r+x-w, 0)
        b2 = np.maximum(r-y, 0)
        b3 = np.maximum(r+y+w-2, 0)
        c = 2*mu*v
        gx = np.bincount(I, -c*dx/d, minlength=N)+np.bincount(J, c*dx/d, minlength=N)
        gy = np.bincount(I, -c*dy/d, minlength=N)+np.bincount(J, c*dy/d, minlength=N)
        gr = -np.ones(N)+np.bincount(I,c,minlength=N)+np.bincount(J,c,minlength=N)
        gx += 2*mu*(-b0+b1)
        gy += 2*mu*(-b2+b3)
        gr += 2*mu*(b0+b1+b2+b3)
        f = -r.sum()+mu*(v@v+b0@b0+b1@b1+b2@b2+b3@b3)
        return f, np.r_[gx,gy,gr,2*mu*(b3.sum()-b1.sum())]

    bounds = [(None,None)]*(2*N)+[(1e-7,0.6)]*N+[(0.35,1.65)]

    def relax(z, mu, nit, free=None):
        check()
        if free is None:
            return minimize(penalty, z, args=(mu,), jac=True, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter':nit,'ftol':1e-12,'gtol':1e-7,'maxls':20}).x
        anchor = z.copy()
        def fun(v):
            check()
            zz = anchor.copy()
            zz[free] = v
            f,g = penalty(zz,mu)
            return f,g[free]
        out = minimize(fun, z[free], jac=True, method='L-BFGS-B',
                       bounds=[bounds[k] for k in free],
                       options={'maxiter':nit,'ftol':1e-11,'gtol':1e-7})
        anchor[free] = out.x
        return anchor

    def consider(z):
        nonlocal best, score, elite, elite_value
        check()
        p = np.stack((z[:N],z[N:2*N]),axis=1)
        r = np.maximum(z[2*N:3*N],1e-12).copy()
        d = np.sqrt(np.sum((p[I]-p[J])**2,axis=1))
        factor = min(1.0,float(np.min(d/(r[I]+r[J]))))
        r *= factor*(1-3e-14)
        lo2 = np.min(p-r[:,None],axis=0)
        hi2 = np.max(p+r[:,None],axis=0)
        s = (2.0/(hi2-lo2).sum())*(1-3e-14)
        candidate = np.column_stack(((p-lo2)*s,r*s))
        value = exact(candidate)
        if value > elite_value:
            elite_value = value
            elite = z.copy()
        if value > score:
            best = {'circles':candidate.tolist()}
            score = value

    objective_gradient = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    def obj(z):
        check()
        return -float(z[2*N:3*N].sum())
    def polish(z):
        check()
        return minimize(obj,z,jac=lambda z:objective_gradient,method='SLSQP',bounds=bounds,
                        constraints=[{'type':'ineq','fun':constraints,'jac':lambda z:constraints(z,True)}],
                        options={'maxiter':int(params['polish_iterations']),'ftol':2e-12}).x

    iteration = 0
    try:
        while time.monotonic() < deadline:
            check()
            z = elite.copy()
            pts = np.column_stack((z[:N],z[N:2*N]))
            center = int(rng.integers(N))
            order = np.argsort(np.sum((pts-pts[center])**2,axis=1))
            k = min(N,int(params['block_size'])+(iteration % 3)-1)
            block = order[:k]
            free = np.r_[block,N+block,2*N+block]
            delta = pts[block]-pts[block].mean(axis=0)
            angle = rng.choice([-1.0,1.0])*(0.25+0.35*rng.random())
            rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            displacement = delta@rotation.T-delta
            displacement += rng.normal(0,float(params['mutation'])*0.3,(k,2))
            z[block] += displacement[:,0]
            z[N+block] += displacement[:,1]
            if iteration % 2:
                z[2*N+block] = np.mean(z[2*N+block])
            z = relax(z,35.0,int(params['local_iterations']),free)
            expanded = order[:min(N,k+5)]
            z = relax(z,200.0,int(params['local_iterations']),np.r_[expanded,N+expanded,2*N+expanded,3*N])
            z = relax(z,2500.0,int(params['global_iterations']))
            consider(z)
            z = polish(z)
            consider(z)
            iteration += 1
    except StopSearch:
        pass
    return best
