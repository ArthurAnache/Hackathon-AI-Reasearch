import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'equalization': (8.0, 0.5, 30.0, 'log'),
    'excursion_iterations': (95, 20, 250),
    'relax_iterations': (110, 20, 300),
    'initial_penalty': (45.0, 5.0, 500.0, 'log'),
    'attempts': (3, 1, 8)
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2] + a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    if w+h > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = -np.inf
    if init is not None:
        a = np.asarray(init['circles'], dtype=float)
        best_score = exact_score(a)
    else:
        a = np.array([[(k % 5 + .5)*.19, (k // 5 + .5)*.19, .07] for k in range(N)])
        best = {'circles': a.tolist()}
        best_score = exact_score(a)
    lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
    sides = np.max(a[:, :2]+a[:, 2, None], axis=0)-lo
    scale = 2.0 / np.sum(sides)
    base = np.r_[(a[:, 0]-lo[0])*scale, (a[:, 1]-lo[1])*scale,
                 a[:, 2]*scale, sides[0]*scale]
    last = base.copy()

    def check():
        if time.monotonic() >= end:
            raise Deadline()

    def constraints(z):
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

    def jacobian(z):
        x,y = z[:N],z[N:2*N]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        m = len(I)
        jac = np.zeros((m+4*N,3*N+1))
        q = np.arange(m)
        jac[q,I] = dx/d
        jac[q,J] = -dx/d
        jac[q,N+I] = dy/d
        jac[q,N+J] = -dy/d
        jac[q,2*N+I] = -1
        jac[q,2*N+J] = -1
        for b in range(4):
            check()
            rows = m+b*N+K
            jac[rows,2*N+K] = -1
            jac[rows,(0 if b < 2 else N)+K] = 1 if b % 2 == 0 else -1
        jac[m+N:m+2*N,-1] = 1
        jac[m+3*N:,-1] = -1
        return jac

    def penalty(z, mu, alpha):
        nonlocal last
        check()
        last = z.copy()
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        v = np.maximum(r[I]+r[J]-d,0)
        b = np.array([np.maximum(r-x,0), np.maximum(r+x-w,0),
                      np.maximum(r-y,0), np.maximum(r+y-(2-w),0)])
        rr = r-r.mean()
        f = -r.sum()+alpha*np.dot(rr,rr)+mu*(np.dot(v,v)+np.sum(b*b))
        gx,gy = np.zeros(N),np.zeros(N)
        gr = -np.ones(N)+2*alpha*rr
        c = 2*mu*v
        np.add.at(gx,I,-c*dx/d)
        np.add.at(gx,J,c*dx/d)
        np.add.at(gy,I,-c*dy/d)
        np.add.at(gy,J,c*dy/d)
        np.add.at(gr,I,c)
        np.add.at(gr,J,c)
        gx += 2*mu*(-b[0]+b[1])
        gy += 2*mu*(-b[2]+b[3])
        gr += 2*mu*np.sum(b,axis=0)
        gw = 2*mu*(-b[1].sum()+b[3].sum())
        return f,np.r_[gx,gy,gr,gw]

    def consider(z):
        nonlocal best,best_score
        if time.monotonic() >= end + .04*budget_s:
            return
        p = np.column_stack((z[:N],z[N:2*N]))
        r = np.maximum(z[2*N:3*N],1e-14).copy()
        d = np.sqrt(np.sum((p[I]-p[J])**2,axis=1))
        ratio = min(1.0,float(np.min(d/(r[I]+r[J]))))
        r *= ratio*(1-2e-14)
        low = np.min(p-r[:,None],axis=0)
        high = np.max(p+r[:,None],axis=0)
        s = min(1.0,2.0/np.sum(high-low))*(1-2e-14)
        out = np.column_stack(((p-low)*s,r*s))
        score = exact_score(out)
        if score > best_score:
            best_score = score
            best = {'circles': out.tolist()}

    bounds = [(0,2)]*(2*N)+[(1e-9,.7)]*N+[(.2,1.8)]
    objective_gradient = np.r_[np.zeros(2*N),-np.ones(N),0.0]
    def objective(z):
        nonlocal last
        check()
        last = z.copy()
        return -np.sum(z[2*N:3*N])

    try:
        for trial in range(int(params['attempts'])):
            check()
            z = base.copy()
            alpha = float(params['equalization']) * (1.65**trial)
            if trial == 1:
                alpha *= -0.35
            if trial > 0:
                z[:2*N] += rng.normal(0,.004,2*N)
            z = minimize(penalty,z,args=(float(params['initial_penalty']),alpha),jac=True,
                         method='L-BFGS-B',bounds=bounds,
                         options={'maxiter':int(params['excursion_iterations']),'ftol':1e-10,'maxls':18}).x
            check()
            z = minimize(penalty,z,args=(450.0,0.0),jac=True,
                         method='L-BFGS-B',bounds=bounds,
                         options={'maxiter':80,'ftol':1e-11,'maxls':18}).x
            check()
            z = minimize(objective,z,jac=lambda z: objective_gradient,method='SLSQP',
                         bounds=bounds,constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                         options={'maxiter':int(params['relax_iterations']),'ftol':2e-13}).x
            consider(z)
    except Deadline:
        consider(last)
    return best
