import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (160, 50, 300), "time_fraction": (0.68, 0.4, 0.75)}

I, J = np.triu_indices(21, 1)

class StopSearch(Exception):
    pass

def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if p > 2:
            return -float('inf')
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        if np.any(d < a[I,2]+a[J,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')

def repair(z):
    a = np.asarray(z[:63], dtype=float).reshape(21,3).copy()
    if not np.isfinite(a).all():
        return None
    a[:,2] = np.maximum(a[:,2], 1e-9)
    d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
    f = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
    if f <= 0:
        return None
    a[:,2] *= f*(1-3e-12)
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    a[:,:2] -= lo
    a *= (2.0/(np.sum(hi-lo)))*(1-3e-12)
    return {'circles':a.tolist()}

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + float(params['time_fraction'])*budget_s
    best = init
    bestscore = exact_score(init) if init is not None else -float('inf')
    saved = []
    rng = np.random.default_rng(seed)

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def consider(z):
        nonlocal best, bestscore
        c = repair(z)
        if c is not None:
            s = exact_score(c)
            if s > bestscore:
                best, bestscore = c, s

    # Full variables are interleaved (x,y,r), followed by rectangle width.
    def constraints(z):
        check()
        a=z[:63].reshape(21,3)
        x,y,r=a.T
        w=z[63]
        dx=x[I]-x[J]; dy=y[I]-y[J]; rr=r[I]+r[J]
        return np.concatenate((x-r,w-x-r,y-r,2-w-y-r,dx*dx+dy*dy-rr*rr))

    def derivative(z):
        check()
        a=z[:63].reshape(21,3)
        x,y,r=a.T
        g=np.zeros((294,64))
        k=np.arange(21)
        g[k,3*k]=1; g[k,3*k+2]=-1
        g[21+k,3*k]=-1; g[21+k,3*k+2]=-1; g[21+k,63]=1
        g[42+k,3*k+1]=1; g[42+k,3*k+2]=-1
        g[63+k,3*k+1]=-1; g[63+k,3*k+2]=-1; g[63+k,63]=-1
        q=np.arange(210)+84
        dx=2*(x[I]-x[J]); dy=2*(y[I]-y[J]); rr=-2*(r[I]+r[J])
        g[q,3*I]=dx; g[q,3*J]=-dx
        g[q,3*I+1]=dy; g[q,3*J+1]=-dy
        g[q,3*I+2]=rr; g[q,3*J+2]=rr
        return g

    obj=np.zeros(64); obj[2:63:3]=-1
    M=np.zeros((64,34)); offset=np.zeros(64)
    for k in range(3):
        M[3*k,2*k]=1
        M[3*k+2,2*k+1]=1
        offset[3*k+1]=1; M[3*k+1,33]=-.5
    for k in range(9):
        for side in range(2):
            n=3+2*k+side
            M[3*n,6+3*k]=1
            M[3*n+2,8+3*k]=1
            offset[3*n+1]=1
            M[3*n+1,33]=-.5
            M[3*n+1,7+3*k]=1 if side==0 else -1
    M[63,33]=1

    def run(v, symmetric, maxiter):
        last=[None]; count=[0]
        def expand(u):
            return offset+M@u if symmetric else u
        gradient=M.T@obj if symmetric else obj
        def fun(u):
            check()
            return float(gradient@u)
        def jac(u):
            check()
            return gradient
        def con(u):
            return constraints(expand(u))
        def cj(u):
            g=derivative(expand(u))
            return g@M if symmetric else g
        def callback(u):
            check()
            last[0]=expand(u).copy()
            count[0]+=1
            if count[0]%8==0:
                saved.append(last[0])
        bounds=[(None,None)]*len(v)
        bounds[-1]=(.12,1.88)
        if symmetric:
            for k in range(3): bounds[2*k+1]=(1e-7,.6)
            for k in range(9): bounds[8+3*k]=(1e-7,.6)
        else:
            for k in range(21): bounds[3*k+2]=(1e-7,.6)
        try:
            res=minimize(fun,v,jac=jac,method='SLSQP',bounds=bounds,
                         constraints={'type':'ineq','fun':con,'jac':cj},callback=callback,
                         options={'maxiter':maxiter,'ftol':2e-11,'disp':False})
            saved.append(expand(res.x).copy())
        except StopSearch:
            pass
        if last[0] is not None:
            saved.append(last[0])

    # Three axis circles and nine reflected pairs, arranged in five rows.
    for attempt in range(5):
        if time.monotonic() >= deadline-.10:
            break
        radius=2/(10+2+4*np.sqrt(3))
        v=np.zeros(34); v[33]=10*radius
        choices=([1,3,7],[1,5,7],[1,3,5],[3,5,7])
        axis=choices[attempt%4]
        for k in range(3):
            v[2*k]=(axis[k]+1)*radius
            v[2*k+1]=radius*.96
        for k in range(9):
            if k<5:
                xx=2*k+1; yy=np.sqrt(3)
            else:
                xx=2*(k-5)+2; yy=2*np.sqrt(3)
            v[6+3*k]=xx*radius
            v[7+3*k]=yy*radius
            v[8+3*k]=radius*.96
        if attempt>=2:
            v[:33] += rng.normal(0,.003,33)
        z=offset+M@v
        consider(z)
        run(v,True,int(params['iterations']))
        while saved:
            if time.monotonic() >= start+.78*budget_s:
                break
            consider(saved.pop())
        if bestscore>2.34 and time.monotonic()<deadline-.18:
            a=np.array(best['circles'])
            lo=np.min(a[:,:2]-a[:,2,None],axis=0)
            a[:,:2]-=lo
            w=float(np.max(a[:,0]+a[:,2]))
            run(np.r_[a.ravel(),w],False,70)
            break
    while saved:
        if time.monotonic() >= start+.86*budget_s:
            break
        consider(saved.pop())
    if best is None:
        a=np.zeros((21,3)); a[:,0]=np.arange(21)*.045; a[:,2]=.02
        best=repair(np.r_[a.ravel(),1.0])
    return best
