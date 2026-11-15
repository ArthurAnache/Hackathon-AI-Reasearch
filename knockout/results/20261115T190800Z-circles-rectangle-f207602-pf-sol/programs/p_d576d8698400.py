import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'perturbation': (0.052, 0.005, 0.2), 'iterations': (95, 20, 250), 'fraction': (0.69, 0.3, 0.7)}

class Expired(Exception):
    pass

def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=float)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -math.inf
        x = [Fraction(float(v)) for v in c[:, 0]]
        y = [Fraction(float(v)) for v in c[:, 1]]
        r = [Fraction(float(v)) for v in c[:, 2]]
        if max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))+max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r)) > 2:
            return -math.inf
        for i in range(21):
            for j in range(i):
                d = np.sqrt((c[i,0]-c[j,0])**2+(c[i,1]-c[j,1])**2)
                if d < c[i,2]+c[j,2]:
                    return -math.inf
        return float(np.sum(c[:,2]))
    except Exception:
        return -math.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + params['fraction'] * budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = []
        for row, count in enumerate([4,5,4,5,3]):
            if time.monotonic() >= deadline:
                break
            for k in range(count):
                if time.monotonic() >= deadline:
                    break
                a.append([(k-(count-1)/2)*0.2, (row-2)*0.18, 0.09])
        if len(a) != 21:
            a = [[(i%7)*0.12, (i//7)*0.12, 0.05] for i in range(21)]
        init = {'circles': a}
    best = init
    best_score = exact_score(init)
    c = np.asarray(init['circles'], float).copy()
    c[:,0] -= (np.max(c[:,0]+c[:,2])+np.min(c[:,0]-c[:,2]))/2
    c[:,1] -= (np.max(c[:,1]+c[:,2])+np.min(c[:,1]-c[:,2]))/2
    def reflection_error(a):
        q = a.copy()
        q[:,0] *= -1
        return np.sum(np.min(np.sum((q[:,None,:]-a[None,:,:])**2,axis=2),axis=1))
    if reflection_error(c[:,[1,0,2]]) < reflection_error(c):
        c = c[:,[1,0,2]]
    groups = []
    remaining = set(range(21))
    while remaining:
        if time.monotonic() >= deadline:
            return best
        i = min(remaining)
        remaining.remove(i)
        target = c[i].copy()
        target[0] *= -1
        if abs(c[i,0]) < 1e-5:
            groups.append([i])
        elif remaining:
            j = min(remaining, key=lambda j: float(np.sum((target-c[j])**2)))
            if np.linalg.norm(target-c[j]) < 1e-4:
                remaining.remove(j)
                groups.append([i,j])
            else:
                groups.append([i])
        else:
            groups.append([i])
    symmetric = sum(len(g)==2 for g in groups) >= 7
    ax=[]; ay=[]; ar=[]; values=[]; bounds=[]
    def variable(value, lo, hi):
        values.append(float(value)); bounds.append((lo,hi))
        return len(values)-1
    assignments=[]
    for g in groups:
        if time.monotonic() >= deadline:
            return best
        if symmetric and len(g)==2:
            p = variable(np.mean(np.abs(c[g,0])), 1e-7, 1.0)
            q = variable(np.mean(c[g,1]), -1.0, 1.0)
            r = variable(np.mean(c[g,2]), 1e-7, 0.6)
            assignments.extend([(g[0],p,1,q,r),(g[1],p,-1,q,r)])
        else:
            for i in g:
                if time.monotonic() >= deadline:
                    return best
                p = -1 if symmetric else variable(c[i,0],-1.0,1.0)
                q = variable(c[i,1],-1.0,1.0)
                r = variable(c[i,2],1e-7,0.6)
                assignments.append((i,p,1,q,r))
    w0 = np.max(c[:,0]+c[:,2])-np.min(c[:,0]-c[:,2])
    wi=variable(w0,0.25,1.75)
    nv=len(values)
    X=np.zeros((21,nv)); Y=X.copy(); R=X.copy()
    for i,p,s,q,r in assignments:
        if time.monotonic() >= deadline:
            return best
        if p>=0: X[i,p]=s
        Y[i,q]=1; R[i,r]=1
    ii,jj=np.triu_indices(21,1)
    DX=X[ii]-X[jj]; DY=Y[ii]-Y[jj]; RR=R[ii]+R[jj]
    wall=np.vstack((-X-R,X-R,-Y-R,Y-R))
    wall[:42,wi]+=0.5
    wall[42:,wi]-=0.5
    offset=np.zeros(84); offset[42:]=1
    obj=-np.sum(R,axis=0)
    base=np.array(values)
    saved=base.copy()
    cache_z=None; cache_v=None; cache_j=None
    def check():
        if time.monotonic() >= deadline:
            raise Expired()
    def constraints(z):
        nonlocal cache_z,cache_v,cache_j
        check()
        if cache_z is None or not np.array_equal(z,cache_z):
            dx=DX@z; dy=DY@z
            d=np.maximum(np.sqrt(dx*dx+dy*dy),1e-16)
            cache_v=np.concatenate((d-RR@z,wall@z+offset))
            cache_j=np.vstack(((dx[:,None]*DX+dy[:,None]*DY)/d[:,None]-RR,wall))
            cache_z=z.copy()
        return cache_v
    def jac(z):
        constraints(z)
        return cache_j
    def consider(z):
        nonlocal best,best_score,saved
        check()
        raw=-float(obj@z)
        if raw <= best_score+2e-13:
            return
        a=np.column_stack((X@z,Y@z,R@z))
        if np.any(a[:,2]<=0):
            return
        d=np.sqrt((a[ii,0]-a[jj,0])**2+(a[ii,1]-a[jj,1])**2)
        ratio=float(np.min(d/(a[ii,2]+a[jj,2])))
        a[:,2] *= min(1.0,ratio)*(1-3e-15)
        perimeter=(np.max(a[:,0]+a[:,2])-np.min(a[:,0]-a[:,2])+np.max(a[:,1]+a[:,2])-np.min(a[:,1]-a[:,2]))
        a *= min(1.0,2/perimeter)*(1-3e-15)
        if float(np.sum(a[:,2])) <= best_score:
            return
        candidate={'circles':a.tolist()}
        s=exact_score(candidate)
        if s>best_score:
            best_score=s; best=candidate; saved=z.copy()
    trial=0
    try:
        while time.monotonic()<deadline:
            check()
            if trial==0:
                z=base.copy()
            else:
                z=saved.copy() if trial%3 else base.copy()
                amplitude=params['perturbation']*[0.6,1.0,1.8,2.6][(trial-1)%4]
                position=(np.sum(np.abs(X)+np.abs(Y),axis=0)>0)
                radius=(np.sum(R,axis=0)>0)
                z[position]+=rng.normal(0,amplitude,np.sum(position))
                z[radius]*=np.exp(rng.normal(0,0.12,np.sum(radius)))
                z[wi]+=rng.normal(0,amplitude*0.5)
                z=np.maximum(np.array([b[0] for b in bounds]),np.minimum(z,np.array([b[1] for b in bounds])))
            result=minimize(lambda z: float(obj@z),z,jac=lambda z: obj,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':constraints,'jac':jac},callback=consider,
                options={'maxiter':params['iterations'],'ftol':2e-13,'disp':False})
            consider(result.x)
            trial+=1
    except Expired:
        pass
    return best
