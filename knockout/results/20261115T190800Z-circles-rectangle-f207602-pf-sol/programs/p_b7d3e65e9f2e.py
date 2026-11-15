import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "iterations": (110, 40, 240),
    "family_seconds": (0.19, 0.05, 0.4),
    "safety": (2e-13, 1e-14, 1e-10, "log"),
}

class StopSearch(Exception):
    pass


def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=np.float64)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        xs = [Fraction(float(v)) for v in c[:, 0]]
        ys = [Fraction(float(v)) for v in c[:, 1]]
        rs = [Fraction(float(v)) for v in c[:, 2]]
        w = max(x+r for x,r in zip(xs,rs))-min(x-r for x,r in zip(xs,rs))
        h = max(y+r for y,r in zip(ys,rs))-min(y-r for y,r in zip(ys,rs))
        if w+h > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        d = np.sqrt((c[i,0]-c[j,0])**2+(c[i,1]-c[j,1])**2)
        if np.any(d < c[i,2]+c[j,2]):
            return -float('inf')
        return float(np.sum(c[:,2]))
    except Exception:
        return -float('inf')


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    safety = float(params['safety'])
    ii,jj = np.triu_indices(21,1)

    def consider(c):
        nonlocal best,best_score
        if time.monotonic() >= deadline:
            return
        c = np.asarray(c,dtype=float).copy()
        if c.shape != (21,3) or not np.isfinite(c).all() or np.any(c[:,2]<=0):
            return
        c[:,:2] -= np.mean(c[:,:2],axis=0)
        ds = np.sqrt((c[ii,0]-c[jj,0])**2+(c[ii,1]-c[jj,1])**2)
        factor = min(1.0,float(np.min(ds/(c[ii,2]+c[jj,2]))))
        if factor <= 0:
            return
        c[:,2] *= factor*(1-safety)
        extent = np.max(c[:,:2]+c[:,2,None],axis=0)-np.min(c[:,:2]-c[:,2,None],axis=0)
        c *= (2.0/float(np.sum(extent)))*(1-safety)
        if float(np.sum(c[:,2])) <= best_score:
            return
        obj = {'circles':c.tolist()}
        sc = exact_score(obj)
        if sc > best_score:
            best,best_score = obj,sc

    # A guaranteed, nondegenerate fallback, independent of the supplied start.
    fallback = np.array([[2*x,2*y,1.] for y in range(3) for x in range(7)],float)
    consider(fallback)

    class Family:
        def __init__(self, rows):
            self.rows = rows
            group_rows=[]
            offsets=[]
            paired=[]
            for row,count in enumerate(rows):
                if time.monotonic() >= deadline:
                    raise StopSearch
                if count%2:
                    group_rows.append(row); offsets.append(0.); paired.append(False)
                for k in range(count//2):
                    group_rows.append(row)
                    offsets.append(float(2*k+1+(count%2)))
                    paired.append(True)
            self.m = len(paired)
            self.p = sum(paired)
            self.nv = self.p+2*self.m+1
            self.ys = slice(self.p,self.p+self.m)
            self.rs = slice(self.p+self.m,self.p+2*self.m)
            self.paired = paired
            self.slot = {}
            t=0
            for a,flag in enumerate(paired):
                if flag:
                    self.slot[a]=t;t+=1
            self.weight=np.array([2. if b else 1. for b in paired])
            self.grad=np.zeros(self.nv)
            self.grad[self.rs]=-self.weight
            dx=[];dy=[];rr=[]
            for a in range(self.m):
                if time.monotonic() >= deadline:
                    raise StopSearch
                for b in range(a+1,self.m):
                    signs=(1.,-1.) if paired[a] and paired[b] else (1.,)
                    for sign in signs:
                        x=np.zeros(self.p)
                        if paired[a]: x[self.slot[a]]=1.
                        if paired[b]: x[self.slot[b]]=-sign
                        y=np.zeros(self.m);y[a]=1.;y[b]=-1.
                        r=np.zeros(self.m);r[a]=1.;r[b]=1.
                        dx.append(x);dy.append(y);rr.append(r)
            self.dx=np.array(dx);self.dy=np.array(dy);self.rr=np.array(rr)
            self.q=len(dx)
            lin=[];off=[]
            for a in range(self.m):
                if time.monotonic() >= deadline:
                    raise StopSearch
                # Right wall, and the contact with the reflected copy.
                z=np.zeros(self.nv);z[-1]=.5;z[self.p+self.m+a]=-1.
                if paired[a]: z[self.slot[a]]=-1.
                lin.append(z);off.append(0.)
                if paired[a]:
                    z=np.zeros(self.nv);z[self.slot[a]]=1.;z[self.p+self.m+a]=-1.
                    lin.append(z);off.append(0.)
                z=np.zeros(self.nv);z[self.p+a]=1.;z[self.p+self.m+a]=-1.
                lin.append(z);off.append(0.)
                z=np.zeros(self.nv);z[self.p+a]=-1.;z[self.p+self.m+a]=-1.;z[-1]=-1.
                lin.append(z);off.append(2.)
            self.lin=np.array(lin);self.off=np.array(off)
            self.cache_x=None
            scale=2./(2*max(rows)+2+np.sqrt(3.)*(len(rows)-1))
            z=np.zeros(self.nv)
            for a in range(self.m):
                if paired[a]:z[self.slot[a]]=offsets[a]*scale
                z[self.p+a]=(1+np.sqrt(3.)*group_rows[a])*scale
                z[self.p+self.m+a]=scale
            z[-1]=2*max(rows)*scale
            self.initial=z
            self.bounds=[(1e-7,1.)]*self.p+[(0.,2.)]*self.m+[(1e-7,.5)]*self.m+[(.15,1.85)]
            self.local_deadline=deadline
            self.last=z.copy()
            self.calls=0

        def values(self,z):
            if time.monotonic() >= self.local_deadline:
                raise StopSearch
            if self.cache_x is not None and np.array_equal(z,self.cache_x):
                return self.cache_c,self.cache_j
            x=self.dx@z[:self.p]
            y=self.dy@z[self.ys]
            d=np.sqrt(x*x+y*y)
            d=np.maximum(d,1e-16)
            c=np.concatenate((d-self.rr@z[self.rs],self.lin@z+self.off))
            jac=np.empty((len(c),self.nv))
            jac[:self.q,:self.p]=self.dx*(x/d)[:,None]
            jac[:self.q,self.ys]=self.dy*(y/d)[:,None]
            jac[:self.q,self.rs]=-self.rr
            jac[:self.q,-1]=0.
            jac[self.q:]=self.lin
            self.cache_x=z.copy();self.cache_c=c;self.cache_j=jac
            return c,jac

        def unpack(self,z):
            out=[]
            for a in range(self.m):
                x=z[self.slot[a]] if self.paired[a] else 0.
                y=z[self.p+a];r=z[self.p+self.m+a]
                out.append([x,y,r])
                if self.paired[a]:out.append([-x,y,r])
            return np.array(out)

        def callback(self,z):
            if time.monotonic() >= self.local_deadline:
                raise StopSearch
            self.last=z.copy()
            self.calls+=1
            if self.calls%6==0:
                consider(self.unpack(z))

        def run(self,z,seconds,iterations):
            self.local_deadline=min(deadline,time.monotonic()+seconds)
            self.last=z.copy();self.cache_x=None
            try:
                result=minimize(lambda v:float(self.grad@v),z,jac=lambda v:self.grad,
                    method='SLSQP',bounds=self.bounds,
                    constraints={'type':'ineq','fun':lambda v:self.values(v)[0],
                                 'jac':lambda v:self.values(v)[1]},
                    callback=self.callback,
                    options={'maxiter':iterations,'ftol':2e-12,'disp':False})
                self.last=result.x.copy()
            except StopSearch:
                pass
            consider(self.unpack(self.last))
            return self.last.copy()

    families=[(4,5,3,5,4),(5,4,3,4,5),(3,4,5,4,5),
              (4,5,4,5,3),(4,4,5,4,4),(3,5,5,5,3)]
    saved=[]
    try:
        for rows in families:
            if time.monotonic() >= deadline-.20:
                break
            fam=Family(rows)
            z=fam.run(fam.initial,min(float(params['family_seconds']),max(.03,(deadline-time.monotonic()-.18)/2)),int(params['iterations']))
            c=fam.unpack(z)
            ds=np.sqrt((c[ii,0]-c[jj,0])**2+(c[ii,1]-c[jj,1])**2)
            rat=min(1.,float(np.min(ds/(c[ii,2]+c[jj,2]))))
            extent=np.max(c[:,:2]+c[:,2,None],axis=0)-np.min(c[:,:2]-c[:,2,None],axis=0)
            estimate=float(np.sum(c[:,2]))*rat*2/max(1e-12,float(np.sum(extent)))
            saved.append((estimate,fam,z))
        saved.sort(key=lambda a:a[0],reverse=True)
        for _,fam,z in saved[:2]:
            if time.monotonic() >= deadline-.015:
                break
            remaining=deadline-time.monotonic()-.008
            fam.run(z,remaining if len(saved)==1 else remaining*.8,180)
    except StopSearch:
        pass
    return best
