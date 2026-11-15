import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (100, 40, 180), "jitter": (0.004, 0.0, 0.03)}

class StopSearch(Exception):
    pass

def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.70 * budget_s
    search_end = started + 0.57 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')

    def check_time():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def exact_score(data):
        try:
            c = np.asarray(data['circles'], dtype=np.float64)
            if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
                return -float('inf')
            d = np.sqrt((c[ii, 0]-c[jj, 0])**2 + (c[ii, 1]-c[jj, 1])**2)
            if np.any(d < c[ii, 2]+c[jj, 2]):
                return -float('inf')
            x = list(map(lambda v: Fraction(float(v)), c[:, 0]))
            y = list(map(lambda v: Fraction(float(v)), c[:, 1]))
            r = list(map(lambda v: Fraction(float(v)), c[:, 2]))
            w = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
            h = max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
            if w+h > 2:
                return -float('inf')
            return float(np.sum(c[:, 2]))
        except Exception:
            return -float('inf')

    if init is not None:
        best_score = exact_score(init)

    def retain(c):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        c = np.array(c, dtype=float, copy=True)
        if c.shape != (21,3) or not np.isfinite(c).all() or np.any(c[:,2] <= 0):
            return
        d = np.sqrt((c[ii,0]-c[jj,0])**2+(c[ii,1]-c[jj,1])**2)
        factor = min(1.0, float(np.min(d/(c[ii,2]+c[jj,2]))))
        if factor <= 0:
            return
        c[:,2] *= factor*(1.0-8e-15)
        c[:,:2] -= np.mean(c[:,:2], axis=0)
        extent = np.ptp(np.array([c[:,0]-c[:,2], c[:,0]+c[:,2]]))
        extent += np.ptp(np.array([c[:,1]-c[:,2], c[:,1]+c[:,2]]))
        c *= (2.0/extent)*(1.0-8e-15)
        if float(np.sum(c[:,2])) <= best_score:
            return
        for _ in range(3):
            if time.monotonic() >= deadline:
                return
            data = {'circles': c.tolist()}
            value = exact_score(data)
            if value > best_score:
                best, best_score = data, value
                return
            c[:,2] *= 1.0-2e-14

    class Model:
        def __init__(self, rows, jitter):
            check_time()
            singles = []
            pairs = []
            for row, count in enumerate(rows):
                check_time()
                yy = 0.5 + row*np.sqrt(3.0)/2
                if count % 2:
                    singles.append((0.0, yy, 0.5))
                    for k in range(1, (count+1)//2):
                        check_time()
                        pairs.append((float(k), yy, 0.5))
                else:
                    for k in range(count//2):
                        check_time()
                        pairs.append((k+0.5, yy, 0.5))
            self.ns = len(singles)
            self.m = len(singles)+len(pairs)
            m = self.m
            nv = 2*len(singles)+3*len(pairs)+1
            self.nv = nv
            self.X = np.zeros((m,nv))
            self.Y = np.zeros((m,nv))
            self.R = np.zeros((m,nv))
            z = np.zeros(nv)
            raw_w = float(max(rows))
            raw_h = 1+(len(rows)-1)*np.sqrt(3.0)/2
            scale = 2/(raw_w+raw_h)
            p = 0
            for k, (x,y,r) in enumerate(singles+pairs):
                check_time()
                if k >= self.ns:
                    self.X[k,p] = 1
                    z[p] = x*scale
                    p += 1
                self.Y[k,p] = 1
                z[p] = y*scale
                p += 1
                self.R[k,p] = 1
                z[p] = r*scale
                p += 1
            z[-1] = raw_w/(raw_w+raw_h)
            if jitter:
                noise = rng.normal(0,jitter*scale,size=nv)
                noise[-1] = 0
                z += noise
            self.z0 = z
            w = np.zeros(nv)
            w[-1] = 1
            self.L = np.concatenate((w-self.X-self.R, self.Y-self.R, -2*w-self.Y-self.R, self.X[self.ns:]-self.R[self.ns:]),axis=0)
            self.b = np.concatenate((np.zeros(2*m),np.full(m,2.0),np.zeros(m-self.ns)))
            a,b = np.triu_indices(m,1)
            ap,bp = np.triu_indices(m-self.ns,1)
            ap += self.ns
            bp += self.ns
            self.DX = np.concatenate((self.X[a]-self.X[b],self.X[ap]+self.X[bp]),axis=0)
            self.DY = np.concatenate((self.Y[a]-self.Y[b],self.Y[ap]-self.Y[bp]),axis=0)
            self.RS = np.concatenate((self.R[a]+self.R[b],self.R[ap]+self.R[bp]),axis=0)
            weights = np.full(m,2.0)
            weights[:self.ns] = 1.0
            self.obj = -(weights @ self.R)
            self.old = None
            self.latest = z.copy()
            self.quality = -float('inf')
            self.best_z = z.copy()
            self.bounds = [(None,None)]*nv
            for k in range(m):
                check_time()
                idx = int(np.argmax(self.R[k]))
                self.bounds[idx] = (1e-7,1.0)
            self.bounds[-1] = (0.03,0.97)

        def circles(self,z):
            x,y,r = self.X@z,self.Y@z,self.R@z
            positive = np.column_stack((x,y,r))
            reflected = positive[self.ns:].copy()
            reflected[:,0] *= -1
            return np.concatenate((positive,reflected),axis=0)

        def calculate(self,z):
            check_time()
            if self.old is not None and np.array_equal(self.old,z):
                return
            self.old = z.copy()
            dx,dy = self.DX@z,self.DY@z
            dist = np.sqrt(dx*dx+dy*dy)
            safe = np.maximum(dist,1e-15)
            self.cv = np.concatenate((self.L@z+self.b,dist-self.RS@z))
            self.cj = np.concatenate((self.L,(dx/safe)[:,None]*self.DX+(dy/safe)[:,None]*self.DY-self.RS),axis=0)

        def con(self,z):
            self.calculate(z)
            return self.cv

        def jac(self,z):
            self.calculate(z)
            return self.cj

        def fun(self,z):
            check_time()
            return float(self.obj@z)

        def grad(self,z):
            check_time()
            return self.obj

        def callback(self,z):
            check_time()
            self.latest = z.copy()
            c = self.circles(z)
            if np.any(c[:,2] <= 0):
                return
            dist = np.sqrt((c[ii,0]-c[jj,0])**2+(c[ii,1]-c[jj,1])**2)
            f = min(1.0,float(np.min(dist/(c[ii,2]+c[jj,2]))))
            r = c[:,2]*f
            ext = (np.max(c[:,0]+r)-np.min(c[:,0]-r)+np.max(c[:,1]+r)-np.min(c[:,1]-r))
            quality = float(np.sum(r))*2/ext if ext > 0 else 0
            if quality > self.quality:
                self.quality = quality
                self.best_z = z.copy()
            if quality > best_score+2e-13:
                retain(c)

        def solve(self,z,iters,tol):
            self.old = None
            try:
                result = minimize(self.fun,z,jac=self.grad,method='SLSQP',bounds=self.bounds,
                    constraints={'type':'ineq','fun':self.con,'jac':self.jac},callback=self.callback,
                    options={'maxiter':iters,'ftol':tol,'disp':False})
                self.callback(result.x)
                return result.x
            except StopSearch:
                return self.best_z.copy()

    models = []
    try:
        families = [(4,5,4,5,3),(5,4,3,4,5),(4,5,3,5,4),(4,4,5,4,4)]
        for k,rows in enumerate(families):
            check_time()
            if k and time.monotonic() >= search_end:
                break
            model = Model(rows,0.0 if k == 0 else params['jitter'])
            models.append(model)
            retain(model.circles(model.z0))
            model.solve(model.z0,int(params['iterations']),2e-12)
            retain(model.circles(model.best_z))
            if model.quality > 2.3658:
                break
        if models and time.monotonic() < deadline:
            model = max(models,key=lambda m:m.quality)
            z = model.best_z.copy()
            model.solve(z,100,1e-14)
            retain(model.circles(model.best_z))
    except StopSearch:
        pass
    return best
