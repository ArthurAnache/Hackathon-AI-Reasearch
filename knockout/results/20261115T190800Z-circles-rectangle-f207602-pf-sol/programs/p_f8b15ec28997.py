import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (105, 40, 180), "margin": (2e-13, 1e-14, 1e-10, "log")}

class TimeLimit(Exception):
    pass

def exact_score(data):
    try:
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        if max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))+max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)) > 2:
            return -float('inf')
        i,j = np.triu_indices(21, 1)
        d = np.sqrt((a[i,0]-a[j,0])**2+(a[i,1]-a[j,1])**2)
        if np.any(d < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    best = init
    best_score = exact_score(init)
    ii,jj = np.triu_indices(21,1)
    margin = params['margin']

    def check():
        if time.monotonic() >= deadline:
            raise TimeLimit()

    def retain(a):
        nonlocal best, best_score
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        a = a.copy()
        d = np.sqrt((a[ii,0]-a[jj,0])**2+(a[ii,1]-a[jj,1])**2)
        factor = min(1.0, float(np.min(d/(a[ii,2]+a[jj,2]))))
        a[:,2] *= factor * (1-margin)
        if np.min(a[:,2]) <= 0:
            return
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        perimeter = float(np.sum(hi-lo))
        if perimeter <= 0:
            return
        a[:,:2] -= (lo+hi)/2
        a *= (2/perimeter)*(1-margin)
        score = float(np.sum(a[:,2]))
        if score <= best_score:
            return
        result = {'circles': a.tolist()}
        score = exact_score(result)
        if score > best_score:
            best, best_score = result, score

    # An unconditional feasible fallback.
    fallback = np.array([[0.2*x,0.2*y,0.1] for y in range(3) for x in range(7)])
    retain(fallback)

    def model(rows):
        check()
        # Each row is invariant under reflection in x=0.
        specs = []
        mirror = []
        reps = []
        initial = []
        k = 0
        maxrow = max(rows)
        raw_h = 2 + (len(rows)-1)*np.sqrt(3)
        radius = 2/(2*maxrow+raw_h)
        width = 2*maxrow*radius
        for row,n in enumerate(rows):
            check()
            yy = radius*(1+row*np.sqrt(3))
            if n%2:
                idx = len(specs)
                specs.append((-1,k,k+1,0.0))
                mirror.append(idx)
                reps.append(idx)
                initial.extend([yy,radius])
                k += 2
            for q in range(n//2):
                check()
                u = radius*(2*q+(2 if n%2 else 1))
                idx = len(specs)
                specs.extend([(k,k+1,k+2,1.0),(k,k+1,k+2,-1.0)])
                mirror.extend([idx+1,idx])
                reps.append(idx)
                initial.extend([u,yy,radius])
                k += 3
        nv = k+1
        initial.append(width)
        bx = np.zeros((21,nv)); by = bx.copy(); br = bx.copy()
        for p,(xi,yi,ri,sgn) in enumerate(specs):
            check()
            if xi >= 0:
                bx[p,xi] = sgn
            by[p,yi] = 1
            br[p,ri] = 1
        pairs = []
        for i,j in zip(ii,jj):
            check()
            reflected = tuple(sorted((mirror[i],mirror[j])))
            if (int(i),int(j)) <= reflected:
                pairs.append((i,j))
        pi = np.array([v[0] for v in pairs]); pj = np.array([v[1] for v in pairs])
        dx = bx[pi]-bx[pj]
        dy = by[pi]-by[pj]
        rr = br[pi]+br[pj]
        linear = []
        offset = []
        for p in reps:
            check()
            # Left/right walls, lower wall, upper wall.
            v = -bx[p]-br[p]; v[-1] += 0.5
            linear.append(v); offset.append(0.)
            linear.append(by[p]-br[p]); offset.append(0.)
            v = -by[p]-br[p]; v[-1] -= 1
            linear.append(v); offset.append(2.)
            if specs[p][0] >= 0:
                linear.append(bx[p]); offset.append(0.)
        linear = np.array(linear); offset = np.array(offset)
        grad = -np.sum(br,axis=0)
        cache_z = None
        cache_c = None
        cache_j = None
        def calculate(z):
            nonlocal cache_z,cache_c,cache_j
            check()
            if cache_z is not None and np.array_equal(z,cache_z):
                return
            xx = dx@z; yy = dy@z
            dist = np.sqrt(xx*xx+yy*yy)
            inv = 1/np.maximum(dist,1e-15)
            cache_c = np.concatenate((dist-rr@z,linear@z+offset))
            cache_j = np.vstack(((xx*inv)[:,None]*dx+(yy*inv)[:,None]*dy-rr,linear))
            cache_z = z.copy()
        def fun(z):
            check()
            return float(grad@z)
        def con(z):
            calculate(z)
            return cache_c
        def jac(z):
            calculate(z)
            return cache_j
        def unpack(z):
            return np.column_stack((bx@z,by@z,br@z))
        counter = [0]
        last = [np.array(initial)]
        def callback(z):
            check()
            last[0] = z.copy()
            counter[0] += 1
            if counter[0]%4 == 0:
                retain(unpack(z))
        bounds = []
        for t in range(nv):
            check()
            if np.any(br[:,t]):
                bounds.append((1e-6,0.5))
            elif t == nv-1:
                bounds.append((0.3,1.7))
            else:
                bounds.append((0.,2.))
        return np.array(initial),fun,grad,con,jac,callback,unpack,bounds,last

    families = [(4,5,4,5,3),(5,4,5,4,3),(5,4,3,4,5),(5,6,5,5),(7,7,7)]
    # The first two families have different relative row offsets despite similar envelopes.
    try:
        for rows in families:
            check()
            z,fun,grad,con,jac,callback,unpack,bounds,last = model(rows)
            retain(unpack(z))
            remaining = deadline-time.monotonic()
            if remaining < 0.05:
                break
            try:
                result = minimize(fun,z,jac=lambda z,g=grad:g,method='SLSQP',bounds=bounds,
                    constraints={'type':'ineq','fun':con,'jac':jac},callback=callback,
                    options={'maxiter':int(params['iterations']),'ftol':2e-11,'disp':False})
                retain(unpack(result.x))
            except TimeLimit:
                retain(unpack(last[0]))
                break
    except TimeLimit:
        pass
    return best
