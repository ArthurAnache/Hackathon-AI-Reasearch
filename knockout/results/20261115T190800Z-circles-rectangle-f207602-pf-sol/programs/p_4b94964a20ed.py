import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {"perturbation": (0.025, 0.001, 0.12), "iterations": (240, 50, 500)}


def exact_score(candidate):
    try:
        a = np.asarray(candidate["circles"], dtype=np.float64)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        w = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        h = max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if w+h > 2:
            return -float('inf')
        i,j = np.triu_indices(21,1)
        d = a[i,:2]-a[j,:2]
        if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except (ValueError, TypeError, KeyError, OverflowError, IndexError):
        return -float('inf')


class StopSearch(Exception):
    pass


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_score = exact_score(init) if init is not None else -float('inf')
    ii,jj = np.triu_indices(21,1)

    def check_time():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def retain(a):
        nonlocal best, best_score
        if time.monotonic() >= deadline:
            return
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        dx = a[ii,0]-a[jj,0]
        dy = a[ii,1]-a[jj,1]
        shrink = min(1.0, float(np.min(np.sqrt(dx*dx+dy*dy)/(a[ii,2]+a[jj,2]))))
        if shrink <= 0:
            return
        a[:,2] *= shrink*(1.0-2e-14)
        lower = np.min(a[:,:2]-a[:,2,None], axis=0)
        upper = np.max(a[:,:2]+a[:,2,None], axis=0)
        perimeter = float(np.sum(upper-lower))
        if perimeter <= 0:
            return
        a[:,:2] -= (lower+upper)/2
        a *= (2.0/perimeter)*(1.0-2e-14)
        if float(np.sum(a[:,2])) <= best_score:
            return
        candidate = {"circles": a.tolist()}
        score = exact_score(candidate)
        if score > best_score:
            best_score, best = score, candidate

    fallback = np.column_stack((2.0*(np.arange(21)%7),2.0*(np.arange(21)//7),np.ones(21)))
    retain(fallback)
    groups = []
    for col, count in enumerate((5,4,5,4,3)):
        if time.monotonic() >= deadline:
            return best
        levels = range(0,count,2) if count%2 else range(1,count,2)
        groups.extend((col,level) for level in levels)
    positive = [g for g,(_,level) in enumerate(groups) if level > 0]
    singles = [g for g,(_,level) in enumerate(groups) if level == 0]
    yindex = {g:12+k for k,g in enumerate(positive)}
    gx,sg = [],[]
    for g,(_,level) in enumerate(groups):
        if time.monotonic() >= deadline:
            return best
        gx.append(g)
        sg.append(1 if level else 0)
        if level:
            gx.append(g)
            sg.append(-1)
    gx = np.array(gx, dtype=int)
    sg = np.array(sg, dtype=float)
    X = np.zeros((21,34))
    Y = np.zeros((21,34))
    R = np.zeros((21,34))
    for k,g in enumerate(gx):
        if time.monotonic() >= deadline:
            return best
        X[k,g] = 1
        R[k,21+g] = 1
        if g in yindex:
            Y[k,yindex[g]] = sg[k]
    keep = []
    signatures = set()
    for i,j in zip(ii,jj):
        if time.monotonic() >= deadline:
            return best
        g,h = sorted((int(gx[i]),int(gx[j])))
        sign = int(sg[i]*sg[j]) if sg[i] and sg[j] else 0
        key = (g,h,sign)
        if key not in signatures:
            signatures.add(key)
            keep.append((i,j))
    pi = np.array([p[0] for p in keep])
    pj = np.array([p[1] for p in keep])
    DX = X[pi]-X[pj]
    DY = Y[pi]-Y[pj]
    RS = R[pi]+R[pj]
    walls = np.zeros((36,34))
    offsets = np.zeros(36)
    for g in range(12):
        if time.monotonic() >= deadline:
            return best
        walls[g,g] = 1
        walls[g,21+g] = -1
        walls[12+g,33] = 1
        walls[12+g,g] = -1
        walls[12+g,21+g] = -1
        walls[24+g,33] = -0.5
        walls[24+g,21+g] = -1
        if g in yindex:
            walls[24+g,yindex[g]] = -1
        offsets[24+g] = 1
    weights = np.array([1 if level == 0 else 2 for _,level in groups],dtype=float)
    objective_gradient = np.zeros(34)
    objective_gradient[21:33] = -weights

    def unpack(z):
        return np.column_stack((X@z,Y@z,R@z))

    def objective(z):
        check_time()
        return float(objective_gradient@z)

    def gradient(z):
        check_time()
        return objective_gradient

    def constraints(z):
        check_time()
        dx,dy = DX@z,DY@z
        return np.concatenate((np.sqrt(dx*dx+dy*dy)-RS@z, walls@z+offsets))

    def jacobian(z):
        check_time()
        dx,dy = DX@z,DY@z
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-20)
        return np.vstack(((dx/d)[:,None]*DX+(dy/d)[:,None]*DY-RS,walls))

    def callback(z):
        check_time()
        retain(unpack(z))

    radius = 2.0/(12.0+8.0*np.sqrt(3.0))
    z0 = np.zeros(34)
    for g,(col,level) in enumerate(groups):
        if time.monotonic() >= deadline:
            return best
        z0[g] = radius*(1+np.sqrt(3.0)*col)
        z0[21+g] = radius
        if g in yindex:
            z0[yindex[g]] = level*radius
    z0[33] = radius*(2+4*np.sqrt(3.0))
    retain(unpack(z0))

    # Project the supplied packing onto the same reflection quotient.
    # Test both coordinate axes so canonical rotation is immaterial.
    current = z0.copy()
    projection_error = float('inf')
    try:
        original = np.asarray(init['circles'],dtype=float)
        if original.shape == (21,3) and np.isfinite(original).all() and np.all(original[:,2]>0):
            for swap in (False,True):
                check_time()
                a = original.copy()
                if swap:
                    a[:,:2] = a[:,[1,0]]
                lower = np.min(a[:,:2]-a[:,2,None],axis=0)
                upper = np.max(a[:,:2]+a[:,2,None],axis=0)
                scale = 2.0/float(np.sum(upper-lower))
                a[:,0] -= lower[0]
                a[:,1] -= (lower[1]+upper[1])/2
                a *= scale
                axis = np.argsort(np.abs(a[:,1]))[:3]
                rest = np.setdiff1d(np.arange(21),axis)
                rest = rest[np.argsort(a[rest,1])]
                neg,pos = rest[:9],rest[9:]
                cost = ((a[neg,0,None]-a[pos,0][None,:])**2
                        +(a[neg,1,None]+a[pos,1][None,:])**2
                        +2*(a[neg,2,None]-a[pos,2][None,:])**2)
                rows,cols = linear_sum_assignment(cost)
                err = float(np.sum(cost[rows,cols])+2*np.sum(a[axis,1]**2))
                if err >= projection_error:
                    continue
                z = np.zeros(34)
                for g,k in zip(singles,axis[np.argsort(a[axis,0])]):
                    check_time()
                    z[g],z[21+g] = a[k,0],a[k,2]
                for g,u,v in zip(positive,neg[rows],pos[cols]):
                    check_time()
                    z[g] = (a[u,0]+a[v,0])/2
                    z[21+g] = (a[u,2]+a[v,2])/2
                    z[yindex[g]] = abs(a[v,1]-a[u,1])/2
                z[33] = (upper[0]-lower[0])*scale
                current,projection_error = z,err
    except StopSearch:
        return best
    except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError):
        pass

    bounds = [(0.0,2.0)]*21+[(1e-7,0.5)]*12+[(0.2,1.8)]
    attempt = 0
    used_projection = np.isfinite(projection_error)
    while time.monotonic() < deadline:
        try:
            result = minimize(objective,current,jac=gradient,method='SLSQP',
                bounds=bounds,
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                callback=callback,
                options={'maxiter':int(params['iterations']), 'ftol':2e-12, 'disp':False})
            retain(unpack(result.x))
            current = result.x.copy()
            attempt += 1
            check_time()
            if used_projection and attempt == 1:
                current = z0.copy()
                continue
            amplitude = float(params['perturbation'])*(1+0.3*(attempt%3))
            current[:21] += rng.normal(0,amplitude,21)
            current[:21] = np.clip(current[:21],1e-5,1.9)
            current[21:33] *= np.exp(rng.normal(0,amplitude,12))
        except StopSearch:
            break
        except (ValueError, FloatingPointError):
            break
    return best
