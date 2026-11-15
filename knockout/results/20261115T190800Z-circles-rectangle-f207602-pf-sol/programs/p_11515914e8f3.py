import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq

PARAMS = {
    'iterations': (85, 30, 150),
    'newton_iterations': (7, 2, 12),
    'active_tolerance': (2e-7, 1e-9, 1e-5, 'log'),
}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def exact(data):
        a = np.asarray(data['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        p = max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r))
        return float(np.sum(a[:, 2])) if p <= 2 else -float('inf')

    if init is not None:
        best_score = exact(init)

    X = np.zeros((21, 34))
    Y = np.zeros((21, 34))
    R = np.zeros((21, 34))
    mirror = np.arange(21)
    for k in range(3):
        check()
        Y[k, 9+k] = 1
        R[k, 21+k] = 1
    for k in range(9):
        check()
        a,b = 3+2*k,4+2*k
        X[a,k], X[b,k] = 1,-1
        Y[a,12+k] = Y[b,12+k] = 1
        R[a,24+k] = R[b,24+k] = 1
        mirror[a],mirror[b] = b,a
    pairs = []
    for a,b in zip(ii,jj):
        check()
        if (int(a),int(b)) <= tuple(sorted((int(mirror[a]),int(mirror[b])))):
            pairs.append((a,b))
    pi,pj = np.asarray(pairs).T
    DX,DY,SR = X[pi]-X[pj],Y[pi]-Y[pj],R[pi]+R[pj]
    walls, constants = [],[]
    for a in [0,1,2]+list(range(3,21,2)):
        check()
        v = -X[a]-R[a]
        v[33] += .5
        walls.append(v); constants.append(0.)
        walls.append(Y[a]-R[a]); constants.append(0.)
        v = -Y[a]-R[a]
        v[33] -= 1
        walls.append(v); constants.append(2.)
    W,wc = np.asarray(walls),np.asarray(constants)
    grad = -R.sum(axis=0)
    bounds = [(0.,1.)]*9+[(0.,2.)]*12+[(1e-7,.7)]*12+[(.15,1.85)]
    best_z = None

    def con(z):
        check()
        dx,dy,rr = DX@z,DY@z,SR@z
        return np.r_[W@z+wc,dx*dx+dy*dy-rr*rr]

    def jac(z):
        check()
        dx,dy,rr = DX@z,DY@z,SR@z
        return np.vstack((W,2*(dx[:,None]*DX+dy[:,None]*DY-rr[:,None]*SR)))

    def retain(z):
        nonlocal best,best_score,best_z
        check()
        a = np.column_stack((X@z,Y@z,R@z))
        if not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = a[ii,:2]-a[jj,:2]
        dist = np.sqrt(d[:,0]**2+d[:,1]**2)
        fac = min(1.,float(np.min(dist/(a[ii,2]+a[jj,2]))))
        if fac <= 0:
            return
        a[:,2] *= fac*(1.-3e-15)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= (lo+hi)/2
        x = [Fraction(float(v)) for v in a[:,0]]
        y = [Fraction(float(v)) for v in a[:,1]]
        r = [Fraction(float(v)) for v in a[:,2]]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if p <= 0:
            return
        a *= float(Fraction(2)/p)*(1.-3e-15)
        if float(np.sum(a[:,2])) <= best_score:
            return
        candidate = {'circles': a.tolist()}
        score = exact(candidate)
        if score > best_score:
            best,best_score,best_z = candidate,score,z.copy()

    def polish(z):
        check()
        active = np.flatnonzero(con(z) < params['active_tolerance'])
        if len(active) < 30:
            return z
        u = z.copy()
        for _ in range(params['newton_iterations']):
            check()
            residual = con(u)[active]
            step = lstsq(jac(u)[active],-residual,lapack_driver='gelsy')[0]
            if np.max(np.abs(step)) > .003:
                break
            u += step
            retain(u)
            if np.max(np.abs(step)) < 3e-16:
                break
        return u

    def optimize(z, iterations):
        last = z.copy()
        calls = 0
        def callback(v):
            nonlocal last,calls
            check()
            last = v.copy()
            calls += 1
            if calls % 8 == 0:
                retain(v)
        result = minimize(lambda v: float(grad@v),z,jac=lambda v: grad,
                          method='SLSQP',bounds=bounds,
                          constraints={'type':'ineq','fun':con,'jac':jac},
                          callback=callback,
                          options={'maxiter':int(iterations),'ftol':8e-15,'disp':False})
        last = result.x
        retain(last)
        return polish(last)

    def warm(data):
        a = np.asarray(data['circles'],dtype=float).copy()
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        mid = (lo+hi)/2
        deviation = np.sort(np.abs(a[:,:2]-mid),axis=0)[:3].sum(axis=0)
        if deviation[1] < deviation[0]:
            a[:,[0,1]] = a[:,[1,0]]
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,0] -= (lo[0]+hi[0])/2
        a[:,1] -= lo[1]
        singles = list(np.argsort(np.abs(a[:,0]))[:3])
        unused = set(range(21))-set(singles)
        z = np.zeros(34)
        for k,i in enumerate(singles):
            check()
            z[9+k],z[21+k] = a[i,1],a[i,2]
        for k in range(9):
            check()
            i = min(unused); unused.remove(i)
            target = a[i].copy(); target[0] *= -1
            j = min(unused,key=lambda j: float(np.sum((a[j]-target)**2)))
            unused.remove(j)
            z[k] = abs(a[i,0]-a[j,0])/2
            z[12+k] = (a[i,1]+a[j,1])/2
            z[24+k] = (a[i,2]+a[j,2])/2
        z[33] = hi[0]-lo[0]
        return z

    def lattice(rows):
        z = np.zeros(34)
        singles,pairs = [],[]
        y,previous = 1.,None
        for count in rows:
            check()
            if previous is not None:
                y += np.sqrt(3.) if count%2 != previous%2 else 2.
            if count%2:
                singles.append(y)
            for x in range(2 if count%2 else 1,count,2):
                check()
                pairs.append((float(x),y))
            previous = count
        width = 2*(max(x for x,y in pairs)+1)
        scale = 2/(width+y+1)
        for k,y in enumerate(singles):
            check()
            z[9+k],z[21+k] = y*scale,scale
        for k,(x,y) in enumerate(pairs):
            check()
            z[k],z[12+k],z[24+k] = x*scale,y*scale,scale
        z[33] = width*scale
        return z

    try:
        if init is not None:
            z = warm(init)
            retain(z)
            z = polish(z)
            z = optimize(z,40)
            retain(z)
        families = [(5,4,3,4,5),(5,4,5,4,3),(3,4,3,4,3,4),(7,6,5,3),(7,7,7)]
        offset = int(rng.integers(2))
        if offset:
            families[0],families[1] = families[1],families[0]
        for rows in families:
            check()
            if deadline-time.monotonic() < .10:
                break
            z = lattice(rows)
            retain(z)
            z = optimize(z,params['iterations'])
            retain(z)
        if best_z is not None and deadline-time.monotonic() > .03:
            polish(best_z)
    except (Stop,ValueError,np.linalg.LinAlgError):
        pass
    if best is None:
        best = {'circles': [[(k%7)*.14,(k//7)*.14,.069] for k in range(21)]}
    return best
