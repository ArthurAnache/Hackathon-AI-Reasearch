import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (85, 20, 160), "repair_trials": (28, 8, 70)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    archive = []

    def exact(data):
        try:
            q = np.asarray(data['circles'], dtype=float)
            if q.shape != (21, 3) or not np.isfinite(q).all() or np.any(q[:, 2] <= 0):
                return -float('inf')
            x = [Fraction(float(v)) for v in q[:, 0]]
            y = [Fraction(float(v)) for v in q[:, 1]]
            r = [Fraction(float(v)) for v in q[:, 2]]
            w = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
            h = max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
            if w+h > 2:
                return -float('inf')
            for i, j in zip(ii, jj):
                if time.monotonic() > deadline + 0.08:
                    return -float('inf')
                d = np.sqrt((q[i,0]-q[j,0])**2+(q[i,1]-q[j,1])**2)
                if d < q[i,2]+q[j,2]:
                    return -float('inf')
            return float(np.sum(q[:,2]))
        except Exception:
            return -float('inf')

    if init is not None:
        best_score = exact(init)

    def save(q):
        nonlocal best, best_score
        if time.monotonic() > deadline + 0.04:
            return
        q = np.asarray(q, dtype=float).copy()
        if not np.isfinite(q).all() or np.min(q[:,2]) <= 0:
            return
        d = np.hypot(q[ii,0]-q[jj,0], q[ii,1]-q[jj,1])
        ratio = min(1.0, float(np.min(d/(q[ii,2]+q[jj,2]))))
        if ratio <= 0:
            return
        q[:,2] *= ratio*(1-3e-13)
        lo = np.min(q[:,:2]-q[:,2,None], axis=0)
        hi = np.max(q[:,:2]+q[:,2,None], axis=0)
        q[:,:2] -= (lo+hi)/2
        q *= (2.0/(hi-lo).sum())*(1-3e-13)
        if q[:,2].sum() <= best_score:
            return
        data = {'circles': q.tolist()}
        score = exact(data)
        if score > best_score:
            best_score, best = score, data

    class Stop(Exception):
        pass

    def model(a, b, c):
        # Three central singles, or another odd central row, and mirrored pairs.
        pairs = a+b
        nvar = 2*c+3*pairs+1
        M = np.zeros((21,3,nvar))
        z = np.zeros(nvar)
        k = 0
        row = 0
        dx = 1.0/max(a,b,c)
        radius = min(dx/2, 0.105)
        for x in np.linspace(-(c-1)*dx/2, (c-1)*dx/2, c):
            if time.monotonic() > deadline:
                raise Stop()
            M[row,0,k] = 1
            M[row,2,k+1] = 1
            z[k:k+2] = [x, radius]
            k += 2
            row += 1
        for count, yy in ((b, 0.195), (a, 0.39)):
            for x in np.linspace(-(count-1)*dx/2, (count-1)*dx/2, count):
                if time.monotonic() > deadline:
                    raise Stop()
                M[row,0,k] = M[row+1,0,k] = 1
                M[row,1,k+1] = 1
                M[row+1,1,k+1] = -1
                M[row,2,k+2] = M[row+1,2,k+2] = 1
                z[k:k+3] = [x, yy, radius]
                k += 3
                row += 2
        z[-1] = 1.04
        return M, z

    def optimize(M, z, until, iterations):
        nv = len(z)
        X, Y, R = M[:,0,:], M[:,1,:], M[:,2,:]
        DX, DY, SR = X[ii]-X[jj], Y[ii]-Y[jj], R[ii]+R[jj]
        walls = np.concatenate((X-R, -X-R, Y-R, -Y-R))
        walls[:42,-1] += 0.5
        walls[42:,-1] -= 0.5
        constants = np.r_[np.zeros(42), np.ones(42)]
        grad = -R.sum(axis=0)
        current = [z.copy()]
        cache = [None, None, None]

        def calc(v):
            if time.monotonic() >= until:
                raise Stop()
            if cache[0] is not None and np.array_equal(v, cache[0]):
                return cache[1], cache[2]
            dx, dy = DX@v, DY@v
            dist = np.sqrt(dx*dx+dy*dy+1e-28)
            val = np.r_[dist-SR@v, walls@v+constants]
            jac = np.concatenate(((dx/dist)[:,None]*DX+(dy/dist)[:,None]*DY-SR, walls), axis=0)
            cache[:] = [v.copy(), val, jac]
            return val, jac

        def callback(v):
            current[0] = v.copy()
            if time.monotonic() >= until:
                raise Stop()

        bounds = []
        for k in range(nv-1):
            if time.monotonic() >= until:
                raise Stop()
            if np.any(R[:,k]):
                bounds.append((0.00005, 0.6))
            else:
                bounds.append((-0.99, 0.99))
        bounds.append((0.25, 1.75))
        try:
            result = minimize(lambda v: float(grad@v), z, jac=lambda v: grad,
                              bounds=bounds, constraints={'type':'ineq','fun':lambda v:calc(v)[0],
                              'jac':lambda v:calc(v)[1]}, method='SLSQP', callback=callback,
                              options={'maxiter':iterations, 'ftol':2e-11, 'disp':False})
            current[0] = result.x
        except Stop:
            pass
        v = current[0]
        q = np.einsum('ijk,k->ij', M, v)
        save(q)
        distances = np.hypot(q[ii,0]-q[jj,0],q[ii,1]-q[jj,1])
        shrink = min(1., float(np.min(distances/(q[ii,2]+q[jj,2]))))
        quality = float(q[:,2].sum()*max(0.,shrink))
        archive.append((quality,M,v.copy()))
        return v

    families = [(4,5,3), (5,4,3), (4,4,5), (3,6,3)]
    try:
        for number, family in enumerate(families):
            if time.monotonic() >= deadline-0.12:
                break
            M,z = model(*family)
            remaining = deadline-time.monotonic()
            allowance = min(0.31, remaining*0.40)
            optimize(M,z,time.monotonic()+allowance,int(params['iterations']))
        if archive and time.monotonic() < deadline-0.06:
            archive.sort(key=lambda t:t[0], reverse=True)
            _, M, z = archive[0]
            z = optimize(M,z,min(deadline-0.035,time.monotonic()+0.27),120)
        if archive and time.monotonic() < deadline-0.10:
            _, M, z = max(archive,key=lambda t:t[0])
            q = np.einsum('ijk,k->ij',M,z)
            # Destroy a mirrored pair near a random surviving circle, then repair in a new void.
            groups = []
            for k in range(len(z)-1):
                if time.monotonic() >= deadline-0.07:
                    break
                ids = np.flatnonzero(M[:,2,k])
                if len(ids) == 2:
                    groups.append((k,ids))
            if groups:
                k,ids = groups[int(rng.integers(len(groups)))]
                survivors = np.ones(21,dtype=bool)
                survivors[ids] = False
                best_void = None
                for trial in range(int(params['repair_trials'])):
                    if time.monotonic() >= deadline-0.07:
                        break
                    x = rng.uniform(-0.42,0.42)*z[-1]
                    y = rng.uniform(0.02,0.43)*(2-z[-1])
                    clearance = min(z[-1]/2-abs(x),(2-z[-1])/2-y,y)
                    clearance = min(clearance,float(np.min(np.hypot(q[survivors,0]-x,q[survivors,1]-y)-q[survivors,2])))
                    if best_void is None or clearance > best_void[0]:
                        best_void = (clearance,x,y)
                if best_void and best_void[0] > 0:
                    v=z.copy()
                    v[k-2:k+1]=[best_void[1],best_void[2],best_void[0]*0.9]
                    optimize(M,v,deadline-0.015,70)
    except Stop:
        pass
    if best is None:
        q = np.array([[0.1*(i%7),0.1*(i//7),0.049] for i in range(21)])
        return {'circles':q.tolist()}
    return best
