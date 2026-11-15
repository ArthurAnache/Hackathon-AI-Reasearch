import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'iterations': (230, 80, 450),
    'tolerance': (8e-12, 1e-13, 1e-8, 'log'),
    'safety': (2e-14, 5e-15, 1e-10, 'log'),
    'perturbation': (0.003, 0.0001, 0.025, 'log'),
    'polish_share': (0.20, 0.10, 0.35)
}
I, J = np.triu_indices(21, 1)

class StopSearch(Exception):
    pass

def valid(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return False
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return False
    x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
    perimeter = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    perimeter += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return perimeter <= 2

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    eps = float(params['safety'])
    best = None
    best_a = None
    best_score = -np.inf

    def offer(a):
        nonlocal best, best_a, best_score
        a = np.asarray(a, dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        ratio = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
        if ratio <= 0:
            return
        a[:,2] *= ratio*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        span = float(np.sum(hi-lo))
        if span <= 0:
            return
        scale = (2-16*eps)/span
        a[:,:2] = (a[:,:2]-(lo+hi)/2)*scale
        a[:,2] *= scale
        score = float(np.sum(a[:,2]))
        if score > best_score and valid(a):
            best_score, best_a = score, a
            best = {'state': a.T.reshape(-1).tolist()}

    try:
        if init is not None:
            original = np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T
            if valid(original):
                best, best_a = init, original.copy()
                best_score = float(np.sum(original[:,2]))
            offer(original)
    except (KeyError, TypeError, ValueError):
        pass
    offer(np.array([[2*(k%3),2*(k//3),1.] for k in range(21)]))

    # Full centered-box model, followed by its reflection quotient.
    wall = np.zeros((84,64))
    k = np.arange(21)
    wall[k,k] = 1
    wall[21+k,k] = -1
    wall[42+k,21+k] = 1
    wall[63+k,21+k] = -1
    wall[np.arange(84),42+np.tile(k,4)] = -1
    wall[:42,63] = .5
    wall[42:,63] = -.5
    off = np.r_[np.zeros(42),np.ones(42)]
    objective = np.r_[np.zeros(42),-np.ones(21),0.]
    ids = np.r_[np.arange(12),np.arange(3,12)]
    T = np.zeros((64,34))
    T[np.arange(21),ids] = 1
    T[21+np.arange(3,12),np.arange(12,21)] = 1
    T[21+np.arange(12,21),np.arange(12,21)] = -1
    T[42+np.arange(21),21+ids] = 1
    T[63,33] = 1
    mirror = np.r_[np.arange(3),np.arange(12,21),np.arange(3,12)]
    selected = []
    for q in range(len(I)):
        if time.monotonic() >= end:
            return best
        p = (int(I[q]),int(J[q]))
        other = tuple(sorted((int(mirror[I[q]]),int(mirror[J[q]]))))
        if p <= other:
            selected.append(q)

    def model(symmetric):
        tr = T if symmetric else np.eye(64)
        sel = np.asarray(selected) if symmetric else np.arange(len(I))
        ii,jj = I[sel],J[sel]
        dx = tr[ii]-tr[jj]
        dy = tr[21+ii]-tr[21+jj]
        rr = tr[42+ii]+tr[42+jj]
        ws = np.r_[np.arange(12),21+np.arange(12),42+np.arange(12),63+np.arange(12)] if symmetric else np.arange(84)
        bounds = [(None,None)]*tr.shape[1]
        for j in (range(21,33) if symmetric else range(42,63)):
            if time.monotonic() >= end:
                break
            bounds[j] = (1e-7,.65)
        if symmetric:
            for j in range(12,21):
                if time.monotonic() >= end:
                    break
                bounds[j] = (0,None)
        bounds[-1] = (.12,1.88)
        return tr,dx,dy,rr,wall[ws]@tr,off[ws],objective@tr,bounds

    sm = model(True)
    fm = model(False)

    def full_state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        b = a.copy()
        b[:,:2] -= (lo+hi)/2
        return np.r_[b.T.reshape(-1),hi[0]-lo[0]]

    def quotient(a):
        u = full_state(a)
        return np.r_[u[:12],u[24:33],u[42:54],u[63]]

    def run(z, mod, stop):
        if time.monotonic() >= stop-.006:
            return z
        tr,dxm,dym,rm,linear,offset,obj,bounds = mod
        cache = [None,None,None]
        last = [z.copy()]
        counter = [0]
        def calc(v):
            if time.monotonic() >= stop:
                raise StopSearch
            if cache[0] is not None and np.array_equal(cache[0],v):
                return cache[1],cache[2]
            dx,dy,rr = dxm@v,dym@v,rm@v
            dist = np.sqrt(dx*dx+dy*dy)
            inv = 1/np.maximum(dist,1e-15)
            c = np.r_[dist-rr,linear@v+offset]
            jac = np.vstack(((dx*inv)[:,None]*dxm+(dy*inv)[:,None]*dym-rm,linear))
            cache[:] = [v.copy(),c,jac]
            last[0] = v.copy()
            return c,jac
        def fun(v):
            if time.monotonic() >= stop:
                raise StopSearch
            return float(obj@v),obj
        def callback(v):
            if time.monotonic() >= stop:
                raise StopSearch
            counter[0] += 1
            if counter[0] % 16 == 0:
                offer((tr@v)[:63].reshape(3,21).T)
        try:
            result = minimize(fun,z,method='SLSQP',jac=True,bounds=bounds,
                constraints={'type':'ineq','fun':lambda v:calc(v)[0],'jac':lambda v:calc(v)[1]},
                callback=callback,options={'maxiter':int(params['iterations']),'ftol':float(params['tolerance']),'disp':False})
            last[0] = result.x
        except StopSearch:
            pass
        offer((tr@last[0])[:63].reshape(3,21).T)
        return last[0]

    if best_score > 2.36:
        run(full_state(best_a),fm,end-.012)
        return best

    # Reorder the first parent's alternating columns into quotient order.
    axis, upper = [], []
    for col,count in enumerate((3,4,5,4,5)):
        if time.monotonic() >= end-.03:
            return best
        for j in range(count):
            if time.monotonic() >= end-.03:
                return best
            y = 2*j-(count-1)
            p = [col*np.sqrt(3.),float(y),1.]
            if y == 0:
                axis.append(p)
            elif y > 0:
                upper.append(p)
    a = np.array(axis+upper+[[x,-y,r] for x,y,r in upper])
    offer(a)
    seeds = [quotient(a)]
    for rows in ([(5,2.,0.),(4,2.+np.sqrt(3.),0.)],
                 [(3,np.sqrt(3.),1.),(3,2*np.sqrt(3.),0.),(3,3*np.sqrt(3.),1.)]):
        if time.monotonic() >= end-.03:
            break
        aa = [[-2.,0.,1.],[0.,0.,1.],[2.,0.,1.]]
        pos = []
        for count,height,shift in rows:
            if time.monotonic() >= end-.03:
                break
            pos.extend([[2*j-(count-1)+shift,height,1.] for j in range(count)])
        if len(pos) == 9:
            aa = np.array(aa+pos+[[x,-y,r] for x,y,r in pos])
            offer(aa)
            z = quotient(aa)
            # Break left/right symmetry without breaking the reflection quotient.
            z[:12] += rng.normal(0,float(params['perturbation']),12)
            seeds.append(z)

    width = end-start
    search_end = end-float(params['polish_share'])*width
    trajectories = []
    for index,z in enumerate(seeds):
        now = time.monotonic()
        if now >= search_end-.02:
            break
        remaining = len(seeds)-index
        cutoff = now+(search_end-now)/remaining
        zz = run(z,sm,cutoff)
        aa = (T@zz)[:63].reshape(3,21).T
        trajectories.append((float(np.sum(aa[:,2])),zz))

    if time.monotonic() < end-.045:
        run(full_state(best_a),fm,end-.018)
    # A completed full solve may leave enough time for a coordinated restart.
    for attempt in range(2):
        if time.monotonic() >= end-.075 or not trajectories:
            break
        z = max(trajectories,key=lambda q:q[0])[1].copy()
        z[:12] += rng.normal(0,0.009,12)
        z[12:21] = np.maximum(0,z[12:21]+rng.normal(0,0.006,9))
        run(z,sm,end-.015)
    return best
