import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'screen_fraction': (0.64, 0.35, 0.8),
    'screen_iterations': (48, 20, 100),
    'iterations': (180, 60, 350),
    'safety': (1e-14, 4e-15, 1e-10, 'log')
}
N = 21
I, J = np.triu_indices(N, 1)
K = np.arange(N)

class Timeout(Exception):
    pass

def exact_score(a):
    if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = [Fraction(float(v)) for v in a[:, 0]]
    y = [Fraction(float(v)) for v in a[:, 1]]
    r = [Fraction(float(v)) for v in a[:, 2]]
    p = (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r)) +
         max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))
    return float(np.sum(a[:, 2])) if p <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + .70*budget_s
    eps = float(params['safety'])
    best, best_a, score = None, None, -np.inf

    def offer(a):
        nonlocal best, best_a, score
        a = np.array(a, dtype=float, copy=True)
        if a.shape != (21,3) or not np.isfinite(a).all() or np.any(a[:,2] <= 0):
            return -np.inf
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        f = min(1., float(np.min(d/(a[I,2]+a[J,2]))))
        if f <= 0:
            return -np.inf
        a[:,2] *= f*(1-eps)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = (2-16*eps)/float(np.sum(hi-lo))
        a[:,:2] = (a[:,:2]-(hi+lo)/2)*scale
        a[:,2] *= scale
        value = float(np.sum(a[:,2]))
        if value > score:
            verified = exact_score(a)
            if verified > score:
                score, best_a = verified, a.copy()
                best = {'state':a.T.reshape(-1).tolist()}
        return value

    try:
        if init is not None:
            a = np.asarray(init['circles'],float) if 'circles' in init else np.asarray(init['state'],float).reshape(3,21).T
            s = exact_score(a)
            if np.isfinite(s):
                best, best_a, score = init, a.copy(), s
            offer(a)
    except (KeyError, ValueError, TypeError):
        pass
    offer(np.column_stack((2*(K%5),2*(K//5),np.ones(21))))

    wall = np.zeros((84,64))
    wall[K,K] = -1
    wall[K,42+K] = -1
    wall[K,63] = .5
    wall[21+K,K] = 1
    wall[21+K,42+K] = -1
    wall[21+K,63] = .5
    wall[42+K,21+K] = -1
    wall[42+K,42+K] = -1
    wall[42+K,63] = -.5
    wall[63+K,21+K] = 1
    wall[63+K,42+K] = -1
    wall[63+K,63] = -.5
    offset = np.r_[np.zeros(42),np.ones(42)]
    objective = np.r_[np.zeros(42),-np.ones(21),0.]
    T = np.zeros((64,34))
    ids = np.r_[np.arange(12),np.arange(3,12)]
    T[K,ids] = 1
    T[42+K,21+ids] = 1
    T[21+np.arange(3,12),12+np.arange(9)] = 1
    T[21+np.arange(12,21),12+np.arange(9)] = -1
    T[63,33] = 1
    reflection = np.r_[np.arange(3),np.arange(12,21),np.arange(3,12)]
    reflected_i = np.minimum(reflection[I],reflection[J])
    reflected_j = np.maximum(reflection[I],reflection[J])
    reps = np.flatnonzero((I < reflected_i) | ((I == reflected_i) & (J <= reflected_j)))

    def model(t, reduced):
        ii,jj = (I[reps],J[reps]) if reduced else (I,J)
        ax = t[ii]-t[jj]
        ay = t[21+ii]-t[21+jj]
        ar = t[42+ii]+t[42+jj]
        bw,bo = wall@t,offset
        if reduced:
            _,keep = np.unique(np.column_stack((bw,bo)),axis=0,return_index=True)
            bw,bo = bw[keep],bo[keep]
        return t,ax,ay,ar,bw,bo,t.T@objective

    def state(a):
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        xy = a[:,:2]-(hi+lo)/2
        return np.r_[xy[:,0],xy[:,1],a[:,2],hi[0]-lo[0]]

    def solve(v, stop, m, reduced, iterations):
        t,ax,ay,ar,bw,bo,obj = m
        accepted = v.copy()
        local_q = v.copy()
        local_value = -np.inf
        cached = None
        cv = cj = None
        count = 0
        def capture(q):
            nonlocal local_q,local_value
            a = (t@q)[:63].reshape(3,21).T
            value = offer(a)
            if value > local_value:
                local_value,local_q = value,q.copy()
        def calc(q):
            nonlocal cached,cv,cj
            if time.monotonic() >= stop:
                raise Timeout()
            if cached is not None and np.array_equal(cached,q):
                return cv,cj
            dx,dy = ax@q,ay@q
            d = np.sqrt(dx*dx+dy*dy)
            inv = 1/np.maximum(d,1e-15)
            cv = np.r_[d-ar@q,bw@q+bo]
            cj = np.vstack(((dx*inv)[:,None]*ax+(dy*inv)[:,None]*ay-ar,bw))
            cached = q.copy()
            return cv,cj
        def fun(q):
            if time.monotonic() >= stop:
                raise Timeout()
            return float(obj@q),obj
        def callback(q):
            nonlocal accepted,count
            if time.monotonic() >= stop:
                raise Timeout()
            accepted = q.copy()
            count += 1
            if count%16 == 0:
                capture(q)
        bounds = [(None,None)]*len(v)
        for k in (range(21,33) if reduced else range(42,63)):
            if time.monotonic() >= stop:
                return v,-np.inf
            bounds[k] = (1e-7,.65)
        bounds[-1] = (.12,1.88)
        if reduced:
            for k in range(12,21):
                if time.monotonic() >= stop:
                    return v,-np.inf
                bounds[k] = (0.,1.)
        try:
            res = minimize(fun,v,jac=True,method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':lambda q:calc(q)[0],'jac':lambda q:calc(q)[1]},
                callback=callback,options={'maxiter':iterations,'ftol':8e-13,'disp':False})
            accepted = res.x
        except Timeout:
            pass
        capture(accepted)
        return local_q,local_value

    def lattice(kind):
        if kind == 0:
            xx = np.r_[[-2.,0.,2.],[-4.,-2.,0.,2.,4.],[-3.,-1.,1.,3.]]
            yy = np.r_[np.full(5,2.),np.full(4,2.+np.sqrt(3))]
        elif kind == 1:
            xx = np.r_[[-2.,0.,2.],[-3.,-1.,1.,3.],[-4.,-2.,0.,2.,4.]]
            yy = np.r_[np.full(4,np.sqrt(3)),np.full(5,2*np.sqrt(3))]
        elif kind == 2:
            xx = np.r_[[-2.,0.,2.],[-1.,1.,3.],[-2.,0.,2.],[-1.,1.,3.]]
            yy = np.repeat(np.arange(1,4)*np.sqrt(3),3)
        else:
            xx = np.r_[[-2.65,0.,2.65],[-4.,-2.,0.,2.,4.],[-3.,-1.,1.,3.]]
            yy = np.r_[np.full(5,np.sqrt(3)),np.full(4,2*np.sqrt(3))]
        q = np.r_[xx,yy,np.ones(12),0.]
        a = (T@q)[:63].reshape(3,21).T
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2/np.sum(hi-lo)
        q[:-1] *= scale
        q[-1] = (hi[0]-lo[0])*scale
        return q

    reduced = model(T,True)
    if score < 2.30:
        screen_end = start+(end-start)*float(params['screen_fraction'])
        pool = []
        for k in range(4):
            now = time.monotonic()
            if now >= screen_end-.015:
                break
            q = lattice(k)
            offer((T@q)[:63].reshape(3,21).T)
            stop = now+(screen_end-now)/(4-k)
            q,value = solve(q,stop,reduced,True,int(params['screen_iterations']))
            pool.append((value,q))
        if pool and time.monotonic() < end-.15:
            pool.sort(key=lambda item:item[0],reverse=True)
            solve(pool[0][1],end-.14,reduced,True,int(params['iterations']))
    else:
        z = state(best_a)
        q = (T.T@z)/np.sum(T*T,axis=0)
        if np.max(np.abs(T@q-z)) < 1e-7 and time.monotonic() < end-.15:
            solve(q,end-.14,reduced,True,int(params['iterations']))
    if time.monotonic() < end-.025:
        solve(state(best_a),end-.008,model(np.eye(64),False),False,int(params['iterations']))
    return best
