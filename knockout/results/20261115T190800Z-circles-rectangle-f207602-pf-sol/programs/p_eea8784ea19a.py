import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

PARAMS = {
    'symmetric_iterations': (105, 30, 180),
    'free_iterations': (65, 20, 120),
    'safety': (3e-14, 1e-15, 1e-11, 'log')
}

class StopSearch(Exception):
    pass

def canonical(obj):
    if obj is None:
        return None
    if 'circles' in obj:
        return obj
    if 'centers' in obj:
        return {'circles': [[p[0], p[1], r] for p, r in zip(obj['centers'], obj['radii'])]}
    rows = []
    for x, y, r, m in obj['orbits']:
        rows.append([x, y, r])
        if m == 2:
            rows.append([x, -y, r])
    return {'circles': rows}

def exact_score(obj):
    try:
        a = np.asarray(canonical(obj)['circles'], dtype=float)
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        wh = max(t+s for t,s in zip(x,r))-min(t-s for t,s in zip(x,r))
        wh += max(t+s for t,s in zip(y,r))-min(t-s for t,s in zip(y,r))
        if wh > 2:
            return -float('inf')
        i,j = np.triu_indices(21, 1)
        dx = a[i,0]-a[j,0]
        dy = a[i,1]-a[j,1]
        if np.any(np.sqrt(dx*dx+dy*dy) < a[i,2]+a[j,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.69*budget_s
    rng = np.random.default_rng(seed)
    best = init
    best_value = exact_score(init)
    safety = float(params['safety'])

    def tick():
        if time.monotonic() >= deadline:
            raise StopSearch()

    fallback = np.array([[2*(i%7), 2*(i//7), 1.] for i in range(21)], dtype=float)/10.00000000001
    if best_value == -float('inf'):
        best = {'circles': fallback.tolist()}
        best_value = exact_score(best)
    try:
        a = np.asarray(canonical(init)['circles'], dtype=float).copy()
        if a.shape != (21,3) or not np.isfinite(a).all():
            a = fallback.copy()
    except Exception:
        a = fallback.copy()
    lo = np.min(a[:,:2]-a[:,2,None], axis=0)
    hi = np.max(a[:,:2]+a[:,2,None], axis=0)
    a[:,:2] -= (lo+hi)/2

    ii,jj = np.triu_indices(21,1)
    DX = np.zeros((210,64)); DY = DX.copy(); RS = DX.copy()
    ar = np.arange(210)
    DX[ar,ii] = 1.; DX[ar,jj] = -1.
    DY[ar,21+ii] = 1.; DY[ar,21+jj] = -1.
    RS[ar,42+ii] = 1.; RS[ar,42+jj] = 1.
    walls = np.zeros((84,64)); offsets = np.zeros(84)
    ids = np.arange(21)
    walls[ids,ids] = 1.; walls[ids,42+ids] = -1.; walls[ids,63] = .5
    walls[21+ids,ids] = -1.; walls[21+ids,42+ids] = -1.; walls[21+ids,63] = .5
    walls[42+ids,21+ids] = 1.; walls[42+ids,42+ids] = -1.; walls[42+ids,63] = -.5
    walls[63+ids,21+ids] = -1.; walls[63+ids,42+ids] = -1.; walls[63+ids,63] = -.5
    offsets[42:] = 1.
    objective = np.zeros(64); objective[42:63] = -1.

    def retain(u):
        nonlocal best,best_value
        tick()
        c = np.column_stack((u[:21],u[21:42],u[42:63])).copy()
        if not np.isfinite(c).all() or np.any(c[:,2] <= 0):
            return
        dx = c[ii,0]-c[jj,0]; dy = c[ii,1]-c[jj,1]
        ratio = float(np.min(np.sqrt(dx*dx+dy*dy)/(c[ii,2]+c[jj,2])))
        c[:,2] *= min(1.,ratio)*(1.-safety)
        low = np.min(c[:,:2]-c[:,2,None],axis=0)
        high = np.max(c[:,:2]+c[:,2,None],axis=0)
        scale = min(1.,2./float(np.sum(high-low)))*(1.-safety)
        if float(np.sum(c[:,2]))*scale <= best_value:
            return
        c[:,:2] -= (low+high)/2
        c *= scale
        obj = {'centers': c[:,:2].tolist(), 'radii': c[:,2].tolist()}
        val = exact_score(obj)
        if val > best_value:
            best,best_value = obj,val

    def optimize(T,z,bounds,iterations,symmetric):
        tick()
        dxm = DX@T; dym = DY@T; rsm = RS@T
        wm = walls@T; off = offsets.copy()
        if symmetric:
            # Reflection creates duplicate constraints; remove them exactly.
            keys = np.concatenate((np.abs(dxm),np.abs(dym),rsm),axis=1)
            _,keep = np.unique(keys,axis=0,return_index=True)
            dxm,dym,rsm = dxm[keep],dym[keep],rsm[keep]
            _,keepw = np.unique(np.column_stack((wm,off)),axis=0,return_index=True)
            wm,off = wm[keepw],off[keepw]
        grad = objective@T
        def fun(v):
            tick()
            return float(grad@v)
        def jac(v):
            tick()
            return grad
        def cons(v):
            tick()
            x,y = dxm@v,dym@v
            return np.r_[np.sqrt(x*x+y*y)-rsm@v,wm@v+off]
        def cjac(v):
            tick()
            x,y = dxm@v,dym@v
            d = np.maximum(np.sqrt(x*x+y*y),1e-24)
            return np.vstack(((x/d)[:,None]*dxm+(y/d)[:,None]*dym-rsm,wm))
        def callback(v):
            retain(T@v)
        result = minimize(fun,z,jac=jac,bounds=bounds,method='SLSQP',constraints={'type':'ineq','fun':cons,'jac':cjac},callback=callback,options={'maxiter':int(iterations),'ftol':2e-13,'disp':False})
        u = T@result.x
        retain(u)
        return u

    try:
        projections = []
        for swap in (False,True):
            tick()
            b = a.copy()
            if swap:
                b[:,[0,1]] = b[:,[1,0]]
            order = np.argsort(b[:,1])
            lower,singles,upper = order[:9],order[9:12],order[12:]
            ref = b[lower].copy(); ref[:,1] *= -1
            cost = np.sum((b[upper,None,:]-ref[None,:,:])**2,axis=2)
            p,q = linear_sum_assignment(cost)
            g = np.zeros((12,3))
            g[:3] = b[singles]; g[:3,1] = 0.
            g[3:] = (b[upper[p]]+ref[q])/2
            g[3:,1] = np.abs(g[3:,1])
            error = float(np.sum(cost[p,q])+np.sum(b[singles,1]**2))
            projections.append((error,g))
        g = min(projections,key=lambda t:t[0])[1]
        T = np.zeros((64,34))
        k = 0
        for h in range(12):
            tick()
            for sign in ((0,) if h < 3 else (1,-1)):
                tick()
                T[k,h] = 1.
                T[42+k,21+h] = 1.
                if h >= 3:
                    T[21+k,12+h-3] = sign
                k += 1
        T[63,33] = 1.
        bounds = [(-1.,1.)]*12+[(0.,1.)]*9+[(1e-7,.5)]*12+[(.05,1.95)]
        z = np.r_[g[:,0],g[3:,1],np.maximum(g[:,2],1e-5),max(.1,float(np.max(g[:,0]+g[:,2])-np.min(g[:,0]-g[:,2])))]
        u = optimize(T,z,bounds,params['symmetric_iterations'],True)
        # A different, regular basin is preferable to polishing a poor cold projection.
        if best_value < 2.365 and time.monotonic() < start+.43*budget_s:
            tick()
            g2 = np.zeros((12,3))
            g2[:3,0] = [-.22,0.,.22]
            for row in range(3):
                tick()
                sl = slice(3+3*row,6+3*row)
                g2[sl,0] = np.array([-.22,0.,.22])+(.065 if row%2 == 0 else -.045)
                g2[sl,1] = .19*(row+1)
            g2[:,2] = .095
            z2 = np.r_[g2[:,0],g2[3:,1],g2[:,2],.70]
            u2 = optimize(T,z2,bounds,params['symmetric_iterations'],True)
            if objective@u2 < objective@u:
                u = u2
        if time.monotonic() < start+.58*budget_s:
            tick()
            # Remove every orbit equality. The original symmetric packing remains feasible.
            c = np.asarray(canonical(best)['circles'],dtype=float)
            low = np.min(c[:,:2]-c[:,2,None],axis=0)
            high = np.max(c[:,:2]+c[:,2,None],axis=0)
            c[:,:2] -= (low+high)/2
            u = np.r_[c[:,0],c[:,1],c[:,2],high[0]-low[0]]
            free_bounds = [(-1.,1.)]*42+[(1e-7,.5)]*21+[(.05,1.95)]
            optimize(np.eye(64),u,free_bounds,params['free_iterations'],False)
    except (StopSearch,ValueError,FloatingPointError):
        pass
    return best
