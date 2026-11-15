import time
import warnings
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize

PARAMS = {
    'lp_steps': (7, 2, 20),
    'trust': (0.24, 0.08, 0.5),
    'safety': (0.70, 0.5, 0.75),
    'switch_gain': (0.000015, 0.000001, 0.001, 'log')
}
N = 21
D = 64
I, J = np.triu_indices(N, 1)
P = np.arange(len(I))
K = np.arange(N)
M = len(I)

class Deadline(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    r = a[:, 2]
    dist = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(dist < r[I]+r[J]):
        return -np.inf
    x = list(map(Fraction, map(float, a[:, 0])))
    y = list(map(Fraction, map(float, a[:, 1])))
    q = list(map(Fraction, map(float, r)))
    w = max(u+v for u,v in zip(x,q))-min(u-v for u,v in zip(x,q))
    h = max(u+v for u,v in zip(y,q))-min(u-v for u,v in zip(y,q))
    return float(np.sum(r)) if w+h <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    end = start + float(params['safety'])*budget_s
    stop = end
    best_obj = None
    best_a = None
    best_score = -np.inf
    raw = None

    def guard():
        if time.monotonic() >= stop:
            raise Deadline()

    def encode(a):
        return {'coordinate_rows': a.T.tolist()}

    if init is not None:
        try:
            raw = np.asarray(init['circles'] if 'circles' in init else np.asarray(init['coordinate_rows']).T, dtype=float)
            s = exact_score(raw)
            if s > best_score:
                best_obj, best_a, best_score = init, raw.copy(), s
        except (KeyError, ValueError, TypeError, OverflowError):
            raw = None
    k = np.arange(N)
    fallback = np.column_stack(((k % 5 + 0.5)*0.2, (k//5+0.5)*0.2, np.full(N,0.1*(1-1e-12))))
    s = exact_score(fallback)
    if s > best_score:
        best_obj, best_a, best_score = encode(fallback), fallback.copy(), s
    if raw is None or raw.shape != (N,3) or not np.isfinite(raw).all():
        raw = fallback.copy()

    objective = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    B = np.zeros((4*N,D))
    B[K,K] = -1
    B[K,2*N+K] = 1
    B[K,-1] = -0.5
    B[N+K,K] = 1
    B[N+K,2*N+K] = 1
    B[N+K,-1] = -0.5
    B[2*N+K,N+K] = -1
    B[2*N+K,2*N+K] = 1
    B[3*N+K,N+K] = 1
    B[3*N+K,2*N+K] = 1
    B[3*N+K,-1] = 1
    bb = np.r_[np.zeros(3*N), np.full(N,2.0)]
    A = np.zeros((M+4*N,D))
    A[M:] = B
    A[P,2*N+I] = 1
    A[P,2*N+J] = 1
    rhs = np.r_[np.zeros(M),bb]

    def pack(a):
        a = a.copy()
        a[:,2] = np.maximum(np.abs(a[:,2]),1e-7)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        sides = hi-lo
        scale = 2/max(float(sides.sum()),1e-12)
        a[:,0] -= (lo[0]+hi[0])*0.5
        a[:,1] -= lo[1]
        a *= scale
        return np.r_[a[:,0],a[:,1],a[:,2],sides[0]*scale]

    def consider(v):
        nonlocal best_obj, best_a, best_score
        guard()
        c = np.column_stack((v[:N],v[N:2*N],np.maximum(v[2*N:3*N],1e-10)))
        d = np.sqrt((c[I,0]-c[J,0])**2+(c[I,1]-c[J,1])**2)
        f = min(1.0,float(np.min(d/(c[I,2]+c[J,2]))))
        if not np.isfinite(f) or f <= 0:
            return
        c[:,2] *= f*(1-2e-12)
        lo = np.min(c[:,:2]-c[:,2,None],axis=0)
        hi = np.max(c[:,:2]+c[:,2,None],axis=0)
        c[:,:2] -= (lo+hi)*0.5
        c *= (2/float((hi-lo).sum()))*(1-2e-12)
        if float(c[:,2].sum()) <= best_score:
            return
        sc = exact_score(c)
        if sc > best_score:
            best_obj, best_a, best_score = encode(c), c.copy(), sc

    def lattice(counts):
        groups = []
        coords = []
        base = 0
        for row, count in enumerate(counts):
            guard()
            for j in range(count):
                guard()
                coords.append((j-(count-1)*0.5,row*np.sqrt(3)*0.5,0.48))
            for j in range((count+1)//2):
                guard()
                other = count-1-j
                groups.append([base+j] if j == other else [base+j,base+other])
            base += count
        columns = []
        bounds = []
        for group in groups:
            guard()
            if len(group) == 2:
                col = np.zeros(D)
                col[group[0]],col[group[1]] = -1,1
                columns.append(col)
                bounds.append((None,None))
            col = np.zeros(D)
            col[N+np.array(group)] = 1
            columns.append(col)
            bounds.append((None,None))
            col = np.zeros(D)
            col[2*N+np.array(group)] = 1
            columns.append(col)
            bounds.append((1e-7,0.5))
        col = np.zeros(D)
        col[-1] = 1
        columns.append(col)
        bounds.append((0.05,1.95))
        return np.asarray(coords),np.column_stack(columns),bounds

    fullT = np.eye(D)
    fullbounds = [(None,None)]*(2*N)+[(1e-7,0.5)]*N+[(0.05,1.95)]

    def run(a,T,bounds,steps):
        guard()
        inv = T.T/np.sum(T*T,axis=0)[:,None]
        z = inv@pack(a)
        obj = objective@T
        BT = B@T
        consider(T@z)
        lp_end = min(stop,time.monotonic()+0.16*budget_s)
        previous = -float(obj@z)
        for it in range(steps):
            guard()
            if time.monotonic() >= lp_end:
                break
            v = T@z
            dx,dy = v[I]-v[J],v[N+I]-v[N+J]
            dist = np.maximum(np.hypot(dx,dy),1e-15)
            nx,ny = dx/dist,dy/dist
            A[P,I],A[P,J] = -nx,nx
            A[P,N+I],A[P,N+J] = -ny,ny
            localbounds = list(bounds)
            for j,(lo,hi) in enumerate(bounds):
                guard()
                if lo is None and hi is None:
                    localbounds[j] = (z[j]-float(params['trust']),z[j]+float(params['trust']))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = linprog(obj,A_ub=A@T,b_ub=rhs,bounds=localbounds,method='highs',options={'threads':1,'time_limit':max(0.001,lp_end-time.monotonic()),'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            guard()
            if not res.success:
                break
            z = res.x
            consider(T@z)
            current = -float(obj@z)
            if it >= 2 and current-previous < float(params['switch_gain']):
                break
            previous = current

        def fun(q):
            guard()
            return float(obj@q)

        def con(q):
            guard()
            v = T@q
            dx,dy = v[I]-v[J],v[N+I]-v[N+J]
            return np.r_[np.sqrt(dx*dx+dy*dy)-v[2*N+I]-v[2*N+J],bb-BT@q]

        def jac(q):
            guard()
            v = T@q
            dx,dy = v[I]-v[J],v[N+I]-v[N+J]
            dist = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
            out = np.zeros_like(A)
            out[P,I],out[P,J] = dx/dist,-dx/dist
            out[P,N+I],out[P,N+J] = dy/dist,-dy/dist
            out[P,2*N+I],out[P,2*N+J] = -1,-1
            out[M:] = -B
            return out@T

        guard()
        res = minimize(fun,z,jac=lambda q:obj,method='SLSQP',bounds=bounds,constraints={'type':'ineq','fun':con,'jac':jac},callback=lambda q:consider(T@q),options={'maxiter':150,'ftol':2e-12,'disp':False})
        consider(T@res.x)

    fractions = [0.40,0.65,0.83,1.0]
    for stage in range(4):
        if time.monotonic() >= end:
            break
        stop = start+(end-start)*fractions[stage]
        if time.monotonic() >= stop-0.015:
            continue
        try:
            if stage == 0:
                a,T,bounds = lattice([5,4,3,4,5])
                run(a,T,bounds,int(params['lp_steps']))
            elif stage == 1:
                run(raw,fullT,fullbounds,int(params['lp_steps']))
            elif stage == 2:
                a,T,bounds = lattice([4,5,4,5,3])
                run(a,T,bounds,4)
            else:
                run(best_a.copy(),fullT,fullbounds,0)
        except (Deadline,ValueError,FloatingPointError):
            pass
    return best_obj
