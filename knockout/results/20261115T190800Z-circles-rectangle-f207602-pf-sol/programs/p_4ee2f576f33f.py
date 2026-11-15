import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'screen_iterations': (48, 20, 90), 'polish_iterations': (100, 40, 180), 'perturbation': (0.018, 0.001, 0.06)}


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    saved = []

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def array(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((np.asarray(obj['centers']), np.asarray(obj['radii'])))

    def perimeter(a):
        x, y, r = [[Fraction(float(v)) for v in a[:, k]] for k in range(3)]
        return (max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
                + max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r)))

    def exact(a):
        if a.shape != (21,3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return -float('inf')
        if perimeter(a) > 2:
            return -float('inf')
        d = a[ii,:2]-a[jj,:2]
        if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < a[ii,2]+a[jj,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))

    if init is not None:
        try:
            best_score = exact(array(init))
        except (ValueError, TypeError, KeyError):
            pass

    def retain(a):
        nonlocal best, best_score
        a = a.copy()
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return -float('inf')
        a[:,:2] -= np.mean(a[:,:2],axis=0)
        d = a[ii,:2]-a[jj,:2]
        dist = np.sqrt(d[:,0]**2+d[:,1]**2)
        f = min(1.,float(np.min(dist/(a[ii,2]+a[jj,2]))))
        if f <= 0:
            return -float('inf')
        a[:,2] *= f*(1-4e-14)
        p = perimeter(a)
        if p <= 0:
            return -float('inf')
        a *= float(Fraction(2)/p)*(1-4e-14)
        score = exact(a)
        if score > best_score:
            best_score = score
            best = {'centers': a[:,:2].tolist(), 'radii': a[:,2].tolist()}
        return score

    def model(counts):
        points = []
        mirrors = []
        groups = []
        for level, count in enumerate(counts):
            check()
            for col in range(count):
                check()
                x = (col-(count-1)/2)*.20
                if level == 0 and count == 3:
                    x *= 1.45
                i = len(points)
                if level == 0:
                    points.append([x,0.,.095])
                    mirrors.append(i)
                    groups.append((i,))
                else:
                    points.extend(([x,level*.177,.095],[x,-level*.177,.095]))
                    mirrors.extend((i+1,i))
                    groups.append((i,i+1))
        a = np.asarray(points)
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        scale = 2/np.sum(hi-lo)
        a *= scale
        a[:,0] -= lo[0]*scale
        n = sum(2 if len(g)==1 else 3 for g in groups)+1
        X = np.zeros((21,n)); Y = X.copy(); R = X.copy()
        z = np.zeros(n)
        c = 0
        for g in groups:
            check()
            if len(g)==1:
                i = g[0]
                X[i,c] = 1; R[i,c+1] = 1
                z[c:c+2] = a[i,[0,2]]
                c += 2
            else:
                i,j = g
                X[i,c] = X[j,c] = 1
                Y[i,c+1] = 1; Y[j,c+1] = -1
                R[i,c+2] = R[j,c+2] = 1
                z[c:c+3] = a[i]
                c += 3
        z[-1] = (hi[0]-lo[0])*scale
        selected = []
        for i,j in zip(ii,jj):
            check()
            reflected = tuple(sorted((mirrors[i],mirrors[j])))
            if (i,j) <= reflected:
                selected.append((i,j))
        p,q = np.asarray(selected).T
        DX = X[p]-X[q]; DY = Y[p]-Y[q]; SR = R[p]+R[q]
        W = np.zeros(n); W[-1] = 1
        L = np.vstack((X-R, W-X-R, Y-R-W/2, -Y-R-W/2))
        b = np.r_[np.zeros(42), np.ones(42)]
        # Exact duplicate linear rows occur on the symmetry axis and in mirror pairs.
        unique = np.unique(np.column_stack((L,b)),axis=0)
        L = unique[:,:n]; b = unique[:,n]
        grad = -np.sum(R,axis=0)
        radcols = np.flatnonzero(np.any(R != 0,axis=0))
        bounds = [(None,None)]*n
        for col in radcols:
            check()
            bounds[col] = (1e-7,None)
        bounds[-1] = (.05,1.95)
        return {'X':X,'Y':Y,'R':R,'DX':DX,'DY':DY,'SR':SR,'L':L,'b':b,'grad':grad,'bounds':bounds,'z':z}

    def unpack(m,z):
        return np.column_stack((m['X']@z,m['Y']@z,m['R']@z))

    def optimize(m,z,iters):
        check()
        cache_z = None
        cache_f = None
        cache_j = None
        latest = z.copy()
        steps = 0
        def evaluate(t):
            nonlocal cache_z, cache_f, cache_j
            check()
            if cache_z is not None and np.array_equal(cache_z,t):
                return
            dx = m['DX']@t; dy = m['DY']@t
            d = np.sqrt(dx*dx+dy*dy)
            den = np.maximum(d,1e-15)
            cache_f = np.r_[d-m['SR']@t,m['L']@t+m['b']]
            cache_j = np.vstack(((dx/den)[:,None]*m['DX']+(dy/den)[:,None]*m['DY']-m['SR'],m['L']))
            cache_z = t.copy()
        def fun(t):
            evaluate(t)
            return cache_f
        def jac(t):
            evaluate(t)
            return cache_j
        def cb(t):
            nonlocal latest, steps
            check()
            latest = t.copy()
            steps += 1
            if steps % 16 == 0:
                retain(unpack(m,t))
        try:
            res = minimize(lambda t: float(m['grad']@t),z,jac=lambda t:m['grad'],
                           method='SLSQP',bounds=m['bounds'],
                           constraints={'type':'ineq','fun':fun,'jac':jac},callback=cb,
                           options={'maxiter':iters,'ftol':8e-12,'disp':False})
            latest = res.x
        except Stop:
            retain(unpack(m,latest))
            raise
        score = retain(unpack(m,latest))
        return latest,score

    try:
        # Screen distinct layer topologies first, then spend the remaining time on their leaders.
        for counts in ((3,5,4),(5,4,4),(5,5,3),(3,4,5)):
            check()
            m = model(counts)
            retain(unpack(m,m['z']))
            z,s = optimize(m,m['z'],params['screen_iterations'])
            saved.append((s,m,z))
        saved.sort(key=lambda t:t[0],reverse=True)
        for _,m,z in saved:
            check()
            z,s = optimize(m,z,params['polish_iterations'])
            m['z'] = z
        attempt = 0
        while time.monotonic() < deadline and saved:
            check()
            _,m,_ = saved[attempt % min(2,len(saved))]
            z = m['z'].copy()
            movable = np.flatnonzero(np.any(m['X'] != 0,axis=0) | np.any(m['Y'] != 0,axis=0))
            chosen = rng.choice(movable,min(5,len(movable)),replace=False)
            z[chosen] += rng.normal(0,params['perturbation'],len(chosen))
            z,s = optimize(m,z,60)
            if s >= best_score-1e-9:
                m['z'] = z
            attempt += 1
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    if best is None or best_score == -float('inf'):
        a = np.array([[(i%7)*.2,(i//7)*.2,.099] for i in range(21)])
        retain(a)
    return best
