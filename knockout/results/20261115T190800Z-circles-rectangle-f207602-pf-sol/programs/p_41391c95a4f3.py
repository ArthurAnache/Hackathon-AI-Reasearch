import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'iterations': (85, 30, 150), 'shake': (0.032, 0.004, 0.09), 'fraction': (0.69, 0.5, 0.7)}


def construct(params, seed, budget_s, init=None):
    deadline = time.monotonic() + params['fraction'] * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    best = init
    best_score = -float('inf')
    pool = []

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def array(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((obj['centers'], obj['radii']))

    def score(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        d = a[ii, :2] - a[jj, :2]
        if np.any(np.sqrt(d[:, 0]**2 + d[:, 1]**2) < a[ii, 2] + a[jj, 2]):
            return -float('inf')
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        p = max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r))
        if p > 2:
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            best_score = score(array(init))
        except (ValueError, TypeError, KeyError):
            pass

    def retain(a):
        nonlocal best, best_score
        a = a.copy()
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = a[ii, :2] - a[jj, :2]
        ds = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
        f = min(1., float(np.min(ds / (a[ii, 2] + a[jj, 2]))))
        if f <= 0:
            return -float('inf')
        a[:, 2] *= f * (1 - 5e-14)
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        a *= float(Fraction(2)/p) * (1-5e-14)
        s = score(a)
        if s > best_score:
            best_score = s
            best = {'centers': a[:, :2].tolist(), 'radii': a[:, 2].tolist()}
        return s

    fallback = np.array([[(i % 7)*.2, (i // 7)*.2, .0999999999999] for i in range(21)])
    retain(fallback)

    def model(counts):
        check()
        singles = counts[0]
        m = 2*singles + 3*((21-singles)//2) + 1
        X = np.zeros((21, m)); Y = X.copy(); R = X.copy()
        z = np.zeros(m)
        mirror = np.arange(21)
        reps = []
        ridx = []
        k = 0; c = 0
        for level, count in enumerate(counts):
            check()
            for col in range(count):
                check()
                xx = (col-(count-1)/2)*.2
                if level == 0:
                    if count == 3:
                        xx *= 1.45
                    X[k,c] = 1; R[k,c+1] = 1
                    z[c] = xx; z[c+1] = .095
                    reps.append((k, False)); ridx.append(c+1)
                    c += 2; k += 1
                else:
                    X[k:k+2,c] = 1
                    Y[k,c+1] = 1; Y[k+1,c+1] = -1
                    R[k:k+2,c+2] = 1
                    z[c:c+3] = [xx, level*.177, .095]
                    mirror[k] = k+1; mirror[k+1] = k
                    reps.append((k, True)); ridx.append(c+2)
                    c += 3; k += 2
        a = np.column_stack((X@z, Y@z, R@z))
        lo = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi = np.max(a[:,:2]+a[:,2,None],axis=0)
        z *= 2/np.sum(hi-lo)
        z[-1] = (hi[0]-lo[0])*2/np.sum(hi-lo)
        shift = -lo[0]*2/np.sum(hi-lo)
        z += shift * np.sum(X, axis=0)
        # Every x coordinate occurs once for a single and twice for a pair.
        for j in range(m-1):
            check()
            if np.any(X[:,j]):
                z[j] -= shift*(np.sum(X[:,j])-1)
        selected = []
        for t, (i,j) in enumerate(zip(ii,jj)):
            check()
            mirrored = tuple(sorted((int(mirror[i]), int(mirror[j]))))
            if (int(i),int(j)) <= mirrored:
                selected.append(t)
        ai = ii[selected]; aj = jj[selected]
        DX = X[ai]-X[aj]; DY = Y[ai]-Y[aj]; RS = R[ai]+R[aj]
        wall = []
        w = np.zeros(m); w[-1] = 1
        for i, paired in reps:
            check()
            wall.extend((X[i]-R[i], w-X[i]-R[i], -.5*w+Y[i]-R[i]))
            if paired:
                wall.append(-.5*w-Y[i]-R[i])
        W = np.array(wall)
        offset = np.zeros(len(W))
        t = 0
        for i, paired in reps:
            check()
            offset[t+2] = 1
            if paired:
                offset[t+3] = 1
            t += 4 if paired else 3
        g = -np.sum(R,axis=0)
        bounds = [(None,None)]*m
        for j in ridx:
            check()
            bounds[j] = (1e-7, .6)
        bounds[-1] = (.05, 1.95)
        return {'X':X,'Y':Y,'R':R,'DX':DX,'DY':DY,'RS':RS,'W':W,'off':offset,'g':g,'bounds':bounds,'z':z,'ridx':ridx}

    def unpack(M,z):
        return np.column_stack((M['X']@z,M['Y']@z,M['R']@z))

    def solve(M,z,n):
        check()
        DX,DY,RS,W = M['DX'],M['DY'],M['RS'],M['W']
        latest = [z.copy()]
        cache = [None,None,None]
        def compute(t):
            check()
            if cache[0] is None or not np.array_equal(t,cache[0]):
                dx = DX@t; dy = DY@t
                d = np.sqrt(dx*dx+dy*dy)
                den = np.maximum(d,1e-16)
                cache[0] = t.copy()
                cache[1] = np.r_[d-RS@t, W@t+M['off']]
                cache[2] = np.vstack(((dx/den)[:,None]*DX+(dy/den)[:,None]*DY-RS,W))
            return cache[1],cache[2]
        count = [0]
        def cb(t):
            latest[0] = t.copy()
            check()
            count[0] += 1
            if count[0] % 18 == 0:
                retain(unpack(M,t))
        try:
            result = minimize(lambda t: float(M['g']@t),z,jac=lambda t:M['g'],method='SLSQP',bounds=M['bounds'],constraints={'type':'ineq','fun':lambda t:compute(t)[0],'jac':lambda t:compute(t)[1]},callback=cb,options={'maxiter':n,'ftol':2e-11,'disp':False})
            latest[0] = result.x
        except Stop:
            retain(unpack(M,latest[0]))
            raise
        val = retain(unpack(M,latest[0]))
        return latest[0],val

    try:
        families = [(3,5,4),(5,4,4),(5,5,3),(3,4,5),(3,3,3,3),(1,4,4,2)]
        for counts in families:
            check()
            M = model(counts)
            z,s = solve(M,M['z'],params['iterations'])
            pool.append([s,M,z])
        attempt = 0
        while time.monotonic() < deadline:
            check()
            pool.sort(key=lambda q:q[0],reverse=True)
            s,M,z0 = pool[0 if attempt % 4 else min(1,len(pool)-1)]
            z = z0.copy()
            if attempt % 5 != 0:
                noise = rng.normal(0,params['shake'],len(z))
                noise[-1] *= .4
                noise[M['ridx']] *= .15
                z += noise
                z[M['ridx']] = np.maximum(z[M['ridx']], .002)
                z[-1] = np.clip(z[-1],.1,1.9)
            z,snew = solve(M,z,params['iterations'])
            if snew > s:
                for entry in pool:
                    check()
                    if entry[1] is M:
                        entry[0] = snew; entry[2] = z
                        break
            attempt += 1
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    return best
