import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {"iterations": (100, 40, 160), "jitter": (0.006, 0.0001, 0.03), "restart": (0.018, 0.002, 0.06)}


def construct(params, seed, budget_s, init=None):
    started = time.monotonic()
    deadline = started + 0.69 * budget_s
    rng = np.random.default_rng(seed)
    ai, aj = np.triu_indices(21, 1)
    best = None
    best_score = -float('inf')
    best_q = None
    best_model = None

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def array_of(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((np.asarray(obj['centers'], dtype=float), np.asarray(obj['radii'], dtype=float)))

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return -float('inf')
        x = [Fraction(float(z)) for z in a[:, 0]]
        y = [Fraction(float(z)) for z in a[:, 1]]
        r = [Fraction(float(z)) for z in a[:, 2]]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if p > 2:
            return -float('inf')
        d = a[ai, :2]-a[aj, :2]
        if np.any(np.sqrt(d[:, 0]**2+d[:, 1]**2) < a[ai, 2]+a[aj, 2]):
            return -float('inf')
        return float(np.sum(a[:, 2]))

    if init is not None:
        try:
            best_score = exact(array_of(init))
            if np.isfinite(best_score):
                best = init
        except (ValueError, KeyError, TypeError):
            pass

    def repair(a):
        nonlocal best, best_score
        a = a.copy()
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return False
        a[:, :2] -= np.mean(a[:, :2], axis=0)
        d = a[ai, :2]-a[aj, :2]
        distances = np.sqrt(d[:, 0]**2+d[:, 1]**2)
        f = min(1.0, float(np.min(distances/(a[ai, 2]+a[aj, 2]))))
        if f <= 0:
            return False
        a[:, 2] *= f*(1-4e-14)
        x = [Fraction(float(z)) for z in a[:, 0]]
        y = [Fraction(float(z)) for z in a[:, 1]]
        r = [Fraction(float(z)) for z in a[:, 2]]
        p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
        p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
        if p <= 0:
            return False
        a *= float(Fraction(2)/p)*(1-4e-14)
        score = exact(a)
        if score > best_score:
            best_score = score
            best = {'centers': a[:, :2].tolist(), 'radii': a[:, 2].tolist()}
            return True
        return False

    def model(counts, jitter=0.0):
        check()
        pts = []
        single = []
        pairs = []
        for level, count in enumerate(counts):
            check()
            for col in range(count):
                check()
                x = (col-(count-1)/2)*0.2
                if level == 0 and count == 3:
                    x *= 1.45
                x += jitter*float(rng.normal())
                if level == 0:
                    single.append(len(pts))
                    pts.append([x, 0.0, 0.095])
                else:
                    k = len(pts)
                    y = level*0.177
                    pts.extend([[x, y, 0.095], [x, -y, 0.095]])
                    pairs.append((k, k+1))
        a = np.array(pts)
        lo = np.min(a[:, :2]-a[:, 2, None], axis=0)
        hi = np.max(a[:, :2]+a[:, 2, None], axis=0)
        scale = 2/np.sum(hi-lo)
        a *= scale
        a[:, 0] -= lo[0]*scale
        m = 2*len(single)+3*len(pairs)+1
        B = np.zeros((63, m))
        mirror = np.arange(21)
        radii_cols = []
        y_cols = []
        c = 0
        for i in single:
            check()
            B[3*i, c] = 1
            B[3*i+2, c+1] = 1
            radii_cols.append(c+1)
            c += 2
        for i,j in pairs:
            check()
            B[3*i, c] = B[3*j, c] = 1
            B[3*i+1, c+1] = 1
            B[3*j+1, c+1] = -1
            B[3*i+2, c+2] = B[3*j+2, c+2] = 1
            radii_cols.append(c+2)
            y_cols.append(c+1)
            mirror[i], mirror[j] = j,i
            c += 3
        q = np.zeros(m)
        denom = np.sum(B*B, axis=0)
        q[:-1] = (B[:, :-1].T @ a.ravel())/denom[:-1]
        q[-1] = (hi[0]-lo[0])*scale
        keep = []
        for k,(i,j) in enumerate(zip(ai,aj)):
            check()
            other = tuple(sorted((int(mirror[i]), int(mirror[j]))))
            if (int(i),int(j)) <= other:
                keep.append(k)
        pi, pj = ai[keep], aj[keep]
        reps = single+[i for i,j in pairs]
        L = []
        offset = []
        for i in reps:
            check()
            bx, by, br = B[3*i:3*i+3]
            row = bx-br
            L.append(row)
            offset.append(0.0)
            row = -bx-br
            row[-1] += 1
            L.append(row)
            offset.append(0.0)
            row = by-br
            row[-1] -= .5
            L.append(row)
            offset.append(1.0)
            row = -by-br
            row[-1] -= .5
            L.append(row)
            offset.append(1.0)
        L = np.asarray(L)
        offset = np.asarray(offset)
        # Nonzero coordinate maps let the pair Jacobian be assembled directly
        # in the quotient, including self-mirror contacts.
        coord_col = np.argmax(np.abs(B), axis=1)
        coord_sign = B[np.arange(63), coord_col]
        pr = np.repeat(np.arange(len(pi)), 6)
        fullcols = np.column_stack((3*pi,3*pj,3*pi+1,3*pj+1,3*pi+2,3*pj+2))
        pc = coord_col[fullcols].ravel()
        signs = coord_sign[fullcols]
        gradient = -np.sum(B[2::3], axis=0)
        bounds = [(None,None)]*m
        for k in radii_cols:
            bounds[k] = (1e-7,None)
        bounds[-1] = (0.05,1.95)
        return {'B': B, 'pi': pi, 'pj': pj, 'L': L, 'offset': offset,
                'pr': pr, 'pc': pc, 'signs': signs, 'g': gradient,
                'bounds': bounds, 'rcols': radii_cols, 'ycols': y_cols}, q

    def optimize(M, q, iterations):
        nonlocal best_q, best_model
        check()
        B, pi, pj = M['B'], M['pi'], M['pj']
        L, offset = M['L'], M['offset']
        g = M['g']
        last = [q.copy()]
        cached = [None, None, None, None]
        count = [0]

        def compute(t):
            check()
            if cached[0] is not None and np.array_equal(t, cached[0]):
                return cached[1], cached[2], cached[3]
            a = (B@t).reshape(21,3)
            d = a[pi,:2]-a[pj,:2]
            dist = np.sqrt(np.sum(d*d,axis=1))
            con = np.r_[dist-a[pi,2]-a[pj,2], L@t+offset]
            cached[:] = [t.copy(),d,dist,con]
            return d,dist,con

        def jac(t):
            d,dist,con = compute(t)
            uv = d/np.maximum(dist[:,None],1e-15)
            entries = np.column_stack((uv[:,0],-uv[:,0],uv[:,1],-uv[:,1],-np.ones(len(pi)),-np.ones(len(pi))))
            J = np.zeros((len(pi),len(t)))
            np.add.at(J,(M['pr'],M['pc']),(entries*M['signs']).ravel())
            return np.vstack((J,L))

        def retain_q(t):
            nonlocal best_q,best_model
            if repair((B@t).reshape(21,3)):
                best_q = t.copy()
                best_model = M

        def callback(t):
            check()
            last[0] = t.copy()
            count[0] += 1
            if count[0] % 20 == 0:
                retain_q(t)

        try:
            res = minimize(lambda t: float(g@t),q,jac=lambda t:g,
                           method='SLSQP',bounds=M['bounds'],
                           constraints={'type':'ineq','fun':lambda t:compute(t)[2],'jac':jac},
                           callback=callback,
                           options={'maxiter':iterations,'ftol':8e-13,'disp':False})
            last[0] = res.x
        except Stop:
            retain_q(last[0])
            raise
        retain_q(last[0])
        return last[0]

    # Establish a good packing before allocating time to exploratory restarts.
    try:
        schedules = [((3,5,4),0.0), ((3,5,4),params['jitter']),
                     ((5,4,4),params['jitter']), ((3,4,5),params['jitter']),
                     ((3,3,3,3),params['jitter']), ((5,5,3),params['jitter'])]
        for counts,jitter in schedules:
            check()
            M,q = model(counts,jitter)
            if best is None:
                repair((M['B']@q).reshape(21,3))
            optimize(M,q,params['iterations'])
            if best_score > 2.3658323758:
                break
        attempt = 0
        while time.monotonic() < deadline and best_q is not None:
            check()
            M = best_model
            q = best_q.copy()
            m = len(q)
            strength = params['restart']*(0.5 if attempt % 3 == 0 else 1.0)
            noise = rng.normal(0,strength,m)
            noise[M['rcols']] *= .25
            noise[-1] *= .3
            q += noise
            q[M['rcols']] = np.maximum(q[M['rcols']], .005)
            q[-1] = np.clip(q[-1],.1,1.9)
            optimize(M,q,70)
            attempt += 1
    except (Stop,ValueError,np.linalg.LinAlgError):
        pass
    if best is None:
        a = np.array([[(i%7)*.2,(i//7)*.2,.09] for i in range(21)])
        repair(a)
    return best
