import time
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

PARAMS = {'iterations': (100, 40, 180), 'perturbation': (0.018, 0.001, 0.08), 'reserve': (0.18, 0.05, 0.3)}


def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + .69 * budget_s
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(21, 1)
    rows = np.arange(210)
    kk = np.arange(21)
    best = init
    best_score = -float('inf')
    saved = None

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() >= deadline:
            raise Stop()

    def array(obj):
        if 'circles' in obj:
            return np.asarray(obj['circles'], dtype=float)
        return np.column_stack((obj['centers'], obj['radii']))

    def perimeter(a):
        x = [Fraction(float(v)) for v in a[:, 0]]
        y = [Fraction(float(v)) for v in a[:, 1]]
        r = [Fraction(float(v)) for v in a[:, 2]]
        return (max(u+v for u,v in zip(x,r)) - min(u-v for u,v in zip(x,r))
                + max(u+v for u,v in zip(y,r)) - min(u-v for u,v in zip(y,r)))

    def exact(a):
        if a.shape != (21, 3) or not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return -float('inf')
        if perimeter(a) > 2:
            return -float('inf')
        d = a[ii,:2] - a[jj,:2]
        if np.any(np.sqrt(d[:,0]**2+d[:,1]**2) < a[ii,2]+a[jj,2]):
            return -float('inf')
        return float(np.sum(a[:,2]))

    if init is not None:
        try:
            best_score = exact(array(init))
        except (ValueError, TypeError, KeyError):
            pass

    def retain(a, model=None, z=None):
        nonlocal best, best_score, saved
        a = a.copy()
        if not np.isfinite(a).all() or np.min(a[:,2]) <= 0:
            return
        a[:,:2] -= np.mean(a[:,:2], axis=0)
        d = a[ii,:2]-a[jj,:2]
        f = min(1., float(np.min(np.sqrt(d[:,0]**2+d[:,1]**2)/(a[ii,2]+a[jj,2]))))
        if f <= 0:
            return
        a[:,2] *= f*(1-4e-14)
        p = perimeter(a)
        if p <= 0:
            return
        a *= float(Fraction(2)/p)*(1-4e-14)
        score = exact(a)
        if score > best_score:
            best_score = score
            best = {'centers': a[:,:2].tolist(), 'radii': a[:,2].tolist()}
            if model is not None:
                saved = (model, z.copy())

    def make_model(counts=None):
        cx = np.zeros(21, dtype=int)
        cy = np.zeros(21, dtype=int)
        cr = np.zeros(21, dtype=int)
        sy = np.ones(21)
        pts = []
        n = 0
        c = 0
        if counts is None:
            cx = np.arange(0,63,3)
            cy = cx+1
            cr = cx+2
            return cx,cy,cr,sy,63
        for level, count in enumerate(counts):
            check()
            for j in range(count):
                check()
                x = (j-(count-1)/2)*.20
                if level == 0 and count == 3:
                    x *= 1.45
                if level == 0:
                    cx[n],cr[n] = c,c+1
                    sy[n] = 0.
                    pts.append([x,0.,.095])
                    c += 2
                    n += 1
                else:
                    cx[n:n+2] = c
                    cy[n:n+2] = c+1
                    cr[n:n+2] = c+2
                    sy[n:n+2] = [1.,-1.]
                    pts.extend([[x,level*.177,.095],[x,-level*.177,.095]])
                    c += 3
                    n += 2
        a = np.array(pts)
        lo = np.min(a[:,:2]-a[:,2,None], axis=0)
        hi = np.max(a[:,:2]+a[:,2,None], axis=0)
        scale = 2/np.sum(hi-lo)
        a *= scale
        a[:,0] -= lo[0]*scale
        model = (cx,cy,cr,sy,c)
        return model, pack(a, model, (hi[0]-lo[0])*scale)

    def unpack(z, m):
        cx,cy,cr,sy,n = m
        return np.column_stack((z[cx],z[cy]*sy,z[cr]))

    def pack(a,m,w):
        cx,cy,cr,sy,n = m
        z = np.zeros(n+1)
        z[cx] = a[:,0]
        z[cr] = a[:,2]
        mask = sy != 0
        z[cy[mask]] = a[mask,1]*sy[mask]
        z[n] = w
        return z

    def optimize(z, m, maxiter):
        check()
        cx,cy,cr,sy,n = m
        grad = np.zeros(n+1)
        np.add.at(grad, cr, -1.)
        last = [z.copy()]
        calls = [0]

        def constraints(t):
            check()
            x,y,r = t[cx],t[cy]*sy,t[cr]
            dx,dy = x[ii]-x[jj],y[ii]-y[jj]
            return np.r_[np.sqrt(dx*dx+dy*dy)-r[ii]-r[jj],
                         x-r,t[n]-x-r,1-t[n]/2+y-r,1-t[n]/2-y-r]

        def jac(t):
            check()
            x,y = t[cx],t[cy]*sy
            dx,dy = x[ii]-x[jj],y[ii]-y[jj]
            den = np.maximum(np.sqrt(dx*dx+dy*dy),1e-15)
            ux,uy = dx/den,dy/den
            J = np.zeros((294,n+1))
            J[rows,cx[ii]] += ux
            J[rows,cx[jj]] -= ux
            J[rows,cy[ii]] += uy*sy[ii]
            J[rows,cy[jj]] -= uy*sy[jj]
            J[rows,cr[ii]] -= 1
            J[rows,cr[jj]] -= 1
            J[210+kk,cx] += 1
            J[210+kk,cr] -= 1
            J[231+kk,cx] -= 1
            J[231+kk,cr] -= 1
            J[231+kk,n] = 1
            J[252+kk,cy] += sy
            J[252+kk,cr] -= 1
            J[252+kk,n] = -.5
            J[273+kk,cy] -= sy
            J[273+kk,cr] -= 1
            J[273+kk,n] = -.5
            return J

        def objective(t):
            check()
            return float(grad@t)

        def callback(t):
            check()
            last[0] = t.copy()
            calls[0] += 1
            if calls[0] % 18 == 0:
                retain(unpack(t,m),m,t)

        bounds = [(None,None)]*(n+1)
        for k in np.unique(cr):
            bounds[int(k)] = (1e-7,1.)
        bounds[n] = (.05,1.95)
        try:
            result = minimize(objective,z,jac=lambda t: grad,method='SLSQP',
                              bounds=bounds,
                              constraints={'type':'ineq','fun':constraints,'jac':jac},
                              callback=callback,
                              options={'maxiter':int(maxiter),'ftol':8e-12,'disp':False})
            last[0] = result.x
        except Stop:
            retain(unpack(last[0],m),m,last[0])
            raise
        retain(unpack(last[0],m),m,last[0])
        return last[0]

    try:
        if best_score > 2.30:
            a = array(init).copy()
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,0] -= lo[0]
            a[:,1] -= (lo[1]+hi[1])/2
            m = make_model()
            optimize(pack(a,m,hi[0]-lo[0]),m,55)
        for counts in ((3,5,4),(5,4,4),(5,5,3),(3,4,5)):
            check()
            m,z = make_model(counts)
            retain(unpack(z,m),m,z)
            optimize(z,m,params['iterations'])
        attempt = 0
        while time.monotonic() < deadline - params['reserve'] and saved is not None:
            check()
            m,z = saved
            z = z.copy()
            cx,cy,cr,sy,n = m
            movable = np.unique(np.r_[cx,cy[sy != 0]])
            choose = rng.choice(movable,min(7,len(movable)),replace=False)
            z[choose] += rng.normal(0,params['perturbation']*(1+attempt%3),len(choose))
            optimize(z,m,65)
            attempt += 1
        if best is not None and time.monotonic() < deadline-.035:
            a = array(best).copy()
            lo = np.min(a[:,:2]-a[:,2,None],axis=0)
            hi = np.max(a[:,:2]+a[:,2,None],axis=0)
            a[:,0] -= lo[0]
            a[:,1] -= (lo[1]+hi[1])/2
            m = make_model()
            optimize(pack(a,m,hi[0]-lo[0]),m,45)
    except (Stop, ValueError, np.linalg.LinAlgError):
        pass
    if best is None or best_score == -float('inf'):
        return {'circles': [[(i%7)*.2,(i//7)*.2,.09999999999998] for i in range(21)]}
    return best
