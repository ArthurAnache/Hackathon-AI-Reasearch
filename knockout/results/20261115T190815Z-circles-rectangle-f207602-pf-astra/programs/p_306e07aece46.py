import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    'cavity_size': (4, 3, 7),
    'grid': (49, 25, 81),
    'iterations': (100, 30, 200),
    'peak_randomness': (0.16, 0.0, 0.4),
}
N = 21
I, J = np.triu_indices(N, 1)
E = np.eye(N)

class StopSearch(Exception):
    pass

def exact_score(a):
    if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(a[:, 2] <= 0):
        return -np.inf
    d = np.sqrt((a[I, 0]-a[J, 0])**2 + (a[I, 1]-a[J, 1])**2)
    if np.any(d < a[I, 2]+a[J, 2]):
        return -np.inf
    x = list(map(lambda v: Fraction(float(v)), a[:, 0]))
    y = list(map(lambda v: Fraction(float(v)), a[:, 1]))
    r = list(map(lambda v: Fraction(float(v)), a[:, 2]))
    w = max(map(lambda t: t[0]+t[1], zip(x,r))) - min(map(lambda t: t[0]-t[1], zip(x,r)))
    h = max(map(lambda t: t[0]+t[1], zip(y,r))) - min(map(lambda t: t[0]-t[1], zip(y,r)))
    if w+h > 2:
        return -np.inf
    return float(np.sum(a[:, 2]))

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k % 7 + .5)*.18, (k // 7 + .5)*.18, .089] for k in range(N)])
        init = {'circles': a.tolist()}
    original = np.asarray(init['circles'], dtype=float)
    best = init
    best_score = exact_score(original)
    lo = np.min(original[:, :2]-original[:, 2,None], axis=0)
    hi = np.max(original[:, :2]+original[:, 2,None], axis=0)
    scale = 2 / np.sum(hi-lo)
    base = original.copy()
    base[:, :2] = (base[:, :2]-lo)*scale
    base[:, 2] *= scale
    width = (hi[0]-lo[0])*scale

    def check():
        if time.monotonic() >= deadline:
            raise StopSearch()

    def consider(z):
        nonlocal best, best_score
        check()
        a = np.column_stack((z[:N], z[N:2*N], z[2*N:3*N]))
        if not np.isfinite(a).all() or np.min(a[:, 2]) <= 0:
            return
        d = np.sqrt((a[I,0]-a[J,0])**2+(a[I,1]-a[J,1])**2)
        f = min(1.0, float(np.min(d/(a[I,2]+a[J,2]))))
        a[:,2] *= f*(1-4e-15)
        lo1 = np.min(a[:,:2]-a[:,2,None],axis=0)
        hi1 = np.max(a[:,:2]+a[:,2,None],axis=0)
        a[:,:2] -= lo1
        a *= min(1.0, 2/np.sum(hi1-lo1))*(1-4e-15)
        if float(a[:,2].sum()) > best_score:
            s = exact_score(a)
            if s > best_score:
                best_score = s
                best = {'circles': a.tolist()}

    objective_gradient = np.r_[np.zeros(2*N), -np.ones(N), 0.0]
    def objective(z):
        check()
        return -np.sum(z[2*N:3*N]), objective_gradient

    def constraints(z):
        check()
        x,y,r,w = z[:N],z[N:2*N],z[2*N:3*N],z[-1]
        dx,dy = x[I]-x[J],y[I]-y[J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        return np.r_[d-r[I]-r[J], x-r, w-x-r, y-r, 2-w-y-r]

    fixed = np.zeros((4*N,3*N+1))
    fixed[:N,:N] = E
    fixed[N:2*N,:N] = -E
    fixed[2*N:3*N,N:2*N] = E
    fixed[3*N:,N:2*N] = -E
    fixed[:N,2*N:3*N] = -E
    fixed[N:2*N,2*N:3*N] = -E
    fixed[2*N:3*N,2*N:3*N] = -E
    fixed[3*N:,2*N:3*N] = -E
    fixed[N:2*N,-1] = 1
    fixed[3*N:,-1] = -1
    rows = np.arange(len(I))
    def jacobian(z):
        check()
        dx,dy = z[I]-z[J],z[N+I]-z[N+J]
        d = np.sqrt(dx*dx+dy*dy+1e-30)
        q = np.zeros((len(I),3*N+1))
        q[rows,I] = dx/d
        q[rows,J] = -dx/d
        q[rows,N+I] = dy/d
        q[rows,N+J] = -dy/d
        q[rows,2*N+I] = -1
        q[rows,2*N+J] = -1
        return np.vstack((q,fixed))

    try:
        trial = 0
        while time.monotonic() < deadline:
            check()
            a = base.copy()
            # Remove a connected cavity, favoring smaller circles near its root.
            order = np.argsort(a[:,2])
            root = int(order[(trial*3 + seed % 7) % 12])
            distance = np.linalg.norm(a[:,:2]-a[root,:2],axis=1)
            priority = distance - .4*a[:,2]
            removed = np.argsort(priority)[:int(params['cavity_size'])]
            retained = [k for k in range(N) if k not in removed]
            g = int(params['grid'])
            xx,yy = np.meshgrid(np.linspace(.01,width-.01,g),np.linspace(.01,2-width-.01,g))
            points = np.column_stack((xx.ravel(),yy.ravel()))
            # Include old sites, but nearby displaced sites offer alternative tilings.
            old = a[removed,:2]
            points = np.vstack((points,old+rng.normal(0,.025,old.shape)))
            clearance = np.minimum.reduce((points[:,0],width-points[:,0],points[:,1],2-width-points[:,1]))
            for k in retained:
                check()
                clearance = np.minimum(clearance,np.linalg.norm(points-a[k,:2],axis=1)-a[k,2])
            for count,k in enumerate(removed):
                check()
                maximum = float(np.max(clearance))
                choices = np.flatnonzero(clearance >= maximum*(1-float(params['peak_randomness'])))
                if len(choices) == 0:
                    break
                # Alternate between maximal filling and randomized near-maximal filling.
                choice = int(np.argmax(clearance)) if (trial+count)%3 == 0 else int(rng.choice(choices))
                a[k,:2] = points[choice]
                a[k,2] = max(1e-5,clearance[choice]*.97)
                clearance = np.minimum(clearance,np.linalg.norm(points-a[k,:2],axis=1)-a[k,2])
            z = np.r_[a[:,0],a[:,1],a[:,2],width]
            calls = [0]
            def callback(v):
                check()
                calls[0] += 1
                if calls[0] % 5 == 0:
                    consider(v)
            result = minimize(objective,z,jac=True,method='SLSQP',
                bounds=[(None,None)]*(2*N)+[(1e-7,.5)]*N+[(.25,1.75)],
                constraints={'type':'ineq','fun':constraints,'jac':jacobian},
                callback=callback,
                options={'maxiter':int(params['iterations']),'ftol':2e-12,'disp':False})
            consider(result.x)
            trial += 1
    except StopSearch:
        pass
    return best
