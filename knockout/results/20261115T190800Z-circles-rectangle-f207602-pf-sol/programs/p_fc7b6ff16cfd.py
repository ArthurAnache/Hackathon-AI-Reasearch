import time
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {
    "removals": (5, 1, 12),
    "collapse_iterations": (90, 20, 200),
    "restore_iterations": (110, 20, 250),
    "floor_radius": (0.000001, 0.00000001, 0.001, "log"),
}
N = 21
I, J = np.triu_indices(N, 1)

class Deadline(Exception):
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
    p = max(u+v for u,v in zip(x,r))-min(u-v for u,v in zip(x,r))
    p += max(u+v for u,v in zip(y,r))-min(u-v for u,v in zip(y,r))
    return float(a[:, 2].sum()) if p <= 2 else -np.inf

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    stop = start + 0.70*budget_s
    rng = np.random.default_rng(seed)
    if init is None:
        a = np.array([[(k % 7 + .5)*.2, (k//7+.5)*.2, .099] for k in range(21)])
    else:
        a = np.array(init['circles'], dtype=float)
    best = init if init is not None else {'circles': a.tolist()}
    best_score = exact_score(a)

    def check():
        if time.monotonic() >= stop:
            raise Deadline()

    def pack(z):
        return np.column_stack((z[:N], z[N:2*N], z[2*N:3*N]))

    def consider(z):
        nonlocal best, best_score
        check()
        b = pack(z)
        if not np.isfinite(b).all() or np.min(b[:,2]) <= 0:
            return
        dist = np.sqrt((b[I,0]-b[J,0])**2+(b[I,1]-b[J,1])**2)
        f = min(1., float(np.min(dist/(b[I,2]+b[J,2]))))
        b[:,2] *= f*(1.-2e-14)
        lo = (b[:,:2]-b[:,2,None]).min(axis=0)
        hi = (b[:,:2]+b[:,2,None]).max(axis=0)
        scale = min(1., 2./float(np.sum(hi-lo)))*(1.-2e-14)
        b[:,:2] = (b[:,:2]-lo)*scale
        b[:,2] *= scale
        if float(b[:,2].sum()) <= best_score:
            return
        s = exact_score(b)
        if s > best_score:
            best_score, best = s, {'circles': b.tolist()}

    lo = (a[:,:2]-a[:,2,None]).min(axis=0)
    side = (a[:,:2]+a[:,2,None]).max(axis=0)-lo
    scale = 2./side.sum()
    p = (a[:,:2]-lo)*scale
    z0 = np.r_[p[:,0],p[:,1],a[:,2]*scale,side[0]*scale]
    m = len(I)
    B = np.zeros((4*N,3*N+1))
    q = np.arange(N)
    B[q,q] = 1; B[q,2*N+q] = -1
    B[N+q,q] = -1; B[N+q,2*N+q] = -1; B[N+q,-1] = 1
    B[2*N+q,N+q] = 1; B[2*N+q,2*N+q] = -1
    B[3*N+q,N+q] = -1; B[3*N+q,2*N+q] = -1; B[3*N+q,-1] = -1
    const = np.r_[np.zeros(3*N),np.full(N,2.)]
    last_z = None
    last_c = None
    last_j = None

    def calculate(z):
        nonlocal last_z,last_c,last_j
        check()
        if last_z is not None and np.array_equal(z,last_z):
            return
        dx = z[I]-z[J]
        dy = z[N+I]-z[N+J]
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-16)
        jac = np.zeros((m,3*N+1))
        rows = np.arange(m)
        jac[rows,I] = dx/d; jac[rows,J] = -dx/d
        jac[rows,N+I] = dy/d; jac[rows,N+J] = -dy/d
        jac[rows,2*N+I] = -1; jac[rows,2*N+J] = -1
        last_c = np.r_[d-z[2*N+I]-z[2*N+J],B@z+const]
        last_j = np.vstack((jac,B))
        last_z = z.copy()

    def fun_c(z):
        calculate(z)
        return last_c
    def jac_c(z):
        calculate(z)
        return last_j
    g = np.r_[np.zeros(2*N),-np.ones(N),0.]
    def objective(z):
        check()
        return -float(z[2*N:3*N].sum())
    bounds = [(None,None)]*(2*N)+[(1e-9,1.)]*N+[(.08,1.92)]
    def solve(z, bnd, steps):
        check()
        return minimize(objective,z,jac=lambda z:g,method='SLSQP',bounds=bnd,
                        constraints={'type':'ineq','fun':fun_c,'jac':jac_c},
                        options={'maxiter':steps,'ftol':2e-12,'disp':False}).x

    # Alternate small-circle removal with contact-rich interior-circle removal.
    radius_order = list(np.argsort(a[:,2]))
    d0 = np.sqrt(((p[I]-p[J])**2).sum(axis=1))
    deg = np.zeros(N)
    active = d0 < z0[2*N+I]+z0[2*N+J]+1e-5
    np.add.at(deg,I[active],1); np.add.at(deg,J[active],1)
    contact_order = list(np.argsort(-deg + .01*rng.random(N)))
    order = []
    for k in range(N):
        if time.monotonic() >= stop:
            break
        for idx in (radius_order[k],contact_order[k]):
            if idx not in order:
                order.append(idx)
    try:
        for removed in order[:int(params['removals'])]:
            check()
            z = z0.copy()
            tiny = float(params['floor_radius'])
            z[2*N+removed] = tiny
            bnd = list(bounds)
            bnd[2*N+removed] = (tiny,tiny)
            z = solve(z,bnd,int(params['collapse_iterations']))
            check()
            w = z[-1]; h = 2-w
            keep = np.arange(N) != removed
            centers = np.column_stack((z[:N],z[N:2*N]))[keep]
            radii = z[2*N:3*N][keep]
            # Rank a broad deterministic mesh by its actual empty-circle radius.
            xx,yy = np.meshgrid(np.linspace(0,w,19),np.linspace(0,h,19))
            sites = np.column_stack((xx.ravel(),yy.ravel()))
            clear = np.min(np.linalg.norm(sites[:,None,:]-centers[None,:,:],axis=2)-radii,axis=1)
            clear = np.minimum(clear,np.minimum.reduce([sites[:,0],w-sites[:,0],sites[:,1],h-sites[:,1]]))
            candidates = np.argsort(-clear)[:3]
            for site in candidates:
                check()
                trial = z.copy()
                trial[removed],trial[N+removed] = sites[site]
                trial[2*N+removed] = max(tiny,float(clear[site])*.98)
                trial = solve(trial,bounds,int(params['restore_iterations']))
                consider(trial)
                if time.monotonic() > start+.60*budget_s:
                    break
    except Deadline:
        pass
    return best
