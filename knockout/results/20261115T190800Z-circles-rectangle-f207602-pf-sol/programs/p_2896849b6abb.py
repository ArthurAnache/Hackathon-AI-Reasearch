import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import time
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

PARAMS = {'stage': (0.31, 0.15, 0.5), 'ftol': (2e-11, 1e-13, 1e-7)}

class StopSearch(Exception):
    pass

def exact_score(data):
    try:
        c = np.asarray(data['circles'], dtype=float)
        if c.shape != (21, 3) or not np.isfinite(c).all() or np.any(c[:, 2] <= 0):
            return -float('inf')
        x = [Fraction(float(t)) for t in c[:, 0]]
        y = [Fraction(float(t)) for t in c[:, 1]]
        r = [Fraction(float(t)) for t in c[:, 2]]
        extent = max(a+b for a,b in zip(x,r))-min(a-b for a,b in zip(x,r))
        extent += max(a+b for a,b in zip(y,r))-min(a-b for a,b in zip(y,r))
        if extent > 2:
            return -float('inf')
        for i in range(21):
            for j in range(i):
                d = np.sqrt((c[i,0]-c[j,0])**2+(c[i,1]-c[j,1])**2)
                if d < c[i,2]+c[j,2]:
                    return -float('inf')
        return float(np.sum(c[:,2]))
    except Exception:
        return -float('inf')

def construct(params, seed, budget_s, init=None):
    start = time.monotonic()
    deadline = start + 0.70 * budget_s
    rng = np.random.default_rng(seed)
    best = init
    bestscore = exact_score(init) if init is not None else -float('inf')
    ii, jj = np.triu_indices(21, 1)

    def retain(c):
        nonlocal best, bestscore
        if time.monotonic() >= deadline:
            return
        c = np.array(c, dtype=float, copy=True)
        if not np.isfinite(c).all() or np.any(c[:,2] <= 0):
            return
        d = np.sqrt((c[ii,0]-c[jj,0])**2+(c[ii,1]-c[jj,1])**2)
        factor = min(1.0, float(np.min(d/(c[ii,2]+c[jj,2]))))
        c[:,2] *= factor * (1-3e-13)
        extent = np.ptp(c[:,0:1] + np.array([0.0])) if False else 0.0
        extent = (np.max(c[:,0]+c[:,2])-np.min(c[:,0]-c[:,2]) +
                  np.max(c[:,1]+c[:,2])-np.min(c[:,1]-c[:,2]))
        c *= min(1.0, 2.0/extent) * (1-3e-13)
        if float(np.sum(c[:,2])) <= bestscore:
            return
        candidate = {'circles': c.tolist()}
        score = exact_score(candidate)
        if score > bestscore:
            bestscore, best = score, candidate

    # Variables: twelve representative x coordinates, nine positive y
    # coordinates, twelve radii, and rectangle width.
    xi = np.r_[np.arange(12), np.arange(3,12)]
    ri = np.r_[np.arange(21,33), np.arange(24,33)]
    yi = np.r_[np.zeros(3,dtype=int), np.arange(12,21), np.arange(12,21)]
    ys = np.r_[np.zeros(3), np.ones(9), -np.ones(9)]
    reflection = np.r_[np.arange(3), np.arange(12,21), np.arange(3,12)]
    pairs = []
    for a,b in zip(ii,jj):
        if time.monotonic() >= deadline:
            break
        reflected = tuple(sorted((int(reflection[a]),int(reflection[b]))))
        if (int(a),int(b)) <= reflected:
            pairs.append((int(a),int(b)))
    pa = np.array([p[0] for p in pairs],dtype=int)
    pb = np.array([p[1] for p in pairs],dtype=int)
    m = len(pa)
    rows = np.arange(m)
    objective = np.zeros(34)
    objective[21:24] = -1
    objective[24:33] = -2
    bounds = [(-1,1)]*12 + [(0,1)]*9 + [(1e-5,0.6)]*12 + [(0.25,1.75)]

    def unpack(z):
        return np.column_stack((z[xi], z[yi]*ys, z[ri]))

    stage_end = deadline
    def check():
        if time.monotonic() >= stage_end:
            raise StopSearch()

    def constraints(z):
        check()
        x, y, r = z[xi], z[yi]*ys, z[ri]
        dx = x[pa]-x[pb]
        dy = y[pa]-y[pb]
        dist = np.sqrt(dx*dx+dy*dy)
        yr = np.r_[np.zeros(3),z[12:21]]
        return np.r_[dist-r[pa]-r[pb], z[33]/2-z[:12]-z[21:33],
                     z[33]/2+z[:12]-z[21:33],
                     1-z[33]/2-yr-z[21:33]]

    def jac(z):
        check()
        x,y = z[xi],z[yi]*ys
        dx,dy = x[pa]-x[pb], y[pa]-y[pb]
        d = np.maximum(np.sqrt(dx*dx+dy*dy),1e-20)
        a,b = dx/d,dy/d
        J = np.zeros((m+36,34))
        np.add.at(J,(rows,xi[pa]),a)
        np.add.at(J,(rows,xi[pb]),-a)
        np.add.at(J,(rows,yi[pa]),b*ys[pa])
        np.add.at(J,(rows,yi[pb]),-b*ys[pb])
        np.add.at(J,(rows,ri[pa]),-1)
        np.add.at(J,(rows,ri[pb]),-1)
        q=np.arange(12)
        J[m+q,q]=-1
        J[m+12+q,q]=1
        J[m+q,21+q]=-1
        J[m+12+q,21+q]=-1
        J[m+24+q,21+q]=-1
        J[m:m+24,33]=0.5
        J[m+24:,33]=-0.5
        J[m+27+np.arange(9),12+np.arange(9)]=-1
        return J

    trajectories=[]
    best_z=None
    best_raw=-1e9
    def callback(z):
        nonlocal best_z,best_raw
        check()
        retain(unpack(z))
        violation = float(np.min(constraints(z)))
        value = -float(objective@z)
        if violation > -2e-7 and value > best_raw:
            best_raw=value
            best_z=z.copy()

    def run(z, seconds, iterations=150):
        nonlocal stage_end
        stage_end=min(deadline,time.monotonic()+seconds)
        last=[z.copy()]
        def cb(v):
            last[0]=v.copy()
            callback(v)
        try:
            result=minimize(lambda v: float(objective@v),z,jac=lambda v: objective,
                method='SLSQP',bounds=bounds,
                constraints={'type':'ineq','fun':constraints,'jac':jac},
                callback=cb,options={'maxiter':iterations,'ftol':params['ftol'],'disp':False})
            last[0]=result.x
        except StopSearch:
            pass
        retain(unpack(last[0]))
        return last[0]

    def lattice(counts, shifts):
        z=np.zeros(34)
        z[:3]=np.array([-2.,0.,2.])
        pos=3
        for k,(count,shift) in enumerate(zip(counts,shifts)):
            if time.monotonic() >= deadline:
                break
            z[pos:pos+count]=2*np.arange(count)-(count-1)+shift
            z[12+pos-3:12+pos-3+count]=(k+1)*np.sqrt(3.)
            pos+=count
        z[21:33]=1
        c=unpack(z)
        width=np.max(c[:,0]+1)-np.min(c[:,0]-1)
        height=np.max(c[:,1]+1)-np.min(c[:,1]-1)
        scale=2/(width+height)
        z[:33]*=scale
        z[33]=width*scale
        return z

    families=[((5,4),(0.,0.)),((4,5),(0.,0.)),((3,3,3),(1.,0.,1.))]
    for counts, shifts in families:
        if time.monotonic() >= deadline-0.12:
            break
        z=lattice(counts,shifts)
        retain(unpack(z))
        z=run(z,min(params['stage'],max(0.05,(deadline-time.monotonic())*0.42)))
        trajectories.append(z)

    if best_z is not None and time.monotonic()<deadline-0.04:
        run(best_z,deadline-time.monotonic(),220)
    if best is None:
        # Always-valid emergency construction.
        c=np.array([[0.08*(i%5),0.08*(i//5),0.035] for i in range(21)])
        best={'circles':c.tolist()}
    return best
