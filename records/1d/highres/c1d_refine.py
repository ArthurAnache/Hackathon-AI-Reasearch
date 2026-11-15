"""Tao constant 1d: split + p-norm continuation L-BFGS (our own code, from the public descriptions of PRs #312,
#345, #270 on teorth/optimizationproblems).

    c1d_refine.py --start FILE(.json|.npy) --out PREFIX [--split 4] [--p0 16] [--pmax 1e10] [--iters 3000]
                  [--budget SECONDS] [--threads 1]

Minimizes log ||v*v||_p - 2 log |sum v| with L-BFGS, p doubled stage by stage from p0 to pmax (stable p-norm: divide
by the max first). The exact float score 2N max|v*v| / (sum v)^2 of every evaluated point is tracked and the best
point is saved to PREFIX.npy (float64) with a PREFIX.json summary. Float search only: the decision is c1d_exact.py.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import scipy.fft as sfft
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1d_exact import load_values

ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--split", type=int, default=1); ap.add_argument("--p0", type=float, default=16.0)
ap.add_argument("--pmax", type=float, default=1e10); ap.add_argument("--iters", type=int, default=3000)
ap.add_argument("--budget", type=float, default=1e9); ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--m", type=int, default=30); ap.add_argument("--quiet", action="store_true")
ap.add_argument("--ckpt-every", type=float, default=0, help="also save PREFIX.npy every S seconds (0: only at the end)")
a = ap.parse_args()

v0 = np.repeat(np.asarray(load_values(a.start), dtype=np.float64), a.split)
v0 = v0 / np.abs(v0).max()
n = len(v0); L = sfft.next_fast_len(2 * n - 1, real=True); W = a.threads


def score(v):
    c = sfft.irfft(sfft.rfft(v, L, workers=W) ** 2, L, workers=W)[: 2 * n - 1]
    return 2 * n * np.abs(c).max() / v.sum() ** 2


t0 = time.time()
S0 = score(v0)
best = {"s": S0, "v": v0.copy()}
st = {"nev": 0, "ck": time.time()}


class Stop(Exception):
    pass


def obj(v, p):
    if time.time() - t0 > a.budget:
        raise Stop
    F = sfft.rfft(v, L, workers=W)
    c = sfft.irfft(F * F, L, workers=W)[: 2 * n - 1]
    ac = np.abs(c); m = ac.max(); sv = v.sum()
    s = 2 * n * m / sv ** 2
    st["nev"] += 1
    if s < best["s"]:
        best["s"] = s; best["v"] = v.copy()
    if a.ckpt_every and time.time() - st["ck"] > a.ckpt_every:
        st["ck"] = time.time()
        np.save(a.out + ".tmp.npy", best["v"]); __import__("os").replace(a.out + ".tmp.npy", a.out + ".npy")
        json.dump(dict(best=float(best["s"]), nev=st["nev"], seconds=time.time() - t0, p=p), open(a.out + ".ckpt.json", "w"))
    r = ac / m
    with np.errstate(under="ignore"):
        rp1 = np.power(r, p - 1.0)
    Z = float(np.dot(rp1, r))
    val = np.log(m) + np.log(Z) / p - 2.0 * np.log(abs(sv))
    wc = np.sign(c) * rp1 / (m * Z)                 # d log||c||_p / dc
    G = sfft.rfft(wc, L, workers=W) * np.conj(F)
    g = 2.0 * sfft.irfft(G, L, workers=W)[:n] - 2.0 / sv
    return val, g


v = v0.copy(); p = a.p0; stages = []
try:
    while p <= a.pmax * 1.0000001:
        res = minimize(obj, v, args=(p,), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=a.iters, maxcor=a.m, ftol=0, gtol=0, maxls=40))
        v = res.x / np.abs(res.x).max()
        stages.append((p, res.nit, float(score(v)), float(best["s"]), time.time() - t0))
        if not a.quiet:
            print(f"p={p:.3g} its={res.nit} cur={score(v):.13f} best={best['s']:.13f} t={time.time()-t0:.0f}s", flush=True)
        p *= 2.0
except Stop:
    pass
np.save(a.out + ".npy", best["v"])
out = dict(start=a.start, n=n, split=a.split, p0=a.p0, pmax=a.pmax, iters=a.iters, start_score=float(S0),
           best=float(best["s"]), nev=st["nev"], seconds=time.time() - t0, stages=stages)
json.dump(out, open(a.out + ".json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "stages"}))
