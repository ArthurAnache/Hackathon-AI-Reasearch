"""Problem plug-in: Tao constant 1a (Sidon autocorrelation, f >= 0, upper bound, lower is better).

score = 2N max_k (a*a)_k / (sum a)^2 for non-negative heights a (wt-compare/problems/p1a.py).

- seed: WS2 optimizer wt-compare/optimizers/1a_tuned.py, search(budget_s, seed, config), from noise.
  SEED_CELLS is its *starting* resolution; it upsamples x3 twice, so a seed has 9 x SEED_CELLS cells.
- refine: our own split + p-norm continuation L-BFGS, the same kind as attack/c1d_refine.py, on a = x^2 so that
  non-negativity is kept by construction (a CLI in this file: python -m autoloop.tao1a refine ...).
- certify: round to integers (max = 2^56, clipped at 0), then two exact checks that must agree: the frozen verifier
  p1a.verify_exact (run in a subprocess inside wt-compare) and attack/c1d_exact.check (direct recomputation of the
  deciding knot, random-point check mod a 61-bit prime).
- reference: live, from the Tao 1a page ("Known upper bounds") and titles of open "1a: ... upper bound ..." PRs.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.fft as sfft

from .safety import clean_env, safe_err

NAME = "tao-1a"
LOWER_IS_BETTER = True
SEED_CELLS = 150                         # 1a_tuned starting resolution; seeds come out at 1350 cells
SPLIT_UP = 4                             # 1350 -> 5400 -> 21600 -> 86400
METHOD_NOTES = ""
WIDE_P0_EARLY = [8.0]
WIDE_P0_FINAL = [4.0, 8.0]
SOFT_RESTARTS = [4.0, 8.0, 2.0, 16.0]
P0_SCHEDULE = [2.0, 4.0, 8.0, 16.0, 64.0, 512.0, 1024.0, 4096.0]
LOCAL_MAX_CELLS = 21600                  # refines above this size get the long (Modal) budget
MODAL_KINDS = dict(seed="seed1a", refine="refine1a")
REPO = Path(__file__).resolve().parent.parent
ATTACK = REPO / "attack"
WORKSPACE = __import__("os").environ.get("AUTOLOOP_WORKSPACE", str((lambda q: q[2] if len(q) > 2 else q[-1])(__import__("pathlib").Path(__file__).resolve().parents)))  # folder that holds repo/, wt-compare/, shared/ and .env
WS2 = Path(__import__("os").environ.get("AUTOLOOP_WS2", WORKSPACE + "/wt-compare"))
PY = sys.executable
REF_PAGE = "https://teorth.github.io/optimizationproblems/constants/1a.html"
REF_REPO = "teorth/optimizationproblems"
BITS = 56


def score(v) -> float:
    a = np.asarray(v, dtype=np.float64)
    n = len(a)
    L = sfft.next_fast_len(2 * n - 1, real=True)
    c = sfft.irfft(sfft.rfft(a, L) ** 2, L)[: 2 * n - 1]
    return float(2 * n * c.max() / a.sum() ** 2)


def vec_id(v) -> str:
    return hashlib.sha256(np.asarray(v, dtype=np.float64).tobytes()).hexdigest()


def load_vec(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    d = json.load(open(path))
    for key in ("best_values", "solution", "values"):
        if isinstance(d, dict) and key in d:
            d = d[key]
    if isinstance(d, dict):
        d = d["values"]
    return np.asarray([float(x) for x in d], dtype=np.float64)


def valid_vec(v) -> bool:
    return len(v) >= 20 and bool(np.isfinite(v).all()) and bool((v >= 0).all()) and v.sum() > 0


from .tao1d import _run_parallel  # noqa: E402  (generic helper: parallel subprocesses + CPU-seconds)


class LocalExecutor:
    real = True
    usd_per_cpu_s = 0.0
    spent_usd = 0.0

    def __init__(self, workdir, max_cells=LOCAL_MAX_CELLS):
        self.dir = Path(workdir); self.max_cells = max_cells
        (self.dir / "certs").mkdir(parents=True, exist_ok=True)

    def seed(self, n, budget_s, cells, seed0):
        outs = [str(self.dir / "certs" / f"seed_s{seed0 + i}_n{cells}") for i in range(n)]
        cmds = [[PY, "-m", "autoloop.tao1a", "seed", "--cells", str(cells), "--budget", str(budget_s),
                 "--seed", str(seed0 + i), "--out", o] for i, o in enumerate(outs)]
        res, cpu = _run_parallel(cmds, timeout=budget_s * 2 + 120)
        rows = []
        for i, (o, r) in enumerate(zip(outs, res)):
            if r.returncode != 0 or not Path(o + ".npy").exists():
                rows.append(dict(ok=False, error=safe_err(r.stderr))); continue
            rows.append(dict(ok=True, path=o + ".npy", reported=json.load(open(o + ".json"))["best"],
                             params=dict(seed=seed0 + i, start_cells=cells, budget_s=budget_s,
                                         optimizer="wt-compare 1a_tuned")))
        return rows, cpu

    def refine(self, parent_path, split, p0s, budget_s, tag):
        outs = [str(self.dir / "certs" / f"{tag}_x{split}_p{p0:g}") for p0 in p0s]
        cmds = [[PY, "-m", "autoloop.tao1a", "refine", "--start", parent_path, "--out", o, "--split", str(split),
                 "--p0", str(p0), "--budget", str(budget_s)] for p0, o in zip(p0s, outs)]
        res, cpu = _run_parallel(cmds, timeout=budget_s + 600)
        rows = []
        for p0, o, r in zip(p0s, outs, res):
            if r.returncode != 0 or not Path(o + ".npy").exists():
                rows.append(dict(ok=False, error=safe_err(r.stderr), params=dict(split=split, p0=p0))); continue
            j = json.load(open(o + ".json"))
            rows.append(dict(ok=True, path=o + ".npy", reported=j["best"],
                             params=dict(split=split, p0=p0, budget_s=budget_s, nev=j["nev"], seconds=j["seconds"],
                                         start_score=j["start_score"])))
        return rows, cpu

    def certify(self, parent_path, tag):
        v = load_vec(parent_path)
        ints = [int(x) for x in np.maximum(np.rint(v / v.max() * 2.0 ** BITS), 0)]
        out = str(self.dir / "certs" / f"{tag}_int{BITS}.json")
        json.dump({"values": ints}, open(out, "w"), separators=(",", ":"))
        code = ("import json,sys; from problems import p1a; f=p1a.verify_exact(json.load(open(sys.argv[1]))); "
                "print(f.numerator, f.denominator)")
        (r,), cpu = _run_in(WS2, [PY, "-c", code, out])     # cwd wt-compare, so `problems` is WS2's package
        if r.returncode != 0:
            return dict(ok=False, error="p1a.verify_exact: " + safe_err(r.stderr)), cpu
        p, q = (int(x) for x in r.stdout.split())
        f1 = Fraction(p, q)
        sys.path.insert(0, str(ATTACK))
        import c1d_exact                      # second, independent exact check (direct knot + modular check)
        chk = c1d_exact.check(ints)
        if chk["score"] != f1:
            return dict(ok=False, error=f"exact checks disagree: p1a {float(f1)!r} vs c1d_exact {float(chk['score'])!r}"), cpu
        return dict(ok=True, path=out, ints=ints, exact=f1, knot=chk["knot"]), cpu


def _run_in(cwd, cmd):
    from .tao1d import _cpu_children
    c0 = _cpu_children()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), timeout=3600, env=clean_env())
    return (r,), _cpu_children() - c0


class FakeExecutor:
    real = False
    usd_per_cpu_s = 0.001
    spent_usd = 0.0

    def __init__(self, workdir, max_cells=LOCAL_MAX_CELLS, seed=0):
        from .tao1d import FakeExecutor as F1
        self._f = F1(workdir, max_cells, seed)
        self.dir, self.max_cells = self._f.dir, max_cells

    def seed(self, *a, **k):
        rows, cpu = self._f.seed(*a, **k)
        for r in rows:
            r["reported"] += 0.04
        return rows, cpu

    def refine(self, *a, **k):
        return self._f.refine(*a, **k)

    def refine_many(self, *a, **k):
        return self._f.refine_many(*a, **k)

    def certify(self, *a, **k):
        return self._f.certify(*a, **k)


NUM = re.compile(r"1\.5[0-9]{4,}")


def fetch_reference() -> dict:
    """Best 1a upper bound, read live; fails closed (see autoloop/refread.py)."""
    from .refread import read
    return read("1a", REF_PAGE, NUM, 1.49, 1.56)


def claim_threshold(ref_value: str) -> Fraction:
    k = len(ref_value.split(".")[1])
    return Fraction(ref_value) - Fraction(5, 10 ** (k + 1))


# ------------------------------------------------------------------------------------------------ workers (CLI)
def _seed_cli(a):
    import importlib.util
    sys.path[:0] = [str(WS2), str(WS2 / "optimizers")]
    spec = importlib.util.spec_from_file_location("ws2_1a_tuned", WS2 / "optimizers" / "1a_tuned.py")
    opt = importlib.util.module_from_spec(spec); spec.loader.exec_module(opt)
    cfg = dict(opt.CONFIG, N_CELLS=a.cells, INIT=None)
    t0 = time.time()
    v = np.maximum(np.asarray(opt.search(a.budget, a.seed, cfg)["values"], dtype=np.float64), 0)
    v = v / v.max()
    np.save(a.out + ".npy", v)
    out = dict(n=len(v), best=score(v), seed=a.seed, budget=a.budget, seconds=time.time() - t0, config=cfg)
    json.dump(out, open(a.out + ".json", "w")); print(json.dumps(out))


def _refine_cli(a):
    from scipy.optimize import minimize
    v0 = np.repeat(np.maximum(load_vec(a.start), 0), a.split); v0 /= v0.max()
    n = len(v0); L = sfft.next_fast_len(2 * n - 1, real=True)
    t0 = time.time(); S0 = score(v0); best = {"s": S0, "v": v0.copy()}; st = {"nev": 0}

    class Stop(Exception):
        pass

    def obj(x, p):
        if time.time() - t0 > a.budget:
            raise Stop
        av = x * x
        F = sfft.rfft(av, L)
        c = sfft.irfft(F * F, L)[: 2 * n - 1]
        m = c.max(); sv = av.sum()
        s = 2 * n * m / sv ** 2
        st["nev"] += 1
        if s < best["s"]:
            best["s"] = s; best["v"] = av.copy()
        r = np.maximum(c, 0) / m
        with np.errstate(under="ignore"):
            rp1 = np.power(r, p - 1.0)
        Z = float(np.dot(rp1, r))
        val = np.log(m) + np.log(Z) / p - 2.0 * np.log(sv)
        wc = rp1 / (m * Z)
        ga = 2.0 * sfft.irfft(sfft.rfft(wc, L) * np.conj(F), L)[:n] - 2.0 / sv
        return val, ga * 2.0 * x                       # chain rule through a = x^2 (keeps a >= 0)

    x = np.sqrt(v0); p = a.p0; stages = []
    try:
        while p <= a.pmax * 1.0000001:
            res = minimize(obj, x, args=(p,), jac=True, method="L-BFGS-B",
                           options=dict(maxiter=a.iters, maxcor=30, ftol=0, gtol=0, maxls=40))
            x = res.x / np.sqrt(np.max(res.x ** 2))
            stages.append((p, res.nit, float(best["s"]), time.time() - t0))
            p *= 2.0
    except Stop:
        pass
    np.save(a.out + ".npy", best["v"])
    out = dict(start=a.start, n=n, split=a.split, p0=a.p0, start_score=float(S0), best=float(best["s"]),
               nev=st["nev"], seconds=time.time() - t0, stages=stages)
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "stages"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed"); s.add_argument("--cells", type=int, default=SEED_CELLS)
    s.add_argument("--budget", type=float, required=True); s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", required=True)
    r = sub.add_parser("refine"); r.add_argument("--start", required=True); r.add_argument("--out", required=True)
    r.add_argument("--split", type=int, default=1); r.add_argument("--p0", type=float, default=16.0)
    r.add_argument("--pmax", type=float, default=1e10); r.add_argument("--iters", type=int, default=3000)
    r.add_argument("--budget", type=float, default=1e9)
    a = ap.parse_args()
    _seed_cli(a) if a.cmd == "seed" else _refine_cli(a)
