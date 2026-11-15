"""Pack: EinsteinArena third-autocorrelation-inequality (C' of AlphaEvolve Problem 6.4; NOT Tao's 1d).

EinsteinArena verifier (origin/ws1-selection:selection/verifiers/third-autocorrelation-inequality.py), signed f on N
cells, dx = 0.5/N:  score = |max_k (f*f)_k dx| / (sum f dx)^2 = 2N |max_k (v*v)_k| / (sum v)^2  (one-sided max).
- seeds: the 1d seeds (WS2 1d_tuned.py minimizes max|v*v|, an upper bound of max(v*v)), kind seed1d;
- refine: split + p-norm continuation L-BFGS on the POSITIVE part of v*v (one-sided), kind refine3 (CLI below);
- certify: round to integers; exact convolution by attack/c1d_exact.autoconv_exact AND by the coordinator's
  check_independent.py (its coefficient list), which must agree at the maximum; direct sum at the deciding knot;
- reference: live EinsteinArena #1 minus minImprovement. Tao's 1d is a different constant (no Tao reference).
Nothing is ever submitted.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.fft as sfft

from . import tao1d as _d
from .safety import clean_env, safe_err

NAME = "ea-third-autocorrelation"
LOWER_IS_BETTER = True
SEED_CELLS, SPLIT_UP, LOCAL_MAX_CELLS = 800, 4, 12800
P0_SCHEDULE, WIDE_P0_EARLY, WIDE_P0_FINAL, SOFT_RESTARTS = _d.P0_SCHEDULE, [8.0], [4.0, 8.0], [4.0, 8.0, 2.0, 16.0]
MODAL_KINDS = dict(seed="seed1d", refine="refine3")
METHOD_NOTES = "One-sided objective: only the largest POSITIVE value of v*v counts; negative lobes are free."
EA, SLUG = "https://einsteinarena.com", "third-autocorrelation-inequality"
EA_DELTA = None
vec_id, load_vec = _d.vec_id, _d.load_vec
PY = sys.executable


def score(v) -> float:
    v = np.asarray(v, dtype=np.float64); n = len(v); L = sfft.next_fast_len(2 * n - 1, real=True)
    c = sfft.irfft(sfft.rfft(v, L) ** 2, L)[: 2 * n - 1]
    return float(2 * n * abs(c.max()) / v.sum() ** 2)


def valid_vec(v) -> bool:
    return len(v) >= 50 and bool(np.isfinite(v).all()) and abs(v.sum()) > 1e-9


def _get(path):
    return json.loads(urllib.request.urlopen(urllib.request.Request(EA + path, headers={"User-Agent": "autoloop/1.0"}),
                                             timeout=30).read())


def fetch_reference() -> dict:
    global EA_DELTA
    try:
        prob = _get(f"/api/problems/{SLUG}")
        board = _get(f"/api/leaderboard?problem_id={prob['id']}&limit=3")
        EA_DELTA = float(prob["minImprovement"])
        return dict(ok=True, value=repr(float(board[0]["bestScore"])), delta=EA_DELTA, leader=board[0]["agentName"],
                    source=f"{EA}/api/leaderboard?problem_id={prob['id']}",
                    candidates=[dict(value=repr(float(b["bestScore"])), agent=b["agentName"]) for b in board])
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, errors=[safe_err(repr(e))])


def claim_threshold(ref_value: str) -> Fraction:
    delta = EA_DELTA if EA_DELTA is not None else float(_get(f"/api/problems/{SLUG}")["minImprovement"])
    return Fraction(ref_value) - Fraction(delta)


def exact(ints):
    sys.path.insert(0, str(_d.ATTACK))
    import c1d_exact
    conv = c1d_exact.autoconv_exact(ints)
    S = sum(ints)
    assert sum(conv) == S * S
    k = max(range(len(conv)), key=lambda i: conv[i])
    n = len(ints)
    assert sum(ints[i] * ints[k - i] for i in range(max(0, k - n + 1), min(k, n - 1) + 1)) == conv[k]
    return Fraction(2 * n * abs(conv[k]), S * S), k, conv[k]


class LocalExecutor(_d.LocalExecutor):
    def certify(self, parent_path, tag):
        v = load_vec(parent_path)
        ints = [int(x) for x in np.rint(v / np.abs(v).max() * 2.0 ** 56)]
        out = str(self.dir / "certs" / f"{tag}_int56.json")
        json.dump({"values": ints}, open(out, "w"), separators=(",", ":"))
        f1, k, ck = exact(ints)
        code = _d.RUN_PINNED + "print('MAXC', max(g['c']))"
        r = subprocess.run([PY, "-I", "-c", code, out], input=_d.pinned_checker_source(), capture_output=True, text=True,
                           timeout=3600, env=clean_env())
        m = [ln for ln in r.stdout.splitlines() if ln.startswith("MAXC ")]
        if not m or int(m[0].split()[1]) != ck:
            return dict(ok=False, error="independent convolution disagrees at the max: " + safe_err(r.stderr)), 0.0
        return dict(ok=True, path=out, ints=ints, exact=f1, knot=k, checks=["c1d_exact.autoconv_exact", "check_independent"]), 0.0


class FakeExecutor(_d.FakeExecutor):
    pass


def _refine_cli(a):
    from scipy.optimize import minimize
    v0 = np.repeat(load_vec(a.start), a.split); v0 /= np.abs(v0).max()
    n = len(v0); L = sfft.next_fast_len(2 * n - 1, real=True)
    t0 = time.time(); S0 = score(v0); best = {"s": S0, "v": v0.copy()}; st = {"nev": 0}

    class Stop(Exception):
        pass

    def obj(v, p):
        if time.time() - t0 > a.budget:
            raise Stop
        F = sfft.rfft(v, L); c = sfft.irfft(F * F, L)[: 2 * n - 1]; sv = v.sum()
        s = 2 * n * abs(c.max()) / sv ** 2; st["nev"] += 1
        if s < best["s"]:
            best["s"] = s; best["v"] = v.copy()
        cp = np.maximum(c, 0); m = cp.max()
        r = cp / m
        with np.errstate(under="ignore"):
            rp1 = np.power(r, p - 1.0)
        Z = float(np.dot(rp1, r))
        val = np.log(m) + np.log(Z) / p - 2.0 * np.log(abs(sv))
        wc = rp1 / (m * Z)
        g = 2.0 * sfft.irfft(sfft.rfft(wc, L) * np.conj(F), L)[:n] - 2.0 / sv
        return val, g

    v = v0.copy(); p = a.p0
    try:
        while p <= 1e10:
            res = minimize(obj, v, args=(p,), jac=True, method="L-BFGS-B",
                           options=dict(maxiter=3000, maxcor=30, ftol=0, gtol=0, maxls=40))
            v = res.x / np.abs(res.x).max(); p *= 2.0
    except Stop:
        pass
    np.save(a.out + ".npy", best["v"])
    json.dump(dict(n=n, split=a.split, p0=a.p0, start_score=float(S0), best=float(best["s"]), nev=st["nev"],
                   seconds=time.time() - t0), open(a.out + ".json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refine"); r.add_argument("--start", required=True); r.add_argument("--out", required=True)
    r.add_argument("--split", type=int, default=1); r.add_argument("--p0", type=float, default=8.0)
    r.add_argument("--budget", type=float, default=300)
    _refine_cli(ap.parse_args())
