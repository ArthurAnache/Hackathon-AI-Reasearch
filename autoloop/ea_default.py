"""Default pack: any frozen EinsteinArena problem from its verifier, its format and its reference only.

    python -m autoloop.loop --problem ea:circles-rectangle --executor modal --max-cells 63 ...

Nothing here is specific to a problem. The pack reads, from WS1's frozen catalog (problems/<slug>/, read-only):
the pinned verifier (`score`, raises on an invalid solution), the independent exact verifier (`independent_score`),
the direction and delta; from the live EinsteinArena API: the #1 score (reference) and the public #1 solution.
- format: the solution JSON's one numeric array, flattened to a vector (shape kept); the vector is the "cells";
- score: the pinned verifier, negated when the problem maximizes (the loop always minimizes); invalid = +inf;
- seed ("adopt"): the public #1 plus a random direction, scaled back by bisection to the largest step the verifier
  still accepts, then the black-box search below. A random draw of the format is not used: for packing formats it
  is invalid or trivially small (Lupa, debate E3), so this run is an adopt start, not a search from noise;
- refine: (1+1) evolution strategy with the 1/5 success rule, black box (verifier calls only), initial step
  max|v| * 2**-p0; only verifier-accepted improvements are kept;
- certify: the pinned verifier and the independent exact verifier must agree (relative 1e-12); exact = its Fraction;
- reference: live #1; claim threshold = #1 improved by max(live minImprovement, frozen delta).
Nothing is ever submitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

import numpy as np

from .safety import safe_err

REPO = Path(__file__).resolve().parent.parent
EA = "https://einsteinarena.com"
SLUG = None
NAME = "ea:?"
LOWER_IS_BETTER = True
SPLIT_UP = 2                     # with --max-cells = dimension, every refine is "top level" (split 1) and certify is legal
P0_SCHEDULE = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
WIDE_P0_EARLY, WIDE_P0_FINAL, SOFT_RESTARTS = [8.0], [6.0, 10.0], [6.0, 10.0, 4.0, 16.0]
METHOD_NOTES = ("Default pack: black-box (1+1)-ES on the verifier only; p0 sets the initial step max|v|*2^-p0 "
                "(small p0 = large moves). Seeds adopt the public #1 and perturb it. Scores are negated sums when the "
                "problem maximizes, so lower is better everywhere.")
SEED_CELLS = LOCAL_MAX_CELLS = 0
SEED_NOTE = "adopted starts (public #1 plus a verifier-accepted random step, then black-box search)"
MODAL_KINDS = {}
_P = _KEY = _SHAPE = None


def configure(slug):
    global SLUG, NAME, _P, SEED_CELLS, LOCAL_MAX_CELLS, MODAL_KINDS, _KEY, _SHAPE
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from problems import catalog
    SLUG, NAME, _P = slug, f"ea:{slug}", catalog.load(slug)
    cert = _P.certificate()                                  # the frozen #1: used ONLY for the format (key, shape)
    _KEY = next(k for k, v in cert.items() if isinstance(v, list))
    _SHAPE = np.asarray(cert[_KEY], dtype=np.float64).shape
    SEED_CELLS = LOCAL_MAX_CELLS = int(np.prod(_SHAPE))
    MODAL_KINDS = dict(seed=f"pack_seed:{slug}", refine=f"pack_refine:{slug}")
    return sys.modules[__name__]


def to_candidate(v):
    return {_KEY: np.asarray(v, dtype=np.float64).reshape(_SHAPE).tolist()}


def score(v) -> float:
    try:
        s = float(_P.score(to_candidate(v)))
    except Exception:  # noqa: BLE001  (the pinned verifier raises on an invalid solution)
        return float("inf")
    if not np.isfinite(s):
        return float("inf")
    return -s if _P.DIRECTION == "maximize" else s


def valid_vec(v) -> bool:
    return len(v) == SEED_CELLS and bool(np.isfinite(v).all()) and np.isfinite(score(v))


def vec_id(v) -> str:
    return hashlib.sha256(np.asarray(v, dtype=np.float64).tobytes()).hexdigest()[:16]


def load_vec(path):
    return np.load(path).astype(np.float64).ravel()


def _get(path):
    return json.loads(urllib.request.urlopen(urllib.request.Request(EA + path, headers={"User-Agent": "autoloop/1.0"}),
                                             timeout=30).read())


def public_best():
    """The public #1 solution as a vector (live API)."""
    pid = _get(f"/api/problems/{SLUG}")["id"]
    best = _get(f"/api/solutions/best?problem_id={pid}&limit=1")[0]
    return np.asarray(best["data"][_KEY], dtype=np.float64).ravel(), best


_DELTA = None


def fetch_reference() -> dict:
    global _DELTA
    try:
        prob = _get(f"/api/problems/{SLUG}")
        board = _get(f"/api/leaderboard?problem_id={prob['id']}&limit=3")
        if not isinstance(board, list) or not board:
            raise ValueError("empty leaderboard")
        _DELTA = max(float(prob.get("minImprovement") or 0.0), float(_P.DELTA))
        sign = -1 if _P.DIRECTION == "maximize" else 1
        return dict(ok=True, value=repr(sign * float(board[0]["bestScore"])), delta=_DELTA, leader=board[0]["agentName"],
                    direction=_P.DIRECTION, source=f"{EA}/api/leaderboard?problem_id={prob['id']}",
                    candidates=[dict(value=repr(sign * float(b["bestScore"])), agent=b["agentName"]) for b in board])
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, errors=[safe_err(repr(e))])


def claim_threshold(ref_value: str) -> Fraction:
    if _DELTA is None:
        raise RuntimeError("no reference read yet (fails closed)")
    return Fraction(ref_value) - Fraction(_DELTA)


# ---------------------------------------------------------------- black-box stages (also run on Modal)
def es(v, p0, budget_s, rng):
    t0 = time.time(); s = score(v); n = len(v)
    sig = max(np.abs(v).max(), 1e-12) * 2.0 ** -p0; nev = 1; acc = 0
    while time.time() - t0 < budget_s and sig > 1e-17:
        k = rng.integers(1, n + 1) if rng.random() < 0.5 else n          # sparse or full moves
        idx = rng.choice(n, size=k, replace=False)
        w = v.copy(); w[idx] += sig * rng.standard_normal(k)
        sw = score(w); nev += 1
        if sw < s:
            v, s = w, sw; acc += 1; sig *= 1.5
        else:
            sig *= 1.5 ** -0.25
    return v, s, dict(nev=nev, accepted=acc, final_step=sig, seconds=time.time() - t0)


def seed_one(seed, budget_s):
    rng = np.random.default_rng(seed)
    v0, _ = public_best()
    d = rng.standard_normal(len(v0)) * np.abs(v0).max() * 10.0 ** -rng.uniform(2, 6)
    lo, hi = 0.0, 1.0                                                    # largest valid step along d (bisection)
    if np.isfinite(score(v0 + d)):
        lo = 1.0
    else:
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.isfinite(score(v0 + mid * d)) else (lo, mid)
    v, s, info = es(v0 + lo * d, 4.0 + 4 * rng.random(), budget_s, rng)
    return v, s, dict(info, step_kept=lo, start_score=score(v0))


class LocalExecutor:
    real = True
    usd_per_cpu_s = 0.0
    spent_usd = 0.0
    workers = 6

    def __init__(self, workdir, max_cells=None):
        self.dir = Path(workdir); self.max_cells = max_cells or SEED_CELLS
        (self.dir / "certs").mkdir(parents=True, exist_ok=True)

    def seed(self, n, budget_s, cells, seed0):
        rows = []
        for i in range(n):
            v, s, info = seed_one(seed0 + i, budget_s)
            p = self.dir / "certs" / f"seed_s{seed0 + i}.npy"; np.save(p, v)
            rows.append(dict(ok=True, path=str(p), reported=s, params=dict(seed=seed0 + i, cells=cells, origin="adopt",
                                                                          budget_s=budget_s, **info)))
        return rows, n * budget_s

    def refine(self, parent_path, split, p0s, budget_s, tag):
        return self.refine_many([parent_path], split, p0s, budget_s, tag)

    def refine_many(self, parent_paths, split, p0s, budget_s, tag):
        rows = []
        for i, pp in enumerate(parent_paths):
            for p0 in p0s:
                v, s, info = es(load_vec(pp), p0, budget_s, np.random.default_rng(abs(hash((tag, i, p0))) % 2**32))
                p = self.dir / "certs" / f"{tag}_{i:02d}_p{p0:g}.npy"; np.save(p, v)
                rows.append(dict(ok=True, pi=i, path=str(p), reported=s, params=dict(split=split, p0=p0, budget_s=budget_s, **info)))
        return rows, len(rows) * budget_s

    def certify(self, parent_path, tag):
        v = load_vec(parent_path); cand = to_candidate(v)
        out = self.dir / "certs" / f"{tag}_solution.json"; out.write_text(json.dumps(cand))
        cand = json.loads(out.read_text())                       # certify exactly the bytes on disk
        try:
            pinned = float(_P.score(cand))
            ind = _P.independent_score(cand)
        except Exception as e:  # noqa: BLE001
            return dict(ok=False, error="a verifier refused the solution: " + safe_err(repr(e))), 0.0
        fr = Fraction(ind["fraction"]) if not isinstance(ind["fraction"], Fraction) else ind["fraction"]
        if abs(float(fr) - pinned) > 1e-12 * max(1.0, abs(pinned)):
            return dict(ok=False, error=f"pinned {pinned!r} and independent {float(fr)!r} disagree"), 0.0
        ex = -fr if _P.DIRECTION == "maximize" else fr
        return dict(ok=True, path=str(out), ints=None, exact=ex, knot=None,
                    checks=["pinned verifier", "independent exact verifier"]), 0.0


class FakeExecutor(LocalExecutor):
    real = False

    def __init__(self, workdir, max_cells=None, seed=0):
        super().__init__(workdir, max_cells)

    def certify(self, parent_path, tag, parent_score=None):
        return super().certify(parent_path, tag)


if __name__ == "__main__":                                    # Modal job entry: seed / refine one job
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["seed", "refine", "stage"]); ap.add_argument("--slug", required=True)
    ap.add_argument("--code", help="stage: a model-written refine(v, score, budget_s, rng) (runs on Modal only)")
    ap.add_argument("--out", required=True); ap.add_argument("--budget", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--start"); ap.add_argument("--p0", type=float, default=8)
    a = ap.parse_args(); configure(a.slug)
    if a.cmd == "stage":
        import signal
        ns = {}
        exec(compile(open(a.code).read(), "model_stage.py", "exec"), ns)            # noqa: S102 (Modal container only)
        v0 = load_vec(a.start) if a.start else None                                 # None: from noise, no solution given
        s0 = score(v0) if v0 is not None else float("inf"); t0 = time.time()

        def _alarm(*_):
            raise TimeoutError("stage passed its budget")
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(int(a.budget * 1.2) + 30)
        try:
            w = np.asarray(ns["refine"](None if v0 is None else v0.copy(), score, a.budget, np.random.default_rng(a.seed)),
                           dtype=np.float64).ravel()
            err = None
        except Exception as e:  # noqa: BLE001
            w, err = (v0 if v0 is not None else np.zeros(SEED_CELLS)), repr(e)[:300]
        signal.alarm(0)
        sw = score(w) if w.shape == (SEED_CELLS,) else float("inf")
        v, s = (w, sw) if sw < s0 else (v0 if v0 is not None else w, s0)            # kept only if the verifier agrees
        info = dict(start_score=s0, stage_score=sw, kept=bool(sw < s0), error=err, seconds=time.time() - t0)
    elif a.cmd == "seed":
        v, s, info = seed_one(a.seed, a.budget)
    else:
        v, s, info = es(load_vec(a.start), a.p0, a.budget, np.random.default_rng(a.seed or int(a.p0 * 1000)))
        info["start_score"] = score(load_vec(a.start))
    np.save(a.out + ".npy", v)
    json.dump(dict(info, best=s, p0=a.p0), open(a.out + ".json", "w"))
