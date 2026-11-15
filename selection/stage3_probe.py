"""WS1 stage 3: margin probe.

For each problem with decision keep_stage3 in registry_stage2.jsonl, run a generic optimizer for a fixed wall-clock
budget, starting from the live #1 solution, and measure how far it gets relative to delta (minImprovement).

Optimizer = multi-scale local perturbation + greedy acceptance (strict improvement only), with a final "polish"
phase using small steps. For step functions (`values` arrays) half of the budget is spent after doubling the
resolution (np.repeat). ring-loading-15 is handled in exact Fractions end to end and re-checked with a second,
independent exact implementation.

Every best solution is re-scored in a fresh process (eval_runner.py) with the pinned verifier before it is reported.
Nothing here claims a record: a gain >= delta is only a *candidate* that needs independent verification.

Usage: .venv/bin/python selection/stage3_probe.py [--budget 600] [--workers 9] [slug ...]
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import os
import subprocess
import sys
import time
from fractions import Fraction
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
STAGE2 = ROOT / "registry_stage2.jsonl"
CACHE2 = ROOT / ".cache" / "stage2"
CACHE3 = ROOT / ".cache" / "stage3"
OUT = ROOT / "registry_stage3.jsonl"

# Priority order decided by Arthur on 2026-11-14.
PRIORITY = [
    "ring-loading-15", "second-autocorrelation-inequality", "circles-rectangle", "erdos-min-overlap",
    "first-autocorrelation-inequality", "third-autocorrelation-inequality",
    "flat-polynomials", "kakeya-needle-128", "circle-packing", "tammes-problem", "thomson-problem",
    "min-distance-ratio-2d", "heilbronn-triangles", "difference-bases",
]


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def load_verifier(path: Path):
    ns: dict = {"__name__": "einstein_verifier"}
    exec(compile("import json\n" + path.read_text(), str(path), "exec"), ns)
    return ns["evaluate"]


# ---------------------------------------------------------------- adapters: solution <-> mutable state

class ArrayAdapter:
    """Float arrays: 1D step functions (`values`) or point sets (list of coordinate lists)."""

    def __init__(self, data: dict, key: str, rng, as_strings: bool = False, upper: float | None = None):
        import numpy as np
        self.np, self.data, self.key, self.rng, self.as_strings = np, data, key, rng, as_strings
        self.x = np.array([float(v) for v in data[key]] if as_strings else data[key], dtype=float)
        self.lower = 0.0 if self.x.min() >= 0 and self.x.ndim == 1 else None
        self.upper = upper
        self.mag = float(np.mean(np.abs(self.x))) or 1.0

    def mutate(self, polish: bool):
        np, rng, x = self.np, self.rng, self.x
        lo, hi = (-14, -6) if polish else (-12, -2)
        sigma = 10 ** rng.uniform(lo, hi) * self.mag
        if x.ndim == 1:
            size = min(len(x), 2 ** int(rng.integers(0, 11)))
            start = int(rng.integers(0, len(x) - size + 1))
            idx = slice(start, start + size)
        else:
            idx = slice(None) if rng.random() < 0.2 else int(rng.integers(0, len(x)))
        old = x[idx].copy()
        shape = x[idx].shape
        noise = rng.normal(0, sigma, shape) if (x.ndim > 1 or rng.random() < 0.5) else rng.normal(0, sigma)
        x[idx] = x[idx] + noise
        if self.lower is not None or self.upper is not None:
            x[idx] = np.clip(x[idx], self.lower, self.upper)

        def undo():
            x[idx] = old
        return undo

    def to_data(self) -> dict:
        vals = [repr(float(v)) for v in self.x] if self.as_strings else self.x.tolist()
        return {**self.data, self.key: vals}

    def upsampled(self):
        if self.x.ndim != 1 or self.key != "values":  # only step functions are upsampled
            return None
        clone = ArrayAdapter.__new__(ArrayAdapter)
        clone.__dict__.update(self.__dict__)
        clone.x = self.np.repeat(self.x, 2)
        return clone


class IntSetAdapter:
    def __init__(self, data: dict, key: str, rng):
        self.data, self.key, self.rng = data, key, rng
        self.x = list(data[key])
        self.span = max(self.x) + 1

    def mutate(self, polish: bool):
        i = int(self.rng.integers(0, len(self.x)))
        old = self.x[i]
        if self.rng.random() < 0.7:
            self.x[i] = max(0, old + int(self.rng.choice([-2, -1, 1, 2])))
        else:
            self.x[i] = int(self.rng.integers(0, self.span))

        def undo():
            self.x[i] = old
        return undo

    def to_data(self) -> dict:
        return {**self.data, self.key: sorted(self.x)}


class SignAdapter:
    def __init__(self, data: dict, key: str, rng):
        self.data, self.key, self.rng = data, key, rng
        self.x = list(data[key])

    def mutate(self, polish: bool):
        idx = self.rng.choice(len(self.x), size=int(self.rng.integers(1, 4)), replace=False)
        for i in idx:
            self.x[i] = -self.x[i]

        def undo():
            for i in idx:
                self.x[i] = -self.x[i]
        return undo

    def to_data(self) -> dict:
        return {**self.data, self.key: list(self.x)}


class RingAdapter:
    """Exact rational pairs (u, v) with u, v >= 0 and u + v <= 1; dyadic steps only."""

    def __init__(self, data: dict, key: str, rng):
        self.data, self.key, self.rng = data, key, rng
        self.x = [[Fraction(a), Fraction(b)] for a, b in data[key]]

    def mutate(self, polish: bool):
        rng = self.rng
        i = int(rng.integers(0, len(self.x)))
        old = list(self.x[i])
        m = int(rng.integers(8, 25)) if polish else int(rng.integers(2, 17))
        step = Fraction(int(rng.choice([-3, -2, -1, 1, 2, 3])), 2 ** m)
        u, v = self.x[i]
        move = rng.random()
        if move < 0.4:
            u = min(max(u + step, Fraction(0)), 1 - v)
        elif move < 0.8:
            v = min(max(v + step, Fraction(0)), 1 - u)
        else:  # transfer mass between u and v, keeping u + v fixed
            t = min(max(step, -u), v)
            u, v = u + t, v - t
        self.x[i] = [u, v]

        def undo():
            self.x[i] = old
        return undo

    def to_data(self) -> dict:
        return {**self.data, self.key: [[str(u), str(v)] for u, v in self.x]}


def ring_score_exact(pairs: list[list[str]]) -> Fraction:
    """Second, independent implementation of the ring-loading score, in exact rationals."""
    us = [Fraction(u) for u, _ in pairs]
    vs = [Fraction(v) for _, v in pairs]
    assert all(u >= 0 and v >= 0 and u + v <= 1 for u, v in zip(us, vs))
    best = None
    for choice in itertools.product((0, 1), repeat=len(us)):
        z = [vs[i] if c else -us[i] for i, c in enumerate(choice)]
        total, prefix, worst = sum(z, Fraction(0)), Fraction(0), Fraction(0)
        for k in range(len(z)):
            prefix += z[k]
            worst = max(worst, abs(prefix - (total - prefix)))
        best = worst if best is None or worst < best else best
    return best


def make_adapter(slug: str, data: dict, rng):
    key = next(iter(data))
    if slug == "ring-loading-15":
        return RingAdapter(data, key, rng)
    if slug == "flat-polynomials":
        return SignAdapter(data, key, rng)
    if slug == "difference-bases":
        return IntSetAdapter(data, key, rng)
    if slug == "kakeya-needle-128":
        return ArrayAdapter(data, key, rng, as_strings=True)
    if slug == "erdos-min-overlap":
        return ArrayAdapter(data, key, rng, upper=1.0)
    return ArrayAdapter(data, key, rng)


# ---------------------------------------------------------------- search loop

def search(adapter, evaluate, direction: str, start_score: float, budget_s: float, t_origin: float) -> dict:
    better = (lambda a, b: a < b) if direction == "minimize" else (lambda a, b: a > b)
    best, n_evals, n_invalid, improvements = start_score, 0, 0, []
    t0 = time.monotonic()
    while (elapsed := time.monotonic() - t0) < budget_s:
        undo = adapter.mutate(polish=elapsed > 0.7 * budget_s)
        try:
            s = float(evaluate(adapter.to_data()))
            ok = math.isfinite(s)
        except Exception:
            ok = False
        n_evals += 1
        if ok and better(s, best):
            best = s
            improvements.append(round(time.monotonic() - t_origin, 2))
        else:
            n_invalid += not ok
            undo()
    return {"best": best, "n_evals": n_evals, "n_invalid": n_invalid, "improvement_times_s": improvements}


def clearance(new: float, old: float, direction: str) -> float:
    return old - new if direction == "minimize" else new - old


def probe(job: dict) -> dict:
    import numpy as np
    slug, budget_s, seed = job["slug"], job["budget_s"], job["seed"]
    rng = np.random.default_rng(seed)
    t_origin = time.monotonic()
    out_dir = CACHE3 / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    verifier_path = CACHE2 / slug / "verifier_pinned.py"
    evaluate = load_verifier(verifier_path)
    data = json.loads((CACHE2 / slug / "record_solution.json").read_text())
    direction, record, delta = job["direction"], job["record"], job["delta"]

    adapter = make_adapter(slug, data, rng)
    start = float(evaluate(adapter.to_data()))
    phases = {}
    up = adapter.upsampled() if hasattr(adapter, "upsampled") else None
    if up is not None:
        res_a = search(adapter, evaluate, direction, start, budget_s / 2, t_origin)
        phases["original_resolution"] = res_a
        up = adapter.upsampled()  # upsample the phase-A best
        try:
            up_start = float(evaluate(up.to_data()))
            up_ok = math.isfinite(up_start)
        except Exception as exc:
            up_ok, up_start = False, f"{type(exc).__name__}: {str(exc)[:200]}"
        phases["upsample_x2_start"] = up_start
        if up_ok:
            res_b = search(up, evaluate, direction, up_start, budget_s / 2, t_origin)
            phases["upsampled_x2"] = res_b
            if clearance(res_b["best"], res_a["best"], direction) > 0:
                adapter = up
        else:
            res_b = search(adapter, evaluate, direction, res_a["best"], budget_s / 2, t_origin)
            phases["original_resolution_cont"] = res_b
    else:
        phases["original_resolution"] = search(adapter, evaluate, direction, start, budget_s, t_origin)

    best_data = adapter.to_data()
    best_path = out_dir / "best.json"
    best_path.write_text(json.dumps(best_data))

    # Independent re-check in a fresh process with the pinned verifier.
    proc = subprocess.run([sys.executable, "-I", str(ROOT / "eval_runner.py"), str(verifier_path), str(best_path),
                           "1", "1"], capture_output=True, text=True, timeout=1200)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    recheck = json.loads(lines[-1]) if lines else {"ok": False, "error": proc.stderr[-500:]}
    score = recheck.get("score") if recheck.get("ok") else None
    gain = clearance(score, record, direction) if score is not None else None

    extra = {}
    if slug == "ring-loading-15":
        exact = ring_score_exact(best_data["pairs"])
        extra = {"exact_score": f"{exact.numerator}/{exact.denominator}",
                 "exact_gain_vs_9_8": str(exact - Fraction(9, 8)),
                 "exact_matches_verifier": score is not None and float(exact) == score}

    n_impr = sum(len(p["improvement_times_s"]) for p in phases.values() if isinstance(p, dict))
    n_evals = sum(p["n_evals"] for p in phases.values() if isinstance(p, dict))
    wall = time.monotonic() - t_origin
    if gain is not None and gain >= delta and gain > 0:
        decision, why = "candidate_needs_independent_check", "Generic probe beat #1 by at least delta."
    elif gain is not None and gain > 0:
        decision, why = "headroom_below_delta", "Generic probe improved on #1, but by less than delta."
    else:
        decision, why = "no_gain", "Generic probe found no strict improvement over #1."
    return {
        "experiment_id": f"ws1-stage3-{slug}-{seed}",
        "slug": slug,
        "stage": 3,
        "checked_on": now_utc(),
        "hypothesis": "A generic local optimizer started from the #1 solution can gain at least delta in the budget.",
        "config": {"optimizer": "multiscale-perturbation+greedy+polish", "upsample_x2": up is not None,
                   "exact_rationals": slug == "ring-loading-15"},
        "seed": seed,
        "budget_s": budget_s,
        "valid": bool(recheck.get("ok")),
        "score": score,
        "score_repr": recheck.get("score_repr"),
        "best_known_score": record,
        "delta": delta,
        "direction": direction,
        "start_score_local": start,
        "gain": gain,
        "gain_over_delta": (gain / delta) if (gain is not None and delta) else None,
        "n_evals": n_evals,
        "n_improvements": n_impr,
        "improvements_per_min": n_impr / (wall / 60),
        "wall_s": wall,
        "phases": phases,
        "best_solution_path": str(best_path.relative_to(REPO_ROOT)),
        **extra,
        "decision": decision,
        "rationale": why,
        "evidence_urls": [f"https://einsteinarena.com/problems/{slug}"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=600)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("slugs", nargs="*")
    args = ap.parse_args()

    rows = {json.loads(l)["slug"]: json.loads(l) for l in STAGE2.read_text().splitlines() if l.strip()}
    keep = [s for s, r in rows.items() if r["decision"] == "keep_stage3"]
    slugs = args.slugs or sorted(keep, key=lambda s: PRIORITY.index(s) if s in PRIORITY else len(PRIORITY))
    jobs = [{"slug": s, "budget_s": args.budget, "seed": 1000 + i, "direction": rows[s]["direction"],
             "record": rows[s]["record"]["value"], "delta": rows[s]["delta"]} for i, s in enumerate(slugs)]

    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    existing = {}
    if OUT.exists():
        existing = {json.loads(l)["slug"]: l for l in OUT.read_text().splitlines() if l.strip()}
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(probe, jobs):
            existing[row["slug"]] = json.dumps(row, sort_keys=True)
            OUT.write_text("\n".join(existing.values()) + "\n")
            print(f"[stage3] {row['slug']}: {row['decision']} gain={row['gain']} gain/delta={row['gain_over_delta']} "
                  f"impr/min={row['improvements_per_min']:.2f} evals={row['n_evals']}", flush=True)


if __name__ == "__main__":
    main()
