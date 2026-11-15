"""Problem scout experiment on third-autocorrelation-inequality: annealed log-sum-exp "melt and re-freeze" hops.

Usage (from the repo root, after `selection/scout_refresh.py` has cached the live problem):

    .venv/bin/python selection/scout_third_autocorr.py scan   --start <solution.json>
    .venv/bin/python selection/scout_third_autocorr.py refine --start <solution.json> --tag NAME [--upsample 2]
    .venv/bin/python selection/scout_third_autocorr.py chain  --start <solution.json> --budget 300 --workers 4 --tag NAME

`scan` prints where one cooling path per starting temperature lands (nothing saved). `refine` runs one timed
cooling path, saves the best point and re-scores it with the pinned verifier. `chain` runs re-melt hops in parallel.
`compare --ref A B C ...` prints how close the coarse shapes of solutions are (same basin or not).

What it measures: how far a *targeted* method gets in minutes from a public solution, as a fraction of the
EinsteinArena threshold delta (1e-5). The method is the one the MILO paper (arXiv:2609.38349, App. E.2, third
listing) describes for this problem: a log-sum-exp surrogate of the max of the autoconvolution, its FFT gradient,
L-BFGS stages on a geometric cooling ladder, and a chain of hops that re-melt the best point with multiplicative
jitter. This is our own re-implementation from that description, not their code (which is not public).

Scoring. The search uses an FFT autoconvolution (relative error about 1e-14). Every reported final score is
recomputed in a clean process (`selection/eval_runner.py`) with the verifier served by the live API, after
checking that its sha256 equals the pinned one recorded at stage 2 (`registry_stage2.jsonl`). Nothing is submitted.

Outputs: one JSON line per run appended to selection/scout_third_autocorr_runs.jsonl; best solutions under
selection/.cache/scout/third-autocorrelation-inequality/runs/<tag>/ (git-ignored).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

# One thread per process, always. Not setdefault: a Modal container with cpu=48 exports OMP_NUM_THREADS=48 and
# OPENBLAS_NUM_THREADS=48, and 48 processes with 48 threads each ran this objective about 250 times slower
# (measured: 541 ms per evaluation instead of about 2 ms).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import scipy.fft as sfft  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parent
SLUG = "third-autocorrelation-inequality"
CACHE = ROOT / ".cache" / "scout" / SLUG
OUT = ROOT / "scout_third_autocorr_runs.jsonl"
DELTA = 1e-5


# ----------------------------------------------------------------------------------------------------------------
# Verifier (live source, checked against the pinned hash)

def pinned_verifier_path() -> tuple[Path, str]:
    """Write the live verifier source to the cache and return (path, sha256). Raises if it is not the pinned one."""
    problem = json.loads((CACHE / "problem.json").read_text())
    source = problem["verifier"]
    sha = hashlib.sha256(source.encode()).hexdigest()
    pinned = None
    for line in (ROOT / "registry_stage2.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("slug") == SLUG:
            pinned = row.get("verifier_pinned_sha256")
    if sha != pinned:
        raise RuntimeError(f"live verifier sha256 {sha} differs from the pinned one {pinned}")
    path = CACHE / "verifier_live.py"
    path.write_text(source)
    return path, sha


def verify_clean(solution_path: Path) -> dict:
    """Score a solution file with the pinned verifier in a clean process (same harness as stage 2)."""
    verifier, sha = pinned_verifier_path()
    cmd = [sys.executable, "-I", str(ROOT / "eval_runner.py"), str(verifier), str(solution_path), "1", "600"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        return {"ok": False, "error": proc.stderr[-500:], "verifier_sha256": sha}
    out = json.loads(lines[-1])
    out["verifier_sha256"] = sha
    return out


# ----------------------------------------------------------------------------------------------------------------
# Objective

class Prob:
    """score(f) = 2n * max(conv(f, f)) / sum(f)^2, the verifier's formula (dx = 0.5 / n)."""

    def __init__(self, n: int):
        self.n = n
        self.L = 2 * n - 1
        self.N = sfft.next_fast_len(self.L, real=True)
        self.scale = 2.0 * n

    def conv(self, f: np.ndarray):
        sp = sfft.rfft(f, self.N)
        return sp, sfft.irfft(sp * sp, self.N)[: self.L]

    def score(self, f: np.ndarray) -> float:
        s = float(f.sum())
        return self.scale * float(self.conv(f)[1].max()) / (s * s)

    def val_grad(self, f: np.ndarray, temp: float):
        """Log-sum-exp surrogate at temperature `temp` (in score units), its gradient, and the exact score."""
        s = float(f.sum())
        if not math.isfinite(s) or s * s < 1e-9:
            return math.inf, None, math.inf
        sp, u = self.conv(f)
        k = self.scale / (s * s)
        c = u * k
        cmax = float(c.max())
        e = np.exp((c - cmax) * (1.0 / temp))
        z = float(e.sum())
        val = cmax + temp * math.log(z)
        w = e * (1.0 / z)
        g = sfft.irfft(sfft.rfft(w, self.N) * np.conj(sp), self.N)[: self.n] * (2.0 * k)
        g -= 2.0 * float(w @ c) / s
        return val, g, cmax


class Ratchet:
    """Keeps the best exact score seen and the point that gave it."""

    def __init__(self):
        self.best = math.inf
        self.x = None

    def offer(self, x: np.ndarray, score: float) -> bool:
        if score < self.best:
            self.best, self.x = score, x.copy()
            return True
        return False


def lbfgs_stage(prob: Prob, f: np.ndarray, temp: float, iters: int, rat: Ratchet, deadline: float) -> np.ndarray:
    class Stop(Exception):
        pass

    last = {"x": f}

    def fun(x):
        if time.monotonic() > deadline:
            raise Stop
        val, g, cmax = prob.val_grad(x, temp)
        if g is None:
            raise Stop
        rat.offer(x, cmax)
        last["x"] = x.copy()
        return val, g

    try:
        res = minimize(fun, f, jac=True, method="L-BFGS-B",
                       options={"maxiter": iters, "maxcor": 16, "ftol": 0.0, "gtol": 0.0, "maxls": 30})
        x = res.x
    except Stop:
        x = last["x"]
    s = float(x.sum())
    return x * (prob.n / s) if math.isfinite(s) and s != 0.0 else f


def run_path(prob: Prob, f: np.ndarray, t_hot: float, t_cold: float, stages: int, iters: int, rat: Ratchet,
             deadline: float, log: list | None = None, timed: bool = False) -> np.ndarray:
    """Geometric cooling t_hot -> t_cold over `stages` stages. Each stage gets `iters / stages` L-BFGS iterations,
    or, with `timed`, an equal share of the time left until `deadline` (so the cold stages always run)."""
    stages = max(2, stages)
    per = max(3, round(iters / stages))
    ratio = (t_cold / t_hot) ** (1.0 / (stages - 1))
    temp = t_hot
    for j in range(stages):
        now = time.monotonic()
        if now > deadline:
            break
        stage_deadline = now + (deadline - now) / (stages - j) if timed else deadline
        f = lbfgs_stage(prob, f, temp, 10 ** 9 if timed else per, rat, stage_deadline)
        if log is not None:
            log.append((temp, prob.score(f)))
        temp *= ratio
    return f


# ----------------------------------------------------------------------------------------------------------------
# Step pipelines (used by the Modal fleet, selection/scout_third_autocorr_modal.py)

MAX_VALUES = 100_000  # zod schema of the problem at the pinned commit: values: z.array(num).min(1).max(100_000)


def resample(f: np.ndarray, m: int) -> np.ndarray:
    """Exact block average of the step function f on m equal cells (plain sum-pooling when m divides len(f))."""
    n = len(f)
    cum = np.concatenate([[0.0], np.cumsum(f)])
    edges = np.interp(np.linspace(0.0, n, m + 1), np.arange(n + 1), cum)
    return np.diff(edges) * (m / n)


def run_steps(f0: np.ndarray, steps: list[dict], seed: int, budget_s: float, on_progress=None):
    """Run a list of steps from f0 and return (best point, its FFT score, trace).

    Steps:
      {"op": "pool", "k": 2}                      sum-pool to a coarser grid (len must be a multiple of k)
      {"op": "resample", "n": 25600}              exact block average onto another grid
      {"op": "decimate", "k": 8}                  keep one value in k (what MILO's page did to its record)
      {"op": "repeat", "k": 2}                    block-repeat (preserves the score exactly)
      {"op": "path", "t_hot", "t_cold", "stages", "seconds", "sigma": 0, "keep_start": True}
                                                  one timed cooling path from the current best (+ relative noise);
                                                  with keep_start False the path's landing replaces the current
                                                  best even when it is worse (exploration)
      {"op": "hops", "t_hot": [lo, hi], "sigma": [lo, hi], "t_cold", "stages", "hop_seconds", "seconds": None}
                                                  re-melt hops with a ratchet, until `seconds` or the budget ends
    Noise is additive: sigma * mean|f| * N(0, 1) per cell. Grids above MAX_VALUES are refused.
    """
    rng = np.random.default_rng(seed)
    t0 = time.monotonic()
    deadline = t0 + budget_s
    f = np.asarray(f0, dtype=np.float64)
    prob = Prob(len(f))
    best = f * (prob.n / f.sum())
    best_score = prob.score(best)
    trace = [{"t": 0.0, "op": "start", "n": prob.n, "score": best_score}]

    def log(**kw) -> None:
        trace.append({"t": round(time.monotonic() - t0, 1), "n": prob.n, **kw})
        if on_progress is not None:
            on_progress(best, best_score, trace)

    def perturb(x: np.ndarray, sigma: float) -> np.ndarray:
        return x + sigma * np.abs(x).mean() * rng.standard_normal(len(x)) if sigma > 0.0 else x.copy()

    def draw(rng_range) -> float:
        lo, hi = rng_range
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) if hi > lo else float(lo)

    for step in steps:
        if time.monotonic() >= deadline:
            break
        op = step["op"]
        if op in ("pool", "repeat", "resample", "decimate"):
            if op == "pool":
                g = best.reshape(-1, step["k"]).mean(axis=1)
            elif op == "decimate":
                g = best[:: step["k"]].copy()
            elif op == "repeat":
                g = np.repeat(best, step["k"])
            else:
                g = resample(best, step["n"])
            if len(g) > MAX_VALUES:
                raise ValueError(f"{len(g)} values exceed the schema limit of {MAX_VALUES}")
            prob = Prob(len(g))
            best = g * (prob.n / g.sum())
            best_score = prob.score(best)
            log(op=op, score=best_score)
        elif op == "path":
            rat = Ratchet()
            keep = step.get("keep_start", True)
            if keep:
                rat.offer(best, best_score)
            end = min(deadline, time.monotonic() + step["seconds"])
            f_end = run_path(prob, perturb(best, step.get("sigma", 0.0)), step["t_hot"], step["t_cold"],
                             step["stages"], 0, rat, end, timed=True)
            landing = prob.score(f_end)
            accepted = rat.best < best_score
            if accepted or not keep:
                best, best_score = rat.x.copy(), rat.best
            log(op="path", t_hot=step["t_hot"], sigma=step.get("sigma", 0.0), landing=landing, accepted=accepted,
                score=best_score)
        elif op == "hops":
            end = min(deadline, time.monotonic() + step["seconds"]) if step.get("seconds") else deadline
            while time.monotonic() < end - 0.6 * step["hop_seconds"]:
                t_hot, sigma = draw(step["t_hot"]), draw(step["sigma"])
                rat = Ratchet()
                hop_end = min(end, time.monotonic() + step["hop_seconds"])
                f_end = run_path(prob, perturb(best, sigma), t_hot, step["t_cold"], step["stages"], 0, rat, hop_end,
                                 timed=True)
                landing = prob.score(f_end)
                accepted = rat.best < best_score
                if accepted:
                    best, best_score = rat.x.copy(), rat.best
                log(op="hop", t_hot=t_hot, sigma=sigma, landing=landing, accepted=accepted, score=best_score)
        else:
            raise ValueError(f"unknown step {op!r}")
    return best, best_score, trace


def run_job_to_dir(job: dict) -> dict:
    """One fleet job: load job["start_path"] (.npy), run its steps, checkpoint the best point and a JSON record
    into job["out_dir"]. Plain files only, so it can run in a child process of a Modal container."""
    import platform

    f0 = np.load(job["start_path"])
    start_sha = hashlib.sha256(f0.tobytes()).hexdigest()
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    state = {"last": 0.0}

    def save(best, score, trace, final=False) -> None:
        if not final and time.monotonic() - state["last"] < 120:
            return
        state["last"] = time.monotonic()
        np.save(out / f"{job['id']}.tmp.npy", best)
        os.replace(out / f"{job['id']}.tmp.npy", out / f"{job['id']}.npy")
        meta = {**{k: job[k] for k in ("id", "track", "round", "start_key", "steps", "seed", "budget_s", "label",
                                       "provenance")},
                "start_sha256_float64": start_sha, "n": int(len(best)), "score_fft": float(score), "final": final,
                "wall_s": round(time.monotonic() - t0, 1), "trace": trace, "device": job.get("device", "CPU"),
                "cpu": platform.processor() or platform.machine(), "numpy": np.__version__}
        tmp = out / f"{job['id']}.tmp.json"
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, out / f"{job['id']}.json")

    best, score, trace = run_steps(f0, job["steps"], job["seed"], job["budget_s"], on_progress=save)
    save(best, score, trace, final=True)
    return {"id": job["id"], "score_fft": float(score), "n": int(len(best))}


def run_job_safe(job: dict) -> dict:
    """run_job_to_dir, but one failing job (for instance a grid above MAX_VALUES) does not take the batch down."""
    try:
        return run_job_to_dir(job)
    except Exception as exc:  # noqa: BLE001
        return {"id": job.get("id"), "error": f"{type(exc).__name__}: {exc}"[:300]}


# ----------------------------------------------------------------------------------------------------------------
# Hop chain (one worker)

def chain(job: dict) -> dict:
    cfg = job["cfg"]
    f0 = np.asarray(job["f0"], dtype=np.float64)
    prob = Prob(len(f0))
    rng = np.random.default_rng(job["seed"])
    t0 = time.monotonic()
    deadline = t0 + job["budget_s"]
    fin_reserve = cfg["fin_frac"] * job["budget_s"]
    f0 = f0 * (prob.n / f0.sum())
    start_score = prob.score(f0)

    # Warm level: the incumbent re-equilibrated at t_warm. Hops are compared at this level.
    warm = Ratchet()
    f_w = run_path(prob, f0, cfg["t_warm"] * 4, cfg["t_warm"], 3, cfg["warm_iters"], warm, deadline)
    warm_ref = prob.score(f_w)
    best_w, best_w_score = f_w.copy(), warm_ref
    trace = [{"t": round(time.monotonic() - t0, 1), "hop": 0, "warm_score": warm_ref}]
    hops = landed = 0
    while time.monotonic() < deadline - fin_reserve:
        alpha = math.exp(rng.uniform(math.log(cfg["alpha_lo"]), math.log(cfg["alpha_hi"])))
        sig = cfg["noise"] * min(1.0, alpha / cfg["alpha_hi"]) * math.exp(rng.uniform(-1.0, 1.0))
        cand = best_w * np.exp(sig * rng.standard_normal(prob.n))
        cand *= prob.n / cand.sum()
        scratch = Ratchet()
        f_h = run_path(prob, cand, alpha * cfg["t_scale"], cfg["t_warm"], cfg["hop_stages"], cfg["hop_iters"],
                       scratch, deadline - fin_reserve)
        sc = prob.score(f_h)
        hops += 1
        if sc < best_w_score:
            landed += 1
            best_w, best_w_score = f_h.copy(), sc
            trace.append({"t": round(time.monotonic() - t0, 1), "hop": hops, "alpha": alpha, "sig": sig,
                          "warm_score": sc})

    # Cold finish of the warm best (and of the untouched start, as the fair baseline of "polish only").
    cold = Ratchet()
    cold.offer(f0, start_score)
    run_path(prob, best_w, cfg["t_warm"], cfg["t_cold"], cfg["fin_stages"], cfg["fin_iters"], cold, deadline)
    return {"seed": job["seed"], "start_score": start_score, "warm_ref": warm_ref, "warm_best": best_w_score,
            "final_score": cold.best, "hops": hops, "landed": landed, "trace": trace,
            "wall_s": round(time.monotonic() - t0, 1), "best": cold.x.tolist()}


# ----------------------------------------------------------------------------------------------------------------

def load_start(path: Path, upsample: int = 1, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Load a solution; optionally block-repeat it (which preserves the score exactly) and add relative noise
    (the "block-repeat + noise + re-optimize" escape of forum thread 181)."""
    f = np.asarray(json.loads(path.read_text())["values"], dtype=np.float64)
    if upsample > 1:
        f = np.repeat(f, upsample)
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        f = f + noise * np.abs(f).mean() * rng.standard_normal(len(f))
    return f


def cmd_scan(args) -> None:
    """One cooling path per t_hot from the start: which temperatures move the solution, and where it lands."""
    f0 = load_start(Path(args.start), args.upsample, args.up_noise)
    prob = Prob(len(f0))
    f0 = f0 * (prob.n / f0.sum())
    print(f"n={prob.n} start score (fft) {prob.score(f0)!r}")
    for t_hot in args.t_hot:
        rat, log = Ratchet(), []
        t = time.monotonic()
        run_path(prob, f0, t_hot, args.t_cold, args.stages, args.iters, rat, t + args.budget, log)
        print(f"t_hot={t_hot:g} -> best exact {rat.best!r} ({time.monotonic() - t:.1f}s) "
              + " ".join(f"[{tt:.0e}:{sc:.9f}]" for tt, sc in log), flush=True)


def cmd_compare(args) -> None:
    """Are two solutions in the same basin? Correlation of their block averages on a common coarse grid."""
    def coarse(path: str) -> np.ndarray:
        v = load_start(Path(path))
        n = len(v)
        cum = np.concatenate([[0.0], np.cumsum(v)]) / v.sum()
        edges = np.interp(np.linspace(0, 1, args.cells + 1), np.linspace(0, 1, n + 1), cum)
        return np.diff(edges)

    ref = coarse(args.ref)
    for path in args.others:
        c = coarse(path)
        print(f"corr {np.corrcoef(ref, c)[0, 1]:+.4f}  mirrored {np.corrcoef(ref, c[::-1])[0, 1]:+.4f}  {path}")


def cmd_refine(args) -> None:
    """One timed cooling path from a start (optionally block-repeated), saved and re-scored by the pinned verifier."""
    start = Path(args.start)
    f0 = load_start(start, args.upsample, args.up_noise, args.seed)
    prob = Prob(len(f0))
    f0 = f0 * (prob.n / f0.sum())
    start_fft = prob.score(f0)
    print(f"start {start} upsample {args.upsample} n={prob.n} score (fft) {start_fft!r}", flush=True)
    rat, log = Ratchet(), []
    rat.offer(f0, start_fft)
    t0 = time.monotonic()
    run_path(prob, f0, args.t_hot, args.t_cold, args.stages, 0, rat, t0 + args.budget, log, timed=True)
    wall = time.monotonic() - t0
    run_dir = CACHE / "runs" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.json"
    best_path.write_text(json.dumps({"values": rat.x.tolist()}, separators=(",", ":")))
    v1 = verify_clean(best_path)
    row = {
        "experiment_id": f"scout-third-autocorr/{args.tag}", "slug": SLUG, "mode": "refine",
        "checked_on": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": args.hypothesis, "start": str(start), "upsample": args.upsample, "up_noise": args.up_noise,
        "n": prob.n, "config": {"t_hot": args.t_hot, "t_cold": args.t_cold, "stages": args.stages},
        "seed": args.seed, "budget_s": args.budget, "wall_s": round(wall, 1), "device": "local CPU, 1 process",
        "start_score_fft": start_fft, "score": v1.get("score"), "valid": bool(v1.get("ok")), "delta": DELTA,
        "verifier_sha256": v1.get("verifier_sha256"),
        "stage_scores": [[t, s] for t, s in log],
        "solution_path": str(best_path.relative_to(ROOT.parent)),
        "solution_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
        "label": "verified: final score recomputed in a clean process with the pinned verifier",
    }
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(" ".join(f"[{t:.0e}:{s:.9f}]" for t, s in log))
    print(f"best pinned score {v1.get('score_repr')} (fft {rat.best!r}) wall {wall:.0f}s -> {best_path}")


def cmd_chain(args) -> None:
    start = Path(args.start)
    f0 = load_start(start)
    v0 = verify_clean(start)
    print(f"start {start} n={len(f0)} pinned verifier score {v0.get('score_repr')}", flush=True)
    cfg = {"t_scale": args.t_scale, "alpha_lo": args.alpha_lo, "alpha_hi": args.alpha_hi, "t_warm": args.t_warm,
           "t_cold": args.t_cold, "noise": args.noise, "hop_stages": args.hop_stages, "hop_iters": args.hop_iters,
           "warm_iters": args.warm_iters, "fin_stages": args.fin_stages, "fin_iters": args.fin_iters,
           "fin_frac": args.fin_frac}
    jobs = [{"f0": f0, "seed": args.seed + i, "budget_s": args.budget, "cfg": cfg} for i in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(chain, jobs)
    wall = time.time() - t0
    best = min(results, key=lambda r: r["final_score"])
    run_dir = CACHE / "runs" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.json"
    best_path.write_text(json.dumps({"values": best["best"]}, separators=(",", ":")))
    v1 = verify_clean(best_path)
    gain = v0["score"] - v1["score"] if v0.get("ok") and v1.get("ok") else None
    row = {
        "experiment_id": f"scout-third-autocorr/{args.tag}", "slug": SLUG,
        "checked_on": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": "annealed log-sum-exp hops (MILO-style) find a better basin than the start within minutes",
        "start": str(start), "n": len(f0), "config": cfg, "seed": args.seed, "workers": args.workers,
        "budget_s": args.budget, "wall_s": round(wall, 1), "device": "local CPU",
        "start_score_pinned": v0.get("score"), "score": v1.get("score"), "valid": bool(v1.get("ok")),
        "gain": gain, "gain_over_delta": gain / DELTA if gain is not None else None, "delta": DELTA,
        "verifier_sha256": v1.get("verifier_sha256"),
        "solution_path": str(best_path.relative_to(ROOT.parent)),
        "solution_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
        "workers_detail": [{k: r[k] for k in ("seed", "start_score", "warm_ref", "warm_best", "final_score", "hops",
                                              "landed", "wall_s")} | {"trace": r["trace"][-12:]} for r in results],
        "label": "verified: final score recomputed in a clean process with the pinned verifier",
    }
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    for r in results:
        print(f"  seed {r['seed']}: start {r['start_score']:.12f} warm_ref {r['warm_ref']:.9f} "
              f"warm_best {r['warm_best']:.9f} final {r['final_score']:.12f} hops {r['hops']} landed {r['landed']}")
    print(f"best pinned score {v1.get('score_repr')}  gain {gain!r}  gain/delta {row['gain_over_delta']!r}  "
          f"wall {wall:.0f}s  -> {best_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    sc = sub.add_parser("scan")
    sc.add_argument("--start", required=True)
    sc.add_argument("--t-hot", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    sc.add_argument("--t-cold", type=float, default=1e-7)
    sc.add_argument("--stages", type=int, default=10)
    sc.add_argument("--iters", type=int, default=600)
    sc.add_argument("--budget", type=float, default=60.0)
    sc.add_argument("--upsample", type=int, default=1)
    sc.add_argument("--up-noise", type=float, default=0.0)
    sc.set_defaults(func=cmd_scan)
    cp = sub.add_parser("compare")
    cp.add_argument("--ref", required=True)
    cp.add_argument("--cells", type=int, default=400)
    cp.add_argument("others", nargs="+")
    cp.set_defaults(func=cmd_compare)
    rf = sub.add_parser("refine")
    rf.add_argument("--start", required=True)
    rf.add_argument("--tag", required=True)
    rf.add_argument("--hypothesis", default="a timed log-sum-exp cooling path improves the start")
    rf.add_argument("--upsample", type=int, default=1)
    rf.add_argument("--up-noise", type=float, default=0.0)
    rf.add_argument("--seed", type=int, default=0)
    rf.add_argument("--t-hot", type=float, default=1e-3)
    rf.add_argument("--t-cold", type=float, default=1e-8)
    rf.add_argument("--stages", type=int, default=14)
    rf.add_argument("--budget", type=float, default=240.0)
    rf.set_defaults(func=cmd_refine)
    ch = sub.add_parser("chain")
    ch.add_argument("--start", required=True)
    ch.add_argument("--tag", required=True)
    ch.add_argument("--budget", type=float, default=300.0)
    ch.add_argument("--workers", type=int, default=4)
    ch.add_argument("--seed", type=int, default=0)
    ch.add_argument("--t-scale", type=float, default=1e-3)
    ch.add_argument("--alpha-lo", type=float, default=0.1)
    ch.add_argument("--alpha-hi", type=float, default=1.0)
    ch.add_argument("--t-warm", type=float, default=1e-5)
    ch.add_argument("--t-cold", type=float, default=1e-8)
    ch.add_argument("--noise", type=float, default=1e-2)
    ch.add_argument("--hop-stages", type=int, default=6)
    ch.add_argument("--hop-iters", type=int, default=240)
    ch.add_argument("--warm-iters", type=int, default=200)
    ch.add_argument("--fin-stages", type=int, default=8)
    ch.add_argument("--fin-iters", type=int, default=2000)
    ch.add_argument("--fin-frac", type=float, default=0.25)
    ch.set_defaults(func=cmd_chain)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
