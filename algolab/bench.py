"""Benchmark ladder for LLM-written plugins against the trusted blockshift control (local CPU only, no paid calls).

    python -m algolab.bench selftest
    python -m algolab.bench ladder algolab/plugins/lse_lbfgs_spectral.py --seeds 101 202 303

Ladder (a plugin is promoted only if it passes each rung; every rung is paired against the control on the same
start, seeds, budget and harness):
  T0 contract  n = 4,096 random sparse start, 6 s: valid output, never below start, no crash, no deadline overrun.
  T1 climb     the incumbent average-pooled to n = 32,768 (a start with headroom), 20 s: can it climb?
  T2 kink      the frozen 2M incumbent, 60 s: can it improve the kinked record itself?
Scores come from the frozen pinned verifier in this (parent) process, never from the plugin.
Output: runs/algolab/<timestamp>/results.jsonl in the shared experiment JSONL format, plus extra fields.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algolab import gate, kit, promote  # noqa: E402
from problems.catalog import MAIN, load  # noqa: E402

CONTROL = ROOT / "algolab" / "plugins" / "control_blockshift.py"
TIERS = {
    "T0": {"n": 4096, "budget_s": 6.0},
    "T1": {"n": 32768, "budget_s": 20.0},
    "T1b": {"n": 4096, "budget_s": 30.0},  # proxy kink: start is a pre-polished local max (see start_for)
    "T2": {"n": None, "budget_s": 60.0},
}


def pinned(values: np.ndarray) -> float:
    return float(load(MAIN).score({"values": np.asarray(values, dtype=np.float64).tolist()}))


def pooled(f: np.ndarray, n: int) -> np.ndarray:
    edges = np.linspace(0, len(f), n + 1).astype(np.int64)
    return np.add.reduceat(f, edges[:-1]) / np.diff(edges)


def start_for(tier: str, run_dir: Path) -> Path:
    path = run_dir / f"start_{tier}.npy"
    if path.exists():
        return path
    if tier == "T0":
        rng = np.random.default_rng(0)
        f = rng.random(TIERS["T0"]["n"]) * (rng.random(TIERS["T0"]["n"]) < 0.3)
    elif tier == "T1b":
        # Proxy kink: average-pool the incumbent to n=4096, then greedily accept tiny block noise until stuck.
        # Escape from this point is a cheap predictor of T2 behaviour without a 2M convolution.
        inc = np.asarray(load(MAIN).certificate()["values"], dtype=np.float64)
        f = pooled(inc / inc.max(), TIERS["T1b"]["n"])
        rng = np.random.default_rng(7)
        cur = kit.score(f)
        for _ in range(200):
            a = int(rng.integers(0, len(f) - 8))
            g = f.copy()
            g[a:a + 8] *= 1.0 + 1e-3 * rng.normal()
            g = np.maximum(g, 0.0)
            s = kit.score(g)
            if s > cur:
                f, cur = g, s
    else:
        inc = np.asarray(load(MAIN).certificate()["values"], dtype=np.float64)
        inc = inc / inc.max()
        f = inc if TIERS[tier]["n"] is None else pooled(inc, TIERS[tier]["n"])
    np.save(path, f)
    return path


def run_job(plugin: Path, tier: str, seed: int, run_dir: Path, trusted: bool, sha: str | None,
            budget_s: float | None = None) -> dict:
    budget = budget_s or TIERS[tier]["budget_s"]
    job_dir = run_dir / f"{tier}_{plugin.stem}_s{seed}"
    job_dir.mkdir(parents=True, exist_ok=True)
    start = start_for(tier, run_dir)
    spec = {"job_dir": str(job_dir), "plugin": str(plugin), "plugin_sha256": sha, "trusted": trusted,
            "seed": seed, "budget_s": budget, "start_npy": str(start), "params": {}}
    (job_dir / "spec.json").write_text(json.dumps(spec))
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "TEMP", "TMP", "PATH") if k in os.environ}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.monotonic()
    try:
        proc = subprocess.run([sys.executable, "-I", str(ROOT / "algolab" / "child.py"), str(job_dir / "spec.json")],
                              cwd=job_dir, env=env, capture_output=True, text=True, timeout=budget * 1.5 + 60)
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}: {proc.stderr.strip()[-300:]}"
    except subprocess.TimeoutExpired:
        status = "killed: wall timeout"
    wall = time.monotonic() - t0
    trace = json.loads((job_dir / "trace.json").read_text()) if (job_dir / "trace.json").exists() else {}
    rec = {"tier": tier, "plugin": plugin.stem, "seed": seed, "budget_s": budget, "status": status,
           "wall_s": round(wall, 2), "errors": trace.get("errors", []), "improve_calls": trace.get("improve_calls"),
           "accepted": trace.get("accepted"), "overruns": trace.get("overruns")}
    f0 = np.load(start)
    rec["start_score"] = pinned(f0)
    if (job_dir / "best.npy").exists():
        best = np.load(job_dir / "best.npy")
        try:
            rec["score"] = pinned(best)
            rec["valid"] = True
        except Exception as e:  # noqa: BLE001
            rec.update(valid=False, score=None, invalid_reason=str(e)[:200])
        rec["n_values"] = int(best.size)
    else:
        rec.update(valid=False, score=None)
    rec["gain"] = (rec["score"] - rec["start_score"]) if rec.get("score") is not None else None
    return rec


def sign_flip_p(diffs: list[float]) -> float:
    """Exact one-sided paired sign-flip p-value for mean(diffs) > 0."""
    obs = float(np.mean(diffs))
    flips = [float(np.mean([s * d for s, d in zip(signs, diffs)]))
             for signs in itertools.product((1, -1), repeat=len(diffs))]
    return sum(m >= obs - 1e-300 for m in flips) / len(flips)


def ladder(plugin: Path, seeds: list[int], tiers: list[str], parallel: int) -> list[dict]:
    load(MAIN)  # catalog.load registers the module before executing it: never call it first from a worker thread
    adm = gate.admit(plugin)
    run_dir = ROOT / "runs" / "algolab" / time.strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = [{"kind": "admission", **adm}]
    print(json.dumps(adm))
    if not adm["admitted"]:
        return out
    for tier in tiers:
        jobs = [(plugin, False, adm["sha256"]), (CONTROL, True, None)]
        with cf.ThreadPoolExecutor(parallel) as ex:
            futs = {ex.submit(run_job, p, tier, s, run_dir, trusted, sha): (p, s)
                    for p, trusted, sha in jobs for s in seeds}
            recs = [f.result() for f in cf.as_completed(futs)]
        by = {(r["plugin"], r["seed"]): r for r in recs}
        diffs = []
        for s in seeds:
            a, b = by.get((plugin.stem, s)), by.get((CONTROL.stem, s))
            if a and b and a.get("gain") is not None and b.get("gain") is not None:
                diffs.append(a["gain"] - b["gain"])
        plugin_gains = [by.get((plugin.stem, s), {}).get("gain") for s in seeds]
        control_gains = [by.get((CONTROL.stem, s), {}).get("gain") for s in seeds]
        decision = promote.decide(tier, plugin_gains, control_gains)
        summary = {"kind": "tier_summary", "tier": tier, "plugin": plugin.stem, "n_pairs": len(diffs),
                   "plugin_gains": plugin_gains, "control_gains": control_gains,
                   "mean_paired_diff": float(np.mean(diffs)) if diffs else None,
                   "p_sign_flip_one_sided": sign_flip_p(diffs) if diffs else None,
                   "plugin_errors": sum(len(r["errors"]) for r in recs if r["plugin"] == plugin.stem),
                   "promote": decision["promote"], "promote_reason": decision["reason"],
                   "promote_detail": decision}
        for r in recs:
            r.update(experiment_id=f"algolab-{run_dir.name}-{r['tier']}-{r['plugin']}-s{r['seed']}",
                     hypothesis=f"plugin {r['plugin']} vs trusted blockshift control, tier {r['tier']}",
                     config={"plugin_sha256": adm["sha256"] if r["plugin"] == plugin.stem else "trusted-control"},
                     best_known_score=None, decision="benchmark_only",
                     rationale="ladder rung; not a record claim (records go through selection/verify_claim.py)",
                     evidence_urls=[], arm=r["plugin"], seed=int(r["seed"]))
        out += recs + [summary]
        print(json.dumps(summary))
        if not summary["promote"]:
            break
        if decision["promote"] and tier == "T2":
            out.append({"kind": "template_pinned", **promote.pin_as_template(plugin)})
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, default=str) + "\n")
    print("results:", run_dir / "results.jsonl")
    return out


def selftest() -> None:
    checks = {}
    good = gate.admit(ROOT / "algolab" / "plugins" / "lse_lbfgs_spectral.py")
    bad = gate.admit(ROOT / "algolab" / "plugins" / "hostile_example.py")
    checks["gate_admits_lse_plugin"] = good["admitted"]
    checks["gate_rejects_hostile_plugin"] = (not bad["admitted"], bad["reasons"])
    rng = np.random.default_rng(1)
    f = rng.random(3001) * (rng.random(3001) < 0.4)
    checks["kit_score_matches_pinned"] = abs(kit.score(f) - pinned(f))
    beta_abs = 1e3 / kit.autoconv(f).max()
    v, g = kit.smoothed_log_score(f, beta_abs)
    d = rng.normal(size=f.size) * (f > 0)
    eps = 1e-6
    fd = (kit.smoothed_log_score(f + eps * d, beta_abs)[0] - kit.smoothed_log_score(f - eps * d, beta_abs)[0]) / (2 * eps)
    checks["kit_gradient_rel_error"] = abs(fd - float(g @ d)) / max(abs(fd), 1e-30)
    run_dir = ROOT / "runs" / "algolab" / ("selftest-" + time.strftime("%Y%m%dT%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = run_job(ROOT / "algolab" / "plugins" / "runtime_probe.py", "T0", 1, run_dir, True, None, budget_s=8.0)
    checks["runtime_hook_refusals"] = rec["errors"]
    leaked = Path(r"C:\Users\Public\algolab_escape.txt").exists()
    checks["no_file_escaped"] = not leaked
    print(json.dumps(checks, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    lp = sub.add_parser("ladder")
    lp.add_argument("plugin")
    lp.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    lp.add_argument("--tiers", nargs="+", default=["T0", "T1", "T2"])
    lp.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        ladder(Path(a.plugin).resolve(), a.seeds, a.tiers, a.parallel)
