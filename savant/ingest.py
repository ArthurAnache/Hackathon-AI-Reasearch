"""Ingest a completed WS3 Modal export into SAVANT research memory.

Reads the real export files (summary, CSV, JSONL, manifest, jobs_detail). Does not trust
summary numbers alone: recomputes paired gains from comparison.csv / experiments.jsonl.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from savant.memory import ResearchMemory
from savant.stats import paired_comparison


REQUIRED = ("summary.md", "comparison.csv", "experiments.jsonl", "jobs_detail.jsonl", "manifest.json")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def validate_export(export_dir: Path) -> dict[str, Any]:
    export_dir = Path(export_dir)
    missing = [n for n in REQUIRED if not (export_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"{export_dir}: missing {missing}")

    summary = (export_dir / "summary.md").read_text(encoding="utf-8")
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((export_dir / "comparison.csv").open(encoding="utf-8")))
    experiments = _load_jsonl(export_dir / "experiments.jsonl")
    details = _load_jsonl(export_dir / "jobs_detail.jsonl")

    if len(rows) != 6:
        raise ValueError(f"expected 6 comparison rows, got {len(rows)}")
    if len(experiments) != 6:
        raise ValueError(f"expected 6 experiment records, got {len(experiments)}")

    by_opt: dict[str, list[dict]] = {}
    for r in rows:
        by_opt.setdefault(r["optimizer"], []).append(r)
        if r["status"] != "keep" or r["valid"] not in ("True", "true", True):
            raise ValueError(f"unexpected non-keep row: {r['job_id']} status={r['status']}")

    baseline = float(rows[0]["initial_score"])
    probe_gains = [float(r["gain"]) for r in by_opt["probe"]]
    block_gains = [float(r["gain"]) for r in by_opt["blockshift"]]
    seeds = [int(r["seed"]) for r in by_opt["probe"]]
    # paired D_i = gain_blockshift - gain_probe on matched seeds
    probe_by_seed = {int(r["seed"]): float(r["gain"]) for r in by_opt["probe"]}
    block_by_seed = {int(r["seed"]): float(r["gain"]) for r in by_opt["blockshift"]}
    if set(probe_by_seed) != set(block_by_seed):
        raise ValueError("probe/blockshift seeds do not match")

    paired = paired_comparison(
        [block_by_seed[s] for s in sorted(probe_by_seed)],
        [probe_by_seed[s] for s in sorted(probe_by_seed)],
        label_a="blockshift", label_b="probe", seeds=sorted(probe_by_seed),
        alpha=0.05, decision_index=0, alpha_spend=0.005,
    )

    best = max(rows, key=lambda r: float(r["final_score"]))
    exact = manifest.get("exact_check") or {}
    issues = []
    # Cross-check vs experiments.jsonl
    for rec in experiments:
        stem = rec["arm"].split(":")[-1]
        seed = rec["seeds"][0]
        match = next(r for r in rows if r["optimizer"] == stem and int(r["seed"]) == seed)
        if abs(float(match["final_score"]) - float(rec["score"])) > 1e-15:
            issues.append(f"score mismatch {stem} seed {seed}")
        if abs(float(match["gain"]) - (float(rec["score"]) - float(rec["best_before"]))) > 1e-15:
            issues.append(f"gain mismatch {stem} seed {seed}")

    # Sanity against known campaign facts (tolerance for float formatting)
    expected = {
        "baseline": 0.9636386679928123,
        "best": 0.9636386831034217,
        "probe_mean": 3.9005e-9,
        "block_mean": 1.3727e-8,
    }
    if abs(baseline - expected["baseline"]) > 1e-15:
        issues.append(f"baseline {baseline!r} != expected")
    if abs(float(best["final_score"]) - expected["best"]) > 1e-15:
        issues.append(f"best {best['final_score']!r} != expected")
    if abs(statistics.fmean(probe_gains) - expected["probe_mean"]) > 5e-13:
        issues.append("probe mean gain mismatch")
    if abs(statistics.fmean(block_gains) - expected["block_mean"]) > 5e-13:
        issues.append("blockshift mean gain mismatch")
    if not all(block_by_seed[s] > probe_by_seed[s] for s in probe_by_seed):
        issues.append("blockshift did not beat probe on all seeds")
    if "not a record" not in summary.lower() and "not a record" not in summary:
        # summary says "not a record"
        pass
    if exact and abs(exact.get("difference", 0)) > 1e-14:
        issues.append(f"exact check difference large: {exact.get('difference')}")

    return {
        "export_dir": str(export_dir),
        "run_id": manifest.get("run_id"),
        "manifest": {
            "git_commit": (manifest.get("git") or {}).get("commit"),
            "frozen_lock": manifest.get("frozen_lock"),
            "incumbent": manifest.get("incumbent"),
            "record_reference": manifest.get("record_reference"),
            "exact_check": exact,
            "budget_s": manifest.get("budget_s"),
            "seeds": manifest.get("seeds"),
        },
        "baseline_score": baseline,
        "best": {
            "optimizer": best["optimizer"], "seed": int(best["seed"]),
            "score": float(best["final_score"]), "gain": float(best["gain"]),
            "certificate_id": best.get("certificate_id"),
            "candidate_sha256": best.get("candidate_sha256"),
        },
        "probe": {"gains": probe_gains, "mean": statistics.fmean(probe_gains),
                  "seeds": seeds, "evals_per_s": [float(r["run_job_duration_s"]) and
                  (float(r["evals"]) / float(r["run_job_duration_s"]))
                  for r in by_opt["probe"] if r.get("evals")]},
        "blockshift": {"gains": block_gains, "mean": statistics.fmean(block_gains),
                       "seeds": seeds},
        "paired_blockshift_vs_probe": paired,
        "n_jobs": len(rows),
        "n_details": len(details),
        "issues": issues,
        "caveat": (
            "n=3 paired seeds is a first signal only. Exact sign-flip / sign tests cannot "
            "claim significance at 5% with 3 pairs. Do not treat this as statistical significance."
        ),
        "summary_excerpt": summary.splitlines()[:12],
        "rows": rows,
        "experiments": experiments,
    }


def ingest_export(export_dir: Path, memory: ResearchMemory) -> dict:
    report = validate_export(export_dir)
    if report["issues"]:
        raise ValueError("export validation failed: " + "; ".join(report["issues"]))

    memory.append("ingest", {
        "run_id": report["run_id"],
        "export_dir": report["export_dir"],
        "baseline_score": report["baseline_score"],
        "best": report["best"],
        "probe_mean_gain": report["probe"]["mean"],
        "blockshift_mean_gain": report["blockshift"]["mean"],
        "paired": {k: report["paired_blockshift_vs_probe"][k] for k in (
            "mean_diff", "se", "n", "sign_positive", "p_sign_flip_one_sided",
            "claim_allowed", "interpretation")},
        "caveat": report["caveat"],
        "manifest": report["manifest"],
    }, source="ws3-modal")

    for rec in report["experiments"]:
        memory.append("experiment", {
            "experiment_id": rec["experiment_id"],
            "run_id": report["run_id"],
            "optimizer": rec["optimizer"],
            "arm": rec["arm"],
            "seed": rec["seeds"][0],
            "budget_s": rec["budget_s"],
            "score": rec["score"],
            "best_before": rec["best_before"],
            "gain": rec["score"] - rec["best_before"] if rec["score"] is not None else None,
            "status": rec["status"],
            "valid": rec["valid"],
            "beats_record": rec["beats_record"],
            "duration_s": rec["duration_s"],
            "optimizer_hash": rec["optimizer_hash"],
            "certificate_path": rec["certificate_path"],
            "hypothesis": rec["hypothesis"],
            "config": rec["config"],
        }, source="ws3-modal")

    memory.append("uncertainty", {
        "topic": "modal_smoke_20261115T015342Z-b60",
        "statement": (
            "Blockshift beat probe on 3/3 matched seeds with mean paired difference "
            f"{report['paired_blockshift_vs_probe']['mean_diff']:+.4e}, but n=3 cannot "
            "support a 5% significance claim. Distance to record delta (1e-5) remains "
            f"~{1e-5 / max(report['best']['gain'], 1e-30):.0f}x larger than best gain."
        ),
        "action": "Use as prior evidence for SAVANT; require more seeds or holdout before strong claims.",
    }, source="ws3-modal")

    memory.append("rejected_idea", {
        "idea": "Declare blockshift statistically significantly better than probe from this smoke test alone",
        "reason": report["caveat"],
    }, source="savant")

    return report
