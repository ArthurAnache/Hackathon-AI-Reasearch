"""Deterministic problem.describe() → problem_report.json. No LLM."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import oaconvolve

from savant.config import PROBLEM, ROOT
from savant.ingest import validate_export


def _quantiles(x: np.ndarray, ps=(0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)) -> dict:
    qs = np.quantile(x, ps)
    return {f"q{int(p*100):02d}" if p > 0 else "min": float(v) for p, v in zip(ps, qs)} | {"max": float(qs[-1])}


def certificate_diagnostics(values: list[float]) -> dict:
    f = np.asarray(values, dtype=np.float64)
    f = np.maximum(f, 0.0)
    n = int(f.size)
    pos = f[f > 0]
    # autocorrelation c = f*f (same as verifier)
    c = oaconvolve(f, f, mode="full")
    m = int(c.size)
    M = float(c.max())
    near = float(np.mean(c >= M * (1 - 1e-7)))
    top1e9 = int(np.sum(c >= M * (1 - 1e-9)))
    # spectral energy of f
    F = np.fft.rfft(f)
    power = np.abs(F) ** 2
    power_share = power / power.sum()
    cum = np.cumsum(np.sort(power_share)[::-1])
    return {
        "n_values": n,
        "mean": float(f.mean()), "variance": float(f.var()), "std": float(f.std()),
        "sum": float(f.sum()), "max": float(f.max()), "min": float(f.min()),
        "sparsity": {"n_positive": int(pos.size), "frac_positive": float(pos.size / n),
                     "n_zero": int(n - pos.size)},
        "quantiles": _quantiles(f),
        "positive_quantiles": _quantiles(pos) if pos.size else {},
        "autocorrelation": {
            "len": m, "max": M, "mean": float(c.mean()), "std": float(c.std()),
            "frac_within_1e-7_of_max": near, "n_within_1e-9_of_max": top1e9,
            "argmax": int(np.argmax(c)),
            "note": "Incumbent described as equioscillating: many c points near max → dense updates fail.",
        },
        "spectrum": {
            "n_rfft": int(F.size),
            "energy_top1pct_modes": float(cum[max(1, int(0.01 * len(cum))) - 1]),
            "energy_top5pct_modes": float(cum[max(1, int(0.05 * len(cum))) - 1]),
            "energy_top10pct_modes": float(cum[max(1, int(0.10 * len(cum))) - 1]),
        },
    }


def build_problem_report(export_dir: Path | None = None, *, offline: bool = True) -> dict[str, Any]:
    """Identical for every researcher. Separates facts / assumptions / unknowns."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # offline: no live leaderboard (same as ws3_bench default)
    if offline:
        try:
            from integrations import einstein_arena as ea
            ea.get_leaderboard = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
        except Exception:
            pass

    from problems.catalog import load
    from certificates import candidate_sha

    p = load(PROBLEM)
    cert = p.certificate()
    score = float(p.score(cert))
    meta = json.loads((ROOT / "problems" / PROBLEM / "meta.json").read_text(encoding="utf-8"))
    sha = candidate_sha(cert)
    diag = certificate_diagnostics(cert["values"])

    hist = None
    if export_dir is not None:
        hist = validate_export(Path(export_dir))
        # drop bulky rows from embedded report (keep aggregates)
        hist = {k: v for k, v in hist.items() if k not in ("rows", "experiments", "summary_excerpt")}

    record = meta["record"]
    delta = float(meta["delta"])
    gap = delta - (hist["best"]["gain"] if hist else 0.0)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "savant.problem_report.build_problem_report",
        "reproducible": True,
        "llm_used": False,
        "problem": {
            "slug": PROBLEM,
            "title": meta["title"],
            "url": meta["url"],
            "direction": meta["direction"],
            "delta": delta,
            "formulation": {
                "objective": "Maximize C = ||f*f||_2^2 / (||f*f||_1 * ||f*f||_inf) over nonnegative step functions f",
                "candidate_schema": '{"values": [f_0,...,f_{n-1}]}, 1 <= n <= 2_000_000, values >= -1e-6 (clipped at 0)',
                "scale_invariance": "C(a f) = C(f) for a > 0",
                "dimension_n": diag["n_values"],
            },
            "constraints": meta["limits"],
            "verifier": {
                "file": meta["verifier"]["file"],
                "sha256": meta["verifier"]["sha256"],
                "commit": meta["verifier"]["commit"],
                "live_identical": meta["verifier"].get("live_identical"),
            },
        },
        "measured_facts": {
            "frozen_certificate_sha256": sha,
            "frozen_certificate_id": f"{PROBLEM}/{sha[:12]}",
            "verified_baseline_score": score,
            "displayed_freeze_score": record["value"],
            "baseline_vs_displayed_diff": score - float(record["value"]),
            "record_snapshot": {"value": record["value"], "agent": record["agent"],
                                "read_at": record["read_at"], "live": False},
            "distance_to_record_threshold": {
                "delta": delta,
                "target_score_at_freeze": float(record["value"]) + delta,
                "gain_needed_from_baseline": float(record["value"]) + delta - score,
                "best_observed_gain": hist["best"]["gain"] if hist else None,
                "gap_best_to_delta": gap if hist else None,
                "best_over_delta_ratio": (hist["best"]["gain"] / delta) if hist else None,
            },
            "certificate_distribution": diag,
            "historical_modal_campaign": hist,
        },
        "mathematical_assumptions": [
            "EinsteinArena C' definition matches the pinned verifier (float64 + oaconvolve).",
            "Scale invariance justifies normalizing f by max(f) inside optimizers.",
            "Independent exact verifier (Fraction/Decimal path) agrees with float within ~1e-15 on verified candidates.",
            "Forum method claims are untrusted until re-verified (digest is evidence of claims only).",
        ],
        "unknowns": [
            "Whether any local move class can reach +1e-5 from the 2M incumbent within hackathon compute.",
            "True sampling variability of probe/blockshift beyond n=3 matched seeds.",
            "Whether structural reformulations (packet/Dinkelbach/spectral) beat incremental block moves at n≈2M.",
            "Live leaderboard may have moved since freeze; claims use verify_claim.py.",
        ],
        "known_failure_modes": [
            "Dense gradient / Adam updates: measured no gain beyond ~5e-16 in 60s (equioscillating max set).",
            "Generic stage-3 local search: no problem reached delta in 10 min (WS1 stage 3).",
            "Upsampling ×2 exceeds the 2M submission limit.",
            "From-scratch constructions stall near ~0.90 (forum claims, unverified).",
        ],
        "search_opportunities": [
            "Exact incremental block moves with optimal step (blockshift) beat random probe on 3/3 seeds (+9.8e-9 paired).",
            "Throughput matters: blockshift ~9.5 proposals/s vs probe ~4.3 evals/s at 60s/1CPU.",
            "Guided blocks near max(c), D-grid / golden search, REFRESH checkpoints (blockshift CONFIG).",
            "Packet/run multipliers, spectral low-mode noise, Dinkelbach (forum claims; need vetted templates).",
            "Replication with holdout seeds 404/505/606 before promoting any strategy.",
        ],
        "optimizer_efficiency": (hist and {
            "probe_mean_gain": hist["probe"]["mean"],
            "blockshift_mean_gain": hist["blockshift"]["mean"],
            "paired_mean_diff_block_minus_probe": hist["paired_blockshift_vs_probe"]["mean_diff"],
            "paired_se": hist["paired_blockshift_vs_probe"]["se"],
            "statistical_caveat": hist["caveat"],
            "budget_s": 60.0, "cpu": 1, "memory_mb": 4096,
        }),
        "content_sha256": None,  # filled below excluding this field and generated_at
    }
    stable = {k: v for k, v in report.items() if k not in ("generated_at", "content_sha256")}
    blob = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    report["content_sha256"] = hashlib.sha256(blob).hexdigest()
    return report


def write_problem_report(out_dir: Path, export_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_problem_report(export_dir)
    path = out_dir / "problem_report.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    # human skim
    md = [
        f"# Problem report: {PROBLEM}",
        "",
        f"- Baseline verified score: `{report['measured_facts']['verified_baseline_score']!r}`",
        f"- Delta: `{report['problem']['delta']}`",
        f"- n: `{report['problem']['formulation']['dimension_n']}`",
        f"- content_sha256: `{report['content_sha256']}`",
        f"- LLM used: `{report['llm_used']}`",
        "",
        "## Measured facts / assumptions / unknowns are separated in problem_report.json.",
        "",
    ]
    (out_dir / "problem_report.md").write_text("\n".join(md), encoding="utf-8")
    return path
