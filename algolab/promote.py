"""Promotion gates: a plugin becomes a SAVANT template only after verified, practical gains.

Thresholds are absolute (practical size vs δ), not p-values. With n=3 the smallest exact one-sided sign-flip p is
0.125, so significance alone cannot authorize promotion. Records still go through selection/verify_claim.py.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DELTA = 1e-5
# Practical floors: far above float noise, far below δ for early tiers; T2/T3 demand more.
THRESHOLDS = {
    "T0": {"min_mean_gain": 1e-4, "min_paired_vs_control": 0.0, "require_all_nonneg": True},
    "T1": {"min_mean_gain": 1e-4, "min_paired_vs_control": 1e-4, "require_all_nonneg": True},
    "T1b": {"min_mean_gain": 1e-8, "min_paired_vs_control": 0.0, "require_all_nonneg": True},
    "T2": {"min_mean_gain": 1e-7, "min_paired_vs_control": 1e-8, "require_all_nonneg": True,
           "or_beats_control_by": 10.0},  # or mean_plugin >= 10 * mean_control when control > 0
    "T3": {"min_mean_gain": 1e-6, "min_paired_vs_control": 0.0, "require_all_nonneg": True,
           "fraction_of_delta": 0.1},  # mean gain >= 0.1 δ against the live/frozen start used in the wave
}


def decide(tier: str, plugin_gains: list[float | None], control_gains: list[float | None],
           *, delta: float = DELTA) -> dict:
    th = THRESHOLDS[tier]
    if any(g is None for g in plugin_gains) or len(plugin_gains) < 3:
        return {"promote": False, "reason": "incomplete plugin gains (need >= 3 seeds)", "tier": tier}
    if th.get("require_all_nonneg") and any(g < 0 for g in plugin_gains):
        return {"promote": False, "reason": "negative gain on a seed", "tier": tier}
    mean_p = sum(plugin_gains) / len(plugin_gains)
    if mean_p < th["min_mean_gain"]:
        return {"promote": False, "reason": f"mean gain {mean_p:.3e} < {th['min_mean_gain']:.3e}",
                "tier": tier, "mean_gain": mean_p}
    diffs = []
    if control_gains and all(g is not None for g in control_gains) and len(control_gains) == len(plugin_gains):
        diffs = [a - b for a, b in zip(plugin_gains, control_gains)]
        mean_d = sum(diffs) / len(diffs)
        mean_c = sum(control_gains) / len(control_gains)
        ok_pair = mean_d >= th["min_paired_vs_control"]
        if not ok_pair and th.get("or_beats_control_by") and mean_c > 0:
            ok_pair = mean_p >= th["or_beats_control_by"] * mean_c
        if not ok_pair:
            return {"promote": False, "reason": f"paired vs control {mean_d:.3e} below floor",
                    "tier": tier, "mean_gain": mean_p, "mean_paired_diff": mean_d}
    if th.get("fraction_of_delta") is not None and mean_p < th["fraction_of_delta"] * delta:
        return {"promote": False,
                "reason": f"mean gain {mean_p:.3e} < {th['fraction_of_delta']}*delta={th['fraction_of_delta']*delta:.3e}",
                "tier": tier, "mean_gain": mean_p}
    return {"promote": True, "reason": "practical thresholds met", "tier": tier, "mean_gain": mean_p,
            "mean_paired_diff": (sum(diffs) / len(diffs)) if diffs else None,
            "note": "Promotion to a hash-pinned template only; record claims still need verify_claim.py"}


def pin_as_template(plugin: Path, *, dest_dir: Path | None = None) -> dict:
    """Copy an admitted plugin into algolab/templates/ with its sha256 manifest entry (local only)."""
    plugin = Path(plugin)
    src = plugin.read_text(encoding="utf-8")
    sha = hashlib.sha256(src.encode()).hexdigest()
    dest_dir = Path(dest_dir or (ROOT / "algolab" / "templates"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / plugin.name
    dest.write_text(src, encoding="utf-8", newline="\n")
    man = dest_dir / "MANIFEST.jsonl"
    entry = {"plugin": plugin.name, "sha256": sha, "source": str(plugin)}
    with man.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry
