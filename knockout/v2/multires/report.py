"""Markdown tables of the coarse-to-fine runs, from the run folders downloaded from the Volume.

    modal volume get knockout-v2-multires <run_id>/summary.json knockout/v2/results/multires/<run_id>/summary.json
    modal volume get knockout-v2-multires <run_id>/levels.jsonl knockout/v2/results/multires/<run_id>/levels.jsonl
    python -m knockout.v2.multires.report knockout/v2/results/multires

Every score in the tables is the frozen verifier's float64 score computed on CPU inside the container.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _g(x) -> str:
    return "n/a" if x is None else f"{x:+.3e}"


def batches(root: Path) -> list[dict]:
    out = []
    for path in sorted(root.glob("*/summary.json")):
        s = json.loads(path.read_text())
        for b in s["batches"]:
            out.append({"run_id": s["run_id"], "gpu": s["env"]["gpu_type"], **b})
    return out


def batch_table(b: dict) -> str:
    sizes = " -> ".join(f"{n:,}" for n in b["level_sizes"])
    evals = " + ".join(str(e) for e in b["evals_done"])
    title = (f"**{b['batch']}** ({b['experiment']}), run `{b['run_id']}`: start `{b['start']}`, pooled by "
             f"{b['pool']}, levels {sizes} cells, evaluations {evals}"
             + (", stopped at the wall-clock budget" if b["stopped_early"] else ""))
    many = len(b["level_sizes"]) > 1
    head = "| Row | Seed | Before pooling | After each upsampling | First beta | Floor |"
    if many:
        head += " Exact score at the end of each level |"
    head += (" Final exact score | vs start | vs live #1 | Envelope L1 (1,000 cells) vs 2782 / 2783 |\n"
             + "|---|---|---|---|---|---|" + ("---|" if many else "") + "---|---|---|---|")
    lines = []
    for r in b["rows"]:
        per_level = " / ".join("n/a" if x["score"] is None else f"{x['score']:.6f}" for x in r["levels"])
        final = "n/a" if r["last_score"] is None else f"{r['last_score']:.13f}"
        if not r["last_is_final"]:
            final += " (not final)"
        prof = ("n/a" if r["profile_vs_frozen"] is None
                else f"{r['profile_vs_frozen']:.4f} / {r['profile_vs_live']:.4f}")
        line = (f"| {r['row']} | {r['seed']} | {r['pre']} | {r['post']} | {r['beta0']:.0e} | "
                f"{r['floor']:g} / {r['up_floor']:g} |")
        if many:
            line += f" {per_level} |"
        lines.append(line + f" {final} | {_g(r['gain_over_start'])} | {_g(r['gain_over_live'])} | {prof} |")
    return title + "\n\n" + head + "\n" + "\n".join(lines)


def best_rows(root: Path, count: int = 8) -> list[dict]:
    """The stored solutions of 2,000,000 cells with the highest verifier score, over all runs."""
    recs = []
    for path in sorted(root.glob("*/levels.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["valid"] and r["n"] >= 1_999_999:
                recs.append(r)
    recs.sort(key=lambda r: -r["score"])
    return recs[:count]


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "knockout/v2/results/multires")
    for b in batches(root):
        print(batch_table(b) + "\n")
    print("| Best stored solutions (2,000,000 cells) | Start | Exact score | vs start | vs live #1 |\n|---|---|---|---|---|")
    for r in best_rows(root):
        print(f"| `{r['experiment_id']}` | {r['config']['start']} | {r['score']:.16f} | {_g(r['gain_over_start'])} "
              f"| {_g(r['gain_over_live'])} |")
