"""Markdown table of the annealing chains, from the run folders downloaded from the Volume.

    modal volume get knockout-v2-gpu <run_id>/summary.json knockout/v2/results/anneal_gpu/<run_id>/summary.json
    python -m knockout.v2.gpu.report knockout/v2/results/anneal_gpu

Every score in the table is the frozen verifier's float64 score computed on CPU inside the container.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _g(x) -> str:
    return "n/a" if x is None else f"{x:+.3e}"


def rows(root: Path) -> list[dict]:
    out = []
    for path in sorted(root.glob("*/summary.json")):
        s = json.loads(path.read_text())
        for t in s["table"]:
            out.append({"run_id": s["run_id"], "gpu": s["env"]["gpu_type"], "wall_s": s["wall_s"],
                        "stopped_early": s["stopped_early"], **t})
    return out


def table(root: Path) -> str:
    head = ("| Run | Chain | Start | beta start -> end (stages) | Floor | Kick | Evaluations "
            "| Lowest stage end vs start | Final exact score | Final vs start | Final vs live #1 "
            "| Best seen vs start |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in rows(root):
        final = "n/a" if r["final_score"] is None else f"{r['final_score']:.16f}"
        start = r["start"] if len(r["start"]) < 24 else "…" + r["start"][-22:]
        lines.append(
            f"| `{r['run_id'][:16]}` | {r['chain']} | {start} | {r['beta_start']:.0e} -> {r['beta_end']:.0e} "
            f"({r['stages']}) | {r['floor']:g} | {r.get('jitter', 0.0):g} | {r['evals']} "
            f"| {_g(r['min_exact_seen_minus_start'])} | {final} "
            f"| {_g(r['final_gain_over_start'])} | {_g(r['final_gain_over_live'])} "
            f"| {_g(r['best_gain_over_start'])} |")
    return head + "\n".join(lines)


if __name__ == "__main__":
    print(table(Path(sys.argv[1] if len(sys.argv) > 1 else "knockout/v2/results/anneal_gpu")))
