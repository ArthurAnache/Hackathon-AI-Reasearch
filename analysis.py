"""Summarise the experiment store: history and best-so-far per problem (for humans and the demo).

    python analysis.py                      # report for experiments.jsonl
    python analysis.py --json               # machine-readable (dashboard)
    python analysis.py path/to/experiments.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import interfaces as I
import problems


def summarise(records: list[dict]) -> dict:
    out = {}
    for pid in dict.fromkeys(r["problem"] for r in records):
        p = I.get_problem(pid)
        hist = [r for r in records if r["problem"] == pid]
        kept = [r for r in hist if r["status"] == "keep"]
        pick = min if p.direction == "minimize" else max
        best = pick(kept, key=lambda r: r["score"]) if kept else None
        traj, cur = [], None
        for r in hist:
            if r["status"] == "keep":
                cur = r["score"] if cur is None else pick(cur, r["score"])
            traj.append(cur)
        tie = bool(best) and not p.beats_record(best["score"]) and \
            abs(best["score"] - p.best_known) <= problems.RECORD_RTOL * max(1.0, abs(p.best_known))
        out[pid] = {
            "name": p.name,
            "direction": p.direction,
            "experiments": len(hist),
            "counts": {s: sum(r["status"] == s for r in hist) for s in I.STATUSES},
            "baseline": kept[0]["score"] if kept else None,
            "best": best["score"] if best else None,
            "best_experiment": best["experiment_id"] if best else None,
            "best_hypothesis": best["hypothesis"] if best else None,
            "best_solution": best["solution_path"] if best else None,
            "best_known": p.best_known,
            "best_known_source": p.best_known_source,
            "gap_to_record": None if best is None else abs(best["score"] - p.best_known),
            "beats_record": bool(best) and p.beats_record(best["score"]),
            "ties_record": tie,
            "best_so_far": traj,
            "history": [{k: r[k] for k in ("experiment_id", "agent", "status", "score", "budget_s", "hypothesis")}
                        for r in hist],
        }
    return out


def report(summary: dict) -> str:
    lines = []
    for pid, s in summary.items():
        arrow = "lower is better" if s["direction"] == "minimize" else "higher is better"
        counts = ", ".join(f"{k} {v}" for k, v in s["counts"].items() if v)
        lines.append(f"== {pid}: {s['name']} ({arrow})")
        lines.append(f"   {s['experiments']} experiments: {counts}")
        if s["best"] is not None:
            verdict = "BEATS RECORD" if s["beats_record"] else "TIES RECORD" if s["ties_record"] else "record not beaten"
            lines.append(f"   baseline {s['baseline']:.10g} -> best {s['best']:.10g} ({s['best_experiment']})")
            lines.append(f"   record   {s['best_known']:.10g}  gap {s['gap_to_record']:.3g}  {verdict}")
        for h in s["history"]:
            sc = "-" if h["score"] is None else f"{h['score']:.10g}"
            lines.append(f"   {h['experiment_id']} {h['status']:8} {sc:>16}  [{h['agent']}, {h['budget_s']:g}s] {h['hypothesis']}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    store = I.JsonlStore(Path(args[0])) if args else I.JsonlStore()
    records = store.read()
    if not records:
        print(f"no experiments in {store.path}")
        return 1
    summary = summarise(records)
    print(json.dumps(summary, indent=2) if "--json" in argv else report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
