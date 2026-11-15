"""Summarise the experiment store: history and best-so-far per problem (for humans and the demo).

    python analysis.py                      # report for experiments.jsonl
    python analysis.py --json               # machine-readable (dashboard)
    python analysis.py path/to/experiments.jsonl
    python analysis.py --csv out.csv        # CSV export (an export only: the JSONL store stays the source of truth)
    python analysis.py --arms               # per-arm comparison (keep rate, mean gain) for the evolution loop
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
            "best_certificate": best["certificate_path"] if best else None,
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


def export_csv(records: list[dict], path: Path) -> int:
    import csv
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(I.RECORD_FIELDS))
        w.writeheader()
        for r in records:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()
                        if k in I.RECORD_FIELDS})
    return len(records)


def arms(records: list[dict]) -> dict:
    """Per (problem, arm): candidates, keep rate, mean gain over the parent, compute used, and the score
    distribution over all seeds (median, best) plus how many new verified bests the arm produced."""
    out: dict = {}
    for r in records:
        if not r.get("arm") or r["arm"] == "manual":
            continue
        a = out.setdefault(f"{r['problem']}:{r['arm']}", {"candidates": 0, "keep": 0, "gains": [], "job_seconds": 0.0,
                                                           "seed_scores": [], "new_bests": 0})
        a["candidates"] += 1
        a["seed_scores"] += [s for s in (r.get("scores") or [r.get("score")]) if s is not None]
        a["new_bests"] += bool(r.get("certificate_path")) and r.get("best_before") is not None
        a["keep"] += r["status"] == "keep"
        a["job_seconds"] += (r.get("budget_s") or 0) * len(r.get("seeds") or [])
        if r.get("score") is not None and r.get("best_before") is not None:
            sign = -1 if I.get_problem(r["problem"]).direction == "minimize" else 1
            a["gains"].append(sign * (r["score"] - r["best_before"]))
    for k, a in out.items():
        g = a.pop("gains")
        a["keep_rate"] = round(a["keep"] / a["candidates"], 3)
        a["mean_gain"] = sum(g) / len(g) if g else None
        s = sorted(a.pop("seed_scores"))           # judge on the distribution over seeds (WS3), not one run
        if s:
            p = I.get_problem(k.split(":")[0])
            a.update(seeds_run=len(s), median=s[len(s) // 2], best=s[0] if p.direction == "minimize" else s[-1])
    return out


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if "--csv" in argv:
        out = Path(argv[argv.index("--csv") + 1])
        args = [a for a in args if a != str(out)]
    store = I.JsonlStore(Path(args[0])) if args else I.JsonlStore()
    records = store.read()
    if not records:
        print(f"no experiments in {store.path}")
        return 1
    if "--csv" in argv:
        print(f"wrote {export_csv(records, out)} rows to {out}")
        return 0
    if "--arms" in argv:
        print(json.dumps(arms(records), indent=2))
        return 0
    summary = summarise(records)
    print(json.dumps(summary, indent=2) if "--json" in argv else report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
