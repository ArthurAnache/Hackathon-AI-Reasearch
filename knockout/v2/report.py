"""Markdown tables from episode tables (the JSONL written by `finalize`).

    .venv/bin/python -m knockout.v2.report knockout/v2/results/moves_cpu/<run_id>.jsonl [more.jsonl ...] [--holds]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load(paths) -> list[dict]:
    rows = []
    for path in paths:
        rows += [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:+.2e}"


def table(rows: list[dict], keys: tuple[str, ...]) -> str:
    """One line per value of `keys`: episodes, value quantiles, best value, episodes above 1e-6, classes."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("value") is not None and not r.get("is_baseline"):
            groups[tuple(r[k] for k in keys)].append(r)
    head = list(keys) + ["episodes", "min value", "median value", "max value", "above +1e-6", "improved", "undone",
                         "worse_basin", "median moved score", "median dist to polished start"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key in sorted(groups, key=lambda k: tuple(str(x) if isinstance(x, str) else x for x in k)):
        g = groups[key]
        values = np.array([r["value"] for r in g])
        classes = Counter(r["decision"] for r in g)
        cells = [f"{k:g}" if isinstance(k, float) else str(k) for k in key]
        cells += [str(len(g)), _fmt(values.min()), _fmt(float(np.median(values))), _fmt(values.max()),
                  str(int((values > 1e-6).sum())), str(classes["improved"]), str(classes["undone"]),
                  str(classes["worse_basin"]), f"{np.median([r['moved_score'] for r in g]):.6f}",
                  f"{np.median([r['dist_to_polished_start'] for r in g]):.2e}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def baselines(rows: list[dict]) -> str:
    lines = ["| start | budget | score after polish alone | gain over the start |", "|---|---|---|---|"]
    for r in sorted((r for r in rows if r.get("is_baseline")), key=lambda r: (r["start"], r["budget"])):
        gain = r["score"] - r["moved_score"]
        lines.append(f"| {r['start']} | {r['budget']} | {r['score']:.16f} | {gain:+.2e} |")
    return "\n".join(lines)


def reheats(rows: list[dict]) -> str:
    """Schedule x budget for the episodes without a move: value against polish alone, and where the schedule
    itself ended (exact score at the end of its last but one stage, minus the start; the polisher's answer is the
    best point seen, so a schedule that ends below its start returns the start and shows the value of "no gain")."""
    done = [r for r in rows if r["family"] == "reheat" and r.get("value") is not None]
    budgets = sorted({r["budget"] for r in done})
    lines = ["| schedule | " + " | ".join(f"{b} evaluations" for b in budgets) + " |", "|---|" + "---|" * len(budgets)]
    for schedule in sorted({r["schedule"] for r in done}):
        cells = []
        for budget in budgets:
            r = next((r for r in done if r["schedule"] == schedule and r["budget"] == budget), None)
            if r is None:
                cells.append("not run")
                continue
            ends = [s["exact_at_end"] for s in r["stages"] if s.get("exact_at_end") is not None]
            end = f"{ends[-1] - r['moved_score']:+.2e}" if ends else "n/a"
            cells.append(f"{r['score'] - r['moved_score']:+.2e} (value {_fmt(r['value'])}, ends at {end})")
        lines.append(f"| {schedule} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def holds(rows: list[dict]) -> str:
    """Per stage of the hold schedules: gain of the smoothed objective inside the stage, exact score at its end."""
    lines = ["| episode | beta | evaluations so far | smoothed gain in the stage | exact score at the end, minus the start |",
             "|---|---|---|---|---|"]
    for r in sorted((r for r in rows if r["schedule"].lstrip("f:").startswith("h")),
                    key=lambda r: (r["start"], r["schedule"], r["budget"])):
        for s in r["stages"]:
            if "smooth_first" not in s:
                continue
            exact = s.get("exact_at_end")
            lines.append(f"| {r['experiment_id']} | {s['beta']:.0e} | {s['calls']:.0f} | "
                         f"{s['smooth_first'] - s['smooth_best']:.2e} | "
                         f"{'n/a' if exact is None else format(exact - r['moved_score'], '+.2e')} |")
    return "\n".join(lines)


def best(rows: list[dict], k: int = 10) -> str:
    done = sorted((r for r in rows if r.get("value") is not None and not r.get("is_baseline")),
                  key=lambda r: -r["value"])[:k]
    lines = ["| episode | score | value | class | dist to polished start |", "|---|---|---|---|---|"]
    for r in done:
        lines.append(f"| {r['experiment_id']} | {r['score']:.16f} | {_fmt(r['value'])} | {r['decision']} | "
                     f"{r['dist_to_polished_start']:.2e} |")
    return "\n".join(lines)


def main() -> None:
    rows = load(a for a in sys.argv[1:] if not a.startswith("--"))
    spend = sum(r.get("usd_estimate", 0.0) for r in rows)
    print(f"{len(rows)} episodes, {sum(r['calls'] for r in rows):.0f} evaluations, estimated spend {spend:.2f} USD\n")
    print("Polish alone\n\n" + baselines(rows) + "\n")
    for start in sorted({r["start"] for r in rows}):
        sub = [r for r in rows if r["start"] == start]
        print(f"From {start}, per family\n\n" + table(sub, ("family",)) + "\n")
        print(f"From {start}, per family and strength\n\n" + table(sub, ("family", "strength")) + "\n")
        print(f"From {start}, per family, schedule and budget\n\n" + table(sub, ("family", "schedule", "budget")) + "\n")
        print(f"From {start}, no move: score gain over the start, per schedule and budget\n\n" + reheats(sub) + "\n")
        print(f"From {start}, best episodes\n\n" + best(sub) + "\n")
    if "--holds" in sys.argv:
        print("Hold schedules, stage by stage\n\n" + holds(rows) + "\n")


if __name__ == "__main__":
    main()
