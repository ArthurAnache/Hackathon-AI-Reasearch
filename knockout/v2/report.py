"""Markdown tables from episode tables (the JSONL written by `finalize`).

    .venv/bin/python -m knockout.v2.report knockout/v2/results/moves_cpu/<run_id>.jsonl [more.jsonl ...]
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


def best(rows: list[dict], k: int = 10) -> str:
    done = sorted((r for r in rows if r.get("value") is not None and not r.get("is_baseline")),
                  key=lambda r: -r["value"])[:k]
    lines = ["| episode | score | value | class | dist to polished start |", "|---|---|---|---|---|"]
    for r in done:
        lines.append(f"| {r['experiment_id']} | {r['score']:.16f} | {_fmt(r['value'])} | {r['decision']} | "
                     f"{r['dist_to_polished_start']:.2e} |")
    return "\n".join(lines)


def main() -> None:
    rows = load(sys.argv[1:])
    spend = sum(r.get("usd_estimate", 0.0) for r in rows)
    print(f"{len(rows)} episodes, {sum(r['calls'] for r in rows):.0f} evaluations, estimated spend {spend:.2f} USD\n")
    print("Polish alone\n\n" + baselines(rows) + "\n")
    for start in sorted({r["start"] for r in rows}):
        sub = [r for r in rows if r["start"] == start]
        print(f"From {start}, per family\n\n" + table(sub, ("family",)) + "\n")
        print(f"From {start}, per family and strength\n\n" + table(sub, ("family", "strength")) + "\n")
        print(f"From {start}, per family, schedule and budget\n\n" + table(sub, ("family", "schedule", "budget")) + "\n")
        print(f"From {start}, best episodes\n\n" + best(sub) + "\n")


if __name__ == "__main__":
    main()
