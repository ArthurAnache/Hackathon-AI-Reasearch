"""experiments.jsonl: the append-only decision log (one JSON object per experiment)."""
from __future__ import annotations

import json
from pathlib import Path

LOG_NAME = "experiments.jsonl"

REQUIRED_FIELDS = (
    "id", "timestamp", "agent", "hypothesis", "rationale_from_history",
    "edited_files", "diff", "seeds", "budget_s", "scores", "score",
    "best_before", "decision", "failure_reason", "frozen_hash", "commit",
)


def read(root: Path) -> list[dict]:
    p = root / LOG_NAME
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append(root: Path, row: dict) -> None:
    missing = [k for k in REQUIRED_FIELDS if k not in row]
    if missing:
        raise ValueError(f"history row missing fields: {missing}")
    with (root / LOG_NAME).open("a") as f:
        f.write(json.dumps(row, sort_keys=False) + "\n")


def best(rows: list[dict], direction: str = "maximize") -> dict | None:
    kept = [r for r in rows if r["decision"] in ("keep", "baseline") and r["score"] is not None]
    if not kept:
        return None
    pick = max if direction == "maximize" else min
    return pick(kept, key=lambda r: r["score"])


def next_id(rows: list[dict]) -> str:
    return f"exp-{len(rows):04d}"
