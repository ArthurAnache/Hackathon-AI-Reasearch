"""Run the current strategy with a fixed budget and fixed seeds, then verify independently."""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from problem import InvalidSolution, verify


@dataclass
class EvalResult:
    scores: dict[int, float] = field(default_factory=dict)
    score: float | None = None
    failure_reason: str | None = None
    duration_s: float = 0.0
    solutions: dict[int, dict] = field(default_factory=dict)


def run_one(root: Path, seed: int, budget_s: float) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "runner._worker", str(seed), str(budget_s)],
        cwd=root, capture_output=True, text=True, timeout=budget_s + 5,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        raise RuntimeError("strategy crashed: " + " | ".join(tail))
    marker = proc.stdout.rfind("__SOLUTION__")
    if marker < 0:
        raise RuntimeError("strategy printed no solution")
    return json.loads(proc.stdout[marker + len("__SOLUTION__"):].strip())


def evaluate(root: Path, seeds: list[int], budget_s: float, aggregate: str = "mean") -> EvalResult:
    res = EvalResult()
    t0 = time.monotonic()
    try:
        for seed in seeds:
            sol = run_one(root, seed, budget_s)
            res.solutions[seed] = sol
            res.scores[seed] = verify(sol)  # independent check: the only score we trust
    except subprocess.TimeoutExpired:
        res.failure_reason = f"timeout: exceeded budget of {budget_s}s"
    except InvalidSolution as e:
        res.failure_reason = f"invalid solution: {e}"
    except Exception as e:  # crash, bad JSON, ...
        res.failure_reason = f"{type(e).__name__}: {e}"
    res.duration_s = round(time.monotonic() - t0, 3)
    if res.failure_reason is None:
        vals = list(res.scores.values())
        res.score = statistics.mean(vals) if aggregate == "mean" else min(vals)
    return res
