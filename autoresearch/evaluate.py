"""Fixed-budget, fixed-seed evaluation of one strategy directory.

Every experiment runs the same seeds with the same per-seed wall-clock budget,
so two experiments differ only in the strategy (common random numbers).
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from problem import DIRECTION, InvalidSolution, verify

from .frozen import ROOT

# Grace on top of the budget for interpreter start-up and output. Beyond this
# the child is killed and the seed counts as a timeout.
TIMEOUT_GRACE_S = 5.0


@dataclass
class SeedResult:
    seed: int
    status: str            # ok | invalid | crash | timeout
    score: float | None
    elapsed_s: float
    error: str | None = None


def better(a: float, b: float, min_delta: float = 0.0) -> bool:
    """True if score a beats score b by more than min_delta, respecting DIRECTION."""
    return a > b + min_delta if DIRECTION == "maximize" else a < b - min_delta


def run_seed(strategy_dir: Path, seed: int, budget_s: float, log_dir: Path) -> SeedResult:
    out = log_dir / f"seed_{seed}.json"
    log = log_dir / f"seed_{seed}.log"
    cmd = [sys.executable, "-m", "autoresearch.worker", str(strategy_dir), str(seed), str(budget_s), str(out)]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              timeout=budget_s * 1.5 + TIMEOUT_GRACE_S)
    except subprocess.TimeoutExpired as e:
        log.write_text(f"TIMEOUT\n--- stdout\n{e.stdout or ''}\n--- stderr\n{e.stderr or ''}")
        return SeedResult(seed, "timeout", None, time.monotonic() - t0, "exceeded hard timeout")
    elapsed = time.monotonic() - t0
    log.write_text(f"returncode={proc.returncode}\n--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr}")
    if proc.returncode != 0 or not out.exists():
        last = (proc.stderr.strip().splitlines() or ["no output"])[-1]
        return SeedResult(seed, "crash", None, elapsed, last[:500])
    try:
        solution = json.loads(out.read_text())["solution"]
        score = float(verify(solution))   # the only trusted score
    except InvalidSolution as e:
        return SeedResult(seed, "invalid", None, elapsed, str(e)[:500])
    except Exception as e:  # malformed output, verifier choking on garbage, ...
        return SeedResult(seed, "invalid", None, elapsed, f"{type(e).__name__}: {e}"[:500])
    return SeedResult(seed, "ok", score, elapsed)


def evaluate(strategy_dir: Path, seeds: list[int], budget_s: float, log_dir: Path) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    results = [run_seed(strategy_dir, s, budget_s, log_dir) for s in seeds]
    failures = [r for r in results if r.status != "ok"]
    scores = [r.score for r in results if r.status == "ok"]
    summary = {
        "seeds": [asdict(r) for r in results],
        "elapsed_s": sum(r.elapsed_s for r in results),
    }
    if failures:
        # One bad seed fails the whole experiment: a strategy that sometimes
        # produces invalid output is not an improvement.
        summary.update(status=failures[0].status, score=None, error=failures[0].error)
    else:
        summary.update(
            status="ok",
            score=statistics.fmean(scores),
            score_min=min(scores),
            score_max=max(scores),
            score_std=statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            error=None,
        )
    return summary
