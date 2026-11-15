"""Problem registry. DO NOT MODIFY (hash-locked by prepare.py).

Every problem module exposes the same contract:
    ID, NAME, URL, DIRECTION ("minimize" | "maximize"),
    BEST_KNOWN (float), BEST_KNOWN_SOURCE (str), SOLUTION_FORMAT (str),
    verify(solution) -> float          raises InvalidSolution
    verify_exact(solution) -> Fraction optional, exact rational certificate
"""
from . import p1a, p1b, p1d, p10c, p11a
from .common import InvalidSolution

PROBLEMS = {m.ID: m for m in (p1a, p1b, p1d, p10c, p11a)}


_WS1_CACHE: dict = {}


def ws1_slugs() -> list[str]:
    """EinsteinArena problems frozen by WS1 in problems/<slug>/ (empty if WS1's catalog is absent)."""
    from pathlib import Path
    here = Path(__file__).resolve().parent
    return sorted(d.name for d in here.iterdir() if (d / "problem.py").exists() and (d / "meta.json").exists())


def get(problem_id: str):
    """A WS2 problem module (1a, 1b, 1d, 10c, 11a) or a WS1 EinsteinArena problem by slug, through
    WS1's own adapter problems.catalog.ws2_module (live #1 as BEST_KNOWN, snapshot when offline)."""
    if problem_id in PROBLEMS:
        return PROBLEMS[problem_id]
    if problem_id in ws1_slugs():
        if problem_id not in _WS1_CACHE:
            from .catalog import ws2_module
            _WS1_CACHE[problem_id] = ws2_module(problem_id)
        return _WS1_CACHE[problem_id]
    raise KeyError(f"unknown problem {problem_id!r}; known: {sorted(PROBLEMS) + ws1_slugs()}")


def better(a: float, b: float, direction: str) -> bool:
    return a < b if direction == "minimize" else a > b


RECORD_RTOL = 1e-9  # float64 verifiers and rounded logs cannot resolve differences below this


def beats_record(score: float, problem) -> bool:
    """True only if `score` beats BEST_KNOWN by more than RECORD_RTOL (relative) AND, for problems that define
    one (EinsteinArena), by at least the platform threshold DELTA. Ties are not records."""
    margin = max(RECORD_RTOL * max(1.0, abs(problem.BEST_KNOWN)), getattr(problem, "DELTA", 0.0) or 0.0)
    if problem.DIRECTION == "minimize":
        return score < problem.BEST_KNOWN - margin
    return score > problem.BEST_KNOWN + margin


__all__ = ["PROBLEMS", "get", "ws1_slugs", "better", "beats_record", "RECORD_RTOL", "InvalidSolution"]
