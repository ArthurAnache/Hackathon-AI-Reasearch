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


def get(problem_id: str):
    try:
        return PROBLEMS[problem_id]
    except KeyError:
        raise KeyError(f"unknown problem {problem_id!r}; known: {sorted(PROBLEMS)}") from None


def better(a: float, b: float, direction: str) -> bool:
    return a < b if direction == "minimize" else a > b


RECORD_RTOL = 1e-9  # float64 verifiers and rounded logs cannot resolve differences below this


def beats_record(score: float, problem) -> bool:
    """True only if `score` beats BEST_KNOWN by more than RECORD_RTOL (relative). Ties are not records."""
    margin = RECORD_RTOL * max(1.0, abs(problem.BEST_KNOWN))
    if problem.DIRECTION == "minimize":
        return score < problem.BEST_KNOWN - margin
    return score > problem.BEST_KNOWN + margin


__all__ = ["PROBLEMS", "get", "better", "beats_record", "RECORD_RTOL", "InvalidSolution"]
