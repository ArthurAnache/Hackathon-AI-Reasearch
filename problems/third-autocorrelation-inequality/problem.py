"""PROPOSED MAIN PROBLEM (not yet accepted by Arthur): EinsteinArena third-autocorrelation-inequality (problem id 4).
FROZEN: never modified by agents.

MINIMIZE C = |max(f*f)| / (integral of f)^2 over step functions f that may take negative values, given as
{"values": [f_0, ..., f_{n-1}]} with 1 <= n <= 100,000 numbers. With dx = 0.5 / n the verifier computes
|max(convolve(f, f) * dx)| / (sum(f) * dx)^2 and rejects (sum(f) * dx)^2 < 1e-9. A LOWER score is better.

Definition trap: this is the constant C' of arXiv:2511.02864 (Problem 6.4), with the absolute value outside the
max. It is not Tao's constant 1d (max |f*f|); the two must not be compared.

Until Arthur accepts the proposal the problem is not registered in problems/catalog.py. The arbiter loads it by
path (selection/verify_claim.py: load_problem):

    from selection.verify_claim import load_problem
    p = load_problem("third-autocorrelation-inequality")
    p.verify(c); p.score(c); p.target()

Limits: <= 100,000 values (zod schema of the pinned commit; the live API does not serve that number). The record to
beat is the LIVE #1 - delta (1e-5); on 2026-11-15 that is <= 1.449099789331911.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_frozen = _load("ws1_problems_frozen", _HERE.parent / "_frozen.py")
_independent = _load("ws1_independent_third_autocorrelation", _HERE / "independent.py")
InvalidSolution = _frozen.InvalidSolution

MAX_VALUES = 100_000


def _number(x) -> bool:
    try:
        return _frozen.number(x)
    except OverflowError:   # an integer too large for a float64
        return False


def check_schema(candidate) -> None:
    """Server-side schema (zod, pinned commit): {"values": array of numbers, length 1..100,000}."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("values"), list):
        raise InvalidSolution('candidate must be {"values": [numbers]}')
    values = candidate["values"]
    if not 1 <= len(values) <= MAX_VALUES:
        raise InvalidSolution(f"values must have 1..{MAX_VALUES} entries (server schema), got {len(values)}")
    if not all(_number(x) for x in values):
        raise InvalidSolution("every value must be a finite JSON number")


_P = _frozen.FrozenProblem(_HERE, check_schema=check_schema, independent=_independent.score)

SLUG, PROBLEM_ID, NAME, URL, ROLE = _P.slug, _P.problem_id, _P.name, _P.url, _P.role
DIRECTION, DELTA, SOLUTION_FORMAT, LIMITS = _P.direction, _P.delta, _P.solution_format, _P.limits
VERIFIER_SHA256 = _P.verifier_sha256

verify = _P.verify
score = _P.score
why_invalid = _P.why_invalid
independent_score = _P.independent_score
best_known_score = _P.best_known_score
target = _P.target
gain = _P.gain
beats_record = _P.beats_record
certificate = _P.certificate
submission_path = _P.submission_path
