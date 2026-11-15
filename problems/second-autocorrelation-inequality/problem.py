"""MAIN PROBLEM: EinsteinArena second-autocorrelation-inequality (problem id 3). FROZEN: never modified by agents.

Maximize C = ||f*f||_2^2 / (||f*f||_1 * ||f*f||_inf) over nonnegative step functions f, given as
{"values": [f_0, ..., f_{n-1}]} with 1 <= n <= 2,000,000 numbers (values below -1e-6 are rejected, the rest is
clipped at 0). The score is scale-invariant (f -> a f).

    from problems.catalog import load
    p = load("second-autocorrelation-inequality")
    p.verify(c); p.score(c); p.target()

Limits: <= 2,000,000 values (server schema). A candidate whose compact JSON exceeds ~2 MB must be submitted through
POST /api/solutions/upload-url (integrations/submit.py does it). The record to beat is the LIVE #1 + delta (1e-5);
on 2026-11-14 that is >= 0.963648667992813.
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
_independent = _load("ws1_independent_second_autocorrelation", _HERE / "independent.py")
InvalidSolution = _frozen.InvalidSolution

MAX_VALUES = 2_000_000


def check_schema(candidate) -> None:
    """Server-side schema (zod): {"values": array of numbers, length 1..2,000,000}."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("values"), list):
        raise InvalidSolution('candidate must be {"values": [numbers]}')
    values = candidate["values"]
    if not 1 <= len(values) <= MAX_VALUES:
        raise InvalidSolution(f"values must have 1..{MAX_VALUES} entries (server schema), got {len(values)}")
    if not all(_frozen.number(x) for x in values):
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
