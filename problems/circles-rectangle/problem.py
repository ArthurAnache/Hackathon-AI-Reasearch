"""BACKUP PROBLEM: EinsteinArena circles-rectangle (problem id 18). FROZEN: never modified by agents.

Place 21 circles {"circles": [[x, y, r], ...]} (21 triples of numbers) inside an axis-aligned rectangle of perimeter 4
(width + height <= 2 for the bounding box, checked exactly), pairwise non-overlapping (float test dist < r_i + r_j
rejects; tangency allowed), and maximize the sum of the radii. delta = 1e-10.

The pinned verifier has no overlap tolerance and computes the perimeter in exact Fractions: the tolerance exploits
reported on the forum (overlap/perimeter slack ~1e-9, large-coordinate float rounding) are rejected; see
tests/test_ws1_frozen_problems.py.
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
_independent = _load("ws1_independent_circles_rectangle", _HERE / "independent.py")
InvalidSolution = _frozen.InvalidSolution

N_CIRCLES = 21


def check_schema(candidate) -> None:
    """Server-side schema (zod): {"circles": array of 21 arrays of 3 numbers}."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("circles"), list):
        raise InvalidSolution('candidate must be {"circles": [[x, y, r], ...]}')
    circles = candidate["circles"]
    if len(circles) != N_CIRCLES:
        raise InvalidSolution(f"expected {N_CIRCLES} circles, got {len(circles)}")
    for c in circles:
        if not isinstance(c, list) or len(c) != 3 or not all(_frozen.number(t) for t in c):
            raise InvalidSolution("each circle must be [x, y, r] with 3 finite numbers")


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
