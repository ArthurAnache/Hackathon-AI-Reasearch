"""RECORD BET: EinsteinArena ring-loading-15 (problem id 30). FROZEN: never modified by agents.

Choose 15 pairs {"pairs": [["u", "v"], ...]} of nonnegative decimal or fraction strings with u + v <= 1, and
maximize A(u, v) = min over adversary choices z_i in {v_i, -u_i} of max over k of
|sum_{i<=k} z_i - sum_{i>k} z_i|. The verifier is exact (Fractions) and delta = 0, so only a strictly larger value
than the live #1 counts. Live #1 on 2026-11-14: 9/8 (it beats the published 47/42).

The forum digest shows local search around 9/8 is provably dead (see selection/digests/ring-loading-15.md): only a
structural change can win. WS1 leaves this bet to the agent.
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
_independent = _load("ws1_independent_ring_loading_15", _HERE / "independent.py")
InvalidSolution = _frozen.InvalidSolution

N_PAIRS = 15


def check_schema(candidate) -> None:
    """Server-side schema (zod): {"pairs": 15 tuples [u, v] of nonnegative rational strings (<= 80 chars)}."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("pairs"), list):
        raise InvalidSolution('candidate must be {"pairs": [["u", "v"], ...]}')
    pairs = candidate["pairs"]
    if len(pairs) != N_PAIRS:
        raise InvalidSolution(f"expected {N_PAIRS} pairs, got {len(pairs)}")
    for p in pairs:
        if not isinstance(p, list) or len(p) != 2 or not all(isinstance(s, str) and 1 <= len(s) <= 80 for s in p):
            raise InvalidSolution("each pair must be [u, v] with two rational strings of 1..80 characters")


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
