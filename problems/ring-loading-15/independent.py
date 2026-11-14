"""Independent, EXACT verifier for ring-loading-15. FROZEN.

Re-implements the definition with integer arithmetic over a common denominator (the pinned verifier loops over
Fractions): A(u, v) = min over the 2^15 adversary choices z_i in {v_i, -u_i} of max over k = 1..15 of
|sum_{i<=k} z_i - sum_{i>k} z_i|. Admissibility mirrors the problem statement: decimal or fraction strings, nonnegative,
reduced numerator/denominator within 64 bits, u_i + v_i <= 1.
"""

from __future__ import annotations

import itertools
import math
import re
from fractions import Fraction

PAIRS = 15
TOLERANCE_ABS = 0.0
_RATIONAL = re.compile(r"^(?:0|[1-9]\d*)(?:(?:\.\d+)|(?:/[1-9]\d*))?$")


def _parse(s) -> Fraction:
    if not isinstance(s, str) or not 1 <= len(s) <= 80 or not _RATIONAL.fullmatch(s):
        raise ValueError(f"not an admissible rational string: {s!r}")
    if "/" in s:
        num, den = s.split("/")
        q = Fraction(int(num), int(den))
    elif "." in s:
        whole, frac = s.split(".")
        q = Fraction(int(whole + frac), 10 ** len(frac))
    else:
        q = Fraction(int(s))
    if q.numerator.bit_length() > 64 or q.denominator.bit_length() > 64:
        raise ValueError("reduced numerator/denominator exceed 64 bits")
    return q


def score(candidate: dict) -> dict:
    pairs = candidate["pairs"]
    if len(pairs) != PAIRS:
        raise ValueError("expected exactly 15 pairs")
    us, vs = [], []
    for p in pairs:
        if not isinstance(p, list) or len(p) != 2:
            raise ValueError("each pair must be [u, v]")
        u, v = _parse(p[0]), _parse(p[1])
        if u + v > 1:
            raise ValueError("u + v > 1")
        us.append(u)
        vs.append(v)
    den = math.lcm(*(q.denominator for q in us + vs))
    U = [int(q * den) for q in us]
    V = [int(q * den) for q in vs]
    best = None
    for choice in itertools.product((False, True), repeat=PAIRS):   # True: the adversary picks -u_i
        z = [-U[i] if pick_u else V[i] for i, pick_u in enumerate(choice)]
        total, prefix, worst = sum(z), 0, 0
        for k in range(PAIRS):
            prefix += z[k]
            worst = max(worst, abs(prefix - (total - prefix)))
        if best is None or worst < best:
            best = worst
    exact = Fraction(best, den)
    return {"score": float(exact), "fraction": exact, "tolerance_abs": TOLERANCE_ABS,
            "method": "integer enumeration of all 2^15 masks over a common denominator"}
