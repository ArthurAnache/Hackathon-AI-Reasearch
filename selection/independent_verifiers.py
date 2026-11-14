"""WS1: second, independent implementations of the scoring functions, used by integrations/submit.py (gate b).

Each implementation re-derives the score from the problem definition, with a different numerical route from the pinned
EinsteinArena verifier, so that a bug or float artefact in one is unlikely to be repeated in the other:

  ring-loading-15                    exact integer arithmetic over a common denominator (pinned: Fraction loop)
  circles-rectangle                  every constraint checked exactly in Fractions, including the pairwise
                                     non-overlap (dx² + dy² >= (r_i + r_j)²); pinned uses float sqrt
  second-autocorrelation-inequality  closed form C = (2 Σc² + Σ c_i c_{i+1}) / (3 Σc · max c) on an rfft
                                     autoconvolution, with exactly rounded sums (math.fsum); pinned uses
                                     scipy oaconvolve + numpy piecewise-linear quadrature

score(slug, solution) returns {"score": float, "exact": str | None, "tolerance_abs": float, "method": str}.
`tolerance_abs` is the largest |independent - pinned| we accept as agreement; it is fixed a priori per problem.
A ValueError means the independent check judges the solution invalid. A problem missing from IMPLEMENTATIONS has no
independent verifier, so submit.py refuses to send it.
"""

from __future__ import annotations

import itertools
import math
import re
from fractions import Fraction


# ---------------------------------------------------------------- ring-loading-15

RING_PAIRS = 15
_RATIONAL = re.compile(r"^(?:0|[1-9]\d*)(?:(?:\.\d+)|(?:/[1-9]\d*))?$")


def _parse_ring_value(s) -> Fraction:
    # Same admissibility rules as the problem statement / schema (strings, nonnegative, reduced num/den <= 64 bits).
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


def ring_loading_15(solution: dict) -> dict:
    pairs = solution.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != RING_PAIRS:
        raise ValueError("expected exactly 15 pairs")
    us, vs = [], []
    for p in pairs:
        if not isinstance(p, list) or len(p) != 2:
            raise ValueError("each pair must be [u, v]")
        u, v = _parse_ring_value(p[0]), _parse_ring_value(p[1])
        if u + v > 1:
            raise ValueError("u + v > 1")
        us.append(u)
        vs.append(v)
    # Common denominator: everything becomes integers, so the enumeration is exact integer arithmetic.
    den = math.lcm(*(q.denominator for q in us + vs))
    U = [int(q * den) for q in us]
    V = [int(q * den) for q in vs]
    best = None
    for choice in itertools.product((False, True), repeat=RING_PAIRS):  # True: adversary picks -u_i
        z = [-U[i] if c else V[i] for i, c in enumerate(choice)]
        total = sum(z)
        prefix = 0
        worst = 0
        for k in range(RING_PAIRS):  # cut after position k+1, k+1 = 1..15
            prefix += z[k]
            left, right = prefix, total - prefix
            worst = max(worst, abs(left - right))
        if best is None or worst < best:
            best = worst
    exact = Fraction(best, den)
    return {"score": float(exact), "exact": f"{exact.numerator}/{exact.denominator}", "tolerance_abs": 0.0,
            "method": "integer enumeration of all 2^15 masks over a common denominator"}


# ---------------------------------------------------------------- circles-rectangle

CIRCLES = 21


def circles_rectangle(solution: dict) -> dict:
    circles = solution.get("circles")
    if not isinstance(circles, list) or len(circles) != CIRCLES:
        raise ValueError("expected 21 circles")
    xs, ys, rs = [], [], []
    for c in circles:
        if not isinstance(c, list) or len(c) != 3:
            raise ValueError("each circle must be [x, y, r]")
        vals = [float(t) for t in c]
        if not all(math.isfinite(t) for t in vals):
            raise ValueError("non-finite coordinate")
        x, y, r = (Fraction(t) for t in vals)  # exact value of the float64 the server will see
        if r <= 0:
            raise ValueError("radius must be positive")
        xs.append(x)
        ys.append(y)
        rs.append(r)
    width = max(x + r for x, r in zip(xs, rs)) - min(x - r for x, r in zip(xs, rs))
    height = max(y + r for y, r in zip(ys, rs)) - min(y - r for y, r in zip(ys, rs))
    if width + height > 2:
        raise ValueError(f"perimeter exceeded: width + height - 2 = {float(width + height - 2):.3e}")
    for i, j in itertools.combinations(range(CIRCLES), 2):
        d2 = (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2
        if d2 < (rs[i] + rs[j]) ** 2:
            raise ValueError(f"circles {i} and {j} overlap (exact check)")
    total = sum(rs, Fraction(0))
    # Pinned returns float(np.sum(radii)): pairwise float summation of 21 terms, error well below 1e-13.
    return {"score": float(total), "exact": None, "tolerance_abs": 1e-13,
            "method": "exact Fraction feasibility (perimeter and squared distances); exact sum of radii"}


# ---------------------------------------------------------------- second-autocorrelation-inequality

def second_autocorrelation(solution: dict) -> dict:
    import numpy as np

    values = solution.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("values must be a non-empty list")
    f = np.asarray(values, dtype=np.float64)
    if f.ndim != 1 or not np.all(np.isfinite(f)):
        raise ValueError("values must be a flat list of finite numbers")
    if np.any(f < -1e-6):  # problem rule: values below -1e-6 are rejected, the rest is clipped at 0
        raise ValueError("function must be nonnegative")
    f = np.clip(f, 0.0, None)
    if not f.any():
        raise ValueError("function must have positive integral")
    m = 2 * len(f) - 1
    size = 1 << (m - 1).bit_length()
    F = np.fft.rfft(f, size)
    c = np.fft.irfft(F * F, size)[:m]
    c = np.clip(c, 0.0, None)  # true autoconvolution of f >= 0 is >= 0; clip FFT round-off below zero
    num = 2.0 * math.fsum((c * c).tolist()) + math.fsum((c[:-1] * c[1:]).tolist())
    den = 3.0 * math.fsum(c.tolist()) * float(c.max())
    return {"score": num / den, "exact": None, "tolerance_abs": 1e-10,
            "method": "rfft autoconvolution + closed-form piecewise-linear L2 with math.fsum"}


IMPLEMENTATIONS = {
    "ring-loading-15": ring_loading_15,
    "circles-rectangle": circles_rectangle,
    "second-autocorrelation-inequality": second_autocorrelation,
}


def score(slug: str, solution: dict) -> dict:
    if slug not in IMPLEMENTATIONS:
        raise KeyError(f"no independent verifier for {slug}")
    return IMPLEMENTATIONS[slug](solution)
