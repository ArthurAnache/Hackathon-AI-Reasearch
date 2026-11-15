"""Independent, EXACT verifier for third-autocorrelation-inequality. FROZEN.

Shares no code and no arithmetic with the pinned verifier (numpy, float64 `np.convolve`). Standard library only, no
floating point in the score. Adapted from the scout's selection/scout_third_autocorr_exact.py: same formula, with
the big multiplications done by the decimal module (libmpdec, quasi-linear) instead of Python integers.

1. Every float64 is a dyadic rational, so the n values become integers a_i over one common power of two D:
   f_i = a_i / D. The values may be negative.
2. The verifier's quantity, with dx = 1 / (2n) and c = a * a the full integer autoconvolution (2n - 1 terms), is

       score = |max_k c_k * dx / D^2| / (sum(a) * dx / D)^2 = 2n * |max_k c_k| / (sum a)^2

   an exact rational: D cancels.
3. c is computed exactly by Kronecker substitution. a = p - q with p, q >= 0 (positive and negative parts), so
   c = p*p - 2 p*q + q*q. Each of the three products packs its factors into one huge base-10 integer, with slots
   wide enough that no carry crosses a slot, multiplies them with decimal in exact integer arithmetic (Inexact is
   trapped), and unpacks the slots. Self-checks: sum(c) == sum(a)^2, c_0 == a_0^2, c_last == a_last^2.

Admissibility follows the pinned verifier: finite numbers, and (sum(f) * dx)^2 >= 1e-9, tested on the exact rational
(1e-9 is read as the binary64 the verifier compares with). A candidate within float rounding of that bound may be
judged differently by the two verifiers; the arbiter then reports it as invalid.

Tolerance: TOLERANCE_ABS = 1e-12 is the accepted |exact - pinned float64|. Observed at the freeze (2026-11-15):
4.4e-16 on the certificate (65,536 values), at most 1.4e-15 on the scout's seven candidates (25,600 and 51,200
values) and 1.2e-14 on a synthetic 100,000-value input. The threshold delta is 1e-5.

Cost, measured on one core of a shared laptop: about 0.5 s for 25,600 values, 1.1 s for 65,536 and 2 s for 100,000
(the server's maximum). On the certificate and on those candidates the exact fraction is equal to the one the
scout's integer scorer returns (19 s for 65,536 values).
"""

from __future__ import annotations

import decimal
import math
from fractions import Fraction

MIN_INTEGRAL_SQUARED = Fraction(1e-9)   # the verifier's float literal, as the exact binary64 it is
TOLERANCE_ABS = 1e-12                   # accepted |exact - pinned float64|


def _integers(values) -> tuple[list[int], int]:
    """The values as integers over one common denominator (a power of two): f_i = a_i / den."""
    ratios = []
    for x in values:
        try:
            v = float(x)
        except OverflowError:
            raise ValueError("a value does not fit in a float64") from None
        if not math.isfinite(v):
            raise ValueError("non-finite value")
        ratios.append(v.as_integer_ratio())
    den = max(d for _, d in ratios)          # all denominators are powers of two
    return [m * (den // d) for m, d in ratios], den


def _pack(x: list[int], width: int, ctx: decimal.Context) -> decimal.Decimal:
    return ctx.create_decimal("".join(str(v).zfill(width) for v in reversed(x)))


def _unpack(product: decimal.Decimal, terms: int, width: int) -> list[int]:
    digits = format(product, "f").zfill(terms * width)
    total = len(digits)
    return [int(digits[total - (k + 1) * width: total - k * width]) for k in range(terms)]


def _autoconvolution(a: list[int]) -> list[int]:
    """Exact full autoconvolution of a signed integer sequence."""
    n = len(a)
    terms = 2 * n - 1
    top = max(abs(x) for x in a)
    width = len(str(n * top * top)) + 1      # decimal digits per slot: no carry can cross a slot
    ctx = decimal.Context(prec=decimal.MAX_PREC, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN,
                          traps=[decimal.Inexact, decimal.InvalidOperation])
    p = _pack([x if x > 0 else 0 for x in a], width, ctx)
    q = _pack([-x if x < 0 else 0 for x in a], width, ctx)
    pp = _unpack(ctx.multiply(p, p), terms, width)
    qq = _unpack(ctx.multiply(q, q), terms, width)
    pq = _unpack(ctx.multiply(p, q), terms, width)
    c = [pp[k] + qq[k] - 2 * pq[k] for k in range(terms)]
    if sum(c) != sum(a) ** 2 or c[0] != a[0] ** 2 or c[-1] != a[-1] ** 2:
        raise AssertionError("exact autoconvolution self-check failed")
    return c


def score(candidate: dict) -> dict:
    values = candidate["values"]
    n = len(values)
    a, den = _integers(values)
    total = sum(a)
    if Fraction(total * total, (2 * n * den) ** 2) < MIN_INTEGRAL_SQUARED:
        raise ValueError("function integral is close to zero: (sum(f) * dx)^2 < 1e-9")
    exact = Fraction(2 * n * abs(max(_autoconvolution(a))), total * total)
    return {"score": float(exact), "fraction": exact, "tolerance_abs": TOLERANCE_ABS,
            "method": "exact rational: dyadic -> integers, signed Kronecker substitution multiplied with decimal "
                      "(libmpdec), score = 2n * |max(a*a)| / (sum a)^2"}
