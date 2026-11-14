"""Independent, EXACT verifier for second-autocorrelation-inequality. FROZEN.

Shares no code with the pinned verifier, and uses no floating-point FFT. It works from the continuous definition.

1. Every float64 is a dyadic rational, so the values become integers a_i over one common power of two:
   f_i = a_i / D.
2. f is a step function with n cells of width h. Its autoconvolution g = f*f is piecewise linear with knots at
   multiples of h. At the knots, g equals h * c_k / D^2, where c = a * a is the integer autoconvolution. g is zero at
   both ends.
3. For such a g, the exact integrals are
       ||g||_2^2 = (h/3) * sum over segments of (y_l^2 + y_l y_r + y_r^2)
       ||g||_1   = h * sum c_k   (up to the common factors)
       ||g||_inf = max c_k
   All h and D factors cancel, which gives the exact rational
       C = (2 * sum c_k^2 + sum c_k c_{k+1}) / (3 * sum c_k * max c_k).
4. c is computed exactly by Kronecker substitution: the a_i are packed into one huge base-10 integer, squared with
   the standard-library decimal module (libmpdec, exact integer arithmetic), and unpacked. Self-checks:
   sum(c) == sum(a)^2, c_0 == a_0^2, c_last == a_last^2.

Admissibility follows the problem statement: finite numbers; any value < -1e-6 rejects the candidate; values in
[-1e-6, 0) count as 0; the integral must be positive.

Cost: about 5 s and ~1 GB of memory for 2,000,000 values.
"""

from __future__ import annotations

import decimal
import math
from fractions import Fraction

NEG_TOL = 1e-6
TOLERANCE_ABS = 1e-12   # accepted |exact - pinned float64| (observed 2e-16 on the frozen certificate)


def _integers(values) -> list[int]:
    ratios = []
    for x in values:
        v = float(x)
        if not math.isfinite(v):
            raise ValueError("non-finite value")
        if v < -NEG_TOL:
            raise ValueError("function must be nonnegative (a value is below -1e-6)")
        ratios.append(v.as_integer_ratio() if v > 0 else (0, 1))
    den = max(d for _, d in ratios)          # all denominators are powers of two
    return [n * (den // d) for n, d in ratios]


def _autoconvolution(a: list[int]) -> list[int]:
    n = len(a)
    m = 2 * n - 1
    top = max(a)
    width = len(str(n * top * top)) + 1      # decimal digits per slot: no carry can cross a slot
    packed = "".join(str(x).zfill(width) for x in reversed(a))
    ctx = decimal.Context(prec=decimal.MAX_PREC, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN,
                          traps=[decimal.Inexact, decimal.InvalidOperation])
    big = ctx.create_decimal(packed)
    digits = format(ctx.multiply(big, big), "f").zfill(m * width)
    total = len(digits)
    c = [int(digits[total - (k + 1) * width: total - k * width]) for k in range(m)]
    if sum(c) != sum(a) ** 2 or c[0] != a[0] ** 2 or c[-1] != a[-1] ** 2:
        raise AssertionError("exact autoconvolution self-check failed")
    return c


def score(candidate: dict) -> dict:
    values = candidate["values"]
    a = _integers(values)
    if not any(a):
        raise ValueError("function must have positive integral")
    c = _autoconvolution(a)
    num = 2 * sum(x * x for x in c) + sum(c[k] * c[k + 1] for k in range(len(c) - 1))
    den = 3 * sum(c) * max(c)
    exact = Fraction(num, den)
    return {"score": float(exact), "fraction": exact, "tolerance_abs": TOLERANCE_ABS,
            "method": "exact rational: dyadic -> integers, Kronecker substitution squared with decimal (libmpdec), "
                      "closed-form piecewise-linear integrals"}
