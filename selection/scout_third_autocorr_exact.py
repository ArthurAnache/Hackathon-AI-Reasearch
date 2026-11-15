"""Independent exact scorer for third-autocorrelation-inequality (no numpy, no floating point in the score).

Usage:  .venv/bin/python -I selection/scout_third_autocorr_exact.py SOLUTION.json [SOLUTION2.json ...]

Each binary64 value is read as the exact rational it is (m / 2^k). With a common denominator the values become
integers a_i, and the verifier's quantity is exactly

    score = 2n * max_k (a * a)_k / (sum a)^2        (dx = 1 / (2n); a * a is the full autoconvolution)

The autoconvolution is computed with exact integers by Kronecker substitution: a = p - q with p, q >= 0, each
packed into one big integer with slots wide enough that no carry crosses a slot, then
a*a = p*p - 2 p*q + q*q, slot by slot. This shares no code and no arithmetic with the pinned verifier
(`np.convolve` in float64), so agreement between the two is an independent check. Standard library only.

It prints the exact score as a float (correctly rounded from the exact fraction), its first 40 digits and the
maximizing lag. About 3 s for 25,600 values, about 20 s for 65,536.
"""

from __future__ import annotations

import json
import sys
import time
from fractions import Fraction


def exact_score(values: list[float]) -> tuple[Fraction, int]:
    n = len(values)
    ratios = [float(v).as_integer_ratio() for v in values]
    den = max(d for _, d in ratios)  # every denominator is a power of two, so the largest is a common multiple
    a = [m * (den // d) for m, d in ratios]
    total = sum(a)
    if total == 0:
        raise ValueError("the integral is zero")
    p = [x if x > 0 else 0 for x in a]
    q = [-x if x < 0 else 0 for x in a]
    bits = 2 * max(abs(x) for x in a).bit_length() + n.bit_length() + 2
    width = (bits + 7) // 8  # bytes per slot

    def pack(coeffs: list[int]) -> int:
        return int.from_bytes(b"".join(c.to_bytes(width, "little") for c in coeffs), "little")

    def unpack(product: int) -> list[int]:
        raw = product.to_bytes(2 * n * width, "little")
        return [int.from_bytes(raw[k * width:(k + 1) * width], "little") for k in range(2 * n - 1)]

    big_p, big_q = pack(p), pack(q)
    pp, qq, pq = unpack(big_p * big_p), unpack(big_q * big_q), unpack(big_p * big_q)
    best_k, best = max(((k, pp[k] + qq[k] - 2 * pq[k]) for k in range(2 * n - 1)), key=lambda t: t[1])
    return Fraction(2 * n * best, total * total), best_k


def main() -> None:
    for path in sys.argv[1:]:
        values = json.load(open(path))["values"]
        t0 = time.perf_counter()
        score, lag = exact_score(values)
        print(json.dumps({"solution": path, "n": len(values), "exact_score": float(score),
                          "exact_score_repr": repr(float(score)), "argmax_lag": lag,
                          "exact_score_40_digits": f"{score.numerator * 10 ** 40 // score.denominator}e-40",
                          "seconds": round(time.perf_counter() - t0, 1)}))


if __name__ == "__main__":
    main()
