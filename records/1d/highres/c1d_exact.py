"""Exact score of a step function for Tao constant 1d (upper bound, lower is better).

    c1d_exact.py SOLUTION(.json|.npy) [--float]   ->  exact fraction, compared STRICTLY with the record fraction

score = 2N * max_k |(v*v)_k| / (sum v)^2 for the exact rational values of the file: integers are used as they are,
floats are converted to their exact binary rational value (common power-of-two denominator), like
`prepare.py verify 1d --exact`.

Check 1: full exact autoconvolution by Kronecker substitution (big integers, (P - Q)^2 = P^2 + Q^2 - 2PQ with
P, Q the positive and negative parts). Check 2 (independent): the deciding knot k* is recomputed as a direct sum
sum_i v_i v_{k*-i} in Python integers; the whole convolution is checked against v(x)^2 at a random point modulo a
61-bit prime, and sum(conv) == (sum v)^2.
"""
import json, math, random, sys
from fractions import Fraction
import numpy as np

RECORD = Fraction(838861227937471890002426104328885370880, 576666392427609677715360509915372739121)
RECORD_SRC = "open PR #345 on teorth/optimizationproblems, 163,840 cells"


def load_values(path):
    if path.endswith(".npy"):
        return [float(x) for x in np.load(path)]
    d = json.load(open(path))
    for key in ("best_values", "solution", "values"):
        if isinstance(d, dict) and key in d:
            d = d[key]
    if isinstance(d, dict):
        d = d["values"]
    return d


def to_ints(vals):
    if all(isinstance(x, int) for x in vals):
        return list(vals)
    fr = [Fraction(x) for x in vals]           # exact value of each float (or int)
    den = math.lcm(*(f.denominator for f in fr))
    return [f.numerator * (den // f.denominator) for f in fr]


def _pack(a, w):
    return int.from_bytes(b"".join(x.to_bytes(w, "little") for x in a), "little")


def _unpack(x, w, n):
    raw = x.to_bytes(w * n, "little")
    return [int.from_bytes(raw[i:i + w], "little") for i in range(0, w * n, w)]


def autoconv_exact(v):
    n = len(v)
    P = [x if x > 0 else 0 for x in v]
    Q = [-x if x < 0 else 0 for x in v]
    m = max(max(P), max(Q), 1)
    w = (n * m * m).bit_length() // 8 + 2
    p, q = _pack(P, w), _pack(Q, w)
    L = 2 * n - 1
    pp, qq, pq = _unpack(p * p, w, L), _unpack(q * q, w, L), _unpack(p * q, w, L)
    return [a + b - 2 * c for a, b, c in zip(pp, qq, pq)]


def check(vals, verbose=True):
    v = to_ints(vals)
    n = len(v)
    s = sum(v)
    if s == 0:
        raise ValueError("sum is zero")
    conv = autoconv_exact(v)
    assert sum(conv) == s * s, "sum(conv) != (sum v)^2"
    mod = (1 << 61) - 1
    x = random.Random(12345).randrange(2, mod - 1)
    def ev(coefs):
        r = 0
        for c in reversed(coefs):
            r = (r * x + c) % mod
        return r
    assert ev(conv) == ev(v) * ev(v) % mod, "random-point modular check failed"
    k = max(range(len(conv)), key=lambda i: abs(conv[i]))
    lo, hi = max(0, k - n + 1), min(k, n - 1)
    direct = sum(v[i] * v[k - i] for i in range(lo, hi + 1))
    assert direct == conv[k], "direct recomputation of the deciding knot disagrees"
    score = Fraction(2 * n * abs(conv[k]), s * s)
    return dict(n=n, score=score, knot=k, beats=score < RECORD)


if __name__ == "__main__":
    r = check(load_values(sys.argv[1]))
    sc = r["score"]
    print(f"n = {r['n']}  deciding knot k = {r['knot']} (checked twice)")
    print(f"exact score = {float(sc):.17g}  ({sc.numerator.bit_length()}-bit / {sc.denominator.bit_length()}-bit)")
    print(f"exact fraction = {sc.numerator}/{sc.denominator}" if sc.numerator.bit_length() < 400 else "exact fraction: (long; use --print)")
    if "--print" in sys.argv:
        print(f"{sc.numerator}/{sc.denominator}")
    print(f"record = {float(RECORD):.17g} ({RECORD_SRC})")
    print(f"strictly below record: {r['beats']}  (difference {float(sc - RECORD):+.3e})")
