"""Independent exact check of a 1d certificate (coordinator's own code; shares nothing with attack/c1d_exact.py).
score = 2N * max_k |(v*v)_k| / (sum v)^2, v integers. Autoconvolution by Kronecker substitution on v = P - Q."""
import json, sys, time
from fractions import Fraction
t0 = time.time()
d = json.load(open(sys.argv[1]))
v = d["values"] if isinstance(d, dict) else d
assert all(isinstance(x, int) and not isinstance(x, bool) for x in v), "values are not all integers"
n = len(v); S = sum(v); assert S != 0
mx = max(abs(x) for x in v)
B = 2 * mx.bit_length() + n.bit_length() + 4          # bits per slot: every coefficient of P*P, P*Q, Q*Q fits
def pack(a):
    out = 0
    for i in range(len(a) - 1, -1, -1):
        out = (out << B) | a[i]
    return out
P = pack([x if x > 0 else 0 for x in v]); Q = pack([-x if x < 0 else 0 for x in v])
mask = (1 << B) - 1
def coeffs(z):
    return [(z >> (B * k)) & mask for k in range(2 * n - 1)]
pp, pq, qq = coeffs(P * P), coeffs(P * Q), coeffs(Q * Q)
c = [pp[k] - 2 * pq[k] + qq[k] for k in range(2 * n - 1)]
assert sum(c) == S * S, "sum of the autoconvolution must equal (sum v)^2"
k = max(range(2 * n - 1), key=lambda i: abs(c[i]))
direct = sum(v[i] * v[k - i] for i in range(max(0, k - n + 1), min(k, n - 1) + 1))   # deciding knot, summed directly
assert direct == c[k], "deciding knot disagrees with the direct sum"
for kk in (0, 1, n - 1, 2 * n - 2, (7 * n) // 5):                                    # spot checks at other knots
    assert c[kk] == sum(v[i] * v[kk - i] for i in range(max(0, kk - n + 1), min(kk, n - 1) + 1))
score = Fraction(2 * n * abs(c[k]), S * S)
rec345 = Fraction(838861227937471890002426104328885370880, 576666392427609677715360509915372739121)
rec270 = Fraction(646233848324886328286774558720, 444246710235775351931155445761)
near = sum(1 for x in c if abs(x) * 10**9 >= abs(c[k]) * (10**9 - 1))
print("cells", n, "| max|v| bits", mx.bit_length(), "| negatives", sum(x < 0 for x in v), "| deciding knot", k, "sign", "+" if c[k] > 0 else "-")
print("exact score      ", float(score), "=", f"{score.numerator}/{score.denominator}"[:90], "...")
print("PR #345 claim    ", float(rec345), "| strictly below:", score < rec345, "| margin", float(rec345 - score))
print("PR #270 claim    ", float(rec270), "| strictly below:", score < rec270)
print("knots within 1e-9 of the max:", near, "| seconds", round(time.time() - t0, 1))
