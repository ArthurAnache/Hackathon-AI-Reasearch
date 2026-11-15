"""Independent, EXACT verifier for circles-rectangle (21 circles, perimeter 4). FROZEN.

Every constraint is checked in exact rational arithmetic on the exact values of the submitted float64 numbers:
  - 21 circles [x, y, r], all finite, r > 0;
  - bounding box: width + height <= 2 (exact; any excess, however small, is a violation);
  - non-overlap for every pair: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 (exact, no square root; tangency is
    allowed, any overlap is a violation).
The score is the exact sum of the radii. This is stricter than any float check: a configuration that only passes
because of float rounding (overlap below 1 ulp, or large coordinates whose float sums lose the radius) is rejected.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

N = 21
TOLERANCE_ABS = 1e-13   # pinned score is float(np.sum(radii)); exact sum differs by at most a few ulp


def score(candidate: dict) -> dict:
    circles = candidate["circles"]
    if len(circles) != N:
        raise ValueError("expected 21 circles")
    xs, ys, rs = [], [], []
    for c in circles:
        x, y, r = (float(t) for t in c)
        if not all(math.isfinite(t) for t in (x, y, r)):
            raise ValueError("non-finite coordinate")
        x, y, r = Fraction(x), Fraction(y), Fraction(r)
        if r <= 0:
            raise ValueError("radius must be positive")
        xs.append(x)
        ys.append(y)
        rs.append(r)
    width = max(x + r for x, r in zip(xs, rs)) - min(x - r for x, r in zip(xs, rs))
    height = max(y + r for y, r in zip(ys, rs)) - min(y - r for y, r in zip(ys, rs))
    if width + height > 2:
        raise ValueError(f"perimeter exceeded: width + height - 2 = {float(width + height - 2):.3e} (exact)")
    for i, j in itertools.combinations(range(N), 2):
        if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 < (rs[i] + rs[j]) ** 2:
            raise ValueError(f"circles {i} and {j} overlap (exact check)")
    total = sum(rs, Fraction(0))
    return {"score": float(total), "fraction": total, "tolerance_abs": TOLERANCE_ABS,
            "method": "exact Fraction feasibility (perimeter, squared distances) and exact sum of radii"}
