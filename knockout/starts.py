"""Trusted start generators for the start portfolio (brief §4.4): cold starts, perturbations and basin descriptors.

Deterministic for a given seed (numpy Generator). Every candidate returned is in the canonical JSON form the frozen
verifier reads (lists of Python floats):

| Problem | Canonical candidate |
|---|---|
| `toy-circles-26` | {"circles": [[x, y, r], ...]}, 26 rows, inside the unit square |
| `circles-rectangle` | {"circles": [[x, y, r], ...]}, 21 rows, bounding box width + height <= 2 |
| `second-autocorrelation-inequality` | {"values": [...]}, non-negative |

Validity: the circle starts are made valid when it is cheap. Cold starts use radius 1e-3, capped at just under half
the distance to the nearest centre, so they pass the verifier unless two random centres coincide (probability 0).
Perturbed candidates shrink radii until the verifier's exact rules pass (toy: exact Fractions; circles-rectangle:
the pinned verifier's float overlap test with no tolerance and its exact Fraction perimeter). A start the verifier
still rejects is allowed by the portfolio (the program must repair it) and is logged with score None by the caller.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

TOY = "toy-circles-26"
CR = "circles-rectangle"
SA = "second-autocorrelation-inequality"
N_CIRCLES = {TOY: 26, CR: 21}
COLD_RADIUS = 1e-3
_SHRINK = 1e-12          # relative margin left on every tight constraint by the repair


def _check_id(problem_id: str) -> str:
    if problem_id in ("toy", TOY):
        return TOY
    if problem_id in (CR, SA):
        return problem_id
    raise ValueError(f"no start generator for problem {problem_id!r}")


# ---------------------------------------------------------------------------------------------------- circles


def _circles(problem_id: str, candidate) -> np.ndarray:
    n = N_CIRCLES[problem_id]
    try:
        a = np.asarray(candidate["circles"], dtype=float)
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError('candidate must be {"circles": [[x, y, r], ...]}') from exc
    if a.shape != (n, 3):
        raise ValueError(f"expected {n} circles of [x, y, r], got shape {a.shape}")
    return a


def _canonical_circles(p: np.ndarray, r: np.ndarray) -> dict:
    return {"circles": [[float(x), float(y), float(rr)] for (x, y), rr in zip(p, r)]}


def _overlap_free(p: np.ndarray, r: np.ndarray, exact: bool) -> bool:
    """No overlap. exact=True: dx^2 + dy^2 >= (ri + rj)^2 in Fractions (toy). False: the pinned circles-rectangle
    rule, float64 sqrt(dx^2 + dy^2) < ri + rj rejects."""
    n = len(r)
    i, j = np.triu_indices(n, 1)
    d = np.sqrt((p[i, 0] - p[j, 0]) ** 2 + (p[i, 1] - p[j, 1]) ** 2)
    if not exact:
        return not bool((d < r[i] + r[j]).any())
    if bool((d < (r[i] + r[j]) * (1 - 1e-9)).any()):
        return False
    q = [(Fraction(float(x)), Fraction(float(y)), Fraction(float(rr))) for (x, y), rr in zip(p, r)]
    return all((q[a][0] - q[b][0]) ** 2 + (q[a][1] - q[b][1]) ** 2 >= (q[a][2] + q[b][2]) ** 2
               for a, b in zip(i.tolist(), j.tolist()))


def _shrink_pairs(p: np.ndarray, r: np.ndarray, exact: bool) -> np.ndarray:
    """Scale both radii of every overlapping pair to just under the centre distance. One sequential pass is enough
    (radii only shrink, so a fixed pair stays fixed); the loop is a guard against float rounding."""
    r = r.copy()
    n = len(r)
    for _ in range(5):
        if _overlap_free(p, r, exact):
            return r
        for a in range(n):
            for b in range(a + 1, n):
                d = math.sqrt((p[a, 0] - p[b, 0]) ** 2 + (p[a, 1] - p[b, 1]) ** 2)
                if r[a] + r[b] > d * (1 - _SHRINK):
                    s = d * (1 - _SHRINK) / (r[a] + r[b])
                    r[a] *= s
                    r[b] *= s
    return r


def _exact_half_perimeter(p: np.ndarray, r: np.ndarray) -> Fraction:
    """Bounding-box width + height, in exact Fractions of the float values (the pinned verifier's rule)."""
    xs, ys, rs = ([Fraction(float(v)) for v in col] for col in (p[:, 0], p[:, 1], r))
    width = max(x + rr for x, rr in zip(xs, rs)) - min(x - rr for x, rr in zip(xs, rs))
    height = max(y + rr for y, rr in zip(ys, rs)) - min(y - rr for y, rr in zip(ys, rs))
    return width + height


def _fit_perimeter(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Shrink radii until the exact bounding-box width + height is <= 2."""
    r = r.copy()
    for _ in range(60):
        excess = _exact_half_perimeter(p, r) - 2
        if excess <= 0:
            return r
        r *= 1 - max(float(excess) / max(float(r.sum()), 1e-300) * 2.0, _SHRINK)
    return r


def _cold_circles(problem_id: str, k: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    n = N_CIRCLES[problem_id]
    out = []
    for _ in range(k):
        if problem_id == TOY:
            w = h = 1.0
            p = rng.uniform(0.1, 0.9, (n, 2))
        else:
            aspect = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))   # width / height
            w, h = 2 * aspect / (1 + aspect), 2 / (1 + aspect)
            # centres at least 0.01 from the rectangle's sides, so the bounding box stays inside it
            p = np.stack([rng.uniform(0.01, w - 0.01, n), rng.uniform(0.01, h - 0.01, n)], 1)
        i, j = np.triu_indices(n, 1)
        d = np.sqrt(((p[i] - p[j]) ** 2).sum(1))
        nearest = np.full(n, np.inf)
        np.minimum.at(nearest, i, d)
        np.minimum.at(nearest, j, d)
        r = np.minimum(COLD_RADIUS, nearest / 2 * (1 - 1e-9))
        out.append(_canonical_circles(p, r))
    return out


def _perturb_circles(problem_id: str, candidate, strength: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    a = _circles(problem_id, candidate)
    p, r = a[:, :2].copy(), np.abs(a[:, 2])
    if problem_id == TOY:
        lo, hi = np.zeros(2), np.ones(2)
    else:
        lo = (p - r[:, None]).min(0)
        hi = (p + r[:, None]).max(0)
        side = hi - lo
        if side.sum() > 2 * (1 - _SHRINK):           # an invalid input: shrink its box about its centre
            mid = (lo + hi) / 2
            side = side * 2 * (1 - _SHRINK) / side.sum()
            lo, hi = mid - side / 2, mid + side / 2
    size = float((hi - lo).max())
    p = p + rng.normal(0.0, strength * size, p.shape)
    # reflect into the box (clipping would stack centres on the sides and corners, where they coincide)
    a, span = lo + 1e-9 * size, (hi - lo) * (1 - 2e-9)
    t = np.mod(p - a, 2 * span)
    p = a + span - np.abs(span - t)
    # containment in the box, then no overlap, then (circles-rectangle) the exact perimeter
    r = np.minimum.reduce([r, p[:, 0] - lo[0], hi[0] - p[:, 0], p[:, 1] - lo[1], hi[1] - p[:, 1]]) * (1 - _SHRINK)
    r = np.maximum(r, 1e-12)
    r = _shrink_pairs(p, r, exact=problem_id == TOY)
    if problem_id == CR:
        r = _fit_perimeter(p, r)
    return _canonical_circles(p, r)


def _descriptors_circles(problem_id: str, candidate) -> dict[str, float]:
    a = _circles(problem_id, candidate)
    p, r = a[:, :2], a[:, 2]
    q = np.quantile(np.sort(r), [0.25, 0.5, 0.75, 1.0])
    # Radii relative to the median radius: scale-free, so they describe the structure of a packing, not its size.
    # Absolute quantiles put every polished packing in the top bin, because the cold starts (radius 1e-3) set
    # the low end of the bin range (seen on the first circles-rectangle run, 2026-11-15).
    med = float(q[1]) if q[1] > 0 else 1.0
    out = {"r_q25_rel": float(q[0]) / med, "r_q75_rel": float(q[2]) / med, "r_max_rel": float(q[3]) / med}
    if problem_id == TOY:
        gap = np.minimum.reduce([p[:, 0] - r, 1 - p[:, 0] - r, p[:, 1] - r, 1 - p[:, 1] - r])
        out["n_boundary"] = float((np.abs(gap) <= 1e-9).sum())
    else:
        side = (p + r[:, None]).max(0) - (p - r[:, None]).min(0)
        out["aspect"] = float(side[0] / side[1]) if side[1] > 0 else float("inf")
        out["n_small"] = float((r < 0.5 * np.median(r)).sum())
    return out


# ---------------------------------------------------------------------------------------- second autocorrelation

SA_COLD_LEN = 50_000


def _values(candidate) -> np.ndarray:
    try:
        v = np.asarray(candidate["values"], dtype=float)
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError('candidate must be {"values": [...]}') from exc
    if v.ndim != 1 or v.size == 0:
        raise ValueError("values must be a non-empty 1-D list")
    return v


def _runs(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) indices of the maximal runs of positive values."""
    pos = np.concatenate([[0], (v > 0).astype(np.int8), [0]])
    edges = np.diff(pos)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _cold_sa(k: int, seed: int) -> list:
    """Even k: log-normal positive vector. Odd k: a coarse comb (blocks of ones on a random period)."""
    rng = np.random.default_rng(seed)
    out = []
    for idx in range(k):
        if idx % 2 == 0:
            v = rng.lognormal(0.0, 1.0, SA_COLD_LEN)
        else:
            period = int(rng.integers(50, 500))
            width = max(1, int(period * rng.uniform(0.1, 0.5)))
            v = ((np.arange(SA_COLD_LEN) % period) < width).astype(float)
        out.append({"values": v.tolist()})
    return out


def _perturb_sa(candidate, strength: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    v = np.maximum(_values(candidate), 0.0)
    starts, ends = _runs(v)
    if len(starts):
        pick = rng.choice(len(starts), max(1, int(round(0.05 * len(starts)))), replace=False)
        for s, e, g in zip(starts[pick], ends[pick], np.exp(rng.normal(0.0, strength, len(pick)))):
            v[s:e] *= g
    return {"values": v.tolist()}


def _descriptors_sa(candidate) -> dict[str, float]:
    v = np.maximum(_values(candidate), 0.0)
    starts, ends = _runs(v)
    total = float(v.sum())
    csum = np.concatenate([[0.0], np.cumsum(v)])
    largest = float((csum[ends] - csum[starts]).max()) if len(starts) else 0.0
    return {"support_fraction": float((v > 0).mean()), "runs_per_1000": 1000.0 * len(starts) / v.size,
            "largest_run_mass_share": largest / total if total > 0 else 0.0}


# ------------------------------------------------------------------------------------------------- public API


def cold_starts(problem_id: str, k: int, seed: int) -> list:
    """k weak canonical starts, deterministic for `seed` (valid or not; see the module docstring)."""
    pid = _check_id(problem_id)
    return _cold_sa(k, seed) if pid == SA else _cold_circles(pid, k, seed)


def perturb(problem_id: str, candidate, strength: float, seed: int):
    """`candidate` with its parts moved by noise of scale `strength` (relative to the container size)."""
    pid = _check_id(problem_id)
    return _perturb_sa(candidate, strength, seed) if pid == SA else _perturb_circles(pid, candidate, strength, seed)


def descriptors(problem_id: str, candidate) -> dict[str, float]:
    """Basin descriptors (brief §4.4). Works on invalid candidates of the right shape."""
    pid = _check_id(problem_id)
    return _descriptors_sa(candidate) if pid == SA else _descriptors_circles(pid, candidate)
