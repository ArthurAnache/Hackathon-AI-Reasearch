"""Shared helpers for the frozen verifiers. DO NOT MODIFY (hash-locked by prepare.py)."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


class InvalidSolution(ValueError):
    """The candidate breaks a constraint of the problem. It gets no score."""


def float_vector(solution, key: str, max_len: int, min_len: int = 1) -> np.ndarray:
    if not isinstance(solution, dict) or key not in solution:
        raise InvalidSolution(f"solution must be a dict with key {key!r}")
    raw = solution[key]
    if not isinstance(raw, (list, tuple)):
        raise InvalidSolution(f"{key!r} must be a list of numbers")
    if not (min_len <= len(raw) <= max_len):
        raise InvalidSolution(f"{key!r} must have between {min_len} and {max_len} entries, got {len(raw)}")
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise InvalidSolution(f"{key!r} contains a non-number: {e}") from e
    if arr.ndim != 1:
        raise InvalidSolution(f"{key!r} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise InvalidSolution(f"{key!r} contains NaN or infinity")
    return arr


def convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full linear convolution. Direct sum for small inputs, FFT for large ones."""
    if min(len(a), len(b)) <= 2048:
        return np.convolve(a, b)
    n = len(a) + len(b) - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b, size), size)[:n]


# ----------------------------------------------------------------- exact arithmetic
# Used by `python prepare.py verify --exact` to certify a candidate record.
# Floats are dyadic rationals, so Fraction(x) is the exact value of the double.

def to_integers(values) -> tuple[list[int], int]:
    fr = [Fraction(v) if isinstance(v, (Fraction, int)) else Fraction(float(v)) for v in values]
    den = math.lcm(*(f.denominator for f in fr))
    return [f.numerator * (den // f.denominator) for f in fr], den


def _poly_mul_nonneg(a: list[int], b: list[int]) -> list[int]:
    """Exact product of polynomials with non-negative integer coefficients (Kronecker substitution)."""
    length = len(a) + len(b) - 1
    if not any(a) or not any(b):
        return [0] * length
    bound = min(len(a), len(b)) * max(a) * max(b)
    width = (bound.bit_length() + 8) // 8 + 1
    pa = int.from_bytes(b"".join(x.to_bytes(width, "little") for x in a), "little")
    pb = pa if a is b else int.from_bytes(b"".join(x.to_bytes(width, "little") for x in b), "little")
    raw = (pa * pb).to_bytes(width * length, "little")
    return [int.from_bytes(raw[i:i + width], "little") for i in range(0, len(raw), width)]


def exact_convolve(a: list[int], b: list[int]) -> list[int]:
    """Exact convolution of signed integer sequences."""
    ap, an = [max(x, 0) for x in a], [max(-x, 0) for x in a]
    bp, bn = [max(x, 0) for x in b], [max(-x, 0) for x in b]
    pp, nn = _poly_mul_nonneg(ap, bp), _poly_mul_nonneg(an, bn)
    pn, np_ = _poly_mul_nonneg(ap, bn), _poly_mul_nonneg(an, bp)
    out = [w + x - y - z for w, x, y, z in zip(pp, nn, pn, np_)]
    assert sum(out) == sum(a) * sum(b), "exact convolution self-check failed"
    return out
