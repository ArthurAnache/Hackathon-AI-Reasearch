"""10c: Spencer discrepancy constant (lower bound). DO NOT MODIFY.

https://teorth.github.io/optimizationproblems/constants/10c.html

C_10c is the least K with disc(A) <= K sqrt(n) for all n and all A in [-1,1]^{n x n},
where disc(A) = min_{x in {±1}^n} ||A x||_inf. Every matrix gives the LOWER bound
disc(A)/sqrt(n) <= C_10c, so a search MAXIMISES

    score = min_{x in {±1}^n} max_i |sum_j a_ij x_j| / sqrt(n)     (higher is better)

computed exhaustively (x and -x give the same value, so 2^(n-1) sign vectors).
"""
from __future__ import annotations

import math

import numpy as np

from .common import InvalidSolution

ID = "10c"
NAME = "Spencer discrepancy constant (lower bound)"
URL = "https://teorth.github.io/optimizationproblems/constants/10c.html"
DIRECTION = "maximize"
BEST_KNOWN = 5 / math.sqrt(8)       # 1.7677669529663689, 8x8 matrix (H2026), proven optimal for n = 8
BEST_KNOWN_SOURCE = "H2026 8x8 {-1,0,1} matrix, disc 5 (5/sqrt 8); page proves this is optimal at n = 8"
SOLUTION_FORMAT = '{"matrix": [[a_11, ..., a_1n], ..., [a_n1, ..., a_nn]]}  entries in [-1,1], 1 <= n <= 20'
MAX_N = 20
CHUNK = 1 << 14


def _matrix(solution) -> np.ndarray:
    if not isinstance(solution, dict) or "matrix" not in solution:
        raise InvalidSolution("solution must be a dict with key 'matrix'")
    try:
        a = np.asarray(solution["matrix"], dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise InvalidSolution(f"matrix is not numeric/rectangular: {e}") from e
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not (1 <= a.shape[0] <= MAX_N):
        raise InvalidSolution(f"matrix must be square n x n with 1 <= n <= {MAX_N}, got shape {a.shape}")
    if not np.all(np.isfinite(a)) or np.any(np.abs(a) > 1):
        raise InvalidSolution("entries must be finite and lie in [-1, 1]")
    return a


def discrepancy(a: np.ndarray) -> float:
    n = a.shape[0]
    best = math.inf
    total = 1 << (n - 1)
    bits = np.arange(n - 1, dtype=np.int64)
    for start in range(0, total, CHUNK):
        idx = np.arange(start, min(start + CHUNK, total), dtype=np.int64)
        signs = np.ones((len(idx), n))
        if n > 1:
            signs[:, 1:] = 1 - 2 * ((idx[:, None] >> bits) & 1)   # x_0 = +1 fixed
        best = min(best, float(np.abs(signs @ a.T).max(axis=1).min()))
    return best


def verify(solution) -> float:
    a = _matrix(solution)
    return discrepancy(a) / math.sqrt(a.shape[0])
