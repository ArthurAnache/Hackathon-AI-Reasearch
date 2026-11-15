"""Trusted math kit for second-autocorrelation-inequality plugins. Plugins may import this module, nothing else local.

Exact verifier arithmetic (problems/second-autocorrelation-inequality/verifier_pinned.py), for c = f*f of length m:
    C(f) = Q(c) / (3 S^2 M),   Q(c) = 2 sum c_i^2 + sum c_i c_{i+1},   S = sum f,   M = max c.
The kit gives plugins this score and the gradient of a smoothed log-score, so that a plugin cannot get the verifier
arithmetic wrong. The parent never trusts a plugin's own score: it re-scores the returned array with the frozen
verifier.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


def conv_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = len(a) + len(b) - 1
    L = next_fast_len(m, real=True)
    return irfft(rfft(a, L) * rfft(b, L), L)[:m]


# Back-compat alias used by older draft plugins in this worktree.
_conv_full = conv_full


def autoconv(f: np.ndarray) -> np.ndarray:
    L = next_fast_len(2 * len(f) - 1, real=True)
    F = rfft(f, L)
    return irfft(F * F, L)[: 2 * len(f) - 1]


def correlate_valid(gamma: np.ndarray, f: np.ndarray) -> np.ndarray:
    """out_j = sum_k gamma_{j+k} f_k for j = 0..n-1 (adjoint of f -> f*f, up to the factor 2)."""
    n = len(f)
    return conv_full(gamma, f[::-1])[n - 1: n - 1 + n]


def q_form(c: np.ndarray) -> float:
    return float(2.0 * np.dot(c, c) + np.dot(c[:-1], c[1:]))


def score(f: np.ndarray) -> float:
    """Fast score with the verifier's formula (FFT, not oaconvolve: may differ from the pinned value by ~1e-15)."""
    f = np.maximum(np.asarray(f, dtype=np.float64), 0.0)
    c = autoconv(f)
    s = float(f.sum())
    return q_form(c) / (3.0 * s * s * float(c.max()))


def smoothed_log_score(f: np.ndarray, beta_abs: float) -> tuple[float, np.ndarray]:
    """log C_beta(f) and its gradient, with max c replaced by LSE_beta(c) = log(sum exp(beta_abs c)) / beta_abs >= max c.

    C_beta <= C everywhere, and C_beta -> C as beta_abs -> inf. beta_abs has units 1/c: pass beta / max(c0).
    """
    c = autoconv(f)
    s = float(f.sum())
    q = q_form(c)
    z = beta_abs * c
    zmax = float(z.max())
    w = np.exp(z - zmax)
    sw = float(w.sum())
    m_beta = (zmax + math.log(sw)) / beta_abs
    p = w / sw
    dq = 4.0 * c
    dq[1:] += c[:-1]
    dq[:-1] += c[1:]
    gamma = dq / q - p / m_beta
    grad = 2.0 * correlate_valid(gamma, f) - 2.0 / s
    val = math.log(q) - 2.0 * math.log(s) - math.log(m_beta) - math.log(3.0)
    return val, grad


def near_max_set(f: np.ndarray, rel_gap: float) -> np.ndarray:
    c = autoconv(np.maximum(f, 0.0))
    return np.flatnonzero(c >= c.max() * (1.0 - rel_gap))


def bilinear(x: np.ndarray, y: np.ndarray) -> float:
    """B(x, y) with B(c, c) = q_form(c)."""
    return float(2.0 * np.dot(x, y) + 0.5 * (np.dot(x[:-1], y[1:]) + np.dot(x[1:], y[:-1])))


def q_gradient(c: np.ndarray) -> np.ndarray:
    """dQ/dc."""
    dq = 4.0 * c
    dq[1:] += c[:-1]
    dq[:-1] += c[1:]
    return dq


def equal_mass_packets(f: np.ndarray, n_packets: int, rng=None, jitter: float = 0.0) -> np.ndarray:
    """Boundaries e_0 = 0 < ... < e_P = n of contiguous packets holding ~equal mass of f (empty packets merged).

    With rng and jitter in [0, 0.5), each internal boundary moves by up to jitter x (one packet's mass).
    """
    n = len(f)
    cm = np.cumsum(f)
    total = float(cm[-1])
    levels = (np.arange(1, n_packets) / n_packets)
    if rng is not None and jitter > 0:
        levels = levels + rng.uniform(-jitter, jitter, size=levels.size) / n_packets
    cuts = np.searchsorted(cm, np.clip(levels, 0, 1) * total, side="right")

    def mass(a: int, b: int) -> float:
        return float(cm[b - 1] - (cm[a - 1] if a else 0.0))

    kept = [0]
    for e in np.unique(cuts):
        if kept[-1] < e < n and mass(kept[-1], int(e)) > 0:
            kept.append(int(e))
    if len(kept) > 1 and mass(kept[-1], n) <= 0:
        kept.pop()
    return np.asarray(kept + [n], dtype=np.int64)


def packet_vectors(f: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For packets b_j = f restricted to [e_j, e_{j+1}): masses m_j and responses R_j = 2 (f * b_j) (one row each).

    A direction h = sum_j z_j b_j moves c = f*f by 2 alpha (f*h) + alpha^2 (h*h), with 2 (f*h) = sum_j z_j R_j.
    Memory: len(edges) - 1 rows of length 2n - 1.
    """
    n = len(f)
    L = next_fast_len(2 * n - 1, real=True)
    F = rfft(f, L)
    rows, masses = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        bj = np.zeros(n)
        bj[a:b] = f[a:b]
        masses.append(float(bj.sum()))
        rows.append(2.0 * irfft(F * rfft(bj, L), L)[: 2 * n - 1])
    return np.asarray(masses), np.vstack(rows)


def exact_step_scores(f: np.ndarray, c: np.ndarray, h: np.ndarray, alphas) -> list[tuple[float, float]]:
    """Exact scores of f + alpha h for each alpha, from c_alpha = c + 2 alpha (f*h) + alpha^2 (h*h).

    Steps that make f + alpha h negative anywhere (beyond -1e-12 relative) are skipped.
    """
    u = 2.0 * conv_full(f, h)
    v = autoconv(h)
    s0, dh = float(f.sum()), float(h.sum())
    fmax = float(f.max())
    out = []
    for a in alphas:
        if float(np.min(f + a * h)) < -1e-12 * fmax:
            continue
        ca = c + a * u + (a * a) * v
        s = s0 + a * dh
        if s <= 0:
            continue
        out.append((float(a), q_form(ca) / (3.0 * s * s * float(ca.max()))))
    return out
