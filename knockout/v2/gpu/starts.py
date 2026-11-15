"""Changes applied to a warm start before a chain (numpy, on CPU, so they are the same on any device).

- kick: independent log-normal factor per cell (a stochastic reheat);
- coarsen_blocks: replace every block of k cells by its mean (a reheat in resolution). Solutions 2782 and 2783
  have the same mass per block of 1,024 cells within 1 % and differ at the scale of a few cells, so this erases
  exactly the part of a solution in which the two records differ, and keeps the envelope.
Both keep f >= 0. Neither is score-preserving: the chain that follows has to earn the score back.
"""
from __future__ import annotations

import numpy as np


def kick(f: np.ndarray, sigma: float, seed: int, block: int = 0) -> np.ndarray:
    """f_i * exp(sigma * N(0, 1)), drawn from `seed`. With block > 0 every block of `block` cells is rescaled to
    the mass it had before the kick, so the kick only moves mass inside the blocks and the envelope is kept.
    (2782 and 2783 agree on block masses within 0.9 % at 1,000 cells and differ by 22.5 % cell by cell.)"""
    if sigma <= 0:
        return f
    out = f * np.exp(sigma * np.random.default_rng(seed).standard_normal(f.size))
    if block > 0:
        n = f.size
        pad = (-n) % block
        before = (np.concatenate([f, np.zeros(pad)]) if pad else f).reshape(-1, block).sum(1, keepdims=True)
        kicked = (np.concatenate([out, np.zeros(pad)]) if pad else out).reshape(-1, block)
        after = kicked.sum(1, keepdims=True)
        out = (kicked * np.where(after > 0, before / np.where(after > 0, after, 1.0), 0.0)).reshape(-1)[:n]
    return out


def sqrt_kick(f: np.ndarray, sigma: float, seed: int, window: int) -> np.ndarray:
    """Additive kick in square-root coordinates, zero cells included, with the mass of every window restored:
    f_i <- (sqrt(f_i) + sigma * sqrt(mu_b) * N(0, 1))^2, mu_b the mean value of the window of `window` cells.
    Unlike the multiplicative kick it opens the zero cells of every window that holds mass; windows with no
    mass stay empty. (Suggested by a second model through the coordinator, 2026-11-14; not our own idea.)"""
    if sigma <= 0 or window <= 0:
        return f
    n = f.size
    pad = (-n) % window
    blocks = (np.concatenate([f, np.zeros(pad)]) if pad else f).reshape(-1, window)
    mass = blocks.sum(1, keepdims=True)
    xi = np.random.default_rng(seed).standard_normal(blocks.shape)
    out = (np.sqrt(blocks) + sigma * np.sqrt(mass / window) * xi) ** 2
    if pad:
        out[-1, window - pad:] = 0.0
    after = out.sum(1, keepdims=True)
    out = out * np.where(after > 0, mass / np.where(after > 0, after, 1.0), 0.0)
    return out.reshape(-1)[:n]


def along_records(f: np.ndarray, older: np.ndarray, newer: np.ndarray, t: float) -> np.ndarray:
    """f moved by t times the step between two records: f/sum(f) + t (newer/sum - older/sum), clipped at 0.

    The step from solution 2782 to solution 2783 is the one rearrangement known to gain (2.3e-5); it keeps the
    envelope by construction, since both records have the same block masses. From `newer`, t > 0 extrapolates
    beyond it and t < 0 goes back towards `older`; from `older`, 0 < t < 1 interpolates."""
    if t == 0:
        return f
    step = newer / newer.sum() - older / older.sum()
    return np.maximum(f / f.sum() + t * step, 0.0)


def coarsen_blocks(f: np.ndarray, k: int, masked: bool) -> np.ndarray:
    """Block means over blocks of k cells (the last block may be shorter). The total mass is unchanged.

    masked=False: every cell of a block gets the block mean, so zero cells inside a block that holds mass open
    (run edges move out to the block boundaries).
    masked=True: the support is kept; every open cell gets the mean of the open cells of its block.
    """
    if k <= 1:
        return f
    n = f.size
    pad = (-n) % k
    g = np.concatenate([f, np.zeros(pad)]) if pad else f
    blocks = g.reshape(-1, k)
    mass = blocks.sum(1, keepdims=True)
    if masked:
        is_open = blocks > 0
        count = is_open.sum(1, keepdims=True)
        out = np.where(is_open, mass / np.maximum(count, 1), 0.0)
    else:
        size = np.full((blocks.shape[0], 1), float(k))
        if pad:
            size[-1, 0] = k - pad
        out = np.broadcast_to(mass / size, blocks.shape).copy()
        if pad:
            out[-1, k - pad:] = 0.0
    return out.reshape(-1)[:n]
