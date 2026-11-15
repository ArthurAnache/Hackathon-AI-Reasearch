"""Changes of resolution of a step function f (numpy only, no torch): sum-pooling to a coarser grid, repeat
upsampling to a finer one, and the block-mass profile used to compare envelopes.

Exact facts used by the driver (checked in tests/test_knockout_v2_multires.py against the frozen verifier):
- the score is invariant under f -> a f and under zero-padding;
- the score is invariant under k-fold repetition `np.repeat(f, k)`: the autoconvolution of the repeated vector
  is k times the piecewise-linear interpolation of the coarse autoconvolution at k times finer nodes, and the
  verifier integrates exactly that piecewise-linear function (L2), sums it (L1) and takes its node maximum.
Sum-pooling is not an inverse: it keeps the mass of every block and forgets where the mass sits inside it.
"""
from __future__ import annotations

import numpy as np

from knockout.v2.gpu.objective import exact_score, fft_length


def pad_right(f: np.ndarray, length: int) -> np.ndarray:
    """f (..., n) with zeros appended up to `length` cells (score-neutral)."""
    n = f.shape[-1]
    if length < n:
        raise ValueError(f"cannot pad {n} cells down to {length}")
    if length == n:
        return f
    widths = [(0, 0)] * (f.ndim - 1) + [(0, length - n)]
    return np.pad(f, widths)


def pad_to_multiple(f: np.ndarray, k: int) -> np.ndarray:
    """f with zeros appended so that its length is a multiple of k."""
    n = f.shape[-1]
    return pad_right(f, -(-n // k) * k)


def sum_pool(f: np.ndarray, k: int) -> np.ndarray:
    """Mass of each block of k consecutive cells: (..., n) -> (..., n / k). The length must be a multiple of k."""
    if k < 1:
        raise ValueError("pooling factor must be >= 1")
    n = f.shape[-1]
    if n % k:
        raise ValueError(f"length {n} is not a multiple of {k}: pad first")
    if k == 1:
        return f.copy()
    return f.reshape(f.shape[:-1] + (n // k, k)).sum(-1)


def repeat_upsample(f: np.ndarray, k: int) -> np.ndarray:
    """Every cell repeated k times: (..., n) -> (..., k n). Keeps the score exactly."""
    if k < 1:
        raise ValueError("upsampling factor must be >= 1")
    return np.repeat(f, k, axis=-1)


def smooth_blocks(f: np.ndarray, k: int) -> np.ndarray:
    """f with the mass of every block of k cells spread evenly inside the block (same length, same block masses):
    pooling by k followed by repeat-upsampling by k. A partial last block is averaged over its own cells."""
    f = np.asarray(f, dtype=np.float64)
    n = f.shape[-1]
    full = n - n % k
    out = np.empty_like(f)
    out[..., :full] = repeat_upsample(sum_pool(f[..., :full], k), k) / k
    if full < n:
        out[..., full:] = f[..., full:].mean(-1, keepdims=True)
    return out


def closed_form_score(f: np.ndarray) -> float:
    """The verifier's score of one non-negative vector, in closed form (numpy float64, linear FFT)."""
    f = np.asarray(f, dtype=np.float64)
    return float(exact_score(np, f[None, :], fft_length(f.size))[0])


def loss_terms(f: np.ndarray) -> dict:
    """1 - C split exactly in two terms. With z = c / max c (c the autoconvolution, z = 0 beyond both ends):

        1 - C = [ sum_k z_k (1 - z_k)  +  (1/6) sum_k (z_{k+1} - z_k)^2 ] / sum_k z_k

    "plateau" is the first term (lags below the maximum), "roughness" the second (steps between neighbouring
    lags, the two end steps included)."""
    f = np.asarray(f, dtype=np.float64)
    n, m = f.size, fft_length(f.size)
    spec = np.fft.rfft(f, m)
    c = np.fft.irfft(spec * spec, m)[: 2 * n - 1]
    z = c / c.max()
    dz = np.diff(np.concatenate(([0.0], z, [0.0])))
    total = z.sum()
    return {"plateau": float((z * (1.0 - z)).sum() / total), "roughness": float((dz * dz).sum() / (6.0 * total))}


def repeat_invariance_gap(f: np.ndarray, k: int) -> float:
    """|score(repeat(f, k)) - score(f)| in closed form; the driver asserts it is at rounding level."""
    return abs(closed_form_score(repeat_upsample(f, k)) - closed_form_score(f))


def block_masses(f: np.ndarray, block: int) -> np.ndarray:
    """Share of the total mass in each block of `block` cells (the envelope of f at that scale)."""
    f = np.asarray(f, dtype=np.float64)
    masses = sum_pool(pad_to_multiple(f, block), block)
    return masses / masses.sum(-1, keepdims=True)


def profile_l1(f: np.ndarray, g: np.ndarray, block: int) -> float:
    """L1 distance between the block-mass profiles of f and g (0 = same envelope, 2 = disjoint). The shorter
    vector is zero-padded on the right, so cell i of f is compared with cell i of g."""
    n = max(f.shape[-1], g.shape[-1])
    a = block_masses(pad_right(np.asarray(f, dtype=np.float64), n), block)
    b = block_masses(pad_right(np.asarray(g, dtype=np.float64), n), block)
    return float(np.abs(a - b).sum())
