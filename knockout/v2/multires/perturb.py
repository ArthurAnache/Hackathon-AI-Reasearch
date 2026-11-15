"""Seeded perturbations of a step function f >= 0 (numpy only). They are drawn on CPU from the row's seed, so a
row is the same on any device. A perturbation is a dict {"kind": ..., parameters}:

    {"kind": "lognormal", "sigma": s}                    f_i <- f_i exp(s N(0, 1)); keeps the support
    {"kind": "uniform", "scale": a}                      f_i <- f_i + U(0, a max f): every cell opens
    {"kind": "blend", "block": k, "t": t}                f <- (1 - t) f + t smooth_blocks(f, k): partial V-cycle,
                                                         block masses kept exactly, no random draw
    {"kind": "shuffle", "block": k, "frac": p}           cells permuted inside a share p of the blocks
    {"kind": "shift", "block": k, "max_shift": s, "frac": p}   blocks rolled by a random offset in [-s, s] \\ {0}
    {"kind": "resample", "block": k, "sigma": s, "frac": p}    log-normal kick, then every block rescaled to its
                                                         old mass (support kept, envelope kept exactly)
    {"kind": "redraw", "block": k, "frac": p}            cells of a block redrawn with replacement from the
                                                         block's own values, then rescaled to the old mass
    {"kind": "dilate", "radius": d, "t": t}              zero cells within d cells of the support open at t times
                                                         the largest value within d cells; no random draw
    {"kind": "roll", "by": s}                            cyclic shift by s cells: changes the pooling phase, and is
                                                         score-neutral when the s cells that wrap are zero

"shuffle", "shift", "resample" and "redraw" keep the mass of every block of k cells: they redraw the
micro-structure at a fixed envelope. `frac` < 1 touches a random share of the blocks that hold mass.
"""
from __future__ import annotations

import numpy as np

from knockout.v2.multires.resample import pad_to_multiple, smooth_blocks

KINDS = ("lognormal", "uniform", "blend", "dilate", "roll", "shuffle", "shift", "resample", "redraw")


def local_max(f: np.ndarray, radius: int) -> np.ndarray:
    """The largest value within `radius` cells of each cell (the ends see zeros)."""
    out = f.copy()
    for s in range(1, radius + 1):
        np.maximum(out[s:], f[:-s], out=out[s:])
        np.maximum(out[:-s], f[s:], out=out[:-s])
    return out


def _blocks(f: np.ndarray, block: int):
    n = f.size
    x = pad_to_multiple(np.asarray(f, dtype=np.float64), block).reshape(-1, block).copy()
    return x, n


def _selected(x: np.ndarray, frac: float, rng, n: int) -> np.ndarray:
    """Indices of the blocks to redraw: a share `frac` of the complete blocks that hold mass (a partial last
    block is left alone, so no mass can move into the padding)."""
    live = np.flatnonzero(x[: n // x.shape[1]].sum(1) > 0.0)
    if frac >= 1.0:
        return live
    count = int(round(frac * live.size))
    return np.sort(rng.choice(live, size=count, replace=False)) if count else live[:0]


def _rescale(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    """Rows of `new` rescaled to the mass of the rows of `old`; a row redrawn to zero mass keeps its old cells."""
    m_new, m_old = new.sum(1), old.sum(1)
    ok = m_new > 0.0
    out = old.copy()
    out[ok] = new[ok] * (m_old[ok] / m_new[ok])[:, None]
    return out


def apply(f: np.ndarray, spec: dict, seed: int) -> np.ndarray:
    """One perturbation of the 1-D vector f; returns a new non-negative vector of the same length."""
    kind = spec["kind"]
    rng = np.random.default_rng(seed)
    f = np.asarray(f, dtype=np.float64)
    if kind == "lognormal":
        return f * np.exp(float(spec["sigma"]) * rng.standard_normal(f.size))
    if kind == "uniform":
        return f + rng.uniform(0.0, float(spec["scale"]) * float(f.max()), size=f.size)
    if kind == "blend":
        t = float(spec["t"])
        if not 0.0 <= t <= 1.0:
            raise ValueError("blend needs 0 <= t <= 1")
        return (1.0 - t) * f + t * smooth_blocks(f, int(spec["block"]))
    if kind == "roll":
        return np.roll(f, int(spec["by"]))
    if kind == "dilate":
        radius, t = int(spec["radius"]), float(spec["t"])
        if radius < 1 or t < 0.0:
            raise ValueError("dilate needs radius >= 1 and t >= 0")
        return np.where(f > 0.0, f, t * local_max(f, radius))
    if kind not in KINDS:
        raise ValueError(f"unknown perturbation {kind!r} (known: {KINDS})")
    block = int(spec["block"])
    x, n = _blocks(f, block)
    sel = _selected(x, float(spec.get("frac", 1.0)), rng, n)
    if sel.size:
        old = x[sel]
        if kind == "shuffle":
            new = rng.permuted(old, axis=1)
        elif kind == "shift":
            s = int(spec["max_shift"])
            if not 1 <= s < block:
                raise ValueError("shift needs 1 <= max_shift < block")
            offsets = rng.integers(1, s + 1, size=sel.size) * rng.choice((-1, 1), size=sel.size)
            idx = (np.arange(block)[None, :] - offsets[:, None]) % block
            new = np.take_along_axis(old, idx, 1)
        elif kind == "resample":
            new = _rescale(old * np.exp(float(spec["sigma"]) * rng.standard_normal(old.shape)), old)
        else:  # redraw
            new = _rescale(np.take_along_axis(old, rng.integers(0, block, size=old.shape), 1), old)
        x[sel] = new
    return x.reshape(-1)[:n]


def apply_all(f: np.ndarray, specs, seed: int) -> np.ndarray:
    """The perturbations of `specs` in order; perturbation j is drawn from seed + j."""
    out = np.asarray(f, dtype=np.float64)
    for j, spec in enumerate(specs):
        out = apply(out, spec, seed + j)
    return out


def describe(specs) -> str:
    if not specs:
        return "none"
    parts = []
    for s in specs:
        args = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                         for k, v in s.items() if k != "kind")
        parts.append(f"{s['kind']}({args})")
    return " + ".join(parts)
