"""Hand-written move families: `move(f, rng, strength) -> f'`.

A move may lower the score: it is followed by a polish (knockout/v2/polish.py), and what counts is where the polish
ends. Every move keeps len(f), f >= 0 and sum(f) > 0, and is deterministic given `rng`.

A "run" is a maximal block of consecutive non-zero cells. The live #1 of 2026-11-14 (solution 2783) has 482,716
non-zero cells in 20,962 runs.

Families and the meaning of `strength`:

| Family | What it does | strength |
|---|---|---|
| `reheat` | nothing (the polish schedule starts at a low beta) | unused |
| `lognormal` | multiplies every non-zero cell by exp(strength * N(0, 1)) | sigma |
| `region_noise` | same noise, only on the ramp (5 % to 65 % of the grid) and on the end block (last 5 %) | sigma |
| `threshold` | zeroes the smallest cells | share of the mass removed |
| `open_boundary` | opens the zero cell next to a run boundary (each with probability 1/2) | amplitude / neighbour |
| `swap_support` | `threshold` then `open_boundary` at 1e-4 of the neighbour | share of the mass removed |
| `split_runs` | zeroes one interior cell of a run and gives its mass to its two neighbours | share of runs split |
| `merge_runs` | fills the gap (at most 4 cells) between two neighbouring runs | share of such gaps filled |
| `transfer_mass` | moves mass from one window (1 % of the grid) to another | share of the window's mass |
| `shift_runs` | shifts the runs of one window by 1 or 2 cells | window width / grid |
| `open_gradient` | opens the zero cells where the gradient of the smoothed score is largest | cells opened / support |
| `envelope` | continues the slow drift of the leader's envelope (needs the reference solutions) | drift, in units of 1e-5 of the leader's score gain |
| `clear_band` | scales down the sparse band between 65 % and 90 % of the grid | share of the band's mass removed |

`region_noise` and `swap_support` come from comparing solutions 2782 and 2783 (the previous and the current #1 of
the same team): 25,554 small cells holding 0.7 % of the mass disappeared, 23,191 cells appeared, and on common
cells the ratio has quartiles 0.79 and 1.24 (a log-normal with sigma about 0.32), all of it on the ramp and the end
block.

`envelope` and `clear_band` come from the four successive solutions of that team on 2026-11-14 (2780, 2781, 2782,
2783; +1.3e-5, +1.3e-5, +2.3e-5). Cell by cell their steps look like a random walk (cosine between successive
displacements 0.03 to 0.10), but the displacement of the mass of blocks of 65,536 cells keeps its direction
(cosine 0.90 to 0.93): the first 5 % of the grid gains mass at every step and the sparse band loses mass.
"""
from __future__ import annotations

import numpy as np

from knockout.v2.objective import Oracle


def runs(f: np.ndarray):
    """(starts, ends) of the runs of non-zero cells, ends exclusive."""
    d = np.diff((f > 0).astype(np.int8), prepend=0, append=0)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def reheat(f, rng, strength):
    return f.copy()


def lognormal(f, rng, strength):
    out = f.copy()
    nz = np.flatnonzero(f > 0)
    out[nz] *= np.exp(strength * rng.standard_normal(len(nz)))
    return out


def region_noise(f, rng, strength):
    n = len(f)
    idx = np.arange(n)
    region = ((idx >= 0.05 * n) & (idx < 0.65 * n)) | (idx >= 0.95 * n)
    cells = np.flatnonzero((f > 0) & region)
    out = f.copy()
    out[cells] *= np.exp(strength * rng.standard_normal(len(cells)))
    return out


def threshold(f, rng, strength):
    out = f.copy()
    nz = np.flatnonzero(f > 0)
    order = nz[np.argsort(f[nz], kind="stable")]
    drop = order[np.cumsum(f[order]) <= strength * f.sum()]
    out[drop] = 0.0
    return out


def open_boundary(f, rng, strength):
    out = f.copy()
    starts, ends = runs(f)
    left = starts[starts > 0]                      # the zero cell at left - 1 takes strength * f[left]
    right = ends[ends < len(f)]                    # the zero cell at right takes strength * f[right - 1]
    left = left[rng.random(len(left)) < 0.5]
    right = right[rng.random(len(right)) < 0.5]
    out[left - 1] = np.maximum(out[left - 1], strength * f[left])
    out[right] = np.maximum(out[right], strength * f[right - 1])
    return out


def swap_support(f, rng, strength):
    return open_boundary(threshold(f, rng, strength), rng, 1e-4)


def split_runs(f, rng, strength):
    out = f.copy()
    starts, ends = runs(f)
    long = np.flatnonzero(ends - starts >= 3)
    chosen = long[rng.random(len(long)) < strength]
    cut = starts[chosen] + 1 + (rng.random(len(chosen)) * (ends[chosen] - starts[chosen] - 2)).astype(np.int64)
    out[cut - 1] += 0.5 * f[cut]
    out[cut + 1] += 0.5 * f[cut]
    out[cut] = 0.0
    return out


def merge_runs(f, rng, strength, max_gap: int = 4):
    out = f.copy()
    starts, ends = runs(f)
    gap_from, gap_to = ends[:-1], starts[1:]       # zero cells gap_from .. gap_to - 1 between two runs
    small = np.flatnonzero(gap_to - gap_from <= max_gap)
    chosen = small[rng.random(len(small)) < strength]
    for a, b in zip(gap_from[chosen], gap_to[chosen]):
        out[a:b] = 0.5 * (f[a - 1] + f[b])
    return out


def transfer_mass(f, rng, strength, width: float = 0.01):
    n = len(f)
    w = max(1, int(width * n))
    mass = np.add.reduceat(f, np.arange(0, n, w))  # mass of each window
    filled = np.flatnonzero(mass > 0)
    if len(filled) < 2:
        return f.copy()
    src, dst = rng.choice(filled, size=2, replace=False)
    out = f.copy()
    moved = strength * mass[src]
    out[src * w:(src + 1) * w] *= 1.0 - strength
    out[dst * w:(dst + 1) * w] *= 1.0 + moved / mass[dst]
    return out


def shift_runs(f, rng, strength, margin: int = 3):
    """Shift the runs of one window of about strength * n cells by -2, -1, 1 or 2 cells. The window starts and ends
    in gaps of at least `margin` zero cells, so no neighbouring run is overwritten."""
    n = len(f)
    starts, ends = runs(f)
    gap_before = starts - np.concatenate(([-margin], ends[:-1]))
    gap_after = np.concatenate((starts[1:], [n + margin])) - ends
    first = np.flatnonzero((gap_before >= margin) & (starts >= margin))
    if len(first) == 0:
        return f.copy()
    i = int(rng.choice(first))
    last = np.flatnonzero((np.arange(len(ends)) >= i) & (gap_after >= margin) & (ends <= n - margin))
    if len(last) == 0:
        return f.copy()
    wide_enough = last[ends[last] >= starts[i] + max(1, int(strength * n))]
    j = int(wide_enough[0]) if len(wide_enough) else int(last[-1])
    a, b = int(starts[i]), int(ends[j])
    k = int(rng.choice([-2, -1, 1, 2]))
    out = f.copy()
    out[a:b] = 0.0
    out[a + k:b + k] = f[a:b]
    return out


def open_gradient(f, rng, strength, beta: float = 1e8, amplitude: float = 0.005):
    """Open the zero cells with the largest positive gradient of the smoothed score (one evaluation). Their number
    is strength * (number of non-zero cells), their amplitude `amplitude` * the median non-zero cell. f = w^2 keeps
    a zero cell at zero for ever, so this is the only way the polisher can use such a cell."""
    oracle = Oracle(len(f))
    g = oracle.grad_f(oracle.normalize(f), beta)
    zero = np.flatnonzero((f <= 0) & (g > 0))
    k = min(len(zero), int(strength * np.count_nonzero(f > 0)))
    out = f.copy()
    if k > 0:
        top = zero[np.argpartition(g[zero], len(zero) - k)[len(zero) - k:]]
        out[top] = amplitude * np.median(f[f > 0])
    return out


def envelope(f, rng, strength, references=None, block: int = 65536, first: str = "sol2780", last: str = "sol2783",
             gain: float = 4.868e-5):
    """Multiply f by exp(strength * e(x)), where e is the drift of the block masses from `first` to `last` (log
    ratio, per 1e-5 of the score gained between them), linear between block centres. Blocks with less than 1e-4
    of the mass are left alone, and e is clipped at 5 times its median size."""
    a, b = references[first], references[last]
    n = len(f)
    block = min(block, max(1, n // 8))
    k = n // block
    mass_a = a[: k * block].reshape(k, block).sum(1) / a.sum()
    mass_b = b[: k * block].reshape(k, block).sum(1) / b.sum()
    drift = np.zeros(k)
    ok = (mass_a > 1e-4) & (mass_b > 1e-4)
    if not ok.any():
        return f.copy()
    drift[ok] = np.log(mass_b[ok] / mass_a[ok]) / (gain / 1e-5)
    cap = 5.0 * np.median(np.abs(drift[ok]))
    drift = np.clip(drift, -cap, cap)
    e = np.interp(np.arange(n), (np.arange(k) + 0.5) * block, drift)
    return f * np.exp(strength * e)


def clear_band(f, rng, strength, lo: float = 0.65, hi: float = 0.90):
    out = f.copy()
    out[int(lo * len(f)):int(hi * len(f))] *= 1.0 - strength
    return out


MOVES = {
    "reheat": reheat,
    "lognormal": lognormal,
    "region_noise": region_noise,
    "threshold": threshold,
    "open_boundary": open_boundary,
    "swap_support": swap_support,
    "split_runs": split_runs,
    "merge_runs": merge_runs,
    "transfer_mass": transfer_mass,
    "shift_runs": shift_runs,
    "open_gradient": open_gradient,
    "envelope": envelope,
    "clear_band": clear_band,
}

# Moves that read other solutions: they get `references`, a dict name -> solution.
NEEDS_REFERENCES = {"envelope"}

# Evaluations a move spends itself; the episode takes them off the polish budget.
MOVE_EVALS = {"open_gradient": 1.5}


def apply_move(family: str, f: np.ndarray, seed: int, strength: float, references: dict | None = None) -> np.ndarray:
    """Run one move and check its contract. Raises ValueError on a broken result."""
    extra = {"references": references or {}} if family in NEEDS_REFERENCES else {}
    out = MOVES[family](np.asarray(f, dtype=np.float64), np.random.default_rng(seed), float(strength), **extra)
    out = np.asarray(out, dtype=np.float64)
    if out.shape != f.shape or not np.all(np.isfinite(out)) or out.min() < 0.0 or out.sum() <= 0.0:
        raise ValueError(f"move {family} broke its contract")
    return out
