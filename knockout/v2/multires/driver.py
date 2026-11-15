"""The coarse-to-fine driver: optimize a batch of rows at one level, repeat-upsample, perturb, optimize at the
next level, up to the finest one. It runs on `numpy` (tests) or `torch` (GPU, float64) through the batched chains
of knockout/v2/gpu; changes of resolution and perturbations are done on CPU in numpy, seeded per row.

    f0 (B, n0) numpy  ->  run_ladder(...)  ->  one call of on_level_end per level with the rows' solutions

The level optimizer is the annealing chain of knockout/v2/gpu/chain.py (f = w^2, LogSumExp maximum, batched
L-BFGS): a cell that is exactly zero stays zero, unless a floor or a perturbation opens it.
"""
from __future__ import annotations

import time

import numpy as np

from knockout.v2.gpu.chain import initial_w, run_chains
from knockout.v2.gpu.objective import exact_score, fft_length, kappa_for
from knockout.v2.multires import perturb
from knockout.v2.multires import plan as pl
from knockout.v2.multires.resample import repeat_upsample

REPEAT_TOL = 1e-11        # |score(repeat(f, k)) - score(f)| allowed on the device (float64 FFT rounding)


def level_start(batch: pl.BatchSpec, start: np.ndarray) -> np.ndarray:
    """Rows of level 0 (B, pad_to / pool): the start zero-padded, perturbed per row (`pre`), then sum-pooled."""
    from knockout.v2.multires.resample import pad_right, sum_pool

    base = pad_right(np.maximum(np.asarray(start, dtype=np.float64), 0.0), batch.pad_to)
    rows = [perturb.apply_all(base, r.pre, r.seed) for r in batch.rows]
    return np.stack([sum_pool(r, batch.pool) for r in rows])


def next_level(batch: pl.BatchSpec, level: int, f_end: np.ndarray) -> np.ndarray:
    """Rows of level `level + 1`: repeat-upsampling by ups[level], then the rows' `post` perturbations (seeded by
    the row's seed and the level, so two levels do not reuse a draw)."""
    up = repeat_upsample(f_end, batch.ups[level])
    return np.stack([perturb.apply_all(up[i], r.post, r.seed + 1000 * (level + 1))
                     for i, r in enumerate(batch.rows)])


def run_ladder(xp, batch: pl.BatchSpec, f0: np.ndarray, *, to_xp, to_np, history: int = 8, on_level_end=None,
               on_progress=None, progress_every: int = 500, deadline=None, clock=time.monotonic) -> dict:
    """Run the levels of `batch` from the level-0 rows f0 (numpy, B x n0).

    on_level_end(level, f_rows (numpy B x n), info) receives, per row, the better of the chain's last point and
    of its best exact point; info = {"n", "evals", "init_exact", "exact", "smooth", "stopped_early",
    "s_per_eval", "accepted", "rejected", "restarts", "repeat_gap"} (lists have one entry per row).
    on_progress(level, evals, total, exact, best) is called every `progress_every` evaluations.
    Returns {"levels": [info ...], "stopped_early": bool, "f": last rows}.
    """
    pl.check_batch(batch)
    if f0.shape != (len(batch.rows), pl.level_sizes(batch)[0]):
        raise ValueError(f"level-0 rows have shape {f0.shape}")
    f = f0
    infos, stopped, repeat_gap = [], False, None
    for level, evals in enumerate(batch.level_evals):
        n = f.shape[1]
        m = fft_length(n)
        ft = to_xp(f)
        init_exact = [float(v) for v in exact_score(xp, ft, m).tolist()]
        t0 = clock()
        kappa = kappa_for(xp, ft, m)
        floors = to_xp(np.array([[r.floor if level == 0 else r.up_floor] for r in batch.rows]))
        w0 = initial_w(xp, ft, kappa, floors)
        del ft
        cb = None
        if on_progress is not None:
            def cb(e, tot, _smooth, ex, be, _stage, lv=level):
                on_progress(lv, e, tot, ex, be)
        out = run_chains(xp, w0, kappa, m, [pl.stages(batch, r, level) for r in batch.rows], history=history,
                         on_progress=cb, progress_every=progress_every, deadline=deadline, clock=clock)
        del w0
        use_best = out["best_exact"] > out["exact"]
        x = xp.where(use_best[:, None], out["best_w"], out["x"])
        f_end_x = x * x
        end_exact = xp.where(use_best, out["best_exact"], out["exact"])
        smooth = out["smooth"]
        counts = {k: out[k] for k in ("accepted", "rejected", "restarts")}
        del out["best_w"], out["x"], x
        loop_s = clock() - t0
        f_end = to_np(f_end_x)
        info = {"n": n, "evals": out["evals"], "planned_evals": evals, "init_exact": init_exact,
                "exact": [float(v) for v in end_exact.tolist()], "smooth": [float(v) for v in smooth.tolist()],
                "stopped_early": bool(out["stopped_early"]), "s_per_eval": loop_s / max(1, out["evals"]),
                "repeat_gap": repeat_gap, **counts}
        infos.append(info)
        if on_level_end is not None:
            on_level_end(level, f_end, info)
        f = f_end
        if out["stopped_early"]:
            stopped = True
            break
        if level < len(batch.ups):
            k = batch.ups[level]
            # the exact invariance the ladder relies on, checked on the scoring backend at every transition
            coarse = exact_score(xp, f_end_x, m)
            fine = exact_score(xp, to_xp(repeat_upsample(f_end, k)), fft_length(n * k))
            repeat_gap = float(abs(fine - coarse).max())
            if not repeat_gap <= REPEAT_TOL:
                raise AssertionError(f"repeat-upsampling by {k} changed the score by {repeat_gap:.3e} at n = {n}")
            del f_end_x, coarse, fine
            f = next_level(batch, level, f_end)
    return {"levels": infos, "stopped_early": stopped, "f": f}
