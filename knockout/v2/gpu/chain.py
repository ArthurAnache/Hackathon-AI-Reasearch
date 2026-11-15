"""The annealing loop for a batch of chains, on `numpy` or `torch` arrays (same code, see objective.py).

Each chain follows its own list of stages (beta, evaluations). At a stage end the chain's current point is handed
to `on_stage_end` (the caller stores it and re-scores it on CPU with the frozen verifier), then beta changes and
the chain goes on from the same point, keeping its L-BFGS history. The loop also keeps, per chain, the point with
the best exact score seen (hard maximum, computed in the same evaluation as the smoothed objective).
"""
from __future__ import annotations

import math
import time

from knockout.v2.gpu.lbfgs import BatchedLBFGS, _copy
from knockout.v2.gpu.objective import evaluate
from knockout.v2.gpu.schedule import boundaries


def initial_w(xp, f0, kappa, floors):
    """Start of a chain in the w parametrization (f = w^2). f0 (B, n) >= 0 is rescaled to sum kappa; `floors`
    (B, 1) adds floor * (mean non-zero value) to every cell, which lets zero cells move (with f = w^2 a cell that
    is exactly zero has a zero gradient and stays zero)."""
    g = f0 * (kappa / xp.sum(f0, -1)[:, None])
    nonzero = xp.sum(xp.where(g > 0, xp.ones_like(g), xp.zeros_like(g)), -1)[:, None]
    mean = xp.sum(g, -1)[:, None] / nonzero
    return xp.sqrt(g + floors * mean)


def run_chains(xp, w0, kappa, m, stage_lists, *, history=8, on_stage_end=None, on_progress=None,
               progress_every=200, deadline=None, clock=time.monotonic):
    """Run the chains. w0 (B, n), kappa (B, 1), stage_lists: one list of Stage per row, equal total evaluations.

    on_stage_end(row, stage_index, evals_done, w_row, info) is called once per row and stage;
    info = {"beta", "smooth", "exact", "accepted", "rejected", "restarts"} describes the row's current point.
    Returns {"evals", "stopped_early", "best_w", "best_exact", "first_exact", "x", "smooth", "exact", "stage"}.
    """
    b = w0.shape[0]
    if len(stage_lists) != b:
        raise ValueError("one stage list per row")
    bounds = [boundaries(s) for s in stage_lists]
    total = bounds[0][-1]
    if any(x[-1] != total for x in bounds):
        raise ValueError("all rows need the same total number of evaluations")
    kw = {"dtype": w0.dtype, "device": w0.device}
    stage = [0] * b
    beta = xp.ones((b, 1), **kw)
    for i in range(b):
        beta[i, 0] = stage_lists[i][0].beta
    opt = BatchedLBFGS(xp, w0, history=history)
    best_w = _copy(w0)
    best_exact = xp.full((b,), -math.inf, **kw)
    cur_smooth = xp.zeros((b,), **kw)
    cur_exact = xp.zeros((b,), **kw)
    first_exact = None
    evals, stopped_early = 0, False
    while evals < total:
        if deadline is not None and clock() >= deadline:
            stopped_early = True
            break
        xt = opt.trial()
        phi, grad, aux = evaluate(xp, xt, beta, kappa, m)
        took = opt.update(xt, phi, grad)
        evals += 1
        cur_smooth = xp.where(took, aux["smooth"], cur_smooth)
        cur_exact = xp.where(took, aux["exact"], cur_exact)
        if first_exact is None:
            first_exact = _copy(aux["exact"])
        improved = took & (aux["exact"] > best_exact)
        if bool(improved.any()):
            best_exact = xp.where(improved, aux["exact"], best_exact)
            best_w = xp.where(improved[:, None], xt, best_w)
        changed = []
        for i in range(b):
            if stage[i] < len(bounds[i]) and evals == bounds[i][stage[i]]:
                if on_stage_end is not None:
                    info = {"beta": stage_lists[i][stage[i]].beta, "smooth": float(cur_smooth[i]),
                            "exact": float(cur_exact[i]), "accepted": opt.accepted[i],
                            "rejected": opt.rejected[i], "restarts": opt.restarts[i]}
                    on_stage_end(i, stage[i], evals, opt.x[i], info)
                stage[i] += 1
                if stage[i] < len(bounds[i]):
                    beta[i, 0] = stage_lists[i][stage[i]].beta
                    changed.append(i)
        if changed:
            opt.mark_fresh(changed)
        if on_progress is not None and (evals % progress_every == 0 or evals == total):
            on_progress(evals, total, [float(v) for v in cur_smooth.tolist()],
                        [float(v) for v in cur_exact.tolist()], [float(v) for v in best_exact.tolist()],
                        list(stage))
    return {"evals": evals, "stopped_early": stopped_early, "best_w": best_w, "best_exact": best_exact,
            "first_exact": first_exact, "x": opt.x, "smooth": cur_smooth, "exact": cur_exact, "stage": stage,
            "accepted": list(opt.accepted), "rejected": list(opt.rejected), "restarts": list(opt.restarts)}
