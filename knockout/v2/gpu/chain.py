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


def reopen_cells(xp, w, grad_f, eta):
    """Open the closed cells that the first-order condition says should not be closed.

    With f >= 0, a local maximum needs d log C / d f_j <= 0 on every cell with f_j = 0. A closed cell (f_j below
    1e-9 of the mean open value) with a positive derivative is given f_j = eta * mean * grad_j / max grad, so the
    strongest violator opens at `eta` times the mean open value. eta is (B, 1); rows with eta = 0 are untouched.
    Returns (w, number of cells opened per row). With f = w^2 this is the only way a zero cell can move.
    """
    f = w * w
    one = xp.ones_like(f)
    zero = xp.zeros_like(f)
    is_open = f > 1e-6 * xp.amax(f, -1)[:, None]
    mean = (xp.sum(xp.where(is_open, f, zero), -1) / xp.sum(xp.where(is_open, one, zero), -1))[:, None]
    violating = (f <= 1e-9 * mean) & (grad_f > 0) & (eta > 0)
    gain = xp.where(violating, grad_f, zero)
    top = xp.amax(gain, -1)[:, None]
    new_f = eta * mean * gain / xp.where(top > 0, top, xp.ones_like(top))
    return xp.where(violating, xp.sqrt(new_f), w), xp.sum(xp.where(violating, one, zero), -1)


def run_chains(xp, w0, kappa, m, stage_lists, *, history=8, on_stage_end=None, on_progress=None,
               progress_every=200, deadline=None, clock=time.monotonic, reopen=None, reopen_every=500,
               reopen_until=0.9, reopen_beta_max=3e7):
    """Run the chains. w0 (B, n), kappa (B, 1), stage_lists: one list of Stage per row, equal total evaluations.

    on_stage_end(row, stage_index, evals_done, w_row, info) is called once per row and stage;
    info = {"beta", "smooth", "exact", "accepted", "rejected", "restarts", "reopened"} describes the row's
    current point.
    reopen: one eta per row (0 = never). Every `reopen_every` evaluations, during the first `reopen_until` of the
    budget, the rows with eta > 0 whose current beta is at most `reopen_beta_max` get their first-order-violating
    closed cells opened (see reopen_cells). Each such event costs one extra batched evaluation, counted in
    "extra_evals" and not in the stage budgets.
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
    reopen_rows = [i for i in range(b) if reopen is not None and reopen[i] > 0]
    eta = xp.zeros((b, 1), **kw)
    for i in reopen_rows:
        eta[i, 0] = reopen[i]
    reopened = [0] * b
    extra_evals = 0
    while evals < total:
        if deadline is not None and clock() >= deadline:
            stopped_early = True
            break
        if reopen_rows and evals % reopen_every == 0 and evals <= reopen_until * total:
            # only while the row is hot: at beta >= 1e9 the smoothed derivative is valid over relative steps of
            # about 1e-9, and an opening costs more exact score than the polish recovers (measured: eta = 1e-3
            # costs 2e-5 at beta = 1e9 and comes back at about 1e-6 per 50 evaluations)
            hot = [i for i in reopen_rows
                   if stage[i] < len(stage_lists[i]) and stage_lists[i][stage[i]].beta <= reopen_beta_max]
            if hot:
                eta_now = xp.zeros_like(eta)
                for i in hot:
                    eta_now[i, 0] = eta[i, 0]
                _, _, aux = evaluate(xp, opt.x, beta, kappa, m)
                opt.x, counts = reopen_cells(xp, opt.x, aux["grad_f"], eta_now)
                extra_evals += 1
                for i, cnt in enumerate(counts.tolist()):
                    reopened[i] += int(cnt)
                opt.mark_fresh(hot)
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
                            "rejected": opt.rejected[i], "restarts": opt.restarts[i], "reopened": reopened[i]}
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
            "accepted": list(opt.accepted), "rejected": list(opt.rejected), "restarts": list(opt.restarts),
            "reopened": reopened, "extra_evals": extra_evals}
