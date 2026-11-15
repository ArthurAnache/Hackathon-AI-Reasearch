"""Deterministic polisher driven by an evaluation budget.

    best_f, info = polish(f, "p0", budget=1500)

A schedule is a list of stages `(beta, use_multipliers)`. Each stage runs L-BFGS on the smoothed objective of
`objective.Oracle` (f = w^2, so the support of the input is kept) from the last iterate of the previous stage.
The remaining budget is shared equally between the remaining stages. Budgets count objective-and-gradient
evaluations, never seconds, so a result does not depend on the speed of the machine.

The polisher returns the best exact point it has evaluated, the input included: it never returns less than its
input. A stage that starts at a low beta may lower the exact score on purpose (a reheat); only the final answer is
protected.

Schedules:
- `p0`: the standard polish, beta = 1e9 then 3e9 then 1e10 ("polish alone" in the episode value).
- `a7`, `a6`, `a5`: anneals that start at beta = 1e7, 1e6, 1e5 and end at 3e9.
- `h6`, `h7`, `h8`: holds. About half of the budget stays at one low beta (1e6, 1e7, 1e8), in six stages whose
  smoothed values are logged, then the anneal goes up to 3e9. They measure how fast the smoothed objective still
  improves at a low beta, where the problem is better conditioned.
- `mm6`, `mm7`, `mm8`: exponential method of multipliers at a fixed beta (1e6, 1e7, 1e8), 8 multiplier updates,
  then the plain LogSumExp at 1e9 and 3e9.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from knockout.v2.objective import BudgetSpent, Oracle

SCHEDULES: dict[str, list[tuple[float, bool]]] = {
    "p0": [(1e9, False), (3e9, False), (1e10, False)],
    "a7": [(1e7, False), (3e7, False), (1e8, False), (3e8, False), (1e9, False), (3e9, False)],
    "a6": [(1e6, False), (1e7, False), (1e8, False), (1e9, False), (3e9, False)],
    "a5": [(1e5, False), (1e6, False), (1e7, False), (1e8, False), (1e9, False), (3e9, False)],
    "h6": [(1e6, False)] * 6 + [(3e6, False), (1e7, False), (3e7, False), (1e8, False), (3e8, False), (1e9, False),
                                (3e9, False)],
    "h7": [(1e7, False)] * 6 + [(3e7, False), (1e8, False), (3e8, False), (1e9, False), (3e9, False)],
    "h8": [(1e8, False)] * 6 + [(3e8, False), (1e9, False), (3e9, False)],
    "mm6": [(1e6, True)] * 8 + [(1e9, False), (3e9, False)],
    "mm7": [(1e7, True)] * 8 + [(1e9, False), (3e9, False)],
    "mm8": [(1e8, True)] * 8 + [(1e9, False), (3e9, False)],
}


class _Stop(Exception):
    pass


def _lbfgs(oracle: Oracle, w: np.ndarray, beta: float, log_lambda, stage_end: float, maxcor: int,
           trace: dict | None = None) -> np.ndarray:
    """L-BFGS from w until the oracle count reaches stage_end. Returns the last accepted iterate. `trace` receives
    the first and the lowest value of the smoothed objective (-log C_beta) seen in the stage."""
    last = {"x": w}

    def fun(x):
        if oracle.calls + 1.0 > stage_end + 1e-9:
            raise _Stop
        value, grad = oracle.value_and_grad(x, beta, log_lambda)
        if trace is not None:
            trace.setdefault("smooth_first", float(value))
            trace["smooth_best"] = min(trace.get("smooth_best", np.inf), float(value))
        return value, grad

    def callback(xk):
        last["x"] = xk.copy()

    for _ in range(4):                       # L-BFGS may stop early on a failed line search: restart it
        before = oracle.calls
        try:
            res = minimize(fun, last["x"], jac=True, method="L-BFGS-B", callback=callback,
                           options={"maxcor": maxcor, "maxiter": 1_000_000, "maxfun": 1_000_000,
                                    "ftol": 0.0, "gtol": 0.0})
            last["x"] = res.x
        except (_Stop, BudgetSpent):
            break
        if oracle.calls - before < 5:
            break
    return last["x"]


def polish(f0, schedule, budget: float, maxcor: int = 8):
    """Polish f0 with at most `budget` evaluations. Returns (best exact point seen, info)."""
    stages = SCHEDULES[schedule] if isinstance(schedule, str) else list(schedule)
    f0 = np.maximum(np.asarray(f0, dtype=np.float64), 0.0)
    oracle = Oracle(len(f0), budget)
    info = {"schedule": schedule if isinstance(schedule, str) else "custom", "budget": budget, "stages": []}
    try:
        info["input_score"] = oracle.exact(f0)
        w = np.sqrt(f0)
        log_lambda = None
        for k, (beta, use_multipliers) in enumerate(stages):
            left = oracle.remaining()
            if left < 3.0:
                break
            stage_end = oracle.calls + left / (len(stages) - k)
            w = np.sqrt(oracle.normalize(w * w))
            if not use_multipliers:
                log_lambda = None
            stage = {"beta": beta, "multipliers": use_multipliers}
            w = _lbfgs(oracle, w, beta, log_lambda, stage_end, maxcor, stage)
            stage["calls"] = oracle.calls
            info["stages"].append(stage)
            if use_multipliers:
                log_lambda = oracle.update_multipliers(w * w, beta, log_lambda)
            stage["exact_at_end"] = oracle.exact(w * w)
    except BudgetSpent:
        pass
    info["calls"] = oracle.calls
    info["best_score"] = oracle.best_score
    return (oracle.best_f if oracle.best_f is not None else f0), info
