"""Non-LLM tuner of a candidate program's PARAMS (section 5.4): CMA-ES, with a random-search fallback.

PARAMS format in a candidate program:
    PARAMS = {"name": (default, low, high), "other": (default, low, high, "log")}
A parameter is an integer when default, low and high are all ints. The tuner works in [0, 1]^d.
This module runs inside the sandbox; it never sees the frozen verifier, only a guide objective.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Import cma once at module load: its first import takes ~0.5 s and must not eat a tuning budget.
try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "could not import matplotlib" is irrelevant here
        import cma as _cma
except Exception:  # noqa: BLE001
    _cma = None


@dataclass
class Param:
    name: str
    default: float
    low: float
    high: float
    log: bool = False
    integer: bool = False

    def encode(self, v: float) -> float:
        lo, hi, v = (math.log(self.low), math.log(self.high), math.log(v)) if self.log else (self.low, self.high, v)
        return 0.0 if hi == lo else min(1.0, max(0.0, (v - lo) / (hi - lo)))

    def decode(self, u: float) -> float | int:
        u = min(1.0, max(0.0, float(u)))
        if self.log:
            v = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            v = self.low + u * (self.high - self.low)
        return int(round(v)) if self.integer else float(v)


class ParamSpace:
    def __init__(self, params: dict[str, tuple]):
        self.params: list[Param] = []
        for name, spec in sorted((params or {}).items()):
            if not isinstance(spec, (tuple, list)) or len(spec) not in (3, 4):
                raise ValueError(f"PARAMS[{name!r}] must be (default, low, high[, 'log'])")
            default, low, high = spec[:3]
            log = len(spec) == 4 and spec[3] == "log"
            if not (low <= default <= high):
                raise ValueError(f"PARAMS[{name!r}]: default outside [low, high]")
            if log and low <= 0:
                raise ValueError(f"PARAMS[{name!r}]: log scale needs low > 0")
            integer = all(isinstance(v, int) and not isinstance(v, bool) for v in (default, low, high))
            self.params.append(Param(name, default, low, high, log, integer))

    @property
    def dim(self) -> int:
        return len(self.params)

    def default_u(self) -> np.ndarray:
        return np.array([p.encode(p.default) for p in self.params], dtype=float)

    def decode(self, u) -> dict[str, Any]:
        return {p.name: p.decode(x) for p, x in zip(self.params, u)}


@dataclass
class Trial:
    params: dict[str, Any]
    value: float                 # guide objective, already signed so that larger is better (-inf if failed)
    candidate: Any = None
    error: str | None = None


@dataclass
class TuneResult:
    trials_top: list[Trial]
    n_evals: int
    method: str
    elapsed_s: float
    errors: list[str] = field(default_factory=list)


def tune(objective: Callable[[dict, float], tuple[float, Any]], space: ParamSpace, *, seed: int, budget_s: float,
         method: str = "cma", sigma0: float = 0.25, calls: int = 8, top_k: int = 3) -> TuneResult:
    """Maximize objective(params, call_budget_s) -> (signed value, candidate) within budget_s seconds.

    The first call always uses the defaults, so a program is never scored worse than its own PARAMS.
    """
    t0 = time.monotonic()
    deadline = t0 + budget_s
    per_call = budget_s if space.dim == 0 else budget_s / max(1, calls)
    top: list[Trial] = []
    errors: list[str] = []
    n = 0

    def run(u) -> float:
        nonlocal n
        params = space.decode(u)
        n += 1
        try:
            value, cand = objective(params, per_call)
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("non-finite guide value")
            trial = Trial(params, value, cand)
        except Exception as exc:  # noqa: BLE001  (a failing trial is data, not a crash)
            errors.append(f"{type(exc).__name__}: {exc}"[:300])
            trial = Trial(params, -math.inf, None, errors[-1])
        if trial.candidate is not None:
            top.append(trial)
            top.sort(key=lambda t: t.value, reverse=True)
            del top[top_k:]
        return trial.value

    def time_left() -> bool:
        return time.monotonic() + per_call <= deadline + 1e-9

    run(space.default_u())
    used = "none" if space.dim == 0 else method
    if space.dim > 0:
        rng = np.random.default_rng(seed)
        es = None
        restarts = 0
        best_u = space.default_u()
        best_v = top[0].value if top else -math.inf

        def new_es(x_start):
            opts = {"bounds": [0.0, 1.0], "seed": int((seed + 7919 * restarts) % (2**31 - 1)) + 1, "verbose": -9,
                    "tolx": 1e-9, "tolfun": 1e-12}
            x0 = np.asarray(x_start, dtype=float)
            if space.dim == 1:  # cma needs dim >= 2; pad with a dummy coordinate
                x0 = np.r_[x0, 0.5]
            return _cma.CMAEvolutionStrategy(x0, sigma0, opts)

        if method == "cma" and _cma is not None:
            es = new_es(best_u)
        else:
            if method == "cma":
                errors.append("cma not installed: random search")
            used = "random"
        while time_left():
            if es is not None:
                xs = es.ask()
                vals = []
                for x in xs:
                    if not time_left():
                        break
                    u = np.asarray(x)[: space.dim]
                    v = run(u)
                    if v > best_v:
                        best_v, best_u = v, u.copy()
                    vals.append(-v)
                if len(vals) < len(xs):
                    break
                try:
                    es.tell(xs, [v if math.isfinite(v) else 1e30 for v in vals])
                    converged = bool(es.stop())
                except Exception:  # noqa: BLE001  (degenerate covariance after convergence)
                    converged = True
                if converged:  # restart around the best point found so far
                    restarts += 1
                    es = new_es(best_u)
            else:
                u = np.clip(space.default_u() + sigma0 * rng.standard_normal(space.dim), 0.0, 1.0)
                run(u)
    return TuneResult(top, n, used, time.monotonic() - t0, errors)
