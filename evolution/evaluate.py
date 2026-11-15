"""Fair, parallel evaluation of candidate specs (the trusted side of L2).

A Spec is one thing to test (optimizer + config + genes + warm start). evaluate_batch() turns every
spec into one job per seed, runs ALL jobs of the batch in a single executor.map (so N Modal workers
are used at once), then re-verifies every candidate here with the frozen problem.

Two different questions, answered separately:
  * "Is this METHOD better?"  -> real_margin(): mean over >= 3 seeds, same budget, same parent,
                                 improvement must exceed max(relative floor, Z x standard error).
  * "Is this SOLUTION better?" -> any verified candidate better than the parent certificate is stored
                                 as a certificate; the island's best certificate always advances.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from execution import make_job

Z = 2.0                  # ~95% one-sided noise band
MIN_REL_MARGIN = 1e-9    # floor below which float noise dominates


@dataclass
class Spec:
    optimizer: str
    config: dict
    warm_start: str | None          # parent certificate id
    arm: str
    island: str
    genes: list = field(default_factory=list)     # gene dicts
    hypothesis: str = ""
    rationale: str = ""
    parent_score: float | None = None
    seeds: list | None = None        # per-spec seeds (an island's own seed set); default: the batch seeds
    llm: dict | None = None          # usage + cost of the LLM call that produced this spec (None: no call)


@dataclass
class Evaluation:
    spec: Spec
    scores: list                    # verified score per seed (None for a failed seed)
    mean: float | None
    best_score: float | None
    best_candidate: dict | None
    status: str                     # ok | invalid | crash
    error: str | None
    duration_s: float
    worker_ids: list
    seeds: list

    @property
    def valid(self) -> bool:
        return self.status == "ok"


def utility(problem, score: float) -> float:
    return -score if problem.direction == "minimize" else score


def real_margin(problem, new_scores: list, ref_scores: list) -> tuple[bool, float, float]:
    """(significant, improvement, required margin), in utility units (positive = better)."""
    u_new = [utility(problem, s) for s in new_scores]
    u_ref = [utility(problem, s) for s in ref_scores]
    if not u_new or not u_ref:
        return False, 0.0, math.inf
    gain = statistics.fmean(u_new) - statistics.fmean(u_ref)
    var = lambda u: statistics.variance(u) / len(u) if len(u) > 1 else 0.0
    noise = Z * math.sqrt(var(u_new) + var(u_ref))
    floor = MIN_REL_MARGIN * max(1.0, abs(statistics.fmean(u_ref)))
    need = max(floor, noise)
    return gain > need, gain, need


def evaluate_batch(problem, executor, specs: list[Spec], seeds: list[int], budget_s: float, certs) -> list[Evaluation]:
    jobs, index = [], []
    for i, spec in enumerate(specs):
        for seed in (spec.seeds or seeds):
            jobs.append(make_job(problem, spec.optimizer, spec.config, seed, budget_s, spec.warm_start, certs,
                                 job_id=f"{i}:{seed}", arm=spec.arm, island=spec.island))
            index.append(i)
    results = executor.map(jobs)
    out = []
    for i, spec in enumerate(specs):
        mine = [r for r, k in zip(results, index) if k == i]
        scores, errors, best, best_c = [], [], None, None
        for r in mine:
            if r.status != "ok":
                scores.append(None)
                errors.append(f"{r.status}: {r.error}")
                continue
            reason = problem.why_invalid(r.candidate)          # trusted re-verification
            if reason is not None:
                scores.append(None)
                errors.append(f"invalid: {reason}")
                continue
            s = problem.score(r.candidate)
            scores.append(s)
            if best is None or problem.better(s, best):
                best, best_c = s, r.candidate
        ok = [s for s in scores if s is not None]
        status = "ok" if len(ok) == len(mine) else ("invalid" if any(e.startswith("invalid") for e in errors) else "crash")
        out.append(Evaluation(spec, scores, statistics.fmean(ok) if len(ok) == len(mine) else None, best, best_c,
                              status, "; ".join(errors[:2]) or None,
                              max((r.duration_s for r in mine), default=0.0), [r.worker_id for r in mine],
                              list(spec.seeds or seeds)))
    return out
