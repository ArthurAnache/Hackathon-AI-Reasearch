"""Run context shared by every arm, plus the one evaluation path: sandboxed tuning, then trusted re-scoring.

Fitness is always the tuned score (section 4): the best score, under the frozen verifier run here in the trusted
process, among the top candidates the sandboxed tuner returned. The sandbox's own guide value is only logged
(as `gpu_objective`, CLAUDE.md WS3 rule 1) and is never used for selection.
"""
from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field
from typing import Any

from knockout import decisions as D
from knockout.budget import BudgetGovernor
from knockout.config import Strategy
from knockout.executor import Executor
from knockout.ledger import Ledger, now_iso
from knockout.problems_api import ProblemHandle
from knockout.sandbox import SandboxJob, SandboxResult


def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:12]


@dataclass
class Evaluation:
    status: str                    # "ok" | "invalid" | "crash"
    valid: bool
    score: float | None
    candidate: Any
    params: dict
    guide: float | None
    error: str | None
    n_evals: int
    tuner: str | None
    elapsed_s: float
    cost_usd: float
    isolation: dict = field(default_factory=dict)
    guide_gap: float | None = None
    claim: str | None = None       # promotion gate: "record_claim" | "numerical_artifact" | None
    claim_inputs: dict = field(default_factory=dict)


_UNSET = object()


@dataclass
class RunContext:
    run_id: str
    problem: ProblemHandle
    strategy: Strategy
    ledger: Ledger
    governor: BudgetGovernor
    executor: Executor
    source_commit: str | None = None
    device: str = "cpu"
    init: Any = None                           # warm-start candidate shared by every job (None for the toy)
    init_score: float | None = None            # its score under the frozen verifier
    llm: Any = None                            # knockout.llm.LLM, shared by the arms (None without an LLM)
    best_known_cache: float | None = None

    def best_known(self) -> float | None:
        if self.best_known_cache is None:
            try:
                self.best_known_cache = float(self.problem.best_known())
            except Exception:  # noqa: BLE001  (offline: unknown)
                return None
        return self.best_known_cache

    # ------------------------------------------------------------------ evaluation
    def submit(self, source: str, budget_s: float, seed: int, init: Any = _UNSET,
               adapter_source: str | None = None):
        """`init` defaults to the context's warm start (the control arm's); the main arm passes its own."""
        job = SandboxJob(program_source=source, seed=seed, budget_s=budget_s, direction=self.problem.direction,
                         guide_ref=self.problem.guide_ref, guide_args=self.problem.guide_args,
                         init=self.init if init is _UNSET else init, adapter_source=adapter_source,
                         tuner={"method": self.strategy["tuner.method"], "sigma0": self.strategy["tuner.sigma0"],
                                "calls": self.strategy.get("tuner.calls", 8), "top_k": self.strategy["tuner.top_k"]})
        return self.executor.submit(job)

    def finish(self, result: SandboxResult, category: str) -> Evaluation:
        cost = self.executor.cost_usd(result)
        self.governor.charge(category, compute_s=result.elapsed_s, usd=cost)
        if not result.ok:
            return Evaluation("crash", False, None, None, {}, None, result.error, result.n_evals, result.tuner,
                              result.elapsed_s, cost, result.isolation)
        best, errors = None, []
        for trial in result.trials:
            try:
                s = self.problem.score(trial["candidate"])
            except Exception as exc:  # noqa: BLE001  (InvalidSolution, or anything the verifier raises)
                errors.append(f"{type(exc).__name__}: {exc}"[:300])
                continue
            if best is None or self.problem.gain(s, best[0]) > 0:
                best = (s, trial)
        if best is None:
            return Evaluation("invalid", False, None, None, result.trials[0]["params"] if result.trials else {},
                              None, "; ".join(errors[:2]) or "no candidate", result.n_evals, result.tuner,
                              result.elapsed_s, cost, result.isolation)
        s, trial = best
        guide = trial.get("guide")
        ev = Evaluation("ok", True, s, trial["candidate"], trial["params"], guide, None, result.n_evals,
                        result.tuner, result.elapsed_s, cost, result.isolation,
                        None if guide is None else abs(float(guide) - s))
        self._promotion_gate(ev)
        return ev

    def _promotion_gate(self, ev: Evaluation) -> None:
        """Section 5.11. A score above the record threshold is a `record_claim` only if `problem.verify` and the
        independent verifier both accept it and the independent score also beats the record. If the float score
        passes and the independent check fails, the candidate is a `numerical_artifact` and is not used."""
        try:
            beats = bool(self.problem.beats_record(ev.score))
        except Exception:  # noqa: BLE001  (record unknown: no claim)
            beats = False
        if not beats:
            return
        verify_ok = bool(self.problem.verify(ev.candidate))
        ind_ok = ind_beats = None
        detail: dict = {}
        if verify_ok and self.problem.independent is not None:
            try:
                ind = self.problem.independent(ev.candidate)
                ind_score = float(ind["score"])
                tol = max(float(ind.get("tolerance_abs") or 0.0), 4e-16 * max(1.0, abs(ev.score)))
                ind_ok = abs(ind_score - ev.score) <= tol
                ind_beats = bool(self.problem.beats_record(ind_score))
                detail = {"independent_score": ind_score, "tolerance_abs": tol, "method": ind.get("method")}
            except Exception as exc:  # noqa: BLE001  (the independent verifier rejected the candidate)
                ind_ok, ind_beats = False, False
                detail = {"independent_error": f"{type(exc).__name__}: {exc}"[:300]}
        ev.claim_inputs = {"beats_record": beats, "verify_ok": verify_ok, "independent_ok": ind_ok,
                           "independent_beats": ind_beats}
        ev.claim = D.promotion(beats, verify_ok, ind_ok, ind_beats)
        ev.claim_inputs["detail"] = {"score": ev.score, "best_known": self.best_known(), **detail}
        if self.problem.exploratory and self.problem.record_info is not None:
            ev.claim_inputs["detail"]["reference"] = {k: v for k, v in self.problem.record_info().items()
                                                      if k in ("value", "binding", "components", "live", "read_at")}
        if ev.claim == "numerical_artifact":
            ev.valid, ev.status = False, "void"
            ev.error = f"numerical artifact: float score {ev.score!r} is not confirmed by the independent verifier"

    # ------------------------------------------------------------------ ledger
    def log_experiment(self, *, experiment_id: str, ev: Evaluation, source: str, program_id: str, chain: str,
                       agent: str, hypothesis: str, rationale: str, decision: str, status: str, seed: int,
                       budget_s: float, best_before: float | None, extra_config: dict | None = None,
                       evidence_urls: list[str] | None = None, save_solution: bool = True) -> dict:
        """`save_solution` matters only when ledger.save_solutions = selected (large candidates): the caller then
        stores the candidates that the run depends on (baseline, kept programs, warm-start updates)."""
        program_path = self.ledger.save_program(program_id, source)
        solution_path = None
        keep_all = self.strategy.get("ledger.save_solutions", "all") == "all"
        exploratory = bool(self.problem.exploratory)
        if ev.claim and exploratory and ev.claim == "record_claim":
            # Exploratory problem (not frozen by WS1): the gate fired against the run's own reference. This is a
            # measurement of the loop, logged under its own decision name, and never a record claim.
            decision = "exploratory_gate"
            rationale = (f"exploratory gate fired ({ev.claim_inputs.get('detail')}): the score passes the run's "
                         f"reference by at least the margin under the pinned verifier and the independent scorer. "
                         f"Not a record claim: the problem is not frozen and WS1 has not arbitrated. " + rationale)
        elif ev.claim:  # promotion gate: the claim replaces the arm's decision, and the candidate is always stored
            decision = ev.claim
            rationale = (f"promotion gate: {ev.claim} ({ev.claim_inputs.get('detail')}); a record_claim is only "
                         f"a candidate until selection/verify_claim.py confirms it. " + rationale)
        if ev.candidate is not None and (ev.claim or (ev.valid and (keep_all or save_solution))):
            solution_path = self.ledger.save_solution(chain, experiment_id, ev.candidate)
        if ev.claim:
            inputs = {k: v for k, v in ev.claim_inputs.items() if k != "detail"}
            # `outputs.decision` is the raw return value of decisions.promotion (replay re-derives it); on an
            # exploratory problem `exploratory` and `logged_as` say how to read it.
            self.ledger.event("promotion", seed=None, refs=[experiment_id], inputs=inputs,
                              outputs={"decision": ev.claim}, detail=ev.claim_inputs.get("detail"),
                              certificate=solution_path, exploratory=exploratory, logged_as=decision)
        record = {
            "experiment_id": experiment_id, "hypothesis": hypothesis,
            "config": {"chain": chain, "program_id": program_id, "params": ev.params, "tuner": ev.tuner,
                       "n_evals": ev.n_evals, "isolation": ev.isolation, "cost_usd": round(ev.cost_usd, 6),
                       "guide_gap": ev.guide_gap, "global_id": f"{self.run_id}/{chain}/{experiment_id}",
                       **(extra_config or {})},
            "seed": seed, "budget_s": budget_s, "valid": ev.valid, "score": ev.score,
            "best_known_score": self.best_known(), "decision": decision, "rationale": rationale,
            "evidence_urls": list(evidence_urls or []),
            "timestamp": now_iso(), "problem": self.problem.id, "agent": agent, "optimizer": program_path,
            "algorithm_version": source_hash(source), "git_commit": self.source_commit, "status": status,
            # Always false on an exploratory problem: such a run never asserts anything about a record.
            "best_before": best_before,
            "beats_record": bool(not exploratory and ev.valid and self.problem.beats_record(ev.score)),
            "duration_s": round(ev.elapsed_s, 3), "error": ev.error, "evidence": [],
            "solution_path": solution_path,
            "run_id": self.run_id, "source_commit": self.source_commit, "device": self.device, "gpu_type": None,
            "dtype": "float64", "torch_version": None, "cuda_version": None, "gpu_objective": ev.guide,
            "wall_s": round(ev.elapsed_s, 3),
        }
        self.ledger.append(record)
        return record


def host_info() -> dict:
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()}
