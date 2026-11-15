"""FeedbackAgent: the research agent. Implements the contract
    propose(history, evidence) -> ExperimentConfig
    review(experiment_result) -> "continue" | "stop"

Each proposal:
  1. rebuilds Knowledge from the store (this optimizer version, this budget);
  2. asks every strategy for candidates and drops those already tried or close to a failure;
  3. picks a strategy by UCB on its improvement rate so far (untried strategies first;
     evidence hints always first), then that strategy's best-predicted candidate;
  4. with warm start on, every experiment starts from the best solution found so far
     (config INIT), so results compound across experiments;
  5. writes the hypothesis and a rationale citing the experiments it relies on,
     tagged `strategy=...; parent=...` so the next Knowledge can attribute the outcome.
"""
from __future__ import annotations

import math
import random

import numpy as np

import interfaces as I

from .knowledge import Knowledge
from .space import INIT_KEY, Space
from .strategies import Candidate, EvidenceHints, default_strategies


class FeedbackAgent:
    name = "feedback"
    UCB_C = 0.8

    def __init__(self, problem: I.Problem, optimizer: I.FileOptimizer, strategies=None,
                 warm_start: bool | None = None, patience: int = 6, max_crash_streak: int = 3):
        self.problem, self.optimizer = problem, optimizer
        self.space = Space.from_optimizer(optimizer)
        supports_init = INIT_KEY in self.space.defaults
        self.warm_start = supports_init if warm_start is None else (warm_start and supports_init)
        self.strategies = strategies if strategies is not None else default_strategies(self.warm_start)
        self.patience, self.max_crash_streak = patience, max_crash_streak
        self.statuses: list[str] = []
        self.last_knowledge: Knowledge | None = None

    # ------------------------------------------------------------ helpers
    def knowledge(self, history: list[dict]) -> Knowledge:
        v = self.optimizer.version()
        mine = [r for r in history if r["problem"] == self.problem.id and r.get("optimizer_hash") == v]
        if mine:
            budget = mine[-1]["budget_s"]
            mine = [r for r in mine if r["budget_s"] == budget]
        return Knowledge(self.problem, self.space, mine)

    def _choose(self, kn: Knowledge, candidates: list[Candidate]) -> Candidate:
        hints = [c for c in candidates if c.strategy == EvidenceHints.name]
        if hints:
            return hints[0]
        stats = kn.strategy_stats()
        total = sum(n for n, _ in stats.values()) + 1

        def ucb(s):
            n, w = stats.get(s, (0, 0))
            return math.inf if n == 0 else w / n + self.UCB_C * math.sqrt(math.log(total) / n)

        order = [s.name for s in self.strategies]
        best_strategy = max({c.strategy for c in candidates}, key=lambda s: (ucb(s), -order.index(s)))
        pool = [c for c in candidates if c.strategy == best_strategy]
        return max(pool, key=lambda c: (c.gain is not None and c.gain > 0, c.gain or 0.0, -c.tries))

    # ------------------------------------------------------------ contract
    def propose(self, history, evidence) -> I.ExperimentConfig:
        kn = self.knowledge(history)
        self.last_knowledge = kn
        cites = "; ".join(f"[{e.source}] {e.content[:120]} ({e.citation})" for e in evidence)
        if kn.best is None:
            return I.ExperimentConfig(
                "Baseline: the optimizer's default CONFIG.", {},
                rationale="strategy=baseline; parent=none" + (f"; evidence: {cites}" if cites else ""))

        rng = np.random.default_rng(len(kn.obs))
        for s in self.strategies:
            if isinstance(s, EvidenceHints):
                s.evidence = evidence
        candidates = []
        for s in self.strategies:
            for c in s.propose(kn, rng):
                if self.warm_start:
                    c.config[INIT_KEY] = kn.best.certificate
                elif INIT_KEY in self.space.defaults:
                    c.config[INIT_KEY] = None
                if not kn.tried(c.config) and not kn.near_failure(c.config):
                    candidates.append(c)
        if not candidates:            # everything nearby tried: random restart inside the full space
            x = np.random.default_rng(len(kn.obs) + 1).uniform(0, 1, kn.space.dim)
            cfg = kn.space.from_unit(x, {**kn.space.defaults, **kn.best.config})
            candidates = [Candidate(cfg, "restart", "neighbourhood exhausted: random restart in the full space")]

        c = self._choose(kn, candidates)
        direction = "lower" if self.problem.direction == "minimize" else "higher"
        expect = f" (predicted gain {abs(c.gain):.3g})" if c.gain else ""
        return I.ExperimentConfig(
            hypothesis=f"[{c.strategy}] {c.why}. Expect a {direction} score than {kn.best.score:.10g}{expect}.",
            config=c.config,
            rationale=f"strategy={c.strategy}; parent={kn.best.id}; {kn.summary()}"
                      + (f"; evidence: {cites}" if cites else ""),
        )

    def review(self, experiment_result) -> str:
        self.statuses.append(experiment_result["status"])
        recent = self.statuses[-self.patience:]
        if len(recent) == self.patience and "keep" not in recent:
            return "stop"
        crashes = self.statuses[-self.max_crash_streak:]
        if len(crashes) == self.max_crash_streak and all(s == "crash" for s in crashes):
            return "stop"
        return "continue"
