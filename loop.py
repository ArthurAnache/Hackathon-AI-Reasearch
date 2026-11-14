"""The research loop, written only against the contract in interfaces.py.

    python loop.py --problem 1a [--optimizer optimizers/1a_baseline.py] [-n 6] [--budget 60]

Each iteration: evidence = source.gather(...) -> exp = agent.propose(history, evidence)
-> run_experiment(...) (run, verify, keep/discard, store.append) -> agent.review(result).

The default agent is research.FeedbackAgent (credential-free). It turns the results
so far into the next experiment: knob effects, a surrogate model, a trust region,
combining moves that helped, evidence hints, and, with warm start, refinement of the best
solution found. Code changes come from a coding agent following program.md, or
from an LLM agent implementing ResearchAgent.
"""
from __future__ import annotations

import argparse
import sys

import interfaces as I
import prepare
from research import FeedbackAgent


def research_loop(problem: I.Problem, optimizer: I.FileOptimizer, agent: I.ResearchAgent,
                  store: I.ExperimentStore, evidence_source: I.EvidenceSource | None = None,
                  n: int = 5, budget_s: float | None = None, on_result=None) -> list[dict]:
    """Run up to n experiments, each chosen by the agent from the history so far."""
    evidence_source = evidence_source or I.NoEvidence()
    results = []
    for _ in range(n):
        history = store.read()
        evidence = evidence_source.gather(problem, history)
        exp = agent.propose(history, evidence)
        opt = optimizer if exp.optimizer is None else I.FileOptimizer(exp.optimizer)
        result = I.run_experiment(problem, opt, exp, store, budget_s=budget_s, agent=agent.name, evidence=evidence)
        results.append(result)
        if on_result:
            on_result(result)
        if agent.review(result) == "stop":
            break
    return results


def _print(r: dict) -> None:
    score = "-" if r["score"] is None else f"{r['score']:.10g}"
    print(f"{r['experiment_id']} {r['status']:8} score={score:<14} {r['hypothesis']}")
    if r["error"]:
        print(f"           error: {r['error']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--optimizer", help="optimizer file (default optimizers/<problem>_baseline.py)")
    ap.add_argument("-n", type=int, default=5, help="max experiments")
    ap.add_argument("--budget", type=float, default=None, help=f"seconds per experiment (default {prepare.TIME_BUDGET})")
    ap.add_argument("--store", default=str(prepare.ROOT / "experiments.jsonl"))
    ap.add_argument("--no-warm-start", action="store_true",
                    help="every experiment starts from scratch (compare algorithms, not chains)")
    ap.add_argument("--insights", action="store_true", help="print what the results so far say and exit")
    a = ap.parse_args(argv)
    problem = I.get_problem(a.problem)
    opt = I.FileOptimizer(a.optimizer or prepare.ROOT / "optimizers" / f"{a.problem}_baseline.py")
    agent = FeedbackAgent(problem, opt, warm_start=False if a.no_warm_start else None)
    store = I.JsonlStore(a.store)
    if a.insights:
        print(agent.knowledge(store.read()).summary())
        return 0
    print(f"agent: {agent.name}, warm start: {agent.warm_start}, knobs: {agent.space.names}")
    research_loop(problem, opt, agent, store, n=a.n, budget_s=a.budget, on_result=_print)
    return 0


if __name__ == "__main__":
    sys.exit(main())
