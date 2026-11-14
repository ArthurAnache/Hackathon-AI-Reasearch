"""Process 7: the shared technical contract (interfaces.py), function by function."""
import inspect

import pytest

import interfaces as I
import problems
from conftest import REPO

P = pytest.mark.process("7. Shared contract (interfaces.py)")


@P
@pytest.mark.parametrize("pid", sorted(problems.PROBLEMS))
def test_problem_verify_bool_and_score_float(pid, record):
    """problem.verify(candidate) -> bool, problem.score(candidate) -> float, on a valid and an invalid candidate."""
    p = I.get_problem(pid)
    good = {"1a": {"values": [1.0]}, "1b": {"values": [0.5, 0.5]}, "1d": {"values": [1.0]},
            "10c": {"matrix": [[1.0]]}, "11a": {"n": 1, "values": [1.0, -1.0]}}[pid]
    bad = {"values": "x", "matrix": "x", "n": 0}
    record(problem=pid, verify_good=p.verify(good), score_good=p.score(good), verify_bad=p.verify(bad))
    assert p.verify(good) is True and isinstance(p.score(good), float)
    assert p.verify(bad) is False
    with pytest.raises(I.InvalidSolution):
        p.score(bad)


@P
def test_optimizer_run_signature(record):
    """optimizer.run(problem, budget_s, seed, config) -> candidate; config overrides CONFIG."""
    opt = I.FileOptimizer(REPO / "optimizers" / "1a_baseline.py")
    params = list(inspect.signature(opt.run).parameters)
    cand = opt.run(I.get_problem("1a"), 0.2, 42, {"N_CELLS": 7})
    record(signature=params, n_values=len(cand["values"]), version=opt.version())
    assert params == ["problem", "budget_s", "seed", "config"] and len(cand["values"]) == 7
    assert isinstance(opt, I.Optimizer)


@P
def test_optimizer_problem_mismatch_rejected(record):
    opt = I.FileOptimizer(REPO / "optimizers" / "1a_baseline.py")
    with pytest.raises(ValueError) as e:
        opt.run(I.get_problem("1d"), 0.2, 42, {})
    record(message=str(e.value))


@P
def test_agent_and_store_protocols(record, tmp_path):
    """research_agent.propose(history, evidence) / review(result); experiment_store.append(result)."""
    from research import FeedbackAgent
    p = I.get_problem("1a")
    agent = FeedbackAgent(p, I.FileOptimizer(REPO / "optimizers" / "1a_baseline.py"))
    store = I.JsonlStore(tmp_path / "e.jsonl")
    exp = agent.propose([], [])
    record(propose_params=list(inspect.signature(agent.propose).parameters),
           review_params=list(inspect.signature(agent.review).parameters),
           append_params=list(inspect.signature(store.append).parameters), first_proposal=exp.hypothesis)
    assert isinstance(agent, I.ResearchAgent) and isinstance(store, I.ExperimentStore)
    assert isinstance(exp, I.ExperimentConfig)
    assert agent.review({"status": "keep"}) in I.NEXT_ACTIONS
    assert isinstance(I.NoEvidence(), I.EvidenceSource) and I.NoEvidence().gather(p, []) == []


@P
def test_record_has_contract_fields(record):
    """The experiment record carries the suggested contract fields."""
    required = {"experiment_id", "hypothesis", "algorithm_version", "config", "seed", "budget_s", "valid", "score"}
    record(fields=list(I.RECORD_FIELDS))
    assert required <= set(I.RECORD_FIELDS)
