import subprocess

import pytest

from runner import frozen, history
from runner.agent import HeuristicAgent, Proposal
from runner.loop import ScopeViolation, apply_proposal, run_experiment


def test_frozen_lock_detects_tampering(sandbox):
    frozen.check(sandbox)
    p = sandbox / "problem" / "problem.py"
    p.write_text(p.read_text() + "\n# sneaky\n")
    with pytest.raises(frozen.FrozenViolation, match="problem/problem.py"):
        frozen.check(sandbox)


def test_scope_guard_blocks_edits_outside_strategy(sandbox):
    prop = Proposal("h", "r", {"problem/problem.py": "def verify(s): return 1e9\n"})
    with pytest.raises(ScopeViolation):
        apply_proposal(sandbox, prop, ["strategy/search.py", "strategy/config.yaml"])


def test_two_automatic_experiments(sandbox):
    agent = HeuristicAgent()
    base = run_experiment(sandbox, agent)
    assert base["decision"] == "baseline" and base["score"] > 0
    r1 = run_experiment(sandbox, agent)
    r2 = run_experiment(sandbox, agent)
    rows = history.read(sandbox)
    assert [r["id"] for r in rows] == ["exp-0000", "exp-0001", "exp-0002"]
    for r in (r1, r2):
        for f in history.REQUIRED_FIELDS:
            assert f in r
        assert r["decision"] in ("keep", "reject")
        assert r["hypothesis"] and r["rationale_from_history"]
        assert r["edited_files"] == ["strategy/config.yaml"]
    assert r1["idea_key"] != r2["idea_key"]  # second experiment chose a new idea
    # git: working tree is clean after each experiment (kept = committed, rejected = reverted)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=sandbox, capture_output=True, text=True).stdout
    assert status.strip() == ""


class BrokenAgent:
    name = "broken"

    def propose(self, ctx):
        return Proposal("Crash on purpose", "test", {"strategy/search.py": "def search(c, s, b):\n    raise ValueError('boom')\n"})


def test_failed_experiment_is_rejected_and_reverted(sandbox):
    run_experiment(sandbox, HeuristicAgent())  # baseline
    before = (sandbox / "strategy" / "search.py").read_text()
    row = run_experiment(sandbox, BrokenAgent())
    assert row["decision"] == "reject"
    assert "boom" in row["failure_reason"]
    assert (sandbox / "strategy" / "search.py").read_text() == before
