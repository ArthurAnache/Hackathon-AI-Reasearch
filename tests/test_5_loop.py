"""Process 5: the research loop, in both modes, built on the same contract.

A. Karpathy mode: the program.md protocol executed literally with git (a scripted
   coding agent with fixed edits, so the mechanics can be asserted exactly).
B. Programmatic mode: loop.research_loop with research.FeedbackAgent: successive experiments
   chosen automatically from the history, evidence passed through, review() stops.
"""
import json
import shutil
import textwrap

import pytest

from conftest import _git, run_search

P = pytest.mark.process("5. Research loop")


def stub(values=None, crash=False):
    body = 'raise RuntimeError("bug")' if crash else f"return {{'values': {values}}}"
    return textwrap.dedent(f'''
        from prepare import run
        PROBLEM = "1a"
        CONFIG = {{}}
        def search(budget_s, seed, config=CONFIG):
            {body}
        if __name__ == "__main__":
            run(__file__)
    ''')


class KarpathyLoop:
    """program.md, step by step: branch, commit, run, grep decision, keep or git reset."""

    def __init__(self, root, tag="test"):
        self.root = root
        _git(root, "checkout", "-q", "-b", f"autoresearch/{tag}-1a")
        shutil.copy(root / "optimizers" / "1a_baseline.py", root / "search.py")
        _git(root, "add", "search.py")
        _git(root, "commit", "-qm", "start")

    def experiment(self, code, hypothesis, first=False):
        if not first:
            (self.root / "search.py").write_text(code)
            _git(self.root, "commit", "-qam", hypothesis)
        fields, _, _ = run_search(self.root, budget=1, hypothesis=hypothesis)
        if fields["decision"] == "reset":
            _git(self.root, "reset", "-q", "--hard", "HEAD~1")
        return fields["status"], fields.get("score")


@P
def test_karpathy_mode(workspace, record):
    """keep advances the branch; discard and crash are reset; the store logs all; main untouched."""
    loop = KarpathyLoop(workspace)
    (workspace / "search.py").write_text(stub("[1.0, 2.0, 1.0]"))          # 2.25
    _git(workspace, "commit", "-qam", "deterministic start")
    steps = [loop.experiment(None, "baseline", first=True),
             loop.experiment(stub("[1.0, 1.0]"), "flat profile"),            # 2.0  keep
             loop.experiment(stub("[1.0, 0.0, 0.0, 1.0]"), "two spikes"),    # 4.0  discard
             loop.experiment(stub(crash=True), "buggy refactor"),            # crash
             loop.experiment(stub("[1.0, 3.0, 3.0, 1.0]"), "smoother")]      # 2.5  discard
    record(steps=steps)
    assert [s for s, _ in steps] == ["keep", "keep", "discard", "crash", "discard"]
    log = _git(workspace, "log", "--format=%s", "autoresearch/test-1a").splitlines()
    record(branch_log=log)
    assert log[:2] == ["flat profile", "deterministic start"]
    recs = [json.loads(l) for l in (workspace / "experiments.jsonl").read_text().splitlines()]
    assert [r["status"] for r in recs] == [s for s, _ in steps]
    assert len({r["algorithm_version"] for r in recs}) == 5            # each experiment's code is identified
    tracked = _git(workspace, "ls-files").split()
    assert not {"experiments.jsonl", "run.log"} & set(tracked)
    assert _git(workspace, "status", "--porcelain").strip() == ""
    assert _git(workspace, "log", "--format=%s", "main").splitlines() == ["init"]


@pytest.fixture
def contract(workspace, monkeypatch):
    """Import prepare/interfaces/loop from the throwaway workspace (solutions and locks stay there);
    the original modules are restored afterwards."""
    import importlib, sys
    names = ("prepare", "interfaces", "loop", "analysis", "problems", "research")
    sub = lambda k: k.startswith("problems.") or k.startswith("research.")
    saved = {k: v for k, v in sys.modules.items() if k in names or sub(k)}
    monkeypatch.chdir(workspace)
    monkeypatch.syspath_prepend(str(workspace))
    for k in saved:
        del sys.modules[k]
    yield importlib.import_module("interfaces"), importlib.import_module("research")
    for k in [k for k in sys.modules if k in names or sub(k)]:
        del sys.modules[k]
    sys.modules.update(saved)


@P
def test_programmatic_loop_chooses_successive_experiments(workspace, record, contract):
    """FeedbackAgent picks each next config from the history; no config is ever repeated."""
    I, R = contract
    import loop as L
    problem = I.get_problem("1a")
    opt = I.FileOptimizer(workspace / "optimizers" / "1a_baseline.py")
    store = I.JsonlStore(workspace / "experiments.jsonl")
    results = L.research_loop(problem, opt, R.FeedbackAgent(problem, opt), store, n=4, budget_s=0.5)
    record(experiments=[{"id": r["experiment_id"], "status": r["status"], "score": r["score"],
                         "config": r["config"], "why": r["rationale"]} for r in results])
    assert len(results) >= 3
    configs = [json.dumps(r["config"], sort_keys=True) for r in results]
    assert len(set(configs)) == len(configs)
    assert results[0]["status"] == "keep" and all(r["agent"] == "feedback" for r in results)
    # the second proposal was made with the first result in view (it names its parent)
    assert "parent=exp_0000" in results[1]["rationale"]


@P
def test_review_stops_after_patience(workspace, record, contract):
    I, R = contract
    problem = I.get_problem("1a")
    agent = R.FeedbackAgent(problem, I.FileOptimizer(workspace / "optimizers" / "1a_baseline.py"), patience=3)
    actions = [agent.review({"status": s}) for s in ("keep", "discard", "crash", "discard")]
    record(actions=actions)
    assert actions == ["continue", "continue", "continue", "stop"]


@P
def test_evidence_reaches_agent_and_log(workspace, record, contract):
    """Evidence (WS4: Exa / past.dev) reaches propose(), is stored, and a config hint in it is acted on."""
    I, R = contract
    import loop as L

    class FakeExa:
        def gather(self, problem, history):
            return [I.Evidence(source="exa", content='Record construction uses 400 cells: {"config": {"N_CELLS": 77}}',
                               citation="https://example.org/c1d", query=f"{problem.id} best construction")]

    problem = I.get_problem("1a")
    opt = I.FileOptimizer(workspace / "optimizers" / "1a_baseline.py")
    store = I.JsonlStore(workspace / "experiments.jsonl")
    r0, r1 = L.research_loop(problem, opt, R.FeedbackAgent(problem, opt), store, FakeExa(), n=2, budget_s=0.5)
    record(evidence=r1["evidence"], hypothesis=r1["hypothesis"], config_used=r1["config"])
    assert r0["evidence"][0]["citation"] == "https://example.org/c1d"
    assert "strategy=evidence" in r1["rationale"] and r1["config"]["N_CELLS"] == 77
