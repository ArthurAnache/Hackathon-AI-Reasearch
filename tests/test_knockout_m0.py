"""M0 tests for Knockout Evolution: tuner, sandbox, toy verifier, ledger schema, budget, config, editor, end to end.

    .venv/bin/python -m pytest tests/test_knockout_m0.py -q
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pytest

from knockout import toy
from knockout.budget import BudgetGovernor
from knockout.config import Strategy, StrategyError
from knockout.editor import OfflineParamEditor, read_params, write_params
from knockout.ledger import ExperimentRecord, Ledger, validate_run
from knockout.problems_api import load_problem
from knockout.sandbox import SandboxJob, run_job
from knockout.tuner import ParamSpace, tune

ROOT = Path(__file__).resolve().parent.parent
SEED_SRC = (ROOT / "knockout" / "seeds" / "toy_circles.py").read_text()


# ------------------------------------------------------------------ tuner
def test_param_space_int_and_log_roundtrip():
    s = ParamSpace({"n": (5, 1, 10), "lr": (0.01, 1e-4, 1.0, "log")})
    d = s.decode(s.default_u())
    assert d["n"] == 5 and isinstance(d["n"], int)
    assert d["lr"] == pytest.approx(0.01)


def test_tuner_improves_on_defaults():
    space = ParamSpace({"a": (0.0, -5.0, 5.0), "b": (0.0, -5.0, 5.0)})

    def objective(p, _):
        return -((p["a"] - 2) ** 2 + (p["b"] + 1) ** 2), dict(p)

    res = tune(objective, space, seed=1, budget_s=0.5, calls=200, top_k=2)
    assert res.n_evals > 10
    assert res.trials_top[0].value > -5.0          # defaults give -5
    assert res.method == "cma"


# ------------------------------------------------------------------ toy verifier (exact)
def _toy_candidate(rows):
    return {"circles": rows + [[0.95, 0.95, 0.001]] * 0}


def test_toy_exact_check_tangent_ok_overlap_rejected():
    # 26 tiny circles on a grid, then make two exactly tangent / overlapping
    rows = [[0.1 + 0.15 * (i % 6), 0.1 + 0.15 * (i // 6), 0.01] for i in range(26)]
    assert toy.verify({"circles": rows})
    tangent = [list(r) for r in rows]
    tangent[0] = [0.25, 0.5, 0.25]
    tangent[1] = [0.75, 0.5, 0.25]
    others = [[0.05 + 0.04 * i, 0.95, 0.01] for i in range(24)]
    assert toy.verify({"circles": tangent[:2] + others})
    overlap = [[0.25, 0.5, 0.25], [0.7499999999999999, 0.5, 0.25]] + others
    assert not toy.verify({"circles": overlap})
    outside = [[0.25, 0.5, 0.2500000000000001], [0.75, 0.5, 0.2]] + others
    assert not toy.verify({"circles": outside})


# ------------------------------------------------------------------ sandbox
def _job(src, budget=3.0, **kw):
    return SandboxJob(program_source=src, seed=0, budget_s=budget, direction="maximize",
                      guide_ref="knockout.toy:guide_score", **kw)


SMALL = "PARAMS = {}\n" + "def construct(params, seed, budget_s, init=None):\n" + \
        "    return {'circles': [[0.1 + 0.15 * (i % 6), 0.1 + 0.15 * (i // 6), 0.01] for i in range(26)]}\n"


def test_sandbox_runs_a_program_and_hides_secrets(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "should-not-leak")
    src = SMALL.replace("    return", "    import os\n    assert 'EXA_API_KEY' not in os.environ\n    return")
    r = run_job(_job(src))
    assert r.ok, r.error
    assert r.trials[0]["guide"] == pytest.approx(0.26)


def test_sandbox_timeout_is_enforced():
    src = "import time\nPARAMS = {}\ndef construct(params, seed, budget_s, init=None):\n    time.sleep(60)\n"
    r = run_job(_job(src, budget=0.5), timeout_slack_s=1.0)
    assert not r.ok and r.timed_out


@pytest.mark.skipif(platform.system() != "Darwin", reason="network/file isolation is checked with sandbox-exec")
def test_sandbox_denies_network_and_outside_writes(tmp_path):
    net = SMALL.replace("    return", "    import socket\n    socket.create_connection(('1.1.1.1', 80), timeout=2)\n    return")
    r = run_job(_job(net))
    assert not r.ok and r.trials == []
    target = tmp_path / "escape.txt"
    write = SMALL.replace("    return", f"    open({str(target)!r}, 'w').write('x')\n    return")
    r = run_job(_job(write))
    assert not r.ok and not target.exists()


def test_sandbox_reports_program_errors_without_raising():
    r = run_job(_job("PARAMS = {}\ndef construct(params, seed, budget_s, init=None):\n    raise RuntimeError('boom')\n"))
    assert not r.ok and "boom" in (r.error or "")


# ------------------------------------------------------------------ ledger
def _record(**over):
    rec = {
        "experiment_id": "exp_000001", "hypothesis": "h", "config": {}, "seed": 1, "budget_s": 2.0, "valid": True,
        "score": 1.0, "best_known_score": 2.0, "decision": "keep", "rationale": "r", "evidence_urls": [],
        "timestamp": "t", "problem": "toy", "agent": "a", "optimizer": "p.py", "algorithm_version": "abc",
        "git_commit": None, "status": "keep", "best_before": None, "beats_record": False, "duration_s": 1.0,
        "error": None, "evidence": [], "solution_path": None, "run_id": "r", "source_commit": None, "device": "cpu",
        "gpu_type": None, "dtype": "float64", "torch_version": None, "cuda_version": None, "gpu_objective": None,
        "wall_s": 1.0,
    }
    rec.update(over)
    return rec


def test_ledger_schema_accepts_union_and_rejects_extra_or_missing(tmp_path):
    led = Ledger(tmp_path, "r")
    led.append(_record())
    with pytest.raises(Exception):
        led.append(_record(extra_field=1))
    bad = _record()
    del bad["best_known_score"]
    with pytest.raises(Exception):
        led.append(bad)
    with pytest.raises(Exception):
        led.event("not_a_type", seed=1)
    led.event("run_start", seed=1, refs=[], x=1)
    assert validate_run(tmp_path) == {"experiments.jsonl": 1, "events.jsonl": 1}


def test_ledger_shared_fields_are_present():
    shared = {"experiment_id", "hypothesis", "config", "seed", "budget_s", "valid", "score", "best_known_score",
              "decision", "rationale", "evidence_urls"}
    ws2 = {"experiment_id", "timestamp", "problem", "hypothesis", "rationale", "agent", "optimizer",
           "algorithm_version", "git_commit", "config", "seed", "budget_s", "valid", "score", "status",
           "best_before", "beats_record", "duration_s", "error", "evidence", "solution_path"}
    assert shared | ws2 <= set(ExperimentRecord.model_fields)


def test_validate_run_detects_duplicate_ids(tmp_path):
    (tmp_path / "experiments.jsonl").write_text(json.dumps(_record()) + "\n" + json.dumps(_record()) + "\n")
    (tmp_path / "events.jsonl").write_text("")
    with pytest.raises(ValueError, match="duplicate"):
        validate_run(tmp_path)


# ------------------------------------------------------------------ budget and config
def test_budget_control_share_and_caps():
    t = [0.0]
    g = BudgetGovernor(100, 0, {"control": 0.2, "gate_max": 0.25, "meta_max": 0.02}, clock=lambda: t[0])
    assert g.control_owed()
    g.charge("main", compute_s=90)
    g.charge("control", compute_s=10)
    assert g.control_owed()                       # 10 < 0.2 * 100
    g.charge("control", compute_s=20)
    assert not g.control_owed()                   # 30 >= 0.2 * 120
    g.charge("gate", compute_s=60)
    assert not g.within_cap("gate")               # 60 >= 0.25 * 180
    assert not g.exhausted()
    t[0] = 100.0
    assert g.exhausted()


def test_budget_usd_cap():
    g = BudgetGovernor(1e9, 1.0, {})
    g.charge("main", usd=0.6)
    assert g.can_spend_usd("main", 0.3) and not g.can_spend_usd("main", 0.5)
    g.charge("control", llm_usd=0.5)
    assert g.exhausted()


def test_strategy_patch_allow_list():
    s = Strategy.load()
    prev = s.apply_patch({"genes.g_max": 6})
    assert s["genes.g_max"] == 6 and prev == {"genes.g_max": 8}
    with pytest.raises(StrategyError):
        s.apply_patch({"sandbox.deny_network": False})


# ------------------------------------------------------------------ editor
def test_offline_editor_rewrites_params_only():
    params = read_params(SEED_SRC)
    assert write_params(SEED_SRC, params).count("def construct") == 1
    props = OfflineParamEditor().propose(SEED_SRC, 6, np.random.default_rng(0))
    assert len(props) == 6 and len({p.source for p in props}) == 6
    for p in props:
        assert p.source != SEED_SRC
        ParamSpace(read_params(p.source))         # still a valid PARAMS literal
        assert p.source.split("PARAMS = {")[0] == SEED_SRC.split("PARAMS = {")[0]


# ------------------------------------------------------------------ Modal executor (no network: fake .remote)
def test_modal_executor_roundtrip_and_cost():
    from dataclasses import asdict

    from knockout.executor import ModalExecutor

    class FakeRemote:
        def remote(self, job: dict, cfg: dict) -> dict:
            json.dumps(job), json.dumps(cfg)          # what Modal would serialize
            return asdict(run_job(SandboxJob(**job), **cfg))

    ex = ModalExecutor(FakeRemote(), 2, {"timeout_slack_s": 5.0, "memory_mb": 2048, "deny_network": True,
                                         "max_output_mb": 64}, cpu=2.0, memory_mb=2048, usd_per_core_s=1e-5,
                       usd_per_gib_s=1e-6, overhead_s=2.0)
    r = ex.submit(_job(SMALL)).result()
    ex.shutdown()
    assert r.ok and r.trials[0]["guide"] == pytest.approx(0.26)
    assert ex.cost_usd(r) == pytest.approx((r.elapsed_s + 2.0) * (2 * 1e-5 + 2 * 1e-6))


# ------------------------------------------------------------------ end to end (toy, local executor)
def test_control_arm_end_to_end(tmp_path):
    from knockout.run import run

    summary = run("toy", runs_root=tmp_path, on_round=lambda *_: None,
                  overrides={"run.wall_clock_s": 20, "fidelity.f0_s": 1.0, "control.edits_per_round": 4,
                             "executor.workers": 4})
    assert summary["validation"]["experiments.jsonl"] >= 5
    assert summary["best"] >= summary["baseline"] > 2.5
    p = load_problem("toy")
    run_dir = tmp_path / summary["run_id"]
    recs = [json.loads(line) for line in (run_dir / "experiments.jsonl").read_text().splitlines()]
    best = next(r for r in recs if r["experiment_id"] == summary["best_experiment"])
    assert p.score(json.loads((run_dir / best["solution_path"]).read_text())) == best["score"]


# ------------------------------------------------------------------ LLM editor, warm-start ratchet, secrets
class _FakeResponses:
    def __init__(self, answers):
        self.answers, self.requests = list(answers), []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        answer = self.answers[(len(self.requests) - 1) % len(self.answers)]
        if isinstance(answer, Exception):
            raise answer
        usage = type("Usage", (), {"input_tokens": 1000, "output_tokens": 2000})()
        return type("Response", (), {"id": f"resp_{len(self.requests)}", "usage": usage,
                                     "output_text": json.dumps(answer)})()


class _FakeClient:
    def __init__(self, answers):
        self.responses = _FakeResponses(answers)
        self.models = type("Models", (), {"retrieve": staticmethod(lambda name: {"id": name})})()


def _llm_ctx(tmp_path, max_usd=1.0, **overrides):
    from knockout.context import RunContext
    from knockout.executor import make_executor

    strategy = Strategy.load(overrides={"run.max_usd": max_usd, "executor.workers": 2, **overrides})
    ledger = Ledger(tmp_path / "run", "run_llm")
    governor = BudgetGovernor(60, max_usd, strategy["budget_shares"])
    return RunContext("run_llm", load_problem("toy"), strategy, ledger, governor, make_executor(strategy))


def test_llm_editor_validates_charges_and_logs(tmp_path):
    from knockout.llm_editor import OpenAIEditor, check_program

    ctx = _llm_ctx(tmp_path)
    child = SEED_SRC.replace('"jitter": (0.03, 0.0, 0.15)', '"jitter": (0.05, 0.0, 0.15)')
    assert child != SEED_SRC and check_program(child, SEED_SRC) is None
    answers = [{"hypothesis": "more jitter escapes the grid", "rationale": "r", "source": "```python\n" + child + "```"},
               {"hypothesis": "broken", "rationale": "r", "source": "def construct(:\n"},
               {"hypothesis": "same", "rationale": "r", "source": SEED_SRC},
               RuntimeError("boom")]
    client = _FakeClient(answers)
    editor = OpenAIEditor(ctx, client=client)
    editor.preflight()
    out = editor.propose(SEED_SRC, 4, np.random.default_rng(0))
    # the 4 calls run in parallel threads, so which operator got the valid answer is not fixed
    assert len(out) == 1 and out[0].source.strip() == child.strip() and out[0].operator in editor.operators
    assert out[0].description == "more jitter escapes the grid"
    events = ctx.ledger.events()
    calls = [e for e in events if e["type"] == "llm_call"]
    assert len(calls) == 4 and sum(e["payload"]["ok"] for e in calls) == 3
    assert sorted(e["payload"]["reason"].split(":")[0] for e in events if e["type"] == "filter_reject") == \
        ["identical to the parent program", "syntax error"]
    # 3 answered calls x (1000 in at 2 USD/M + 2000 out at 10 USD/M); the failed call costs nothing
    assert ctx.governor.llm_usd["control"] == pytest.approx(3 * (1000 * 2e-6 + 2000 * 10e-6))
    request = client.responses.requests[0]
    assert request["model"] == ctx.strategy["llm.model"] and "Incumbent program" in request["input"]
    assert "API_KEY" not in request["input"] and "API_KEY" not in request["instructions"]
    ctx.executor.shutdown()


def test_llm_editor_stops_at_the_usd_cap(tmp_path):
    from knockout.llm_editor import OpenAIEditor

    ctx = _llm_ctx(tmp_path, max_usd=0.3)       # one worst-case call is ~0.2 USD (20k output tokens)
    client = _FakeClient([{"hypothesis": "h", "rationale": "r", "source": SEED_SRC + "\n# edit\n"}])
    editor = OpenAIEditor(ctx, client=client)
    out = editor.propose(SEED_SRC, 4, np.random.default_rng(0))
    assert len(client.responses.requests) == 1 and len(out) == 1 and editor.budget_stopped
    ctx.executor.shutdown()


def test_check_program_rejects_bad_programs():
    from knockout.llm_editor import check_program

    assert check_program("", SEED_SRC) == "empty source"
    assert "construct" in check_program("PARAMS = {}\ndef build(p, s, b, init=None):\n    return 1\n", SEED_SRC)
    assert "bad PARAMS" in check_program("PARAMS = {'a': (5, 0, 1)}\ndef construct(p, s, b, init=None):\n    return 1\n",
                                         SEED_SRC)


def test_ratchet_run_end_to_end(tmp_path):
    """Toy problem with the warm-start ratchet: replays are logged, the best candidate never gets worse."""
    from knockout.run import run

    s = run("toy", runs_root=tmp_path, run_id="ratchet", on_round=lambda *_: None,
            overrides={"run.wall_clock_s": 14, "fidelity.f0_s": 2.0, "control.edits_per_round": 2,
                       "control.ratchet": True, "executor.workers": 3, "ledger.save_solutions": "selected"})
    rows = [json.loads(line) for line in (tmp_path / "ratchet" / "experiments.jsonl").read_text().splitlines()]
    replays = [r for r in rows if r["config"].get("operator") == "replay"]
    assert s["rounds"] >= 1 and len(replays) == s["rounds"]
    assert s["best"] >= s["baseline"] and s["stop_reason"] == "budget"
    best = next(r for r in rows if r["experiment_id"] == s["best_experiment"])
    assert best["solution_path"] == s["best_solution_path"]
    assert toy.score(json.loads((tmp_path / "ratchet" / best["solution_path"]).read_text())) == s["best"]
    discarded = [r for r in rows if r["decision"] == "discard"]
    assert all(r["solution_path"] is None for r in discarded)      # "selected" storage


@pytest.mark.skipif(platform.system() != "Darwin", reason="sandbox-exec profile (macOS)")
def test_sandbox_denies_reading_env_file():
    env = ROOT / ".env"
    if not env.exists():
        pytest.skip("no .env in this checkout")
    src = SMALL.replace("    return", f"    open({str(env)!r}).read()\n    return")
    r = run_job(_job(src))
    assert not r.ok and "PermissionError" in (r.error or "") + r.stderr_tail
