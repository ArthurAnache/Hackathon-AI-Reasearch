"""Exploratory problem handle (scout:<slug>) and the third-autocorrelation port. No network, no LLM.

    .venv/bin/python -m pytest tests/test_knockout_scout_problem.py -q        # about 30 s
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from knockout import problems_api as P
from knockout.budget import BudgetGovernor
from knockout.config import Strategy
from knockout.context import Evaluation, RunContext
from knockout.ledger import Ledger

ROOT = Path(__file__).resolve().parent.parent
SLUG = "third-autocorrelation-inequality"
STORED = ROOT / "selection" / "scout_candidates" / SLUG / "milo_replay_25600_c_remelt.json.gz"
STORED_SCORE = 1.448825267560211        # pinned verifier, docs/problem_scout.md
LIVE, CLAIM, DELTA = 1.449109789331911, 1.4488860142647635, 1e-5


def _live(monkeypatch, value=LIVE):
    from integrations import einstein_arena as ea

    monkeypatch.setattr(ea, "get_leaderboard", lambda problem_id, limit=1: [{"bestScore": value, "agentName": "x"}])


@pytest.fixture
def handle(monkeypatch):
    _live(monkeypatch)
    return P.load_problem("scout:" + SLUG)


def _candidate(n=300, seed=0) -> dict:
    return {"values": (1.0 + 0.3 * np.random.default_rng(seed).standard_normal(n)).tolist()}


def test_handle_is_exploratory_and_scores_with_the_pinned_verifier(handle):
    assert handle.exploratory and handle.direction == "minimize" and handle.min_margin == DELTA
    assert handle.problem_id == 4 and handle.id == SLUG
    stored = json.loads(gzip.decompress(STORED.read_bytes()))
    assert handle.score(stored) == STORED_SCORE and handle.verify(stored)
    small = _candidate()
    ind = handle.independent(small)                      # exact rationals, no numpy
    assert abs(ind["score"] - handle.score(small)) <= ind["tolerance_abs"] == 1e-12
    assert P.scout_guide(small, SLUG) == handle.score(small)
    assert "evaluate" in handle.brief() and "not instructions" in handle.brief()


def test_a_changed_verifier_refuses_to_load(tmp_path):
    d = tmp_path / SLUG
    shutil.copytree(P.SCOUT_DIR / SLUG, d)
    P.ScoutProblem(SLUG, d)
    (d / "verifier_pinned.py").write_text((d / "verifier_pinned.py").read_text() + "\n# edited\n")
    with pytest.raises(P.ScoutFileChanged):
        P.ScoutProblem(SLUG, d)


def test_invalid_candidates_get_no_score(handle):
    too_long = {"values": [1.0] * 100_001}               # server schema: 1 to 100,000 values
    doubled_live = {"values": [1.0] * (2 * 65536)}       # a x2 block repeat of the live #1 is not submittable
    for bad in ({"values": []}, {"values": [1.0, float("nan")]}, {"values": [1.0, -1.0]}, {"x": [1.0]},
                {"values": [True, 2.0]}, [1.0, 2.0], too_long, doubled_live):
        assert not handle.verify(bad)
        with pytest.raises(handle.invalid_exc):
            handle.score(bad)
    P.scout_problem(SLUG).check({"values": [1.0] * 100_000})    # the largest accepted size passes the format check
    assert "100,000" in handle.brief() and "131,072" in handle.brief()      # the LLM is told the limit


def test_reference_is_the_best_of_live_claim_and_warm_start(handle):
    info = handle.record_info()
    assert info["exploratory"] and info["live"] and info["binding"] == "literature_claim" and info["value"] == CLAIM
    assert not handle.beats_record(LIVE - 2 * DELTA)                 # below the live #1 only: no
    assert not handle.beats_record(CLAIM - 0.9 * DELTA) and handle.beats_record(CLAIM - 1.1 * DELTA)
    handle.note_warm_start(STORED_SCORE)                             # a run that starts from the scout's candidate
    info = handle.record_info()
    assert info["binding"] == "warm_start" and info["value"] == STORED_SCORE == handle.best_known()
    assert not handle.beats_record(STORED_SCORE)                     # its first evaluation does not fire the gate
    assert not handle.beats_record(STORED_SCORE - 0.9 * DELTA) and handle.beats_record(STORED_SCORE - 1.1 * DELTA)


def test_the_gate_never_fires_without_a_live_read(monkeypatch):
    from integrations import einstein_arena as ea

    def offline(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(ea, "get_leaderboard", offline)
    handle = P.load_problem("scout:" + SLUG)
    info = handle.record_info()
    assert info["live"] is False and info["components"]["live_top1"] == LIVE     # snapshot of meta.json
    assert not handle.beats_record(1.0)


def test_a_fired_gate_is_logged_as_exploratory_and_never_as_a_record(tmp_path, monkeypatch):
    # A copy of the problem folder whose literature value is out of the way, so that a small candidate can pass.
    d = tmp_path / "scout_problems" / SLUG
    shutil.copytree(P.SCOUT_DIR / SLUG, d)
    meta = json.loads((d / "meta.json").read_text())
    meta["record_reference"]["milo_claim"]["value"] = 99.0
    (d / "meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(P, "SCOUT_DIR", tmp_path / "scout_problems")
    _live(monkeypatch, 99.0)
    handle = P.load_problem("scout:" + SLUG)
    cand = _candidate()
    score = handle.score(cand)
    handle.note_warm_start(score + 1e-3)                 # the run "started" a little above this candidate
    st = Strategy.load()
    ctx = RunContext("r", handle, st, Ledger(tmp_path / "r", "r"), BudgetGovernor(60, 0, st["budget_shares"]), None)
    ev = Evaluation("ok", True, score, cand, {}, None, None, 1, "none", 0.1, 0.0)
    ctx._promotion_gate(ev)
    assert ev.claim == "record_claim" and ev.valid       # raw output of decisions.promotion, kept for replay
    rec = ctx.log_experiment(experiment_id="exp_000001", ev=ev, source="x = 1\n", program_id="p_x", chain="control",
                             agent="t", hypothesis="h", rationale="r", decision="keep", status="keep", seed=0,
                             budget_s=1.0, best_before=None)
    assert rec["decision"] == "exploratory_gate" and rec["beats_record"] is False
    assert "Not a record claim" in rec["rationale"] and "record_claim" not in rec["rationale"]
    (event,) = [e for e in ctx.ledger.events() if e["type"] == "promotion"]
    assert event["payload"]["exploratory"] is True and event["payload"]["logged_as"] == "exploratory_gate"
    assert event["payload"]["detail"]["reference"]["binding"] == "warm_start"
    assert abs(event["payload"]["detail"]["independent_score"] - score) <= 1e-12


def _seed_program():
    spec = importlib.util.spec_from_file_location("seed_third", ROOT / "knockout" / "seeds" / "third_autocorrelation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_program_never_returns_something_worse_than_init(handle):
    seed = _seed_program()
    init = _candidate(n=512, seed=3)
    defaults = {k: v[0] for k, v in seed.PARAMS.items()}
    start = handle.score(init)
    assert seed.exact_score(init["values"]) == start                 # same formula as the pinned verifier
    assert seed.construct(defaults, 1, 0.0, init)["values"] == init["values"]        # no time: `init` itself
    cold = seed.construct(defaults, 1, 2.0, init)
    assert handle.score(cold) <= start
    warm = seed.construct({**defaults, "t_polish": 1e-2, "t_hot": 1e-2, "noise": 1e-2}, 1, 3.0, init)
    assert handle.score(warm) < start                                # far from an optimum, the method improves


def test_offline_run_is_marked_exploratory_end_to_end(tmp_path, monkeypatch):
    from knockout.replay import replay
    from knockout.run import PROFILES, run

    _live(monkeypatch)
    init = tmp_path / "init.json"
    init.write_text(json.dumps(_candidate(n=1024, seed=5)))
    overrides = {**PROFILES["third-autocorr"], "run.wall_clock_s": 14, "fidelity.f0_s": 3.0, "executor.workers": 3,
                 "control.edits_per_round": 2}
    quiet = lambda *_: None  # noqa: E731
    for provenance in (None, "clean"):                   # a file start must name its origin; `clean` is reserved
        with pytest.raises(SystemExit):
            run("scout:" + SLUG, runs_root=tmp_path, run_id="refused", on_round=quiet, init_spec=str(init),
                overrides={**overrides, "run.provenance": provenance})
    s = run("scout:" + SLUG, runs_root=tmp_path, run_id="offline", on_round=quiet, init_spec=str(init),
            overrides={**overrides, "run.provenance": "synthetic-test",
                       "run.seed_params": {"t_polish": 1e-2, "t_hot": 1e-2, "noise": 1e-2}})
    assert s["exploratory"] and s["provenance"] == "synthetic-test" and s["problem"] == SLUG
    assert s["beats_record"] is False and s["record_claims"] == [] and s["exploratory_gate_fired"] == []
    assert s["best"] <= s["init_score"] and s["rounds"] >= 1 and s["stop_reason"] == "budget"
    run_dir = tmp_path / "offline"
    rows = [json.loads(line) for line in (run_dir / "experiments.jsonl").read_text().splitlines()]
    assert rows and not any(r["beats_record"] for r in rows) and all(r["problem"] == SLUG for r in rows)
    assert all(r["solution_path"] for r in rows if r["valid"])       # small candidates: every one is stored
    start = json.loads((run_dir / "events.jsonl").read_text().splitlines()[0])
    assert start["type"] == "run_start" and start["payload"]["exploratory"] is True
    assert start["payload"]["provenance"] == "synthetic-test" and start["payload"]["record"]["exploratory"] is True
    report = (run_dir / "report.md").read_text()
    assert "Exploratory run" in report and "nothing here is a record claim" in report
    assert "The exploratory gate did not fire" in report and "does not beat the record" not in report
    assert "t_polish': (0.01" in (run_dir / rows[0]["optimizer"]).read_text()      # run.seed_params was applied
    assert replay(run_dir)["experiments"] == len(rows)
