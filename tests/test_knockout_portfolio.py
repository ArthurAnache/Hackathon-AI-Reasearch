"""Start portfolio (docs/knockout_portfolio_brief.md, section 8). Fast unit tests plus one recorded toy run with the
fake LLM of tests/test_knockout_m1_m5.py (no network, no cost). Tests 2 of the brief (cold starts, perturbations)
are in tests/test_knockout_portfolio_starts.py; test 9 is the v1 test suite with the portfolio off (the default).
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from knockout import decisions as D
from knockout import portfolio as P
from knockout.basins import Archive
from knockout.context import Evaluation

sys.path.insert(0, str(Path(__file__).parent))
from test_knockout_m1_m5 import _FakeLLMClient  # noqa: E402


def _ev(score, valid=True, cand=None):
    return Evaluation("ok" if valid else "invalid", valid, score if valid else None, cand, {}, None,
                      None if valid else "bad", 1, None, 1.0, 0.01)


# ------------------------------------------------------------------ 1. regime choice
def test_pick_regime_floor_then_thompson():
    arms = {"record": [1.0, 1.0], "neighbour": [1.0, 1.0], "cold": [1.0, 1.0]}
    floors = {"record": 0.2, "neighbour": 0.2, "cold": 0.2}
    out = D.pick_regime(arms, {}, floors, 3)
    assert out == {"choice": "cold", "reason": "floor", "shares": {"cold": 0.0, "neighbour": 0.0, "record": 0.0},
                   "draws": {}}                                            # equal shortfalls: ties by name
    out = D.pick_regime(arms, {"record": 10.0, "neighbour": 1.0, "cold": 3.0}, floors, 3)
    assert out["choice"] == "neighbour" and out["reason"] == "floor"       # largest shortfall
    spend = {"record": 1.0, "neighbour": 1.0, "cold": 1.0}
    a, b = D.pick_regime(arms, spend, floors, 11), D.pick_regime(arms, spend, floors, 11)
    assert a == b and a["reason"] == "thompson" and set(a["draws"]) == set(arms)


# ------------------------------------------------------------------ 4. per-start aggregation
def test_regime_score_is_the_mean_and_the_best_start_is_kept():
    sset = P.StartSet("cold:0", "cold", ("a", "b", "c"), (None, 0.5, None), ("cold:0", "cold:1", "cold:2"))
    agg = P.aggregate([_ev(2.0, cand="x"), _ev(None, valid=False), _ev(3.0, cand="z")], sset,
                      [None, None, None], "maximize")
    assert agg.valid and agg.score == 3.0 and agg.candidate == "z"
    assert agg.regime_score == pytest.approx((2.0 + 0.5 + 3.0) / 3)      # invalid on start 1: the start's score
    assert [p["rule"] for p in agg.per_start] == ["own", "start_score", "own"]
    # invalid on a start that is itself invalid: the worst score seen there, else the program is invalid
    agg = P.aggregate([_ev(2.0), _ev(1.0), _ev(None, valid=False)], sset, [None, None, 1.5], "maximize")
    assert agg.regime_score == pytest.approx((2.0 + 1.0 + 1.5) / 3)
    agg = P.aggregate([_ev(2.0), _ev(1.0), _ev(None, valid=False)], sset, [None, None, None], "maximize")
    assert not agg.valid and agg.regime_score is None and agg.candidate is None
    assert P.update_worst([None, 0.9, None], [_ev(2.0), _ev(1.0), _ev(None, valid=False)], "maximize") == \
        [2.0, 0.9, None]
    assert P.selection_value(_ev(4.0)) == 4.0


# ------------------------------------------------------------------ 6. archive
def test_archive_inserts_replaces_on_strict_improvement_only():
    describe = lambda c: {"x": c["x"]}                                    # noqa: E731
    arc = Archive.from_samples([{"x": 0.0}, {"x": 1.0}], describe, 4, "maximize", 1e-6)
    assert arc.edges == {"x": [0.25, 0.5, 0.75]}
    first = arc.offer({"x": 0.1}, 1.0, "exp_1", "cold", "p1")
    assert first["cell"] == "c0" and first["previous"] is None
    assert arc.offer({"x": 0.2}, 1.0 + 1e-9, "exp_2", "cold", "p2") is None     # not a strict improvement
    second = arc.offer({"x": 0.2}, 1.5, "exp_3", "record", "p3")
    assert second["previous"] == {"score": 1.0, "experiment": "exp_1"}
    arc.offer({"x": 0.9}, 2.0, "exp_4", "record", "p4")
    assert [e.experiment for e in arc.others({"x": 0.95})] == ["exp_3"]       # the incumbent's cell is excluded
    assert [e.experiment for e in arc.others({"x": 0.05})] == ["exp_4"]


def test_strategist_ranges_for_portfolio_keys():
    from knockout.config import Strategy, StrategyError

    s = Strategy.load()
    s.apply_patch({"portfolio.floors": {"record": 0.3, "neighbour": 0.3, "cold": 0.2}})
    for bad in ({"portfolio.floors": {"record": 0.5}}, {"portfolio.floors": {"a": 0.4, "b": 0.4, "c": 0.4}},
                {"portfolio.k_cold": 9}, {"portfolio.promote_gap": 0.5}):
        with pytest.raises(StrategyError):
            s.apply_patch(bad)


# ------------------------------------------------------------------ recorded toy run (tests 3, 5, 7, 8, 10)
class _RecordingClient(_FakeLLMClient):
    def __init__(self):
        super().__init__()
        self.prompts: list[str] = []

    def create(self, **kwargs):
        if kwargs["text"]["format"]["name"] in ("program", "program_reformulate"):   # main arm only
            self.prompts.append(kwargs["input"])
        return super().create(**kwargs)


@pytest.fixture(scope="module")
def portfolio_run(tmp_path_factory):
    import knockout.meta
    import knockout.run as run_module
    from knockout.llm_editor import OpenAIEditor

    tmp = tmp_path_factory.mktemp("pf")
    client = _RecordingClient()
    real_scout, knockout.meta.scout = knockout.meta.scout, lambda arm, reason: 0
    overrides = {"run.wall_clock_s": 45, "run.max_usd": 100.0, "fidelity.f0_s": 1.0, "fidelity.f1_s": 2.0,
                 "fidelity.levels": 2, "tuner.calls": 1, "tuner.top_k": 1, "executor.workers": 6,
                 "control.edits_per_round": 2, "main.batch_size": 6, "gate.pairs_per_batch": 3,
                 "editor.kind": "openai", "selection.eps_rel": 1e-6, "islands.max_islands": 3,
                 "portfolio.enabled": True, "prompt.show_record": False,
                 "portfolio.promote_gap": 1.0, "portfolio.deep_factor": 2}   # deep polish fires on the toy

    class Editor(OpenAIEditor):
        def __init__(self, ctx):
            super().__init__(ctx, client=client)

    real_make = run_module.make_editor
    run_module.make_editor = lambda ctx: Editor(ctx)
    try:
        summary = run_module.run("toy", runs_root=tmp, run_id="pf", overrides=overrides, on_round=lambda *_: None)
    finally:
        run_module.make_editor = real_make
        knockout.meta.scout = real_scout
    return tmp / "pf", summary, client


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_run_validates_replays_reports_and_uses_every_regime(portfolio_run):
    from knockout.ledger import validate_run
    from knockout.replay import replay

    run_dir, summary, _ = portfolio_run
    assert validate_run(run_dir) == summary["validation"]
    out = replay(run_dir)
    assert out["checked"].get("regime_pick", 0) >= 3
    picks = {e["payload"]["outputs"]["choice"] for e in _jsonl(run_dir / "events.jsonl") if e["type"] == "regime_pick"}
    assert picks == {"record", "neighbour", "cold"}
    report = (run_dir / "report.md").read_text()
    assert "## Start portfolio" in report and "### Basin archive" in report


def test_parent_is_evaluated_once_per_start_set(portfolio_run):
    run_dir, _, _ = portfolio_run
    replays = Counter((r["config"]["member"], r["config"]["start_set"]) for r in _jsonl(run_dir / "experiments.jsonl")
                      if r["config"].get("operator") == "replay")
    assert replays and max(replays.values()) == 1


def test_ledger_score_is_the_stored_candidate_and_regime_score_the_mean(portfolio_run):
    from knockout import toy

    run_dir, _, _ = portfolio_run
    rows = [r for r in _jsonl(run_dir / "experiments.jsonl") if r["config"].get("per_start") and r["valid"]]
    assert rows
    for r in rows:
        vals = [p["value"] for p in r["config"]["per_start"]]
        assert r["config"]["regime_score"] == pytest.approx(sum(vals) / len(vals), rel=0, abs=1e-12)
        own = [p["score"] for p in r["config"]["per_start"] if p["score"] is not None]
        assert r["score"] == max(own)
    stored = [r for r in rows if r.get("solution_path")][:10]
    assert stored
    for r in stored:
        path = Path(r["solution_path"])
        path = path if path.is_absolute() else run_dir / path
        if not path.exists():
            path = run_dir.parent / r["solution_path"]
        raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        assert toy.score(json.loads(raw)) == r["score"]


def test_gene_stats_are_keyed_by_regime_and_phase(portfolio_run):
    run_dir, _, _ = portfolio_run
    state = json.loads((run_dir / "state.json").read_text())
    keys = {k for g in state["genes"] for k in g.get("stats", {})}
    assert keys and all(k.split("/")[0] in P.REGIMES and "/" in k for k in keys)
    assert D.gene_posterior({"cold/explore": [2.0, 0.0, 1], "record/climb": [0.0, 3.0, 5]}, "cold/explore", 3) == \
        [3.0, 4.0]                                                            # few observations: pooled


def test_deep_polish_fires_once_per_entry_and_per_batch(portfolio_run):
    run_dir, _, _ = portfolio_run
    events = _jsonl(run_dir / "events.jsonl")
    promos = [e for e in events if e["type"] == "basin_promotion"]
    assert promos, "promote_gap = 1.0 should promote at least one archive entry on the toy"
    entries = [(p["payload"]["entry"]["cell"], p["payload"]["entry"]["experiment"]) for p in promos]
    assert len(entries) == len(set(entries))
    results = {p["refs"][0] for p in promos}
    assert not any(p["payload"]["entry"]["experiment"] in results for p in promos)   # a result is not re-promoted
    batch, per_batch = 0, Counter()
    for e in events:
        batch += e["type"] == "regime_pick"
        per_batch[batch] += e["type"] == "basin_promotion"
    assert max(per_batch.values()) <= 1
    for p in promos:      # a promotion above the incumbent must have made it the new incumbent
        pl = p["payload"]
        if pl["new_incumbent_from_other_basin"]:
            assert any(e["type"] == "warm_start_update" and e["refs"] == [p["refs"][0]] for e in events)


def test_prompt_hides_the_record_and_names_the_regime(portfolio_run):
    _, _, client = portfolio_run
    assert client.prompts
    assert all("# Regime of this batch" in p for p in client.prompts)
    assert not any("Record to beat" in p or "record threshold" in p for p in client.prompts)
    assert any("Regime: `cold`" in p for p in client.prompts) and any("Regime: `neighbour`" in p
                                                                       for p in client.prompts)
