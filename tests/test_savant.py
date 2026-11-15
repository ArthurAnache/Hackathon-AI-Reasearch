"""Offline SAVANT tests: ingest, stats, problem report, mock council. No paid LLM/Modal."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from savant.config import resolve_seed_export

EXPORT = resolve_seed_export()


@pytest.mark.skipif(not EXPORT.exists(), reason="WS3 Modal export missing")
@pytest.mark.process("savant")
def test_ingest_real_modal_export(tmp_path, record):
    from savant.ingest import ingest_export, validate_export
    from savant.memory import ResearchMemory

    rep = validate_export(EXPORT)
    assert rep["issues"] == []
    assert rep["run_id"] == "20261115T015342Z-b60"
    assert abs(rep["baseline_score"] - 0.9636386679928123) < 1e-15
    assert abs(rep["best"]["score"] - 0.9636386831034217) < 1e-15
    assert abs(rep["probe"]["mean"] - 3.9005e-9) < 5e-13
    assert abs(rep["blockshift"]["mean"] - 1.3727e-8) < 5e-13
    assert rep["paired_blockshift_vs_probe"]["sign_positive"] == 3
    assert rep["paired_blockshift_vs_probe"]["claim_allowed"] is False
    assert abs(rep["paired_blockshift_vs_probe"]["p_sign_flip_one_sided"] - 0.125) < 1e-12

    mem = ResearchMemory(tmp_path / "research_memory.jsonl")
    ingest_export(EXPORT, mem)
    kinds = {e["kind"] for e in mem.read()}
    assert {"ingest", "experiment", "uncertainty", "rejected_idea"} <= kinds
    assert sum(1 for e in mem.read() if e["kind"] == "experiment") == 6
    record(probe_mean=rep["probe"]["mean"], block_mean=rep["blockshift"]["mean"],
           p=rep["paired_blockshift_vs_probe"]["p_sign_flip_one_sided"])


@pytest.mark.process("savant")
def test_stats_n3_cannot_claim_significance(record):
    from savant.stats import paired_comparison
    # all-positive tiny diffs like the smoke test
    a = [1.26e-8, 1.35e-8, 1.51e-8]
    b = [2.60e-9, 3.35e-9, 5.75e-9]
    rep = paired_comparison(a, b, label_a="blockshift", label_b="probe", seeds=[101, 202, 303],
                            decision_index=0, alpha_spend=0.005)
    assert rep["n"] == 3
    assert rep["claim_allowed"] is False
    assert rep["p_sign_flip_one_sided"] == 0.125
    assert "cannot" in rep["interpretation"].lower() or "exploratory" in rep["interpretation"].lower()
    record(mean_diff=rep["mean_diff"], p=rep["p_sign_flip_one_sided"])


@pytest.mark.process("savant")
def test_problem_report_deterministic(tmp_path, record):
    from savant.problem_report import build_problem_report
    if not EXPORT.exists():
        pytest.skip("export missing")
    r1 = build_problem_report(EXPORT)
    r2 = build_problem_report(EXPORT)
    assert r1["llm_used"] is False
    assert r1["content_sha256"] == r2["content_sha256"]
    assert r1["measured_facts"]["verified_baseline_score"] > 0.96
    assert "facts" in str(r1.keys()) or "measured_facts" in r1
    assert r1["unknowns"] and r1["mathematical_assumptions"]
    record(sha=r1["content_sha256"][:12], n=r1["problem"]["formulation"]["dimension_n"])


@pytest.mark.process("savant")
def test_openai_strict_schemas():
    """Every SAVANT structured-output schema must satisfy OpenAI strict mode rules."""
    from savant.schemas import (
        CONFIG_OVERRIDE_KEYS, CRITIQUE_JSON_SCHEMA, HYPOTHESIS_JSON_SCHEMA,
        REVISION_JSON_SCHEMA, STOP_VOTE_JSON_SCHEMA, VOTE_JSON_SCHEMA,
        assert_openai_strict_compatible, ensure, normalize_config_overrides,
    )
    for name, schema in (
        ("hypothesis", HYPOTHESIS_JSON_SCHEMA),
        ("critique", CRITIQUE_JSON_SCHEMA),
        ("revision", REVISION_JSON_SCHEMA),
        ("vote", VOTE_JSON_SCHEMA),
        ("stop_vote", STOP_VOTE_JSON_SCHEMA),
    ):
        errs = assert_openai_strict_compatible(schema, path=name)
        assert errs == [], f"{name}: {errs}"
    cfg_schema = HYPOTHESIS_JSON_SCHEMA["properties"]["config_overrides"]
    assert cfg_schema["additionalProperties"] is False
    assert set(cfg_schema["required"]) == set(CONFIG_OVERRIDE_KEYS) == set(cfg_schema["properties"])
    # Nullable overrides (as OpenAI returns unused keys) normalize to a sparse dict
    cleaned = normalize_config_overrides({
        "GUIDED": 0.85, "GAP_REL": None, "BLOCK_LOG2_MAX": None, "UNKNOWN": 1,
    })
    assert cleaned == {"GUIDED": 0.85}
    h = ensure("hypothesis", {
        "hypothesis_id": "H-test", "role": "numerical", "title": "t", "claim": "c",
        "mechanism": "m", "algorithmic_change": "a", "template": "blockshift_mutate",
        "config_overrides": {k: None for k in CONFIG_OVERRIDE_KEYS} | {"D_GRID": 18},
        "expected_gain": 1e-8, "expected_gain_lo": 0, "expected_gain_hi": 1e-7,
        "risk": 0.3, "assumptions": ["a"], "falsification": "f", "compute_jobs": 3,
        "novelty": 0.4, "literature_refs": None, "notes": None,
    })
    assert h["config_overrides"] == {"D_GRID": 18}
    assert h["literature_refs"] == []


@pytest.mark.process("savant")
def test_polish_frac_rejected_on_blockshift():
    """POLISH_FRAC belongs to probe. On a blockshift template it is dropped, not a crash."""
    from savant.debate import manager_select
    from savant.memory import ResearchMemory
    from savant.templates import materialize, sanitize_overrides
    cfg, rejected = sanitize_overrides("blockshift_mutate", {
        "GUIDED": 0.7, "POLISH_FRAC": 0.8, "BLOCK_LOG2_MAX": None, "INIT": "/tmp/x",
    })
    assert cfg == {"GUIDED": 0.7}
    reasons = {r["key"]: r["reason"] for r in rejected}
    assert "not a parameter" in reasons["POLISH_FRAC"]
    assert reasons["BLOCK_LOG2_MAX"] == "null"
    assert "INIT" in reasons
    mat = materialize({
        "hypothesis_id": "H-polish", "role": "mathematical", "template": "blockshift_mutate",
        "title": "t", "config_overrides": {"POLISH_FRAC": 0.8, "GUIDED": 0.7},
        "claim": "c", "expected_gain": 1e-8, "falsification": "f",
    })
    assert mat["executable"] is True and "POLISH_FRAC" not in mat["config"]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        memory = ResearchMemory(Path(tmp) / "research_memory.jsonl")
        picked = manager_select(
            [{"hypothesis_id": "H-polish", "score": 1, "hypothesis": {
                "hypothesis_id": "H-polish", "role": "mathematical", "template": "blockshift_mutate",
                "config_overrides": {"POLISH_FRAC": 0.8, "GUIDED": 0.85},
            }}],
            [{"hypothesis_id": "H-polish", "role": "mathematical", "template": "blockshift_mutate",
              "config_overrides": {"POLISH_FRAC": 0.8, "GUIDED": 0.85}}],
            max_select=2, memory=memory)
        assert [h["hypothesis_id"] for h in picked] == ["H-polish"]
        saved = REPO / "runs" / "savant" / "iter1" / "research_memory.jsonl"
        if saved.exists():
            import shutil
            from savant.debate import council_from_memory, finish_council
            copy = Path(tmp) / "iter1.jsonl"
            shutil.copy(saved, copy)
            cached = council_from_memory(ResearchMemory(copy))
            assert cached is not None and len(cached["hypotheses"]) == 5
            assert len(cached["critiques"]) == 20 and len(cached["votes"]) == 5
            done = finish_council(cached["hypotheses"], cached["critiques"], cached["votes"],
                                  ResearchMemory(copy), voting_seed=20261116, max_select=2)
            for mat in done["materialized"]:
                assert mat["executable"] and "POLISH_FRAC" not in mat["config"]
            assert 1 <= len(done["selected"]) <= 2


@pytest.mark.process("savant")
def test_templates_sanitize_and_queue_novel():
    from savant.templates import materialize, sanitize_overrides
    cfg, rejected = sanitize_overrides("blockshift_mutate", {"GUIDED": 2.0, "D_GRID": 100, "INIT": "/tmp/x"})
    assert cfg["GUIDED"] == 1.0 and cfg["D_GRID"] == 24 and "INIT" not in cfg
    assert any(r["key"] == "INIT" for r in rejected)
    h = {"hypothesis_id": "x", "role": "literature", "title": "t", "template": "queue_novel",
         "config_overrides": {}, "claim": "c", "mechanism": "m", "algorithmic_change": "a"}
    m = materialize({**h, "expected_gain": 1e-5, "expected_gain_lo": 0, "expected_gain_hi": 1,
                     "risk": 1, "assumptions": [], "falsification": "f", "compute_jobs": 0, "novelty": 1})
    assert m["executable"] is False and m["queued"] is True


@pytest.mark.process("savant")
def test_mock_council_and_job_plan(tmp_path, monkeypatch, record):
    from savant.config import SavantConfig
    from savant.loop import offline_smoke_council
    if not EXPORT.exists():
        pytest.skip("export missing")
    cfg = SavantConfig()
    cfg.runs_dir = tmp_path / "runs"
    cfg.seed_export = EXPORT
    cfg.llm.provider = "mock"
    out = offline_smoke_council(cfg)
    plan = out["step"]["plan"]
    assert plan.n_jobs <= cfg.budgets.max_jobs_per_iteration
    assert plan.afford or plan.n_jobs == 0 or plan.afford_reason
    selected = out["step"]["council"]["selected"]
    assert 1 <= len(selected) <= cfg.budgets.max_strategies_per_iteration
    assert all(h["role"] for h in out["step"]["council"]["hypotheses"])
    roles = {h["role"] for h in out["step"]["council"]["hypotheses"]}
    assert roles == {"mathematical", "numerical", "evolutionary", "statistical", "literature"}
    # distinct templates preferred
    mats = out["step"]["council"]["materialized"]
    assert any(m.get("executable") for m in mats) or any(m.get("queued") for m in mats)
    mem = out["session"]["memory"].read()
    assert any(e["kind"] == "manager_decision" for e in mem)
    record(n_jobs=plan.n_jobs, selected=[h["hypothesis_id"] for h in selected],
           llm_usd=out["usage"]["usd"])


@pytest.mark.process("savant")
def test_run_mock_loop_no_modal(tmp_path, record):
    from savant.config import SavantConfig
    from savant.loop import run_loop
    if not EXPORT.exists():
        pytest.skip("export missing")
    cfg = SavantConfig()
    cfg.runs_dir = tmp_path / "runs"
    cfg.seed_export = EXPORT
    out = run_loop(cfg, max_iterations=2, confirm_llm=False, confirm_modal=False, mock=True,
                   session_id="test-loop")
    assert len(out["history"]) >= 1
    assert out["history"][0]["execution"]["status"] == "planned_only"
    assert (out["session"]["dir"] / "iteration_decisions.jsonl").exists()
    decisions = [h["decision"] for h in out["history"] if "decision" in h]
    assert decisions and decisions[-1] in ("CONTINUE", "PIVOT", "MORE_REPLICATION", "STOP")
    record(n_it=len(out["history"]), decision=decisions[-1], stop_reason=out["stop_reason"])


@pytest.mark.process("savant")
def test_seed_export_is_configurable(tmp_path, monkeypatch):
    """SAVANT_SEED_EXPORT wins; otherwise the in-repo copy, then the sibling worktree."""
    from savant.config import SMOKE_EXPORT_REL, SavantConfig, resolve_seed_export, seed_export_candidates

    monkeypatch.delenv("SAVANT_SEED_EXPORT", raising=False)
    repo = tmp_path / "repo"
    in_repo = repo / SMOKE_EXPORT_REL
    sibling = repo.parent / "ws3-v2-runner" / SMOKE_EXPORT_REL
    assert seed_export_candidates(repo) == (in_repo, sibling)
    assert resolve_seed_export(repo) == in_repo

    sibling.mkdir(parents=True)
    assert resolve_seed_export(repo) == sibling

    in_repo.mkdir(parents=True)
    assert resolve_seed_export(repo) == in_repo

    chosen = tmp_path / "configured"
    monkeypatch.setenv("SAVANT_SEED_EXPORT", str(chosen))
    assert resolve_seed_export(repo) == chosen
    assert SavantConfig().seed_export == chosen
