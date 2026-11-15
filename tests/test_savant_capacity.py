"""High-compute caps: a wave is 4 strategies x 10 seeds, not 6, and $50 is a hard stop."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from savant.config import CAMPAIGN_SEEDS, BudgetCaps, estimate_modal_usd
from savant.planner import ResourceLedger, allocate_wave, batch_spec, build_job_plan
from ws3_bench.modal_app import MAX_CONTAINERS


def _hyp(i: int, guided: float) -> dict:
    return {
        "hypothesis_id": f"H{i}", "role": "numerical", "template": "blockshift_mutate",
        "title": f"t{i}", "config_overrides": {"GUIDED": guided, "D_GRID": 12 + i},
        "claim": "c", "expected_gain": 1e-8, "falsification": "f",
    }


def test_wave_is_forty_concurrent_jobs_not_six():
    caps = BudgetCaps()
    assert caps.max_llm_usd == 50 and caps.max_modal_usd == 50
    assert caps.max_jobs_per_iteration == 40 and caps.max_jobs_total == 400
    assert caps.seeds_per_strategy == 10 and len(CAMPAIGN_SEEDS) == 10
    assert caps.max_search_seconds_total == 400 * 300
    plan = build_job_plan([_hyp(i, 0.2 * i) for i in range(1, 5)], ResourceLedger(), caps,
                          iteration=1, seeds=CAMPAIGN_SEEDS)
    assert plan.n_jobs == 40
    assert len(plan.strategies) == 4 and len(plan.seeds) == 10
    assert plan.budget_s == 60 and plan.afford
    spec = batch_spec(plan)
    assert spec["n_calls"] == 1 and spec["n_jobs"] == 40 and spec["concurrent"]
    assert MAX_CONTAINERS >= plan.n_jobs


def test_modal_dollar_cap_blocks_the_wave():
    caps = BudgetCaps()
    ledger = ResourceLedger(modal_usd_est=49.99)
    plan = build_job_plan([_hyp(1, 0.5)], ledger, caps, iteration=2, seeds=CAMPAIGN_SEEDS)
    assert plan.n_jobs == 0 or not plan.afford
    assert "modal usd" in plan.afford_reason or plan.n_jobs == 0
    # a fresh ledger stays under the cap
    assert estimate_modal_usd(40, 60, caps) < 50
    assert estimate_modal_usd(400, 300, caps) < 50


def test_unproductive_strategy_is_not_scheduled_again():
    from savant.templates import materialize
    from savant.planner import strategy_signature
    caps = BudgetCaps()
    hyp = _hyp(1, 0.5)
    sig = strategy_signature(materialize(hyp))
    plan = allocate_wave([hyp], ResourceLedger(), caps, CAMPAIGN_SEEDS, iteration=2,
                         completed={sig: {"runs": 1, "best_gain": 0.0, "first_gain": 0.0}})
    assert plan.n_jobs == 0
    assert any("unproductive" in n for n in plan.notes)


def test_resume_does_not_repeat_a_stored_council():
    import tempfile
    from savant.debate import run_council
    from savant.memory import ResearchMemory
    from savant.schemas import ROLES

    class Boom:
        def complete_json(self, *a, **k):
            raise AssertionError("paid call")

    tmp = Path(tempfile.mkdtemp())
    memory = ResearchMemory(tmp / "research_memory.jsonl")
    ids = []
    for i, role in enumerate(ROLES):
        hid = f"H-{role}"
        ids.append(hid)
        memory.append("hypothesis", {
            "stage": "revised", "hypothesis_id": hid, "role": role, "template": "blockshift_mutate",
            "title": role, "config_overrides": {"GUIDED": 0.1 * (i + 1), "D_GRID": 10 + i},
            "claim": "c", "expected_gain": 1e-8, "falsification": "f",
        }, source=f"savant:{role}")
        memory.append("note", {"kind_detail": "revision", "hypothesis_id": hid, "role": role},
                      source=f"savant:{role}")
    for reviewer in ROLES:
        for hid, role in zip(ids, ROLES):
            if reviewer == role:
                continue
            memory.append("critique", {
                "reviewer_role": reviewer, "target_id": hid, "score": 3, "text": "ok",
            }, source=f"savant:{reviewer}")
    for role in ROLES:
        memory.append("vote", {"voter_role": role, "ranking": list(ids)}, source=f"savant:{role}")
    out = run_council({}, memory, Boom(), max_select=4)
    assert len(out["selected"]) == 4
    assert all(h["hypothesis_id"].startswith("H-") for h in out["selected"])


def test_budget_expands_only_after_a_verified_increase():
    caps = BudgetCaps()
    plan = allocate_wave([_hyp(1, 0.4), _hyp(2, 0.8)], ResourceLedger(), caps, CAMPAIGN_SEEDS,
                         iteration=3, prior_best_gains=[1e-9, 2e-9])
    assert plan.budget_s == 300 and plan.n_jobs == 20
    first = allocate_wave([_hyp(1, 0.4)], ResourceLedger(), caps, CAMPAIGN_SEEDS,
                          iteration=1, prior_best_gains=[])
    assert first.budget_s == 60
