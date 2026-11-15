"""Astra migration: request shape, token accounting, hard spending stops, resume. Offline (fake OpenAI)."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from savant.config import ASTRA_MAX_OUTPUT_TOKENS, BudgetCaps, LLMConfig, SavantConfig, astra_config
from savant.llm import BudgetExceeded, LLMCallFailed, LLMClient, request_cost
from savant.schemas import (CRITIQUE_BATCH_JSON_SCHEMA, MANAGER_JSON_SCHEMA, ROLES,
                            assert_openai_strict_compatible)
from savant.templates import default_hypotheses_for_mock

EXP1 = REPO / "runs" / "savant" / "savant-exp1"


class FakeResponses:
    def __init__(self, payload: dict, *, status="completed", inp=1000, cached=200, out=5000, reasoning=4000):
        self.calls = []
        self.payload, self.status = payload, status
        self.inp, self.cached, self.out, self.reasoning = inp, cached, out, reasoning

    polls_before_done = 0
    fail_polls = 0
    fail_after_accept = False
    retrieves = 0

    def _final(self):
        usage = SimpleNamespace(input_tokens=self.inp, output_tokens=self.out,
                                input_tokens_details=SimpleNamespace(cached_tokens=self.cached),
                                output_tokens_details=SimpleNamespace(reasoning_tokens=self.reasoning))
        return SimpleNamespace(id="resp_test", status=self.status, usage=usage, output=[],
                               output_text=json.dumps(self.payload))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.polls_before_done or self.fail_polls or self.fail_after_accept:
            return SimpleNamespace(id="resp_test", status="queued", usage=None, output=[], output_text="")
        return self._final()

    def retrieve(self, rid):
        self.retrieves += 1
        if self.fail_after_accept:
            raise ConnectionError("network down")
        if self.fail_polls:
            self.fail_polls -= 1
            raise ConnectionError("poll dropped")
        if self.polls_before_done:
            self.polls_before_done -= 1
            return SimpleNamespace(id=rid, status="in_progress", usage=None, output=[], output_text="")
        return self._final()

    def cancel(self, rid):
        return None


def astra_client(tmp: Path, fake: FakeResponses, *, max_usd=50.0) -> LLMClient:
    """Build a client without ``__init__`` so tests never load a real API key.

    ``last_meta`` lives on thread-local state, so ``_init_runtime`` has to run
    before that property is set. The rest follows ``LLMClient.__init__``.
    """
    client = LLMClient.__new__(LLMClient)
    client.cfg = astra_config(LLMConfig())
    client.max_usd, client.max_tokens = max_usd, 40_000_000
    from savant.llm import LLMUsage
    client.usage = LLMUsage()
    client.allow_mock = False
    client.ledger_path = tmp / "spend_ledger.json"
    client.prior = {"usd": 0.0, "tokens": 0, "entries": []}
    client._init_runtime()
    client.last_meta = None
    client._load_ledger()
    client._openai_client = lambda: SimpleNamespace(responses=fake)
    return client


def _hyp_payload(role="mathematical"):
    h = dict(next(x for x in default_hypotheses_for_mock() if x["role"] == role))
    h.setdefault("mechanism", "m")
    for k in ("expected_gain_lo", "expected_gain_hi", "risk", "novelty"):
        h.setdefault(k, 0.1)
    h.setdefault("assumptions", [])
    h.setdefault("compute_jobs", 10)
    h.setdefault("literature_refs", None)
    h.setdefault("notes", None)
    return h


def test_astra_request_shape_and_token_accounting(tmp_path):
    from savant import researchers as R
    fake = FakeResponses(_hyp_payload())
    client = astra_client(tmp_path, fake)
    h = R.propose("mathematical", {"measured_facts": {}}, [], client, evidence={"best_verified": None})
    kw = fake.calls[0]
    assert kw["model"] == "gpt-6-astra"
    assert kw["reasoning"] == {"effort": "max"}
    assert kw["max_output_tokens"] == ASTRA_MAX_OUTPUT_TOKENS
    assert kw["text"]["format"]["type"] == "json_schema" and kw["text"]["format"]["strict"] is True
    assert "temperature" not in kw and kw["store"] is False
    expected = (800 * 10 + 200 * 1 + 5000 * 50) / 1e6
    assert abs(client.usage.usd - expected) < 1e-12
    assert client.usage.reasoning_tokens == 4000 and client.usage.cached_input_tokens == 200
    assert h["llm"]["model"] == "gpt-6-astra" and h["llm"]["reasoning_effort"] == "max"
    led = json.loads((tmp_path / "spend_ledger.json").read_text())
    assert led["pending"] is None and abs(led["spent_usd"] - expected) < 1e-9
    assert led["by_model"]["gpt-6-astra"]["reasoning_tokens"] == 4000


def test_reservation_blocks_before_sending(tmp_path):
    fake = FakeResponses(_hyp_payload())
    client = astra_client(tmp_path, fake, max_usd=3.0)  # one max-effort reservation is ~ $3.2 + input
    with pytest.raises(BudgetExceeded):
        client.complete_json("s", "u", schema_name="hypothesis")
    assert fake.calls == []


def test_prior_spend_and_orphaned_reservation_are_counted(tmp_path):
    (tmp_path / "spend_ledger.json").write_text(json.dumps({
        "entries": [{"kind": "request", "model": "gpt-6-astra", "usd": 45.0, "input_tokens": 1, "output_tokens": 1}],
        "pending": {"model": "gpt-6-astra", "reserved_usd": 3.5, "est_input_tokens": 100,
                    "max_output_tokens": ASTRA_MAX_OUTPUT_TOKENS}}))
    fake = FakeResponses(_hyp_payload())
    client = astra_client(tmp_path, fake)
    assert abs(client.spent_usd() - 48.5) < 1e-9
    with pytest.raises(BudgetExceeded):
        client.complete_json("s", "u", schema_name="hypothesis")
    assert fake.calls == []


def test_incomplete_response_is_charged_and_not_retried(tmp_path):
    fake = FakeResponses(_hyp_payload(), status="incomplete", out=ASTRA_MAX_OUTPUT_TOKENS,
                         reasoning=ASTRA_MAX_OUTPUT_TOKENS)
    client = astra_client(tmp_path, fake)
    with pytest.raises(LLMCallFailed):
        client.complete_json("s", "u", schema_name="hypothesis")
    assert len(fake.calls) == 1 and client.usage.output_tokens == ASTRA_MAX_OUTPUT_TOKENS


def test_background_mode_survives_a_dropped_poll(tmp_path, monkeypatch):
    import savant.llm as L
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    fake = FakeResponses(_hyp_payload())
    fake.polls_before_done, fake.fail_polls = 2, 1
    client = astra_client(tmp_path, fake)
    out = client.complete_json("s", "u", schema_name="hypothesis")
    assert out["hypothesis_id"] and len(fake.calls) == 1  # one paid request, several free polls
    assert fake.calls[0]["background"] is True
    assert client.usage.calls == 1


def test_connection_loss_after_accept_books_full_reservation(tmp_path, monkeypatch):
    import savant.llm as L
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    fake = FakeResponses(_hyp_payload())
    fake.fail_after_accept = True
    client = astra_client(tmp_path, fake)
    client.cfg.request_timeout_s = 0.0  # deadline reached on the first loop
    with pytest.raises(L.LLMUnavailable):
        client.complete_json("s", "u", schema_name="hypothesis")
    assert client.usage.output_tokens == ASTRA_MAX_OUTPUT_TOKENS  # booked as if fully used
    assert len(fake.calls) == 1


def test_effort_by_stage_is_sent_and_recorded(tmp_path):
    fake = FakeResponses(_hyp_payload())
    client = astra_client(tmp_path, fake)
    for purpose in ("propose:mathematical", "critique:numerical", "revise:literature", "vote:statistical",
                    "manager", "stop:evolutionary"):
        client.complete_json("s", "u", schema_name="hypothesis", purpose=purpose)
    sent = [c["reasoning"]["effort"] for c in fake.calls]
    assert sent == ["max", "xhigh", "xhigh", "xhigh", "max", "max"]
    led = json.loads((tmp_path / "spend_ledger.json").read_text())
    assert [e["reasoning_effort"] for e in led["entries"]] == sent
    assert {e["model"] for e in led["entries"]} == {"gpt-6-astra"}


def test_restart_adopts_the_in_flight_response(tmp_path, monkeypatch):
    import savant.llm as L
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    reserved = 3.33
    (tmp_path / "spend_ledger.json").write_text(json.dumps({"entries": [], "pending": {
        "model": "gpt-6-astra", "reasoning_effort": "max", "schema": "hypothesis", "purpose": "propose:statistical",
        "est_input_tokens": 17000, "max_output_tokens": 64000, "reserved_usd": reserved,
        "response_id": "resp_inflight"}}))
    fake = FakeResponses(_hyp_payload(), inp=17000, cached=0, out=9000, reasoning=8000)
    fake.polls_before_done = 1
    client = astra_client(tmp_path, fake)
    assert client.spent_usd() == pytest.approx(reserved)  # orphan counted until resolved
    out = client.complete_json("s", "u", schema_name="hypothesis", purpose="propose:statistical")
    assert out["hypothesis_id"] and fake.calls == []  # no new paid request
    led = json.loads((tmp_path / "spend_ledger.json").read_text())
    assert [e["kind"] for e in led["entries"]] == ["request"]
    assert led["entries"][0]["response_id"] == "resp_test" or led["entries"][0]["response_id"]
    assert client.spent_usd() == pytest.approx(request_cost(client.cfg.prices(), input_tokens=17000,
                                                            output_tokens=9000))
    client.complete_json("s", "u", schema_name="hypothesis", purpose="propose:literature")
    assert len(fake.calls) == 1  # adoption is one-shot


def test_cost_formula_long_context():
    p = astra_config().prices()
    assert abs(request_cost(p, input_tokens=300_000, output_tokens=1000)
               - (300_000 * 20 + 1000 * 75) / 1e6) < 1e-12


def test_new_schemas_are_strict():
    assert assert_openai_strict_compatible(CRITIQUE_BATCH_JSON_SCHEMA) == []
    assert assert_openai_strict_compatible(MANAGER_JSON_SCHEMA) == []


def test_confirm_llm_refuses_mock_provider(tmp_path):
    from savant.loop import run_loop
    cfg = SavantConfig(runs_dir=tmp_path)
    cfg.llm.provider = "mock"
    with pytest.raises(SystemExit):
        run_loop(cfg, confirm_llm=True, mock=False, session_id="x")


def test_batched_critiques_five_requests():
    from savant.debate import run_council
    from savant.memory import ResearchMemory
    tmp = Path(tempfile.mkdtemp())
    cfg = LLMConfig(provider="mock", reasoning_effort="max", model="gpt-6-astra")
    client = LLMClient(cfg, allow_mock=True)
    mem = ResearchMemory(tmp / "m.jsonl")
    out = run_council({"measured_facts": {}}, mem, client, max_select=4, batch_critiques=True)
    n_crit = sum(1 for e in mem.read() if e["kind"] == "critique")
    assert n_crit == 20
    assert client.usage.calls == 5 + 5 + 5 + 5  # propose, batched critique, revise, vote
    assert out["selected"]


@pytest.mark.skipif(not (EXP1 / "research_memory.jsonl").exists(), reason="savant-exp1 not present")
def test_resume_audit_on_copy_of_savant_exp1(tmp_path):
    from savant.loop import (audit_mock_outputs, book_legacy_llm, iteration_state, open_session,
                             rebuild_modal_ledger, build_evidence)
    dst = tmp_path / "savant-exp1"
    shutil.copytree(EXP1, dst, ignore=shutil.ignore_patterns("volume", "candidates", "certificates", "logs"))
    (dst / "campaign.lock").unlink(missing_ok=True)
    cfg = SavantConfig(runs_dir=tmp_path)
    cfg.llm = astra_config(cfg.llm)
    session = open_session(cfg, "savant-exp1")
    n_before = len(session["memory"].read_all())
    audit = audit_mock_outputs(session)
    n_after = len(session["memory"].read_all())
    assert n_after == n_before + (1 if audit["invalidated_records"] else 0)  # append-only, nothing deleted
    assert audit_mock_outputs(session)["invalidated_records"] == 0  # idempotent
    assert not any(e.get("hypothesis_id") in {"H-math-symmetry", "H-num-dgrid"} for e in session["memory"].read())
    assert iteration_state(dst, 1) in {"executed", "decided"}  # a mock decision never counts as decided
    if (dst / "void_mock_iteration_02").exists():
        assert not any(json.loads(l).get("source") == "mock" for l in
                       (dst / "iteration_decisions.jsonl").read_text(encoding="utf-8").splitlines()
                       if l.strip() and json.loads(l).get("iteration") == 2)
    led = rebuild_modal_ledger(dst)
    assert led.jobs_used >= 30 and led.modal_usd_est > 0.2
    client = LLMClient(LLMConfig(provider="mock"), allow_mock=True, ledger_path=dst / "spend_ledger.json")
    book_legacy_llm(client, session["memory"])
    assert client.spent_usd() >= 0.29
    assert book_legacy_llm(client, session["memory"]) is None  # booked once
    ev = build_evidence(session, cfg)
    assert ev["best_verified"]["score"] > ev["baseline_score"]
    assert {c["hypothesis_id"] for c in ev["completed_strategies"]} >= {"math-hypo-20261115-001"}


def test_control_arm_and_300s_gate():
    from savant.loop import CONTROL_HYPOTHESIS
    from savant.planner import ResourceLedger, allocate_wave
    from savant.config import CAMPAIGN_SEEDS
    hyps = [dict(_hyp_payload(r), hypothesis_id=f"H{i}", template="blockshift_mutate",
                 config_overrides={"GUIDED": 0.1 * (i + 1)}) for i, r in enumerate(ROLES[:4])]
    caps = BudgetCaps()
    plan = allocate_wave(hyps, ResourceLedger(), caps, CAMPAIGN_SEEDS, iteration=2, control=CONTROL_HYPOTHESIS,
                         control_done_budgets=set(), budget_request=300, n_seeds=10)
    assert plan.strategies[0]["hypothesis_id"] == "control-blockshift-stock"
    assert len(plan.strategies) == 4 and plan.n_jobs == 40 and plan.budget_s == 60
    assert any("refused" in n for n in plan.notes)
    plan2 = allocate_wave(hyps, ResourceLedger(), caps, CAMPAIGN_SEEDS, iteration=3, control=CONTROL_HYPOTHESIS,
                          control_done_budgets={60.0}, prior_best_gains=[1e-8, 2e-8], budget_request=300, n_seeds=5)
    assert plan2.budget_s == 300 and len(plan2.seeds) == 5
    assert plan2.strategies[0]["hypothesis_id"] == "control-blockshift-stock"  # new budget level needs a control
