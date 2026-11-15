"""Process 12: the LLM structural editor and its spend cap (no real API calls, no money spent).

A fake OpenAI client returns a fixed, valid optimizer edit with known token usage, so every cost is
known exactly: gpt-6.1-terra at $1 / $5 per 1M tokens -> 2,000 in + 3,000 out = $0.017 per call.
"""
import json
from types import SimpleNamespace

import pytest

from test_10_evolution import STUB, InProcess, evo  # noqa: F401  (evo is a fixture)

P = pytest.mark.process("12. LLM editor (OpenAI / Claude) and $ cap")

EDIT = STUB.replace('"HELP": 1', '"HELP": 2')          # a structural edit that really helps


class FakeOpenAI:
    """Mimics openai.OpenAI().chat.completions.create with fixed usage."""

    def __init__(self, source=EDIT, prompt_tokens=2000, completion_tokens=3000, cached=0, finish="stop"):
        self.calls, self.kwargs = 0, []
        self.source, self.pt, self.ct, self.cached, self.finish = source, prompt_tokens, completion_tokens, cached, finish
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        self.kwargs.append(kw)
        content = json.dumps({"hypothesis": f"HELP=2 halves the peak (call {self.calls})", "source": self.source})
        msg = SimpleNamespace(content=content, refusal=None)
        usage = SimpleNamespace(prompt_tokens=self.pt, completion_tokens=self.ct,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=self.cached))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=self.finish)], usage=usage)


def _context(EV, I, root):
    from evolution.genes import GenePool
    return dict(genome=[], pool=GenePool(), problem=I.get_problem("1a"),
                base_optimizer="optimizers/1a_stub.py", best_score=2.25, evidence=[])


@P
def test_cost_from_usage_and_unknown_model_refused(evo, record):
    I, EV, root = evo
    from evolution import llm
    c = llm.cost_usd("gpt-6.1-terra", 2000, 3000)
    cached = llm.cost_usd("gpt-6.1-terra", 2000, 3000, cached_input_tokens=1000)
    record(cost=c, with_cache=cached)
    assert c == pytest.approx(0.017) and cached == pytest.approx(0.017 - 1000 * (1.00 - 0.05) / 1e6)
    with pytest.raises(KeyError):
        llm.cost_usd("gpt-unknown", 1, 1)
    with pytest.raises(KeyError):
        llm.editor("openai", model="gpt-unknown", client=FakeOpenAI(), spend_file=root / "s.json")


@P
def test_budget_cutoff_never_exceeds_cap_and_persists(evo, record):
    """Calls stop BEFORE the cap could be crossed; the total survives a restart (new process = new ledger)."""
    I, EV, root = evo
    from evolution import llm
    spend = root / "llm_spend.json"
    fake = FakeOpenAI()
    ed = llm.editor("openai", budget_usd=0.10, client=fake, spend_file=spend,
                    policy=llm.EditorPolicy(stall_rounds=0, max_edits_per_round=100))
    genes = []
    for r in range(10):
        ed.new_round(r)
        genes.append(ed.propose(**_context(EV, I, root)))
    on_disk = json.loads(spend.read_text())
    record(calls=fake.calls, spent=ed.ledger.spent, on_disk=on_disk["total_usd"], skipped=ed.skipped)
    assert fake.calls == 3 and ed.skipped["budget"] == 7                  # 3 x $0.017; a 4th could cross $0.10
    assert ed.ledger.spent == pytest.approx(0.051) and ed.ledger.spent <= 0.10
    assert on_disk["total_usd"] == pytest.approx(0.051) and len(on_disk["calls"]) == 3
    assert all(g is not None and g.kind == "operator" for g in genes[:3]) and genes[3] is None
    # restart: a new editor reads the same ledger and is still blocked
    fake2 = FakeOpenAI()
    ed2 = llm.editor("openai", budget_usd=0.10, client=fake2, spend_file=spend,
                     policy=llm.EditorPolicy(stall_rounds=0))
    assert ed2.propose(**_context(EV, I, root)) is None and fake2.calls == 0
    # raising the budget explicitly lets it continue from the same total
    ed3 = llm.editor("openai", budget_usd=0.20, client=fake2, spend_file=spend, policy=llm.EditorPolicy(stall_rounds=0))
    assert ed3.propose(**_context(EV, I, root)) is not None and ed3.ledger.spent == pytest.approx(0.068)


@P
def test_stall_trigger_and_round_cap(evo, record):
    I, EV, root = evo
    from evolution import llm
    fake = FakeOpenAI()
    ed = llm.editor("openai", client=fake, spend_file=root / "s.json",
                    policy=llm.EditorPolicy(stall_rounds=2, max_edits_per_round=1))
    ctx = _context(EV, I, root)
    ed.new_round(1)
    assert ed.propose(**ctx, stalled_rounds=1) is None and fake.calls == 0      # not stalled long enough
    assert ed.propose(**ctx, stalled_rounds=2) is not None and fake.calls == 1
    assert ed.propose(**ctx, stalled_rounds=5) is None and fake.calls == 1      # round cap reached
    ed.new_round(2)
    assert ed.propose(**ctx, stalled_rounds=5) is not None and fake.calls == 2
    record(skipped=ed.skipped)
    assert ed.skipped == {"not stalled": 1, "round cap": 1}


@P
def test_compact_prompt_and_request_shape(evo, record):
    I, EV, root = evo
    from evolution import llm
    fake = FakeOpenAI()
    ed = llm.editor("openai", client=fake, spend_file=root / "s.json",
                    policy=llm.EditorPolicy(stall_rounds=0, max_tokens=8000))
    ed.propose(**_context(EV, I, root))
    kw = fake.kwargs[0]
    user = kw["messages"][1]["content"]
    record(model=kw["model"], max_completion_tokens=kw["max_completion_tokens"], prompt_chars=len(user))
    assert kw["model"] == "gpt-6.1-terra" and kw["max_completion_tokens"] == 8000
    assert kw["response_format"]["json_schema"]["strict"] is True
    assert "Current optimizer" in user and "Graveyard" in user and len(user) < 4000


@P
def test_invalid_or_truncated_edit_is_paid_but_not_proposed(evo, record):
    I, EV, root = evo
    from evolution import llm
    spend = root / "s.json"
    bad = llm.editor("openai", client=FakeOpenAI(source="def nope(:"), spend_file=spend,
                     policy=llm.EditorPolicy(stall_rounds=0))
    cut = llm.editor("openai", client=FakeOpenAI(finish="length"), spend_file=spend,
                     policy=llm.EditorPolicy(stall_rounds=0))
    assert bad.propose(**_context(EV, I, root)) is None
    assert cut.propose(**_context(EV, I, root)) is None
    calls = json.loads(spend.read_text())["calls"]
    record(errors=[c["error"] for c in calls])
    assert len(calls) == 2 and all(c["error"] and c["cost_usd"] > 0 for c in calls)


@P
def test_auto_without_keys_is_off(evo, monkeypatch, record):
    I, EV, root = evo
    from evolution import llm
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm.editor("auto", spend_file=root / "s.json") is None
    assert llm.editor("none") is None


@P
def test_evolution_logs_llm_cost_per_record(evo, record):
    """End to end: stalled islands trigger capped LLM edits; each record carries its call's cost;
    records + failed calls add up to the ledger, which stays under the cap."""
    I, EV, root = evo
    from certificates import CertificateStore
    from evolution import llm
    fake = FakeOpenAI(source=STUB.replace('"JUNK": 1.0', '"JUNK": 3.0'))   # a useless edit: never improves
    ed = llm.editor("openai", budget_usd=1.0, client=fake, spend_file=root / "spend.json",
                    policy=llm.EditorPolicy(stall_rounds=1, max_edits_per_round=1))
    store = I.JsonlStore(root / "e.jsonl")
    orch = EV.Orchestrator("1a", "optimizers/1a_stub.py", InProcess(), store, CertificateStore(root / "certs"),
                           islands=1, seeds=3, budget_s=0.2, candidates=6, seed=3, editor=ed)
    reports = orch.run(5)
    recs = store.read()
    paid = [r for r in recs if r["llm_cost_usd"]]
    record(llm_per_round=[rep.tools.get("llm") for rep in reports], calls=fake.calls,
           records_with_cost=len(paid), ledger=ed.ledger.spent)
    assert all(set(r) == set(I.RECORD_FIELDS) for r in recs)
    assert all(rep.tools["llm"]["calls"] <= 1 for rep in reports)                 # round cap
    assert fake.calls >= 1 and paid                                               # stall triggered real calls
    assert all(r["llm_usage"]["model"] == "gpt-6.1-terra" for r in paid)
    assert sum(r["llm_cost_usd"] for r in paid) <= ed.ledger.spent + 1e-12 <= 1.0
    assert ed.ledger.spent == pytest.approx(0.017 * fake.calls)
