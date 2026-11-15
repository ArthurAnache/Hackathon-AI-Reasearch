"""Shared memory of the Knockout loop (knockout/pastdev.py, knockout/memory.py). Offline: LocalStore, a fake HTTP
layer for PastDevStore, and recorded toy runs with the fake LLM of tests/test_knockout_m1_m5.py."""
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from knockout.config import Strategy
from knockout.ledger import Ledger, validate_run
from knockout.memory import HEADER, RunMemory, jaccard
from knockout.pastdev import LocalStore, PastDevError, PastDevStore

ROOT = Path(__file__).resolve().parent.parent
SLUG = "toy-circles-26"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Tests never reach past.dev, even when PAST_API_KEY is set in .env: the store is always the local one."""
    monkeypatch.setattr("knockout.pastdev.api_key", lambda: "")


def _ctx(tmp_path, run_id="runA", slug=SLUG, **mem):
    st = Strategy({"memory": {"enabled": True, "recall_k": 8, "max_chars": 2500, "timeout_s": 1,
                              "write_failed_batches": True, "agent_name": "tester", **mem}})
    return SimpleNamespace(strategy=st, problem=SimpleNamespace(id=slug, gain=lambda a, b: a - b), run_id=run_id,
                           ledger=Ledger(tmp_path / run_id, run_id), source_commit="test", init_score=None)


def _events(ctx, type_):
    return [e["payload"] for e in ctx.ledger.events() if e["type"] == type_]


# ---------------------------------------------------------------------------------------------- stores
def test_local_store_roundtrip_and_idempotent_ingest(tmp_path):
    s = LocalStore(tmp_path / "m.jsonl", log_path=tmp_path / "log.jsonl")
    for _ in range(2):
        s.ingest("f1", "[refuted] packet weights do not help", "refuted", "2026-11-15T10:00:00Z", "main@r1",
                 {"problem": SLUG, "kind": "refuted"})
    assert len((tmp_path / "m.jsonl").read_text().splitlines()) == 1
    facts = s.recall("packet weights", "main@r2", 5)
    assert facts[0]["fact_id"] == "f1" and facts[0]["author"] == "main@r1"
    assert s.recall("packet", "x", 5, occurred_from="2026-11-16T00:00:00Z") == []
    log = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert log[-2]["op"] == "recall" and log[-2]["authors"] == ["main@r1"] and log[-1]["fact_ids"] == []


class _FakeHTTP:
    def __init__(self, answers):
        self.answers, self.requests = list(answers), []

    def __call__(self, req, timeout):
        self.requests.append(req)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return io.BytesIO(json.dumps(a).encode())


def test_pastdev_store_calls_and_parsing_without_leaking_the_key(tmp_path):
    http = _FakeHTTP([{"ingestionId": "ing1", "status": "queued"},
                      {"results": [{"content": "summary", "sources": [
                          {"sourceId": "f1", "metadata": {"problem": SLUG, "kind": "win", "author": "main@r1"},
                           "excerpts": ["[win] a"]},
                          {"sourceId": "f1", "metadata": {}, "excerpts": ["dup"]}]}]}])
    s = PastDevStore("past_sk_SECRET", opener=http, log_path=tmp_path / "log.jsonl")
    assert s.ingest("f1", "c", "win", "t", "main@r1", {"kind": "win"}) == "ing1"
    facts = s.recall("q", "main@r2", 8, occurred_from="2026-11-15T00:00:00Z")
    assert [f["fact_id"] for f in facts] == ["f1"] and facts[0]["author"] == "main@r1"
    first, second = http.requests
    assert first.full_url.endswith("/ingest") and first.get_header("Authorization") == "Bearer past_sk_SECRET"
    assert json.loads(second.data)["occurredFrom"] == "2026-11-15T00:00:00Z"
    assert "past_sk_SECRET" not in (tmp_path / "log.jsonl").read_text() and "SECRET" not in repr(s)


def test_pastdev_store_turns_http_errors_into_pastdev_error():
    s = PastDevStore("k", opener=_FakeHTTP([urllib.error.URLError("down")]))
    with pytest.raises(PastDevError):
        s.recall("q", "x", 3)


# ---------------------------------------------------------------------------------------------- RunMemory
def test_disabled_memory_changes_nothing(tmp_path):
    from knockout.llm_editor import memory_section
    ctx = _ctx(tmp_path, enabled=False)
    ctx.memory = RunMemory(ctx)
    assert ctx.memory.store is None and ctx.memory.recall_block("main", 0, "x") == ""
    assert memory_section(ctx, "main", 0, "x", 24) == [] and memory_section(SimpleNamespace(), "main", 0, "", 1) == []
    assert ctx.memory.write_fact("win", "x", arm="main") is None


def test_written_facts_are_recalled_by_the_next_run_and_other_problems_never_shown(tmp_path):
    a = _ctx(tmp_path, "runA")
    ma = RunMemory(a)
    ma.write_fact("refuted", "Packet-weight reparametrization", "0 wins / 3 losses", arm="main",
                  experiment_ids=["exp_000001"], gain=-3.5e-10)
    ma.write_fact("win", "refine: keep full radii", "beat parent", arm="control", gain=2e-6)
    ma.close()
    other = _ctx(tmp_path, "runO", slug="another-problem")
    mo = RunMemory(other)
    mo.write_fact("refuted", "Packet idea on another problem", arm="main")
    mo.close()
    assert {e["kind"] for e in _events(a, "memory_write")} >= {"refuted", "win", "lesson"}

    b = _ctx(tmp_path, "runB")
    mb = RunMemory(b)
    block = mb.recall_block("main", 0, "refine packet weights")
    assert block.startswith(HEADER)
    refuted_part = block.split("Already tried and refuted", 1)[1].split("Past wins", 1)[0]
    assert "Packet-weight reparametrization" in refuted_part
    assert "another problem" not in block
    rec = _events(b, "memory_recall")[0]
    assert rec["runs"] and "runA" in rec["runs"] and "runO" not in rec["runs"]
    assert mb.stats["foreign_facts"] >= 2 and mb.stats["prompts_with_foreign"] == 1
    assert mb.recall_block("main", 0, "something else") == block               # cached per batch
    assert len(_events(b, "memory_recall")) == 1


def test_scope_suffix_and_since_isolate_an_ab_memory(tmp_path):
    a = _ctx(tmp_path, "runA")
    m = RunMemory(a)
    m.write_fact("refuted", "Old idea", arm="main")
    m.close()
    iso = RunMemory(_ctx(tmp_path, "runB", scope_suffix="ab-1"))
    assert iso.recall_block("main", 0, "old idea") == ""
    late = RunMemory(_ctx(tmp_path, "runC", since="2999-01-01T00:00:00Z"))
    assert late.recall_block("main", 0, "old idea") == ""


def test_current_run_facts_already_in_history_are_dropped(tmp_path):
    a = _ctx(tmp_path, "runA")
    m = RunMemory(a)
    m.write_fact("failed_batch", "Batch 0: 2 edits did not beat their parent", arm="main",
                 experiment_ids=["exp_000002", "exp_000003"])
    m._q.join()
    assert m.recall_block("main", 1, "batch", visible_ids={"exp_000002", "exp_000003"}) == ""
    assert "Batch 0" in m.recall_block("main", 2, "batch", visible_ids={"exp_000002"})


def test_block_is_capped(tmp_path):
    a = _ctx(tmp_path, "runA", max_chars=400, recall_k=50)
    m = RunMemory(a)
    for i in range(40):
        m.write_fact("refuted", f"idea number {i} with a fairly long description of what was tried", arm="main")
    m.close()
    block = RunMemory(_ctx(tmp_path, "runB", max_chars=400, recall_k=50)).recall_block("main", 0, "idea")
    assert 0 < len(block) <= 400


def test_many_small_wins_do_not_crowd_out_refuted_ideas_or_failed_edits(tmp_path):
    a = _ctx(tmp_path, "runA", recall_k=8)
    m = RunMemory(a)
    for i in range(30):
        m.write_fact("win", f"refine variant {i}: a tiny polish gain", "x" * 400, arm="main", gain=1e-11 * (i + 1))
    for i in range(3):
        m.write_fact("refuted", f"refuted idea {i} about epigraph constraints", arm="main")
        m.write_fact("failed_batch", f"Batch {i}: 6 edits did not beat their parent", arm="main")
    m.close()
    block = RunMemory(_ctx(tmp_path, "runB", recall_k=8)).recall_block("main", 0, "refine polish")
    tried = block.split("Already tried and refuted", 1)[1].split("Past wins", 1)[0]
    assert tried.count("refuted idea") == 3 and tried.count("did not beat their parent") == 3
    wins = block.split("Past wins on this problem:", 1)[1] if "Past wins" in block else ""
    assert wins.count("refine variant") <= 3
    assert all(len(x) <= 200 for x in block.splitlines()[1:]) and len(block) <= 2500


class _NoSinceStore:
    """Answers like past.dev did on 2026-11-15: HTTP 400 whenever occurredFrom is sent."""
    name = "fake-pastdev"

    def __init__(self, facts):
        self.facts, self.calls = facts, []

    def ingest(self, *a, **k):
        return "ok"

    def recall(self, query, identity, limit, occurred_from=None):
        self.calls.append(occurred_from)
        if occurred_from:
            raise PastDevError("HTTP 400 on POST /recall")
        return self.facts[:limit]


def test_since_falls_back_to_a_local_date_filter(tmp_path):
    def fact(i, ts):
        return {"fact_id": f"f{i}", "author": "main@old", "kind": "refuted", "content": f"idea {i}",
                "metadata": {"problem": SLUG, "scope": SLUG, "kind": "refuted", "run_id": f"r{i}",
                             "headline": f"idea {i}", "timestamp": ts}}
    store = _NoSinceStore([fact(1, "2026-11-15T10:00:00Z"), fact(2, "2026-11-15T12:00:00Z"),
                           {**fact(3, ""), "metadata": {**fact(3, "")["metadata"], "timestamp": None}}])
    a = _ctx(tmp_path, "runS", since="2026-11-15T11:00:00Z")
    m = RunMemory(a, store=store)
    block = m.recall_block("main", 0, "idea")
    assert "idea 2" in block and "idea 1" not in block and "idea 3" not in block
    assert None in store.calls and "2026-11-15T11:00:00Z" in store.calls
    errs = _events(a, "memory_error")
    assert errs and all(e["op"] == "recall_since" for e in errs)          # the 400 stays visible


class _BrokenStore:
    name = "broken"

    def ingest(self, *a, **k):
        raise PastDevError("down")

    def recall(self, *a, **k):
        raise PastDevError("down")


def test_store_failures_never_stop_the_run(tmp_path):
    a = _ctx(tmp_path, "runA")
    m = RunMemory(a, store=_BrokenStore())
    assert m.recall_block("main", 0, "x") == ""
    m.write_fact("win", "x y z", arm="main")
    m.close()
    errs = _events(a, "memory_error")
    assert {e["op"] for e in errs} >= {"recall", "ingest"} and m.stats["errors"] >= 2
    assert not _events(a, "memory_write")


def test_repeat_rate_helper():
    assert jaccard("reparametrize packet weights by softmax", "packet weights reparametrize by a softmax") >= 0.6
    assert jaccard("shift runs left", "spectral damping of high frequencies") < 0.2


def test_branch_touches_nothing_under_integrations_or_problems():
    out = subprocess.run(["git", "diff", "--name-only", "60e3856"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("git history not available")
    assert not [p for p in out.stdout.split() if p.startswith(("integrations/", "problems/"))]


# ---------------------------------------------------------------------------------------------- recorded toy runs
def _fake_client_class():
    spec = importlib.util.spec_from_file_location("m1m5", ROOT / "tests" / "test_knockout_m1_m5.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._FakeLLMClient


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    import knockout.meta
    import knockout.run as run_module
    from knockout.llm_editor import OpenAIEditor
    from knockout.run import run

    tmp = tmp_path_factory.mktemp("mem")
    client = _fake_client_class()()
    prompts: list[str] = []
    real_create = client.create

    def create(**kwargs):                     # record every prompt the "LLM" receives
        prompts.append(kwargs["input"])
        return real_create(**kwargs)

    client.create = create
    real_scout, knockout.meta.scout = knockout.meta.scout, lambda arm, reason: 0
    overrides = {"run.wall_clock_s": 22, "run.max_usd": 100.0, "fidelity.f0_s": 1.0, "fidelity.f1_s": 2.0,
                 "fidelity.levels": 2, "tuner.calls": 1, "tuner.top_k": 1, "executor.workers": 6,
                 "control.edits_per_round": 2, "main.batch_size": 4, "gate.pairs_per_batch": 1,
                 "editor.kind": "openai", "selection.eps_rel": 1e-6, "islands.max_islands": 2,
                 "memory.enabled": True, "memory.local_path": str(tmp / "shared_memory.jsonl")}

    class Editor(OpenAIEditor):
        def __init__(self, ctx):
            super().__init__(ctx, client=client)

    import knockout.pastdev as pastdev
    real_key, pastdev.api_key = pastdev.api_key, (lambda: "")     # module-scoped: the autouse patch is not active yet
    real_make, run_module.make_editor = run_module.make_editor, (lambda ctx: Editor(ctx))
    try:
        s1 = run("toy", runs_root=tmp, run_id="run1", overrides=overrides, on_round=lambda *_: None)
        split = len(prompts)
        s2 = run("toy", runs_root=tmp, run_id="run2", overrides=overrides, on_round=lambda *_: None)
    finally:
        run_module.make_editor, knockout.meta.scout, pastdev.api_key = real_make, real_scout, real_key
    return tmp, s1, s2, (prompts[:split], prompts[split:])


def test_tests_use_the_local_store(two_runs):
    tmp, s1, s2, _ = two_runs
    assert s1["memory"]["store"] == s2["memory"]["store"] == "local"


def test_run2_recalls_facts_written_by_run1_and_replays(two_runs):
    from knockout.replay import replay
    tmp, s1, s2, _ = two_runs
    ev2 = [json.loads(x) for x in (tmp / "run2" / "events.jsonl").read_text().splitlines()]
    recalls = [e["payload"] for e in ev2 if e["type"] == "memory_recall"]
    assert recalls and any("run1" in r["runs"] for r in recalls)
    assert s1["memory"]["writes"] > 0 and s2["memory"]["foreign_facts"] > 0
    for d in ("run1", "run2"):
        validate_run(tmp / d)
        replay(tmp / d)
    log = (tmp / "run2" / "pastdev_log.jsonl").read_text()
    assert "main@run1" in log or "control@run1" in log or "run@run1" in log


def test_prompts_of_run2_carry_the_memory_section(two_runs):
    _, _, _, (p1, p2) = two_runs
    with_mem = [p for p in p2 if HEADER in p]
    assert with_mem, "no prompt of run 2 contains the memory section"
    assert any("run run1" in p for p in with_mem)                       # a fact written by run 1
    assert all(p.index(HEADER) > p.index("# Evidence from this run") for p in with_mem)



def test_report_has_a_memory_section(two_runs):
    from knockout.report import build
    tmp, _, _, _ = two_runs
    text = build(tmp / "run2")
    assert "## Memory (past.dev)" in text and "Prompts that contained a fact from another run" in text
