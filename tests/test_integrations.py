"""Offline tests for the WS4 integrations: no network, no real keys.

    python -m pytest tests/test_integrations.py      or      python tests/test_integrations.py

A fake local HTTP server stands in for Exa and past.dev.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import types
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
_TMP = tempfile.mkdtemp()
os.environ.update(INTEGRATIONS_ENV_FILE=os.path.join(_TMP, "no.env"),   # never read the real .env
                  INTEGRATIONS_STATE_DIR=os.path.join(_TMP, "state"),
                  EXA_LOG_PATH=os.path.join(_TMP, "exa_log.jsonl"),
                  PASTDEV_LOG_PATH=os.path.join(_TMP, "pastdev_log.jsonl"))
for _v in ("EXA_API_KEYS", "EXA_API_KEY", "PAST_API_KEYS", "PAST_API_KEY", "EXA_BUDGET_USD", "EXA_MAX_CALLS"):
    os.environ.pop(_v, None)

try:                                   # real WS2 contract when present, else a minimal stand-in
    import interfaces  # noqa: F401
except Exception:
    stub = types.ModuleType("interfaces")

    @dataclass
    class Evidence:
        source: str
        content: str
        citation: str = ""
        query: str = ""
        author: str = ""
        retrieved_at: str = ""
        latency_s: float = 0.0

    class JsonlStore:
        def __init__(self, path):
            self.path = path

        def append(self, r):
            with open(self.path, "a") as f:
                f.write(json.dumps(r) + "\n")

        def read(self):
            return [json.loads(l) for l in open(self.path)] if os.path.exists(self.path) else []

    stub.Evidence, stub.JsonlStore = Evidence, JsonlStore
    sys.modules["interfaces"] = stub

from integrations import _common, evidence, exa_client, pastdev_client  # noqa: E402

KEYS = {"dead": None, "ok1": 1, "ok2": 1, "other": 2}    # key -> past.dev project (None = no credit)
CALLS = {"search": 0}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        key = self.headers.get("x-api-key") or self.headers.get("Authorization", "").replace("Bearer ", "")
        if KEYS.get(key, "missing") in (None, "missing"):
            return self._send(402, {"error": "insufficient credits"})
        if self.path.endswith("/search"):
            CALLS["search"] += 1
            return self._send(200, {"costDollars": {"total": 0.009}, "results": [
                {"title": "Record paper", "url": "https://example.org/r", "publishedDate": "2026-01-01",
                 "highlights": ["new bound 1.5027"]}]})
        if self.path.endswith("/ingest"):
            Fake.facts.append(body)
            return self._send(202, {"projectId": KEYS[key], "status": "queued", "ingestionId": "i1",
                                    "sourceId": body.get("id")})
        if self.path.endswith("/recall"):
            Fake.last_recall = body
            return self._send(200, {"results": [{"content": f["content"], "sources": [
                {"sourceId": f["id"], "excerpts": [f["content"]], "metadata": f.get("metadata")}]}
                for f in Fake.facts]})
        self._send(404, {})

    def do_GET(self):
        self._send(200, {"status": "completed", "projectId": 1})


Fake.facts = []
Fake.last_recall = {}
_srv = HTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
BASE = "http://127.0.0.1:%d" % _srv.server_port
exa_client.URL = BASE + "/search"
pastdev_client.PastDevMemory.BASE = BASE


class P:                                   # minimal stand-in for interfaces.Problem
    id, name = "1a", "Sidon set autocorrelation constant"


def _reset_state():
    import shutil
    shutil.rmtree(os.environ["INTEGRATIONS_STATE_DIR"], ignore_errors=True)


def test_keypool_labels_and_empty_template():
    os.environ["EXA_API_KEYS"] = "murilo:dead, arthur:ok1, marie:"
    pool = _common.KeyPool("exa", "EXA_API_KEYS", "EXA_API_KEY")
    assert [l for l, _ in pool.keys] == ["murilo", "arthur"]


def test_exa_failover_spend_cache_and_wire_format():
    _reset_state()
    os.environ["EXA_API_KEYS"] = "murilo:dead,arthur:ok1"
    hits = exa_client.search("q1", since="2025-01-01", n=3, decision="test")
    assert hits == [{"url": "https://example.org/r", "title": "Record paper",
                     "published_date": "2026-01-01", "highlights": ["new bound 1.5027"]}]
    s = exa_client.spend()
    assert s["by_key"] == {"arthur": 0.009} and s["calls"] == 1
    before = CALLS["search"]
    exa_client.search("q1", since="2025-01-01", n=3, decision="test")       # cached: free
    assert CALLS["search"] == before and exa_client.spend()["calls"] == 1


def test_exa_stop_loss_and_missing_key():
    _reset_state()
    os.environ["EXA_API_KEYS"] = "arthur:ok1"
    os.environ["EXA_MAX_CALLS"] = "0"
    try:
        exa_client.search("q-blocked")
        raise AssertionError("expected stop loss")
    except exa_client.BudgetExceeded:
        pass
    finally:
        os.environ.pop("EXA_MAX_CALLS")
    os.environ.pop("EXA_API_KEYS")
    try:
        exa_client.search("q-nokey")
        raise AssertionError("expected ExaUnavailable")
    except exa_client.ExaUnavailable:
        pass


def test_pastdev_failover_and_project_guard():
    _reset_state()
    os.environ["PAST_API_KEYS"] = "k1:dead,k2:ok1"
    m = pastdev_client.get_memory()
    m.write("fact", author="scout")
    assert m.project_id == 1
    os.environ["PAST_API_KEYS"] = "p1:ok1,p2:other"
    m = pastdev_client.get_memory()
    m.write("fact", author="scout")
    m.pool.mark_dead("p1", "simulated")
    try:
        m.write("fact 2", author="scout")
        raise AssertionError("expected refusal of a key from another project")
    except _common.NoUsableKey as e:
        assert "expected 1" in str(e)
    os.environ.pop("PAST_API_KEYS")


def test_mirror_store_and_cross_agent_hint():
    _reset_state()
    Fake.facts.clear()
    os.environ["PAST_API_KEYS"] = "murilo:ok1"
    mem = pastdev_client.get_memory()
    store = evidence.PastDevMirrorStore(os.path.join(_TMP, "exp.jsonl"), mem, author="laptop-a")
    fields = getattr(sys.modules["interfaces"], "RECORD_FIELDS", ())
    record = {k: None for k in fields}
    record.update({"experiment_id": "e1", "problem": "1a", "status": "keep", "score": 1.51,
                   "best_before": 1.52, "hypothesis": "h", "agent": "laptop-a", "evidence": [],
                   "config": {"LR": 0.02, "INIT": "/tmp/x"}})
    store.append(record)
    assert store.read()[0]["experiment_id"] == "e1"
    ev_b = evidence.PastDevEvidence(mem, reader="laptop-b").gather(P, [])
    assert ev_b and ev_b[0].author == "laptop-a"
    assert '{"config": {"LR": 0.02}}' in ev_b[0].content          # actionable, INIT stripped
    ev_a = evidence.PastDevEvidence(mem, reader="laptop-a").gather(P, [])
    assert '"config"' not in ev_a[0].content                       # own facts are not re-suggested
    os.environ.pop("PAST_API_KEYS")


def test_exa_evidence_triggers_only_on_audit_and_plateau():
    _reset_state()
    os.environ["EXA_API_KEYS"] = "arthur:ok1"
    src = evidence.ExaEvidence(memory=pastdev_client.LocalMemory(), plateau=2)
    assert src.gather(P, []) and src.gather(P, []) == []                     # audit once
    hist = [{"problem": "1a", "status": "keep"}, {"problem": "1a", "status": "discard"},
            {"problem": "1a", "status": "discard"}]
    ev = src.gather(P, hist)
    assert ev and ev[0].source == "exa" and ev[0].citation == "https://example.org/r"
    assert src.gather(P, hist) == []                                         # once per plateau
    os.environ.pop("EXA_API_KEYS")


def test_memory_scope_isolates_comparison_runs():
    _reset_state()
    # local memory: scope tag + since filter
    mem = pastdev_client.LocalMemory()
    mem.write("old run: LR 0.5 helped", author="old", metadata={"problem": "1a"})
    mem.write("run A1: LR 0.02 helped", author="a", metadata={"problem": "1a"}, scope="ab-1")
    mem.write("run B1: LR 0.9 helped", author="b", metadata={"problem": "1a"}, scope="ab-2")
    got = mem.read("LR helped", reader="x", scope="ab-1")
    assert [f["author"] for f in got] == ["a"], got
    assert len(mem.read("LR helped", reader="x")) == 3                 # no scope: accumulated memory
    assert mem.read("LR helped", reader="x", since="2999-01-01T00:00:00Z") == []
    # past.dev: scope tag filtered client-side, since sent as occurredFrom
    Fake.facts.clear()
    os.environ["PAST_API_KEYS"] = "murilo:ok1"
    pm = pastdev_client.get_memory()
    pm.write("A fact", author="a", metadata={"problem": "1a"}, scope="ab-1", wait=False)
    pm.write("B fact", author="b", metadata={"problem": "1a"}, scope="ab-2", wait=False)
    got = pm.read("fact", reader="x", scope="ab-1", since="2026-11-14T00:00:00Z")
    assert [f["author"] for f in got] == ["a"], got
    assert Fake.last_recall.get("occurredFrom") == "2026-11-14T00:00:00Z"
    ev = evidence.PastDevEvidence(pm, reader="x", scope="ab-2").gather(P, [])
    assert [e.author for e in ev] == ["b"]
    os.environ.pop("PAST_API_KEYS")


def test_tagged_store_records_switches():
    path = os.path.join(_TMP, "tagged.jsonl")
    inner = sys.modules["interfaces"].JsonlStore(path)
    store = evidence.TaggedStore(inner, {"exa": False, "pastdev": True, "memory_scope": "ab-1"})
    fields = getattr(sys.modules["interfaces"], "RECORD_FIELDS", ())
    store.append(dict({k: None for k in fields}, experiment_id="t1", problem="1a"))
    row = store.read()[-1]
    assert row["integrations"] == {"exa": False, "pastdev": True, "memory_scope": "ab-1"}


def test_no_keys_means_local_fallbacks_and_no_secret_in_logs():
    for v in ("EXA_API_KEYS", "PAST_API_KEYS"):
        os.environ.pop(v, None)
    assert pastdev_client.get_memory().name == "local"
    blob = ""
    for root, _, files in os.walk(_TMP):
        for f in files:
            if f.endswith((".json", ".jsonl")):
                blob += open(os.path.join(root, f)).read()
    for secret in ("ok1", "ok2", "other"):
        assert '"%s"' % secret not in blob and ":%s" % secret not in blob, secret


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print("all %d integration tests passed" % len(tests))
