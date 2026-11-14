"""Shared research memory between agents, backed by past.dev.

    from integrations.pastdev_client import get_memory
    memory = get_memory()                     # past.dev if a key is set, else local JSONL
    fact_id = memory.write("Lesson: ...", author="loop-laptop", kind="lesson",
                           metadata={"problem": "1a", "config": {...}})
    facts = memory.read("lessons for problem 1a", reader="loop-modal")
    # -> [{fact_id, author, kind, metadata, content, memory_summary}]

PastDevMemory: https://api.past.dev/api/v1 (Authorization: Bearer past_sk_...).
    Keys: PAST_API_KEYS=name:key,... (failover). All keys must belong to the
    SAME past.dev project, otherwise agents would silently stop sharing one
    memory; a key from another project is refused.
LocalMemory: same interface in a local JSONL file, for runs without a key.

Memory scopes (fair comparisons): a fact written with scope="ab-1" carries that
tag; read(..., scope="ab-1") only returns facts with the same tag, and
read(..., since=ISO time) asks past.dev for facts written after that time
(occurredFrom), so old runs cannot leak into a new comparison. Without a scope,
agents read the accumulated memory of every run (the "learn from the past" mode).

Every write and read is logged to logs/pastdev_log.jsonl (author, reader, fact
ids, latency), which is how "agent B used a fact written by agent A" is shown.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._common import (REPO_ROOT, STATE_DIR, AdapterError, KeyPool, append_jsonl, http_json,
                      load_env, utc_now)

LOG_PATH = Path(os.environ.get("PASTDEV_LOG_PATH", REPO_ROOT / "logs" / "pastdev_log.jsonl"))

Fact = Dict[str, Any]


def _log(**event: Any) -> None:
    append_jsonl(LOG_PATH, {"timestamp": utc_now(), **event})


class Memory:
    name = "abstract"

    def write(self, content: str, author: str, kind: str = "note", label: str = "",
              metadata: Optional[Dict[str, Any]] = None, wait: bool = True,
              scope: Optional[str] = None) -> str:
        raise NotImplementedError

    def read(self, query: str, reader: str, limit: int = 10, scope: Optional[str] = None,
             since: Optional[str] = None) -> List[Fact]:
        raise NotImplementedError


def _meta(metadata, author, kind, scope):
    meta = dict(metadata or {}, author=author, kind=kind)
    if scope:
        meta["scope"] = scope
    return meta


def _in_scope(fact: Fact, scope: Optional[str]) -> bool:
    return scope is None or (fact.get("metadata") or {}).get("scope") == scope


class PastDevMemory(Memory):
    name = "past.dev"
    BASE = "https://api.past.dev/api/v1"

    def __init__(self, pool: KeyPool, wait_s: float = 90):
        self.pool = pool
        self.wait_s = wait_s
        self.project_id = None

    def _req(self, method: str, path: str, body=None):
        def attempt(label, key):
            resp = http_json(method, self.BASE + path, {"Authorization": "Bearer " + key}, body)
            pid = resp.get("projectId") if isinstance(resp, dict) else None
            if pid is not None:
                if self.project_id is None:
                    self.project_id = pid
                elif pid != self.project_id:
                    raise AdapterError("key belongs to project %s, expected %s"
                                       % (pid, self.project_id), 403)
            return resp
        return self.pool.call(attempt)[1]

    def write(self, content, author, kind="note", label="", metadata=None, wait=True, scope=None):
        fact_id = "%s-%s" % (kind, uuid.uuid4().hex[:10])
        meta = _meta(metadata, author, kind, scope)
        if scope:
            content = "[scope %s] %s" % (scope, content)
        t0 = time.perf_counter()
        resp = self._req("POST", "/ingest", {
            "id": fact_id, "content": content, "label": label or kind,
            "timestamp": utc_now(), "identity": author, "metadata": meta,
        })
        status = resp.get("status")
        if wait and status != "completed":
            status = self._wait(resp.get("ingestionId"))
        _log(tool="past.ingest", author=author, fact_id=fact_id, kind=kind, scope=scope, content=content[:300],
             status=status, project_id=self.project_id, latency_s=round(time.perf_counter() - t0, 3))
        return fact_id

    def _wait(self, ingestion_id: Optional[str]) -> str:
        """Ingest is asynchronous: poll until processed so other agents can recall it."""
        if not ingestion_id:
            return "unknown"
        deadline, status = time.time() + self.wait_s, "queued"
        while time.time() < deadline:
            status = self._req("GET", "/ingest/%s" % ingestion_id).get("status", status)
            if status in ("completed", "failed"):
                break
            time.sleep(2)
        return status

    def read(self, query, reader, limit=10, scope=None, since=None):
        t0 = time.perf_counter()
        body = {"query": ("[scope %s] " % scope if scope else "") + query, "identity": reader,
                "limit": min(200, limit * 4) if scope else limit, "level": "low"}
        if since:
            body["occurredFrom"] = since          # server-side: older facts never come back
        resp = self._req("POST", "/recall", body)
        facts, seen = [], set()
        for r in resp.get("results", []):
            for s in r.get("sources", []) or []:
                sid = s.get("sourceId")
                if sid in seen:
                    continue
                seen.add(sid)
                meta = s.get("metadata") or {}
                facts.append({"fact_id": sid, "author": meta.get("author"), "kind": meta.get("kind"),
                              "metadata": meta,
                              "content": " ".join(s.get("excerpts") or []) or r.get("content"),
                              "memory_summary": r.get("content")})
        facts = [f for f in facts if _in_scope(f, scope)][:limit]
        _log(tool="past.recall", reader=reader, query=query, scope=scope, since=since,
             fact_ids=[f["fact_id"] for f in facts],
             authors=sorted({str(f["author"]) for f in facts}),
             latency_s=round(time.perf_counter() - t0, 3))
        return facts


class LocalMemory(Memory):
    """Keyword-match fallback with the same interface (STATE_DIR/memory.jsonl)."""
    name = "local"

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or STATE_DIR / "memory.jsonl")

    def write(self, content, author, kind="note", label="", metadata=None, wait=True, scope=None):
        fact_id = "%s-%s" % (kind, uuid.uuid4().hex[:10])
        meta = _meta(metadata, author, kind, scope)
        append_jsonl(self.path, {"fact_id": fact_id, "content": content, "author": author,
                                 "kind": kind, "label": label, "metadata": meta, "ts": utc_now()})
        _log(tool="local.write", author=author, fact_id=fact_id, kind=kind, scope=scope, content=content[:300])
        return fact_id

    def read(self, query, reader, limit=10, scope=None, since=None):
        rows = []
        if self.path.exists():
            rows = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if _in_scope(r, scope) and (since is None or r["ts"] >= since)]
        words = {w for w in query.lower().split() if len(w) > 2}
        ranked = sorted(reversed(rows), key=lambda r: -sum(w in r["content"].lower() for w in words))
        facts = ranked[:limit]
        _log(tool="local.read", reader=reader, query=query, scope=scope, since=since,
             fact_ids=[f["fact_id"] for f in facts],
             authors=sorted({f["author"] for f in facts}))
        return facts


def get_memory() -> Memory:
    load_env()
    pool = KeyPool("past", "PAST_API_KEYS", "PAST_API_KEY")
    return PastDevMemory(pool) if len(pool) else LocalMemory()
