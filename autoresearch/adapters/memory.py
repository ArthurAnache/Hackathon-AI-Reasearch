"""Shared research memory between agents.

  MemoryAdapter.write(content, author, kind, label, metadata) -> fact_id
  MemoryAdapter.read(query, reader, limit) -> [ {fact_id, content, author, ...} ]

PastDevMemory - https://api.past.dev/api/v1  (Authorization: Bearer past_sk_...)
LocalMemory   - JSONL fallback with the same interface (no credentials needed).

Every write and read is logged to runs/provenance.jsonl, so the demo can show
"agent B used fact X written by agent A".
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from .common import RUNS_DIR, http_json, log_provenance

Fact = Dict[str, Any]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class MemoryAdapter:
    name = "abstract"

    def write(self, content: str, author: str, kind: str = "note",
              label: str = "", metadata: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError

    def read(self, query: str, reader: str, limit: int = 10) -> List[Fact]:
        raise NotImplementedError


class PastDevMemory(MemoryAdapter):
    name = "past.dev"
    BASE = "https://api.past.dev/api/v1"

    def __init__(self, api_key: str, wait_s: float = 90):
        self.headers = {"Authorization": "Bearer " + api_key}
        self.wait_s = wait_s

    def write(self, content, author, kind="note", label="", metadata=None):
        fact_id = "%s-%s" % (kind, uuid.uuid4().hex[:10])
        meta = dict(metadata or {}, author=author, kind=kind)
        t0 = time.perf_counter()
        resp = http_json("POST", self.BASE + "/ingest", self.headers, {
            "id": fact_id, "content": content, "label": label or kind,
            "timestamp": _now(), "identity": author, "metadata": meta,
        })
        status = self._wait(resp.get("ingestionId")) if resp.get("status") != "completed" else "completed"
        log_provenance(tool="past.ingest", agent=author, fact_id=fact_id, kind=kind,
                       content=content, status=status,
                       latency_s=round(time.perf_counter() - t0, 3))
        return fact_id

    def _wait(self, ingestion_id: Optional[str]) -> str:
        """Ingest is asynchronous: poll until processed so readers can see it."""
        if not ingestion_id:
            return "unknown"
        deadline = time.time() + self.wait_s
        status = "queued"
        while time.time() < deadline:
            st = http_json("GET", "%s/ingest/%s" % (self.BASE, ingestion_id), self.headers)
            status = st.get("status", status)
            if status in ("completed", "failed"):
                break
            time.sleep(2)
        return status

    def read(self, query, reader, limit=10):
        t0 = time.perf_counter()
        resp = http_json("POST", self.BASE + "/recall", self.headers, {
            "query": query, "identity": reader, "limit": limit, "level": "low",
        })
        facts, seen = [], set()
        for r in resp.get("results", []):
            for s in r.get("sources", []) or []:
                sid = s.get("sourceId")
                if sid in seen:
                    continue
                seen.add(sid)
                meta = s.get("metadata") or {}
                facts.append({"fact_id": sid, "author": meta.get("author"),
                              "kind": meta.get("kind"), "metadata": meta,
                              "content": " ".join(s.get("excerpts") or []) or r.get("content"),
                              "memory_summary": r.get("content")})
        log_provenance(tool="past.recall", agent=reader, query=query,
                       fact_ids=[f["fact_id"] for f in facts],
                       authors=sorted({str(f["author"]) for f in facts}),
                       latency_s=round(time.perf_counter() - t0, 3))
        return facts


class LocalMemory(MemoryAdapter):
    """Keyword-match fallback. Same interface, stored in runs/memory.jsonl."""
    name = "local"

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(RUNS_DIR, "memory.jsonl")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def write(self, content, author, kind="note", label="", metadata=None):
        fact_id = "%s-%s" % (kind, uuid.uuid4().hex[:10])
        meta = dict(metadata or {}, author=author, kind=kind)
        with open(self.path, "a") as f:
            f.write(json.dumps({"fact_id": fact_id, "content": content, "author": author,
                                "kind": kind, "label": label, "metadata": meta,
                                "ts": _now()}) + "\n")
        log_provenance(tool="local_memory.write", agent=author, fact_id=fact_id,
                       kind=kind, content=content)
        return fact_id

    def read(self, query, reader, limit=10):
        rows = []
        if os.path.exists(self.path):
            with open(self.path) as f:
                rows = [json.loads(l) for l in f if l.strip()]
        words = {w for w in query.lower().split() if len(w) > 3}
        ranked = sorted(rows, key=lambda r: -sum(w in r["content"].lower() for w in words))
        facts = ranked[:limit]
        log_provenance(tool="local_memory.read", agent=reader, query=query,
                       fact_ids=[f["fact_id"] for f in facts],
                       authors=sorted({f["author"] for f in facts}))
        return facts


def get_memory() -> MemoryAdapter:
    key = os.environ.get("PAST_API_KEY")
    return PastDevMemory(key) if key else LocalMemory()
