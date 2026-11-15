"""A small past.dev client for the Knockout loop, plus a local JSONL store with the same interface.

    store = get_store(log_path=run_dir / "pastdev_log.jsonl")      # past.dev if PAST_API_KEY is set, else local
    ingestion_id = store.ingest(fact_id, content, label, timestamp, identity, metadata)
    facts = store.recall(query, identity, limit, occurred_from=None)
    # -> [{"fact_id", "author", "kind", "metadata", "content", "summary"}]

past.dev Memory API (https://past.dev/docs/memory-api/): base https://api.past.dev/api/v1, header
`Authorization: Bearer past_sk_...`; `POST /ingest` {id, content, label, timestamp, identity, metadata} is
asynchronous (poll `GET /ingest/{ingestionId}`); `POST /recall` {query, identity, limit, level, occurredFrom}
returns results whose `sources[]` carry `sourceId`, `metadata` and `excerpts`. The same calls as WS4's client
(integrations/pastdev_client.py on ws4-sponsor-integrations), rewritten here so the Knockout loop does not depend
on that branch.

Every call is appended to `log_path` (operation, identity, fact ids, authors of recalled facts, latency, status).
The key is never logged, printed or written anywhere. Only the orchestrator uses this module: never a sandboxed
program, never a Modal worker.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from knockout.problems_api import REPO_ROOT

BASE = "https://api.past.dev/api/v1"


class PastDevError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class _Log:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def __call__(self, **event) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps({"timestamp": utc_now(), **event}, default=str) + "\n")


def _fact_from_source(src: dict, summary: str | None) -> dict:
    meta = src.get("metadata") or {}
    return {"fact_id": src.get("sourceId"), "author": meta.get("author"), "kind": meta.get("kind"),
            "metadata": meta, "content": " ".join(src.get("excerpts") or []) or summary or "",
            "summary": summary}


class PastDevStore:
    name = "past.dev"

    def __init__(self, api_key: str, *, base: str = BASE, timeout_s: float = 8.0, log_path=None, opener=None):
        if not api_key:
            raise PastDevError("no past.dev key")
        self._key = api_key
        self.base, self.timeout_s = base.rstrip("/"), float(timeout_s)
        self.log = _Log(log_path)
        self._open = opener or urllib.request.urlopen     # injectable for offline tests

    def __repr__(self) -> str:                          # never show the key
        return f"PastDevStore(base={self.base!r})"

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Authorization": "Bearer " + self._key,
                                              "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self._open(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise PastDevError(f"HTTP {exc.code} on {method} {path}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PastDevError(f"{type(exc).__name__} on {method} {path}") from None
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            raise PastDevError(f"non-JSON answer on {method} {path}") from None

    def ingest(self, fact_id: str, content: str, label: str, timestamp: str, identity: str,
               metadata: dict) -> str | None:
        t0 = time.perf_counter()
        status = "error"
        try:
            resp = self._request("POST", "/ingest", {"id": fact_id, "content": content, "label": label,
                                                     "timestamp": timestamp, "identity": identity,
                                                     "metadata": metadata})
            status = resp.get("status", "queued")
            return resp.get("ingestionId")
        finally:
            self.log(op="ingest", store=self.name, identity=identity, fact_id=fact_id, kind=metadata.get("kind"),
                     status=status, latency_s=round(time.perf_counter() - t0, 3))

    def status(self, ingestion_id: str) -> str:
        return str(self._request("GET", f"/ingest/{ingestion_id}").get("status", "unknown"))

    def recall(self, query: str, identity: str, limit: int, occurred_from: str | None = None) -> list[dict]:
        t0 = time.perf_counter()
        facts: list[dict] = []
        status = "error"
        try:
            body = {"query": query, "identity": identity, "limit": int(limit), "level": "low"}
            if occurred_from:
                body["occurredFrom"] = occurred_from
            resp = self._request("POST", "/recall", body)
            seen = set()
            for r in resp.get("results", []) or []:
                for src in r.get("sources", []) or []:
                    sid = src.get("sourceId")
                    if sid and sid not in seen:
                        seen.add(sid)
                        facts.append(_fact_from_source(src, r.get("content")))
            status = "ok"
            return facts
        finally:
            self.log(op="recall", store=self.name, identity=identity, query=query[:300], status=status,
                     fact_ids=[f["fact_id"] for f in facts], authors=sorted({str(f["author"]) for f in facts}),
                     latency_s=round(time.perf_counter() - t0, 3))


class LocalStore:
    """Same interface over one JSONL file shared by the runs of a machine (keyword ranking). Used when
    PAST_API_KEY is not set, so the loop behaves the same way, only without cross-machine sharing."""
    name = "local"

    def __init__(self, path: str | Path, *, log_path=None):
        self.path = Path(path)
        self.log = _Log(log_path)
        self._lock = threading.Lock()

    def ingest(self, fact_id, content, label, timestamp, identity, metadata) -> str:
        row = {"fact_id": fact_id, "content": content, "label": label, "timestamp": timestamp,
               "author": identity, "kind": metadata.get("kind"), "metadata": metadata}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            existing = self._rows()
            if not any(r["fact_id"] == fact_id for r in existing):      # idempotent by fact id
                with self.path.open("a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
        self.log(op="ingest", store=self.name, identity=identity, fact_id=fact_id, kind=metadata.get("kind"),
                 status="completed")
        return fact_id

    def status(self, ingestion_id: str) -> str:
        return "completed"

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out

    def recall(self, query: str, identity: str, limit: int, occurred_from: str | None = None) -> list[dict]:
        rows = [r for r in self._rows() if occurred_from is None or r["timestamp"] >= occurred_from]
        words = {w for w in query.lower().replace("[", " ").replace("]", " ").split() if len(w) > 2}
        ranked = sorted(reversed(rows), key=lambda r: -sum(w in r["content"].lower() for w in words))[:int(limit)]
        facts = [{"fact_id": r["fact_id"], "author": r["author"], "kind": r["kind"], "metadata": r["metadata"],
                  "content": r["content"], "summary": None} for r in ranked]
        self.log(op="recall", store=self.name, identity=identity, query=query[:300], status="ok",
                 fact_ids=[f["fact_id"] for f in facts], authors=sorted({str(f["author"]) for f in facts}))
        return facts


def api_key() -> str:
    """PAST_API_KEY from the environment, else from the repo's .env (that one key only)."""
    key = os.environ.get("PAST_API_KEY", "").strip()
    if not key and (REPO_ROOT / ".env").exists():
        try:
            from dotenv import dotenv_values
            key = (dotenv_values(REPO_ROOT / ".env").get("PAST_API_KEY") or "").strip()
        except ImportError:
            pass
    return key


def get_store(*, log_path=None, local_path=None, timeout_s: float = 8.0):
    key = api_key()
    if key:
        return PastDevStore(key, timeout_s=timeout_s, log_path=log_path)
    return LocalStore(local_path or REPO_ROOT / "runs" / "pastdev_local.jsonl", log_path=log_path)
