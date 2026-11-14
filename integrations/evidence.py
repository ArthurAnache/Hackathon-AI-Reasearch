"""WS4 plug-ins for the shared contract in interfaces.py (WS2):

    EvidenceSource.gather(problem, history) -> [Evidence]
    ExperimentStore.append(result) / read()

ExaEvidence       literature search, only when it can change a decision:
                  a record audit the first time a problem is seen, then idea
                  mining each time the loop plateaus. Findings are also written
                  to shared memory, so other agents reuse them for free.
PastDevEvidence   what other agents learned, recalled from past.dev. A config
                  that another agent kept is turned into an actionable hint
                  ({"config": {...}}, which FeedbackAgent's EvidenceHints acts on).
CombinedEvidence  several sources; one failing never stops the loop.
PastDevMirrorStore the local JSONL store stays the source of truth; every
                  experiment is also mirrored to shared memory as a fact.
TaggedStore       stamps every experiment record with the integration switches
                  in force ({"exa": bool, "pastdev": bool, "memory_scope", ...}),
                  so "tool off" is never confused with "tool on, found nothing".

memory_scope / since isolate a comparison run from facts of earlier runs (see
pastdev_client); without a scope, agents read the accumulated memory.

Every Evidence carries source, citation (URL or memory fact id) and author, and
run_experiment() stores it in the experiment record, so each decision can be
traced back to the search result or the agent that informed it.
"""
from __future__ import annotations

import json
import time
from typing import List, Optional

from . import exa_client
from ._common import utc_now
from .pastdev_client import Memory, get_memory

INIT_KEY = "INIT"   # warm-start path: machine-specific, never shared as a hint


def _evidence_cls():
    from interfaces import Evidence   # WS2 contract; imported lazily
    return Evidence


def _problem_rows(problem, history):
    return [r for r in history if r.get("problem") in (None, problem.id)]


class ExaEvidence:
    name = "exa"

    def __init__(self, memory: Optional[Memory] = None, plateau: int = 3, n: int = 3,
                 author: str = "exa-scout", scope: Optional[str] = None):
        self.memory, self.plateau, self.n, self.author = memory, plateau, n, author
        self.scope = scope
        self._fired = set()

    def _trigger(self, problem, history):
        rows = _problem_rows(problem, history)
        if ("audit", problem.id) not in self._fired:
            return ("audit", problem.id)
        tail = 0
        for r in reversed(rows):
            if r.get("status") == "keep":
                break
            tail += 1
        if tail >= self.plateau:
            kept = sum(r.get("status") == "keep" for r in rows)
            return ("ideas", problem.id, kept)       # once per plateau episode
        return None

    def gather(self, problem, history) -> list:
        trig = self._trigger(problem, history)
        if trig is None or trig in self._fired:
            return []
        self._fired.add(trig)
        if trig[0] == "audit":
            query = "%s best known bound record" % problem.name
        else:
            query = "%s new construction or optimization method that improved the bound" % problem.name
        decision = "loop %s: problem %s" % (trig[0], problem.id)
        t0 = time.perf_counter()
        try:
            hits = exa_client.search(query, n=self.n, decision=decision)
        except exa_client.ExaUnavailable:
            return []                               # already logged; the loop carries on
        latency = round(time.perf_counter() - t0, 3)
        Evidence = _evidence_cls()
        out = [Evidence(source="exa",
                        content="%s: %s" % (h.get("title") or "", " ".join(h.get("highlights") or [])[:400]),
                        citation=h.get("url") or "", query=query, author=self.author,
                        retrieved_at=utc_now(), latency_s=latency) for h in hits]
        if self.memory is not None and hits:
            try:
                self.memory.write(
                    "Literature (%s) for problem %s (%s): " % (trig[0], problem.id, problem.name)
                    + " | ".join("%s <%s>" % (h.get("title"), h.get("url")) for h in hits),
                    author=self.author, kind="literature", label="%s %s" % (problem.id, trig[0]),
                    metadata={"problem": problem.id, "trigger": trig[0], "query": query,
                              "urls": [h.get("url") for h in hits]}, wait=False, scope=self.scope)
            except Exception:
                pass
        return out


class PastDevEvidence:
    name = "past.dev"

    def __init__(self, memory: Optional[Memory] = None, reader: str = "agent", limit: int = 8,
                 scope: Optional[str] = None, since: Optional[str] = None):
        self.memory = memory or get_memory()
        self.reader, self.limit, self.scope, self.since = reader, limit, scope, since

    def gather(self, problem, history) -> list:
        query = "problem %s %s: kept experiments, failed ideas, literature" % (problem.id, problem.name)
        t0 = time.perf_counter()
        try:
            facts = self.memory.read(query, reader=self.reader, limit=self.limit,
                                     scope=self.scope, since=self.since)
        except Exception:
            return []
        latency = round(time.perf_counter() - t0, 3)
        Evidence = _evidence_cls()
        out = []
        for f in facts:
            meta = f.get("metadata") or {}
            if meta.get("problem") != problem.id:
                continue                              # unrelated or untagged fact
            author = f.get("author") or ""
            content = (f.get("content") or "")[:400]
            cfg = meta.get("config")
            if (meta.get("kind") == "experiment" and meta.get("status") == "keep"
                    and isinstance(cfg, dict) and author != self.reader):
                hint = {k: v for k, v in cfg.items() if k != INIT_KEY}
                if hint:
                    content += " " + json.dumps({"config": hint})
            out.append(Evidence(source="past.dev", content=content, citation=f.get("fact_id") or "",
                                query=query, author=author, retrieved_at=utc_now(), latency_s=latency))
        return out


class CombinedEvidence:
    name = "combined"

    def __init__(self, *sources):
        self.sources = [s for s in sources if s is not None]

    def gather(self, problem, history) -> list:
        out: List = []
        for s in self.sources:
            try:
                out.extend(s.gather(problem, history))
            except Exception:
                continue
        return out


class PastDevMirrorStore:
    """ExperimentStore: local JSONL (source of truth) + a fact per experiment."""

    def __init__(self, path, memory: Optional[Memory] = None, author: str = "agent",
                 scope: Optional[str] = None):
        from interfaces import JsonlStore
        self.inner = JsonlStore(path)
        self.memory = memory or get_memory()
        self.author, self.scope = author, scope

    def append(self, result: dict) -> None:
        self.inner.append(result)
        if result.get("status") in ("void",):
            return
        cfg = {k: v for k, v in (result.get("config") or {}).items() if k != INIT_KEY}
        text = ("Experiment %s on problem %s by %s: %s, score %s (best before %s). Hypothesis: %s"
                % (result.get("experiment_id"), result.get("problem"), self.author, result.get("status"),
                   result.get("score"), result.get("best_before"), result.get("hypothesis")))
        try:
            self.memory.write(text, author=self.author, kind="experiment",
                              label="%s %s" % (result.get("problem"), result.get("status")),
                              metadata={"problem": result.get("problem"),
                                        "experiment_id": result.get("experiment_id"),
                                        "status": result.get("status"), "score": result.get("score"),
                                        "beats_record": result.get("beats_record"), "config": cfg},
                              wait=False, scope=self.scope)
        except Exception:
            pass                                     # mirroring must never break the loop

    def read(self) -> list:
        return self.inner.read()


class TaggedStore:
    """ExperimentStore wrapper: adds an "integrations" field to every record.
    Extra fields are allowed by interfaces.JsonlStore (it only checks missing ones)."""

    def __init__(self, inner, tags: dict):
        self.inner, self.tags = inner, dict(tags)

    def append(self, result: dict) -> None:
        self.inner.append({**result, "integrations": self.tags})

    def read(self) -> list:
        return self.inner.read()


def from_flags(agent_id: str, use_exa: bool, use_pastdev: bool, memory: Optional[Memory] = None,
               scope: Optional[str] = None, since: Optional[str] = None) -> CombinedEvidence:
    if use_pastdev and memory is None:
        memory = get_memory()
    return CombinedEvidence(
        ExaEvidence(memory=memory if use_pastdev else None, scope=scope) if use_exa else None,
        PastDevEvidence(memory, reader=agent_id, scope=scope, since=since) if use_pastdev else None)
