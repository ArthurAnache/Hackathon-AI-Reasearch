"""Append-only research memory (JSONL). Auditable lineage for SAVANT."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable


KINDS = (
    "experiment", "hypothesis", "critique", "vote", "manager_decision",
    "statistical_report", "stop_vote", "iteration_decision", "prediction_check",
    "uncertainty", "rejected_idea", "ingest", "note",
)


def record_sha(entry: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class ResearchMemory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def append(self, kind: str, payload: dict, *, source: str = "savant") -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown memory kind {kind!r}")
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kind": kind,
            "source": source,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return entry

    def read_all(self) -> list[dict]:
        """Every record, including invalidated ones. A truncated last line (killed writer) is skipped."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        out = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def invalidated(self, rows: list[dict] | None = None) -> set[str]:
        rows = self.read_all() if rows is None else rows
        out: set[str] = set()
        for e in rows:
            if e.get("kind") == "note" and e.get("kind_detail") == "invalidate":
                out.update(e.get("record_sha256") or [])
        return out

    def read(self) -> list[dict]:
        """Valid records only: records listed by an `invalidate` note are kept on disk but hidden."""
        rows = self.read_all()
        bad = self.invalidated(rows)
        if not bad:
            return rows
        return [e for e in rows if record_sha(e) not in bad]

    def invalidate(self, records: list[dict], reason: str) -> int:
        shas = sorted({record_sha(e) for e in records} - self.invalidated())
        if shas:
            self.append("note", {"kind_detail": "invalidate", "reason": reason, "record_sha256": shas},
                        source="savant:audit")
        return len(shas)

    def by_kind(self, kind: str) -> list[dict]:
        return [e for e in self.read() if e.get("kind") == kind]

    def summary_for_agents(self, limit: int = 40) -> list[dict]:
        """Compact recent facts for prompts (no secrets, no full solutions)."""
        rows = []
        for e in self.read()[-limit:]:
            if e.get("kind_detail") == "invalidate":
                continue
            rows.append({k: e[k] for k in e if k not in ("solution", "candidate", "raw")})
        return rows

    def extend(self, entries: Iterable[dict]) -> int:
        n = 0
        for e in entries:
            kind = e.get("kind", "note")
            payload = {k: v for k, v in e.items() if k not in ("kind", "ts", "source")}
            self.append(kind, payload, source=e.get("source", "savant"))
            n += 1
        return n
