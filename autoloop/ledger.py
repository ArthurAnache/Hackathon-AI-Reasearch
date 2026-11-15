"""Append-only ledger (JSONL) with two record kinds: certificate and decision (plus claim and note).

Certificate: id (sha256 of the float64 bytes, or of the integer JSON once certified), problem, cells, score (float),
exact (fraction "p/q" or null), parent, stage, params, cpu_s, completed, path, origin ("autoloop" | "imported").
Decision: t, options, move, args, reason, policy, numbers.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        if self.path.exists():
            self.records = [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def append(self, rec: dict) -> dict:
        rec = dict(rec)
        rec.setdefault("t", time.time())
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            f.flush(); os.fsync(f.fileno())
        self.records.append(rec)
        return rec

    # --- views ---
    def certs(self) -> list[dict]:
        return [r for r in self.records if r["kind"] == "certificate"]

    def cert(self, cid: str) -> dict | None:
        for r in self.records:
            if r["kind"] == "certificate" and r["id"] == cid:
                return r
        return None

    def decisions(self) -> list[dict]:
        return [r for r in self.records if r["kind"] == "decision"]

    def of_kind(self, kind: str) -> list[dict]:
        return [r for r in self.records if r["kind"] == kind]


def cert_record(*, problem, cid, cells, score_, parent, stage, params, cpu_s, path, origin="autoloop", exact=None,
                completed=True, root=None) -> dict:
    return dict(kind="certificate", id=cid, problem=problem, cells=int(cells), score=float(score_), exact=exact,
                parent=parent, root=root or cid, stage=stage, params=params, cpu_s=float(cpu_s), completed=completed,
                path=str(path), origin=origin)


def import_memory(ledger: Ledger, source: str, prob) -> int:
    """Copy certificate records of another ledger (origin -> imported), or import a directory / file of solutions."""
    have = {c["id"] for c in ledger.certs()}
    n = 0
    src = Path(source)
    if src.suffix == ".jsonl":
        for r in [json.loads(x) for x in src.read_text().splitlines() if x.strip()]:
            if r.get("kind") == "certificate" and r["id"] not in have:
                ledger.append({**r, "origin": "imported", "t": time.time()}); have.add(r["id"]); n += 1
            elif r.get("kind") == "memory_baseline":
                ledger.append({**r, "source_ledger": str(src), "t": time.time()})
        return n
    files = sorted(src.rglob("*")) if src.is_dir() else [src]
    for f in files:
        if f.suffix not in (".npy", ".json") or f.name.endswith((".ckpt.json",)):
            continue
        try:
            v = prob.load_vec(str(f))
        except Exception:  # noqa: BLE001  (summaries and other JSON files without values)
            continue
        if not prob.valid_vec(v):
            continue
        cid = prob.vec_id(v)
        if cid in have:
            continue
        ledger.append(cert_record(problem=prob.NAME, cid=cid, cells=len(v), score_=prob.score(v), parent=None, stage="imported",
                                  params={"file": f.name}, cpu_s=0.0, path=str(f.resolve()), origin="imported"))
        have.add(cid); n += 1
    return n
