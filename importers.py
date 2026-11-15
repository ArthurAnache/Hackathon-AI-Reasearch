"""Bring external results into the ONE experiment store (so the loop can read them back).

    python importers.py results/ws2-v1                 # WS3 / Modal pilot results (JSONL + *_best.json)

Each external record is mapped onto the unified schema (interfaces.RECORD_FIELDS). Every best-solution
file is re-verified by the frozen problem before it becomes a certificate; one that does not verify is
reported and skipped, never trusted. Re-importing the same file adds nothing (experiment ids are stable).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import interfaces as I
from certificates import CertificateError, CertificateStore


def import_pilot(results_dir: str | Path, store=None, certs=None) -> dict:
    store, certs = store or I.JsonlStore(), certs or CertificateStore()
    results_dir = Path(results_dir)
    existing = {r["experiment_id"] for r in store.read()}
    report = {"records": 0, "certificates": [], "rejected": [], "skipped_existing": 0}
    for jsonl in sorted(results_dir.glob("*.jsonl")):
        run = jsonl.stem
        best_files = {p.stem.rsplit("_", 2)[-2]: p for p in results_dir.glob(f"{run}_*_best.json")}
        certs_by_variant = {}
        for variant, path in best_files.items():
            data = json.loads(path.read_text())
            try:
                problem = I.get_problem(data["problem"])
                cert = certs.put(problem, data["solution"], data["score"],
                                 {"imported_from": str(path), "experiment_id": data.get("experiment_id")})
                certs_by_variant[variant] = (data.get("experiment_id"), cert.id)
                report["certificates"].append(cert.id)
            except (CertificateError, KeyError) as e:
                report["rejected"].append(f"{path.name}: {e}")
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            variant = r.get("variant", "pilot")
            eid = f"{run}:{variant}:{r['experiment_id']}"
            if eid in existing:
                report["skipped_existing"] += 1
                continue
            best_eid, cert_id = certs_by_variant.get(variant, (None, None))
            store.append(I.new_record(
                experiment_id=eid, timestamp=r.get("timestamp"), problem=r["problem"], agent=r.get("agent"),
                arm=f"pilot:{variant}", island=f"pilot:{run}", worker_id="modal" if r.get("modal_cpu_requested") else None,
                optimizer=r.get("optimizer"), optimizer_hash=r.get("algorithm_version"), source_sha=r.get("git_commit"),
                parent_id=None, genes=[], hypothesis=r.get("hypothesis"), rationale=r.get("rationale"),
                config=r.get("config"), warm_start=None, evidence=r.get("evidence", []),
                seeds=[r["seed"]] if r.get("seed") is not None else [], budget_s=r.get("budget_s"),
                scores=[r["score"]] if r.get("score") is not None else [], score=r.get("score"),
                valid=bool(r.get("valid")), status=r.get("status"), best_before=r.get("best_before"),
                beats_record=bool(r.get("beats_record")), duration_s=r.get("duration_s"), error=r.get("error"),
                certificate_path=cert_id if r["experiment_id"] == best_eid else None))
            report["records"] += 1
    return report


if __name__ == "__main__":
    for d in sys.argv[1:] or ["results/ws2-v1"]:
        print(json.dumps(import_pilot(d), indent=2))
