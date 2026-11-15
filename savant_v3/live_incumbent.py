"""Fetch the PUBLIC live #1 solution (EinsteinArena GET /api/solutions/best).

This is not an Arthur-private certificate: the values are served by the public API.
Always re-score with the frozen WS1 verifier before use.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "second-autocorrelation-inequality"
CACHE = ROOT / "runs" / "v3" / "live_incumbent.json.gz"


def fetch_and_verify(*, force: bool = False) -> dict:
    import gzip
    from integrations import einstein_arena as ea
    from problems.catalog import load

    log = ROOT / "runs" / "v3" / "arena_logs"
    log.mkdir(parents=True, exist_ok=True)
    ea.LOG_PATH = log / "arena_log.jsonl"

    if CACHE.is_file() and not force:
        raw = json.loads(gzip.decompress(CACHE.read_bytes()).decode())
        # still re-verify
        p = load(SLUG)
        score = float(p.score({"values": raw["values"]}))
        raw["verified_score"] = score
        raw["verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return raw

    rows = ea.get_best_solutions(3, limit=1)
    if not rows:
        raise RuntimeError("public best solutions empty")
    row = rows[0]
    values = (row.get("data") or {}).get("values")
    if not values:
        raise RuntimeError("best solution missing data.values")
    cand = {"values": values}
    p = load(SLUG)
    score = float(p.score(cand))
    meta = {
        "source": "GET /api/solutions/best",
        "solution_id": row.get("id"),
        "agentName": row.get("agentName"),
        "api_score": row.get("score"),
        "verified_score": score,
        "n": len(values),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Public API incumbent. Parent-verified with frozen WS1 verifier.",
        "values": values,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(gzip.compress(json.dumps(meta).encode(), compresslevel=6))
    # also write values-only warm start for INIT
    warm = ROOT / "runs" / "v3" / "live_incumbent_warm.json"
    warm.write_text(json.dumps({"solution": {"values": values}}), encoding="utf-8")
    return meta
