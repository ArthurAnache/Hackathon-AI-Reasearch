"""Refresh the live EinsteinArena #1 and compute the correct record target.

DEFAULT: offline. Reads ONLY problems/<slug>/meta.json["record"] — no network.
LIVE READ: requires arthur_authorized=True in THIS call. Never auto-reads from a campaign loop.
Does not submit. Does not write credentials. Logs go to the caller's log_dir only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def frozen_snapshot(slug: str = "second-autocorrelation-inequality") -> dict:
    """Strictly offline: meta.json record + local certificate score. No leaderboard GET."""
    from problems.catalog import load
    p = load(slug)
    meta = json.loads((ROOT / "problems" / slug / "meta.json").read_text(encoding="utf-8"))
    rec = meta["record"]
    best = {
        "value": float(rec["value"]),
        "agent": rec["agent"],
        "live": False,
        "source": f"frozen snapshot ({rec['read_at']})",
        "read_at": rec["read_at"],
    }
    target = float(best["value"]) + float(p.DELTA)
    cert_score = float(p.score(p.certificate()))
    return {
        "slug": slug,
        "mode": "frozen_snapshot",
        "best": best,
        "delta": float(p.DELTA),
        "target": target,
        "frozen_certificate_score": cert_score,
        "gain_needed_from_frozen_certificate": target - cert_score,
        "note": "Offline (meta.json only). Pass arthur_authorized=True to refresh from the live leaderboard.",
        "read_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def refresh(slug: str = "second-autocorrelation-inequality", *, arthur_authorized: bool = False,
            log_dir: Path | None = None) -> dict:
    """Return the record to beat. Live GET only when Arthur has authorized this call."""
    if not arthur_authorized:
        return frozen_snapshot(slug)
    from problems.catalog import load
    log_dir = Path(log_dir or (ROOT / "runs" / "algolab" / "live_record"))
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from integrations import einstein_arena as ea
        ea.LOG_PATH = log_dir / "arena_log.jsonl"
    except ImportError as e:
        out = frozen_snapshot(slug)
        out["error"] = f"integrations.einstein_arena unavailable: {e}"
        return out
    p = load(slug)
    best = p.best_known_score(max_age_s=0)  # force refresh; Arthur authorized
    target = float(best["value"]) + float(p.DELTA)
    cert_score = float(p.score(p.certificate()))
    out = {
        "slug": slug,
        "mode": "live",
        "arthur_authorized": True,
        "best": best,
        "delta": float(p.DELTA),
        "target": target,
        "frozen_certificate_score": cert_score,
        "gain_needed_from_frozen_certificate": target - cert_score,
        "gain_needed_from_live_one": float(p.DELTA),
        "read_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Live #1 re-read under Arthur authorization. Warm-start future waves from a re-verified copy "
                "of this solution when best.live is True. Still no submission.",
    }
    (log_dir / "last_refresh.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out
