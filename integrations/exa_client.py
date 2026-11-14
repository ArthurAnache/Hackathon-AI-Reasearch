"""Thin adapter around the official Exa SDK (exa-py 2.x).

    from integrations.exa_client import search
    hits = search("ring loading problem improved bound", since="2025-01-01", n=8,
                  decision="stage2 literature check: ring-loading-15")

Returns a list of {url, title, published_date, highlights}. Every call, successful or not, is appended to
selection/exa_log.jsonl with: timestamp, query, parameters, duration, returned URLs, and the decision it served.

The API key is read from EXA_API_KEY in the repo's .env (git-ignored) via python-dotenv. It is never printed or logged.
Importing this module never fails when the key is missing: only `search` raises ExaUnavailable, so callers can
catch it and carry on.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "selection" / "exa_log.jsonl"

_client = None


class ExaUnavailable(RuntimeError):
    """Raised when Exa cannot be used (missing key or missing SDK)."""


def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        from dotenv import load_dotenv
        from exa_py import Exa
    except ImportError as exc:
        raise ExaUnavailable(f"Exa SDK not installed ({exc.name}); run .venv/bin/pip install -r requirements.txt")
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("EXA_API_KEY", "").strip()
    if not key:
        raise ExaUnavailable("EXA_API_KEY is missing: add it to .env at the repo root (see .env.example)")
    _client = Exa(api_key=key)
    return _client


def _log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def search(query: str, since: str | None = None, n: int = 10, *, decision: str = "unspecified",
           search_type: str = "auto") -> list[dict]:
    """Search Exa and return [{url, title, published_date, highlights}].

    since: ISO date (YYYY-MM-DD) passed as start_published_date, or None for no date filter.
    decision: which project decision this call serves (logged for traceability).
    """
    params = {"type": search_type, "num_results": n, "start_published_date": since,
              "contents": {"highlights": True}}
    entry = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "query": query,
             "params": params, "decision": decision}
    t0 = time.perf_counter()
    try:
        client = _get_client()
        kwargs = {k: v for k, v in params.items() if v is not None}
        resp = client.search(query, **kwargs)
    except Exception as exc:
        entry.update(duration_s=round(time.perf_counter() - t0, 3), ok=False,
                     error=f"{type(exc).__name__}: {exc}"[:300], urls=[])
        _log(entry)
        if isinstance(exc, ExaUnavailable):
            raise
        raise ExaUnavailable(f"Exa search failed: {type(exc).__name__}") from exc

    hits = [{"url": r.url, "title": r.title, "published_date": r.published_date,
             "highlights": list(r.highlights or [])} for r in resp.results]
    cost = getattr(resp, "cost_dollars", None)
    entry.update(duration_s=round(time.perf_counter() - t0, 3), ok=True, urls=[h["url"] for h in hits],
                 resolved_search_type=getattr(resp, "resolved_search_type", None),
                 cost_dollars=getattr(cost, "total", None) if cost is not None else None)
    _log(entry)
    return hits
