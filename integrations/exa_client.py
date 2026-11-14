"""Exa search adapter. Drop-in replacement for the WS1 client (same `search`
signature, same return shape, same log file), plus:

- several API keys with automatic failover (EXA_API_KEYS=name:key,name:key)
- spending limit = each key's own prepaid balance on exa.ai: a key that runs out
  of credit is disabled and the next key is used. An optional extra cap for all
  keys together can be set with EXA_BUDGET_USD / EXA_MAX_CALLS (unset = no cap)
- a cache: the same query with the same parameters is never paid for twice
- no SDK needed: plain HTTPS to https://api.exa.ai/search

    from integrations.exa_client import search
    hits = search("ring loading problem improved bound", since="2025-01-01", n=8,
                  decision="stage2 literature check: ring-loading-15")

Returns [{url, title, published_date, highlights}]. Every call, successful,
cached, blocked or failed, is appended to selection/exa_log.jsonl with the
query, parameters, duration, URLs, cost, key label and the decision it served.
Keys are never printed or logged. Importing never fails; only `search` raises
ExaUnavailable, so callers can catch it and carry on.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from ._common import (REPO_ROOT, STATE_DIR, AdapterError, KeyPool, NoUsableKey, append_jsonl,
                      http_json, load_env, read_json, utc_now, write_json)

URL = "https://api.exa.ai/search"
LOG_PATH = Path(os.environ.get("EXA_LOG_PATH", REPO_ROOT / "selection" / "exa_log.jsonl"))
SPEND_PATH = STATE_DIR / "exa_spend.json"
CACHE_DIR = STATE_DIR / "exa_cache"
EST_COST_PER_CALL = 0.01   # used only when the response carries no costDollars


class ExaUnavailable(RuntimeError):
    """Raised when Exa cannot be used (no key, stop loss reached, all keys failed)."""


class BudgetExceeded(ExaUnavailable):
    pass


def _limits():
    """Optional extra caps. Unset or empty = no cap: the limit is each key's balance."""
    usd = os.environ.get("EXA_BUDGET_USD", "").strip()
    calls = os.environ.get("EXA_MAX_CALLS", "").strip()
    return (float(usd) if usd else float("inf"), int(calls) if calls else 10 ** 12)


def spend() -> Dict:
    """Total Exa spend recorded on this machine: {"usd", "calls", "by_key"}."""
    return read_json(SPEND_PATH, {"usd": 0.0, "calls": 0, "by_key": {}})


def search(query: str, since: Optional[str] = None, n: int = 10, *, decision: str = "unspecified",
           search_type: str = "auto") -> List[Dict]:
    """Search Exa and return [{url, title, published_date, highlights}].

    since: ISO date (YYYY-MM-DD) for start_published_date, or None.
    decision: which project decision this call serves (logged for traceability).
    """
    load_env()
    params = {"type": search_type, "num_results": n, "start_published_date": since,
              "contents": {"highlights": True}}
    entry = {"timestamp": utc_now(), "query": query, "params": params, "decision": decision}
    body = {"query": query, "type": search_type, "numResults": n, "contents": {"highlights": True}}
    if since:
        body["startPublishedDate"] = since

    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / (key + ".json")
    if cache_path.exists():
        hits = read_json(cache_path, [])
        entry.update(duration_s=0.0, ok=True, cached=True, cost_dollars=0.0,
                     urls=[h["url"] for h in hits])
        append_jsonl(LOG_PATH, entry)
        return hits

    pool = KeyPool("exa", "EXA_API_KEYS", "EXA_API_KEY")
    if not len(pool):
        entry.update(duration_s=0.0, ok=False, error="no Exa key configured", urls=[])
        append_jsonl(LOG_PATH, entry)
        raise ExaUnavailable("EXA_API_KEYS / EXA_API_KEY missing: add it to .env (see .env.example)")

    budget_usd, max_calls = _limits()
    s = spend()
    if s["usd"] >= budget_usd or s["calls"] >= max_calls:
        entry.update(duration_s=0.0, ok=False, blocked="stop loss", spend=s,
                     budget_usd=budget_usd, max_calls=max_calls, urls=[])
        append_jsonl(LOG_PATH, entry)
        raise BudgetExceeded("Exa optional cap reached: $%.3f / %d calls used (caps $%s / %s)"
                             % (s["usd"], s["calls"], budget_usd, max_calls))

    t0 = time.perf_counter()
    try:
        label, resp = pool.call(
            lambda _l, k: http_json("POST", URL, {"x-api-key": k}, body, retries=2))
    except (AdapterError, NoUsableKey) as exc:
        entry.update(duration_s=round(time.perf_counter() - t0, 3), ok=False,
                     error=("%s: %s" % (type(exc).__name__, exc))[:300], urls=[])
        append_jsonl(LOG_PATH, entry)
        raise ExaUnavailable("Exa search failed: %s" % type(exc).__name__) from exc

    hits = [{"url": r.get("url"), "title": r.get("title"), "published_date": r.get("publishedDate"),
             "highlights": list(r.get("highlights") or [])} for r in resp.get("results", [])]
    cost = float((resp.get("costDollars") or {}).get("total", EST_COST_PER_CALL))
    s["usd"] = round(s["usd"] + cost, 6)
    s["calls"] += 1
    s.setdefault("by_key", {})[label] = round(s["by_key"].get(label, 0.0) + cost, 6)
    write_json(SPEND_PATH, s)
    write_json(cache_path, hits)
    entry.update(duration_s=round(time.perf_counter() - t0, 3), ok=True, urls=[h["url"] for h in hits],
                 resolved_search_type=resp.get("resolvedSearchType"), cost_dollars=cost,
                 key_label=label, spent_total_usd=s["usd"])
    append_jsonl(LOG_PATH, entry)
    return hits
