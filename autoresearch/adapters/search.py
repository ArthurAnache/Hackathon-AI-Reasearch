"""Literature / record search.  SearchAdapter.search(query, agent, purpose)

ExaSearch  - https://api.exa.ai/search (header x-api-key)
NullSearch - fallback when EXA_API_KEY is missing; returns nothing, says so.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List

from .common import RUNS_DIR, http_json, log_provenance

# Stop loss: set in .env.  Spend is persisted across runs in runs/exa_spend.json.
def _limits():
    """Read at call time so values from .env (loaded later) are honoured."""
    return (float(os.environ.get("EXA_BUDGET_USD", "5")),
            int(os.environ.get("EXA_MAX_CALLS", "200")))

EXA_EST_COST_PER_CALL = 0.01   # used only if the API response has no costDollars
SPEND_PATH = os.path.join(RUNS_DIR, "exa_spend.json")
CACHE_DIR = os.path.join(RUNS_DIR, "exa_cache")


class BudgetExceeded(RuntimeError):
    pass


def _load_spend() -> Dict[str, float]:
    try:
        with open(SPEND_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"usd": 0.0, "calls": 0}


def _save_spend(spend: Dict[str, float]) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(SPEND_PATH, "w") as f:
        json.dump(spend, f)

Evidence = Dict[str, Any]   # {"title", "url", "snippet", "published"}


class SearchAdapter:
    name = "abstract"

    def search(self, query: str, agent: str, purpose: str,
               num_results: int = 5) -> List[Evidence]:
        raise NotImplementedError


class ExaSearch(SearchAdapter):
    name = "exa"
    URL = "https://api.exa.ai/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query, agent, purpose, num_results=5):
        body = {"query": query, "numResults": num_results,
                "contents": {"text": {"maxCharacters": 1200}}}
        key = hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
        cache_path = os.path.join(CACHE_DIR, key + ".json")
        if os.path.exists(cache_path):           # same question again: free
            with open(cache_path) as f:
                out = json.load(f)
            log_provenance(tool="exa.search", agent=agent, purpose=purpose, query=query,
                           cached=True, cost_usd=0, urls=[e["url"] for e in out])
            return out

        EXA_BUDGET_USD, EXA_MAX_CALLS = _limits()
        spend = _load_spend()
        if spend["usd"] >= EXA_BUDGET_USD or spend["calls"] >= EXA_MAX_CALLS:
            log_provenance(tool="exa.search", agent=agent, purpose=purpose, query=query,
                           blocked="stop loss", spend=spend,
                           budget_usd=EXA_BUDGET_USD, max_calls=EXA_MAX_CALLS)
            raise BudgetExceeded("Exa stop loss hit: $%.2f / %d calls used (limits $%.2f / %d)"
                                 % (spend["usd"], spend["calls"], EXA_BUDGET_USD, EXA_MAX_CALLS))

        t0 = time.perf_counter()
        error, out, cost = None, [], 0.0
        try:
            resp = http_json("POST", self.URL, {"x-api-key": self.api_key}, body, retries=2)
            cost = float((resp.get("costDollars") or {}).get("total", EXA_EST_COST_PER_CALL))
            for r in resp.get("results", []):
                out.append({"title": r.get("title"), "url": r.get("url"),
                            "published": r.get("publishedDate"),
                            "snippet": (r.get("text") or "")[:1200]})
        except Exception as e:
            error = str(e)
        spend["usd"] += cost
        spend["calls"] += 1
        _save_spend(spend)
        if not error:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(out, f)
        log_provenance(tool="exa.search", agent=agent, purpose=purpose, query=query,
                       latency_s=round(time.perf_counter() - t0, 3), cost_usd=cost,
                       spent_total_usd=round(spend["usd"], 4), calls_total=spend["calls"],
                       urls=[e["url"] for e in out], error=error)
        if error:
            raise RuntimeError(error)
        return out


class NullSearch(SearchAdapter):
    name = "none"

    def search(self, query, agent, purpose, num_results=5):
        log_provenance(tool="search.disabled", agent=agent, purpose=purpose,
                       query=query, note="EXA_API_KEY not set; no evidence gathered")
        return []


def get_search() -> SearchAdapter:
    key = os.environ.get("EXA_API_KEY")
    return ExaSearch(key) if key else NullSearch()
