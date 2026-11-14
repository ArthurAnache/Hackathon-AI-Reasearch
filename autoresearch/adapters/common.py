"""Shared plumbing for sponsor adapters: .env loading, HTTP with retries,
and the provenance trace (runs/provenance.jsonl).

Standard library only, so the core runner never depends on sponsor SDKs.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

RUNS_DIR = os.environ.get("AUTORESEARCH_RUNS_DIR", "runs")
PROVENANCE_PATH = os.path.join(RUNS_DIR, "provenance.jsonl")


def load_env(path: str = ".env") -> None:
    """Read KEY=VALUE lines into os.environ (existing variables win)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v:
                os.environ.setdefault(k.strip(), v)


class AdapterError(Exception):
    pass


def http_json(method: str, url: str, headers: Dict[str, str],
              body: Optional[Dict[str, Any]] = None, retries: int = 3,
              timeout: float = 30) -> Dict[str, Any]:
    """JSON request with exponential backoff on 429 / 5xx / network errors."""
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Content-Type": "application/json", "Accept": "application/json",
            "User-Agent": "autoresearch/0.1", **headers}
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            last = AdapterError("HTTP %d %s: %s" % (e.code, url, detail))
            if e.code != 429 and e.code < 500:
                raise last            # client error: retrying will not help
        except (urllib.error.URLError, TimeoutError) as e:
            last = AdapterError("network error %s: %s" % (url, e))
        time.sleep(2 ** attempt)
    raise last


def log_provenance(**event: Any) -> None:
    """Append one external-tool event. Never stores API keys."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}
    with open(PROVENANCE_PATH, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")
