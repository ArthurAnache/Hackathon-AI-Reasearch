"""Shared plumbing for the WS4 integrations: .env loading, multi-key pools with
failover, HTTP with retries, and append-only JSONL logs.

Standard library only, so the research loop never depends on a sponsor SDK.
Secrets are never written to any log: keys appear only as their labels.
"""
from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
# Mutable local state (spend counters, query cache, disabled keys). Git-ignored.
STATE_DIR = Path(os.environ.get("INTEGRATIONS_STATE_DIR", REPO_ROOT / ".cache" / "integrations"))


def load_env(path: Optional[Path] = None) -> None:
    """Read KEY=VALUE lines from the git-ignored .env (existing variables win).
    INTEGRATIONS_ENV_FILE overrides the location (tests point it at nothing)."""
    path = Path(path or os.environ.get("INTEGRATIONS_ENV_FILE") or REPO_ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v:
            os.environ.setdefault(k.strip(), v)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_jsonl(path: Path, entry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


class AdapterError(Exception):
    def __init__(self, msg: str, status: Optional[int] = None):
        super().__init__(msg)
        self.status = status


class NoUsableKey(RuntimeError):
    pass


def http_json(method: str, url: str, headers: Dict[str, str],
              body: Optional[Dict[str, Any]] = None, retries: int = 3,
              timeout: float = 30) -> Dict[str, Any]:
    """JSON request with exponential backoff on 429 / 5xx / network errors."""
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Content-Type": "application/json", "Accept": "application/json",
            "User-Agent": "hackathon-autoresearch/0.1", **headers}
    last: Optional[Exception] = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            last = AdapterError("HTTP %d %s: %s" % (e.code, url, detail), e.code)
            if e.code != 429 and e.code < 500:
                raise last            # client error: retrying will not help
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            # URLError, timeouts, connection reset / remote disconnect
            last = AdapterError("network error %s: %s" % (url, e))
        time.sleep(2 ** attempt)
    raise last  # type: ignore[misc]


class KeyPool:
    """Several API keys for one service, tried in order with automatic failover.

    .env formats (labels only appear in logs, keys never do):
        EXA_API_KEYS=murilo:key1,arthur:key2,marie:key3
        EXA_API_KEYS=key1,key2
        EXA_API_KEY=key1                       (single key, backward compatible)
    "name:" with no key is ignored. A key that fails authentication or runs out
    of credit is disabled and remembered in STATE_DIR/keys_<service>.json.
    """

    def __init__(self, service: str, multi_var: str, single_var: str):
        self.service = service
        raw = os.environ.get(multi_var) or os.environ.get(single_var) or ""
        self.keys: List[Tuple[str, str]] = []
        for i, item in enumerate(x.strip() for x in raw.split(",")):
            if not item:
                continue
            label, key = item.split(":", 1) if ":" in item else ("key%d" % (i + 1), item)
            if key.strip():
                self.keys.append((label.strip(), key.strip()))
        self.status_path = STATE_DIR / ("keys_%s.json" % service)
        self.dead: Dict[str, str] = read_json(self.status_path, {})

    def __len__(self) -> int:
        return len(self.keys)

    def usable(self) -> List[Tuple[str, str]]:
        return [(l, k) for l, k in self.keys if l not in self.dead]

    def mark_dead(self, label: str, reason: str) -> None:
        self.dead[label] = reason
        write_json(self.status_path, self.dead)

    def call(self, fn: Callable[[str, str], Any], dead_on=(401, 402, 403), skip_on=(429,)):
        """Run fn(label, key) with each usable key until one succeeds.
        Returns (label, result)."""
        errors = []
        for label, key in self.usable():
            try:
                return label, fn(label, key)
            except AdapterError as e:
                errors.append("%s: %s" % (label, e))
                if e.status in dead_on:
                    self.mark_dead(label, str(e)[:200])
                elif e.status in skip_on:
                    continue          # rate limited: try the next key this time
                else:
                    raise
        raise NoUsableKey("%s: no usable key left (%s)" % (
            self.service, "; ".join(errors) or "all keys disabled: %s" % self.dead))
