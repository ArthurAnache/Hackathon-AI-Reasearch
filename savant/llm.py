"""Configurable LLM provider. Keys never logged. Mock provider for offline tests.

Spending is persisted per session (spend_ledger.json). Before every request the worst-case cost
(estimated input + the full output allowance, reasoning included) is reserved on disk, so a crash or a
kill mid-request is still counted at the next start.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from savant.config import LLMConfig, env_llm_config, load_dotenv, require_openai_key
from savant.schemas import HYPOTHESIS_JSON_SCHEMA, ensure
from savant.templates import default_hypotheses_for_mock


def request_cost(prices: dict, *, input_tokens: int, cached_tokens: int = 0, output_tokens: int = 0,
                 cache_write_tokens: int = 0) -> float:
    """USD for one request. cached and cache-write tokens are subsets of input_tokens."""
    long_at = prices.get("long_context_tokens")
    long = bool(long_at) and input_tokens > long_at
    mi = prices.get("long_input_mult", 1.0) if long else 1.0
    mo = prices.get("long_output_mult", 1.0) if long else 1.0
    cached = min(max(0, cached_tokens), input_tokens)
    writes = min(max(0, cache_write_tokens), input_tokens - cached)
    uncached = input_tokens - cached - writes
    usd = (uncached * prices["input"] * mi + cached * prices["cached_input"] * mi
           + writes * prices.get("cache_write", prices["input"]) * mi
           + output_tokens * prices["output"] * mo)
    return usd / 1_000_000


@dataclass
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    usd: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, inp: int, out: int, cfg: LLMConfig, *, cached: int = 0, reasoning: int = 0,
            usd: float | None = None, effort: str | None = None) -> float:
        cost = usd if usd is not None else request_cost(cfg.prices(), input_tokens=inp, cached_tokens=cached,
                                                        output_tokens=out)
        self.calls += 1
        self.input_tokens += inp
        self.cached_input_tokens += cached
        self.output_tokens += out
        self.reasoning_tokens += reasoning
        self.usd += cost
        slot = self.by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0,
                                                "output_tokens": 0, "reasoning_tokens": 0, "usd": 0.0,
                                                "reasoning_effort": effort})
        slot["calls"] += 1
        slot["input_tokens"] += inp
        slot["cached_input_tokens"] = slot.get("cached_input_tokens", 0) + cached
        slot["output_tokens"] += out
        slot["reasoning_tokens"] = slot.get("reasoning_tokens", 0) + reasoning
        slot["usd"] += cost
        return cost

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BudgetExceeded(RuntimeError):
    pass


class LLMCallFailed(RuntimeError):
    """One request failed (incomplete, refusal, API error). Its cost is already charged. Not retried."""


class LLMUnavailable(LLMCallFailed):
    """The API could not be reached. The caller pauses and resumes from memory; nothing is retried blindly."""


def _is_connectivity_error(exc: Exception) -> bool:
    name = type(exc).__name__
    return name in ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout", "ConnectTimeout",
                    "RemoteProtocolError", "InternalServerError") or isinstance(exc, (ConnectionError, TimeoutError))


class LLMClient:
    def __init__(self, cfg: LLMConfig | None = None, *, max_usd: float = 5.0,
                 max_tokens: int = 500_000, allow_mock: bool = True, ledger_path: Path | None = None):
        load_dotenv()
        self.cfg = cfg if cfg is not None else env_llm_config()
        self.max_usd = max_usd
        self.max_tokens = max_tokens
        self.usage = LLMUsage()
        self.allow_mock = allow_mock
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.prior: dict = {"usd": 0.0, "tokens": 0, "entries": []}
        self._init_runtime()
        if self.cfg.provider == "mock" or (allow_mock and not os.environ.get("OPENAI_API_KEY")):
            self.cfg.provider = "mock"
        elif self.cfg.provider == "openai":
            require_openai_key(allow_mock=False)
        self._load_ledger()

    def _init_runtime(self) -> None:
        """Thread-safe state: several council requests of one stage may be in flight at once."""
        self._lock = threading.RLock()
        self._tls = threading.local()
        self._pending: dict[str, dict] = {}
        self._adoptable: dict[tuple, dict] = {}

    # Per-thread view of the request in progress.
    @property
    def last_meta(self) -> dict | None:
        return getattr(self._tls, "last_meta", None)

    @last_meta.setter
    def last_meta(self, value: dict | None) -> None:
        self._tls.last_meta = value

    @property
    def _accepted_id(self) -> str | None:
        return getattr(self._tls, "accepted_id", None)

    @_accepted_id.setter
    def _accepted_id(self, value: str | None) -> None:
        self._tls.accepted_id = value

    @property
    def _effort(self) -> str | None:
        return getattr(self._tls, "effort", None)

    @_effort.setter
    def _effort(self, value: str | None) -> None:
        self._tls.effort = value

    # ------------------------------------------------------------------ persistent spending
    def _load_ledger(self) -> None:
        if not hasattr(self, "_lock"):
            self._init_runtime()
        self._adoptable = {}
        if not self.ledger_path or not self.ledger_path.exists():
            return
        data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        pending = data.get("pending") or []
        if isinstance(pending, dict):
            pending = [pending]
        for p in pending:
            # A request was in flight when the process died: count its full reservation as spent.
            data.setdefault("entries", []).append({**p, "kind": "orphaned_reservation", "usd": p["reserved_usd"]})
            if p.get("response_id"):
                # Background responses keep running server-side: the next call for the same stage polls it
                # instead of paying for a new one.
                self._adoptable[(p.get("model"), p.get("schema"), p.get("purpose"))] = p
        self.prior = {
            "usd": sum(e.get("usd", 0.0) for e in data.get("entries", [])),
            "tokens": sum(e.get("input_tokens", 0) + e.get("output_tokens", 0) for e in data.get("entries", [])),
            "entries": data.get("entries", []),
        }
        self._write_ledger()

    def _write_ledger(self, *_ignored) -> None:
        if not self.ledger_path:
            return
        with self._lock:
            self._write_ledger_locked()

    def _write_ledger_locked(self) -> None:
        entries = self.prior["entries"]
        by_model: dict = {}
        for e in entries:
            slot = by_model.setdefault(e.get("model", "?"), {"calls": 0, "usd": 0.0, "input_tokens": 0,
                                                             "cached_input_tokens": 0, "output_tokens": 0,
                                                             "reasoning_tokens": 0, "reasoning_effort": None})
            slot["calls"] += e.get("calls", 1)
            for k in ("usd", "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"):
                slot[k] += e.get(k, 0) or 0
            slot["reasoning_effort"] = e.get("reasoning_effort") or slot["reasoning_effort"]
        data = {
            "max_usd": self.max_usd, "max_tokens": self.max_tokens,
            "spent_usd": round(self.spent_usd(), 6), "spent_tokens": self.spent_tokens(),
            "by_model": by_model, "pending": list(self._pending.values()) or None, "entries": entries,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = self.ledger_path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.ledger_path)

    def _set_pending(self, key: str, pending: dict | None) -> None:
        with self._lock:
            if pending is None:
                self._pending.pop(key, None)
            else:
                self._pending[key] = pending
            self._write_ledger()

    def spent_usd(self) -> float:
        with self._lock:
            return sum(e.get("usd", 0.0) for e in self.prior["entries"])

    def reserved_usd(self) -> float:
        with self._lock:
            return sum(p.get("reserved_usd", 0.0) for p in self._pending.values())

    def spent_tokens(self) -> int:
        with self._lock:
            return sum(e.get("input_tokens", 0) + e.get("output_tokens", 0) for e in self.prior["entries"])

    def remaining_usd(self) -> float:
        return self.max_usd - self.spent_usd() - self.reserved_usd()

    def record_legacy(self, entry: dict) -> None:
        """Book spending that happened before this ledger existed (conservative estimate)."""
        with self._lock:
            self.prior["entries"].append({**entry, "kind": "legacy_estimate"})
            self._write_ledger()

    # ------------------------------------------------------------------ budget checks
    def estimate_input_tokens(self, system: str, user: str, schema: dict | None) -> int:
        chars = len(system) + len(user) + len(json.dumps(schema or {}))
        return int(math.ceil(chars / 3.0)) + 200  # ~4 chars/token in English; /3 is conservative

    def reserve(self, model: str, est_input: int) -> float:
        """Check the caps, counting every request still in flight at its full reservation."""
        reserved = request_cost(self.cfg.prices(), input_tokens=est_input,
                                output_tokens=self.cfg.max_output_tokens)
        with self._lock:
            committed = self.spent_usd() + self.reserved_usd()
            if committed + reserved > self.max_usd + 1e-12:
                raise BudgetExceeded(
                    f"LLM USD cap: spent+in flight ${committed:.4f} + reservation ${reserved:.4f} for one {model} "
                    f"request > ${self.max_usd:.2f}")
            in_flight = sum(p.get("est_input_tokens", 0) + p.get("max_output_tokens", 0)
                            for p in self._pending.values())
            if self.spent_tokens() + in_flight + est_input + self.cfg.max_output_tokens > self.max_tokens:
                raise BudgetExceeded(f"LLM token cap {self.max_tokens} would be exceeded by one request")
        return reserved

    def _book(self, model: str, inp: int, out: int, *, cached: int = 0, reasoning: int = 0,
              cache_write: int = 0, meta: dict | None = None, effort: str | None = None,
              key: str | None = None) -> float:
        """Record the real cost and release the request's reservation in one ledger write."""
        effort = effort or self.cfg.reasoning_effort
        usd = request_cost(self.cfg.prices(), input_tokens=inp, cached_tokens=cached, output_tokens=out,
                           cache_write_tokens=cache_write)
        with self._lock:
            self.usage.add(model, inp, out, self.cfg, cached=cached, reasoning=reasoning, usd=usd, effort=effort)
            self.prior["entries"].append({
                "kind": "request", "model": model, "reasoning_effort": effort,
                "input_tokens": inp, "cached_input_tokens": cached, "cache_write_tokens": cache_write,
                "output_tokens": out, "reasoning_tokens": reasoning, "usd": usd, "calls": 1,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(meta or {}),
            })
            if key is not None:
                self._pending.pop(key, None)
            self._write_ledger()
        return usd

    def _charge(self, model: str, inp: int, out: int) -> None:
        """Mock path: same checks, same ledger."""
        if self.spent_tokens() + inp + out > self.max_tokens:
            raise BudgetExceeded(f"LLM token cap {self.max_tokens} exceeded")
        cost = request_cost(self.cfg.prices(), input_tokens=inp, output_tokens=out)
        if self.spent_usd() + cost > self.max_usd:
            raise BudgetExceeded(f"LLM USD cap ${self.max_usd} exceeded")
        self._book(model, inp, out)

    # ------------------------------------------------------------------ calls
    def complete_json(self, system: str, user: str, *, schema_name: str = "hypothesis",
                      schema: dict | None = None, purpose: str | None = None) -> dict:
        if self.cfg.provider == "mock":
            out = self._mock_json(system, user, schema_name)
            self.last_meta = {"model": "mock", "reasoning_effort": self.cfg.reasoning_effort,
                              "schema": schema_name, "purpose": purpose}
            return out
        if self.cfg.provider == "openai":
            return self._openai_json(system, user, schema_name, schema or HYPOTHESIS_JSON_SCHEMA, purpose)
        raise ValueError(f"unsupported provider {self.cfg.provider}")

    def _openai_client(self):
        from openai import OpenAI
        # max_retries=0: an SDK retry would silently send (and bill) the request again.
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=self.cfg.request_timeout_s)

    def _openai_json(self, system: str, user: str, schema_name: str, schema: dict,
                     purpose: str | None) -> dict:
        model = self.cfg.model
        effort = self.cfg.effort_for(purpose)
        client = self._openai_client()
        adopted = self._adopt(client, model, schema_name, purpose)
        if adopted is not None:
            return adopted
        est = self.estimate_input_tokens(system, user, schema)
        key = uuid.uuid4().hex
        with self._lock:  # check and register atomically: concurrent requests cannot overshoot the cap together
            reserved = self.reserve(model, est)
            pending = {"model": model, "reasoning_effort": effort, "schema": schema_name, "purpose": purpose,
                       "est_input_tokens": est, "max_output_tokens": self.cfg.max_output_tokens,
                       "reserved_usd": reserved, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            self._set_pending(key, pending)
        t0 = time.time()
        self._accepted_id = None
        self._effort = effort
        try:
            if self.cfg.is_reasoning:
                resp = self._responses_call(client, system, user, schema_name, schema, pending, effort=effort,
                                            key=key)
            else:
                resp = self._chat_call(client, system, user, schema_name, schema)
        except Exception as exc:  # noqa: BLE001  error before a usable response: keep it booked conservatively
            wall = time.time() - t0
            # Accepted by the server, or connected long enough to have reasoned: book the full reservation.
            if self._accepted_id or wall > 30:
                usd_out = self.cfg.max_output_tokens
            else:
                usd_out = 0
            self._book(model, est, usd_out, effort=effort, key=key,
                       meta={"schema": schema_name, "purpose": purpose, "status": "error",
                             "error": type(exc).__name__, "wall_s": round(wall, 1),
                             "response_id": self._accepted_id,
                             "conservative": "full reservation" if usd_out else "input only"})
            if _is_connectivity_error(exc):
                raise LLMUnavailable(f"{model} unreachable: {type(exc).__name__}: {str(exc)[:200]}") from exc
            raise LLMCallFailed(f"{model} request failed: {type(exc).__name__}: {str(exc)[:300]}") from exc
        return self._finish(resp, model, schema_name, purpose, time.time() - t0, key=key)

    def _adopt(self, client, model: str, schema_name: str, purpose: str | None) -> dict | None:
        """Finish a request a previous process left in flight, if this call is the same stage."""
        with self._lock:
            orphan = self._adoptable.pop((model, schema_name, purpose), None)
        if not orphan:
            return None
        rid = orphan["response_id"]
        try:
            resp = client.responses.retrieve(rid)
        except Exception:  # noqa: BLE001  expired or unreachable: the orphan stays booked at full reservation
            return None
        key = uuid.uuid4().hex
        with self._lock:
            entries = self.prior["entries"]
            idx = next((i for i in range(len(entries) - 1, -1, -1)
                        if entries[i].get("kind") == "orphaned_reservation"
                        and entries[i].get("response_id") == rid), None)
            if idx is None:
                return None
            entries.pop(idx)  # the pending reservation below covers it until the real usage is booked
            pending = {k: v for k, v in orphan.items() if k != "kind"}
            pending["adopted"] = True
            self._set_pending(key, pending)
        self._accepted_id = rid
        self._effort = orphan.get("reasoning_effort")
        t0 = time.time()
        try:
            resp = self._poll(client, resp)
        except Exception as exc:  # noqa: BLE001
            self._book(model, orphan.get("est_input_tokens", 0), self.cfg.max_output_tokens, key=key,
                       effort=self._effort, meta={"schema": schema_name, "purpose": purpose, "status": "error",
                                                  "error": type(exc).__name__, "response_id": rid,
                                                  "adopted": True, "conservative": "full reservation"})
            if _is_connectivity_error(exc):
                raise LLMUnavailable(f"{model} unreachable while adopting {rid}") from exc
            raise LLMCallFailed(f"adopted response {rid} failed: {type(exc).__name__}") from exc
        return self._finish(resp, model, schema_name, purpose, time.time() - t0, key=key)

    POLL_S = 5.0

    def _responses_call(self, client, system, user, schema_name, schema, pending: dict | None = None, *,
                        effort: str | None = None, key: str | None = None):
        """Background mode: the server keeps reasoning if our connection drops; polling is free to retry."""
        effort = effort or self.cfg.reasoning_effort
        kwargs: dict[str, Any] = dict(
            model=self.cfg.model,
            input=[{"role": "developer", "content": system}, {"role": "user", "content": user}],
            max_output_tokens=self.cfg.max_output_tokens,
            text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
            store=False,
            background=True,
        )
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        resp = client.responses.create(**kwargs)
        self._accepted_id = getattr(resp, "id", None)
        if pending is not None and self._accepted_id and key is not None:
            self._set_pending(key, {**pending, "response_id": self._accepted_id})
        return self._poll(client, resp)

    def _poll(self, client, resp):
        deadline = time.time() + self.cfg.request_timeout_s
        misses = 0
        while getattr(resp, "status", "completed") in ("queued", "in_progress"):
            if time.time() > deadline:
                try:
                    client.responses.cancel(self._accepted_id)
                except Exception:  # noqa: BLE001
                    pass
                raise TimeoutError(f"background response {self._accepted_id} exceeded {self.cfg.request_timeout_s}s")
            time.sleep(self.POLL_S)
            try:
                resp = client.responses.retrieve(self._accepted_id)
                misses = 0
            except Exception as exc:  # noqa: BLE001  a lost poll costs nothing; keep polling until the deadline
                misses += 1
                if not _is_connectivity_error(exc) and misses > 3:
                    raise
                time.sleep(min(60.0, self.POLL_S * 2 ** min(misses, 4)))
        return resp

    def wait_for_api(self, max_wait_s: float = 900.0) -> bool:
        """Free connectivity probe (model retrieve) with backoff."""
        if self.cfg.provider != "openai":
            return True
        t0, delay = time.time(), 5.0
        while time.time() - t0 < max_wait_s:
            try:
                self._openai_client().models.retrieve(self.cfg.model)
                return True
            except Exception:  # noqa: BLE001
                time.sleep(delay)
                delay = min(60.0, delay * 2)
        return False

    def _chat_call(self, client, system, user, schema_name, schema):
        return client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_output_tokens,
            response_format={"type": "json_schema",
                             "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )

    def _finish(self, resp, model: str, schema_name: str, purpose: str | None, wall_s: float, *,
                key: str | None = None) -> dict:
        usage = getattr(resp, "usage", None)
        if hasattr(resp, "output"):  # Responses API
            inp = getattr(usage, "input_tokens", 0) or 0
            out = getattr(usage, "output_tokens", 0) or 0
            if usage is None and getattr(resp, "status", "completed") != "completed":
                out = self.cfg.max_output_tokens  # no usage reported for a failed run: assume the worst
            idet = getattr(usage, "input_tokens_details", None)
            odet = getattr(usage, "output_tokens_details", None)
            cached = getattr(idet, "cached_tokens", 0) or 0
            cache_write = getattr(idet, "cache_write_tokens", 0) or 0
            reasoning = getattr(odet, "reasoning_tokens", 0) or 0
            status = getattr(resp, "status", "completed")
            text = getattr(resp, "output_text", None) or ""
            refusal = None
            for item in getattr(resp, "output", []) or []:
                for part in getattr(item, "content", None) or []:
                    if getattr(part, "type", "") == "refusal":
                        refusal = getattr(part, "refusal", "refused")
            resp_id = getattr(resp, "id", None)
        else:  # Chat Completions
            inp = getattr(usage, "prompt_tokens", 0) or 0
            out = getattr(usage, "completion_tokens", 0) or 0
            pdet = getattr(usage, "prompt_tokens_details", None)
            cdet = getattr(usage, "completion_tokens_details", None)
            cached = getattr(pdet, "cached_tokens", 0) or 0
            cache_write = 0
            reasoning = getattr(cdet, "reasoning_tokens", 0) or 0
            choice = resp.choices[0]
            status = "incomplete" if getattr(choice, "finish_reason", "stop") == "length" else "completed"
            text = choice.message.content or ""
            refusal = getattr(choice.message, "refusal", None)
            resp_id = getattr(resp, "id", None)
        meta = {"schema": schema_name, "purpose": purpose, "status": status, "response_id": resp_id,
                "wall_s": round(wall_s, 1)}
        effort = getattr(self, "_effort", None) or self.cfg.reasoning_effort
        usd = self._book(model, inp, out, cached=cached, reasoning=reasoning, cache_write=cache_write, meta=meta,
                         effort=effort, key=key)
        self.last_meta = {"model": model, "reasoning_effort": effort, "schema": schema_name,
                          "purpose": purpose, "response_id": resp_id, "input_tokens": inp,
                          "cached_input_tokens": cached, "output_tokens": out, "reasoning_tokens": reasoning,
                          "usd": round(usd, 6), "wall_s": round(wall_s, 1)}
        if refusal:
            raise LLMCallFailed(f"{model} refused ({schema_name})")
        if status != "completed":
            raise LLMCallFailed(f"{model} response {status} ({schema_name}); output allowance "
                                f"{self.cfg.max_output_tokens} tokens, reasoning used {reasoning}")
        try:
            return json.loads(text or "{}")
        except json.JSONDecodeError as exc:
            raise LLMCallFailed(f"{model} returned non-JSON for {schema_name}") from exc

    def _mock_json(self, system: str, user: str, schema_name: str) -> dict:
        # Deterministic role-specific outputs for offline tests
        approx_in = max(1, (len(system) + len(user)) // 4)
        approx_out = 400
        self._charge("mock", approx_in, approx_out)
        role = None
        # Prefer the explicit role= tag we append to the system prompt (user JSON may mention other roles).
        sys_l = system.lower()
        for r in ("mathematical", "numerical", "evolutionary", "statistical", "literature"):
            if f"role={r}" in sys_l or f"you are the {r} savant" in sys_l:
                role = r
                break
        if schema_name == "hypothesis":
            for h in default_hypotheses_for_mock():
                if h["role"] == role:
                    return ensure("hypothesis", copy_h(h))
            return ensure("hypothesis", copy_h(default_hypotheses_for_mock()[0]))
        if schema_name in ("critique", "critique_batch"):
            import re
            targets = re.findall(r'"hypothesis_id":\s*"([^"]+)"', user) or ["H-unknown"]
            items = [ensure("critique", {
                "reviewer_role": role or "statistical",
                "target_id": t,
                "validity": 0.7, "novelty": 0.5, "expected_gain": 0.5,
                "compute_cost": 0.4, "testability": 0.8,
                "objections": ["Mock critique: verify against matched-seed evidence."],
                "score": 0.62,
            }) for t in dict.fromkeys(targets)]
            return {"critiques": items} if schema_name == "critique_batch" else items[0]
        if schema_name == "revision":
            return ensure("revision", {
                "hypothesis_id": "H-mock", "role": role or "numerical",
                "response": "Acknowledged; keeping template within vetted bounds.",
                "revised": False, "changes": [], "hypothesis": None,
            })
        if schema_name == "vote":
            ranking = [h["hypothesis_id"] for h in default_hypotheses_for_mock()
                       if h["role"] != role and h["template"] != "queue_novel"]
            ranking += [h["hypothesis_id"] for h in default_hypotheses_for_mock()
                        if h["role"] != role and h["template"] == "queue_novel"]
            return ensure("vote", {
                "voter_role": role or "statistical",
                "ranking": ranking[:4],
                "rationale": "Mock vote prefers executable, diversified templates.",
            })
        if schema_name == "stop_vote":
            return ensure("stop_vote", {
                "role": role or "statistical",
                "decision": "MORE_REPLICATION" if role == "statistical" else "CONTINUE",
                "additional_jobs": 3 if role == "statistical" else 6,
                "rationale": "Mock stop vote.",
                "expected_information_gain": 0.4,
                "remaining_budget_note": "Within caps.",
                "selected_hypothesis_ids": [],
            })
        if schema_name == "manager":
            import re
            ids = list(dict.fromkeys(re.findall(r'"hypothesis_id":\s*"([^"]+)"', user)))
            return {"selected_ids": ids[:4], "seeds_per_strategy": 10, "budget_s_request": 60,
                    "rationale": "Mock manager keeps the council order.", "rejections": []}
        return {"ok": True, "schema_name": schema_name}


def copy_h(h: dict) -> dict:
    return json.loads(json.dumps(h))
