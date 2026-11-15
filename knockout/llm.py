"""One door to the LLM provider (OpenAI): JSON answers, response cache, cost accounting, `llm_call` events.

- Key: OPENAI_API_KEY from the environment or the repo's .env, read in the trusted process only, never logged.
- Cache: every answer is stored under runs/<run_id>/llm_cache/<request_hash>.json (section 6). The hash covers the
  model, both prompts, the schema, the reasoning effort and the seed of the call, so `knockout.replay` can check
  every recorded decision against it without calling any API.
- Cost: token usage times the price of the model (`llm.prices`, or `llm.usd_per_mtok_*`), charged to the budget
  governor under the caller's category. A call whose worst-case cost does not fit the USD cap is not sent.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from knockout.problems_api import REPO_ROOT


class LLMUnavailable(RuntimeError):
    """No key, no SDK, or a configured model cannot be reached."""


def request_hash(model: str, system: str, prompt: str, schema: dict, effort, seed: int) -> str:
    blob = json.dumps({"model": model, "system": system, "prompt": prompt, "schema": schema, "effort": effort,
                       "seed": int(seed)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class LLM:
    def __init__(self, strategy, ledger, governor, client=None):
        self.strategy, self.ledger, self.governor = strategy, ledger, governor
        self.model = str(strategy["llm.model"])
        self.models = [str(m) for m in (strategy.get("llm.models") or [self.model])]
        self.effort = strategy.get("llm.reasoning_effort")
        self.max_output_tokens = int(strategy["llm.max_output_tokens"])
        self.timeout_s = float(strategy["llm.timeout_s"])
        self.cache_dir = Path(ledger.dir) / "llm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or self._client()
        self.budget_stopped = False       # True once a call was refused for the USD cap
        self.last_error: str | None = None
        self.calls = 0

    def _client(self):
        try:
            from dotenv import load_dotenv
            from openai import OpenAI
        except ImportError as exc:
            raise LLMUnavailable(f"missing package: {exc}") from exc
        load_dotenv(REPO_ROOT / ".env")
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise LLMUnavailable("OPENAI_API_KEY is not set (add it to .env; see .env.example)")
        return OpenAI(api_key=key, timeout=self.timeout_s, max_retries=2)

    def preflight(self) -> None:
        """Fail fast, before any compute is spent, when the key or a model name is wrong. Costs nothing."""
        for model in dict.fromkeys([self.model, *self.models]):
            try:
                self.client.models.retrieve(model)
            except Exception as exc:  # noqa: BLE001
                raise LLMUnavailable(f"cannot use model {model!r}: {type(exc).__name__}: {str(exc)[:300]}") from exc

    # ------------------------------------------------------------------ cost
    def prices(self, model: str) -> tuple[float, float]:
        table = self.strategy.get("llm.prices") or {}
        if model in table:
            return float(table[model][0]) / 1e6, float(table[model][1]) / 1e6
        return float(self.strategy["llm.usd_per_mtok_in"]) / 1e6, float(self.strategy["llm.usd_per_mtok_out"]) / 1e6

    def estimate_usd(self, system: str, prompt: str, model: str | None = None) -> float:
        """Worst case of one call: the prompt at ~3 characters per token plus a full-length answer."""
        p_in, p_out = self.prices(model or self.model)
        return (len(system) + len(prompt)) / 3.0 * p_in + self.max_output_tokens * p_out

    # ------------------------------------------------------------------ one call
    def call(self, role: str, system: str, prompt: str, schema: dict, *, seed: int, category: str,
             model: str | None = None, tags: dict | None = None, reserved: bool = False) -> tuple[dict | None, dict]:
        """Returns (parsed JSON answer or None, meta). Never raises for API or parsing errors.
        `reserved=True` means the caller already checked the USD cap for this call."""
        model = model or self.model
        h = request_hash(model, system, prompt, schema, self.effort, seed)
        meta = {"role": role, "model": model, "request_hash": h, **(tags or {})}
        path = self.cache_dir / f"{h}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            self.ledger.event("llm_call", seed=seed, ok=True, error=None, cached=True, input_tokens=0,
                              output_tokens=0, usd=0.0, latency_s=0.0, response_id=cached.get("response_id"), **meta)
            return json.loads(cached["text"]), {**meta, "cached": True, "usd": 0.0}
        if not reserved and not self.governor.can_spend_usd(category, self.estimate_usd(system, prompt, model)):
            self.budget_stopped = True
            return None, {**meta, "error": "usd cap"}
        kwargs = dict(model=model, instructions=system, input=prompt, max_output_tokens=self.max_output_tokens,
                      text={"format": {"type": "json_schema", "name": role, "strict": True, "schema": schema}})
        if self.effort:
            kwargs["reasoning"] = {"effort": self.effort}
        t0 = time.monotonic()
        usage_in = usage_out = 0
        error = response_id = text = data = None
        self.calls += 1
        try:
            resp = self.client.responses.create(**kwargs)
            response_id = getattr(resp, "id", None)
            usage = getattr(resp, "usage", None)
            usage_in = int(getattr(usage, "input_tokens", 0) or 0)
            usage_out = int(getattr(usage, "output_tokens", 0) or 0)
            text = resp.output_text
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("the answer is not a JSON object")
        except Exception as exc:  # noqa: BLE001  (API error, truncated or non-JSON answer: logged, not fatal)
            error = f"{type(exc).__name__}: {str(exc)[:400]}"
            data = None
        p_in, p_out = self.prices(model)
        usd = usage_in * p_in + usage_out * p_out
        self.governor.charge(category, llm_usd=usd)
        latency = round(time.monotonic() - t0, 2)
        if error is None:
            path.write_text(json.dumps({"request_hash": h, "role": role, "model": model, "seed": int(seed),
                                        "response_id": response_id, "input_tokens": usage_in,
                                        "output_tokens": usage_out, "text": text}))
        else:
            self.last_error = error
        self.ledger.event("llm_call", seed=seed, ok=error is None, error=error, cached=False,
                          input_tokens=usage_in, output_tokens=usage_out, usd=round(usd, 6), latency_s=latency,
                          response_id=response_id, **meta)
        return data, {**meta, "input_tokens": usage_in, "output_tokens": usage_out, "usd": round(usd, 6),
                      "latency_s": latency, "error": error}

    def cached_text(self, h: str) -> str | None:
        path = self.cache_dir / f"{h}.json"
        return json.loads(path.read_text())["text"] if path.exists() else None
