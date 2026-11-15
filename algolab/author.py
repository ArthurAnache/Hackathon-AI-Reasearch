"""Author and repair LLM-written optimizer plugins (up to 3 repair rounds). Default: offline / mock only.

Paid LLM calls require explicit confirm_llm=True AND an injected client. This module never reads API keys from
the environment by itself. Gate rejection reasons and child tracebacks are fed back as repair context.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from algolab import gate

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = ROOT / "algolab" / "plugins"
MAX_REPAIRS = 3

SYSTEM = """You write ONE Python plugin for Algorithm Lab.
Requirements:
- Define PARAMS = {...} and class Algorithm with __init__(self, params, rng) and improve(self, f, deadline).
- Optional: perturb(self, f, intensity) -> array.
- Import only from the allowlist: math, time, heapq, itertools, functools, bisect, collections, dataclasses,
  typing, numpy, scipy.*, algolab.kit.
- No open/exec/eval/__import__/socket/subprocess/os/pathlib/file I/O.
- improve must return a nonnegative float64 vector; the harness owns the clock and scoring.
- Use algolab.kit for autoconvolution, q_form, score, packet helpers, exact_step_scores.
Return ONLY the Python source, no markdown fences.
"""


def seed_hypotheses() -> list[dict]:
    """The three queued novel ideas from savant-exp1, mapped to draft plugins already in-tree."""
    return [
        {
            "hypothesis_id": "math-20261115-002-active-envelope-packets",
            "title": "Mass-normalized packet moves against the full maximum envelope",
            "template": "algolab_plugin",
            "plugin": "algolab/plugins/lp_packet_ascent.py",
            "role": "mathematical",
            "expected_gain": 1e-7,
        },
        {
            "hypothesis_id": "numerical-20261115-two-box-envelope-queue",
            "title": "Amortized two-box transfers with peak-envelope line search",
            "template": "algolab_plugin",
            "plugin": "algolab/plugins/two_box_envelope.py",
            "role": "numerical",
            "expected_gain": 3e-8,
        },
        {
            "hypothesis_id": "literature-dinkelbach-kelley-packet-exchange-v1",
            "title": "Dinkelbach–Kelley peak exchange in packet coordinates",
            "template": "algolab_plugin",
            "plugin": "algolab/plugins/dinkelbach_kelley.py",
            "role": "literature",
            "expected_gain": 5e-08,
        },
    ]


def admit_or_repair(src: str, *, client=None, confirm_llm: bool = False, max_repairs: int = MAX_REPAIRS,
                    context: dict | None = None) -> dict:
    """Run the gate; optionally ask an LLM to repair up to max_repairs times. Default path is offline."""
    history = []
    current = src
    for attempt in range(max_repairs + 1):
        reasons = gate.check_source(current)
        history.append({"attempt": attempt, "admitted": not reasons, "reasons": reasons,
                        "sha256": hashlib.sha256(current.encode()).hexdigest()})
        if not reasons:
            return {"status": "admitted", "source": current, "history": history,
                    "sha256": history[-1]["sha256"]}
        if attempt == max_repairs or not confirm_llm or client is None:
            return {"status": "rejected", "source": current, "history": history,
                    "reason": "gate failed" + ("" if confirm_llm else " (repair disabled: no --confirm-llm)")}
        prompt = {"system": SYSTEM, "user": json.dumps({
            "task": "repair plugin so it passes the Algorithm Lab gate",
            "gate_reasons": reasons,
            "context": context or {},
            "current_source": current,
        }, indent=2)}
        # Client is whatever SAVANT injects; we do not construct OpenAI clients here.
        current = client.complete(prompt)  # type: ignore[attr-defined]
        if not isinstance(current, str):
            current = str(current)
        if current.startswith("```"):
            current = current.strip("`")
            if current.startswith("python"):
                current = current[6:]
            current = current.strip()
    return {"status": "rejected", "source": current, "history": history, "reason": "exhausted repairs"}


def materialize_seed_plugins(out_dir: Path | None = None) -> list[dict]:
    """Admit the three queued drafts (no LLM). Writes a session manifest under runs/algolab/seeds/."""
    out_dir = Path(out_dir or (ROOT / "runs" / "algolab" / "seeds" /
                               time.strftime("%Y%m%dT%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in seed_hypotheses():
        path = ROOT / h["plugin"]
        src = path.read_text(encoding="utf-8")
        result = admit_or_repair(src, confirm_llm=False)
        row = {**h, **{k: result[k] for k in ("status", "sha256", "history") if k in result},
               "plugin_path": str(path)}
        rows.append(row)
        (out_dir / f"{path.stem}.admission.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows
