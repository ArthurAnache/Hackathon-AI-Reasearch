"""Problem plug-in: EinsteinArena first-autocorrelation-inequality = Tao constant 1a, with the EinsteinArena target.

Definitions agree (checked 2026-11-14 against Arthur's copy of the EinsteinArena verifier,
origin/ws1-selection:selection/verifiers/first-autocorrelation-inequality.py): f >= 0 on N cells, dx = 0.5/N,
score = max(f*f) dx / (sum f dx)^2 = 2N max(f*f) / (sum f)^2, which is exactly tao1a.score. So the stages, executors and
exact checks are tao1a's; only the reference differs:

- reference: live #1 of GET https://einsteinarena.com/api/leaderboard?problem_id=2 (read-only, no key);
- claim threshold: #1 minus the problem's minImprovement (delta, read live from /api/problems/<slug>).

Nothing is ever submitted to EinsteinArena or anywhere else.
"""
from __future__ import annotations

import json
import urllib.request
from fractions import Fraction

from .tao1a import *  # noqa: F401,F403  (stages, executors, exact checks)
from .tao1a import fetch_reference as fetch_tao_reference  # noqa: F401

NAME = "ea-first-autocorrelation"
EA = "https://einsteinarena.com"
SLUG = "first-autocorrelation-inequality"
EA_PROBLEM_ID = 2
EA_DELTA = None          # set by fetch_reference (minImprovement, read live)


def _get(path):
    req = urllib.request.Request(EA + path, headers={"User-Agent": "autoloop/1.0"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def fetch_reference() -> dict:
    """EinsteinArena live #1 and delta; the Tao 1a reference is attached for information."""
    global EA_DELTA
    try:
        prob = _get(f"/api/problems/{SLUG}")
        board = _get(f"/api/leaderboard?problem_id={prob['id']}&limit=3")
        EA_DELTA = float(prob["minImprovement"])
        top = board[0]
        out = dict(ok=True, value=repr(float(top["bestScore"])), delta=EA_DELTA, leader=top["agentName"],
                   source=f"{EA}/api/leaderboard?problem_id={prob['id']}",
                   candidates=[dict(value=repr(float(b["bestScore"])), agent=b["agentName"]) for b in board])
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, error=repr(e)[:300])
    try:
        t = fetch_tao_reference()
        out["tao_1a"] = dict(value=t.get("value"), source=t.get("source"))
    except Exception:  # noqa: BLE001
        pass
    return out


def claim_threshold(ref_value: str) -> Fraction:
    """Beat the live #1 by at least delta (EinsteinArena's minImprovement)."""
    delta = EA_DELTA if EA_DELTA is not None else float(_get(f"/api/problems/{SLUG}")["minImprovement"])
    return Fraction(ref_value) - Fraction(delta)
