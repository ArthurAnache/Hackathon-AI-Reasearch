"""Budget governor (section 5.12): one object owns the wall clock and the USD cap.

Spend is tracked per category: "control" (Hill Sampling arm, fixed share), "main", "gate" (capped), "meta"
(strategist + scout, capped). USD covers compute (Modal) and LLM calls together, under one cap.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

CATEGORIES = ("control", "main", "gate", "meta")


class BudgetGovernor:
    def __init__(self, wall_clock_s: float, max_usd: float, shares: dict[str, float],
                 clock: Callable[[], float] = time.monotonic):
        self.wall_clock_s = float(wall_clock_s)
        self.max_usd = float(max_usd)
        self.control_share = float(shares.get("control", 0.2))
        self.gate_max = float(shares.get("gate_max", 0.25))
        self.meta_max = float(shares.get("meta_max", 0.02))
        self._clock = clock
        self._t0 = clock()
        self._lock = threading.Lock()
        self.compute_s = {c: 0.0 for c in CATEGORIES}
        self.usd = {c: 0.0 for c in CATEGORIES}
        self.llm_usd = {c: 0.0 for c in CATEGORIES}

    # ------------------------------------------------------------------ accounting
    def charge(self, category: str, compute_s: float = 0.0, usd: float = 0.0, llm_usd: float = 0.0) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"unknown budget category {category!r}")
        with self._lock:
            self.compute_s[category] += compute_s
            self.usd[category] += usd + llm_usd
            self.llm_usd[category] += llm_usd

    def elapsed_s(self) -> float:
        return self._clock() - self._t0

    def total_usd(self) -> float:
        return sum(self.usd.values())

    def total_compute_s(self) -> float:
        return sum(self.compute_s.values())

    def spend(self, category: str) -> float:
        """Spend in comparable units: USD when a cap is set, otherwise compute seconds."""
        return self.usd[category] if self.max_usd > 0 and self.total_usd() > 0 else self.compute_s[category]

    def total_spend(self) -> float:
        return sum(self.spend(c) for c in CATEGORIES)

    # ------------------------------------------------------------------ decisions
    def exhausted(self) -> bool:
        return self.elapsed_s() >= self.wall_clock_s or (self.max_usd > 0 and self.total_usd() >= self.max_usd)

    def fraction_used(self) -> float:
        """Share of the run already used: the larger of the wall-clock share and the USD share."""
        f = self.elapsed_s() / self.wall_clock_s if self.wall_clock_s > 0 else 1.0
        if self.max_usd > 0:
            f = max(f, self.total_usd() / self.max_usd)
        return min(1.0, f)

    def remaining_s(self) -> float:
        return max(0.0, self.wall_clock_s - self.elapsed_s())

    def control_owed(self) -> bool:
        """True while the control arm has received less than its fixed share (it can never be starved)."""
        total = self.total_spend()
        return total == 0 or self.spend("control") < self.control_share * total

    def within_cap(self, category: str) -> bool:
        cap = {"gate": self.gate_max, "meta": self.meta_max}.get(category)
        if cap is None:
            return True
        total = self.total_spend()
        return total == 0 or self.spend(category) < cap * total

    def can_spend_usd(self, category: str, estimate: float) -> bool:
        if self.max_usd <= 0:
            return estimate <= 0
        return self.total_usd() + estimate <= self.max_usd and self.within_cap(category)

    def summary(self) -> dict:
        return {"elapsed_s": round(self.elapsed_s(), 1), "wall_clock_s": self.wall_clock_s,
                "max_usd": self.max_usd, "total_usd": round(self.total_usd(), 4),
                "compute_s": {k: round(v, 1) for k, v in self.compute_s.items()},
                "usd": {k: round(v, 4) for k, v in self.usd.items()},
                "llm_usd": {k: round(v, 4) for k, v in self.llm_usd.items()}}
