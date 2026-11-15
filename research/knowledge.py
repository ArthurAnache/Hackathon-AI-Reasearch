"""What the results so far say. Rebuilt from the experiment store on every proposal,
so the agent is stateless and a resumed run reasons exactly like an uninterrupted one.

Each experiment's rationale carries two tags written by FeedbackAgent:
    strategy=<name>; parent=<experiment_id>
`parent` is the best experiment the proposal started from, so every result can be
read as a controlled comparison "parent -> this change -> this score".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .space import INIT_KEY, Space

TAG = re.compile(r"\b(strategy|parent)=([\w\-.]+)")
FAILED = ("crash", "invalid", "void")
TIE_RTOL = 1e-12


@dataclass
class Obs:
    id: str
    config: dict
    x: np.ndarray                 # position in the unit cube
    utility: float | None         # +score when maximising, -score when minimising; None if failed
    score: float | None
    status: str
    strategy: str
    parent: str | None
    certificate: str | None       # certificate id (the only allowed warm start)
    tags: dict = field(default_factory=dict)


@dataclass
class KnobEffect:
    knob: str
    du: float          # signed move in unit space
    gain: float        # utility change vs the parent (positive = better)
    exp_id: str


class Knowledge:
    def __init__(self, problem, space: Space, records: list[dict]):
        self.problem, self.space = problem, space
        self.sign = -1.0 if problem.direction == "minimize" else 1.0
        self.obs: list[Obs] = []
        for r in records:
            tags = dict(TAG.findall(r.get("rationale") or ""))
            ok = r["status"] in ("keep", "discard") and r["score"] is not None
            self.obs.append(Obs(
                id=r["experiment_id"], config=r["config"] or {}, x=space.to_unit(r["config"] or {}),
                utility=self.sign * r["score"] if ok else None, score=r["score"] if ok else None,
                status=r["status"], strategy=tags.get("strategy", "manual"), parent=tags.get("parent"),
                certificate=r.get("certificate_path"), tags=tags))
        self.by_id = {o.id: o for o in self.obs}
        kept = [o for o in self.obs if o.status == "keep"]
        self.best: Obs | None = max(kept, key=lambda o: o.utility) if kept else None
        self.valid = [o for o in self.obs if o.utility is not None]
        self.failed = [o for o in self.obs if o.status in FAILED]

    # ------------------------------------------------------------ memory
    def tried(self, config: dict) -> bool:
        key = self.space.key(config)
        return any(self.space.key(o.config) == key for o in self.obs)

    def near_failure(self, config: dict, radius: float = 0.03) -> bool:
        """Close (L-inf, unit cube) to a config that crashed, was invalid or ran over budget."""
        x = self.space.to_unit(config)
        return any(np.max(np.abs(o.x - x)) <= radius for o in self.failed)

    def gain_vs_parent(self, o: Obs) -> float | None:
        p = self.by_id.get(o.parent or "")
        if o.utility is None or p is None or p.utility is None:
            return None
        return o.utility - p.utility

    # ------------------------------------------------------------ what has been learned
    def knob_effects(self) -> dict[str, list[KnobEffect]]:
        """Single-knob controlled comparisons: experiments that differ from their parent in one knob."""
        out: dict[str, list[KnobEffect]] = {k: [] for k in self.space.names}
        for o in self.obs:
            p = self.by_id.get(o.parent or "")
            if p is None or p.utility is None:
                continue
            diff = o.x - p.x
            moved = np.flatnonzero(np.abs(diff) > 1e-9)
            if len(moved) == 1:
                k = self.space.names[moved[0]]
                gain = -np.inf if o.utility is None else o.utility - p.utility
                if abs(gain) <= TIE_RTOL * max(1.0, abs(p.utility)):
                    gain = 0.0                     # a tie is not an improvement (float noise)
                out[k].append(KnobEffect(k, float(diff[moved[0]]), float(gain), o.id))
        return out

    def strategy_stats(self) -> dict[str, tuple[int, int]]:
        """strategy -> (tries, improvements over the best at the time)."""
        stats: dict[str, list[int]] = {}
        for o in self.obs:
            s = stats.setdefault(o.strategy, [0, 0])
            s[0] += 1
            s[1] += o.status == "keep"
        return {k: (v[0], v[1]) for k, v in stats.items()}

    def trust_radius(self) -> float:
        """Grows after improvements and shrinks after misses (a TuRBO-style trust region)."""
        r = 0.2
        for o in self.obs[1:]:
            r = min(0.5, r * 1.6) if o.status == "keep" else max(0.02, r * 0.75)
        return r

    def non_keep_streak(self) -> int:
        n = 0
        for o in reversed(self.obs):
            if o.status == "keep":
                break
            n += 1
        return n

    def warm_start_gains(self) -> list[float]:
        return [g for o in self.obs if o.strategy == "refine" and (g := self.gain_vs_parent(o)) is not None]

    # ------------------------------------------------------------ explanation
    def summary(self) -> str:
        if self.best is None:
            return "no kept result yet"
        parts = [f"best {self.best.id} score {self.best.score:.10g}", f"trust radius {self.trust_radius():.2f}"]
        for k, effs in self.knob_effects().items():
            if effs:
                good = [e for e in effs if e.gain > 0]
                trend = ("up helps" if good[-1].du > 0 else "down helps") if good else "no improving move yet"
                parts.append(f"{k}: {len(good)}/{len(effs)} moves helped ({trend})")
        stats = self.strategy_stats()
        parts.append("strategies " + ", ".join(f"{s} {w}/{n}" for s, (n, w) in stats.items()))
        return "; ".join(parts)
