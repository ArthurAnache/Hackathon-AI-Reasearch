"""Proposal strategies: each one turns Knowledge into candidate experiments.

A strategy is any object with `name` and `propose(kn, rng) -> list[Candidate]`.
To add one, write a class and pass it to FeedbackAgent(strategies=[...]).

Built in:
  CoordinateStep  per-knob step with adaptive size and direction (Rprop-like):
                  continue in a direction that helped, reverse and halve after one that hurt
  Surrogate       ridge regression (linear + diagonal quadratic) over all valid results;
                  maximises the prediction inside the trust region
  TrustRegion     random point in the trust region around the best (exploration)
  CombineWins     applies two single-knob moves that each helped, together
  EvidenceHints   configs suggested by evidence (WS4: Exa / past.dev) as JSON {"config": {...}}
  Refine          warm start: re-run the best config starting from the best solution found
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from .knowledge import Knowledge
from .space import INIT_KEY


@dataclass
class Candidate:
    config: dict
    strategy: str
    why: str
    gain: float | None = None      # predicted utility gain over the best (None = unknown)
    tries: int = 0                 # how often this direction/knob was tried already (tie-break: fewer first)


def _around_best(kn: Knowledge) -> dict:
    return {**kn.space.defaults, **kn.best.config}


class CoordinateStep:
    name = "coordinate"
    INITIAL = 0.12        # unit-cube step for a knob never moved (about x1.6 on a default x8 range)

    def propose(self, kn: Knowledge, rng) -> list[Candidate]:
        out, base = [], _around_best(kn)
        x0 = kn.space.to_unit(base)
        for i, (k, effs) in enumerate(kn.knob_effects().items()):
            if not effs:
                moves = [(+self.INITIAL, "never moved: probe upwards"), (-self.INITIAL, "never moved: probe downwards")]
                gain = None
            else:
                last = effs[-1]
                if last.gain > 0:
                    moves = [(np.sign(last.du) * abs(last.du) * 1.6,
                              f"{last.exp_id} moved it {'up' if last.du > 0 else 'down'} and improved: go further")]
                else:
                    moves = [(-np.sign(last.du) * abs(last.du) * 0.5,
                              f"{last.exp_id} moved it {'up' if last.du > 0 else 'down'} and got worse: reverse, half step")]
                # trust a slope only once this knob has produced an improvement
                slopes = [e.gain / e.du for e in effs if np.isfinite(e.gain) and abs(e.du) > 1e-9]
                gain = float(np.mean(slopes)) if slopes and any(e.gain > 0 for e in effs) else None
            for step, why in moves:
                x = x0.copy()
                x[i] = np.clip(x[i] + step, 0, 1)
                if abs(x[i] - x0[i]) < 1e-9:
                    continue
                cfg = kn.space.from_unit(x, base)
                out.append(Candidate(cfg, self.name, f"{k} {base[k]} -> {cfg[k]}: {why}",
                                     None if gain is None else gain * (x[i] - x0[i]), tries=len(effs)))
        return out


class Surrogate:
    name = "surrogate"
    SAMPLES = 2048
    RIDGE = 1e-3

    def propose(self, kn: Knowledge, rng) -> list[Candidate]:
        pts = kn.valid
        d = kn.space.dim
        if d == 0 or len(pts) < 2 * d + 2:
            return []
        X = np.array([o.x for o in pts])
        y = np.array([o.utility for o in pts])
        mu, sd = y.mean(), y.std() or 1.0
        feats = lambda Z: np.hstack([np.ones((len(Z), 1)), Z, Z ** 2])
        F = feats(X)
        w = np.linalg.solve(F.T @ F + self.RIDGE * np.eye(F.shape[1]), F.T @ ((y - mu) / sd))
        target = (y - mu) / sd
        r2 = 1 - ((F @ w - target) ** 2).sum() / max((target ** 2).sum(), 1e-12)
        r = kn.trust_radius()
        base = _around_best(kn)
        x0 = kn.space.to_unit(base)
        Z = np.clip(x0 + rng.uniform(-r, r, size=(self.SAMPLES, d)), 0, 1)
        pred = feats(Z) @ w * sd + mu
        i = int(np.argmax(pred))
        gain = float(pred[i] - kn.best.utility)
        if gain <= 0:
            return []
        cfg = kn.space.from_unit(Z[i], base)
        changed = ", ".join(f"{k} {base[k]} -> {cfg[k]}" for k in kn.space.names if cfg[k] != base[k])
        return [Candidate(cfg, self.name, f"quadratic model of {len(pts)} results (R^2 {r2:.2f}) predicts a gain: {changed}", gain)]


class TrustRegion:
    name = "explore"

    def propose(self, kn: Knowledge, rng) -> list[Candidate]:
        if kn.space.dim == 0:
            return []
        r = kn.trust_radius()
        base = _around_best(kn)
        x = np.clip(kn.space.to_unit(base) + rng.uniform(-r, r, size=kn.space.dim), 0, 1)
        cfg = kn.space.from_unit(x, base)
        return [Candidate(cfg, self.name, f"random point within trust radius {r:.2f} of the best", None)]


class CombineWins:
    name = "combine"

    def propose(self, kn: Knowledge, rng) -> list[Candidate]:
        wins = {k: max(effs, key=lambda e: e.gain) for k, effs in kn.knob_effects().items()
                if effs and max(e.gain for e in effs) > 0}
        if len(wins) < 2:
            return []
        base = _around_best(kn)
        x = kn.space.to_unit(base)
        for i, k in enumerate(kn.space.names):
            if k in wins:
                x[i] = np.clip(x[i] + wins[k].du, 0, 1)
        cfg = kn.space.from_unit(x, base)
        total = sum(e.gain for e in wins.values())
        src = ", ".join(f"{k} ({e.exp_id})" for k, e in wins.items())
        return [Candidate(cfg, self.name, f"each of these moves helped on its own: {src}; apply them together", total)]


class EvidenceHints:
    name = "evidence"
    PATTERN = re.compile(r"\{.*\}", re.S)

    def __init__(self):
        self.evidence = []

    def propose(self, kn: Knowledge, rng) -> list[Candidate]:
        out, base = [], _around_best(kn)
        for e in self.evidence:
            m = self.PATTERN.search(e.content or "")
            try:
                hint = json.loads(m.group(0)).get("config") if m else None
            except (json.JSONDecodeError, AttributeError):
                hint = None
            if isinstance(hint, dict):
                cfg = {**base, **{k: v for k, v in hint.items() if k in kn.space.knobs or k == INIT_KEY}}
                out.append(Candidate(cfg, self.name, f"suggested by {e.source} ({e.citation})", None))
        return out


class Refine:
    """Warm start: spend another budget refining the best solution found so far."""
    name = "refine"

    def propose(self, kn: Knowledge, rng) -> list[Candidate]:
        if not kn.best.solution_path:
            return []
        gains = kn.warm_start_gains()
        if len(gains) >= 2 and max(gains[-2:]) <= 0:
            return []                       # refining stopped paying off
        cfg = {**_around_best(kn), INIT_KEY: kn.best.solution_path}
        why = f"start from the best solution so far ({kn.best.id})"
        if gains:
            why += f"; last refinement gained {gains[-1]:.3g}"
        return [Candidate(cfg, self.name, why, gains[-1] if gains else None)]


def default_strategies(warm_start: bool) -> list:
    refine = [Refine()] if warm_start else []
    return [EvidenceHints(), *refine, CoordinateStep(), Surrogate(), CombineWins(), TrustRegion()]
