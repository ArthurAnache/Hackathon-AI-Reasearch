"""One read-only handle over the toy fixture and the frozen WS1 catalog (problems/<slug>/).

The loop never wraps or patches the frozen verifier: `score` and `verify` call the catalog functions as they are.
This module only adds the bits the loop needs around them (direction-aware comparison, the minimum margin,
the reference to a sandbox-side guide score, and the warm-start certificate).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent

TOY = "toy-circles-26"


@dataclass
class ProblemHandle:
    id: str
    direction: str                         # "maximize" | "minimize"
    min_margin: float                      # the problem's minimum gain for a record (delta)
    verify: Callable[[Any], bool]
    score: Callable[[Any], float]          # raises the problem's InvalidSolution
    invalid_exc: type[Exception]
    guide_ref: str                         # "module:function" importable inside the sandbox
    guide_args: list = field(default_factory=list)
    best_known: Callable[[], float] = lambda: math.nan
    best_known_source: str = ""
    independent: Callable[[Any], dict] | None = None   # {"score", "fraction", "tolerance_abs", "method"}
    initial_candidate: Callable[[], Any] | None = None
    beats_record_fn: Callable[[float], bool] | None = None
    evidence_urls: list[str] = field(default_factory=list)

    def sign(self) -> float:
        return 1.0 if self.direction == "maximize" else -1.0

    def gain(self, new: float, old: float) -> float:
        """Signed improvement of `new` over `old` (positive = better)."""
        return self.sign() * (new - old)

    def improves(self, new: float | None, old: float | None, eps_rel: float) -> bool:
        if new is None:
            return False
        if old is None:
            return True
        return self.gain(new, old) > eps_rel * max(1.0, abs(old))

    def beats_record(self, score: float | None) -> bool:
        if score is None:
            return False
        if self.beats_record_fn is not None:
            return bool(self.beats_record_fn(score))
        best = self.best_known()
        g = self.gain(score, best)
        return g >= self.min_margin and g > 0


def _toy() -> ProblemHandle:
    from knockout import toy

    def independent(candidate) -> dict:
        frac = toy.exact_check(candidate)
        return {"score": float(frac), "fraction": frac, "tolerance_abs": 0.0, "method": "exact Fraction check"}

    return ProblemHandle(
        id=TOY, direction="maximize", min_margin=1e-6, verify=toy.verify, score=toy.score,
        invalid_exc=toy.InvalidSolution, guide_ref="knockout.toy:guide_score",
        best_known=lambda: toy.REFERENCE_SCORE, best_known_source=toy.REFERENCE_SOURCE,
        independent=independent, initial_candidate=None,
    )


def _catalog(slug: str) -> ProblemHandle:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from problems.catalog import load

    p = load(slug)
    return ProblemHandle(
        id=p.SLUG, direction=p.DIRECTION, min_margin=float(p.DELTA), verify=p.verify, score=p.score,
        invalid_exc=p.InvalidSolution if hasattr(p, "InvalidSolution") else ValueError,
        guide_ref="knockout.problems_api:catalog_guide", guide_args=[slug],
        best_known=lambda: float(p.best_known_score()["value"]),
        best_known_source=f"EinsteinArena live #1 for {slug}",
        independent=p.independent_score, initial_candidate=p.certificate,
        beats_record_fn=lambda s: p.beats_record(s), evidence_urls=[p.URL],
    )


def load_problem(problem_id: str) -> ProblemHandle:
    return _toy() if problem_id in (TOY, "toy") else _catalog(problem_id)


_GUIDES: dict[str, Any] = {}


def catalog_guide(candidate, slug: str) -> float:
    """Guide score inside the sandbox for a frozen problem: the pinned verifier itself (re-checked by the parent)."""
    if slug not in _GUIDES:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from problems.catalog import load

        _GUIDES[slug] = load(slug)
    return _GUIDES[slug].score(candidate)
