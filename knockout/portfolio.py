"""Start portfolio (docs/knockout_portfolio_brief.md, sections 4.1 to 4.3): start sets and per-start aggregation.

Trusted code, run in the orchestrator. A program of the `neighbour` or `cold` regime runs once per start of a start
set; its regime score is the mean of the frozen-verifier scores over the starts, and that mean is what selection
uses. The experiment line keeps the best per-start candidate (`Evaluation.score` and `.candidate`), so the ledger's
`score` always matches a stored candidate; the mean goes to `config.regime_score`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from knockout import decisions as D
from knockout.context import Evaluation

REGIMES = ("record", "neighbour", "cold")


@dataclass(frozen=True)
class StartSet:
    id: str                      # "record:<epoch>", "cold:0", "neighbour:<n>"; changes whenever a member changes
    regime: str                  # record | neighbour | cold
    starts: tuple                # canonical candidates (None = "no init")
    start_scores: tuple          # frozen-verifier score of each start (None if the start is not a valid candidate)
    origin: tuple                # per start: "incumbent", "perturb:<strength>", "archive:<cell>", "cold:<k>"

    def __len__(self) -> int:
        return len(self.starts)


@dataclass
class PortfolioEvaluation(Evaluation):
    """An Evaluation over a start set. `score`/`candidate` are those of the best valid per-start result."""
    regime_score: float | None = None
    per_start: list[dict] = field(default_factory=list)
    per_start_evals: list[Evaluation] = field(default_factory=list)


def selection_value(ev: Evaluation) -> float | None:
    """The value selection compares: the regime score over a start set, else the plain tuned score."""
    if isinstance(ev, PortfolioEvaluation):
        return ev.regime_score
    return ev.score


def aggregate(evs: list[Evaluation], sset: StartSet, worst: list[float | None], direction: str) -> PortfolioEvaluation:
    """Combine the per-start evaluations of one program on one start set.

    Rule for a start where the program gave no valid candidate (brief 4.3, written for either direction): the
    start's own score if the start is a valid candidate, else the worst valid score seen so far on that start of
    that start set (`worst[i]`), else nothing. If any start has no value at all, the program is invalid on this
    start set (it must return a valid candidate from every start, or at least from starts someone else solved).
    """
    assert len(evs) == len(sset.starts) == len(worst)
    per, vals = [], []
    for i, ev in enumerate(evs):
        if ev.valid:
            v, used = ev.score, "own"
        elif sset.start_scores[i] is not None:
            v, used = sset.start_scores[i], "start_score"
        elif worst[i] is not None:
            v, used = worst[i], "worst_seen"
        else:
            v, used = None, "none"
        per.append({"origin": sset.origin[i], "score": ev.score if ev.valid else None, "value": v, "rule": used,
                    "status": ev.status})
        vals.append(v)
    valid_evs = [e for e in evs if e.valid]
    best = None
    for e in valid_evs:
        if best is None or D.gain(e.score, best.score, direction) > 0:
            best = e
    base = best or evs[0]
    kw = {f.name: getattr(base, f.name) for f in fields(Evaluation)}
    kw["cost_usd"] = sum(e.cost_usd for e in evs)
    kw["elapsed_s"] = sum(e.elapsed_s for e in evs)
    kw["n_evals"] = sum(e.n_evals for e in evs)
    regime_score = None if any(v is None for v in vals) else sum(vals) / len(vals)
    out = PortfolioEvaluation(**kw, regime_score=regime_score, per_start=per, per_start_evals=list(evs))
    if best is not None and regime_score is None:
        out.valid, out.status, out.candidate = False, "invalid", None
        out.error = "no valid candidate on some start, and no fallback score for that start"
    return out


def update_worst(worst: list[float | None], evs: list[Evaluation], direction: str) -> list[float | None]:
    out = list(worst)
    for i, ev in enumerate(evs):
        if ev.valid and (out[i] is None or D.gain(ev.score, out[i], direction) < 0):
            out[i] = ev.score
    return out


def progress(score: float | None, mean_start: float | None, reference: float | None) -> float | None:
    """Normalized progress for the prompt and the report only (never for selection), clipped to [-1, 2]."""
    if score is None or mean_start is None or reference is None or reference == mean_start:
        return None
    return max(-1.0, min(2.0, (score - mean_start) / (reference - mean_start)))


def mean_start_score(sset: StartSet) -> float | None:
    vals = [v for v in sset.start_scores if v is not None]
    return sum(vals) / len(vals) if vals else None


def safe_score(problem, candidate: Any) -> float | None:
    if candidate is None:
        return None
    try:
        return float(problem.score(candidate))
    except Exception:  # noqa: BLE001  (an invalid start is allowed: the program must repair it)
        return None
