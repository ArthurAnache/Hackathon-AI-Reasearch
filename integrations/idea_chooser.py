"""The idea chooser: a bandit over ideas, above WS2's knob tuner.

    IdeaChooser       ranks the runnable ideas of the map by UCB1 on what our own
                      experiments returned, and never returns a dead end.
    IdeaSearchAgent   a ResearchAgent (interfaces.py contract): each proposal picks
                      an idea with the chooser, lets that idea's own knob tuner
                      (one FeedbackAgent per idea) pick the config, and tags the
                      rationale with idea=<id>.

Which ideas can be chosen:
- testable kinds only (worked, dead_end, untested_idea), on the problem(s) asked for;
- never a dead end: a dead_end node, or any idea our experiments refuted
  (ideas.is_dead). The store is re-read at every choice, so an idea refuted by the
  experiment that just finished is gone from the very next proposal;
- only ideas with a runner ({"optimizer": path, "config": {...}}): an idea nobody
  implemented cannot be run honestly, and running the default optimizer under its
  name would credit or blame the wrong idea. Such ideas stay on the frontier of the
  map as "no runner yet".

The bandit. One arm per idea, one pull per experiment (crashes included: they cost
compute). The reward of an experiment has two parts, both in [0, 0.5]:
- standing: how its score ranks among every valid run of the candidate ideas
  (0.5 x the share of the other runs it beats). This is what lets the bandit
  concentrate on the best idea while every idea is still below the record, which
  is the usual case: a verdict needs a gain over the record, a search does not.
  Without it every reward is 0 and UCB degenerates into round robin.
- gain: 0.5 + 0.5 * sqrt(min(1, gain/delta)) when the run is positive evidence
  (integrations.verdicts); with delta unknown, 1. The square root keeps a
  0.01-delta gain visible without letting it look like a record-sized one.
The standing is a search heuristic only: it never touches a verdict.
Exploration: c = 0.4, i.e. the classic 0.8 scaled to rewards that span 0.5 below
the record. After one experiment per idea the next one goes to the best idea; on
the first real comparison (4 ideas, 8 experiments) it then gets 4 of the 8. Score = mean reward + c * sqrt(ln(total pulls) / pulls).
Ideas never tried come first, in the order: claimed to have worked, then untested
ideas, then revived dead ends; ties keep the order of the map.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import ideas, verdicts

_PRIOR = {"worked": 0, "untested_idea": 1, "dead_end": 2}


class NoIdeaLeft(RuntimeError):
    """No runnable idea is left on the frontier."""


def reward(row: Dict[str, Any], standing: Optional[float] = None) -> float:
    """standing: share of the other runs this run beats, 0..1 (None: unknown, or not a valid run)."""
    if row["evidence"] == "positive":
        g = row["gain_over_delta"]
        return 1.0 if g is None else 0.5 + 0.5 * math.sqrt(min(1.0, max(0.0, g)))
    return 0.5 * standing if standing is not None else 0.0


def standings(rows_by_idea: Dict[str, List[Dict[str, Any]]], sign: float = 1.0) -> Dict[tuple, float]:
    """{(idea id, run key): share of the other valid runs it beats (a tie counts half)}."""
    scored = [(i, x["key"], sign * x["score"]) for i, rows in rows_by_idea.items() for x in rows
              if x["score"] is not None and x["evidence"] not in ("ambiguous", "copy")]
    out = {}
    for i, key, u in scored:
        if len(scored) < 2:
            out[(i, key)] = 0.5                         # alone: nothing to rank against
            continue
        beaten = sum(1.0 if u > v else 0.5 if u == v else 0.0 for j, k, v in scored if (j, k) != (i, key))
        out[(i, key)] = beaten / (len(scored) - 1)
    return out


def local_history(history: Iterable[Dict[str, Any]], idea_id: str, sign: float = 1.0,
                  problems: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """The experiments of one idea, as its knob tuner should see them: keep/discard
    and best_before are re-derived against the idea's OWN best. In the shared store
    an idea's first run is a "discard" whenever another idea already scored better,
    and a tuner that sees no kept result would repeat its baseline forever."""
    out, best, accepted = [], None, (set(problems) if problems is not None else None)
    for r in history:
        if ideas.idea_of(r) != idea_id or (accepted is not None and r.get("problem") not in accepted):
            continue
        r = dict(r)
        score = verdicts.valid_score(r)                # None for a crash, a NaN, a missing score
        if score is not None:
            r["best_before"] = best
            r["status"] = "keep" if best is None or sign * (score - best) > 0 else "discard"
            if r["status"] == "keep":
                best = score
        elif r.get("status") in verdicts.VALID:
            r["status"] = "invalid"                     # "keep" with no usable score: never the idea's best
        out.append(r)
    return out


class IdeaChooser:
    def __init__(self, store: ideas.IdeaStore, ref_for: Callable[[Dict[str, Any]], verdicts.Reference],
                 problems: Optional[Iterable[str]] = None, c: float = 0.4, need_runner: bool = True,
                 min_gain_over_delta: float = verdicts.MIN_GAIN_OVER_DELTA,
                 aliases: Optional[Dict[str, str]] = None):
        """problems: map slugs to choose from (None = every problem on the map).
        aliases: {experiment record problem id: map slug}, for WS2's short ids."""
        self.store, self.ref_for, self.c, self.need_runner = store, ref_for, c, need_runner
        self.problems = set(problems) if problems is not None else None    # an empty set means no problem
        self.aliases = aliases
        self.min_gain_over_delta = min_gain_over_delta

    def frontier(self) -> List[Dict[str, Any]]:
        """Every live, testable idea of the problem(s), runnable or not, in map order."""
        return [n for n in self.store.all()
                if ideas.is_frontier(n) and (self.problems is None or n["problem"] in self.problems)]

    def candidates(self, exclude: Iterable[str] = ()) -> List[Dict[str, Any]]:
        skip = set(exclude)
        return [n for n in self.frontier()
                if n["id"] not in skip and (n.get("runner") or not self.need_runner)]

    def ranking(self, history: List[Dict[str, Any]], exclude: Iterable[str] = ()) -> List[Dict[str, Any]]:
        """Candidates, best first: [{"id", "node", "pulls", "mean", "score", "why"}]."""
        cands = self.candidates(exclude)
        rows_by_idea = {n["id"]: verdicts.observe(n, history, self.ref_for(n), self.aliases,
                                                  self.min_gain_over_delta) for n in cands}
        rank = standings(rows_by_idea, self.ref_for(cands[0]).sign) if cands else {}
        stats = {i: (len(rows), sum(reward(x, rank.get((i, x["key"]))) for x in rows))
                 for i, rows in rows_by_idea.items()}
        total = sum(p for p, _ in stats.values()) + 1
        out = []
        for order, n in enumerate(cands):
            pulls, won = stats[n["id"]]
            if pulls == 0:
                score, mean, why = math.inf, None, "never tried (%s, status %s)" % (n["kind"].replace("_", " "), n["claimed_status"])
            else:
                mean = won / pulls
                score = mean + self.c * math.sqrt(math.log(total) / pulls)
                why = "mean reward %.3f over %d experiments, UCB %.3f" % (mean, pulls, score)
            out.append({"id": n["id"], "node": n, "pulls": pulls, "mean": mean, "score": score, "why": why,
                        "_key": (-score, _PRIOR.get(n["kind"], 9), order)})
        out.sort(key=lambda x: x.pop("_key"))
        return out

    def choose(self, history: List[Dict[str, Any]], exclude: Iterable[str] = ()) -> Optional[Dict[str, Any]]:
        ranked = self.ranking(history, exclude)
        return ranked[0] if ranked else None


class IdeaSearchAgent:
    """ResearchAgent over ideas. make_inner(node) builds the knob tuner of one idea
    (e.g. a FeedbackAgent on the idea's optimizer file); it is built once per idea."""

    def __init__(self, store: ideas.IdeaStore, chooser: IdeaChooser, make_inner: Callable[[Dict[str, Any]], Any],
                 name: str = "ideas", sign: float = 1.0, record_problems: Optional[Iterable[str]] = None):
        """record_problems: the `problem` values of this run's experiment records; the
        knob tuner of an idea only sees experiments of the idea on those."""
        self.store, self.chooser, self.make_inner, self.name, self.sign = store, chooser, make_inner, name, sign
        self.record_problems = set(record_problems) if record_problems is not None else None
        self._inner: Dict[str, Any] = {}
        self.paused: set = set()         # ideas whose knob tuner asked to stop, for the rest of this run
        self._best: Dict[str, Optional[float]] = {}   # each idea's own best score, as of its last proposal
        self.last_choice: Optional[Dict[str, Any]] = None

    def _tuner(self, node: Dict[str, Any]):
        if node["id"] not in self._inner:
            self._inner[node["id"]] = self.make_inner(node)
        return self._inner[node["id"]]

    def propose(self, history: List[dict], evidence: list):
        pick = self.chooser.choose(history, exclude=self.paused)
        if pick is None:
            raise NoIdeaLeft("no runnable idea left: every idea is dead, paused, or has no runner")
        node = pick["node"]
        self.last_choice = pick
        local = local_history(history, node["id"], self.sign, self.record_problems)
        kept = [r["score"] for r in local if r.get("status") == "keep"]
        self._best[node["id"]] = kept[-1] if kept else None
        exp = self._tuner(node).propose(local, evidence)
        runner = node.get("runner") or {}
        exp.config = {**(runner.get("config") or {}), **(exp.config or {})}
        if runner.get("optimizer"):
            exp.optimizer = runner["optimizer"]
        exp.rationale = ideas.tag_rationale(exp.rationale, node["id"])
        return exp

    def review(self, experiment_result: dict) -> str:
        idea_id = ideas.idea_of(experiment_result)
        node = self.store.get(idea_id) if idea_id else None
        if node is not None and idea_id in self._inner:
            local = dict(experiment_result)          # the tuner judges the idea against itself
            score = verdicts.valid_score(local)
            if score is not None:
                best = self._best.get(idea_id)
                local["status"] = "keep" if best is None or self.sign * (score - best) > 0 else "discard"
            if self._inner[idea_id].review(local) == "stop":
                self.paused.add(idea_id)
        return "continue" if self.chooser.candidates(exclude=self.paused) else "stop"
