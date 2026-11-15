"""The idea layer above WS2's knob tuner, written only against the contract in interfaces.py.

    IdeaAgent   a ResearchAgent that wraps another one (FeedbackAgent). The inner
                agent still tunes the knobs; IdeaAgent says which IDEA the experiment
                is for, and writes it in the rationale as `idea=<id>` next to the
                `strategy=...; parent=...` tags the inner agent wrote.

No WS2 file changes: the wrapper only edits the ExperimentConfig it passes on.
WS2's Knowledge keeps reading its own two tags and ignores the third.
"""
from __future__ import annotations

import sys
from typing import Any, Iterable, List, Optional

from . import ideas


class DeadIdea(ValueError):
    """The idea is a dead end (or was refuted by us): it must not be proposed again."""


class WrongProblem(ValueError):
    """The idea is about another problem than the one being run."""


def check_runnable(node: Optional[dict], idea_id: str, problems: Optional[Iterable[str]] = None) -> dict:
    """problems: the names under which the problem of this run is known to the map
    (its WS2 id, plus the map's slug when they differ). None skips the check."""
    if node is None:
        raise KeyError("unknown idea %r: import it first (python -m integrations.import_knowledge)" % idea_id)
    if node["kind"] not in ideas.TESTABLE:
        raise ValueError("%s is a %s, not an idea an experiment can test" % (idea_id, node["kind"]))
    if problems is not None and node["problem"] not in set(problems):
        raise WrongProblem("%s is about problem %s, this run is on %s (if they are the same problem, "
                           "say so with --idea-problem %s)"
                           % (idea_id, node["problem"], "/".join(sorted(set(problems))), node["problem"]))
    if ideas.is_dead(node):
        raise DeadIdea("%s is a dead end (%s, status %s): %s"
                       % (idea_id, node["kind"], node["status"], node["content"][:160]))
    return node


class IdeaAgent:
    """Runs every experiment of the wrapped agent under one idea."""

    def __init__(self, inner: Any, store: ideas.IdeaStore, idea_id: str,
                 problems: Optional[Iterable[str]] = None):
        self.inner, self.store, self.idea_id = inner, store, ideas.check_id(idea_id)
        self.problems = set(problems) if problems is not None else None
        check_runnable(store.get(idea_id), idea_id, self.problems)   # refuse before any compute is spent

    @property
    def name(self) -> str:
        return getattr(self.inner, "name", "agent")

    def propose(self, history: List[dict], evidence: list):
        # it may have died since the last experiment
        check_runnable(self.store.get(self.idea_id), self.idea_id, self.problems)
        exp = self.inner.propose(history, evidence)
        exp.rationale = ideas.tag_rationale(exp.rationale, self.idea_id)
        return exp

    def review(self, experiment_result: dict) -> str:
        action = self.inner.review(experiment_result)
        try:
            node = self.store.get(self.idea_id)
        except ideas.CorruptStore as e:          # cannot confirm the idea is alive: stop, do not guess
            print("stopping: %s" % e, file=sys.stderr)
            return "stop"
        return "stop" if node is None or ideas.is_dead(node) else action
