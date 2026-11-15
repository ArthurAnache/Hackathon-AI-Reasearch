"""Verdicts: what OUR experiments say about each idea, in units of delta.

An experiment belongs to an idea when its rationale carries `idea=<id>`
(integrations.ideas) and it was run on the idea's problem. A verdict is always
recomputed from the experiments themselves, never incremented, so it does not
depend on the order of calls and a resumed run reaches the same verdict.

THE BAR. A run is measured against the incumbent: the best of
    - the live record, when it was read (never a record copied from a note);
    - the best score any OTHER idea (or an untagged run) reached earlier in the same store;
    - the score the run started from (`best_before`; `start_score` for logs made
      outside the loop);
    - the idea's own best so far.
gain = score - incumbent, reported as gain/delta (delta = the smallest improvement
that makes a new record on the problem, 1e-5 on the main target).

EVIDENCE of one run (valid = status keep/discard with a finite score):
    positive   gain >= min_gain_over_delta * delta (default 0.01 delta; never less
               than 1e-12 relative, so float noise is never a gain), AND the bar was
               external: the record, another idea's result, or an explicit starting
               score. An idea that only improves on its own earlier runs, from
               scratch and far below the record, has shown progress, not a gain.
               For a `worked` node with a claimed_value the criterion is instead:
               our score reached the value the claim reported.
    baseline   the idea's first valid run in a store
    progress   still improving on itself: its score beats the idea's own best by the
               threshold, and its lift beats the idea's best lift so far. The lift is
               how far the run moved from where it started (or, with no separate
               start, from the record or the other ideas): a run that merely starts
               from a better solution scores higher without the idea having done anything
    repeat     same optimizer, config, seed and budget as an earlier run: no new information
    short      a budget smaller than the idea's baseline run: not a fair try
    forgiven   a negative run a human set aside (retest)
    negative   a valid run with none of the above
    none       crash, invalid, void, no score: counted, and its compute too, but evidence of nothing
    ambiguous  its experiment id appears twice in the same store: not evidence either
               (the runs are kept apart as "<id>#1", "<id>#2" so the verdict stays readable)

THE VERDICT (ideas.verdict_status is the only mapping to a word):
    gained     at least one positive run
    exhausted  no positive, at least `refute_after` negative runs (default 3), AND
               those negative runs add up to the idea's fair budget: the runner's
               min_compute_s, else `min_compute_s` (default 3600 s). Four one-second
               runs refute nothing. A human can reopen a refuted idea: retest().
    worked / untested_idea   gained -> reproduced    exhausted -> refuted
    dead_end                 gained -> refuted (alive again: it beat everything we
                             had, the record included when known)   exhausted -> reproduced

SEVERAL STORES. Each comparison arm has its own experiment store. A verdict keeps
its runs per store ("<store label>/<experiment id>"); updating from one store
replaces that store's runs and keeps the others, inside one locked write of the
idea store. A positive found by one arm is never erased by another.

What a verdict stores, so the referee can re-check it: every run (store,
experiment id, problem, valid, score, seed, budget, solution path and sha256, the
bar and where it came from), the reference (record value, when it was read,
delta), the rule, when it was decided and by which version of this file.

A verdict is never a record. Only selection/verify_claim.py printing beats_record
makes one; `record_candidates` only says "worth sending to the referee".

    python -m integrations.verdicts retest ws1-018 --reason "only ran 60 s budgets" --by Murilo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from . import ideas
from ._common import utc_now

VALID = ("keep", "discard")
WS2_STATUSES = ("keep", "discard", "invalid", "crash", "void")
MIN_GAIN_OVER_DELTA = 0.01
REFUTE_AFTER = 3
MIN_COMPUTE_S = 3600.0
RELATIVE_FLOOR = 1e-12            # WS2's own tie tolerance (research/knowledge.py TIE_RTOL)
_FLOAT = r"([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"
try:
    DECIDED_BY = "integrations.verdicts@" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
except OSError:
    DECIDED_BY = "integrations.verdicts@unknown"


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


@dataclass
class Reference:
    """The yardstick of one problem. delta and record may be unknown (None).
    record must come from a live read, never from a node or a note: the #1 of the
    main target moved by 2.3 delta during the first day."""
    problem: Optional[str] = None
    direction: str = "maximize"
    delta: Optional[float] = None
    record: Optional[float] = None
    record_read_at: Optional[str] = None
    source: str = "unknown"

    @property
    def sign(self) -> float:
        return -1.0 if self.direction == "minimize" else 1.0

    def over_delta(self, gain: Optional[float]) -> Optional[float]:
        return None if gain is None or not self.delta else gain / self.delta

    def threshold(self, min_gain_over_delta: float = MIN_GAIN_OVER_DELTA, at: float = 1.0) -> float:
        """Smallest gain that counts, for scores of magnitude `at`."""
        floor = RELATIVE_FLOOR * max(1.0, abs(at))
        return max(min_gain_over_delta * self.delta, floor) if self.delta else floor

    def as_dict(self) -> Dict[str, Any]:
        return {"record": self.record, "record_read_at": self.record_read_at, "delta": self.delta,
                "direction": self.direction, "source": self.source}


def reference_from_problem(problem: Any, read_at: Optional[str] = None) -> Reference:
    """From a WS2 `interfaces.Problem`. When WS1's catalog module backs it
    (problems.catalog.ws2_module), BEST_KNOWN is the live #1 at load time and DELTA
    is there; otherwise delta is unknown and gains are reported in raw units."""
    m = getattr(problem, "_m", None)
    delta = getattr(m, "DELTA", None)
    return Reference(problem=getattr(problem, "id", None), direction=getattr(problem, "direction", "maximize"),
                     delta=float(delta) if delta else None, record=getattr(problem, "best_known", None),
                     record_read_at=read_at or utc_now(),
                     source=str(getattr(problem, "best_known_source", "") or "problem module"))


def reference_from_nodes(nodes: Iterable[Dict[str, Any]], problem: str,
                         direction: str = "maximize") -> Reference:
    """Offline fallback for the map page: delta as WS1 wrote it in the problem's
    record node ("...; delta 1e-5; ..."), read only from a node a trusted import
    wrote. The record value is deliberately NOT taken from the node: it goes stale.
    Read it live (research_map --live)."""
    for n in nodes:
        if n["kind"] == "record" and n["problem"] == problem and n.get("trusted"):
            dlt = re.search(r"\bdelta\s+" + _FLOAT, n["content"])
            if dlt:
                return Reference(problem=problem, direction=direction, delta=float(dlt.group(1)),
                                 source="delta from node %s, written %s; record not read"
                                        % (n["id"], n["written_on"]))
    return Reference(problem=problem, direction=direction, source="delta and record unknown")


# ------------------------------------------------------------------ stores and records
def store_label(path: Any) -> str:
    """A short, stable name for an experiment store: "<file stem>@<hash of its full path>".
    Two stores that are both called experiments.jsonl get different labels."""
    p = Path(path).resolve()
    return "%s@%s" % (p.stem, hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:6])


def labelled(records: Iterable[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    """Copies of the records carrying their store label (never changes the input)."""
    return [dict(r, _store=label) for r in records]


def normalise_record(r: Dict[str, Any], sign: float = 1.0, default_problem: Optional[str] = None,
                     labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Make an experiment logged outside the WS2 loop readable here. Such logs (an
    agent's own experiments.jsonl) carry `idea`, `score`, `start_score`, `budget_s`
    and a free-text status. A record that already has a WS2 status is returned as is.
        idea         translated through `labels` ({the agent's label: map id}) and
                     written as the idea=<id> tag at the head of the rationale, the
                     only place an attribution is ever read from. The original text
                     follows it, with its key=value tokens broken so it adds no tag.
        best_before  the score the run started from (`start_score`): the bar it had to beat
        status       keep if the score beat that bar, else discard; "none" when there is
                     no final score (a run still going), which is evidence of nothing
        duration_s   `duration_s`, else `killed_at_s`, else `budget_s`
    The agent's own status word is kept in `reported_status`; it is never a verdict."""
    if r.get("status") in WS2_STATUSES:
        return r
    out = dict(r)
    idea_id = r.get("idea")
    if labels and isinstance(idea_id, str) and idea_id in labels:
        idea_id = labels[idea_id]
    text = ideas._plain(r["rationale"]) if isinstance(r.get("rationale"), str) else ""
    head = "strategy=external; parent=none"
    if isinstance(idea_id, str) and ideas.ID_RE.fullmatch(idea_id):
        head += "; idea=%s" % idea_id
    out["rationale"] = head + ("; " + text if text else "")
    out["reported_status"] = r.get("status")
    if out.get("problem") is None:
        out["problem"] = default_problem
    if out.get("best_before") is None:
        out["best_before"] = r.get("start_score")
    score, bar = r.get("score"), out["best_before"]
    if not _num(score):
        out["status"] = "none"
    else:
        out["status"] = "keep" if not _num(bar) or sign * (score - bar) > 0 else "discard"
    if not out.get("duration_s"):
        out["duration_s"] = r.get("killed_at_s") or r.get("budget_s") or 0.0
    return out


def valid_score(r: Dict[str, Any]) -> Optional[float]:
    """The score of a valid run (status keep/discard, finite number), else None."""
    s = r.get("score")
    return float(s) if r.get("status") in VALID and _num(s) else None


def _signature(r: Dict[str, Any]) -> str:
    return json.dumps([r.get("optimizer"), r.get("algorithm_version"), r.get("config"), r.get("seed"),
                       r.get("budget_s")], sort_keys=True, default=str)


_SHA_CACHE: Dict[Tuple[str, int, int], str] = {}


def solution_sha256(solution_path: Any, root: Optional[Path]) -> Optional[str]:
    """sha256 of the saved solution file, or None when hashing is off or the file is absent."""
    if root is None or not isinstance(solution_path, str) or not solution_path:
        return None
    path = Path(solution_path)
    path = path if path.is_absolute() else Path(root) / path
    try:
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in _SHA_CACHE:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            _SHA_CACHE[key] = h.hexdigest()
        return _SHA_CACHE[key]
    except OSError:
        return None


def records_for(node: Dict[str, Any], records: Iterable[Dict[str, Any]],
                aliases: Optional[Dict[str, str]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(experiments of this idea on its own problem, experiments tagged with the idea
    but run on another problem). The second list is never evidence."""
    mine, foreign = [], []
    for r in records:
        if ideas.idea_of(r) != node["id"]:
            continue
        (mine if ideas.same_problem(node["problem"], r.get("problem"), aliases) else foreign).append(r)
    return mine, foreign


def by_idea(records: Iterable[Dict[str, Any]]) -> Dict[Optional[str], List[Dict[str, Any]]]:
    """Group experiment records by their idea tag (None = no idea tag), keeping order."""
    out: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for r in records:
        out.setdefault(ideas.idea_of(r), []).append(r)
    return out


# ------------------------------------------------------------------ evidence, run by run
def observe(node: Dict[str, Any], history: Iterable[Dict[str, Any]], ref: Reference,
            aliases: Optional[Dict[str, str]] = None, min_gain_over_delta: float = MIN_GAIN_OVER_DELTA,
            hash_root: Optional[Path] = None, **_ignored: Any) -> List[Dict[str, Any]]:
    """One row per experiment of the idea, store by store. `history` is everything the
    stores hold (all ideas, tagged or not): other ideas' scores set the bar."""
    by_value = node["kind"] == "worked" and node.get("claimed_value") is not None
    forgiven = set((node.get("retest") or {}).get("forgiven") or [])
    stores: Dict[str, List[Dict[str, Any]]] = {}
    for r in history:
        stores.setdefault(str(r.get("_store") or ""), []).append(r)
    rows = []
    for label, records in stores.items():
        ids = [str(r.get("experiment_id")) for r in records if ideas.idea_of(r) == node["id"]
               and ideas.same_problem(node["problem"], r.get("problem"), aliases)]
        twice = {i for i in ids if ids.count(i) > 1}
        occurrence: Dict[str, int] = {}
        own = others = best_lift = baseline_budget = None
        seen: Set[str] = set()
        for r in records:
            if not ideas.same_problem(node["problem"], r.get("problem"), aliases):
                continue
            score = valid_score(r)
            if ideas.idea_of(r) != node["id"]:
                # another idea's run, or an untagged one, on the same problem: part of the bar
                if score is not None and (others is None or ref.sign * (score - others) > 0):
                    others = score
                continue
            eid = str(r.get("experiment_id"))
            ambiguous = eid in twice
            if ambiguous:                      # which of the two is the real one? neither counts
                occurrence[eid] = occurrence.get(eid, 0) + 1
                eid = "%s#%d" % (eid, occurrence[eid])
            row = {"experiment_id": eid, "store": label, "problem": r.get("problem"), "status": r.get("status"),
                   "valid": score is not None, "score": score, "seed": r.get("seed"), "budget_s": r.get("budget_s"),
                   "agent": r.get("agent"), "timestamp": r.get("timestamp"),
                   "duration_s": float(r["duration_s"]) if _num(r.get("duration_s")) else 0.0,
                   "solution_path": r.get("solution_path"),
                   "solution_sha256": solution_sha256(r.get("solution_path"), hash_root),
                   "bar": None, "bar_source": None, "gain": None, "gain_over_delta": None, "local_gain": None,
                   "vs_record_over_delta": None, "evidence": "none", "record_candidate": False}
            row["key"] = ideas.run_key(row)
            if ambiguous:
                row["evidence"] = "ambiguous"
            elif score is not None:
                start = r.get("best_before") if _num(r.get("best_before")) else None
                bars = [(ref.record, "record read %s" % ref.record_read_at), (others, "best of the other ideas"),
                        (start, "the score it started from"), (own, "the idea's own best")]
                bars = [(v, s) for v, s in bars if v is not None]
                if bars:
                    bar, source = max(bars, key=lambda b: ref.sign * b[0])
                    row.update(bar=bar, bar_source=source, gain=ref.sign * (score - bar))
                if own is not None:
                    row["local_gain"] = ref.sign * (score - own)
                thr = ref.threshold(min_gain_over_delta, at=score)
                external = ref.record is not None or others is not None or _num(r.get("start_score"))
                if by_value:
                    positive = ref.sign * (score - node["claimed_value"]) >= 0
                else:
                    positive = bool(external and row["gain"] is not None and row["gain"] > 0 and row["gain"] >= thr)
                # lift: how far the run moved from where it stood. Its own starting score when it has
                # one that is not just the idea's previous best; else the record or the other ideas.
                outside = [v for v in (ref.record, others) if v is not None]
                anchor = start if start is not None and start != own else (
                    max(outside, key=lambda v: ref.sign * v) if outside else None)
                lift = ref.sign * (score - anchor) if anchor is not None else None
                progress = (row["local_gain"] is not None and row["local_gain"] > 0 and row["local_gain"] >= thr
                            and (lift is None or best_lift is None or lift - best_lift >= thr))
                sig = _signature(r)
                budget = r.get("budget_s")
                if positive:
                    row["evidence"] = "positive"
                elif own is None:
                    row["evidence"] = "baseline"
                elif progress:
                    row["evidence"] = "progress"
                elif sig in seen:
                    row["evidence"] = "repeat"
                elif _num(budget) and _num(baseline_budget) and budget < baseline_budget:
                    row["evidence"] = "short"
                elif row["key"] in forgiven:
                    row["evidence"] = "forgiven"
                else:
                    row["evidence"] = "negative"
                seen.add(sig)
                if own is None:
                    baseline_budget = budget
                if own is None or ref.sign * (score - own) > 0:
                    own = score
                if lift is not None and (best_lift is None or lift > best_lift):
                    best_lift = lift
                row["gain_over_delta"] = ref.over_delta(row["gain"])
                if ref.record is not None:
                    above = ref.sign * (score - ref.record)
                    row["vs_record_over_delta"] = ref.over_delta(above)
                    # same test as WS1's beats_record (gain >= delta and gain > 0); a hint, never a ruling
                    row["record_candidate"] = bool(ref.delta is not None and above > 0 and above >= ref.delta)
            rows.append(row)
    return rows


# ------------------------------------------------------------------ the verdict of one idea
def summarise(node: Dict[str, Any], rows: List[Dict[str, Any]], reference: Dict[str, Any],
              min_gain_over_delta: float = MIN_GAIN_OVER_DELTA, refute_after: int = REFUTE_AFTER,
              min_compute_s: float = MIN_COMPUTE_S, **_ignored: Any) -> Optional[Dict[str, Any]]:
    """A verdict from run rows (possibly of several stores). None without runs."""
    if not rows or node["kind"] not in ideas.TESTABLE:
        return None
    sign = -1.0 if reference.get("direction") == "minimize" else 1.0
    by_value = node["kind"] == "worked" and node.get("claimed_value") is not None
    forgiven = set((node.get("retest") or {}).get("forgiven") or [])
    rows = [dict(x, evidence="forgiven") if x["evidence"] == "negative" and x["key"] in forgiven else x for x in rows]
    positives = [x for x in rows if x["evidence"] == "positive"]
    negatives = [x for x in rows if x["evidence"] == "negative"]
    budget = (node.get("runner") or {}).get("min_compute_s") or min_compute_s
    spent_against = sum(x["duration_s"] for x in negatives)
    gained = bool(positives)
    exhausted = not gained and len(negatives) >= refute_after and spent_against >= budget
    scored = [x for x in rows if x["score"] is not None]
    top = max(scored, key=lambda x: sign * x["score"]) if scored else None
    with_gain = [x for x in rows if x["gain"] is not None]
    best = max(with_gain, key=lambda x: x["gain"]) if with_gain else None
    if gained:                     # the run that carries the positive evidence
        best = max(positives, key=(lambda x: sign * x["score"]) if by_value else (lambda x: x["gain"]))
    dead_end = node["kind"] == "dead_end"
    return {
        "status": ideas.verdict_status(node["kind"], gained, exhausted),
        "gained": gained, "exhausted": exhausted,
        "criterion": "claimed_value" if by_value else "gain",
        "claimed_value": node["claimed_value"] if by_value else None,
        "experiments": [x["key"] for x in rows],
        "supporting": [x["key"] for x in (negatives if dead_end else positives)],
        "contradicting": [x["key"] for x in (positives if dead_end else negatives)],
        "trials": len(rows), "valid": len(scored), "failed": len(rows) - len(scored),
        "best_experiment": best["key"] if best else None,
        "best_gain": best["gain"] if best else None,
        "best_gain_over_delta": best["gain_over_delta"] if best else None,
        "best_score": (best if by_value and gained else top)["score"] if top else None,
        "best_vs_record_over_delta": top["vs_record_over_delta"] if top else None,
        "record_candidates": [x["key"] for x in rows if x["record_candidate"]],
        "compute_s": round(sum(x["duration_s"] for x in rows), 2),
        "negative_compute_s": round(spent_against, 2), "fair_budget_s": float(budget),
        "reference": dict(reference),
        "rule": {"min_gain_over_delta": min_gain_over_delta, "refute_after": refute_after,
                 "min_compute_s": float(min_compute_s)},
        "runs": rows,
        "decided_by": DECIDED_BY,
    }


def compute_verdict(node: Dict[str, Any], history: Iterable[Dict[str, Any]], ref: Reference,
                    aliases: Optional[Dict[str, str]] = None, **rule: Any) -> Optional[Dict[str, Any]]:
    """The verdict of one idea from everything the given stores hold. None when the
    idea has no experiment there, or is a record or a fact."""
    if node["kind"] not in ideas.TESTABLE:
        return None
    return summarise(node, observe(node, history, ref, aliases, **rule), ref.as_dict(), **rule)


def _same(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
    strip = lambda v: {k: x for k, x in (v or {}).items() if k != "decided_on"}
    return (a is None) == (b is None) and strip(a) == strip(b)


def update_verdict(store: ideas.IdeaStore, idea_id: str, history: List[Dict[str, Any]], ref: Reference,
                   memory=None, scope: Optional[str] = ideas.SCOPE, aliases: Optional[Dict[str, str]] = None,
                   **rule: Any) -> Optional[Dict[str, Any]]:
    """Recompute the verdict of one idea from the experiment stores present in
    `history` (records carry their store in `_store`, see labelled()), and write it
    to the idea store and to the idea's memory fact. Runs that other stores
    contributed earlier are kept: the merge is one locked write. Returns the node,
    or None if the idea is unknown. Records and facts have no verdict."""
    history = list(history)
    labels = {str(r.get("_store") or "") for r in history}
    changed = []

    def merge(node):
        if node["kind"] not in ideas.TESTABLE:
            return node
        old = node.get("verdict") or {}
        kept = [x for x in old.get("runs", []) if x.get("store", "") not in labels]
        rows = kept + observe(node, history, ref, aliases, **rule)
        verdict = summarise(node, rows, ref.as_dict(), **rule)
        if _same(verdict, node.get("verdict")):
            return node
        if verdict is not None:
            verdict["decided_on"] = utc_now()
        changed.append(True)
        return ideas.with_verdict(node, verdict)

    try:
        node = store.update(idea_id, merge)
    except KeyError:
        return None
    if changed and memory is not None:
        ideas.mirror(memory, node, scope)
    return node


def update_all(store: ideas.IdeaStore, history: List[Dict[str, Any]], ref_for: Callable[[Dict[str, Any]], Reference],
               memory=None, scope: Optional[str] = ideas.SCOPE, aliases: Optional[Dict[str, str]] = None,
               **rule: Any) -> Dict[str, str]:
    """Recompute every idea that has experiments. Returns {idea id: verdict status};
    ids found in tags but absent from the map map to "unknown"."""
    out = {}
    for idea_id in by_idea(history):
        if idea_id is None:
            continue
        node = store.get(idea_id)
        if node is None:
            out[idea_id] = "unknown"
            continue
        node = update_verdict(store, idea_id, history, ref_for(node), memory=memory, scope=scope,
                              aliases=aliases, **rule)
        out[idea_id] = (node.get("verdict") or {}).get("status", "no verdict")
    return out


def retest(store: ideas.IdeaStore, idea_id: str, reason: str, by: str, memory=None,
           scope: Optional[str] = ideas.SCOPE) -> Dict[str, Any]:
    """A human reopens an idea our experiments refuted: its negative runs so far are
    set aside (kept in the verdict as "forgiven", with who and why), so it needs a new
    full set of negative runs to be refuted again. It cannot revive a dead end
    reported by a source: only a run that gains does that."""
    if not reason or not by:
        raise ideas.InvalidIdea("a retest needs a reason and a name")

    def change(node):
        verdict = node.get("verdict")
        if not verdict:
            raise ideas.InvalidIdea("%s has no verdict: nothing to retest" % idea_id)
        if node["kind"] == "dead_end":
            raise ideas.InvalidIdea("%s is a dead end reported by its source: a retest cannot revive it" % idea_id)
        negatives = [x["key"] for x in verdict["runs"] if x["evidence"] == "negative"]
        before = (node.get("retest") or {}).get("forgiven") or []
        node = dict(node, retest={"forgiven": sorted(set(before) | set(negatives)), "reason": str(reason),
                                  "by": str(by), "at": utc_now()})
        fresh = summarise(node, verdict["runs"], verdict["reference"], **verdict["rule"])
        fresh["decided_on"] = utc_now()
        return ideas.with_verdict(node, fresh)

    node = store.update(idea_id, change)
    if memory is not None:
        ideas.mirror(memory, node, scope)
    return node


class VerdictStore:
    """ExperimentStore wrapper (interfaces.py contract): after each experiment is
    appended, the verdict of its idea is updated. A failure here is reported on
    stderr and never stops the loop: the experiment record is already safe.
    label names this experiment store inside the verdicts (store_label(path))."""

    def __init__(self, inner, idea_store: ideas.IdeaStore, ref: Reference, memory=None,
                 scope: Optional[str] = ideas.SCOPE, aliases: Optional[Dict[str, str]] = None,
                 hash_root: Optional[Path] = None, label: Optional[str] = None, **rule: Any):
        self.inner, self.idea_store, self.ref = inner, idea_store, ref
        self.memory, self.scope, self.aliases = memory, scope, aliases
        self.rule = dict(rule, hash_root=hash_root)
        if label is None:
            path, obj = None, inner
            for _ in range(4):                       # TaggedStore -> PastDevMirrorStore -> JsonlStore
                path = getattr(obj, "path", None)
                obj = getattr(obj, "inner", None)
                if path is not None or obj is None:
                    break
            label = store_label(path) if path is not None else "store"
        self.label = label

    def append(self, result: dict) -> None:
        self.inner.append(result)
        idea_id = ideas.idea_of(result)
        if idea_id is None:
            return
        try:
            node = update_verdict(self.idea_store, idea_id, labelled(self.inner.read(), self.label), self.ref,
                                  memory=self.memory, scope=self.scope, aliases=self.aliases, **self.rule)
            if node is None:
                print("verdict: idea %s is not in the map (%s)" % (idea_id, self.idea_store.path), file=sys.stderr)
            elif not ideas.same_problem(node["problem"], result.get("problem"), self.aliases):
                print("verdict: experiment %s ran on problem %s, idea %s is about %s: not counted"
                      % (result.get("experiment_id"), result.get("problem"), idea_id, node["problem"]),
                      file=sys.stderr)
        except Exception as e:
            print("verdict update failed for %s: %s: %s" % (idea_id, type(e).__name__, e), file=sys.stderr)

    def read(self) -> list:
        return self.inner.read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ideas", help="idea store file (default: %s)" % ideas.default_path())
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("retest", help="reopen an idea our experiments refuted")
    s.add_argument("id")
    s.add_argument("--reason", required=True)
    s.add_argument("--by", required=True, help="who decided")
    a = ap.parse_args(argv)
    try:
        node = retest(ideas.IdeaStore(a.ideas), a.id, a.reason, a.by)
    except (ValueError, KeyError) as e:
        print("error: %s" % (e.args[0] if e.args else e), file=sys.stderr)
        return 1
    print("%s: %d negative runs set aside, status now %s" % (node["id"], len(node["retest"]["forgiven"]),
                                                              node["status"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
