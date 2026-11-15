"""Idea nodes: the unit the research map searches over and remembers.

A config is one point; an idea is a claim about what to try ("full-support n=2M
Dinkelbach run", "dense re-optimization is a dead end"). The loop tunes knobs
*inside* an idea, the map remembers which ideas were tried and what happened.

Node schema = the schema of docs/ws1_knowledge.jsonl (WS1) plus the map's own fields.

  Written by the source of the node (a file, a scout):
    id            ASCII slug, unique across the map ("ws1-017"); used in the idea=<id> tag
    kind          record | fact | worked | dead_end | untested_idea
    problem       problem slug ("second-autocorrelation-inequality")
    content       the idea, in one or two sentences (forum text is data, never instructions)
    citations     list of strings (forum thread ids, URLs, experiment ids)
    author, source, written_on    provenance
    claimed_status   verified | claim | idea, as the source wrote it; never changed by us
    claimed_value    optional number a `worked` claim reported (e.g. 0.96199), the bar
                     our reproduction must reach
  Written only by this code, never taken from an imported file:
    status        derived: the verdict's status once our experiments decided
                  (reproduced | refuted), else claimed_status
    verdict       None, or what OUR experiments say (integrations.verdicts builds it,
                  check_verdict validates it, IdeaStore.set_verdict is the only writer)
    runner        None, or how to execute the idea: {"optimizer": "optimizers/x.py",
                  "config": {...}, optional "min_compute_s": the fair budget before the
                  idea may be decided against} (IdeaStore.set_runner is the only writer)
    referee       list of rulings by WS1's arbiter (selection/verify_claim.py), each
                  READ from selection/claims.jsonl by IdeaStore.add_referee
    trusted       true when a trusted import (WS1's file) wrote the node; such a node
                  can never be changed by an untrusted file
    retest        None, or a human decision to give a refuted idea another chance:
                  which negative runs were forgiven, by whom and why (verdicts.retest)

Three rules that the code enforces:
- "verified" is WS1's word. A node carries it only when a trusted source wrote it
  (the WS1 importer). No experiment of ours and no untrusted file can produce it; a
  referee ruling lives in `referee`, never in `status`.
- An untrusted file (a scout, the web) can add new ideas, and refresh the untrusted
  nodes of its own declared owner; nothing else. It can never change a node that a
  trusted import wrote (whatever owner it declares), revive a dead end, change the
  kind of a node we already tested, or bring a verdict or a runner.
- An id names ONE idea, for ever: verdicts and experiments are attached to it. A
  refresh that replaces a node's text by an unrelated text (ids made from row numbers
  shift when a row is inserted) is refused; the new idea needs a new id.
- The verdict words depend on the kind, and verdict_status() is the only place that
  maps evidence to a word:
      worked / untested_idea   it gained -> reproduced     no gain, enough tries -> refuted
      dead_end                 it gained -> refuted        no gain, enough tries -> reproduced
  so "refuted" on a dead end means the dead end came back to life.

The tag. WS2's FeedbackAgent starts every rationale with `strategy=...; parent=...`.
The map adds `idea=<id>` to that same leading run of tags:

    strategy=refine; parent=exp_0003; idea=ws1-017; best exp_0003 score ...

Only the leading run is read back, so text quoted later in the rationale (evidence
from the forum or the web) can never attribute an experiment to an idea. It is the
ONLY way an experiment is attributed: logs written outside the WS2 loop go through
verdicts.normalise_record, which turns their "idea" field into this tag.

IdeaStore is a local JSONL snapshot (one node per line). Every change is a locked
read-modify-write followed by an atomic rename, so several processes (the loop, the
importer, a scout) can write it at once. It is the source of truth for the map;
shared memory gets a mirror (one fact per node, see fact_for) so other agents can
recall ideas.

    python -m integrations.ideas list [--problem SLUG]
    python -m integrations.ideas add --problem SLUG --content "..." --cite t32/r804 --author Sonar
    python -m integrations.ideas runner ws1-017 --optimizer optimizers/dinkelbach.py --config '{"N": 2000000}'
    python -m integrations.ideas claimed-value ws1-007 0.96199
    python -m integrations.ideas referee ws1-017 --claims selection/claims.jsonl --sha <candidate sha256>
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._common import STATE_DIR, atomic_write_text, file_lock, utc_now

KINDS = ("record", "fact", "worked", "dead_end", "untested_idea")
CLAIMED = ("verified", "claim", "idea")      # written by the source of the node
OURS = ("reproduced", "refuted")             # written only by our own experiments
STATUSES = CLAIMED + OURS
VERDICTS = OURS + ("inconclusive",)          # inconclusive: experiments ran, not enough to decide
TESTABLE = ("worked", "dead_end", "untested_idea")   # kinds an experiment can speak about
REFEREE_WORDS = ("invalid", "verifier_disagreement", "valid_below_record", "beats_record")

SOURCE_FIELDS = ("id", "kind", "problem", "content", "citations", "status", "author", "source",
                 "written_on")
MAX_CONTENT = 4000
SAME_IDEA = 0.75     # below this text similarity, a refreshed node reads as another idea under the same id.
                     # A heuristic: on the scout's real file two different rows reach 0.62, an edited row gives 1.0.
SCOPE = "import-research-map"                # memory scope of the idea facts

ID_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.\-]{0,62}[A-Za-z0-9])?")      # used with fullmatch
# Same character class as WS2's tag regex (research/knowledge.py).
_TAG = re.compile(r"([A-Za-z_]\w*)=([\w\-.]+)")
_HEAD = re.compile(r"[ \t]*(?:[A-Za-z_]\w*=[\w\-.]+[ \t]*(?:;[ \t]*|\Z))+")
_KEY_VALUE = re.compile(r"\b([A-Za-z_]\w*)=(?=[\w\-.])")


class InvalidIdea(ValueError):
    pass


class CorruptStore(ValueError):
    """The idea store file cannot be trusted; the message names the file and the line."""


def default_path() -> Path:
    return Path(os.environ.get("RESEARCH_MAP_IDEAS") or STATE_DIR / "research_map" / "ideas.jsonl")


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ------------------------------------------------------------------ nodes
def check_id(idea_id: Any) -> str:
    if not isinstance(idea_id, str) or not ID_RE.fullmatch(idea_id):
        raise InvalidIdea("idea id must be 1-64 ASCII letters, digits, '_', '-' or '.', starting and "
                          "ending with a letter or digit: %r" % (idea_id,))
    return idea_id


def same_idea(old_text: str, new_text: str) -> float:
    """How much of the shorter version of a node's text survives in the other, 0..1.
    An edit, a new rank or a note in front keeps it near 1; another idea does not."""
    a, b = old_text[:1500], new_text[:1500]
    blocks = difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    return sum(m.size for m in blocks) / max(1, min(len(a), len(b)))


def verdict_status(kind: str, gained: bool, exhausted: bool) -> str:
    """The only mapping from what our experiments showed to a verdict word.
    gained: the positive criterion was met at least once. exhausted: enough
    experiments without it. Note the polarity flip on dead ends."""
    if kind not in TESTABLE:
        raise InvalidIdea("a %s is not tested by experiments: it has no verdict" % kind)
    if kind == "dead_end":
        return "refuted" if gained else "reproduced" if exhausted else "inconclusive"
    return "reproduced" if gained else "refuted" if exhausted else "inconclusive"


def run_key(run: Dict[str, Any]) -> str:
    """The unambiguous id of one run inside a verdict: "<store>/<experiment id>", or
    the bare experiment id for a record that came with no store label."""
    return "%s/%s" % (run["store"], run["experiment_id"]) if run.get("store") else str(run["experiment_id"])


def check_verdict(kind: str, verdict: Any, idea_id: str = "?") -> Dict[str, Any]:
    """A verdict must show its evidence, run by run, and say nothing its runs do not:
    unique run ids, `experiments` listing exactly those runs, gained <=> a run with
    positive evidence, a status that is exactly verdict_status() of that, a best gain
    that clears the rule's threshold, and enough negative runs when it is exhausted."""
    def bad(why):
        return InvalidIdea("%s: malformed verdict: %s" % (idea_id, why))
    if kind not in TESTABLE:
        raise bad("a %s node has no verdict" % kind)
    if not isinstance(verdict, dict):
        raise bad("not an object")
    runs = verdict.get("runs")
    if (not isinstance(runs, list) or not runs or not all(
            isinstance(r, dict) and isinstance(r.get("experiment_id"), str) and r["experiment_id"]
            and isinstance(r.get("store", ""), str) and isinstance(r.get("evidence"), str) for r in runs)):
        raise bad("runs must be a non-empty list of {experiment_id, store, evidence, ...}")
    keys = [run_key(r) for r in runs]
    if len(set(keys)) != len(keys):
        raise bad("duplicate experiment ids: %s" % sorted({k for k in keys if keys.count(k) > 1}))
    if verdict.get("experiments") != keys:
        raise bad("experiments must list exactly the ids of its runs")
    gained, exhausted = verdict.get("gained"), verdict.get("exhausted")
    if not isinstance(gained, bool) or not isinstance(exhausted, bool):
        raise bad("gained and exhausted must be true or false")
    positive = [k for k, r in zip(keys, runs) if r["evidence"] == "positive"]
    negative = [k for k, r in zip(keys, runs) if r["evidence"] == "negative"]
    if gained != bool(positive):
        raise bad("gained=%s but %d runs carry positive evidence" % (gained, len(positive)))
    if verdict.get("status") != verdict_status(kind, gained, exhausted):
        raise bad("status %r does not follow from gained=%s, exhausted=%s on a %s"
                  % (verdict.get("status"), gained, exhausted, kind))
    for k in ("best_gain", "best_gain_over_delta", "best_score", "best_vs_record_over_delta", "compute_s"):
        if verdict.get(k) is not None and not _finite(verdict[k]):
            raise bad("%s must be a finite number or null" % k)
    rule = verdict.get("rule") if isinstance(verdict.get("rule"), dict) else {}
    if exhausted and (gained or len(negative) < max(1, int(rule.get("refute_after") or 1))):
        raise bad("exhausted needs no gain and at least %s negative runs" % (rule.get("refute_after") or 1))
    if gained:
        if verdict.get("best_experiment") not in positive:
            raise bad("a gain needs best_experiment, one of its positive runs")
        if verdict.get("criterion") == "claimed_value":
            if not _finite(verdict.get("best_score")):
                raise bad("reaching a claimed value needs best_score")
        else:
            gain = verdict.get("best_gain")
            delta = (verdict.get("reference") or {}).get("delta") if isinstance(verdict.get("reference"), dict) else None
            floor = (rule.get("min_gain_over_delta") or 0.0) * delta if _finite(delta) else 0.0
            if not (_finite(gain) and gain > 0 and gain >= floor * (1 - 1e-9)):
                raise bad("a gain needs best_gain > 0 and above the rule's threshold (%g)" % floor)
    return verdict


def check_runner(runner: Any, idea_id: str = "?") -> Optional[Dict[str, Any]]:
    if runner is None:
        return None
    if not isinstance(runner, dict) or not isinstance(runner.get("config") or {}, dict):
        raise InvalidIdea("%s: runner must be {\"optimizer\": path or null, \"config\": {...}}" % idea_id)
    opt = runner.get("optimizer")
    if opt is not None:
        parts = Path(opt).parts if isinstance(opt, str) else ()
        if (not isinstance(opt, str) or not opt.endswith(".py") or Path(opt).is_absolute()
                or ".." in parts or not parts):
            raise InvalidIdea("%s: runner optimizer must be a relative path to a .py file inside the "
                              "checkout, got %r" % (idea_id, opt))
    out = {"optimizer": opt, "config": dict(runner.get("config") or {})}
    budget = runner.get("min_compute_s")
    if budget is not None:
        if not _finite(budget) or budget <= 0:
            raise InvalidIdea("%s: runner min_compute_s must be a positive number of seconds" % idea_id)
        out["min_compute_s"] = float(budget)
    return out


def read_claim(claims_path: Any, sha256: Optional[str] = None, timestamp: Optional[str] = None
               ) -> Tuple[Dict[str, Any], str]:
    """Read ONE ruling from the arbiter's log (selection/claims.jsonl), by candidate
    sha256 and/or timestamp. Returns (the line as an object, sha256 of the raw line).
    With several matches the last one wins (the arbiter's latest word on that candidate)."""
    if not sha256 and not timestamp:
        raise InvalidIdea("say which ruling: the candidate sha256, the claim timestamp, or both")
    try:
        lines = Path(claims_path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise InvalidIdea("cannot read the claims file %s: %s" % (claims_path, e)) from None
    found = None
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and (not sha256 or row.get("candidate_sha256") == sha256) \
                and (not timestamp or row.get("timestamp") == timestamp):
            found = (row, hashlib.sha256(line.encode("utf-8")).hexdigest())
    if found is None:
        raise InvalidIdea("no ruling with sha256=%s timestamp=%s in %s" % (sha256, timestamp, claims_path))
    return found


def referee_entry(claim: Any, problem: str, experiment_id: Optional[str] = None,
                  claims_file: Optional[str] = None, line_sha256: Optional[str] = None) -> Dict[str, Any]:
    """One ruling of WS1's arbiter, from one line of selection/claims.jsonl."""
    if not isinstance(claim, dict):
        raise InvalidIdea("a referee ruling must be one JSON line of selection/claims.jsonl")
    word, sha = claim.get("verdict"), claim.get("candidate_sha256")
    if word not in REFEREE_WORDS:
        raise InvalidIdea("unknown referee verdict %r (expected one of %s)" % (word, ", ".join(REFEREE_WORDS)))
    if claim.get("slug") != problem:
        raise InvalidIdea("the claim is about problem %r, the idea about %r" % (claim.get("slug"), problem))
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise InvalidIdea("the claim line has no candidate_sha256")
    if not isinstance(claim.get("timestamp"), str) or not claim["timestamp"]:
        raise InvalidIdea("the claim line has no timestamp")
    pinned, indep, live = claim.get("pinned") or {}, claim.get("independent") or {}, claim.get("live_best") or {}
    delta, gain = claim.get("delta"), claim.get("beats_record_by")
    return {"verdict": word, "claim_timestamp": claim["timestamp"],
            "candidate_file": claim.get("candidate_file"), "candidate_sha256": sha,
            "pinned_score": pinned.get("score"), "independent_score": indep.get("score"),
            "record_value": live.get("value"), "record_read_at": live.get("read_at"), "delta": delta,
            "gain": gain, "gain_over_delta": gain / delta if _finite(gain) and _finite(delta) and delta else None,
            "experiment_id": experiment_id, "claims_file": claims_file, "claim_line_sha256": line_sha256}


def _derive_status(node: Dict[str, Any]) -> str:
    verdict = node.get("verdict") or {}
    return verdict["status"] if verdict.get("status") in OURS else node["claimed_status"]


def make_node(raw: Dict[str, Any], allow_verified: bool = False) -> Dict[str, Any]:
    """Validate and normalise one node AS WRITTEN BY ITS SOURCE. Raises InvalidIdea.
    Whatever the row says, the result has no verdict, no runner, no referee ruling,
    no retest and is not trusted: those are written only by this code (see IdeaStore)."""
    if not isinstance(raw, dict):
        raise InvalidIdea("a node must be a JSON object, got %s" % type(raw).__name__)
    idea_id = check_id(raw.get("id"))
    kind = raw.get("kind")
    if kind not in KINDS:
        raise InvalidIdea("%s: kind must be one of %s, got %r" % (idea_id, "|".join(KINDS), kind))
    problem, content = raw.get("problem"), raw.get("content")
    if not isinstance(problem, str) or not problem.strip():
        raise InvalidIdea("%s: problem is missing" % idea_id)
    if not isinstance(content, str) or not content.strip():
        raise InvalidIdea("%s: content is missing" % idea_id)
    citations = raw.get("citations") or []
    if isinstance(citations, str):
        citations = [citations]
    if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
        raise InvalidIdea("%s: citations must be a list of strings" % idea_id)

    claimed = raw.get("claimed_status") or raw.get("status") or ("idea" if kind == "untested_idea" else "claim")
    if claimed in OURS:                    # "reproduced" with no claimed_status: not a source status
        raise InvalidIdea("%s: status %r is set only by our experiments (verdict)" % (idea_id, claimed))
    if claimed not in CLAIMED:
        raise InvalidIdea("%s: status must be one of %s, got %r" % (idea_id, "|".join(CLAIMED), claimed))
    if claimed == "verified" and not allow_verified:
        raise InvalidIdea("%s: only WS1 / the referee may write status 'verified'" % idea_id)
    claimed_value = raw.get("claimed_value")
    if claimed_value is not None and not _finite(claimed_value):
        raise InvalidIdea("%s: claimed_value must be a finite number" % idea_id)

    return {"id": idea_id, "kind": kind, "problem": problem.strip(),
            "content": content.strip()[:MAX_CONTENT], "citations": [c.strip() for c in citations if c.strip()],
            "status": claimed, "author": str(raw.get("author") or "unknown"),
            "source": str(raw.get("source") or ""), "written_on": str(raw.get("written_on") or utc_now()[:10]),
            "claimed_status": claimed, "claimed_value": claimed_value,
            "verdict": None, "runner": None, "referee": [], "trusted": False, "retest": None}


def with_verdict(node: Dict[str, Any], verdict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A copy of a node carrying this verdict (validated), with the status that follows."""
    out = dict(node, verdict=None if verdict is None else check_verdict(node["kind"], verdict, node["id"]))
    out["status"] = _derive_status(out)
    return out


def _stored(raw: Any) -> Dict[str, Any]:
    """Validate one line of the store, map-owned fields included."""
    if not isinstance(raw, dict):
        raise InvalidIdea("not a JSON object")
    node = make_node(dict(raw, status=raw.get("claimed_status")), allow_verified=True)
    node["runner"] = check_runner(raw.get("runner"), node["id"])
    referee = raw.get("referee") or []
    if not isinstance(referee, list) or not all(isinstance(x, dict) and x.get("verdict") in REFEREE_WORDS
                                                 for x in referee):
        raise InvalidIdea("%s: malformed referee rulings" % node["id"])
    node["referee"] = referee
    node["trusted"] = raw.get("trusted") is True
    retest = raw.get("retest")
    if retest is not None and not (isinstance(retest, dict) and isinstance(retest.get("forgiven"), list)):
        raise InvalidIdea("%s: malformed retest" % node["id"])
    node["retest"] = retest
    return with_verdict(node, raw.get("verdict"))


def is_dead(node: Dict[str, Any]) -> bool:
    """A dead end must never be proposed again: a dead_end node (unless our own
    experiments refuted the dead end, i.e. it gained), or any idea we refuted."""
    if node["kind"] == "dead_end":
        return node["status"] != "refuted"
    return node["status"] == "refuted"


def is_frontier(node: Dict[str, Any]) -> bool:
    """Still worth an experiment: a testable idea that is not dead."""
    return node["kind"] in TESTABLE and not is_dead(node)


def same_problem(node_problem: str, record_problem: Any, aliases: Optional[Dict[str, str]] = None) -> bool:
    """Does an experiment record belong to the problem of a node? WS2 names problems
    by short ids ("1a"); aliases maps a record's problem id to the map's slug."""
    return record_problem == node_problem or (aliases or {}).get(record_problem) == node_problem


# ------------------------------------------------------------------ the idea=<id> tag
def _split(rationale: Optional[str]) -> Tuple[List[Tuple[str, str]], str]:
    text = rationale or ""
    m = _HEAD.match(text)
    if not m:
        return [], text.strip()
    return _TAG.findall(m.group(0)), text[m.end():].strip()


def tag_rationale(rationale: Optional[str], idea_id: str) -> str:
    """Put idea=<id> in the leading run of tags, right after strategy=...; parent=...
    Replaces an idea tag that is already there; never touches the rest of the text."""
    check_id(idea_id)
    tags, rest = _split(rationale)
    head = ["%s=%s" % (k, v) for k, v in tags if k != "idea"] + ["idea=%s" % idea_id]
    return "; ".join(head + ([rest] if rest else []))


def idea_of(record_or_rationale: Any) -> Optional[str]:
    """The idea an experiment was run for, or None. Accepts a record or a rationale.
    Only the idea=<id> tag at the head of the rationale counts."""
    text = record_or_rationale
    if isinstance(text, dict):
        text = text.get("rationale")
    if not isinstance(text, str):
        return None
    for k, v in _split(text)[0]:
        if k == "idea":
            v = v.rstrip(".")
            return v if ID_RE.fullmatch(v) else None
    return None


# ------------------------------------------------------------------ local store
class IdeaStore:
    """Local JSONL snapshot of the map: one node per line.
    Writers lock, re-read, change, then rename a unique temp file over the store."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else default_path()

    # ---- reading (no lock needed: the file only ever changes by an atomic rename)
    def all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        nodes, seen = [], set()
        for n, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                node = _stored(json.loads(line))
            except ValueError as e:                    # bad JSON or InvalidIdea
                raise CorruptStore("idea store %s, line %d: %s" % (self.path, n, e)) from None
            if node["id"] in seen:
                raise CorruptStore("idea store %s, line %d: duplicate id %s" % (self.path, n, node["id"]))
            seen.add(node["id"])
            nodes.append(node)
        return nodes

    def get(self, idea_id: str) -> Optional[Dict[str, Any]]:
        return next((n for n in self.all() if n["id"] == idea_id), None)

    def _write(self, nodes: List[Dict[str, Any]]) -> None:
        atomic_write_text(self.path, "".join(json.dumps(n, ensure_ascii=False) + "\n" for n in nodes))

    # ---- writing
    def upsert(self, raw: Dict[str, Any], trusted: bool = False, owner: Optional[str] = None) -> str:
        """Add a node, or refresh it from its source. Returns "new", "updated" or "unchanged".
        trusted=True (WS1's file): the source fields are refreshed and the node is
        marked trusted. What we learned ourselves always survives (verdict, runner,
        referee rulings, retest); the verdict is dropped only if the kind changed,
        because its words depend on the kind.
        trusted=False (a scout, any other file): can ADD. An id that already exists is
        refused unless the row says exactly what the node says, with one exception:
        owner="name" makes every row be attributed to that name, and lets it refresh
        the UNTRUSTED nodes that name wrote, except the kind of a node that already has
        a verdict. A node written by a trusted import is out of reach, whatever the
        declared owner."""
        if owner is not None and not trusted and isinstance(raw, dict):
            raw = dict(raw, author=owner)             # a file cannot speak in somebody else's name
        node = make_node(raw, allow_verified=trusted)
        node["trusted"] = bool(trusted)
        with file_lock(self.path):
            nodes = self.all()
            for i, old in enumerate(nodes):
                if old["id"] != node["id"]:
                    continue
                if not (isinstance(raw, dict) and raw.get("written_on")):
                    node["written_on"] = old["written_on"]      # an undated row is not a change of date
                same_kind = old["kind"] == node["kind"]
                merged = dict(node, runner=old["runner"], referee=old["referee"], retest=old["retest"],
                              trusted=old["trusted"] or bool(trusted),
                              claimed_value=node["claimed_value"] if node["claimed_value"] is not None
                              else old["claimed_value"])
                merged = with_verdict(merged, old["verdict"] if same_kind else None)
                if merged == old:
                    return "unchanged"
                if not trusted:
                    if old["trusted"]:
                        raise InvalidIdea("%s was written by a trusted source (%s): an untrusted file can "
                                          "never change it" % (node["id"], old["author"]))
                    if owner is None or old["author"] != owner:
                        raise InvalidIdea("%s already exists (written by %s): an untrusted file can only "
                                          "add new ideas or refresh its owner's" % (node["id"], old["author"]))
                    if not same_kind and old["verdict"] is not None:
                        raise InvalidIdea("%s already has a verdict from our experiments: its kind cannot "
                                          "be changed by an untrusted file" % node["id"])
                    alike = same_idea(old["content"], node["content"])
                    if alike < SAME_IDEA:
                        raise InvalidIdea("%s would become a different idea (text similarity %.2f): an id names "
                                          "one idea for ever, verdicts are attached to it. Use a new id; ids "
                                          "must not come from row numbers" % (node["id"], alike))
                nodes[i] = merged
                self._write(nodes)
                return "updated"
            nodes.append(node)
            self._write(nodes)
            return "new"

    def update(self, idea_id: str, change) -> Dict[str, Any]:
        """One locked read-modify-write of a node: change(node) -> node. The node is
        validated again before it is written. Use it for anything that must not lose
        a concurrent writer's work (e.g. merging a verdict)."""
        with file_lock(self.path):
            nodes = self.all()
            for i, old in enumerate(nodes):
                if old["id"] == idea_id:
                    node = _stored(change(dict(old)))
                    if node != old:
                        nodes[i] = node
                        self._write(nodes)
                    return node
        raise KeyError("unknown idea %r" % idea_id)

    def set_verdict(self, idea_id: str, verdict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """The only writer of a verdict. Refused on records and facts, and unless the
        verdict shows its evidence (check_verdict)."""
        return self.update(idea_id, lambda n: with_verdict(n, verdict))

    def set_runner(self, idea_id: str, optimizer: Optional[str], config: Optional[Dict[str, Any]] = None,
                   min_compute_s: Optional[float] = None) -> Dict[str, Any]:
        """min_compute_s: the fair budget of the idea. It cannot be decided against
        before its negative runs add up to that much compute."""
        def change(n):
            if n["kind"] not in TESTABLE:
                raise InvalidIdea("%s is a %s: nothing to run" % (idea_id, n["kind"]))
            return dict(n, runner=check_runner({"optimizer": optimizer, "config": config or {},
                                                "min_compute_s": min_compute_s}, idea_id))
        return self.update(idea_id, change)

    def clear_runner(self, idea_id: str) -> Dict[str, Any]:
        return self.update(idea_id, lambda n: dict(n, runner=None))

    def set_claimed_value(self, idea_id: str, value: Optional[float]) -> Dict[str, Any]:
        if value is not None and not _finite(value):
            raise InvalidIdea("%s: claimed_value must be a finite number" % idea_id)
        return self.update(idea_id, lambda n: dict(n, claimed_value=value))

    def add_referee(self, idea_id: str, claims_path: Any, sha256: Optional[str] = None,
                    timestamp: Optional[str] = None, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Attach a ruling of selection/verify_claim.py. The ruling is READ from the
        arbiter's own log (claims_path), never typed. With experiment_id, the ruled
        candidate must be that run's saved solution (same sha256). It never changes
        `status`. The same ruling (candidate hash + timestamp) is stored once."""
        claim, line_sha = read_claim(claims_path, sha256, timestamp)

        def change(n):
            entry = referee_entry(claim, n["problem"], experiment_id, str(Path(claims_path).resolve()), line_sha)
            if experiment_id is not None:
                runs = [r for r in (n["verdict"] or {}).get("runs", []) if experiment_id in (run_key(r), r["experiment_id"])]
                if not runs:
                    raise InvalidIdea("%s has no run %s" % (idea_id, experiment_id))
                if entry["candidate_sha256"] not in {r.get("solution_sha256") for r in runs}:
                    raise InvalidIdea("the ruled candidate (%s...) is not the saved solution of run %s"
                                      % (entry["candidate_sha256"][:12], experiment_id))
            key = (entry["candidate_sha256"], entry["claim_timestamp"])
            if any((x.get("candidate_sha256"), x.get("claim_timestamp")) == key for x in n["referee"]):
                return n
            return dict(n, referee=n["referee"] + [entry])
        return self.update(idea_id, change)


def next_id(store: IdeaStore, prefix: str) -> str:
    """Next free id "<prefix>-NNN" (for ideas that arrive without one, e.g. from a scout)."""
    check_id(prefix)
    used = {n["id"] for n in store.all()}
    i = 1
    while "%s-%03d" % (prefix, i) in used:
        i += 1
    return "%s-%03d" % (prefix, i)


# ------------------------------------------------------------------ shared memory mirror
def _plain(text: str) -> str:
    """Idea text is quoted as evidence inside WS2 rationales, where two parsers scan it:
    EvidenceHints takes the first {...} as a config hint, and Knowledge reads every
    `strategy=`/`parent=` it finds. Neutralise both: no braces, no key=value token."""
    return _KEY_VALUE.sub(r"\1 = ", text.replace("{", "(").replace("}", ")"))


def fact_id_for(idea_id: str, scope: Optional[str] = None) -> str:
    """Stable memory id of a node. The scope is part of it outside the map's own scope,
    so mirroring under another scope never moves the facts of this one."""
    return "idea-" + idea_id if scope in (None, SCOPE) else "idea-%s--%s" % (idea_id, scope)


def fact_for(node: Dict[str, Any], scope: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
    """(fact_id, text, metadata) of the memory fact that mirrors a node. Writing the
    same id again replaces the fact, so the mirror follows the node's verdict."""
    verdict = node.get("verdict") or {}
    text = "Idea %s (%s, status %s) on problem %s: %s" % (
        node["id"], node["kind"], node["status"], node["problem"], node["content"])
    if node["citations"]:
        text += " Citations: %s." % "; ".join(node["citations"])
    if verdict.get("status") in OURS:
        text += " Our experiments: %s (%s), best gain %s delta." % (
            verdict["status"], ", ".join(verdict["experiments"]), verdict.get("best_gain_over_delta"))
    rulings = [r["verdict"] for r in node.get("referee") or []]
    if rulings:
        text += " Referee: %s." % ", ".join(rulings)
    meta = {"problem": node["problem"], "idea_id": node["id"], "idea_kind": node["kind"],
            "status": node["status"], "claimed_status": node["claimed_status"],
            "citations": node["citations"], "source": node["source"], "written_on": node["written_on"],
            "dead": is_dead(node), "experiments": verdict.get("experiments") or [],
            "best_gain_over_delta": verdict.get("best_gain_over_delta"), "referee": rulings,
            "imported": True}
    return fact_id_for(node["id"], scope), _plain(text), meta


def mirror(memory, node: Dict[str, Any], scope: Optional[str] = SCOPE) -> str:
    """Write (or replace) the memory fact of a node, attributed to the node's author."""
    fact_id, text, meta = fact_for(node, scope)
    return memory.write(text, author=node["author"], kind="idea",
                        label="%s %s" % (node["problem"], node["kind"]), metadata=meta,
                        wait=False, scope=scope, fact_id=fact_id)


# ------------------------------------------------------------------ cli
def _memory(choice: str):
    from .pastdev_client import LocalMemory, get_memory
    return None if choice == "none" else LocalMemory() if choice == "local" else get_memory()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ideas", help="idea store file (default: %s)" % default_path())
    ap.add_argument("--memory", choices=("auto", "local", "none"), default="local",
                    help="where to mirror a change (default local: no network call)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list", help="print the nodes")
    s.add_argument("--problem")
    s = sub.add_parser("add", help="add one new idea (untrusted: it can never overwrite a node)")
    s.add_argument("--problem", required=True)
    s.add_argument("--content", required=True)
    s.add_argument("--kind", default="untested_idea", choices=TESTABLE)
    s.add_argument("--cite", action="append", default=[])
    s.add_argument("--author", required=True)
    s.add_argument("--source", default="")
    s.add_argument("--id", help="default: <author>-NNN")
    s.add_argument("--claimed-value", type=float)
    s = sub.add_parser("runner", help="say how an idea is executed")
    s.add_argument("id")
    s.add_argument("--optimizer", help="relative path of the optimizer file")
    s.add_argument("--config", default="{}", help="JSON object of config overrides")
    s.add_argument("--min-compute", type=float, help="fair budget in seconds before the idea may be refuted")
    s.add_argument("--clear", action="store_true")
    s = sub.add_parser("claimed-value", help="the value a `worked` claim reported")
    s.add_argument("id")
    s.add_argument("value", type=float)
    s = sub.add_parser("referee", help="attach a ruling of selection/verify_claim.py")
    s.add_argument("id")
    s.add_argument("--claims", required=True, help="path of the arbiter's selection/claims.jsonl")
    s.add_argument("--sha", help="candidate_sha256 of the ruling")
    s.add_argument("--timestamp", help="timestamp of the ruling")
    s.add_argument("--experiment", help="the run this candidate came from (its solution sha256 must match)")
    a = ap.parse_args(argv)
    store = IdeaStore(a.ideas)
    try:
        if a.cmd == "list":
            for n in store.all():
                if a.problem in (None, n["problem"]):
                    flag = "DEAD" if is_dead(n) else "frontier" if is_frontier(n) else "-"
                    print("%-12s %-13s %-10s %-8s %s%s" % (n["id"], n["kind"], n["status"], flag,
                                                         "[runner] " if n["runner"] else "", n["content"][:100]))
            return 0
        if a.cmd == "add":
            prefix = re.sub(r"[^a-z0-9]+", "-", a.author.lower()).strip("-") or "idea"
            raw = {"id": a.id or next_id(store, prefix), "kind": a.kind, "problem": a.problem,
                   "content": a.content, "citations": a.cite, "author": a.author, "source": a.source,
                   "claimed_value": a.claimed_value}
            outcome = store.upsert(raw, trusted=False)
            node = store.get(raw["id"])
            print("%s %s" % (outcome, node["id"]))
        elif a.cmd == "runner":
            node = store.clear_runner(a.id) if a.clear else store.set_runner(
                a.id, a.optimizer, json.loads(a.config), a.min_compute)
            print("%s runner: %s" % (node["id"], json.dumps(node["runner"])))
        elif a.cmd == "claimed-value":
            node = store.set_claimed_value(a.id, a.value)
            print("%s claimed value: %r" % (node["id"], node["claimed_value"]))
        else:
            node = store.add_referee(a.id, a.claims, a.sha, a.timestamp, a.experiment)
            print("%s referee rulings: %s" % (node["id"], ", ".join(r["verdict"] for r in node["referee"])))
    except (ValueError, KeyError) as e:
        print("error: %s" % (e.args[0] if e.args else e), file=sys.stderr)
        return 1
    memory = _memory(a.memory)
    if memory is not None:
        mirror(memory, node)
    return 0


if __name__ == "__main__":
    sys.exit(main())
