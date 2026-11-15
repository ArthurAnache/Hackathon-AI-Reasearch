"""Offline tests for the research map: idea nodes, the WS1 knowledge importer, the
idea=<id> tag, verdicts, the idea chooser and the static page.

    python -m pytest tests/test_research_map.py      or      python tests/test_research_map.py

No network, no keys: shared memory is the local JSONL fallback inside a temp directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
if "integrations._common" not in sys.modules:      # else test_integrations.py already isolated the state
    _T = tempfile.mkdtemp()
    os.environ.update(INTEGRATIONS_ENV_FILE=os.path.join(_T, "no.env"),   # never read the real .env
                      INTEGRATIONS_STATE_DIR=os.path.join(_T, "state"),
                      EXA_LOG_PATH=os.path.join(_T, "exa_log.jsonl"),
                      PASTDEV_LOG_PATH=os.path.join(_T, "pastdev_log.jsonl"))
for _v in ("EXA_API_KEYS", "EXA_API_KEY", "PAST_API_KEYS", "PAST_API_KEY", "RESEARCH_MAP_IDEAS"):
    os.environ.pop(_v, None)

try:                                   # real WS2 contract when present, else a minimal stand-in
    import interfaces  # noqa: F401
except Exception:
    _stub = types.ModuleType("interfaces")

    class _JsonlStore:
        def __init__(self, path):
            self.path = path

        def append(self, r):
            with open(self.path, "a") as f:
                f.write(json.dumps(r) + "\n")

        def read(self):
            return [json.loads(l) for l in open(self.path)] if os.path.exists(self.path) else []

    _stub.JsonlStore = _JsonlStore
    sys.modules["interfaces"] = _stub

from integrations import (_common, evidence, idea_chooser, ideas, import_knowledge, import_results,  # noqa: E402
                          pastdev_client, research_map, verdicts)
from integrations.idea_agent import DeadIdea, IdeaAgent, WrongProblem  # noqa: E402

assert str(_common.STATE_DIR).startswith(tempfile.gettempdir()), "tests must never touch the real state dir"
TMP = tempfile.mkdtemp()
PY = sys.executable
SLUG = "second-autocorrelation-inequality"
RECORD, DELTA = 0.9636615620620852, 1e-5
REF = verdicts.Reference(problem=SLUG, delta=DELTA, record=RECORD, record_read_at="2026-11-14T22:41:00Z",
                         source="test", live=True)                      # the record as read from the leaderboard
TYPED = verdicts.Reference(problem=SLUG, delta=DELTA, record=RECORD, source="typed, not read")
NO_RECORD = verdicts.Reference(problem=SLUG, delta=DELTA, source="test, record not read")
LONG = 1300.0                          # seconds: three such negative runs exceed the default fair budget (3600 s)
# WS2's own tag regex (research/knowledge.py on feature/ws2-autoresearch-loop), copied to prove
# that a rationale carrying idea=<id> still reads exactly as WS2 expects.
WS2_TAG = re.compile(r"\b(strategy|parent)=([\w\-.]+)")

ROWS = [
    {"id": "ws1-001", "author": "ws1-selection", "kind": "record", "problem": SLUG,
     "content": "Live #1 on 2026-11-14: 0.963638667992813 (MLAI_YONSEI); delta 1e-5; target >= 0.963648667992813.",
     "citations": [], "status": "verified", "source": "selection/registry_stage2.jsonl", "written_on": "2026-11-14"},
    {"id": "ws1-002", "author": "ws1-selection", "kind": "fact", "problem": SLUG,
     "content": "Closed form: C = (2 sum c_k^2 + sum c_k c_{k+1}) / (3 sum c_k max c_k).",
     "citations": ["einsteinarena thread t32/r70"], "status": "verified", "source": "digest", "written_on": "2026-11-14"},
    {"id": "ws1-007", "author": "ws1-selection", "kind": "worked", "problem": SLUG,
     "content": "Iterated Dinkelbach: 0.96199 at 100k.", "citations": ["einsteinarena thread t133/r279"],
     "status": "claim", "source": "digest", "written_on": "2026-11-14"},
    {"id": "ws1-010", "author": "ws1-selection", "kind": "dead_end", "problem": SLUG,
     "content": "Dense re-optimization of all coordinates is much worse than packet-level moves.",
     "citations": ["einsteinarena thread t32"], "status": "claim", "source": "digest", "written_on": "2026-11-14"},
    {"id": "ws1-017", "author": "ws1-selection", "kind": "untested_idea", "problem": SLUG,
     "content": "Full-support n = 2M run of the whole Dinkelbach pipeline.",
     "citations": ["einsteinarena thread t255", "einsteinarena thread t127/r1063"], "status": "idea",
     "source": "digest", "written_on": "2026-11-14"},
    {"id": "ws1-018", "author": "ws1-selection", "kind": "untested_idea", "problem": SLUG,
     "content": "Finer comb (4x finer than Hyra) at larger n.", "citations": ["einsteinarena thread t254"],
     "status": "idea", "source": "digest", "written_on": "2026-11-14"},
    {"id": "ws1-025", "author": "ws1-selection", "kind": "untested_idea", "problem": "circles-rectangle",
     "content": "Another contact graph.", "citations": [], "status": "idea", "source": "digest",
     "written_on": "2026-11-14"},
]
BY_ID = {r["id"]: r for r in ROWS}


def _fresh(name):
    """A new directory for one test, and a clean shared state (local memory, scopes)."""
    shutil.rmtree(os.environ["INTEGRATIONS_STATE_DIR"], ignore_errors=True)
    d = os.path.join(TMP, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def _store(d, rows=ROWS):
    """An idea store filled by a trusted import (as WS1's file is)."""
    store = ideas.IdeaStore(os.path.join(d, "ideas.jsonl"))
    for raw in rows:
        store.upsert(raw, trusted=True)
    return store


def _write_rows(d, rows, name="knowledge.jsonl"):
    path = os.path.join(d, name)
    with open(path, "w") as f:
        f.write("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return path


def _idea_facts(scope=ideas.SCOPE, path=None):
    facts = pastdev_client.LocalMemory(path).read("idea", reader="test", scope=scope, limit=500)
    return [f for f in facts if f["kind"] == "idea"]


def _raises(exc, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except exc as e:
        return e
    raise AssertionError("expected %s from %s%r" % (exc.__name__, getattr(fn, "__name__", fn), args))


def _run(eid, evidence="none", gain=None, dur=60.0):
    score = None if gain is None else RECORD + gain
    return {"experiment_id": eid, "store": "", "key": eid, "problem": SLUG, "status": "keep", "valid": score is not None,
            "score": score, "seed": 42, "budget_s": 60.0, "duration_s": dur, "solution_path": None,
            "solution_sha256": None, "bar": RECORD, "bar_source": "record read t", "gain": gain,
            "gain_over_delta": None if gain is None else gain / DELTA, "local_gain": None,
            "vs_record_over_delta": None if gain is None else gain / DELTA, "evidence": evidence,
            "record_candidate": False}


def _verdict(kind, gained, exhausted=False, exps=("exp_0001",), gain=None):
    """A well-formed verdict, as integrations.verdicts would build it."""
    gain = gain if gain is not None else (3e-6 if gained else -1e-4)
    if gained:
        runs = [_run(exps[0], "positive", gain)] + [_run(e, "baseline", -1e-4) for e in exps[1:]]
    else:
        names = list(exps) + ["neg_%d" % i for i in range(max(0, 3 - len(exps)))] if exhausted else list(exps)
        runs = [_run(e, "negative" if exhausted else "baseline", gain, LONG) for e in names]
    return {"status": ideas.verdict_status(kind, gained, exhausted), "gained": gained, "exhausted": exhausted,
            "criterion": "gain", "experiments": [r["key"] for r in runs],
            "best_experiment": runs[0]["key"], "best_gain": gain, "best_gain_over_delta": gain / DELTA,
            "trials": len(runs), "compute_s": sum(r["duration_s"] for r in runs), "runs": runs,
            "reference": REF.as_dict(),
            "rule": {"min_gain_over_delta": 0.01, "refute_after": 3, "min_compute_s": 3600.0}}


_N = [0]


def rec(idea, score, best_before, status=None, dur=60.0, problem=SLUG, cfg=None, eid=None, **extra):
    """One experiment record, shaped like interfaces.run_experiment writes it."""
    _N[0] += 1
    if status is None:
        status = "keep" if best_before is None or score > best_before else "discard"
    tags = "strategy=refine; parent=exp_0000" + ("; idea=%s" % idea if idea else "") + "; best exp_0000"
    r = {"experiment_id": eid or "exp_%04d" % _N[0], "problem": problem, "status": status, "score": score,
         "best_before": best_before, "duration_s": dur, "config": cfg or {"K": _N[0]}, "seed": 42,
         "budget_s": max(60.0, dur), "agent": "feedback@test", "optimizer": "optimizers/x.py",
         "algorithm_version": "abc",
         "rationale": tags, "timestamp": "2026-11-14T23:00:00+00:00", "solution_path": None, "hypothesis": "h",
         "valid": score is not None, "error": None, "evidence": [], "beats_record": False, "git_commit": None}
    r.update(extra)
    return r


def _evidence(node, history, ref=REF, **kw):
    return [x["evidence"] for x in verdicts.observe(node, history, ref, **kw)]


@dataclass
class Exp:                              # stand-in for interfaces.ExperimentConfig
    hypothesis: str
    config: dict
    optimizer: str = None
    rationale: str = ""


class Inner:                            # stand-in for research.FeedbackAgent
    name = "feedback@test"

    def __init__(self, stop_after=None):
        self.reviews, self.seen, self.stop_after = [], [], stop_after

    def propose(self, history, evidence):
        self.seen.append(list(history))
        return Exp("[refine] start from the best", {"LR": 0.02},
                   rationale="strategy=refine; parent=exp_%04d; best exp_0000 score 1.5" % len(history))

    def review(self, result):
        self.reviews.append(result["status"])
        return "stop" if self.stop_after and len(self.reviews) >= self.stop_after else "continue"


# ====================================================================== 1. idea nodes
def test_node_schema_matches_ws1_knowledge_file():
    for raw in ROWS:
        node = ideas.make_node(raw, allow_verified=True)
        for k in ideas.SOURCE_FIELDS:                       # every WS1 field survives unchanged
            assert node[k] == raw[k], (k, node[k], raw[k])
        assert node["claimed_status"] == raw["status"]
        assert node["verdict"] is None and node["runner"] is None and node["referee"] == []
    assert set(ideas.KINDS) == {r["kind"] for r in ROWS}


def test_node_validation_rejects_bad_input():
    good = BY_ID["ws1-017"]
    bad = [dict(good, id="has space"), dict(good, id="x;idea=y"), dict(good, id=""), dict(good, id="trailing."),
           dict(good, id="ws1-017\n"), dict(good, id="a\u0661b"), dict(good, id="a\u00e9b"), dict(good, id="x" * 65),
           dict(good, kind="rumour"), dict(good, status="proved"), dict(good, status="reproduced"),
           dict(good, content="  "), dict(good, problem=None), dict(good, citations=[1, 2]),
           dict(good, claimed_value=float("nan")), "not a dict"]
    for raw in bad:
        _raises(ideas.InvalidIdea, ideas.make_node, raw, allow_verified=True)
    _raises(ideas.InvalidIdea, ideas.make_node, dict(good, status="verified"))   # WS1's word, not ours
    node = ideas.make_node(dict(good, citations="t255", status=None))
    assert node["citations"] == ["t255"] and node["status"] == "idea"           # defaults for a scout's idea
    assert ideas.check_id("x" * 64) and ideas.check_id("a.b_c-1")


def test_a_source_row_cannot_carry_a_verdict_a_runner_a_ruling_or_trust():
    row = dict(BY_ID["ws1-010"], verdict=_verdict("dead_end", True), runner={"optimizer": "/etc/passwd"},
               referee=[{"verdict": "beats_record"}], trusted=True, retest={"forgiven": ["x"]})
    node = ideas.make_node(row)
    assert node["verdict"] is None and node["runner"] is None and node["referee"] == []
    assert node["trusted"] is False and node["retest"] is None
    assert node["status"] == "claim" and ideas.is_dead(node)


def test_verdict_words_flip_on_dead_ends_and_only_one_function_maps_them():
    table = {("untested_idea", True): "reproduced", ("worked", True): "reproduced", ("dead_end", True): "refuted"}
    for (kind, gained), word in table.items():
        assert ideas.verdict_status(kind, gained, False) == word
        assert ideas.verdict_status(kind, gained, True) == word              # a gain wins over "exhausted"
    assert ideas.verdict_status("untested_idea", False, True) == "refuted"
    assert ideas.verdict_status("dead_end", False, True) == "reproduced"     # it is dead for us too
    for kind in ideas.TESTABLE:
        assert ideas.verdict_status(kind, False, False) == "inconclusive"
    _raises(ideas.InvalidIdea, ideas.verdict_status, "record", True, False)

    nodes = {r["id"]: ideas.make_node(r, allow_verified=True) for r in ROWS}
    assert ideas.is_dead(nodes["ws1-010"]) and not ideas.is_frontier(nodes["ws1-010"])
    assert ideas.is_frontier(nodes["ws1-017"]) and ideas.is_frontier(nodes["ws1-007"])
    assert not ideas.is_frontier(nodes["ws1-001"]) and not ideas.is_frontier(nodes["ws1-002"])   # nothing to test
    refuted = ideas.with_verdict(nodes["ws1-017"], _verdict("untested_idea", False, True))
    assert refuted["status"] == "refuted" and refuted["claimed_status"] == "idea" and ideas.is_dead(refuted)
    # a dead end stays dead when our runs did not gain, whatever word a careless writer would pick
    confirmed = ideas.with_verdict(nodes["ws1-010"], _verdict("dead_end", False, True))
    assert confirmed["status"] == "reproduced" and ideas.is_dead(confirmed)
    revived = ideas.with_verdict(nodes["ws1-010"], _verdict("dead_end", True))     # it GAINED: alive again
    assert revived["status"] == "refuted" and not ideas.is_dead(revived) and ideas.is_frontier(revived)
    undecided = ideas.with_verdict(nodes["ws1-017"], _verdict("untested_idea", False, False))
    assert undecided["status"] == "idea" and ideas.is_frontier(undecided)


def test_a_verdict_must_show_its_evidence():
    ok = _verdict("untested_idea", True)
    assert ideas.check_verdict("untested_idea", ok) is ok
    no_runs = {k: v for k, v in ok.items() if k != "runs"}
    forged = [{"status": "reproduced"}, no_runs, dict(ok, runs=[]), dict(ok, experiments=[]),
              dict(ok, experiments="exp_1"), dict(ok, experiments=["made-up"]),
              dict(ok, runs=[_run("exp_0001", "baseline", 3e-6)]),              # gained, but no positive run
              dict(ok, runs=ok["runs"] * 2, experiments=ok["experiments"] * 2),  # the same id twice
              dict(ok, best_gain=0.0), dict(ok, best_gain=-3), dict(ok, best_gain=float("nan")),
              dict(ok, best_gain=1e-300),                                       # below the rule's own threshold
              dict(ok, best_gain_over_delta=float("inf")), dict(ok, best_experiment="exp_9999"),
              dict(ok, gained=False), dict(ok, gained="yes"), dict(ok, status="refuted"), "reproduced", None]
    for v in forged:
        _raises(ideas.InvalidIdea, ideas.check_verdict, "untested_idea", v)
    dead = _verdict("dead_end", False, True)
    assert ideas.check_verdict("dead_end", dead)["status"] == "reproduced"
    # "our run did not gain" written as 'refuted' on a dead end would revive it: refused
    _raises(ideas.InvalidIdea, ideas.check_verdict, "dead_end", dict(dead, status="refuted"))
    # exhausted with fewer negative runs than the rule asks for
    _raises(ideas.InvalidIdea, ideas.check_verdict, "untested_idea",
            dict(dead, status="refuted", runs=dead["runs"][:2], experiments=dead["experiments"][:2]))
    _raises(ideas.InvalidIdea, ideas.check_verdict, "record", ok)

    d = _fresh("verdict-writer")
    store = _store(d)
    for v in forged[:-1]:
        _raises(ideas.InvalidIdea, store.set_verdict, "ws1-017", v)
    # a hand-made "it gained" on a dead end, with an invented run and a gain of 1e-300: refused, still dead
    made_up = dict(_verdict("dead_end", True, exps=("made-up",)), best_gain=1e-300)
    _raises(ideas.InvalidIdea, store.set_verdict, "ws1-010", made_up)
    assert ideas.is_dead(store.get("ws1-010"))
    _raises(ideas.InvalidIdea, store.set_verdict, "ws1-001", ok)             # a 'verified' record has no verdict
    assert store.get("ws1-001")["status"] == "verified" and store.get("ws1-017")["status"] == "idea"
    assert store.set_verdict("ws1-017", ok)["status"] == "reproduced"
    assert store.set_verdict("ws1-017", None)["status"] == "idea"
    _raises(KeyError, store.set_verdict, "nope-1", ok)


def test_runner_has_one_writer_and_a_fair_budget():
    d = _fresh("runner")
    store = _store(d)
    assert store.set_runner("ws1-017", "optimizers/dinkelbach_2m.py", {"N": 2000000})["runner"] == {
        "optimizer": "optimizers/dinkelbach_2m.py", "config": {"N": 2000000}}
    assert store.set_runner("ws1-017", "optimizers/dinkelbach_2m.py", min_compute_s=7200)["runner"] == {
        "optimizer": "optimizers/dinkelbach_2m.py", "config": {}, "min_compute_s": 7200.0}
    for bad in ("/etc/passwd", "../outside.py", "optimizers/x.sh", "", 7):
        _raises(ideas.InvalidIdea, store.set_runner, "ws1-017", bad)
    _raises(ideas.InvalidIdea, store.set_runner, "ws1-017", "optimizers/x.py", {}, -5)
    _raises(ideas.InvalidIdea, store.set_runner, "ws1-001", "optimizers/x.py")       # nothing to run on a record
    assert store.clear_runner("ws1-017")["runner"] is None
    assert ideas.main(["--ideas", str(store.path), "--memory", "none", "runner", "ws1-017", "--optimizer",
                       "optimizers/a.py", "--config", '{"N": 5}', "--min-compute", "900"]) == 0
    assert store.get("ws1-017")["runner"] == {"optimizer": "optimizers/a.py", "config": {"N": 5}, "min_compute_s": 900.0}


def test_a_referee_ruling_is_read_from_the_arbiters_file_never_typed():
    d = _fresh("referee")
    store = _store(d)
    sol = os.path.join(d, "sol.json")
    open(sol, "wb").write(b"candidate")
    sha = hashlib.sha256(b"candidate").hexdigest()
    hist = [rec("ws1-017", RECORD + 2e-6, RECORD, eid="exp_a", solution_path="sol.json"),
            rec("ws1-017", RECORD + 3e-6, RECORD + 2e-6, eid="exp_b", solution_path="missing.json")]
    verdicts.update_verdict(store, "ws1-017", hist, REF, hash_root=Path(d))
    assert store.get("ws1-017")["verdict"]["runs"][0]["solution_sha256"] == sha
    claim = {"timestamp": "2026-11-15T01:00:00+00:00", "slug": SLUG, "candidate_file": "/tmp/c.json",
             "candidate_sha256": sha, "delta": DELTA, "pinned": {"ok": True, "score": RECORD + 2e-6},
             "independent": {"ok": True, "score": RECORD + 2e-6},
             "live_best": {"value": RECORD, "read_at": "2026-11-15T01:00:01Z"},
             "beats_record_by": 2e-6, "verdict": "valid_below_record"}
    lines = [json.dumps(dict(claim, slug="circles-rectangle", timestamp="t-other")),
             json.dumps(claim), "{torn", json.dumps(dict(claim, verdict="verified", timestamp="t-word"))]
    os.makedirs(os.path.join(d, "selection"))
    claims = os.path.join(d, "selection", "claims.jsonl")
    open(claims, "w").write("\n".join(lines) + "\n")
    # a file that merely looks like the arbiter's log is not the arbiter's log
    e = _raises(ideas.InvalidIdea, store.add_referee, "ws1-017", claims, "exp_a", sha256=sha)
    assert "not the arbiter's log" in str(e)
    handmade = os.path.join(d, "my_claims.jsonl")
    shutil.copy(claims, handmade)
    open(os.path.join(d, "selection", "verify_claim.py"), "w").write("# the arbiter\n")
    _raises(ideas.InvalidIdea, store.add_referee, "ws1-017", handmade, "exp_a", sha256=sha)
    node = store.add_referee("ws1-017", claims, "exp_a", sha256=sha, timestamp=claim["timestamp"])
    assert node["status"] == "reproduced"                                  # from our runs; a ruling never moves it
    r = node["referee"][0]
    assert r["verdict"] == "valid_below_record" and abs(r["gain_over_delta"] - 0.2) < 1e-9
    assert r["candidate_sha256"] == sha and r["record_value"] == RECORD and r["experiment_id"] == "exp_a"
    assert r["record_read_at"] == "2026-11-15T01:00:01Z" and r["pinned_score"] == r["independent_score"]
    assert r["claims_file"] == str(Path(claims).resolve())
    assert r["claim_line_sha256"] == hashlib.sha256(lines[1].encode()).hexdigest()
    assert len(store.add_referee("ws1-017", claims, "exp_a", timestamp=claim["timestamp"])["referee"]) == 1   # once
    # nothing that is not in the file, and nothing about another candidate, run, problem or word
    for exp, kw in (("exp_a", {"sha256": "ab" * 32}), ("exp_a", {"timestamp": "never"}), ("exp_a", {}),
                    ("exp_a", {"timestamp": "t-other"}), ("exp_a", {"timestamp": "t-word"}),
                    ("exp_b", {"sha256": sha}), ("exp_zzz", {"sha256": sha}), ("", {"sha256": sha})):
        _raises(ideas.InvalidIdea, store.add_referee, "ws1-017", claims, exp, **kw)
    _raises(ideas.InvalidIdea, store.add_referee, "ws1-018", claims, "exp_a", sha256=sha)   # not that idea's run
    open(claims, "a").write(json.dumps(dict(claim, timestamp="t2", verdict="beats_record")) + "\n")
    assert ideas.main(["--ideas", str(store.path), "--memory", "none", "referee", "ws1-017", "--claims", claims,
                       "--timestamp", "t2", "--experiment", "exp_a"]) == 0
    node = store.get("ws1-017")
    assert [(x["verdict"], x["experiment_id"]) for x in node["referee"]] == [("valid_below_record", "exp_a"),
                                                                             ("beats_record", "exp_a")]
    assert "Referee: valid_below_record, beats_record." in ideas.fact_for(node)[1]


def test_store_fails_closed_on_a_corrupt_file_and_names_the_line():
    d = _fresh("corrupt")
    store = _store(d)
    good = store.path.read_text()
    for extra, needle in (("{not json\n", "line 8"), (json.dumps({"id": "x-1", "problem": SLUG}) + "\n", "line 8"),
                          (good.splitlines()[0] + "\n", "duplicate id ws1-001"),
                          (json.dumps(dict(store.get("ws1-017"), verdict={"status": "reproduced"})) + "\n", "line 8")):
        store.path.write_text(good + extra)
        e = _raises(ideas.CorruptStore, store.all)
        assert needle in str(e) and str(store.path) in str(e), str(e)
        _raises(ideas.CorruptStore, store.upsert, BY_ID["ws1-017"], trusted=True)
        assert store.path.read_text() == good + extra                      # a failed write changes nothing
    store.path.write_text(good)
    assert len(store.all()) == len(ROWS)


_WRITER = """
import sys
sys.path.insert(0, %r)
from integrations import ideas, verdicts
store = ideas.IdeaStore(sys.argv[1])
who, n = sys.argv[2], int(sys.argv[3])
ref = verdicts.Reference(problem="p", delta=1e-5, record=0.5, record_read_at="t")
for i in range(n):
    if who.startswith("arm"):
        # one comparison arm = one experiment store; arm-a gains, arm-b never does
        gain = 5e-6 * (i + 1) if who == "arm-a" else -1e-3 * (i + 1)
        rows = [{"experiment_id": "exp_%%04d" %% j, "problem": "p", "status": "keep", "score": 0.5 + (
                 5e-6 * (j + 1) if who == "arm-a" else -1e-3 * (j + 1)), "best_before": None, "duration_s": 2000.0,
                 "config": {"j": j}, "seed": 1, "budget_s": 60, "rationale": "strategy=x; parent=none; idea=shared-1"}
                for j in range(i + 1)]
        verdicts.update_verdict(store, "shared-1", verdicts.labelled(rows, who), ref)
    else:
        store.upsert({"id": "%%s-%%03d" %% (who, i), "kind": "untested_idea", "problem": "p", "content": "c"})
""" % ROOT


def test_concurrent_writers_do_not_corrupt_or_lose_anything():
    d = _fresh("race")
    store = _store(d)
    store.upsert({"id": "shared-1", "kind": "untested_idea", "problem": "p", "content": "tested by two arms"})
    n = 30
    procs = [subprocess.Popen([PY, "-c", _WRITER, str(store.path), who, str(n)], stderr=subprocess.PIPE)
             for who in ("a", "b", "c", "arm-a", "arm-b")]
    errors = [p.communicate()[1].decode() for p in procs]
    assert all(p.returncode == 0 for p in procs), errors
    nodes = store.all()                                                    # still readable, nothing lost
    assert len(nodes) == len(ROWS) + 1 + 3 * n
    assert {x["id"] for x in nodes} >= {"%s-%03d" % (w, i) for w in "abc" for i in range(n)}
    # two experiment stores updated the same idea at the same time: both sets of runs are there, and
    # the arm that never gained did not erase the other arm's positive
    v = store.get("shared-1")["verdict"]
    assert sorted({r["store"] for r in v["runs"]}) == ["arm-a", "arm-b"] and len(v["runs"]) == 2 * n
    assert v["status"] == "reproduced" and v["best_experiment"].startswith("arm-a/")
    assert [f for f in os.listdir(d) if f.endswith(".tmp")] == []          # no temp file left behind


_MEMORY_WRITER = """
import sys
sys.path.insert(0, %r)
from integrations import pastdev_client
mem = pastdev_client.LocalMemory(sys.argv[1])
who, n = sys.argv[2], int(sys.argv[3])
for i in range(n):
    if who == "replace":
        mem.write("verdict v%%d" %% i, author="map", kind="idea", fact_id="idea-x", scope="import-m")
    else:
        mem.write("experiment %%s %%d" %% (who, i), author=who, kind="experiment")
""" % ROOT


def test_local_memory_replace_and_append_can_race():
    d = _fresh("memory-race")
    path = os.path.join(d, "memory.jsonl")
    env = dict(os.environ, PASTDEV_LOG_PATH=os.path.join(d, "log.jsonl"))
    n = 60
    procs = [subprocess.Popen([PY, "-c", _MEMORY_WRITER, path, who, str(n)], stderr=subprocess.PIPE, env=env)
             for who in ("loop-a", "loop-b", "replace")]
    errors = [p.communicate()[1].decode() for p in procs]
    assert all(p.returncode == 0 for p in procs), errors
    rows = [json.loads(l) for l in open(path)]
    assert sum(r["kind"] == "experiment" for r in rows) == 2 * n           # no append lost to a replace
    ideas_rows = [r for r in rows if r["fact_id"] == "idea-x"]
    assert len(ideas_rows) == 1 and ideas_rows[0]["content"] == "verdict v%d" % (n - 1)


def test_local_memory_replaces_a_fact_written_again_with_the_same_id():
    d = _fresh("replace")
    mem = pastdev_client.LocalMemory(os.path.join(d, "memory.jsonl"))
    mem.write("v1", author="a", kind="idea", metadata={"status": "idea"}, fact_id="idea-x", scope="import-m")
    mem.write("other", author="a", kind="idea", fact_id="idea-y", scope="import-m")
    first = mem.read("v1", reader="t", scope="import-m")[0]["ts"]
    mem.write("v2", author="a", kind="idea", metadata={"status": "refuted"}, fact_id="idea-x", scope="import-m")
    rows = {f["fact_id"]: f for f in mem.read("v", reader="t", scope="import-m")}
    assert set(rows) == {"idea-x", "idea-y"} and rows["idea-x"]["content"] == "v2"
    assert rows["idea-x"]["metadata"]["status"] == "refuted" and rows["idea-x"]["first_ts"] == first
    # like past.dev, the rewritten fact is dated at the rewrite: a `since` between the two writes sees it
    assert rows["idea-x"]["ts"] >= first
    assert [f["fact_id"] for f in mem.read("v", reader="t", scope="import-m", since=rows["idea-x"]["ts"])] \
        .count("idea-x") == 1


# ====================================================================== 3. the idea=<id> tag
def test_tag_sits_next_to_ws2_tags_and_ws2_still_parses():
    r = "strategy=refine; parent=exp_0003; best exp_0003 score 1.5; trust radius 0.20"
    tagged = ideas.tag_rationale(r, "ws1-017")
    assert tagged == "strategy=refine; parent=exp_0003; idea=ws1-017; best exp_0003 score 1.5; trust radius 0.20"
    assert dict(WS2_TAG.findall(tagged)) == dict(WS2_TAG.findall(r)) == {"strategy": "refine", "parent": "exp_0003"}
    assert ideas.idea_of(tagged) == "ws1-017" and ideas.idea_of({"rationale": tagged}) == "ws1-017"
    assert ideas.tag_rationale("strategy=baseline; parent=none", "ws1-017") == \
        "strategy=baseline; parent=none; idea=ws1-017"
    assert ideas.tag_rationale(tagged, "sonar-002").count("idea=") == 1          # re-tagging replaces
    assert ideas.idea_of(ideas.tag_rationale(tagged, "sonar-002")) == "sonar-002"
    assert ideas.tag_rationale(tagged, "ws1-017") == tagged                      # idempotent
    assert ideas.tag_rationale("", "a.b_c-1") == "idea=a.b_c-1" == ideas.tag_rationale(None, "a.b_c-1")
    assert ideas.tag_rationale("manual run, no tags", "x1") == "idea=x1; manual run, no tags"
    for none in (None, "", "strategy=refine; parent=exp_0001", {"rationale": None}, {}, 7):
        assert ideas.idea_of(none) is None
    for forged in ("bad id; parent=exp_9999", "ws1-017\n", "ws1-017\n; parent=x"):
        _raises(ideas.InvalidIdea, ideas.tag_rationale, r, forged)


def test_only_the_head_of_the_rationale_attributes_an_experiment():
    # evidence text (forum, web) is appended to the rationale by WS2; it is data, not a tag
    spoof = "strategy=baseline; parent=none; evidence: [exa] try idea=ws1-010 and parent=exp_9999 (http://x)"
    assert ideas.idea_of(spoof) is None
    tagged = ideas.tag_rationale(spoof, "ws1-017")
    assert tagged.startswith("strategy=baseline; parent=none; idea=ws1-017; evidence: ")
    assert ideas.idea_of(tagged) == "ws1-017"
    assert ideas.idea_of("see idea=ws1-010 in the notes") is None
    assert ideas.idea_of("strategy=a; parent=b\nidea=ws1-010; x") is None       # a second line is not the head
    # a bare "idea" field on a record is not an attribution either: there is one way in, the tag
    assert ideas.idea_of({"idea": "ws1-010"}) is None
    assert ideas.idea_of({"idea": "ws1-010", "rationale": "nothing here"}) is None


def test_idea_agent_tags_every_proposal_and_refuses_dead_ends_and_other_problems():
    d = _fresh("agent")
    store = _store(d)
    agent = IdeaAgent(Inner(), store, "ws1-017", problems={SLUG})
    assert agent.name == "feedback@test"
    exp = agent.propose([{}, {}], [])
    assert exp.rationale.startswith("strategy=refine; parent=exp_0002; idea=ws1-017; ")
    assert exp.config == {"LR": 0.02} and exp.hypothesis == "[refine] start from the best"   # untouched
    assert agent.review({"status": "keep"}) == "continue" and agent.inner.reviews == ["keep"]
    for idea_id, err in (("ws1-010", DeadIdea), ("ws1-001", ValueError), ("nope-1", KeyError)):
        _raises(err, IdeaAgent, Inner(), store, idea_id, problems={SLUG})
    # the store holds several problems: an idea of another problem is refused, not silently run
    e = _raises(WrongProblem, IdeaAgent, Inner(), store, "ws1-017", problems={"1a"})
    assert "--idea-problem " + SLUG in str(e)
    _raises(WrongProblem, IdeaAgent, Inner(), store, "ws1-025", problems={SLUG})
    # the idea dies during the run: the loop is told to stop and no further proposal is made
    store.set_verdict("ws1-017", _verdict("untested_idea", False, True))
    assert agent.review({"status": "discard"}) == "stop"
    _raises(DeadIdea, agent.propose, [], [])
    # the store becomes unreadable mid-run: stop, never guess that the idea is alive
    store.set_verdict("ws1-017", None)
    assert agent.review({"status": "discard"}) == "continue"
    store.path.write_text(store.path.read_text() + "{broken\n")
    assert agent.review({"status": "discard"}) == "stop"


def test_mirrored_experiment_fact_carries_the_idea():
    d = _fresh("mirror")
    mem = pastdev_client.LocalMemory(os.path.join(d, "memory.jsonl"))
    store = evidence.PastDevMirrorStore(os.path.join(d, "exp.jsonl"), mem, author="laptop-a")
    fields = getattr(sys.modules["interfaces"], "RECORD_FIELDS", ())
    base = dict({k: None for k in fields}, experiment_id="exp_0001", problem="1a", status="keep", score=1.5,
                config={"LR": 0.02}, evidence=[], rationale="strategy=refine; parent=exp_0000; idea=ws1-017; x")
    store.append(base)
    store.append(dict(base, experiment_id="exp_0002", rationale="strategy=refine; parent=exp_0001"))
    by_id = {f["metadata"]["experiment_id"]: f["metadata"].get("idea") for f in mem.read("experiment", reader="t")}
    assert by_id == {"exp_0001": "ws1-017", "exp_0002": None}


def test_run_loop_wires_the_idea_flags():
    """Executes integrations.run_loop.main with WS2 stand-ins: --idea tags and updates the
    verdict, a dead end and a wrong problem are refused, --ideas uses the chooser."""
    d = _fresh("run-loop")
    added = []
    for name in ("prepare", "loop", "research"):                 # WS2 modules, absent from this branch
        if name not in sys.modules:
            try:
                __import__(name)
            except Exception:
                sys.modules[name] = types.ModuleType(name)
                added.append(name)
    sys.modules["prepare"].__dict__.setdefault("ROOT", Path(d))
    sys.modules["loop"].__dict__.setdefault("_print", lambda r: None)
    sys.modules["loop"].__dict__.setdefault("research_loop", None)
    sys.modules["research"].__dict__.setdefault("FeedbackAgent", None)
    from integrations import run_loop
    saved = {k: getattr(run_loop, k) for k in ("I", "prepare", "research_loop", "FeedbackAgent", "_print")}

    class Problem:
        id, name, direction, best_known = "1a", "autocorrelation", "maximize", RECORD

    class Optimizer:
        def __init__(self, path):
            self.path = path

    def fake_loop(problem, opt, agent, store, evidence_source=None, n=5, budget_s=None, on_result=None):
        for i in range(n):
            history = store.read()
            exp = agent.propose(history, [])
            score = RECORD + 4e-6 * (len(history) + 1)
            result = rec(None, score, history[-1]["score"] if history else None, problem=problem.id,
                         eid="exp_%04d" % len(history), cfg=exp.config, optimizer=exp.optimizer or "default")
            result["rationale"] = exp.rationale
            store.append(result)
            on_result(result)
            if agent.review(result) == "stop":
                break

    seeds = []

    def fake_run_experiment(problem, opt, exp, store, *, budget_s=None, seed=None, agent="?", evidence=None):
        seeds.append(seed)
        history = store.read()
        result = rec(None, RECORD + 4e-6 * (len(history) + 1), None, problem=problem.id, seed=seed,
                     eid="exp_%04d" % len(history), cfg=exp.config)
        result["rationale"] = exp.rationale
        store.append(result)
        return result

    class NoEvidence:
        def gather(self, problem, history):
            return []

    run_loop.I = types.SimpleNamespace(get_problem=lambda pid: Problem, FileOptimizer=Optimizer,
                                       JsonlStore=sys.modules["interfaces"].JsonlStore, NoEvidence=NoEvidence,
                                       run_experiment=fake_run_experiment)
    run_loop.prepare = types.SimpleNamespace(ROOT=Path(d))
    run_loop.research_loop, run_loop._print = fake_loop, lambda r: None
    run_loop.FeedbackAgent = lambda problem, opt, warm_start=None: Inner()
    try:
        store = _store(d)
        common = ["--problem", "1a", "--ideas-file", str(store.path), "--delta", str(DELTA)]
        for bad in (["--idea", "ws1-017"],                                   # idea of another problem
                    ["--idea", "ws1-010", "--idea-problem", SLUG],           # dead end
                    ["--ideas", "--idea-problem", SLUG],                     # no idea has a runner yet
                    ["--idea", "ws1-017", "--ideas"]):
            _raises(SystemExit, run_loop.main, common + bad + ["--store", os.path.join(d, "bad.jsonl")])
        assert not os.path.exists(os.path.join(d, "bad.jsonl"))              # refused before any experiment

        exp_path = os.path.join(d, "exp.jsonl")
        assert run_loop.main(common + ["--idea", "ws1-017", "--idea-problem", SLUG, "-n", "2", "--store", exp_path]) == 0
        rows = [json.loads(l) for l in open(exp_path)]
        assert [ideas.idea_of(r) for r in rows] == ["ws1-017", "ws1-017"]
        assert rows[0]["integrations"]["idea"] == "ws1-017"
        node, label = store.get("ws1-017"), verdicts.store_label(exp_path)
        assert node["status"] == "reproduced"
        assert node["verdict"]["experiments"] == [label + "/exp_0000", label + "/exp_0001"]
        assert abs(node["verdict"]["best_gain_over_delta"] - 0.4) < 1e-6 and node["verdict"]["reference"]["delta"] == DELTA

        store.set_runner("ws1-018", "optimizers/comb.py", {"FINE": 4})
        exp2 = os.path.join(d, "exp2.jsonl")
        assert run_loop.main(common + ["--ideas", "--idea-problem", SLUG, "-n", "1", "--store", exp2]) == 0
        row = json.loads(open(exp2).read())
        assert ideas.idea_of(row) == "ws1-018" and row["optimizer"] == "optimizers/comb.py"
        assert row["config"] == {"FINE": 4, "LR": 0.02} and row["integrations"]["idea"] == "chooser"
        # a second experiment store for the same idea adds its runs; the first store's are kept
        assert run_loop.main(common + ["--idea", "ws1-017", "--idea-problem", SLUG, "-n", "1", "--store", exp2]) == 0
        stores = {r["store"] for r in store.get("ws1-017")["verdict"]["runs"]}
        assert stores == {label, verdicts.store_label(exp2)} and store.get("ws1-017")["status"] == "reproduced"
        # --seed reaches run_experiment and the record; without it WS2's own loop and seed are used
        exp3 = os.path.join(d, "exp3.jsonl")
        assert seeds == [] and run_loop.main(common + ["--ideas", "--idea-problem", SLUG, "-n", "2", "--seed", "7",
                                                       "--store", exp3]) == 0
        rows = [json.loads(l) for l in open(exp3)]
        assert seeds == [7, 7] and [r["seed"] for r in rows] == [7, 7] and rows[0]["integrations"]["seed"] == 7
        assert all(ideas.idea_of(r) for r in rows)
    finally:
        for k, v in saved.items():
            setattr(run_loop, k, v)
        for name in added:
            sys.modules.pop(name, None)
        sys.modules.pop("integrations.run_loop", None)


# ====================================================================== 2. the importer
def test_import_is_idempotent_scoped_and_cannot_be_parsed_as_hints_or_tags():
    d = _fresh("import")
    path, store_path = _write_rows(d, ROWS), os.path.join(d, "ideas.jsonl")
    argv = [path, "--ideas", store_path, "--memory", "local", "--trusted"]
    assert import_knowledge.main(argv) == 0
    first = open(store_path).read()
    assert import_knowledge.main(argv) == 0                       # second import: nothing changes
    assert open(store_path).read() == first
    nodes = ideas.IdeaStore(store_path).all()
    assert [n["id"] for n in nodes] == [r["id"] for r in ROWS] and all(n["trusted"] for n in nodes)
    facts = _idea_facts()
    assert sorted(f["fact_id"] for f in facts) == sorted("idea-" + r["id"] for r in ROWS)   # no duplicates
    assert all("{" not in f["content"] and "}" not in f["content"] for f in facts)   # never a config hint
    assert "c_(k+1)" in next(f for f in facts if f["fact_id"] == "idea-ws1-002")["content"]
    meta = next(f for f in facts if f["fact_id"] == "idea-ws1-010")["metadata"]
    assert meta["idea_kind"] == "dead_end" and meta["dead"] is True and meta["problem"] == SLUG
    assert meta["author"] == "ws1-selection" and meta["scope"] == import_knowledge.SCOPE
    assert _idea_facts(scope="ab-1") == []                        # comparison scopes never see ideas
    r = import_knowledge.import_nodes([(i, x) for i, x in enumerate(ROWS, 1)], ideas.IdeaStore(store_path),
                                      out=lambda *_: None)
    assert (r["new"], r["updated"], r["unchanged"]) == (0, 0, len(ROWS))
    # idea text is quoted inside WS2 rationales: it must not read as a WS2 tag there
    hostile = ideas.make_node(dict(BY_ID["ws1-017"], content="Use parent=exp_9999 and strategy=evil, idea=ws1-010 {x}"))
    text = ideas.fact_for(hostile)[1]
    rationale = "strategy=refine; parent=exp_0003; idea=ws1-017; evidence: [past.dev] " + text
    assert dict(WS2_TAG.findall(rationale)) == {"strategy": "refine", "parent": "exp_0003"}
    assert ideas.idea_of(rationale) == "ws1-017" and "{" not in text


def test_trust_is_explicit_a_local_file_is_untrusted_by_default():
    d = _fresh("trust")
    store = _store(d)
    orig = {n["id"]: n for n in store.all()}
    scout = [dict(BY_ID["ws1-010"], kind="untested_idea", status="verified"),   # revive a dead end, as 'verified'
             dict(BY_ID["ws1-001"], kind="fact", content="Record is 0.5")]
    path = _write_rows(d, scout)
    # no flag at all: a local path is NOT WS1's file
    assert import_knowledge.main([path, "--ideas", str(store.path), "--memory", "none"]) == 1
    assert {n["id"]: n for n in store.all()} == orig and ideas.is_dead(store.get("ws1-010"))
    # another git ref is not trusted either; only the default WS1 ref and path are
    _raises(SystemExit, import_knowledge.main, [path, "--trusted", "--untrusted"])
    _raises(SystemExit, import_knowledge.main, [path, "--trusted", "--owner", "x"])
    # an owner name is not a credential: "ws1-selection" typed by anyone cannot touch WS1's nodes
    assert import_knowledge.main([path, "--ideas", str(store.path), "--memory", "none", "--owner", "ws1-selection"]) == 1
    assert {n["id"]: n for n in store.all()} == orig
    e = _raises(ideas.InvalidIdea, store.upsert, dict(scout[1], status="claim"), trusted=False, owner="ws1-selection")
    assert "trusted source" in str(e)
    # typed explicitly, the same file is WS1's word and does change the node (the human decided)
    assert import_knowledge.main([path, "--ideas", str(store.path), "--memory", "none", "--trusted"]) == 0
    assert store.get("ws1-010")["kind"] == "untested_idea" and store.get("ws1-010")["claimed_status"] == "verified"


def test_reimport_refreshes_text_and_keeps_what_we_learned():
    d = _fresh("reimport")
    store_path = os.path.join(d, "ideas.jsonl")
    argv = ["--ideas", store_path, "--memory", "local", "--trusted"]
    assert import_knowledge.main([_write_rows(d, ROWS)] + argv) == 0
    store = ideas.IdeaStore(store_path)
    store.set_verdict("ws1-017", _verdict("untested_idea", True, exps=("exp_0004",), gain=3e-6))
    store.set_runner("ws1-017", "optimizers/dinkelbach_2m.py", {"N": 2000000})
    # WS1 edits the text; the file also carries a forged verdict and an empty runner: both ignored
    changed = [dict(r, content=r["content"] + " (edited by WS1)", runner={}, verdict={"status": "refuted"})
               if r["id"] == "ws1-017" else r for r in ROWS]
    assert import_knowledge.main([_write_rows(d, changed)] + argv) == 0
    node = store.get("ws1-017")
    assert node["content"].endswith("(edited by WS1)")
    assert node["status"] == "reproduced" and node["claimed_status"] == "idea"
    assert node["verdict"]["experiments"] == ["exp_0004"]
    assert node["runner"] == {"optimizer": "optimizers/dinkelbach_2m.py", "config": {"N": 2000000}}
    facts = [f for f in _idea_facts() if f["fact_id"] == "idea-ws1-017"]
    assert len(facts) == 1                                         # replaced in place, not duplicated
    assert "(edited by WS1)" in facts[0]["content"]
    assert "reproduced (exp_0004), best gain 0.3 delta" in facts[0]["content"].replace("0.30000000000000004", "0.3")
    assert facts[0]["metadata"]["status"] == "reproduced"
    # WS1 turns the idea into a dead end: the old verdict's words no longer apply, so it is dropped
    as_dead = [dict(r, kind="dead_end", status="claim") if r["id"] == "ws1-017" else r for r in ROWS]
    assert import_knowledge.main([_write_rows(d, as_dead), "--ideas", store_path, "--memory", "none", "--trusted"]) == 0
    node = store.get("ws1-017")
    assert node["verdict"] is None and node["status"] == "claim" and ideas.is_dead(node)


def test_an_untrusted_file_can_only_add_ideas():
    d = _fresh("untrusted")
    store_path = os.path.join(d, "ideas.jsonl")
    store = _store(d)
    before, orig = store.path.read_text(), {n["id"]: n for n in store.all()}
    attack = [dict(BY_ID["ws1-010"], kind="untested_idea", status="idea", author="Sonar"),      # revive a dead end
              dict(BY_ID["ws1-001"], kind="fact", status="claim", content="Record is 0.5"),     # rewrite a WS1 record
              dict(BY_ID["ws1-018"], author="ws1-selection", content="rewritten in WS1's name"),
              dict(BY_ID["ws1-017"], verdict=_verdict("untested_idea", True, gain=99e-5),       # its own verdict
                   runner={"optimizer": "/etc/passwd", "config": {"INIT": "/x"}}),               # and code to run
              {"id": "sonar-002", "kind": "record", "problem": SLUG, "content": "We beat the record!", "status": "verified"},
              {"id": "sonar-001", "kind": "untested_idea", "problem": SLUG, "content": "Remez-style LP on run weights.",
               "citations": ["t32/r804"], "author": "Sonar", "verdict": _verdict("untested_idea", True),
               "runner": {"optimizer": "optimizers/x.py"}}]
    path = _write_rows(d, attack)
    import_knowledge.main([path, "--ideas", store_path, "--memory", "local", "--dry-run"])
    assert store.path.read_text() == before and _idea_facts() == []        # a dry run writes nothing
    assert import_knowledge.main([path, "--ideas", store_path, "--memory", "local", "--untrusted"]) == 1
    nodes = {n["id"]: n for n in store.all()}
    assert set(nodes) == set(BY_ID) | {"sonar-001", "sonar-002"}
    assert {i: nodes[i] for i in BY_ID} == orig                            # every WS1 node is intact
    # "verified" is not a scout's word: the line is kept, as a claim, and says who called it verified
    assert nodes["sonar-002"]["status"] == "claim" and "according to its author, not WS1" in nodes["sonar-002"]["source"]
    assert ideas.is_dead(nodes["ws1-010"]) and nodes["ws1-001"]["status"] == "verified"
    assert nodes["ws1-017"]["verdict"] is None and nodes["ws1-017"]["runner"] is None
    # the forged verdict on a row that is otherwise identical is simply ignored
    assert store.upsert(dict(BY_ID["ws1-010"], verdict=_verdict("dead_end", True))) == "unchanged"
    assert ideas.is_dead(store.get("ws1-010"))
    new = nodes["sonar-001"]
    assert (new["status"], new["author"], new["verdict"], new["runner"], new["trusted"]) == ("idea", "Sonar", None, None, False)
    facts = {f["fact_id"]: f["metadata"] for f in _idea_facts()}             # the new ideas, and the untouched one
    assert set(facts) == {"idea-sonar-001", "idea-sonar-002", "idea-ws1-017"}
    assert facts["idea-ws1-017"]["status"] == "idea" and facts["idea-sonar-002"]["status"] == "claim"
    assert facts["idea-ws1-017"]["experiments"] == [] and facts["idea-sonar-001"]["author"] == "Sonar"
    # the same scout file again: still nothing but skips for the taken ids, and no change
    after = store.path.read_text()
    assert import_knowledge.main([path, "--ideas", store_path, "--memory", "none", "--untrusted"]) == 1
    assert store.path.read_text() == after
    assert ideas.next_id(store, "sonar") == "sonar-003"
    _raises(SystemExit, import_knowledge.main, [path, "--scope", "ab-1"])   # idea scopes are tag-only scopes
    # the add command is the scout's door: new ids only
    assert ideas.main(["--ideas", store_path, "--memory", "none", "add", "--problem", SLUG, "--author", "Sonar",
                       "--content", "Adaptive mesh near the critical lags.", "--cite", "t127/r333"]) == 0
    assert store.get("sonar-003")["content"].startswith("Adaptive mesh")
    assert ideas.main(["--ideas", store_path, "--memory", "none", "add", "--problem", SLUG, "--author", "Sonar",
                       "--content", "overwrite", "--id", "ws1-010"]) == 1
    assert ideas.is_dead(store.get("ws1-010"))


def test_an_owner_can_refresh_its_own_nodes_and_nobody_elses():
    d = _fresh("owner")
    store = _store(d)
    orig = {n["id"]: n for n in store.all()}
    v1 = [{"id": "S1", "kind": "untested_idea", "problem": SLUG, "content": "Block-coordinate ramp, v1.",
           "author": "sonar-scout", "written_on": "2026-11-14"},
          {"id": "S2", "kind": "dead_end", "problem": SLUG, "content": "Uniform rescale.", "author": "sonar-scout"}]
    argv = ["--ideas", str(store.path), "--memory", "local", "--owner", "sonar-scout"]
    assert import_knowledge.main([_write_rows(d, v1)] + argv) == 0
    assert import_knowledge.main([_write_rows(d, v1)] + argv) == 0          # the scout rebuilds its file: no change
    assert store.get("S1")["author"] == "sonar-scout" and ideas.is_dead(store.get("S2"))
    store.set_verdict("S1", _verdict("untested_idea", False, True))         # we refuted S1
    v2 = [dict(v1[0], content="Block-coordinate ramp, v2 (new numbers)."),   # its own text: refreshed
          dict(v1[1], author="ws1-selection"),                              # the declared author is ignored
          dict(BY_ID["ws1-010"], kind="untested_idea", status="idea", author="sonar-scout"),   # not its node
          {"id": "S3", "kind": "dead_end", "problem": SLUG, "content": "Measured: no gain.", "status": "verified"}]
    assert import_knowledge.main([_write_rows(d, v2)] + argv) == 1
    assert store.get("S1")["content"].endswith("v2 (new numbers).") and store.get("S1")["status"] == "refuted"
    assert store.get("S2")["author"] == "sonar-scout"
    s3 = store.get("S3")                                                   # its own measurement: a claim, still a dead end
    assert s3["status"] == "claim" and ideas.is_dead(s3) and "according to sonar-scout, not WS1" in s3["source"]
    assert {i: store.get(i) for i in BY_ID} == orig
    # the owner cannot bring a refuted idea back by renaming its kind
    assert import_knowledge.main([_write_rows(d, [dict(v2[0], kind="worked", status="claim")])] + argv) == 1
    assert store.get("S1")["kind"] == "untested_idea" and ideas.is_dead(store.get("S1"))
    facts = {f["fact_id"]: f for f in _idea_facts()}
    assert "v2 (new numbers)" in facts["idea-S1"]["content"] and facts["idea-S1"]["metadata"]["dead"] is True
    # ids made from row numbers shift when the scout inserts a row: S2 would silently become another
    # idea, and every verdict attached to it would change owner. Refused; a note in front of the text is fine.
    shifted = [dict(v1[1], content="Averaging two record solutions scores -281 delta at the midpoint.")]
    assert import_knowledge.main([_write_rows(d, shifted)] + argv) == 1
    assert store.get("S2")["content"] == "Uniform rescale."
    note = [dict(v1[1], content="REFUTED 16:40 PT, see row S9 | Uniform rescale.")]
    assert import_knowledge.main([_write_rows(d, note)] + argv) == 0 and store.get("S2")["content"].startswith("REFUTED")
    assert ideas.same_idea("RANK 2 | S2 slow-manifold extrapolation of the window masses",
                           "REFUTED (see S2 row) | was RANK 2 | S2 slow-manifold extrapolation of the window masses") > 0.7


def test_bad_lines_are_skipped_and_reported_good_lines_still_imported():
    d = _fresh("bad")
    store_path = os.path.join(d, "ideas.jsonl")
    rows = [ROWS[0], "{not json", dict(ROWS[2], kind="rumour"), ROWS[4], dict(ROWS[4], content="again")]
    lines = []
    r = import_knowledge.import_nodes(import_knowledge.load_rows(_write_rows(d, rows)),
                                      ideas.IdeaStore(store_path), out=lines.append)
    assert r["new"] == 2 and [line for line, _ in r["skipped"]] == [2, 3, 5]
    assert "duplicate id" in r["skipped"][2][1] and any("SKIPPED line 2" in l for l in lines)
    assert [n["id"] for n in ideas.IdeaStore(store_path).all()] == ["ws1-001", "ws1-017"]
    assert import_knowledge.main([_write_rows(d, rows), "--ideas", store_path, "--memory", "none", "--trusted"]) == 1


def test_a_second_scope_never_moves_the_facts_of_the_first():
    d = _fresh("scopes")
    store_path, path = os.path.join(d, "ideas.jsonl"), _write_rows(d, ROWS)
    argv = [path, "--ideas", store_path, "--memory", "local", "--trusted"]
    assert import_knowledge.main(argv) == 0
    assert import_knowledge.main(argv + ["--scope", "import-other"]) == 0
    assert len(_idea_facts()) == len(ROWS) == len(_idea_facts(scope="import-other"))
    assert {f["fact_id"] for f in _idea_facts()}.isdisjoint(f["fact_id"] for f in _idea_facts(scope="import-other"))
    # same guarantee for the experiment importer (import_results)
    results = os.path.join(d, "ws3.jsonl")
    with open(results, "w") as f:
        f.write(json.dumps({"experiment_id": "exp_0000", "problem": "1a", "status": "keep", "score": 1.5,
                            "variant": "tuned", "git_commit": "abc", "timestamp": "t0", "config": {"LR": 0.01}}))
    for scope in ([], ["--scope", "import-second"], []):
        import_results.main([results] + scope)
    mem = pastdev_client.LocalMemory()
    assert len(mem.read("imported experiment", reader="x", scope="import-ws3")) == 1
    assert len(mem.read("imported experiment", reader="x", scope="import-second")) == 1


def test_real_ws1_knowledge_file_imports_cleanly():
    """The real file, read from the local git objects (no network). Skipped where the ref is absent."""
    try:
        rows = import_knowledge.load_rows(None, import_knowledge.DEFAULT_REF, import_knowledge.DEFAULT_PATH)
    except (subprocess.CalledProcessError, OSError):
        reason = "%s is not in this checkout" % import_knowledge.DEFAULT_REF
        if "pytest" in sys.modules:
            import pytest
            pytest.skip(reason)
        print("SKIPPED test_real_ws1_knowledge_file_imports_cleanly: " + reason)
        return
    d = _fresh("real")
    store = ideas.IdeaStore(os.path.join(d, "ideas.jsonl"))
    r = import_knowledge.import_nodes(rows, store, pastdev_client.LocalMemory(os.path.join(d, "m.jsonl")),
                                      out=lambda *_: None)
    assert r["skipped"] == [] and r["new"] == len(rows) >= 37
    nodes = store.all()
    assert {n["kind"] for n in nodes} == set(ideas.KINDS)
    main = [n for n in nodes if n["problem"] == SLUG]
    assert sum(ideas.is_dead(n) for n in main) >= 5 and sum(ideas.is_frontier(n) for n in main) >= 5
    for n in nodes:                                    # no node text reads as a WS2 tag or a config hint
        text = ideas.fact_for(n)[1]
        assert not WS2_TAG.search(text) and "{" not in text, n["id"]


# ====================================================================== 4. verdicts
def test_gain_is_measured_in_delta_against_the_best_we_had():
    node = ideas.make_node(BY_ID["ws1-017"])
    b = RECORD
    recs = [rec("ws1-017", b + 1e-10, b),                 # polishing: 0.00001 delta, the idea's baseline
            rec("ws1-017", b + 2e-10, b + 1e-10),         # again 0.00001 delta: negative
            rec("ws1-017", b + 3e-6, b + 2e-10),          # 0.3 delta over the best before: positive
            rec("ws1-017", None, b + 3e-6, status="crash", dur=5.0)]
    rows = verdicts.observe(node, recs, REF)
    assert [x["evidence"] for x in rows] == ["baseline", "negative", "positive", "none"]
    assert abs(rows[2]["gain_over_delta"] - 0.29998) < 1e-4 and rows[2]["bar"] == b + 2e-10
    assert rows[0]["bar_source"].startswith("record read 2026-11-14T22:41")
    assert abs(rows[2]["vs_record_over_delta"] - 0.3) < 1e-6 and not rows[2]["record_candidate"]
    v = verdicts.compute_verdict(node, recs, REF)
    assert v["status"] == "reproduced" and v["gained"] and v["supporting"] == [recs[2]["experiment_id"]]
    assert v["best_experiment"] == recs[2]["experiment_id"] and abs(v["best_gain_over_delta"] - 0.29998) < 1e-4
    assert (v["trials"], v["valid"], v["failed"], v["compute_s"]) == (4, 3, 1, 185.0)
    assert v["reference"] == {"record": RECORD, "record_read_at": "2026-11-14T22:41:00Z", "delta": DELTA,
                              "direction": "maximize", "source": "test", "live": True}
    assert v["decided_by"].startswith("integrations.verdicts@")
    assert v["rule"] == {"min_gain_over_delta": 0.01, "refute_after": 3, "min_compute_s": 3600.0}
    for k in ("experiment_id", "store", "problem", "valid", "score", "seed", "budget_s", "solution_path",
              "solution_sha256", "bar", "bar_source"):
        assert k in v["runs"][0], k                        # what a referee needs to re-check each run
    assert ideas.check_verdict("untested_idea", v) is v
    # the first experiment of an empty store is measured against the live record
    first = verdicts.observe(node, [rec("ws1-017", b + 1.2e-5, None)], REF)[0]
    assert first["evidence"] == "positive" and first["record_candidate"] and first["bar_source"].startswith("record read")
    assert abs(first["gain_over_delta"] - 1.2) < 1e-6
    # the bar includes what OTHER ideas reached in the same store: matching them is not a gain
    others = [rec(None, b, None), rec("ws1-018", b + 5e-6, b), rec("ws1-017", b + 5e-6, b), rec("ws1-017", b + 6e-6, b)]
    rows = verdicts.observe(node, others, REF)
    assert [x["evidence"] for x in rows] == ["baseline", "positive"] and rows[0]["bar_source"] == "best of the other ideas"
    assert abs(rows[1]["gain_over_delta"] - 0.1) < 1e-6
    # with no record at all, beating the other ideas is progress, never a gain
    assert _evidence(node, others, NO_RECORD) == ["baseline", "progress"]


def test_improving_only_on_itself_is_progress_never_a_gain():
    node, dead = ideas.make_node(BY_ID["ws1-017"]), ideas.make_node(BY_ID["ws1-010"])
    for ref in (REF, NO_RECORD, verdicts.Reference(problem=SLUG)):         # record known, unknown, nothing known
        # a from-scratch optimizer climbing 0.50 -> 0.60 -> 0.70, far below the record
        hist = [rec("ws1-017", 0.50, None), rec("ws1-017", 0.60, 0.50), rec("ws1-017", 0.70, 0.60)]
        assert _evidence(node, hist, ref) == ["baseline", "progress", "progress"]
        v = verdicts.compute_verdict(node, hist, ref)
        assert v["status"] == "inconclusive" and not v["gained"]
        # the same two records under a dead end must NOT bring it back
        hist = [rec("ws1-010", 0.50, None), rec("ws1-010", 0.60, 0.50)]
        v = verdicts.compute_verdict(dead, hist, ref)
        assert v["status"] == "inconclusive" and ideas.is_dead(ideas.with_verdict(dead, v))
    d = _fresh("self-improvement")
    store = _store(d)
    vs = verdicts.VerdictStore(sys.modules["interfaces"].JsonlStore(os.path.join(d, "e.jsonl")), store, REF)
    for r in (rec("ws1-010", 0.50, None), rec("ws1-010", 0.60, 0.50)):
        vs.append(r)
    assert ideas.is_dead(store.get("ws1-010")) and store.get("ws1-010")["status"] == "claim"
    assert verdicts.update_all(store, verdicts.labelled(vs.read(), vs.label), lambda n: REF) == {"ws1-010": "inconclusive"}
    _raises(DeadIdea, IdeaAgent, Inner(), store, "ws1-010", problems={SLUG})
    # with delta unknown, float noise is still not a gain (relative floor)
    noise = [rec(None, 1.0, None), rec("ws1-017", 1.0 + 2e-16, 1.0), rec("ws1-017", 1.0 + 4e-16, 1.0)]
    assert _evidence(node, noise, verdicts.Reference(problem=SLUG)) == ["baseline", "negative"]
    # no record read: nothing is a gain, whatever else is in the store (an untagged run, another idea,
    # an extra start_score field, an external log), for a normal idea and for a dead end, max or min
    low = [rec(None, 0.30, None), rec("ws1-018", 0.30, None)]
    for kind_node, tag in ((node, "ws1-017"), (dead, "ws1-010")):
        for ref in (NO_RECORD, verdicts.Reference(problem=SLUG)):
            assert _evidence(kind_node, low + [rec(tag, 0.50, 0.30)], ref) == ["baseline"]
            assert _evidence(kind_node, [rec(tag, 0.50, None, start_score=0.50), rec(tag, 0.60, 0.50, start_score=0.50)],
                             ref) == ["baseline", "progress"]
            ext = verdicts.normalise_record({"experiment_id": "x", "idea": tag, "score": 0.60, "start_score": 0.50,
                                             "status": "DONE"}, 1.0, SLUG)
            assert _evidence(kind_node, [ext], ref) == ["baseline"]
    lower = verdicts.Reference(problem=SLUG, direction="minimize")
    assert _evidence(dead, [rec(None, 9.0, None), rec("ws1-010", 1.6, 9.0, status="keep")], lower) == ["baseline"]
    row = verdicts.observe(node, [rec(None, 1.0, None), rec("ws1-017", 1.5, 1.0)], verdicts.Reference(problem=SLUG))[0]
    assert row["evidence"] == "baseline" and row["gain"] == 0.5 and row["gain_over_delta"] is None
    # a record that was typed, or is a constant, is enough for a normal idea but never revives a dead end
    win = RECORD + 5e-6
    assert _evidence(node, [rec("ws1-017", win, RECORD)], TYPED) == ["positive"]
    assert _evidence(dead, [rec("ws1-010", win, RECORD)], TYPED) == ["baseline"]
    assert _evidence(dead, [rec("ws1-010", win, RECORD)], REF) == ["positive"]         # read live: it is back


def test_refuting_needs_three_fair_negatives_and_a_fair_budget():
    node = ideas.make_node(BY_ID["ws1-018"])
    b = RECORD
    base = rec("ws1-018", b - 1e-4, b, dur=LONG)
    same = rec("ws1-018", b - 2e-4, b, cfg={"K": 1}, dur=LONG)
    repeat = dict(same, experiment_id="exp_rep")
    crash = rec("ws1-018", None, b, status="crash", dur=LONG)
    invalid = rec("ws1-018", None, b, status="invalid", dur=LONG)
    short = rec("ws1-018", b - 5e-4, b, budget_s=10.0, dur=LONG)          # a smaller budget than the baseline's
    recs = [base, same, repeat, crash, invalid, short, rec("ws1-018", b - 3e-4, b, dur=LONG)]
    v = verdicts.compute_verdict(node, recs, REF)
    assert [x["evidence"] for x in v["runs"]] == ["baseline", "negative", "repeat", "none", "none", "short", "negative"]
    assert v["status"] == "inconclusive" and not v["exhausted"]            # 2 fair negatives are not enough
    v = verdicts.compute_verdict(node, recs + [rec("ws1-018", b - 4e-4, b, dur=LONG)], REF)
    assert v["status"] == "refuted" and v["exhausted"] and len(v["contradicting"]) == 3 and v["failed"] == 2
    assert v["negative_compute_s"] == 3 * LONG and v["fair_budget_s"] == 3600.0
    assert ideas.is_dead(ideas.with_verdict(node, v))
    # four runs of one second each refute nothing, however many there are
    quick = [rec("ws1-018", b - 1e-4 * (i + 1), b, dur=1.0) for i in range(8)]
    v = verdicts.compute_verdict(node, quick, REF)
    assert v["status"] == "inconclusive" and len(v["contradicting"]) == 7 and v["negative_compute_s"] == 7.0
    assert verdicts.compute_verdict(node, quick, REF, min_compute_s=5.0)["status"] == "refuted"
    # a run counts for at most its own budget, whatever duration its record claims
    liar = [rec("ws1-018", b - 1e-4 * (i + 1), b, dur=99999.0, budget_s=60.0) for i in range(4)]
    v = verdicts.compute_verdict(node, liar, REF)
    assert v["status"] == "inconclusive" and v["negative_compute_s"] == 180.0
    # the runner declares the idea's own fair budget, and it wins over the default
    patient = dict(node, runner={"optimizer": "optimizers/x.py", "config": {}, "min_compute_s": 10000.0})
    v = verdicts.compute_verdict(patient, recs + [rec("ws1-018", b - 4e-4, b, dur=LONG)], REF)
    assert v["status"] == "inconclusive" and v["fair_budget_s"] == 10000.0
    # an idea that keeps improving on itself is not refuted, even below the global best
    climbing = [rec("ws1-018", b - 9e-4 + i * 1e-6, b, dur=LONG) for i in range(6)]
    v = verdicts.compute_verdict(node, climbing, REF)
    assert v["status"] == "inconclusive" and [x["evidence"] for x in v["runs"]] == ["baseline"] + ["progress"] * 5
    # three runs each gaining 0.005 delta in 10 minutes (the pace of polishing the #1): undecided, not refuted
    polish = [rec("ws1-018", b + 5e-8 * (i + 1), b + 5e-8 * i, dur=600.0) for i in range(4)]
    assert verdicts.compute_verdict(node, polish, REF)["status"] == "inconclusive"
    assert verdicts.compute_verdict(node, [], REF) is None
    assert verdicts.compute_verdict(ideas.make_node(BY_ID["ws1-001"], allow_verified=True), recs, REF) is None


def test_a_human_can_reopen_a_refuted_idea():
    d = _fresh("retest")
    store = _store(d)
    b = RECORD
    hist = [rec("ws1-018", b - 1e-4 * (i + 1), b, dur=LONG, eid="e%d" % i) for i in range(4)]
    verdicts.update_verdict(store, "ws1-018", hist, REF)
    assert store.get("ws1-018")["status"] == "refuted" and ideas.is_dead(store.get("ws1-018"))
    _raises(ideas.InvalidIdea, verdicts.retest, store, "ws1-018", "", "Murilo")
    _raises(ideas.InvalidIdea, verdicts.retest, store, "ws1-017", "no runs yet", "Murilo")
    node = verdicts.retest(store, "ws1-018", "only ran the default comb", "Murilo")
    assert node["status"] == "idea" and ideas.is_frontier(node) and node["verdict"]["status"] == "inconclusive"
    assert node["retest"]["forgiven"] == ["e1", "e2", "e3"] and node["retest"]["log"][0]["by"] == "Murilo"
    # no forgiving in advance: an idea that is not refuted cannot be reopened again
    _raises(ideas.InvalidIdea, verdicts.retest, store, "ws1-018", "just in case", "agent-x")
    assert [x["evidence"] for x in node["verdict"]["runs"]] == ["baseline", "forgiven", "forgiven", "forgiven"]
    # recomputing from the same experiments does not refute it again: the forgiven runs stay forgiven
    assert verdicts.update_verdict(store, "ws1-018", hist, REF)["status"] == "idea"
    # two new negatives are not enough; a third one refutes it again
    more = hist + [rec("ws1-018", b - 1e-3 * (i + 1), b, dur=LONG, eid="n%d" % i) for i in range(2)]
    assert verdicts.update_verdict(store, "ws1-018", more, REF)["status"] == "idea"
    more.append(rec("ws1-018", b - 5e-3, b, dur=LONG, eid="n2"))
    node = verdicts.update_verdict(store, "ws1-018", more, REF)
    assert node["status"] == "refuted" and node["verdict"]["contradicting"] == ["n0", "n1", "n2"]
    assert verdicts.main(["--ideas", str(store.path), "retest", "ws1-018", "--reason", "new runner", "--by", "Marie"]) == 0
    node = store.get("ws1-018")
    assert ideas.is_frontier(node) and len(node["retest"]["forgiven"]) == 6
    assert [(e["action"], e["by"], e["reason"]) for e in node["retest"]["log"]] == [
        ("retest", "Murilo", "only ran the default comb"), ("retest", "Marie", "new runner")]   # nothing overwritten
    # a dead end reported by its source is not reopened by a switch: only a run that gains does that
    dead = [rec("ws1-010", b - 1e-4 * (i + 1), b, dur=LONG) for i in range(4)]
    assert verdicts.update_verdict(store, "ws1-010", dead, REF)["status"] == "reproduced"
    _raises(ideas.InvalidIdea, verdicts.retest, store, "ws1-010", "try again", "Murilo")
    assert verdicts.main(["--ideas", str(store.path), "retest", "ws1-010", "--reason", "x", "--by", "y"]) == 1
    assert ideas.is_dead(store.get("ws1-010"))


def test_dead_end_polarity_and_claimed_value_criterion():
    b = RECORD
    dead = ideas.make_node(BY_ID["ws1-010"])
    losing = [rec("ws1-010", b - 1e-3 * (i + 1), b, dur=LONG) for i in range(4)]
    v = verdicts.compute_verdict(dead, losing, REF)
    assert v["status"] == "reproduced" and len(v["supporting"]) == 3      # no gain: the dead end is confirmed
    assert ideas.is_dead(ideas.with_verdict(dead, v))
    # it comes back only by beating everything we had, the record included
    v = verdicts.compute_verdict(dead, losing + [rec("ws1-010", b + 5e-6, b)], REF)
    assert v["status"] == "refuted" and v["gained"] and len(v["contradicting"]) == 1
    assert ideas.is_frontier(ideas.with_verdict(dead, v))
    # a `worked` claim with a claimed value is reproduced when we REACH that value, record or not
    claim = ideas.make_node(dict(BY_ID["ws1-007"], claimed_value=0.96199))
    runs = [rec("ws1-007", 0.95, b), rec("ws1-007", 0.9612, b)]
    v = verdicts.compute_verdict(claim, runs, REF)
    assert v["status"] == "inconclusive" and v["criterion"] == "claimed_value" and v["claimed_value"] == 0.96199
    v = verdicts.compute_verdict(claim, runs + [rec("ws1-007", 0.96201, b)], REF)
    assert v["status"] == "reproduced" and v["best_score"] == 0.96201 and v["best_gain"] < 0
    assert ideas.check_verdict("worked", v) is v
    # the same claim without a claimed value falls back to the gain criterion
    v = verdicts.compute_verdict(ideas.make_node(BY_ID["ws1-007"]), runs + [rec("ws1-007", 0.96201, b)], REF)
    assert v["criterion"] == "gain" and v["status"] == "inconclusive"
    # minimisation: a lower score is the gain
    lower = verdicts.Reference(problem=SLUG, direction="minimize", delta=1e-3, record=1.7, record_read_at="t")
    idea = ideas.make_node(BY_ID["ws1-017"])
    hist = [rec("ws1-017", 1.6, None, status="keep"), rec("ws1-017", 1.55, 1.6, status="keep"),
            rec("ws1-017", 1.75, 1.55, status="discard")]
    v = verdicts.compute_verdict(idea, hist, lower)
    assert [x["evidence"] for x in v["runs"]] == ["positive", "positive", "negative"]
    assert v["status"] == "reproduced" and abs(v["best_gain_over_delta"] - 100) < 1e-6 and v["best_score"] == 1.55


def test_duplicate_experiment_ids_are_never_evidence():
    node = ideas.make_node(BY_ID["ws1-017"])
    b = RECORD
    hist = [rec("ws1-017", b + 3e-6, b, eid="exp_0000"), rec("ws1-017", b + 9e-6, b, eid="exp_0000"),
            rec("ws1-017", b - 1e-6, b, eid="exp_0001")]
    rows = verdicts.observe(node, hist, REF)
    assert [x["evidence"] for x in rows] == ["ambiguous", "ambiguous", "baseline"]
    v = verdicts.summarise(node, rows, REF.as_dict())
    assert not v["gained"] and v["status"] == "inconclusive"
    assert v["experiments"] == ["exp_0000#1", "exp_0000#2", "exp_0001"] and ideas.check_verdict("untested_idea", v) is v
    d = _fresh("duplicates")
    store = _store(d)
    vs = verdicts.VerdictStore(sys.modules["interfaces"].JsonlStore(os.path.join(d, "e.jsonl")), store, REF)
    vs.append(hist[0])
    assert store.get("ws1-017")["status"] == "reproduced"                  # one run, a real gain
    vs.append(hist[1])                                                     # the same id again: the gain is withdrawn
    vs.append(hist[2])
    node = store.get("ws1-017")
    assert node["status"] == "idea" and len(vs.read()) == 3
    assert [x["evidence"] for x in node["verdict"]["runs"]] == ["ambiguous", "ambiguous", "baseline"]
    # the same id in two different stores is two different runs
    two = verdicts.labelled(hist[:1], "arm-a") + verdicts.labelled(hist[1:2], "arm-b")
    v = verdicts.compute_verdict(node, two, REF)
    assert v["experiments"] == ["arm-a/exp_0000", "arm-b/exp_0000"] and v["status"] == "reproduced"
    assert ideas.check_verdict("untested_idea", v) is v


def test_verdict_is_updated_after_each_experiment_and_mirrored():
    d = _fresh("verdict-store")
    store = _store(d)
    mem = pastdev_client.LocalMemory(os.path.join(d, "memory.jsonl"))
    for n in store.all():
        ideas.mirror(mem, n)
    sol = os.path.join(d, "solutions", "exp_a.json")
    os.makedirs(os.path.dirname(sol))
    open(sol, "w").write('{"values": [1, 2, 3]}')
    inner = sys.modules["interfaces"].JsonlStore(os.path.join(d, "exp.jsonl"))
    vs = verdicts.VerdictStore(inner, store, REF, memory=mem, hash_root=Path(d))
    L = verdicts.store_label(inner.path)
    assert vs.label == L and L.startswith("exp@")
    b = RECORD
    vs.append(rec(None, b, None, eid="exp_base"))                          # untagged: no verdict anywhere
    assert all(n["verdict"] is None for n in store.all())
    vs.append(rec("ws1-017", b - 1e-6, b, eid="exp_a", solution_path="solutions/exp_a.json"))
    node = store.get("ws1-017")
    assert node["status"] == "idea" and node["verdict"]["status"] == "inconclusive"
    assert node["verdict"]["runs"][0]["solution_sha256"] == hashlib.sha256(open(sol, "rb").read()).hexdigest()
    decided = node["verdict"]["decided_on"]
    vs.append(rec("ws1-017", b + 2.5e-6, b, eid="exp_b"))
    node = store.get("ws1-017")
    assert node["status"] == "reproduced" and node["verdict"]["experiments"] == [L + "/exp_a", L + "/exp_b"]
    assert abs(node["verdict"]["best_gain_over_delta"] - 0.25) < 1e-6 and node["verdict"]["decided_on"] >= decided
    fact = next(f for f in _idea_facts(path=mem.path) if f["fact_id"] == "idea-ws1-017")
    assert fact["metadata"]["status"] == "reproduced" and "reproduced (%s/exp_a, %s/exp_b)" % (L, L) in fact["content"]
    assert len(vs.read()) == 3
    # recomputing from the same records changes nothing (not even the decision time)
    history = verdicts.labelled(vs.read(), L)
    again = verdicts.update_verdict(store, "ws1-017", history, REF, hash_root=Path(d))
    assert again["verdict"] == node["verdict"]
    assert verdicts.update_all(store, history, lambda n: REF, hash_root=Path(d)) == {"ws1-017": "reproduced"}
    # an experiment run on another problem is never evidence for the idea, unless an alias says it is the same
    vs.append(rec("ws1-018", b + 9e-6, b, eid="exp_c", problem="1a"))
    assert store.get("ws1-018")["verdict"] is None
    aliased = verdicts.VerdictStore(inner, store, REF, aliases={"1a": SLUG})
    aliased.append(rec("ws1-018", b + 9e-6, b, eid="exp_d", problem="1a"))
    node = store.get("ws1-018")
    assert node["status"] == "reproduced" and node["verdict"]["experiments"] == [L + "/exp_c", L + "/exp_d"]
    assert abs(node["verdict"]["best_gain_over_delta"] - 0.65) < 1e-6      # over ws1-017's best, not over the record
    # tags that point nowhere, or at a record, never break the loop and never write a verdict
    vs.append(rec("nope-9", b, b, eid="exp_e"))
    vs.append(rec("ws1-001", b + 1e-3, b, eid="exp_f"))
    assert store.get("ws1-001")["status"] == "verified" and store.get("ws1-001")["verdict"] is None
    store.path.write_text(store.path.read_text() + "{broken\n")
    vs.append(rec("ws1-017", b, b, eid="exp_g"))                           # the record is still stored
    assert vs.read()[-1]["experiment_id"] == "exp_g"


def test_two_experiment_stores_never_erase_each_others_runs():
    d = _fresh("two-stores")
    store = _store(d)
    JsonlStore = sys.modules["interfaces"].JsonlStore
    arm_a = verdicts.VerdictStore(JsonlStore(os.path.join(d, "arm_a.jsonl")), store, REF)
    arm_b = verdicts.VerdictStore(JsonlStore(os.path.join(d, "arm_b.jsonl")), store, REF)
    b = RECORD
    arm_a.append(rec("ws1-017", b - 1e-6, b, eid="exp_0000"))
    arm_a.append(rec("ws1-017", b + 4e-6, b, eid="exp_0001"))              # arm A: a real gain
    assert store.get("ws1-017")["status"] == "reproduced"
    for i in range(4):                                                     # arm B: four long runs, no gain
        arm_b.append(rec("ws1-017", b - 1e-4 * (i + 1), b, eid="exp_%04d" % i, dur=LONG))
    node = store.get("ws1-017")
    assert node["status"] == "reproduced" and not ideas.is_dead(node)      # A's positive is still there
    assert node["verdict"]["experiments"] == [arm_a.label + "/exp_0000", arm_a.label + "/exp_0001"] + [
        "%s/exp_%04d" % (arm_b.label, i) for i in range(4)]
    assert node["verdict"]["best_experiment"] == arm_a.label + "/exp_0001" and len(node["verdict"]["contradicting"]) == 3
    # in the other order too: B first (refuted), then A's gain reopens it
    for i in range(4):
        arm_b.append(rec("ws1-018", b - 1e-4 * (i + 1), b, eid="exp_1%03d" % i, dur=LONG))
    assert store.get("ws1-018")["status"] == "refuted"
    arm_a.append(rec("ws1-018", b + 8e-6, b, eid="exp_0002"))
    assert store.get("ws1-018")["status"] == "reproduced" and len(store.get("ws1-018")["verdict"]["runs"]) == 5
    # the same file seen under a second label (copied, moved) adds nothing: each run is counted once
    before = store.get("ws1-017")["verdict"]
    node = verdicts.update_verdict(store, "ws1-017", verdicts.labelled(arm_b.read(), "copy-of-arm-b"), REF)
    v = node["verdict"]
    assert [x["evidence"] for x in v["runs"] if x["store"] == "copy-of-arm-b"] == ["copy"] * 4
    assert (v["trials"], v["negative_compute_s"], len(v["contradicting"])) == (
        before["trials"], before["negative_compute_s"], len(before["contradicting"]))
    assert verdicts.forget_store(store, "copy-of-arm-b", "a copy", "Murilo", idea_id="ws1-017") == {"ws1-017": 4}
    # a store that no longer holds an idea's runs takes only ITS runs out of the verdict
    node = verdicts.update_verdict(store, "ws1-017", verdicts.labelled([], arm_b.label) or
                                   [dict(rec(None, b, None), _store=arm_b.label)], REF)
    assert {r["store"] for r in node["verdict"]["runs"]} == {arm_a.label}


def test_a_positive_does_not_outlive_the_record_it_was_measured_against():
    d = _fresh("stale")
    store = _store(d)
    old_record = 0.963638667992813
    then = verdicts.Reference(problem=SLUG, delta=DELTA, record=old_record, record_read_at="15:35Z", live=True)
    win = [rec("ws1-010", old_record + 1.5e-5, old_record, eid="e0")]      # +1.5 delta over the OLD record
    node = verdicts.update_verdict(store, "ws1-010", verdicts.labelled(win, "old"), then)
    assert node["status"] == "refuted" and not ideas.is_dead(node)         # the dead end was back, rightly, then
    # the record moves by 2.29 delta; a new store reports under the live record; "old" is not given again
    new = [rec("ws1-010", RECORD - 1e-6, RECORD, eid="n0"), rec("ws1-010", RECORD - 2e-6, RECORD, eid="n1")]
    node = verdicts.update_verdict(store, "ws1-010", verdicts.labelled(new, "new"), REF)
    v = node["verdict"]
    assert [(x["key"], x["evidence"]) for x in v["runs"]] == [("old/e0", "outdated"), ("new/n0", "baseline"),
                                                              ("new/n1", "negative")]
    assert node["status"] == "claim" and ideas.is_dead(node) and not v["gained"]
    assert v["runs"][0]["record"] == RECORD and v["runs"][0]["gain"] < 0   # judged again, against today's record
    # a store that is gone can be taken out for good, and the node says who did it and why
    _raises(ideas.InvalidIdea, verdicts.forget_store, store, "old", "", "Murilo")
    assert verdicts.forget_store(store, "old", "store deleted", "Murilo") == {"ws1-010": 1}
    node = store.get("ws1-010")
    assert [x["key"] for x in node["verdict"]["runs"]] == ["new/n0", "new/n1"]
    assert node["retest"]["log"] == [dict(node["retest"]["log"][0], action="forget", store="old", runs=["old/e0"],
                                          reason="store deleted", by="Murilo")]
    assert verdicts.main(["--ideas", str(store.path), "forget", "--store", "new", "--reason", "r", "--by", "b"]) == 0
    assert verdicts.forget_store(store, "new", "again", "x") == {}


def test_logs_written_outside_the_loop_count_through_normalise_record():
    b = RECORD
    brasa = {"experiment_id": "b0_live_w2", "idea": "B0", "hypothesis": "plain polish from the live #1",
             "config": {"script": "attack/probe_lbfgs.py"}, "seed": None, "budget_s": 600, "score": b + 5.2e-8,
             "start_score": b, "status": "DEAD", "rationale": "idea=ws1-017; parent=exp_1 is what the text says"}
    r = verdicts.normalise_record(brasa, 1.0, SLUG, {"B0": "ws1-010"})
    assert ideas.idea_of(r) == "ws1-010"                                    # the field, written as the tag
    assert ideas.idea_of(dict(brasa, rationale="plain text")) is None       # the bare field attributes nothing
    assert r["rationale"].startswith("strategy=external; parent=none; idea=ws1-010; ")
    assert dict(WS2_TAG.findall(r["rationale"])) == {"strategy": "external", "parent": "none"}   # the text adds no tag
    assert (r["status"], r["best_before"], r["duration_s"], r["problem"]) == ("keep", b, 600, SLUG)
    assert r["reported_status"] == "DEAD" and "reported_status" not in brasa   # the input is not modified
    assert ideas.idea_of(verdicts.normalise_record(dict(brasa, idea="bad id"), 1.0, SLUG)) is None
    killed = verdicts.normalise_record(dict(brasa, experiment_id="killed", score=b - 1e-9, killed_at_s=300), 1.0, SLUG)
    assert (killed["status"], killed["duration_s"], ideas.idea_of(killed)) == ("discard", 300, "B0")
    running = verdicts.normalise_record({"experiment_id": "r1", "idea": "ws1-010", "status": "RUNNING",
                                         "budget_s": 1200}, 1.0, SLUG)
    assert running["status"] == "none" and running["problem"] == SLUG       # no problem named: the page's problem
    assert verdicts.normalise_record(dict(brasa, score=1.4, start_score=1.5), -1.0)["status"] == "keep"   # minimise
    ws2 = rec("ws1-017", b, b)
    assert verdicts.normalise_record(ws2, 1.0, "other", {"ws1-017": "x"}) is ws2     # WS2 records pass through
    node = ideas.make_node(BY_ID["ws1-010"])
    v = verdicts.compute_verdict(node, [r, killed, running], REF)
    assert v["experiments"] == ["b0_live_w2", "r1"] and v["status"] == "inconclusive"
    assert abs(v["best_gain_over_delta"] - 0.0052) < 1e-6 and v["compute_s"] == 1800.0 and v["failed"] == 1
    # a start score is part of the bar; with the record read, beating both is a gain
    won = [verdicts.normalise_record(dict(brasa, idea="ws1-017", score=b + 2e-6), 1.0, SLUG)]
    assert _evidence(ideas.make_node(BY_ID["ws1-017"]), won, REF) == ["positive"]
    assert _evidence(ideas.make_node(BY_ID["ws1-017"]), won, NO_RECORD) == ["baseline"]
    # four polish runs from two different starting points: a higher score that only reflects a better
    # start is not progress of the idea
    runs = [verdicts.normalise_record(dict(brasa, experiment_id=e, score=s, start_score=s0, config={"run": e}),
                                      1.0, SLUG, {"B0": "ws1-010"})
            for e, s0, s in (("from_cert", b - 2.29e-5, b - 2.29e-5 + 5e-8), ("from_live", b, b + 5.2e-8),
                             ("free_cells", b, b), ("alm", b, b + 3e-8))]
    v = verdicts.compute_verdict(node, runs, REF)
    assert [x["evidence"] for x in v["runs"]] == ["baseline", "negative", "negative", "negative"]
    assert v["status"] == "inconclusive" and v["negative_compute_s"] == 1800.0     # 30 minutes: not a fair budget yet
    v = verdicts.compute_verdict(node, runs, REF, min_compute_s=1500.0)
    assert v["status"] == "reproduced" and ideas.is_dead(ideas.with_verdict(node, v))   # dead for us too
    d = _fresh("outside")
    store = _store(d)
    path = os.path.join(d, "attack.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps(brasa) + "\n")
    out = os.path.join(d, "map.html")
    assert research_map.main(["--ideas", str(store.path), "--experiments", path, "--out", out]) == 0
    assert "B0 (not on the map)" in open(out).read()                       # an unknown label is shown, not dropped
    assert research_map.main(["--ideas", str(store.path), "--experiments", path, "--out", out,
                              "--label", "B0=ws1-010"]) == 0
    page = open(out).read()
    assert "b0_live_w2" in page and "+0.0052 δ" in page and "not on the map" not in page


def test_the_record_is_never_taken_from_a_node():
    d = _fresh("live")
    nodes = _store(d).all()
    ref = verdicts.reference_from_nodes(nodes, SLUG)
    assert ref.delta == 1e-5 and ref.record is None and "record not read" in ref.source   # the node's #1 is stale
    assert verdicts.reference_from_nodes(nodes, "circles-rectangle").delta is None
    # delta is read only from a record node a trusted import wrote
    scout = [ideas.make_node(dict(BY_ID["ws1-001"], status="claim", content="Live #1: 0.5; delta 5e-1"))]
    assert verdicts.reference_from_nodes(scout, SLUG).delta is None
    # --live reads WS1's catalog in its own checkout; a catalog that is offline (snapshot) gives NO record
    os.makedirs(os.path.join(d, "problems"))
    catalog = ("class P:\n    DELTA = 1e-5\n    DIRECTION = 'maximize'\n"
               "    def best_known_score(self, max_age_s=300.0):\n"
               "        return {'value': %r, 'agent': 'MLAI_YONSEI', 'live': %s, 'read_at': '2026-11-15T00:30:00Z'}\n"
               "def load(slug):\n    return P()\n")
    open(os.path.join(d, "problems", "__init__.py"), "w").close()
    open(os.path.join(d, "problems", "catalog.py"), "w").write(catalog % (RECORD, True))
    live = research_map.reference_live(SLUG, d)
    assert (live.record, live.delta, live.record_read_at) == (RECORD, 1e-5, "2026-11-15T00:30:00Z")
    assert "live #1 by MLAI_YONSEI" in live.source
    open(os.path.join(d, "problems", "catalog.py"), "w").write(catalog % (0.963638667992813, False))
    stale = research_map.reference_live(SLUG, d)
    assert stale.record is None and stale.delta == 1e-5 and "record not read" in stale.source
    assert os.listdir(os.path.join(d, "problems")) == ["__init__.py", "catalog.py"] or \
        sorted(os.listdir(os.path.join(d, "problems"))) == ["__init__.py", "catalog.py"]   # no bytecode left there
    missing = research_map.reference_live(SLUG, os.path.join(d, "nowhere"))
    assert missing.record is None and missing.delta is None and "live read failed" in missing.source

    class Module:
        DELTA = 1e-5

    class Problem:
        id, direction, best_known, _m = "x", "maximize", RECORD, Module
        best_known_source = "EinsteinArena #1 MLAI_YONSEI (live, 2026-11-15T00:30:00Z); a new record needs gain >= delta"
    live = verdicts.reference_from_problem(Problem)
    assert (live.record, live.delta, live.live, live.record_read_at) == (RECORD, 1e-5, True, "2026-11-15T00:30:00Z")
    # a snapshot, or a constant in a WS2 module, is a record nobody read: said so, and never "read <now>"
    for source in ("EinsteinArena #1 MLAI_YONSEI (snapshot, 2026-11-14T15:35Z)", "Matolcsi-Vinuesa 2010", ""):
        Problem.best_known_source = source
        const = verdicts.reference_from_problem(Problem)
        assert const.record == RECORD and const.live is False and const.record_read_at is None
        assert const.source.startswith("constant in the problem module, not a live read")


# ====================================================================== 5. the idea chooser
def _runnable(d, ids=("ws1-007", "ws1-017", "ws1-018")):
    store = _store(d)
    for i in ids:
        store.set_runner(i, "optimizers/%s.py" % i.replace("-", "_"), {"PRESET": i})
    return store


def test_chooser_never_proposes_a_dead_end_and_needs_a_runner():
    d = _fresh("chooser")
    store = _store(d)
    chooser = idea_chooser.IdeaChooser(store, lambda n: REF, problems={SLUG})
    assert [n["id"] for n in chooser.frontier()] == ["ws1-007", "ws1-017", "ws1-018"]
    assert chooser.candidates() == [] and chooser.choose([]) is None        # nothing has a runner yet
    store = _runnable(d)
    # even a dead end that somebody gave a runner is never a candidate
    dead_with_runner = dict(store.get("ws1-010"), runner={"optimizer": "optimizers/dense.py", "config": {}})
    store._write([dead_with_runner if n["id"] == "ws1-010" else n for n in store.all()])
    ranked = chooser.ranking([])
    assert [x["id"] for x in ranked] == ["ws1-007", "ws1-017", "ws1-018"]   # untried: worked claims first, map order
    assert all(x["score"] == math.inf and x["pulls"] == 0 for x in ranked)
    assert chooser.choose([], exclude={"ws1-007"})["id"] == "ws1-017"
    assert idea_chooser.IdeaChooser(store, lambda n: REF, problems={"circles-rectangle"}).candidates() == []
    assert idea_chooser.IdeaChooser(store, lambda n: REF, problems=set()).candidates() == []   # no problem, not all
    assert len(idea_chooser.IdeaChooser(store, lambda n: REF, problems=None, need_runner=False).candidates()) == 4

    # simulate a long run: ws1-007 and ws1-018 never gain, ws1-017 gains on every 2nd run of its own;
    # verdicts are updated after each experiment, as the loop does
    history, chosen, b = [], [], RECORD
    for step in range(60):
        pick = chooser.choose(history)
        if pick is None:
            break
        pulls = chosen.count(pick["id"])
        chosen.append(pick["id"])
        best = max([r["score"] for r in history if r["status"] == "keep"] or [b])
        gain = {"ws1-017": 2e-6 if pulls % 2 == 0 else -1e-6, "ws1-007": -1e-5 * (step + 1)}.get(
            pick["id"], -1e-4 * (step + 1))
        history.append(rec(pick["id"], best + gain, best, eid="sim_%03d" % step, dur=LONG))
        verdicts.update_verdict(store, pick["id"], history, REF)
        dead_now = {n["id"] for n in store.all() if ideas.is_dead(n)}
        nxt = chooser.choose(history)
        assert nxt is None or nxt["id"] not in dead_now                    # gone from the very next choice
    assert "ws1-010" not in chosen
    for idea_id in ("ws1-018", "ws1-007"):                                 # refuted by us, then never again
        assert store.get(idea_id)["status"] == "refuted"
        last = len(chosen) - 1 - chosen[::-1].index(idea_id)
        assert idea_id not in chosen[last + 1:] and chosen.count(idea_id) == 4   # baseline + 3 negatives
    assert store.get("ws1-017")["status"] == "reproduced" and chosen.count("ws1-017") == 60 - 8
    assert chooser.choose(history)["id"] == "ws1-017"


def test_chooser_is_a_bandit_on_gain_over_delta():
    d = _fresh("bandit")
    store = _runnable(d, ids=("ws1-017", "ws1-018"))
    chooser = idea_chooser.IdeaChooser(store, lambda n: REF, problems={SLUG}, c=0.8)
    b = RECORD
    history = [rec("ws1-017", b + 1e-6, b), rec("ws1-018", b + 6e-6, b)]    # both tried once: 0.1 vs 0.5 delta
    ranked = chooser.ranking(history)
    assert [x["id"] for x in ranked] == ["ws1-018", "ws1-017"]
    assert abs(ranked[0]["mean"] - math.sqrt(0.5)) < 1e-6 and abs(ranked[1]["mean"] - math.sqrt(0.1)) < 1e-6
    assert "UCB" in ranked[0]["why"]
    # exploration: after many pulls of the better arm, the other one gets another try
    many = history + [rec("ws1-018", b, b + 6e-6) for _ in range(40)]
    assert chooser.choose(many)["id"] == "ws1-017"
    assert idea_chooser.reward({"evidence": "negative", "gain_over_delta": 5}) == 0.0
    assert idea_chooser.reward({"evidence": "positive", "gain_over_delta": 4.0}) == 1.0      # capped at a record
    assert idea_chooser.reward({"evidence": "positive", "gain_over_delta": None}) == 1.0     # delta unknown
    assert idea_chooser.reward({"evidence": "positive", "gain_over_delta": -2.0}) == 0.1     # a claim reproduced
    # experiments of another problem do not count as pulls
    assert chooser.ranking([rec("ws1-017", b, b, problem="1a")])[0]["pulls"] == 0
    aliased = idea_chooser.IdeaChooser(store, lambda n: REF, problems={SLUG}, aliases={"1a": SLUG})
    assert {x["id"]: x["pulls"] for x in aliased.ranking([rec("ws1-017", b, b, problem="1a")])}["ws1-017"] == 1


def test_idea_search_agent_runs_each_idea_with_its_own_tuner():
    d = _fresh("search-agent")
    store = _runnable(d, ids=("ws1-017", "ws1-018"))
    chooser = idea_chooser.IdeaChooser(store, lambda n: REF, problems={SLUG})
    tuners = {}

    def make(node):
        tuners[node["id"]] = Inner(stop_after=2 if node["id"] == "ws1-018" else None)
        return tuners[node["id"]]

    agent = idea_chooser.IdeaSearchAgent(store, chooser, make, name="ideas@test", record_problems={SLUG})
    b = RECORD
    history = [rec(None, b, None, eid="exp_base")]
    exp = agent.propose(history, [])
    assert ideas.idea_of(exp.rationale) == "ws1-017" and exp.optimizer == "optimizers/ws1_017.py"
    assert exp.config == {"PRESET": "ws1-017", "LR": 0.02}                 # the idea's preset under the tuner's knobs
    assert dict(WS2_TAG.findall(exp.rationale)) == {"strategy": "refine", "parent": "exp_0000"}
    assert tuners["ws1-017"].seen == [[]]                                  # the tuner sees only its idea's history
    # ws1-017's first run is worse than the global best: "discard" in the store, "keep" for its own tuner
    r1 = rec("ws1-017", b - 1e-4, b, eid="exp_1")
    history.append(r1)
    assert r1["status"] == "discard" and agent.review(r1) == "continue" and tuners["ws1-017"].reviews == ["keep"]
    exp = agent.propose(history, [])
    assert ideas.idea_of(exp.rationale) == "ws1-018"                       # untried ideas first
    r2 = rec("ws1-018", b - 2e-4, b, eid="exp_2")
    history.append(r2)
    assert agent.review(r2) == "continue"
    local = idea_chooser.local_history(history + [rec("ws1-017", b - 5e-5, b, eid="exp_3"),
                                                  rec("ws1-017", b - 9e-5, b, eid="exp_4"),
                                                  rec("ws1-017", b, b, eid="exp_5", problem="1a")], "ws1-017",
                                       problems={SLUG})
    assert [(r["experiment_id"], r["status"], r["best_before"]) for r in local] == [
        ("exp_1", "keep", None), ("exp_3", "keep", b - 1e-4), ("exp_4", "discard", b - 5e-5)]
    # a "keep" whose score is NaN never becomes the idea's best
    nan = idea_chooser.local_history([rec("ws1-017", float("nan"), None, status="keep", eid="n1"),
                                      rec("ws1-017", 0.5, None, eid="n2"), rec("ws1-017", 0.6, 0.5, eid="n3")], "ws1-017")
    assert [r["status"] for r in nan] == ["invalid", "keep", "keep"]
    # ws1-018's tuner asks to stop after its 2nd result: the idea is paused, the search goes on with the other
    history.append(rec("ws1-018", b - 3e-4, b, eid="exp_6"))
    assert agent.review(history[-1]) == "continue" and agent.paused == {"ws1-018"}
    assert all(ideas.idea_of(agent.propose(history, []).rationale) == "ws1-017" for _ in range(3))
    # when every idea is dead or paused, the loop is told to stop and no proposal is made
    store.set_verdict("ws1-017", _verdict("untested_idea", False, True))
    assert agent.review(rec("ws1-017", b - 1e-4, b)) == "stop"
    _raises(idea_chooser.NoIdeaLeft, agent.propose, history, [])


# ====================================================================== 6. the static page
def _page_inputs(d):
    store = _runnable(d, ids=("ws1-017",))
    store.upsert({"id": "sonar-001", "kind": "untested_idea", "problem": SLUG, "author": "Sonar",
                  "content": "<script>alert('x')</script> Remez-style LP & \"quotes\"", "citations": ["<b>t32</b>"]})
    b = RECORD
    records = [rec(None, b, None, dur=12.0, eid="exp_0000"),
               rec("ws1-017", b + 2.4e-6, b, dur=610.0, eid="exp_0001"),
               rec("ws1-017", b + 2.5e-6, b + 2.4e-6, dur=600.0, eid="exp_0002"),
               rec("ws1-018", b - 1e-4, b + 2.5e-6, dur=LONG, eid="exp_0003"),
               rec("ws1-018", b - 2e-4, b + 2.5e-6, dur=LONG, eid="exp_0004"),
               rec("ws1-018", b - 3e-4, b + 2.5e-6, dur=LONG, eid="exp_0005"),
               rec("ws1-018", b - 4e-4, b + 2.5e-6, dur=LONG, eid="exp_0006"),
               rec("ws1-010", b - 1e-3, b + 2.5e-6, dur=90.0, eid="exp_0007"),
               rec("zzz-404", b, b + 2.5e-6, dur=5.0, eid="exp_0008"),
               rec("ws1-017", b + 9e-6, b, dur=7.0, eid="exp_0009", problem="1a")]
    path = os.path.join(d, "experiments.jsonl")
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(r) for r in records) + "\n{torn line")
    return store, path


def test_map_model_dead_ends_frontier_verdicts_and_compute():
    d = _fresh("page-model")
    store, path = _page_inputs(d)
    L = verdicts.store_label(path)
    records = research_map.load_experiments([path])
    assert len(records) == 10 and {r["_store"] for r in records} == {L}
    m = research_map.build(store.all(), records, REF, SLUG)
    tested = {v["node"]["id"]: v["verdict"] for v in m["tested"]}
    assert {k: v["status"] for k, v in tested.items()} == {"ws1-017": "reproduced", "ws1-018": "refuted",
                                                           "ws1-010": "inconclusive"}
    assert tested["ws1-017"]["experiments"] == [L + "/exp_0001", L + "/exp_0002"]   # the 1a run is not counted
    assert abs(tested["ws1-017"]["best_gain_over_delta"] - 0.24) < 1e-6 and m["single_store"]
    assert sorted(v["node"]["id"] for v in m["dead"]) == ["ws1-010", "ws1-018"]
    frontier = [f["id"] for f in m["frontier"]]
    assert set(frontier) == {"ws1-007", "sonar-001", "ws1-017"} and frontier[-1] == "ws1-017"   # untried first
    assert not set(frontier) & {"ws1-010", "ws1-018", "ws1-001", "ws1-025"}
    compute = {c["id"]: (c["seconds"], c["experiments"]) for c in m["compute"]}
    assert compute == {"ws1-017": (1210.0, 2), "ws1-018": (4 * LONG, 4), "ws1-010": (90.0, 1),
                       research_map.UNTAGGED: (12.0, 1), "zzz-404 (not on the map)": (5.0, 1),
                       "(other problem)": (7.0, 1)}
    assert m["counts"]["experiments"] == 10 and m["counts"]["compute_s"] == 1324.0 + 4 * LONG   # every second counted
    assert m["unknown_tags"] == ["zzz-404"] and m["foreign"] == [("ws1-017", "exp_0009", "1a")]
    assert [v["node"]["id"] for v in m["context"]] == ["ws1-001", "ws1-002"]
    assert dict(m["other_problems"]) == {"circles-rectangle": 1}
    # with the alias, the 1a experiment counts; 0.9 delta above the record is not a record candidate
    m2 = research_map.build(store.all(), records, REF, SLUG, aliases={"1a": SLUG})
    v = {x["node"]["id"]: x["verdict"] for x in m2["tested"]}["ws1-017"]
    assert v["experiments"] == [L + "/%s" % e for e in ("exp_0001", "exp_0002", "exp_0009")] and v["record_candidates"] == []
    # several files: each is its own store, so two stores that both have exp_0001 do not collide
    other = os.path.join(d, "armB.jsonl")
    shutil.copy(path, other)
    both = research_map.load_experiments([path, other])
    m3 = research_map.build(store.all(), both, REF, SLUG)
    v = {x["node"]["id"]: x["verdict"] for x in m3["tested"]}["ws1-017"]
    assert len(both) == 20 and not m3["single_store"] and len(set(v["experiments"])) == 4
    assert v["experiments"][0] == L + "/exp_0001" and v["experiments"][2] == verdicts.store_label(other) + "/exp_0001"
    # an idea with no experiment in the files shows the verdict stored in the map
    store.set_verdict("ws1-007", _verdict("worked", False, True))
    m4 = research_map.build(store.all(), records, REF, SLUG)
    assert {x["node"]["id"]: x["verdict"]["status"] for x in m4["tested"]}["ws1-007"] == "refuted"
    assert "ws1-007" in {x["node"]["id"] for x in m4["dead"]}


def test_page_renders_escaped_and_never_shows_a_stale_record():
    d = _fresh("page")
    store, path = _page_inputs(d)
    out = os.path.join(d, "map.html")
    argv = ["--ideas", str(store.path), "--experiments", path, "--out", out]
    before = store.path.read_text()
    assert research_map.main(argv) == 0
    blind = open(out, encoding="utf-8").read()
    # the node says #1 = 0.963638667992813, which is stale: the page does not present it as the record to
    # beat, and with no record read nothing is called a gain
    assert "The record was not read" in blind and "Record to beat" not in blind and "Send to the referee" not in blind
    assert "reproduced by us" not in blind and "refuted by us" in blind
    assert research_map.main(argv + ["--record", repr(RECORD)]) == 0
    page = open(out, encoding="utf-8").read()
    assert store.path.read_text() == before                                # rendering writes nothing but the page
    for needle in ("Research map: " + SLUG, "Dead ends</h3>", "Tested by us</h3>", "Where the compute went</h3>",
                   "Dead ends, and why", "Frontier: what to try next", "Verdicts: what our experiments say",
                   "Compute spent per idea", "reproduced by us", "refuted by us", "no runner yet",
                   "optimizers/ws1_017.py", "<code>exp_0001</code>", "zzz-404", "20.2 min", "+0.24 δ",
                   "never a record", "Dense re-optimization of all coordinates", "it never gained",
                   "best of the other ideas"):
        assert needle in page, needle
    assert "<script>alert" not in page and "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in page
    assert "<b>t32</b>" not in page and "&lt;b&gt;t32&lt;/b&gt;" in page
    assert page.count("<table>") == page.count("</table>") and page.strip().endswith("</html>")
    # a typed record is shown as typed, never as read
    assert research_map.main(argv + ["--record", repr(RECORD), "--alias", "1a=" + SLUG]) == 0
    page = open(out, encoding="utf-8").read()
    assert "Record to beat <b>%r</b>" % RECORD in page and "Send to the referee" not in page
    assert "record TYPED on the command line, not read: it may be stale" in page
    assert research_map.main(argv + ["--record", repr(RECORD - 2e-5), "--alias", "1a=" + SLUG,
                                     "--record-read-at", "2026-11-14T15:35Z"]) == 0
    page = open(out, encoding="utf-8").read()
    # 2.24 delta above a TYPED record: shown as such in the header, and never offered to the referee,
    # because only a record read live can back that
    assert "record TYPED on the command line, said to be read 2026-11-14T15:35Z" in page
    assert "Send to the referee" not in page
    # --save-verdicts writes the recomputed verdicts back; an empty map is an error, not an empty page
    assert research_map.main(argv + ["--save-verdicts", "--record", repr(RECORD)]) == 0
    assert store.get("ws1-018")["status"] == "refuted" and store.get("ws1-017")["status"] == "reproduced"
    assert research_map.main(argv) == 0 and store.get("ws1-017")["verdict"]["experiments"] == [
        verdicts.store_label(path) + "/exp_0001", verdicts.store_label(path) + "/exp_0002"]
    assert research_map.main(["--ideas", os.path.join(d, "none.jsonl"), "--out", out]) == 1
    assert research_map.main(["--ideas", str(store.path), "--experiments", "--problem", "circles-rectangle",
                              "--out", out]) == 0
    assert "Another contact graph." in open(out, encoding="utf-8").read()
    # a problem this file does not know: with experiments, the direction must be said, never guessed
    store.upsert({"id": "x-1", "kind": "untested_idea", "problem": "sidon-45-set", "content": "c"})
    _raises(SystemExit, research_map.main, argv + ["--problem", "sidon-45-set"])
    assert research_map.main(argv + ["--problem", "sidon-45-set", "--direction", "minimize"]) == 0
    assert "Goal: minimize the score" in open(out, encoding="utf-8").read()
    # the experiments table is capped, and says so
    many = os.path.join(d, "many.jsonl")
    with open(many, "w") as f:
        f.write("\n".join(json.dumps(rec("ws1-017", RECORD - 1e-6 * i, RECORD, eid="m%04d" % i)) for i in range(305)))
    assert research_map.main(["--ideas", str(store.path), "--experiments", many, "--out", out]) == 0
    # 305 runs from this file, plus the 5 stored runs of the two ideas that have none in it
    assert "Showing the last 300 of 310 experiments." in open(out, encoding="utf-8").read()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print("all %d research map tests passed" % len(tests))
