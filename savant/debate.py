"""Independent proposals → blind review → debate → ranked voting → manager selection."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from savant import researchers as R
from savant.llm import LLMCallFailed, LLMClient, LLMUnavailable
from savant.memory import ResearchMemory
from savant.schemas import ROLES
from savant.templates import materialize


def _open_round(memory: ResearchMemory) -> list[dict]:
    """Records since the last manager or iteration decision. An interrupted council lives here."""
    rows = memory.read()
    start = 0
    for i, row in enumerate(rows):
        if row.get("kind") in ("manager_decision", "iteration_decision"):
            start = i + 1
    return rows[start:]


def _body(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in ("ts", "kind", "source")}


def run_council(problem_report: dict, memory: ResearchMemory, client: LLMClient,
                *, voting_seed: int = 20261115, max_select: int = 2, evidence: dict | None = None,
                batch_critiques: bool = False, manager_context: dict | None = None) -> dict:
    mem = memory.summary_for_agents()
    pending = _open_round(memory)

    def attempt(stage: str, fn, *args, **kwargs):
        """One paid request. Budget stops propagate; a failed request is logged and skipped, never retried."""
        try:
            return fn(*args, **kwargs)
        except LLMUnavailable as exc:
            memory.append("note", {"kind_detail": "llm_call_failed", "stage": stage, "error": str(exc)[:400],
                                   "pause": True}, source="savant:llm")
            raise
        except LLMCallFailed as exc:
            memory.append("note", {"kind_detail": "llm_call_failed", "stage": stage, "error": str(exc)[:400],
                                   "llm": getattr(client, "last_meta", None)}, source="savant:llm")
            return None
    saved_h = [r for r in pending if r.get("kind") == "hypothesis"]
    latest: dict[str, dict] = {}
    for row in saved_h:
        if row.get("role"):
            latest[row["role"]] = _body(row)
    critiques_saved = [_body(r) for r in pending if r.get("kind") == "critique"]
    crit_key = {(c.get("reviewer_role"), c.get("target_id")): c for c in critiques_saved}
    revised_ids = {r.get("hypothesis_id") for r in pending
                   if r.get("kind") == "note" and r.get("kind_detail") == "revision"}
    votes_saved = {_body(r).get("voter_role"): _body(r) for r in pending if r.get("kind") == "vote"}

    # --- Independent proposals (reuse a role already stored in this open round) ---
    hypotheses = []
    for role in ROLES:
        if role in latest:
            hypotheses.append(latest[role])
            continue
        h = attempt(f"propose:{role}", R.propose, role, problem_report, mem, client, evidence=evidence)
        if h is None:
            continue
        hypotheses.append(h)
        memory.append("hypothesis", {"stage": "independent", **h}, source=f"savant:{role}")

    # --- Round 1: blind critiques (anonymized in prompt) ---
    critiques = []
    for reviewer in ROLES:
        todo = []
        for h in hypotheses:
            if h["role"] == reviewer:
                continue
            key = (reviewer, h.get("hypothesis_id"))
            if key in crit_key:
                critiques.append(crit_key[key])
                continue
            todo.append(h)
        if not todo:
            continue
        if batch_critiques:
            got = attempt(f"critique:{reviewer}", R.critique_batch, reviewer, todo, problem_report, client,
                          evidence=evidence) or []
        else:
            got = [c for c in (attempt(f"critique:{reviewer}", R.critique, reviewer, h, problem_report, client)
                               for h in todo) if c is not None]
        for c in got:
            critiques.append(c)
            memory.append("critique", c, source=f"savant:{reviewer}")

    # --- Round 2: revisions + skeptical challenge ---
    revised = []
    for h in hypotheses:
        if h.get("hypothesis_id") in revised_ids or h.get("stage") == "revised":
            revised.append(h)
            continue
        mine = [c for c in critiques if c["target_id"] == h["hypothesis_id"]]
        rev = attempt(f"revise:{h['role']}", R.revise, h["role"], h, mine, client)
        if rev is None:
            revised.append(h)
            continue
        memory.append("note", {"kind_detail": "revision", **rev}, source=f"savant:{h['role']}")
        if rev.get("revised") and rev.get("hypothesis"):
            h = rev["hypothesis"]
            h["role"] = rev.get("role", h.get("role"))
            memory.append("hypothesis", {"stage": "revised", **h}, source=f"savant:{h['role']}")
        # skeptical statistical challenge recorded
        if h["expected_gain"] >= 1e-5 and h["template"] != "queue_novel":
            memory.append("uncertainty", {
                "topic": h["hypothesis_id"],
                "statement": "Skeptical reviewer: expected_gain near record delta without prior evidence at this scale.",
                "challenger": "statistical",
            }, source="savant:skeptic")
        revised.append(h)
    hypotheses = revised

    # --- Round 3: ranked voting (no self-votes) ---
    votes = []
    for role in ROLES:
        if role in votes_saved:
            votes.append(votes_saved[role])
            continue
        if not any(h["role"] != role for h in hypotheses):
            continue
        v = attempt(f"vote:{role}", R.vote, role, hypotheses, critiques, client)
        if v is None:
            continue
        votes.append(v)
        memory.append("vote", v, source=f"savant:{role}")

    if manager_context is None:
        return finish_council(hypotheses, critiques, votes, memory,
                              voting_seed=voting_seed, max_select=max_select)
    # Deterministic ranking produces a wider shortlist; the manager model chooses the wave from it.
    council = finish_council(hypotheses, critiques, votes, memory, voting_seed=voting_seed,
                             max_select=max(max_select * 2, max_select), log_decision=False)
    shortlist = [{k: v for k, v in materialize(h).items() if k not in ("rejected_overrides",)}
                 for h in council["selected"]]
    ranking_scores = {r["hypothesis_id"]: r["score"] for r in council["ranking"]}
    for s in shortlist:
        s["council_score"] = ranking_scores.get(s["hypothesis_id"])
    review = attempt("manager", R.manager_review, shortlist, manager_context, client, max_select=max_select)
    by_id = {h["hypothesis_id"]: h for h in council["selected"]}
    if review is not None:
        picked = [by_id[i] for i in dict.fromkeys(review.get("selected_ids") or []) if i in by_id][:max_select]
        override = None
        if not picked:
            picked = council["selected"][:max_select]
            override = "manager selected no shortlisted id; council order kept"
    else:
        picked = council["selected"][:max_select]
        override = "manager call failed; council order kept"
    memory.append("manager_decision", {
        "selected_ids": [h["hypothesis_id"] for h in picked],
        "shortlist_ids": [s["hypothesis_id"] for s in shortlist],
        "ranking_ids": [r["hypothesis_id"] for r in council["ranking"]],
        "override": override is not None,
        "override_reason": override,
        "manager_review": review,
        "note": "Consensus is not mathematical proof; frozen verifier decides scores.",
    }, source="savant:manager")
    council["selected"] = picked
    council["materialized"] = [materialize(h) for h in picked]
    council["manager_review"] = review
    return council


def finish_council(hypotheses, critiques, votes, memory: ResearchMemory, *,
                   voting_seed: int, max_select: int, log_decision: bool = True) -> dict:
    """Rank and select. Every proposal is materialized first; a bad one is logged, not fatal."""
    materialized = []
    for h in hypotheses:
        mat = materialize(h)
        materialized.append(mat)
        if mat.get("rejected_overrides"):
            memory.append("note", {
                "kind_detail": "rejected_overrides",
                "hypothesis_id": h.get("hypothesis_id"),
                "template": h.get("template"),
                "rejected": mat["rejected_overrides"],
            }, source="savant:templates")
        if not mat.get("executable") and h.get("template") != "queue_novel":
            memory.append("rejected_idea", {
                "idea": h.get("hypothesis_id"),
                "reason": mat.get("reason") or "not executable",
                "queued": False,
            }, source="savant:templates")
    ranking = aggregate_votes(hypotheses, votes, critiques, seed=voting_seed)
    selected = manager_select(ranking, hypotheses, max_select=max_select, memory=memory,
                              log_decision=log_decision)
    return {
        "hypotheses": hypotheses,
        "critiques": critiques,
        "votes": votes,
        "ranking": ranking,
        "selected": selected,
        "materialized": [materialize(h) for h in selected],
    }


_MEMORY_META = ("ts", "kind", "source", "stage")


def council_from_memory(memory: ResearchMemory) -> dict | None:
    """Rebuild a finished propose/critique/vote round from research_memory.jsonl.

    Uses the latest record per hypothesis id (a revision replaces the independent proposal).
    Returns None when the round is incomplete, so the caller does not skip a real council.
    """
    rows = memory.read()
    order: list[str] = []
    latest: dict[str, dict] = {}
    for row in rows:
        if row.get("kind") != "hypothesis" or not row.get("hypothesis_id"):
            continue
        hid = row["hypothesis_id"]
        if hid not in latest:
            order.append(hid)
        latest[hid] = {k: v for k, v in row.items() if k not in _MEMORY_META}
    hypotheses = [latest[hid] for hid in order]
    critiques = [{k: v for k, v in row.items() if k not in _MEMORY_META}
                 for row in rows if row.get("kind") == "critique"]
    votes = [{k: v for k, v in row.items() if k not in _MEMORY_META}
             for row in rows if row.get("kind") == "vote"]
    roles = {h.get("role") for h in hypotheses}
    if len(hypotheses) < len(ROLES) or roles != set(ROLES):
        return None
    if len(critiques) < len(ROLES) * (len(ROLES) - 1) or len(votes) < len(ROLES):
        return None
    return {"hypotheses": hypotheses, "critiques": critiques, "votes": votes}


def aggregate_votes(hypotheses: list[dict], votes: list[dict], critiques: list[dict],
                    *, seed: int) -> list[dict]:
    """Borda-like scores with penalties; ties broken by seeded randomness."""
    ids = [h["hypothesis_id"] for h in hypotheses]
    by_id = {h["hypothesis_id"]: h for h in hypotheses}
    scores = defaultdict(float)
    n = len(ids)
    for v in votes:
        for rank, hid in enumerate(v["ranking"]):
            if hid in by_id:
                scores[hid] += (n - rank)
    # critique average score
    crit_avg = defaultdict(list)
    for c in critiques:
        crit_avg[c["target_id"]].append(c["score"])
    for hid, xs in crit_avg.items():
        scores[hid] += 0.5 * (sum(xs) / len(xs))
    # penalties
    titles = defaultdict(list)
    for h in hypotheses:
        key = (h["template"], tuple(sorted((h.get("config_overrides") or {}).items())))
        titles[key].append(h["hypothesis_id"])
    for key, group in titles.items():
        if len(group) > 1:
            for hid in group:
                scores[hid] -= 1.0  # duplicate penalty
    for h in hypotheses:
        if h["template"] == "queue_novel":
            scores[h["hypothesis_id"]] -= 0.25  # not executable this round
        if h["expected_gain"] > 1e-5 and h["template"] != "queue_novel":
            scores[h["hypothesis_id"]] -= 0.5  # unsupported optimism vs measured ~1e-8
        if not h.get("falsification"):
            scores[h["hypothesis_id"]] -= 2.0

    rng = random.Random(seed)
    ordered = sorted(ids, key=lambda hid: (scores[hid], rng.random()), reverse=True)
    return [{"hypothesis_id": hid, "score": scores[hid], "hypothesis": by_id[hid]} for hid in ordered]


def manager_select(ranking: list[dict], hypotheses: list[dict], *, max_select: int,
                   memory: ResearchMemory, log_decision: bool = True) -> list[dict]:
    """Select up to max_select distinct executable templates; log overrides."""
    picked = []
    seen_templates = set()
    override_reason = None
    for row in ranking:
        h = row["hypothesis"]
        try:
            mat = materialize(h)
        except Exception as exc:  # noqa: BLE001  a bad proposal is a rejection, not a crash
            memory.append("rejected_idea", {
                "idea": h.get("hypothesis_id"), "reason": f"{type(exc).__name__}: {exc}", "queued": False,
            }, source="savant:manager")
            continue
        if not mat.get("executable"):
            memory.append("rejected_idea", {
                "idea": h["hypothesis_id"], "reason": mat.get("reason"), "queued": bool(mat.get("queued")),
            }, source="savant:manager")
            continue
        # diversify: avoid two identical template+config
        sig = (mat["optimizer"], tuple(sorted(mat["config"].items())))
        if sig in seen_templates:
            continue
        seen_templates.add(sig)
        picked.append(h)
        if len(picked) >= max_select:
            break
    # Prefer including at most one pure replication if two mutations are clones — already handled
    if len(picked) < max_select:
        override_reason = f"Only {len(picked)} distinct executable proposals available"
    decision = {
        "selected_ids": [h["hypothesis_id"] for h in picked],
        "ranking_ids": [r["hypothesis_id"] for r in ranking],
        "override": False,
        "override_reason": override_reason,
        "note": "Consensus is not mathematical proof; frozen verifier decides scores.",
    }
    if log_decision:
        memory.append("manager_decision", decision, source="savant:manager")
    return picked


def stop_meeting(context: dict, memory: ResearchMemory, client: LLMClient) -> dict:
    votes = []
    rows = memory.read()
    start = max((i + 1 for i, r in enumerate(rows) if r.get("kind") == "iteration_decision"), default=0)
    stored = {r.get("role"): _body(r) for r in rows[start:] if r.get("kind") == "stop_vote"}
    for role in ROLES:
        if role in stored:
            votes.append(stored[role])
            continue
        try:
            v = R.stop_vote(role, context, client)
        except LLMUnavailable:
            raise
        except LLMCallFailed as exc:
            memory.append("note", {"kind_detail": "llm_call_failed", "stage": f"stop:{role}",
                                   "error": str(exc)[:400]}, source="savant:llm")
            continue
        votes.append(v)
        memory.append("stop_vote", v, source=f"savant:{v['role']}")
    if not votes:
        decision = {"decision": "STOP", "additional_jobs": 0, "vote_counts": {},
                    "reason": "no stop vote could be obtained; stopping rather than spending blind",
                    "votes": [], "selected_hypothesis_ids": context.get("candidate_ids", []),
                    "llm_model": getattr(client.cfg, "model", None),
                    "reasoning_effort": getattr(client.cfg, "reasoning_effort", None)}
        memory.append("iteration_decision", decision, source="savant:manager")
        return decision
    # Manager aggregation: majority with statistical savant tie-break preference for replication
    counts = defaultdict(int)
    for v in votes:
        counts[v["decision"]] += 1
    stat = next((v for v in votes if v["role"] == "statistical"), {"decision": "MORE_REPLICATION"})
    # Priority if tied: STOP > MORE_REPLICATION > PIVOT > CONTINUE (conservative under uncertainty)
    order = ["STOP", "MORE_REPLICATION", "PIVOT", "CONTINUE"]
    best = max(order, key=lambda d: (counts[d], order.index(d) == order.index(stat["decision"])))
    # If statistical insists MORE_REPLICATION and uncertainty high, prefer that over CONTINUE
    if stat["decision"] == "MORE_REPLICATION" and context.get("uncertainty_high"):
        best = "MORE_REPLICATION"
    jobs = max(v["additional_jobs"] for v in votes if v["decision"] == best) if any(
        v["decision"] == best for v in votes) else 0
    jobs = min(40, max(0, jobs))  # hint only; allocate_wave sizes the Modal wave
    decision = {
        "decision": best,
        "additional_jobs": jobs if best != "STOP" else 0,
        "vote_counts": dict(counts),
        "statistical_vote": stat,
        "votes": votes,
        "selected_hypothesis_ids": context.get("candidate_ids", []),
        "budget_justification": context.get("budget"),
        "expected_information_gain": sum(v["expected_information_gain"] for v in votes) / len(votes),
        "llm_model": getattr(client.cfg, "model", None),
        "reasoning_effort": getattr(client.cfg, "reasoning_effort", None),
    }
    memory.append("iteration_decision", decision, source="savant:manager")
    return decision
