"""Five independent SAVANT researchers. Shared report; distinct system prompts."""
from __future__ import annotations

import json
from typing import Any

from savant.llm import LLMClient
from savant.schemas import (
    CRITIQUE_BATCH_JSON_SCHEMA,
    CRITIQUE_JSON_SCHEMA,
    HYPOTHESIS_JSON_SCHEMA,
    MANAGER_JSON_SCHEMA,
    REVISION_JSON_SCHEMA,
    ROLES,
    STOP_VOTE_JSON_SCHEMA,
    VOTE_JSON_SCHEMA,
    ensure,
)

NOVELTY_RULES = (
    "Executable work must use a vetted template (probe, probe_mutate, blockshift, blockshift_mutate, "
    "hybrid_refresh) with config overrides inside its bounds; anything that needs new code goes to "
    "queue_novel with a precise algorithmic description. Do not resubmit a strategy listed in "
    "completed_strategies, and do not propose a reseeded copy of probe/blockshift: a new executable "
    "proposal must change the search mechanism (scales, guidance, refresh, step grid) for a stated "
    "mathematical or numerical reason that the verified results support or would falsify."
)


def _tag(raw: dict, client: LLMClient) -> dict:
    meta = getattr(client, "last_meta", None)
    if meta:
        raw["llm"] = {k: meta.get(k) for k in ("model", "reasoning_effort", "response_id", "usd",
                                               "reasoning_tokens", "output_tokens") if k in meta}
    return raw

ROLE_PROMPTS = {
    "mathematical": (
        "You are the Mathematical Savant. Exploit structure of the second-autocorrelation inequality: "
        "symmetries, equioscillation, scale invariance, analytic bounds. Prefer reformulations and "
        "geometry of the feasible set over brute force. Map ideas to vetted templates "
        "(blockshift_mutate / queue_novel). Never invent fake theorems."
    ),
    "numerical": (
        "You are the Numerical Savant. Focus on FFT/oaconvolve cost, exact incremental updates, "
        "D-grids, golden search, numerical stability, vectorization, gradients only if they can beat "
        "the known equioscillation failure mode. Prefer blockshift_mutate or probe_mutate."
    ),
    "evolutionary": (
        "You are the Evolutionary Savant. Focus on stochastic search, diversity of block sizes, "
        "mutation rates, adaptive polish, population-style restarts via seeds. Prefer probe_mutate "
        "or blockshift_mutate. Avoid claiming significance from n=3."
    ),
    "statistical": (
        "You are the Statistical Savant (STAT 201B). Formulate falsifiable hypotheses, demand matched "
        "seeds and equal budgets, watch power/MDE, overfitting to seeds, multiple comparisons, and "
        "sequential testing. Prefer replication (stock blockshift/probe) before exotic mutations. "
        "Never claim p<0.05 from 3 pairs."
    ),
    "literature": (
        "You are the Literature Savant. Cite real references (arXiv ids, digest entries) when possible. "
        "Clearly separate verified facts from speculative applications. If proposing a method without a "
        "vetted optimizer file, use template queue_novel. Do not fabricate papers."
    ),
}


def propose(role: str, problem_report: dict, memory_summary: list[dict], client: LLMClient,
            evidence: dict | None = None) -> dict:
    if role not in ROLES:
        raise ValueError(role)
    system = ROLE_PROMPTS[role] + (
        "\nReturn ONE JSON hypothesis matching the schema. Be concrete and distinct from other roles. "
        + NOVELTY_RULES + f" role={role}"
    )
    user = json.dumps({
        "role": role,
        "problem_report_sha": problem_report.get("content_sha256"),
        "problem_report": _compact_report(problem_report),
        "verified_evidence": evidence or {},
        "research_memory_tail": memory_summary[-20:],
        "template_enum": list(HYPOTHESIS_JSON_SCHEMA["properties"]["template"]["enum"]),
        "instruction": "Propose one executable or queued hypothesis. No Python code.",
    }, default=str)
    raw = client.complete_json(system, user, schema_name="hypothesis", schema=HYPOTHESIS_JSON_SCHEMA,
                               purpose=f"propose:{role}")
    raw["role"] = role
    return _tag(ensure("hypothesis", raw), client)


def critique_batch(reviewer_role: str, hypotheses: list[dict], problem_report: dict, client: LLMClient,
                   evidence: dict | None = None) -> list[dict]:
    """One request: the reviewer critiques every listed (anonymized) proposal."""
    system = (
        f"You are the {reviewer_role} Savant in an adversarial scientific review of ANONYMIZED proposals. "
        "For each proposal, attack its weakest assumption, check it against the verified evidence, and score "
        "validity, novelty, expected_gain realism, compute_cost, testability and an overall score in [0,1]. "
        "Return exactly one critique per proposal with target_id equal to its hypothesis_id. "
        f"role={reviewer_role}"
    )
    anon = [{k: v for k, v in h.items() if k not in ("role", "llm")} for h in hypotheses]
    user = json.dumps({
        "anonymized_hypotheses": anon,
        "baseline_facts": _compact_report(problem_report)["measured_facts_brief"],
        "verified_evidence": evidence or {},
    }, default=str)
    raw = client.complete_json(system, user, schema_name="critique_batch", schema=CRITIQUE_BATCH_JSON_SCHEMA,
                               purpose=f"critique:{reviewer_role}")
    wanted = {h["hypothesis_id"] for h in hypotheses}
    out, seen = [], set()
    for c in raw.get("critiques") or []:
        tid = c.get("target_id")
        if tid not in wanted or tid in seen:
            continue
        seen.add(tid)
        c["reviewer_role"] = reviewer_role
        out.append(_tag(ensure("critique", c), client))
    return out


def critique(reviewer_role: str, hypothesis: dict, problem_report: dict, client: LLMClient) -> dict:
    system = (
        f"You are the {reviewer_role} Savant reviewing an ANONYMIZED proposal. "
        "Score validity, novelty, expected_gain realism, compute_cost, testability in [0,1]. "
        f"role={reviewer_role}"
    )
    anon = {k: v for k, v in hypothesis.items() if k != "role"}
    user = json.dumps({
        "anonymized_hypothesis": anon,
        "baseline_facts": _compact_report(problem_report)["measured_facts_brief"],
    }, default=str)
    raw = client.complete_json(system, user, schema_name="critique", schema=CRITIQUE_JSON_SCHEMA,
                               purpose=f"critique:{reviewer_role}")
    raw["reviewer_role"] = reviewer_role
    raw["target_id"] = hypothesis["hypothesis_id"]
    return _tag(ensure("critique", raw), client)


def revise(role: str, hypothesis: dict, critiques: list[dict], client: LLMClient) -> dict:
    system = (
        f"You are the {role} Savant responding to peer critiques. You may revise once with explicit "
        f"reasoning, or keep the proposal. {NOVELTY_RULES} role={role}"
    )
    user = json.dumps({"hypothesis": {k: v for k, v in hypothesis.items() if k != "llm"},
                       "critiques": [{k: v for k, v in c.items() if k != "llm"} for c in critiques]},
                      default=str)
    raw = client.complete_json(system, user, schema_name="revision", schema=REVISION_JSON_SCHEMA,
                               purpose=f"revise:{role}")
    raw["role"] = role
    raw["hypothesis_id"] = hypothesis["hypothesis_id"]
    out = _tag(ensure("revision", raw), client)
    if out.get("hypothesis"):
        out["hypothesis"]["llm"] = out.get("llm")
    return out


def vote(role: str, hypotheses: list[dict], critiques: list[dict], client: LLMClient) -> dict:
    system = (
        f"You are the {role} Savant casting a ranked vote. No self-voting. Penalize duplicates, "
        f"untestable claims, unsupported optimism and reseeded repeats of strategies that already ran. "
        f"Prefer a diversified portfolio. role={role}"
    )
    others = [{k: v for k, v in h.items() if k != "llm"} for h in hypotheses if h["role"] != role]
    user = json.dumps({"candidates": others,
                       "critiques": [{k: v for k, v in c.items() if k != "llm"} for c in critiques]},
                      default=str)
    raw = client.complete_json(system, user, schema_name="vote", schema=VOTE_JSON_SCHEMA, purpose=f"vote:{role}")
    raw["voter_role"] = role
    # strip self votes if model included them
    own_ids = {h["hypothesis_id"] for h in hypotheses if h["role"] == role}
    raw["ranking"] = [x for x in raw.get("ranking", []) if x not in own_ids]
    return _tag(ensure("vote", raw), client)


def stop_vote(role: str, context: dict, client: LLMClient) -> dict:
    system = (
        f"You are the {role} Savant in a Continue/Pivot/Stop meeting. Decide among CONTINUE, "
        "PIVOT, MORE_REPLICATION, STOP from the verified results, paired statistics, novelty left to test, "
        "diminishing returns and the remaining budget. STOP only for scientific evidence or budget. "
        f"additional_jobs in 0..40 (a hint; the allocator enforces caps). role={role}"
    )
    user = json.dumps(context, default=str)
    raw = client.complete_json(system, user, schema_name="stop_vote", schema=STOP_VOTE_JSON_SCHEMA,
                               purpose=f"stop:{role}")
    raw["role"] = role
    return _tag(ensure("stop_vote", raw), client)


def manager_review(shortlist: list[dict], context: dict, client: LLMClient, *, max_select: int) -> dict:
    """Manager picks the wave from the council's executable shortlist. Caps are enforced afterwards in code."""
    system = (
        "You are the SAVANT research manager. From the council's ranked, executable shortlist choose up to "
        f"{max_select} strategies for the next Modal wave. Maximize expected information and verified gain: "
        "favour diversity of mechanism, drop reseeded repeats and strategies whose verified gains were flat, "
        "and keep at least one comparison that the paired statistics can falsify. seeds_per_strategy in 3..10 "
        "(paired seeds), budget_s_request 60 or 300 (300 is only granted after verified improvement). "
        "List every shortlisted id you drop in rejections with the reason."
    )
    user = json.dumps({"shortlist": shortlist, "context": context}, default=str)
    raw = client.complete_json(system, user, schema_name="manager", schema=MANAGER_JSON_SCHEMA, purpose="manager")
    return _tag(ensure("manager", raw), client)


def _compact_report(report: dict) -> dict:
    mf = report.get("measured_facts", {})
    return {
        "problem": report.get("problem"),
        "measured_facts_brief": {
            "verified_baseline_score": mf.get("verified_baseline_score"),
            "distance_to_record_threshold": mf.get("distance_to_record_threshold"),
            "certificate_n": (mf.get("certificate_distribution") or {}).get("n_values"),
            "autocorr_near_max": ((mf.get("certificate_distribution") or {})
                                  .get("autocorrelation") or {}).get("frac_within_1e-7_of_max"),
            "historical": mf.get("historical_modal_campaign") and {
                "run_id": mf["historical_modal_campaign"].get("run_id"),
                "probe_mean": mf["historical_modal_campaign"]["probe"]["mean"],
                "blockshift_mean": mf["historical_modal_campaign"]["blockshift"]["mean"],
                "paired": mf["historical_modal_campaign"]["paired_blockshift_vs_probe"]["mean_diff"],
                "caveat": mf["historical_modal_campaign"]["caveat"],
            },
        },
        "known_failure_modes": report.get("known_failure_modes"),
        "search_opportunities": report.get("search_opportunities"),
        "unknowns": report.get("unknowns"),
        "content_sha256": report.get("content_sha256"),
    }
