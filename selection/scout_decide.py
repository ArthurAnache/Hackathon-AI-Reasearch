"""Problem scout, last step: write selection/problem_scout.jsonl (one line per live problem, with a decision).

Usage (from the repo root):  .venv/bin/python selection/scout_decide.py

Inputs: selection/scout_live.jsonl (live facts, from scout_refresh.py), selection/scout_third_autocorr_runs.jsonl
and selection/scout_two_deletion_runs.jsonl (experiments), selection/scout_milo_replay.json (MILO's published
claim), and the curated notes below. Like `LITERATURE` in literature_check.py, a literature value counts only once
it is curated here with its URL, access date, label and whether its definition matches the EinsteinArena verifier.

Decisions: `switch_candidate` (worth moving the main effort to), `keep_as_backup`, `not_now`.
Labels: verified = reproduced locally with the pinned verifier; claim / hypothesis = read, not reproduced;
limitation = known gap. This script proposes; Arthur decides, and only WS1 freezes a problem.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "problem_scout.jsonl"
ACCESSED = "2026-11-15"  # UTC date of every web page read for this file (2026-11-14 evening PT)
EA = "https://einsteinarena.com/problems/"


def runs(path: str) -> dict:
    p = ROOT / path
    return {json.loads(l)["experiment_id"].split("/")[-1]: json.loads(l) for l in p.read_text().splitlines()} \
        if p.exists() else {}


THIRD = runs("scout_third_autocorr_runs.jsonl")
TWO = runs("scout_two_deletion_runs.jsonl")
MILO = json.loads((ROOT / "scout_milo_replay.json").read_text())


def third_candidates() -> list[dict]:
    out = []
    for tag, r in THIRD.items():
        if tag.startswith("milo_replay") and r.get("valid"):
            stored = ROOT / "scout_candidates" / "third-autocorrelation-inequality" / f"{tag}.json.gz"
            out.append({"tag": tag, "n": r["n"], "score_pinned": r["score"],
                        "stored_file": str(stored.relative_to(ROOT.parent)) if stored.exists() else None,
                        "solution_sha256": r["solution_sha256"], "wall_s": r["wall_s"],
                        "label": "verified (pinned verifier, clean process); candidate, not a record claim"})
    return sorted(out, key=lambda c: c["score_pinned"])


CURATED: dict[str, dict] = {
    "third-autocorrelation-inequality": {
        "rank": 1, "decision": "switch_candidate",
        "rationale": (
            "Only problem with literature headroom of many delta under the same verifier AND a way to reach it "
            "tonight. MILO (arXiv:2609.38349, App. E) claims 1.4488860143 on 25,600 grid points, scored with the "
            "arena verifier, 22 delta below the live #1; its construction is not published and not on the "
            "leaderboard. Its project page publishes a plot-resolution copy of the record (3,200 of the 25,600 "
            "values, 6 decimals), which scores 1.6558 as given. Block-repeating that copy x8 and re-optimizing it "
            "with one log-sum-exp cooling path (our re-implementation, 5 to 7 minutes on one core) gives candidates "
            "that the pinned verifier scores 23.7 to 28.5 delta below the live #1 and 1.3 to 6.1 delta below MILO's "
            "claimed value (seven candidates: five direct paths, one re-melt hop, one x2 cascade; all below both); "
            "an independent exact-rational scorer agrees to 1.3e-15. "
            "Without that starting point the same method gains nothing from the live #1 in 5 minutes (0 delta) and "
            "0.4 delta from BasinHopper's solution in 40 s. The candidates derive from MILO's published data and "
            "must be credited to it: whether "
            "that counts as our record is Arthur's call. Evaluation is cheap (FFT of 51,200 to 262,144 points), so "
            "CPU is enough."),
        "literature": [
            {"value": MILO["claimed_record"], "what": "MILO record, 25,600 grid points, 2026-09-15",
             "source": "https://jprithwish.github.io/MILO/ (demo data static/einstein-run.js) and "
                       "https://arxiv.org/abs/2609.38349 (App. E.1, Table 13: 1.4488860143)",
             "accessed_on": ACCESSED, "definition_matches_verifier": True,
             "definition_note": "paper: 'minimize |max(f*f)| / (int f)^2, f signed', 'rescored with the arena's own "
                                "verifier'; same quantity as the pinned verifier",
             "label": "claim: the construction is not public; we reproduced the basin, not their exact value"},
            {"value": 1.4557, "what": "AlphaEvolve, C' (signed max)", "source": "https://arxiv.org/abs/2511.02864",
             "accessed_on": "2026-11-14", "definition_matches_verifier": True, "label": "read in source"},
            {"value": 1.457854884921989, "what": "Tao's table constant 1d = max |f*f| / (int f)^2",
             "source": "https://teorth.github.io/optimizationproblems/constants/1d.html", "accessed_on": ACCESSED,
             "definition_matches_verifier": False, "label": "not comparable: a different constant (abs inside the max)"},
        ],
        "method_evidence": [
            "MILO App. E.2: log-sum-exp objective with FFT gradient, L-BFGS stages on a geometric cooling ladder, "
            "chain of re-melt hops with multiplicative jitter, 60 s runs, on 25,600 points, started from arena "
            "rank 3 (BasinHopper, 1.4515829).",
            "MILO project page: the run took 12.7 days to its record and 11.8 days to go below today's live #1; "
            "41 harness evaluations, 403 sub-record constructions (claim).",
            "Forum t181 (JSAgent, untrusted): block-repeat + noise + re-optimization gains at each resolution "
            "doubling; a reply fits C(n) = C_inf + A/n with A about 1.6 in that basin (claim).",
        ],
        "unverified": [
            "MILO's own value (construction not public).",
            "Server-side score of our candidates (nothing was submitted; server numpy version is not pinned).",
            "Whether a result derived from MILO's published replay is acceptable as the team's record.",
            "The problem is not frozen in problems/: verify_claim.py has not arbitrated these candidates.",
        ],
        "to_freeze_if_switch": [
            "verifier_pinned.py: sha256 b8288d5943d72032... (live == pinned on 2026-11-15, see registry_stage2)",
            "certificate: live #1, solution 2617 (Atris), 65,536 values, score 1.449109789331911",
            "independent.py: selection/scout_third_autocorr_exact.py (exact rationals, Kronecker substitution)",
            "record to beat: the lower of the live #1 and MILO's claimed 1.4488860142647635, minus delta",
        ],
        "evidence_urls": [
            EA + "third-autocorrelation-inequality", "https://arxiv.org/abs/2609.38349",
            "https://jprithwish.github.io/MILO/", "https://jprithwish.github.io/MILO/static/einstein-run.js",
            EA + "third-autocorrelation-inequality/threads/181",
            "https://teorth.github.io/optimizationproblems/constants/1d.html",
            "https://discoveria.ai/frontiers/einsteinarena-third-autocorrelation",
        ],
    },
    "kakeya-needle-128": {
        "rank": 2, "decision": "keep_as_backup",
        "rationale": (
            "Strongest sign of headroom after the lead: the record moved by 2.55e-6 (2,550 delta) four days ago and "
            "by 4.6e-5 over 60 days, through five successive record holders since 2026-09-08; delta is 1e-9. 128 "
            "variables, exact rational verifier, and the leaderboard is ahead of the literature (0.107067, the "
            "Station). The method that worked is described: group moves (resize, transfer, reflect groups of "
            "nearby directions) alternated with relaxation of all offsets, accepting temporary increases. Not "
            "tested here; stage 3's generic probe found no gain in 10 minutes at 1.07 s per evaluation, so it "
            "needs a fast float event scorer first. The platform marks statement and verifier 'under active "
            "review'."),
        "literature": [
            {"value": 0.107067, "what": "the Station", "source": "https://arxiv.org/abs/2608.23691 (via "
             "https://eventcompute.com/en/challenges/kakeya/ and https://discoveria.ai/frontiers/einsteinarena-kakeya-needle-128)",
             "accessed_on": ACCESSED, "definition_matches_verifier": None,
             "label": "read in secondary sources; definition not checked by us; worse than the live #1"},
            {"value": 0.114810, "what": "AlphaEvolve", "source": "https://arxiv.org/abs/2511.02864",
             "accessed_on": ACCESSED, "definition_matches_verifier": None, "label": "read in a secondary source"},
        ],
        "method_evidence": ["Forum t265 (CrossingPaths, untrusted): group moves + relaxation reached 0.1067978; "
                            "reply r1102: a float event scorer matched the rational verifier to 2.8e-17, and the "
                            "local event cell of that solution is closed (positive-definite Hessian)."],
        "unverified": ["No experiment by us beyond stage 3.", "Literature definitions not checked."],
        "evidence_urls": [EA + "kakeya-needle-128", EA + "kakeya-needle-128/threads/265",
                          "https://eventcompute.com/en/challenges/kakeya/",
                          "https://discoveria.ai/frontiers/einsteinarena-kakeya-needle-128"],
    },
    "second-autocorrelation-inequality": {
        "rank": 3, "decision": "keep_as_backup",
        "rationale": (
            "Current main problem. The #1 moved four times on 2026-11-14 (one team, +7.3 delta over the best older "
            "solution), so headroom exists, but our best verified gain is 0.5 % of delta and five runs stop there "
            "(knockout/README.md). Keep the two running attacks; do not add effort unless one of them finds the "
            "basin jump."),
        "evidence_urls": [EA + "second-autocorrelation-inequality"],
    },
    "two-deletion-code-16": {
        "rank": 4, "decision": "not_now",
        "rationale": (
            "Integer record that rose three times in 28 hours on 2026-11-11 and 11-12 (224, 225, 226, by three "
            "agents) and is 11 above the best reported value (215, Weindel and Heckel v2, as quoted by the problem "
            "page), with an LP upper "
            "bound of 487. But our probe shows the live #1 code is rigid to simple moves: no free word, and one "
            "single word in the whole graph conflicts with exactly one codeword, so add / (1,2)-swap / plateau "
            "local search gained 0 in 60 s. It needs a real maximum-independent-set solver on a 65,536-vertex "
            "graph, and the platform marks the problem 'under active review'."),
        "literature": [{"value": 215, "what": "Weindel and Heckel v2 (KaMIS), reported without a public construction",
                        "source": EA + "two-deletion-code-16 (problem statement)", "accessed_on": ACCESSED,
                        "definition_matches_verifier": True, "label": "read in the problem statement, not in the paper"}],
        "experiment": {k: TWO.get("live1_local_60s", {}).get(k) for k in
                       ("start_score_pinned", "score", "gain", "rounds", "at_start", "wall_s", "label")},
        "evidence_urls": [EA + "two-deletion-code-16"],
    },
    "erdos-min-overlap": {
        "rank": 5, "decision": "not_now",
        "rationale": (
            "Active (four new top solutions in 30 days, +23 delta in 60 days), but the last two records were steps "
            "of 1.8 and 1.4 delta that took a multi-stage pipeline (coarse-to-fine, log-sum-exp continuation, "
            "sequential minimax LP; forum t281, claim), and trust-region LP steps from the then-leader gained 0.04 "
            "delta (forum r1152, claim). MILO's value (0.3808567744) is already beaten by the live #1, so there is "
            "no literature headroom, and a lower bound of 0.380827 claimed in the title of forum t243 would cap "
            "the remaining room at about 290 delta. Stage 3: 2e-5 delta in 10 minutes."),
        "literature": [
            {"value": 0.3808567743668193, "what": "MILO, 512 grid points, 2026-08-31",
             "source": "https://jprithwish.github.io/MILO/", "accessed_on": ACCESSED,
             "definition_matches_verifier": True, "label": "claim; worse than the live #1, so EA is best known"}],
        "evidence_urls": [EA + "erdos-min-overlap", EA + "erdos-min-overlap/threads/281",
                          EA + "erdos-min-overlap/threads/279", "https://jprithwish.github.io/MILO/"],
    },
    "first-autocorrelation-inequality": {
        "rank": 6, "decision": "not_now",
        "rationale": (
            "Three new top solutions in 60 days (+16 delta), but the gap #1 to #2 is 1.3 delta and MILO's value "
            "(1.5027435984) is already beaten by the live #1: no literature headroom. Stage 3: 5.8e-3 delta in 10 "
            "minutes. Threads not digested."),
        "literature": [
            {"value": 1.5027435984, "what": "MILO, 65,536 grid points", "source": "https://arxiv.org/abs/2609.38349",
             "accessed_on": ACCESSED, "definition_matches_verifier": True,
             "label": "claim; worse than the live #1, so EA is best known"}],
        "evidence_urls": [EA + "first-autocorrelation-inequality", "https://arxiv.org/abs/2609.38349"],
    },
    "flat-polynomials": {
        "rank": 7, "decision": "not_now",
        "rationale": (
            "One jump of 0.0213 (21,270 delta) nine days ago by a single agent, after a leader that forum threads "
            "describe as isolated (titles of t225 and t242: exhaustive Hamming-4 ball; bodies not read). That shows "
            "the old record was far from optimal, not that the new one is; we found no description of the method, "
            "and stage 3 found no gain with 1 to 3 sign flips."),
        "evidence_urls": [EA + "flat-polynomials", EA + "flat-polynomials/threads/242"],
    },
    "snake-in-the-box-13": {
        "rank": 8, "decision": "not_now",
        "rationale": "Integer record that moved by +2 on 2026-11-14 (2934 to 2950 in two months, four agents). "
                     "Live facts only: verifier and literature not examined; it needs a specialized search.",
        "evidence_urls": [EA + "snake-in-the-box-13"],
    },
    "kissing-number-d11-605": {
        "rank": 9, "decision": "not_now",
        "rationale": "delta = 0 and a continuous penalty: eight agents sit within 7e-8 of each other and the #1 "
                     "changed two days ago, so any strict polish tops the board. But the score is the penalty of an "
                     "infeasible configuration: topping it is not a mathematical record unless it reaches 0 "
                     "(stage 1: moonshot_only). Live facts only.",
        "evidence_urls": [EA + "kissing-number-d11-605"],
    },
    "kissing-number-d12-842": {
        "rank": 9, "decision": "not_now",
        "rationale": "Same situation as kissing-number-d11-605: delta = 0, the #1 beat #2 by 7.6e-12 two days ago. "
                     "A leaderboard polish, not a mathematical record. Live facts only.",
        "evidence_urls": [EA + "kissing-number-d12-842"],
    },
    "circles-rectangle": {
        "rank": 10, "decision": "not_now",
        "rationale": "Current backup. No new top solution in 77 days; the #1 is a certified non-degenerate KKT "
                     "point and published escapes failed (selection/digests/circles-rectangle.md).",
        "evidence_urls": [EA + "circles-rectangle"],
    },
    "ring-loading-15": {
        "rank": 10, "decision": "not_now",
        "rationale": "Record bet. 9/8 is a certified strict local maximum; nothing new in 64 days "
                     "(selection/digests/ring-loading-15.md).",
        "evidence_urls": [EA + "ring-loading-15"],
    },
}


def complete_run_logs(live: dict) -> None:
    """Add the fields of the shared experiment-log format that the run scripts do not know (best_known_score,
    decision, rationale, evidence_urls) to both run logs. Idempotent."""
    for name, slug in (("scout_third_autocorr_runs.jsonl", "third-autocorrelation-inequality"),
                       ("scout_two_deletion_runs.jsonl", "two-deletion-code-16")):
        path = ROOT / name
        if not path.exists():
            continue
        top1, delta = live[slug]["top1"]["score"], live[slug]["delta"]
        sign = -1.0 if live[slug]["direction"] == "minimize" else 1.0
        rows = []
        for line in path.read_text().splitlines():
            r = json.loads(line)
            margin = sign * (r["score"] - top1) if r.get("valid") and r.get("score") is not None else None
            r["best_known_score"] = top1
            r["beats_live_top1_by"] = margin
            r["decision"] = "candidate" if margin is not None and margin >= delta and margin > 0 else "no_gain"
            r["rationale"] = (
                f"scores {margin / delta:.2f} delta better than the live #1 under the pinned verifier; a candidate "
                "for WS1 arbitration, not a record claim" if r["decision"] == "candidate"
                else "does not beat the live #1 by delta")
            r["evidence_urls"] = [EA + slug] + ([MILO["source_url"], "https://arxiv.org/abs/2609.38349"]
                                                if "milo" in str(r.get("start", "")) else [])
            rows.append(r)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def generic(row: dict) -> dict:
    ties, age = row.get("within_delta_of_1"), row.get("days_since_top1")
    if ties:
        why = f"Plateau: {ties} other entries within delta of the #1, which is {age} days old."
    else:
        why = (f"The #1 is {age} days old; {row.get('top_created_last_30d')} of the top 8 solutions were created "
               f"in the last 30 days, none of them the #1.")
    return {"rank": None, "decision": "not_now", "rationale": why + " Live facts only; not examined further.",
            "evidence_urls": [EA + row["slug"]]}


def main() -> None:
    live = [json.loads(l) for l in (ROOT / "scout_live.jsonl").read_text().splitlines()]
    complete_run_logs({r["slug"]: r for r in live})
    rows = []
    for row in live:
        cur = CURATED.get(row["slug"]) or generic(row)
        out = {
            "problem_id": row["problem_id"], "rank": cur["rank"], "decision": cur["decision"],
            "rationale": cur["rationale"], "evidence_urls": cur["evidence_urls"], "checked_on": row["checked_on"],
            "direction": row.get("direction"), "delta": row.get("delta"),
            "live_top1": {**row.get("top1", {}), "created_at": row.get("top1_created_at"),
                          "source": row["source"], "accessed_on": row["checked_on"],
                          "label": "displayed by the live API"},
            "live_top2": row.get("top2"), "gap_1_2_over_delta": row.get("gap_1_2_over_delta"),
            "within_delta_of_1": row.get("within_delta_of_1"), "days_since_top1": row.get("days_since_top1"),
            "top_created_last_30d": row.get("top_created_last_30d"),
            "gain_over_best_older_than_60d_over_delta": row.get("gain_over_best_older_than_60d_over_delta"),
            "literature": cur.get("literature", []),
        }
        for key in ("method_evidence", "unverified", "to_freeze_if_switch", "experiment"):
            if key in cur:
                out[key] = cur[key]
        if row["slug"] == "third-autocorrelation-inequality":
            out["live_top1"]["label"] = ("verified: pinned verifier reproduces 1.449109789331911 in a clean process; "
                                         "exact rational score 1.4491097893319105")
            cands = third_candidates()
            out["candidates"] = cands
            if cands:
                best = cands[0]["score_pinned"]
                out["best_candidate_vs_live_top1_over_delta"] = (row["top1"]["score"] - best) / row["delta"]
                out["best_candidate_vs_milo_claim_over_delta"] = (MILO["claimed_record"] - best) / row["delta"]
            out["experiment"] = {
                "from_live_top1_hops_300s_4_workers": {k: THIRD.get("live1_hops_300s", {}).get(k)
                                                       for k in ("score", "gain", "gain_over_delta", "wall_s")},
                "note": "see docs/problem_scout.md for the scan measurements that were printed, not saved",
            }
        rows.append(out)
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0, r["problem_id"]))
    with OUT.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in rows:
        print(f"{str(r['rank']):5s} {r['decision']:16s} {r['problem_id']}")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
