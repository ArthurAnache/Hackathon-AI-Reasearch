# WS1 decision: which problem the team attacks

Decided by Arthur (WS1), 2026-11-14. Written by WS1. Data as of 2026-11-14. The live leaderboards were re-read at
22:15 UTC and were unchanged since stage 2.

| Role | Problem | Record to beat (live #1) | δ | Target |
|---|---|---|---|---|
| **Main** | `second-autocorrelation-inequality` (maximize) | 0.963638667992813, MLAI_YONSEI, solution 2782, 2026-11-14 15:35 UTC | 1e-5 | ≥ 0.963648667992813 |
| **Backup** | `circles-rectangle` (maximize) | 2.3658323759098576, ExoMind-TTS, solution 2544, 2026-08-30 | 1e-10 | ≥ 2.3658323760098576 |
| **Record bet** (left to the agent) | `ring-loading-15` (maximize) | 9/8 exactly, JSAgent-Ring15, solution 2576, 2026-09-12 | 0 (strict) | > 9/8 |

All three records were reproduced locally with the pinned verifier (stage 2, *verified*). The frozen versions are in
`problems/<slug>/`. WS1 does not optimize: improving these records is the job of the agent (WS2) and the optimizers
(WS3). WS1 arbitrates claims with `selection/verify_claim.py`.

## 1. The rule (written before the results)

CLAUDE.md fixed the method and the weights before stage 3 ran:

| Criterion | Weight |
|---|---|
| Empirical margin (stage 3) | 30 % |
| Evaluation cost | 20 % |
| Verifier quality | 15 % |
| Baseline | 15 % |
| Code | 10 % |
| Recent activity | 10 % |

The stage 3 note (CLAUDE.md, 2026-11-14 12:13 PT) added this: generic local search reached no δ anywhere, so the
weights must lean on the gap to δ, the literature status and the evaluation cost, not on near-zero gains.

**Transparency.** The weights were fixed in advance. The 0–5 rating of each criterion, defined below, was **not**
written before the results: it is defined here and applied mechanically to the recorded data. Each rating cites its
source.

| Criterion | 0–5 rating rule | Source |
|---|---|---|
| Empirical margin | `clamp(5 + log10(gain/δ), 0, 5)`, with the gain at a submittable resolution; 0 if there is no gain. So gain = δ gives 5, and gain = δ/1000 gives 2. | `registry_stage3.jsonl`, phase `original_resolution` |
| Evaluation cost | < 1 ms → 5; < 0.2 s → 4; < 1 s → 3; < 5 s → 2; otherwise 1 | `registry_stage2.jsonl` `eval_s` |
| Verifier quality | 5: exact arithmetic on the server. 4: float64 on the server, but an exact independent check is feasible and no exploit is open. 3 or less: known open issues. | verifier source; digests |
| Baseline | 5: record reproduced and confirmed as best known in the literature. 4: reproduced, literature not yet curated. 3: reproduced but a plateau (several agents within δ). 2: the literature claims a better value we cannot reproduce. | stage 2; `literature_check.jsonl` |
| Code | 5: open-source code for the record family. 3: methods described in the threads, no code (or threads not digested yet). 1: nothing. | `selection/digests/` |
| Recent activity | #1 changed ≤ 7 days ago → 5; ≤ 30 d → 4; ≤ 90 d → 3; older → 2 | stage 2 `last_record_date` |

## 2. Data and scores

Candidates are the live problems that made **any** progress in stage 3, plus ring-loading-15. These problems made
none: circle-packing, flat-polynomials, kakeya-needle-128, tammes, thomson, heilbronn-triangles and
difference-bases (0 improvements in 10 min). Most of them are also plateaus. All score 0 on the 30 % criterion, so
none can win.

| Problem | gain/δ (stage 3) | Improvements / min | eval | Last #1 | Margin | Cost | Verifier | Baseline | Code | Activity | **Weighted** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| circles-rectangle | 4.3e-3 | 13.7 | 0.26 ms | 2026-08-30 | 2.63 | 5 | 4 | 4 | 3 | 3 | **3.59** |
| second-autocorrelation | 9.0e-4 | 67.6 | 0.15 s | 2026-11-14 | 1.95 | 4 | 4 | 4 | 5 | 5 | **3.59** |
| first-autocorrelation | 4.9e-3 | 8.0 | 0.78 s | 2026-10-13 | 2.69 | 3 | 4 | 4 | 3 | 3 | 3.21 |
| min-distance-ratio-2d | 2.2e-4 | 12.5 | 0.02 ms | 2026-03-19 | 1.35 | 5 | 4 | 3 | 3 | 2 | 2.95 |
| erdos-min-overlap | 1.8e-5 | 6.0 | 0.79 s | 2026-10-29 | 0.26 | 3 | 4 | 4 | 3 | 4 | 2.58 |
| ring-loading-15 | 0 (δ = 0) | 0 | 1.21 s | 2026-09-12 | 0 | 2 | 5 | 5 | 3 | 3 | 2.50 |
| third-autocorrelation | 1.8e-5 | 8.5 | 0.80 s | 2026-10-13 | 0.26 | 3 | 4 | 2 | 3 | 3 | 2.18 |

Notes on the data:

- **second-autocorrelation:**
  - The 9.0e-4 is the gain at the original resolution (2M values). The larger 1.1e-3 in the registry comes from
    ×2 upsampling, which gives 4M values and cannot be submitted (the schema allows at most 2M).
  - The "Code" 5 refers to the open-source Dinkelbach optimizer of the record family
    (github.com/justinkang221/second-autocorrelation-inequality) [t133/r415, ClaudeExplorer].
  - The "Activity" 5 is the #1 that changed today.
- **circles-rectangle:**
  - The gain/δ of 4.3e-3 is an absolute gain of 4.3e-13, at the level of float noise.
  - The rating 4 for verifier quality holds because the pinned verifier closes the reported tolerance exploits.
    `problems/circles-rectangle/` has tests for this.
- **third-autocorrelation:** baseline 2 because MILO (arXiv:2609.38349) claims 1.44889 under the same definition,
  and we cannot reproduce it (README1 §7.2).
- **ring-loading-15:** the verifier is exact and the record (9/8) already beats the literature (47/42). But δ = 0
  means only a strict exact improvement counts, and stage 3 found none in 331 evaluations.

**Result of the rule:** circles-rectangle and second-autocorrelation are **tied** (3.589 vs 3.585, a difference far
below the resolution of the ratings). first-autocorrelation is third (3.21). Every other problem is at 2.95 or below.

## 3. Tie-break and decision

The weighted rule cannot separate the top two. The tie is broken by two pieces of evidence that the rule's own
criteria point to (margin and activity), read beyond the 0–5 rounding:

1. **Sustained progress at a submittable resolution (stage 3):**
   - second-autocorrelation improved 681 times in 10 min (67.6/min) and was still improving at the end of the run.
   - circles-rectangle improved 137 times. Most of these came in a final polish burst at t ≈ 420 s, after long
     stalls, for an absolute gain of 4.3e-13.
2. **The forum record (digests):**
   - circles-rectangle:
     - The incumbent is a certified non-degenerate KKT point: 65 active constraints on 65 variables, full-rank
       Jacobian, all multipliers > 0 [t141/r899, CHRONOS; t141/r1067, NumaroTech].
     - Every published local, single-contact and contact-forcing escape failed [t141/r478, EinsteinAgent3384;
       t141/r931, r934, r940, Asper].
     - The best changed-topology result is 2.3658322057, below #1 [t141/r931, Asper].
   - second-autocorrelation:
     - The record has moved repeatedly through new constructions at higher effective resolution:
       0.96199 → 0.96272 [t133, ClaudeExplorer], then 0.96290 [t254, ClaudeExplorer on Hyra], then 0.963433
       [t255, NelsonFrontier], then today 0.963639.
     - A concrete, untested route is documented: a full-support n = 2M run with Dinkelbach [t255, NelsonFrontier].
     - The full pipeline is open-source [t133/r415, ClaudeExplorer].

**Decision.**

- **Main:** second-autocorrelation-inequality. Target ≥ live #1 + 1e-5 (today ≥ 0.963648668).
- **Backup:** circles-rectangle. It is the cheapest evaluation of all (0.26 ms), its verifier is exact on the
  perimeter and strict on overlaps, and δ is 1e-10. #1 − #2 is 1.08e-9, so a strictly feasible gain of 1e-10 is in
  principle possible, but the digest says it needs a coordinated change of topology.
- **Record bet:** ring-loading-15. It scores low on the rule (no margin, slow evaluation), but it is the only problem
  where any strict improvement is a new lower bound on a classical constant, beyond the published 47/42.
  - The digest shows that local search around 9/8 is provably dead: a certified local maximum in a box of radius
    1/160 [t261/r1147, CodexRingLab-20261019], and no gain from inserting 3 pairs around the fixed 12-pair core
    [t274, CrossingPaths].
  - Only a structural change can win, so this is left to the agent as a bet. WS1 spends no optimization effort on
    it.
- **Not chosen:** first-autocorrelation (3rd, 3.21). Its threads are not digested yet, and its verifier is float
  with δ = 1e-8. It stays the next candidate if the backup is abandoned.

## 4. What this decision commits WS1 to

- Freeze the three problems in `problems/<slug>/`:
  - the pinned verifier, with its commit and sha256;
  - the #1 certificate, with its sha256;
  - an independent verifier;
  - a test that fails if the verifier hash changes.
- Arbitrate every claim with `selection/verify_claim.py`. It runs the pinned and the independent verifier and
  re-reads the live #1. It never submits.
- Hand over the knowledge from the digests (`docs/ws1_handoff.md`) as data for the agent, not as WS1 tasks.
