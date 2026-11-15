# Problem scout: is another record more reachable?

WS1, 2026-11-14 evening PT (live data read 2026-11-15 04:36 UTC; the third-autocorrelation leaderboard re-read at
06:20 UTC). Written for the team; Arthur decides. Section 3.4 is a follow-up run the same evening.
Labels: **verified** = reproduced here with the pinned verifier; **claim** = read, not reproduced; **limitation**.
Nothing was submitted, and nothing here is a record claim.

## 1. Answer

**Yes: `third-autocorrelation-inequality` (minimize, δ = 1e-5).**

We hold two kinds of candidates, kept strictly apart (sections 3.3 to 3.5). Every one is scored by the pinned
verifier in a clean process and by an independent exact-rational scorer, which agree within 2e-15.

- **Track A, from MILO's published copy:** best 1.4488128798873552, **29.7 δ below the live #1** and **7.3 δ
  below the best published claim** (MILO, 1.4488860143). These are re-optimized from a plot-resolution copy of
  MILO's record that its authors publish on their project page. Any use must credit MILO, and whether that
  counts is Arthur's call.
- **Track B, from the live #1 only, no MILO data:** best 1.4490529129656413, **5.7 δ below the live #1**. It
  would top the leaderboard, but it stays about 17 δ above MILO's claimed value.
- Polishing the live #1 on its own grid gains nothing: 0 δ in 5 minutes.
- A first candidate costs 5 to 15 minutes of one CPU core.

Recommendation: move the main effort to this problem, keep the two running attacks on the current problem as they
are, and have WS1 freeze and arbitrate before anything else (section 5).

## 2. Ranking

Live facts for all 28 live problems are in `selection/scout_live.jsonl`; decisions and evidence, one line per
problem, in `selection/problem_scout.jsonl`.

| Rank | Problem | δ | Live #1 (agent, date of the solution) | Sign of headroom | What we measured | Decision |
|---|---|---|---|---|---|---|
| 1 | third-autocorrelation | 1e-5 | 1.449109789331911 (Atris, 2026-10-13), **verified** | MILO claims 1.4488860143 with the same verifier, 22 δ lower (**claim**) | candidates 29.7 δ below the live #1 from MILO's published copy, and 5 δ below it from public data only (**verified**, section 3) | `switch_candidate` |
| 2 | kakeya-needle-128 | 1e-9 | 0.10675138995168142 (kgrey, 2026-11-10) | record moved 2,550 δ four days ago, 46,400 δ in 60 days, five successive holders | nothing beyond stage 3 (no gain) | `keep_as_backup` |
| 3 | second-autocorrelation (current main) | 1e-5 | 0.9636615620620852 (MLAI_YONSEI, 2026-11-14) | #1 moved four times on 2026-11-14, +7.3 δ | our best is 0.5 % of δ (knockout v1) | `keep_as_backup` |
| 4 | two-deletion-code-16 | 1 | 226 (kgrey, 2026-11-12) | +1 three times in 28 hours, 11 above the best reported value | live #1 is rigid to simple moves: 0 gain in 60 s (**verified**) | `not_now` |
| 5 | erdos-min-overlap | 1e-7 | 0.38085625977797644 (Blob, 2026-10-29) | +23 δ in 60 days, but last steps of 1.8 and 1.4 δ | stage 3: 2e-5 δ | `not_now` |
| 6 | first-autocorrelation | 1e-8 | 1.502743486709583 (Atris, 2026-10-13) | +16 δ in 60 days | stage 3: 5.8e-3 δ | `not_now` |
| 7 | flat-polynomials | 1e-6 | 1.2594571732205586 (apollo-preview, 2026-11-05) | one jump of 21,270 δ nine days ago | stage 3: no gain | `not_now` |
| 8 | snake-in-the-box-13 | 1 | 2950 (kgrey, 2026-11-14) | +2 on 2026-11-14 | not examined | `not_now` |
| 9 | kissing-number-d11-605, -d12-842 | 0 | 1.71023812080097, 0.5309634950279605 (kgrey, 2026-11-12) | δ = 0, #1 changed two days ago | not examined; a leaderboard polish, not a mathematical record | `not_now` |
| 10 | circles-rectangle, ring-loading-15 | 1e-10, 0 | unchanged since stage 2 | none: 77 and 64 days without a new top solution | certified local optima (digests) | `not_now` |

The other 16 live problems are plateaus of ties or have had no new top solution for months (`not_now`).

Unless marked verified, live scores are the values displayed by the API.

## 3. The lead: third-autocorrelation

### 3.1 Evidence

| Fact | Value | Source, access date | Label |
|---|---|---|---|
| Live #1 | 1.449109789331911, Atris, solution 2617, 65,536 values, 2026-10-13 | https://einsteinarena.com/problems/third-autocorrelation-inequality, 2026-11-15 | **verified**: pinned verifier in a clean process gives 1.449109789331911; exact rational score 1.4491097893319105 |
| Live #2 | 1.449121951852342, Ghazt, 2026-10-06 | same | displayed |
| Verifier | live source sha256 `b8288d59…` equals the pinned one (stage 2) | `selection/registry_stage2.jsonl` | **verified** |
| MILO record | 1.4488860142647635 on 25,600 grid points, 2026-09-15 | https://jprithwish.github.io/MILO/ (demo data); arXiv:2609.38349 App. E.1 Table 13 prints 1.4488860143; 2026-11-15 | **claim**: construction not public, not on the leaderboard |
| MILO definition | "minimize \|max(f⋆f)\| / (∫f)², f signed", "rescored with the arena's own verifier" | arXiv:2609.38349 App. E.1 | matches the pinned verifier (abs outside the max). Tao's 1d is the other constant and stays not comparable |
| MILO method | log-sum-exp objective with FFT gradient, L-BFGS stages on a geometric cooling ladder, chain of re-melt hops with multiplicative jitter, 60 s runs, started from arena rank 3 (BasinHopper, 1.4515829) | arXiv:2609.38349 App. E.2, third listing | read in source; the 574-line program is only excerpted |
| MILO cost | 12.7 days to the record, 11.8 days to pass below today's live #1, 41 harness evaluations with Opus 5, 403 sub-record constructions | project page, demo data | **claim** |
| What MILO publishes | a plot-resolution copy of the record: 3,200 of the 25,600 values (one in eight), 6 decimals | `static/einstein-run.js` on the project page, sha256 in `selection/scout_milo_replay.json` | **verified** that, scored as given, the copy gives 1.6558: it is not the construction |
| Other literature | nothing else below 1.4491 found (Exa, 3 queries) | DiscoverIA frontier page lists 1.449122 as the record | **limitation**: a short search |

Forum: 20 threads, the latest from 2026-07-04; none describes how Ghazt and Atris reached 1.4491 (untrusted data).

### 3.2 Experiments

Script: `selection/scout_third_autocorr.py` (our re-implementation of the method MILO describes: log-sum-exp
surrogate, FFT gradient, L-BFGS on a cooling ladder). Local CPU, one core per run unless stated; the machine was
shared with other jobs. Final scores come from the pinned verifier in a clean process (`selection/eval_runner.py`);
"exact" is `selection/scout_third_autocorr_exact.py` (exact rationals, no numpy). Rows marked *printed* come from
`scan`, which prints the FFT score (within 6e-15 of the pinned verifier on the live #1) and saves nothing.

| # | Start | Method, budget | Result | Gain |
|---|---|---|---|---|
| A1 | live #1 (65,536) | cold polish, T 1e-6 → 1e-9, 52 s | ends at 1.449109797, never below the start (*printed*) | **0 δ** |
| A2 | live #1 block-repeated ×2 (131,072) + noise | cooling 1e-6 → 1e-9, 75 s | 1.4491115017, 1.7e-6 above the live #1 (*printed*). Not submittable anyway: the schema allows at most 100,000 values (section 3.4) | **0 δ** |
| A3 | live #1 (65,536) | re-melt hops, 4 workers × 300 s, 12 hops | none landed; pinned score unchanged (**verified**) | **0 δ** |
| B1 | BasinHopper #2493 (25,600; 1.4515718639), the family MILO started from | one cooling path 1e-4 → 1e-7, 40 s | 1.4515679350 (*printed*) | 0.39 δ over its own start; still 246 δ above the live #1 |
| C0 | decimated copy of MILO's *starting point* (BasinHopper), block-repeated ×8 — control | one cooling path 1e-3 → 1e-7, 45 s | 1.4517381 (*printed*) | returns to BasinHopper's level: the method alone does not find MILO's basin |
| C1 | decimated copy of MILO's *record*, block-repeated ×8 (25,600; scores 1.6558 as given) | cooling 1e-3 → 1e-8, 300 s | **1.4488729104664002** (**verified**; exact 1.4488729104664004) | 23.7 δ below the live #1; 1.3 δ below MILO's claim |
| C2 | same | cooling 3e-3 → 1e-8, 300 s | **1.4488567088845645** (**verified**; exact 1.4488567088845636) | 25.3 δ below the live #1; 2.9 δ below MILO's claim |
| C3 | same | cooling 1e-2 → 1e-8, 420 s | **1.448827989657653** (**verified**; exact 1.448827989657653) | 28.2 δ below the live #1; 5.8 δ below MILO's claim |
| C4 | C2 block-repeated ×2 (51,200) + noise 1e-4 | cooling 1e-4 → 1e-9, 420 s | **1.448853861372726** (**verified**; exact 1.4488538613727269) | 0.28 δ better than C2: the finer grid adds little |
| C5 | same as C1 | cooling 3e-2 → 1e-8, 420 s | **1.448845291011433** (**verified**; exact 1.4488452910114331) | 26.4 δ below the live #1; 4.1 δ below MILO's claim |
| C6 | same as C1 | cooling 1e-1 → 1e-8, 420 s | **1.448835478669337** (**verified**; exact 1.448835478669337) | 27.4 δ below the live #1; 5.1 δ below MILO's claim |
| C7 | C3 + noise 1e-3 (one re-melt hop) | cooling 1e-3 → 1e-8, 420 s | **1.448825267560211** (**verified**; exact 1.4488252675602122) | **28.5 δ below the live #1; 6.1 δ below MILO's claim**; 0.27 δ better than C3 |

Reading:

- **The method is not what was missing; the basin is.** From the live #1 (A1 to A3) and from BasinHopper (B1, C0)
  the same code gains between 0 and 0.4 δ. From a coarse, rounded copy of MILO's record (C1 to C7) it lands
  23.7 to 28.5 δ below the live #1 within minutes. On a 400-cell block average (`compare` mode), candidate C2
  correlates at 0.9997 with the published copy, at 0.72 with the live #1 and at 0.76 with BasinHopper's solution:
  the candidates sit in MILO's basin, the live #1 does not.
- **Every candidate ends below MILO's claimed value**, by 1.3 to 6.1 δ. A plausible reason is that MILO's
  optimizer ran 60 s per call and ours cools for 5 to 7 minutes (hypothesis). MILO's exact number stays a claim.
- **Where the remaining room is.** The five direct paths (C1, C2, C3, C5, C6) differ only by their starting
  temperature and spread over 4.5 δ, without a monotonic trend. One re-melt hop added 0.27 δ (C7) and one ×2
  resolution step added 0.28 δ (C4). So more paths and hops should still gain a few δ; a finer grid gains little
  (hypothesis, from one run each).

Compute spent on this problem: about 75 CPU-minutes over about 35 minutes of wall clock, the longest single run
7 minutes. That is above the 20 minutes planned for the experiment: the first 45 s run from MILO's copy landed
below the live #1, and the rest went into verifying and repeating it.

### 3.3 The candidates

Stored in `selection/scout_candidates/third-autocorrelation-inequality/` with `MANIFEST.json` (sha256, both scores,
margins, run settings):

| File (`.json.gz`) | Run | Values | Pinned verifier | Exact rational | Below live #1 | Below MILO's claim |
|---|---|---|---|---|---|---|
| `milo_replay_25600_c_remelt` | C7 | 25,600 | 1.448825267560211 | 1.4488252675602122 | 2.845e-4 = 28.5 δ | 6.07e-5 = 6.1 δ |
| `milo_replay_25600_c` | C3 | 25,600 | 1.448827989657653 | 1.448827989657653 | 2.818e-4 = 28.2 δ | 5.80e-5 = 5.8 δ |
| `milo_replay_25600_e` | C6 | 25,600 | 1.448835478669337 | 1.448835478669337 | 2.743e-4 = 27.4 δ | 5.05e-5 = 5.1 δ |
| `milo_published_copy_3200` | start | 3,200 | 1.6557955266483575 as given | — | — | — |

The four other candidates (C1, C2, C4, C5) are only in the git-ignored cache of the worktree that ran them; their
scores and sha256 are in `selection/scout_third_autocorr_runs.jsonl`. The live #1 was re-read at 05:15 UTC on
2026-11-15 and was unchanged.

What they are:

- **Verified:** valid inputs of the pinned verifier (sha256 `b8288d59…`, identical to the live one), scored in a
  clean process, and re-scored in exact rational arithmetic by a second implementation. The two agree within
  1.3e-15. The stored files, decompressed, give the same scores with the stage 2 copy of the pinned verifier.
- **Candidates, not records.** They were not submitted, not scored by the server, and not arbitrated by
  `selection/verify_claim.py`, because the problem is not in the frozen catalog.
- **Derived from MILO's published data.** The starting point is the 3,200-value copy on MILO's project page. The
  EinsteinArena page of this problem asks to check that a result is the agent's own "and not just a download from
  the literature". This is not a download (the published copy scores 1.6558, and MILO's construction is not
  public), but it is not independent either. The forum has a precedent on erdos-min-overlap, where a #1 was
  presented as "a balanced repair of MILO's rounded public replay" with credit [t279, CristianBudala] (untrusted
  data).

### 3.4 Follow-up: two tracks at scale (2026-11-14, 21:40 to 22:25 PT)

The coordinator re-scored the best candidate of section 3.3 independently (1.448825267560211) and asked for two
things: widen the margin, and test whether anything is reachable without MILO's data. Arthur has not ruled on the
provenance question, so nothing was frozen or submitted.

The two tracks never share a start vector, a Modal call, a Volume folder, a log or a candidate name. Each job
records the sha256 of the vector it started from.

| Track | Provenance | Starts | Log |
|---|---|---|---|
| A | `milo_published_copy` | MILO's published copy (block-repeated ×8) and the candidates of section 3.3 | `selection/scout_third_autocorr_trackA_runs.jsonl` |
| B | `public_leaderboard_only` | the live #1 (solution 2617, Atris) and #2 (solution 2609, Ghazt), nothing else | `selection/scout_third_autocorr_trackB_runs.jsonl` |

**Setup**

- Runner: `selection/scout_third_autocorr_modal.py` (Modal app and Volume `scout-third-autocorr`), and
  `selection/scout_third_autocorr_local.py` for four local jobs. A job is a list of steps (`run_steps` in
  `selection/scout_third_autocorr.py`): sum-pool, block average onto another grid, keep one value in k,
  block-repeat, one cooling path, re-melt hops with a ratchet.
- One 48-core container per track and round, one core per job. The workspace's container limit was taken by the
  team's other campaigns (99 containers running), so one-core containers stayed queued.
- **Schema limit: at most 100,000 values** (`values: z.array(num).min(1).max(100_000)` in the problem file of the
  pinned commit `5baf636`; the live API does not expose it, limitation). From 25,600 values only ×2 and ×3 are
  allowed; from 65,536 nothing. `run_steps` refuses larger grids.
- Every kept point is re-scored by the pinned verifier in a clean process and by the exact-rational scorer. The
  other jobs carry the FFT score computed on Modal.

**Rounds**

| Round | Jobs A + B | Budget per job | What ran |
|---|---|---|---|
| r1 | 48 + 48 | 1200 s (A stopped after 10 min) | A: hop chains with 216 s hops, direct paths, ×2 and ×3. B: hot re-melts at 65,536 values, coarse-to-fine (sum-pool by 2, 4, 8, then repeat ×2 step by step), hop chains, other grids |
| r1b | 48 + 48 | 1080 s (A stopped after 7 min) | A: direct paths. B: block average to 3,200–8,192 values, repeat ×8, hot path, hops |
| r1c | 48 + 48 | 1980 s planned, stopped after 20 min | one long path per job. A: 30 hops from the candidates, 12 direct paths, 6 at ×2 or ×3. B: block average to 1,600–6,400 values, repeat to 25,600 or 32,768 |
| r1d | 48 + 48 | 1860 s planned, stopped after 17 min | one long path per job. A: 48 direct paths. B: sum-pool ×2, keep one value in 4, 8 or 16, repeat back: what MILO's page did to its record, applied to the public solutions |
| local1 | 2 + 2 | 840 s on local cores | A: two hop chains. B: two coarse pipelines |

| r2 | 48 + 48 | 1020 s, after the thread fix | one long path per job. A: 24 hops from the best candidates, 20 direct paths, 4 at ×2 or ×3. B: 24 coarse copies of the live #1 (3,200 values ×8), 12 hops from the best track B point, 12 other coarse variants |
| local2 | 0 + 5 | 780 s on local cores | B: five coarse copies of the live #1, one path each |
| local3 | 0 + 4 | 700 s on local cores | B: hops, one long hop and one ×2 step from the best local1 point |
| local4 | 0 + 5 | 600 s on local cores | B: five long hops from the best local3 point |

**The Modal thread problem (limitation).** r1, r1b, r1c and r1d ran about 250 times too slowly and are not
converged. A Modal container with `cpu=48` exports `OMP_NUM_THREADS=48` and `OPENBLAS_NUM_THREADS=48`; our module
used `setdefault`, so each of the 48 job processes kept 48 threads. Measured in such a container: 541 ms per
objective evaluation with 48 processes, against about 2 ms on a local core. Their results say nothing about the
method; they were stopped as soon as this was found, the runner now forces one thread, and r2 ran with the fix.

**Track B: yes, by a thin margin.** Everything below starts from the live #1 (solution 2617) only.

| Step | Round | Score (pinned verifier unless noted) | Against the live #1 |
|---|---|---|---|
| block average to 3,200 values, repeat ×8, one cooling path from T = 1e-2 (462 s) | local1 | 1.4491102 (FFT scorer) | 0.04 δ above |
| + one hop (378 s) | local1 | 1.449101225026528 (exact 1.449101225026529) | 0.86 δ below |
| + one long hop (679 s, T = 1e-3, noise 5e-4) | local3 | **1.449094982848945** (exact 1.449094982848944) | **1.48 δ below** |
| + two chains of two hops (336 s per hop) | local3 | 1.4490977656912827 and 1.4490985295580978 | 1.20 and 1.13 δ below |
| + repeat ×2 (51,200 values) and one path | local3 | 1.4490989105229044 | 1.09 δ below |
| + five more long hops from the 1.48 δ point (570 s each) | local4 | best 1.4490894751686874 (exact 1.4490894751686876); the four others by the FFT scorer | best 2.03 δ below; the others 1.80 to 1.98 δ below |
| **again from the live #1:** block average to 1,600 values, repeat ×16, one long path (990 s on a Modal core), 3 jobs | r2 | **1.4490567002326498** (exact 1.4490567002326498), 1.4490666598430821, 1.4490749223996897 | **5.31**, 4.31 and 3.49 δ below |
| block average to 3,200 values, repeat ×8, one long path, 24 jobs | r2 | best 1.4490763441446004 and 1.4490776603516209; the others by the FFT scorer | best 3.34 δ below, median 1.09 δ below; 13 of 24 at least δ below |
| block average to 6,400 values, repeat ×4, one long path, 3 jobs | r2 | best 1.4490868 (FFT scorer) | best 2.29 δ below |
| 12 long hops from the 0.86 δ point | r2 | best 1.4490851 (FFT scorer) | 2.15 to 2.47 δ below |
| sum-pool ×2, keep one value in 8, repeat ×8 (32,768 values), from the #1 and the #2, 6 jobs | r2 | FFT scorer | 33 to 35 δ above |
| + from the 5.31 δ point: 24 long hops, 12 ×2 steps (51,200 values) and 12 ×3 steps (76,800 values), 510 s each | r4 | best **1.4490529129656413** at ×2 (exact 1.4490529129656424); three more ×2 results re-scored, the rest by the FFT scorer | **5.69 δ below** at ×2 (median 5.49); ×3 best 5.49; hops best 5.37 |
| five more coarse copies, one path each, no hop (757 s) | local2 | 1.4491154 to 1.4491425 (FFT scorer) | 0.56 to 3.27 δ above |
| sum-pool to 4,096 values, repeat ×8, path, hop | local1 | 1.4491311065380568 | 2.13 δ above |
| 96 jobs before the thread fix: hot re-melts at 65,536 values, coarse-to-fine by pooling, hop chains, other grids, from the #1 and the #2 | r1, r1b | nothing below the live #1; not converged, so inconclusive | — |

- **87 jobs end at least δ below the live #1 without any MILO data (48 of them continue the same 5.31 δ point),
  and the 14 we re-scored hold under the pinned verifier and the exact scorer.** The chain is: solution 2617,
  block average, repeat, our optimizer. Each job's log line holds
  the sha256 of the vector it started from, so the chain can be audited in
  `selection/scout_third_autocorr_trackB_runs.jsonl`.
- **The margin is modest**: 5.7 δ for the best, and 3.5 to 5.3 δ for the three first paths of its kind; hops and
  ×2 steps added 0.4 δ. Other agents moved this record by 1.2 δ on 2026-10-13 and by 169 δ on
  2026-10-06, so the lead is not safe. And track B stays about 17 δ *above* MILO's claimed value: it would top
  the leaderboard, not the literature.
- **The coarser the copy, the better, as far as we looked**: 1,600 values (3.5 to 5.3 δ, 3 jobs) beat 3,200
  (median 1.1 δ, 24 jobs) and 6,400 (best 2.3 δ, 3 jobs). Keeping one value in 8 instead of averaging fails
  (33 δ above). Only three jobs ran at 1,600, so this is a lead, not a result.
- **What track B still owes to MILO, to be exact:** the optimizer follows the method MILO's paper describes, and
  the idea of re-optimizing a coarse copy was first seen to work on MILO's published copy. No number from MILO
  enters track B.
- Why it works (hypothesis): the live #1 has 65,536 values; the candidates have 25,600. The coarse copy keeps the
  large-scale shape and lets the optimizer rebuild the fine structure on another grid, with a path long enough to
  converge. Polishing the #1 on its own grid gains nothing (section 3.2).

**Track A: the margin grows slowly.**

| Source | Jobs | Best score (pinned verifier) | Below the live #1 | Below MILO's claim |
|---|---|---|---|---|
| section 3.2: one path, then one hop | 7 | 1.448825267560211 | 28.45 δ | 6.07 δ |
| local1: two chains of two hops (400 s per hop) | 2 | 1.44882467417151 (exact 1.4488246741715105) | 28.51 δ | 6.13 δ |
| r2 (Modal, after the fix): 20 long direct paths from the copy | 20 | 1.4488194732357842 (exact identical) | 29.03 δ (median 27.85 δ) | 6.65 δ |
| r2: 24 long hops from the best candidates | 24 | 1.4488217929 (FFT scorer) | 28.80 δ (median 28.66 δ) | 6.42 δ |
| r2: ×2 (51,200 values) and ×3 (76,800 values) from the best candidate, 2 jobs each | 4 | **1.4488128798873552** at ×2 (exact 1.448812879887355); 1.4488149360933422 at ×3 | **29.69 δ**; 29.49 δ | **7.31 δ**; 7.11 δ |
| r1, r1b, r1c, r1d (before the thread fix) | 16 logged, the rest stopped | nothing better; not converged | — | — |

How the gain flattens with compute, along the best lineage:

| Compute spent | Best | Step |
|---|---|---|
| one path from the copy, 7 min of a local core | 28.18 δ below the live #1 | — |
| + one hop, 7 min local | 28.45 δ | +0.27 δ |
| + two hops, 13 min local | 28.51 δ | +0.06 δ |
| + one long hop, 12 min of a Modal core (best of 24) | 28.80 δ | +0.29 δ |
| or + one ×2 resolution step, 16 min of a Modal core (best of 2; the other gives 28.92 δ) | 29.69 δ | +1.18 δ |
| for comparison: 20 fresh paths from the copy, 16 min of a Modal core each | best 29.03 δ, median 27.85 δ | — |

The first path does almost all the work: 28 of the 29.7 δ. After it, about 50 jobs and 14 core-hours bought
1.5 δ. Hops add a few hundredths to a few tenths of δ each. The largest single step was one ×2 resolution step,
which the 100,000-value limit caps (76,800 values is the last grid of this family, and it did no better). Fresh
first paths are as good a use of compute as hop chains: the best of 20 beats every hop chain at 25,600 values.

**Suggestions of a second model for track B** (relayed by the coordinator; unverified ideas, no MILO data):

- *Structure (its point 1), measured on the live #1:* with dx·Σf = 1, the reflection-even part has energy 14.86
  and the odd part 13.41, so the central lag equals the maximum (14.86 − 13.41 = 1.449). Positive mass 1.766,
  negative mass −0.766. The autoconvolution goes down to −4.61. 58,972, 60,335 and 62,913 of the 131,071 lags lie
  within 1e-6, 1e-5 and 1e-4 of the maximum. This agrees with its claim that the odd part must carry a lot of
  energy.
- *Movable signed boxes with CMA-ES, coarse-grid basin enumeration, continuation in the central lag (its
  strategies A, B, C):* **not tried.** Each needs a new optimizer and its own stop rule asks for an hour per
  strategy; the follow-up had about 75 minutes in all.
- *Its remark on resolution:* consistent with what we see. Block-repeat preserves the score exactly, and 25,600
  has no special meaning: track B works on 25,600 values simply because 3,200 × 8 is cheap to optimize.

**Compute and spend**

- Modal: 13 runs of one container with 48 cores and 24 GiB, about 185 container-minutes in total, **about
  7.6 USD** at 0.047 USD per core-hour and 0.008 USD per GiB-hour (estimate from the apps' run times; the cap was
  20 USD). About 5.4 USD of it went into the four rounds before the thread fix, which produced nothing usable.
  All apps are stopped.
- Local: 18 jobs, about 3.6 core-hours.

### 3.5 Where the candidates stand

Best candidate of each track, live #1 = 1.449109789331911 (re-read at 06:20 UTC on 2026-11-15, unchanged), MILO's
claimed value = 1.4488860142647635:

| Track | Provenance | File (`.json.gz`) | Values | Pinned verifier | Exact rational | Against the live #1 | Against MILO's claim |
|---|---|---|---|---|---|---|---|
| A | `milo_published_copy` | `trackA_r2_A045_longrepeat2` | 51,200 | 1.4488128798873552 | 1.448812879887355 | 29.69 δ below | 7.31 δ below |
| B | `public_leaderboard_only` | `trackB_r4_B035_longrepeat2` | 51,200 | 1.4490529129656413 | 1.4490529129656424 | 5.69 δ below | 16.69 δ above |

- `MANIFEST.json` lists every stored candidate with its track, provenance, sha256, both scores and their
  difference (at most 2e-15). All stored files were decompressed and re-scored once more with the stage 2 copy of
  the pinned verifier: same scores.
- The coordinator reports that the WS1 arbiter, on the proposal branch `ws1/freeze-third-autocorrelation`, judged
  the first seven track A candidates against the live #1. This scout did not run the arbiter; the later
  candidates, track B included, were committed for it to judge.
- Still candidates: nothing was submitted, and no record is claimed.

## 4. Other problems, briefly

- **kakeya-needle-128** (backup): best sign of headroom among the problems we did not attack. The record fell five
  times since 2026-09-08, the last time by 2,550 δ on 2026-11-10; the leaderboard is ahead of the literature
  (0.107067, the Station, arXiv:2608.23691, read in secondary sources). 128 variables and an exact rational
  verifier. The method that worked is described in the forum [t265, CrossingPaths]: group moves alternated with
  relaxation of all offsets. A reply reports a float event scorer within 2.8e-17 of the rational verifier
  [t265/r1102, cidral]. Limits: statement and verifier are "under active review"; we ran no experiment; it needs a
  fast scorer before any search (the verifier takes 1.07 s).
- **two-deletion-code-16**: the record rose from 223 to 226 in three steps within 28 hours, on 2026-11-11 and
  11-12 (three agents). Probe
  (`selection/scout_two_deletion.py`, 60 s, **verified** with the live verifier): the live #1 code has no free
  word, and one single word in the whole graph conflicts with exactly one codeword, so add / (1,2)-swap / plateau
  moves gain nothing. It needs a real maximum-independent-set solver on 65,536 vertices.
- **erdos-min-overlap**, **first-autocorrelation**: MILO's values (0.3808567744 and 1.5027435984) are already
  beaten by the live #1, so there is no literature headroom. The last leaderboard steps were 1.3 to 1.8 δ each;
  for erdos they came from a multi-stage pipeline [t281, NIGHTFALL-DevinWright] (untrusted data).
- **kissing-number-d11-605 and -d12-842** (δ = 0): any strict polish tops the board (the d12 #1 leads by 7.6e-12),
  but the score is the penalty of an infeasible configuration. Whether that counts for Track 1 is Arthur's call; we
  did not examine the verifiers.

## 5. Recommendation

1. **Switch the main effort to third-autocorrelation.** The switch no longer hangs on the provenance question:
   track B gives candidates below the live #1 with public data only. Arthur still has to choose what the team
   stands on:
   - track A, the stronger number (below the leaderboard *and* below MILO's claim), built on MILO's published copy
     with credit;
   - track B, clean but only a leaderboard lead, 17 δ short of MILO's claimed value. Under the WS1 rule "the record
     to beat is the best known value", track B alone does not beat it if MILO's claim is counted.
2. **WS1 first:**
   - Freeze the problem: pinned verifier (sha256 `b8288d59…`), certificate = live #1 (solution 2617), independent
     verifier = `selection/scout_third_autocorr_exact.py` (3 s for 25,600 values, 11 s for 65,536), schema limit
     of 100,000 values.
   - Fix the record to beat. If it is the lower of the live #1 and MILO's claimed 1.4488860142647635, the target
     is ≤ 1.4488760142647635: the best track A candidate passes it by 6.3 δ, no track B candidate does.
   - Arbitrate the candidates with `verify_claim.py`, re-reading the live #1 just before.
3. **Attack** (WS2 and WS3):
   - Track B first: only three jobs ran from a 1,600-value copy, and they gave the three best results. Run
     hundreds of them (other coarse sizes, starting temperatures, seeds), then hops and one ×2 step from the best.
     One job is 17 minutes of one Modal core, so 100 jobs cost about 1.5 USD.
   - Track A: more first paths from the copy and one ×2 step each; hops bring little (section 3.4).
   - Not tried: a sequential minimax LP polish at the end, as the erdos threads describe [t281] (untrusted data),
     and the three strategies of the second model (section 3.4).
   - Compute: CPU on Modal, with one thread per process forced (`OMP_NUM_THREADS=1`). No GPU needed.
4. **Keep** the two running attacks on second-autocorrelation; add nothing to them.

Odds before Sunday 14:00 PT, as estimates:

| Outcome | Estimate | Why |
|---|---|---|
| A track A candidate stays at least δ below the live #1 and below MILO's claimed value | about 85 % | margins of 29.7 δ and 7.3 δ under two verifiers; the risk is a better result, published or not, by MILO or others before Sunday |
| A track B candidate stays at least δ below the live #1 | about 70 % | margin of 5.7 δ today, and more coarse copies should widen it; but this record moved by 1.2 δ and by 169 δ in October, and anyone can run the same pipeline |
| Track B gets below MILO's claimed value | under 10 % | it is 17 δ short, and the best step we saw after a first path is about 1 δ |
| A record on the current main problem | not estimated here | see `docs/knockout_v2_brief.md` |

## 6. What remains unverified

- MILO's own value: their construction is not public. We reproduced the basin, not their number.
- The server's score of the candidates: nothing was submitted, and the server's numpy version is not pinned. The
  margin is far above float rounding (1e-15).
- WS1 arbitration: this scout did not run `verify_claim.py`. The coordinator reports a verdict on the first seven
  track A candidates only (section 3.5).
- Whether a MILO-derived result (track A) is acceptable for Track 1 and for EinsteinArena.
- The live schema limit: 100,000 values is read in the pinned commit; the live API does not expose it.
- Rows marked *printed* in section 3.2 and rows marked "FFT scorer" in section 3.4 were not re-scored by the
  pinned verifier. On the points we did re-score, the FFT score and the pinned score agree within 1e-14.
- The Modal rounds r1, r1b, r1c and r1d ran with 48 threads per process and are not converged: they say nothing
  for or against the pipelines they ran (hot re-melts at 65,536 values, pooling by 2, 4, 8, hop chains from the
  #1 and the #2).
- Track B was run from the #1 almost only; from the #2 only the pre-fix rounds and three r2 jobs (which failed).
- Why a coarser copy works better, and how far the 1,600-value variant goes: three jobs only.
- The literature search was short (three Exa queries for this problem, one for kakeya); kakeya's literature
  values were read in secondary sources.
- We store the 3,200 published values in the repository for replay
  (`selection/scout_candidates/.../milo_published_copy_3200.json.gz`). The page states no license; remove the file
  if that is a concern.

## 7. Reproduce

```bash
.venv/bin/python selection/scout_refresh.py --top 8          # live facts -> selection/scout_live.jsonl
.venv/bin/python selection/scout_milo_replay.py              # MILO's published copy -> cache + scout_milo_replay.json
C=selection/.cache/scout/third-autocorrelation-inequality
.venv/bin/python selection/scout_third_autocorr.py refine --start $C/milo/milo_record_x8_25600.json \
    --tag milo_replay_25600_c --t-hot 1e-2 --t-cold 1e-8 --stages 16 --budget 420            # C3
.venv/bin/python selection/scout_third_autocorr.py refine --start $C/runs/milo_replay_25600_c/best.json \
    --up-noise 1e-3 --seed 1 --tag milo_replay_25600_c_remelt --t-hot 1e-3 --t-cold 1e-8 --stages 14 --budget 420  # C7
.venv/bin/python -I selection/scout_third_autocorr_exact.py $C/runs/milo_replay_25600_c_remelt/best.json
.venv/bin/python selection/scout_pack_candidates.py milo_replay_25600_c_remelt milo_replay_25600_c
.venv/bin/python selection/scout_third_autocorr.py chain --start $C/record_solution.json \
    --tag live1_hops_300s --budget 300 --workers 4 --seed 100 --alpha-lo 0.03 --t-warm 1e-6 --t-cold 1e-9 \
    --hop-stages 8 --hop-iters 4000 --warm-iters 1500 --fin-iters 5000 --fin-frac 0.2          # A3
.venv/bin/python selection/scout_two_deletion.py --budget 60 --seed 1 --tag live1_local_60s

# Follow-up (section 3.4). Track B, local: coarse copy of the live #1, then hops
.venv/bin/python selection/scout_third_autocorr_local.py --round-name local1 --budget 840
.venv/bin/python selection/scout_third_autocorr_modal.py harvest --round-name local1 --keep 2
.venv/bin/python selection/scout_third_autocorr_local.py --round-name local3 --budget 700 --plan hopsB
# Follow-up on Modal: one track per call; the plans of r1b to r4 were JSON files of the same step format
.venv/bin/modal run --detach selection/scout_third_autocorr_modal.py::launch --round-name r1 --track B --budget 1200
.venv/bin/modal volume get --force scout-third-autocorr /trackB/r1 $C/modal/trackB/
.venv/bin/python selection/scout_third_autocorr_modal.py harvest --round-name r1 --keep 5
.venv/bin/python selection/scout_pack_candidates.py --track B --round r2 B037_coarse1600x16
.venv/bin/python selection/scout_decide.py                   # -> selection/problem_scout.jsonl
```

The step lists of every Modal and local job are in the `config` field of the two track logs, so any job can be
replayed with `run_steps`.

Runs are time-budgeted, so a replay on another machine gives a nearby score, not the same one. The stored
candidates are the evidence.
