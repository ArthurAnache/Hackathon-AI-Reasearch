# Problem scout: is another record more reachable?

WS1, 2026-11-14 evening PT (live data read 2026-11-15 04:36 UTC). Written for the team; Arthur decides.
Labels: **verified** = reproduced here with the pinned verifier; **claim** = read, not reproduced; **limitation**.
Nothing was submitted, and nothing here is a record claim.

## 1. Answer

**Yes: `third-autocorrelation-inequality` (minimize, δ = 1e-5).**

- We hold seven candidates that the pinned verifier scores **23.7 to 28.5 δ below the live #1** and **1.3 to
  6.1 δ below the best published claim** (MILO, 1.4488860143). The best is 1.448825267560211. An independent
  exact-rational scorer agrees within 1.3e-15 on every one (section 3).
- They cost 5 to 7 minutes of one CPU core each.
- **Caveat that Arthur must rule on:** they are not found from scratch. They are re-optimized from a
  plot-resolution copy of MILO's record that the MILO authors publish on their project page. Any use must credit
  MILO.
- Without that starting point the same method gains nothing: 0 δ from the live #1 in 5 minutes.

Recommendation: move the main effort to this problem, keep the two running attacks on the current problem as they
are, and have WS1 freeze and arbitrate before anything else (section 5).

## 2. Ranking

Live facts for all 28 live problems are in `selection/scout_live.jsonl`; decisions and evidence, one line per
problem, in `selection/problem_scout.jsonl`.

| Rank | Problem | δ | Live #1 (agent, date of the solution) | Sign of headroom | What we measured | Decision |
|---|---|---|---|---|---|---|
| 1 | third-autocorrelation | 1e-5 | 1.449109789331911 (Atris, 2026-10-13), **verified** | MILO claims 1.4488860143 with the same verifier, 22 δ lower (**claim**) | candidates up to 28.5 δ below the live #1 (**verified**, section 3) | `switch_candidate` |
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
| A2 | live #1 block-repeated ×2 (131,072) + noise | cooling 1e-6 → 1e-9, 75 s | 1.4491115017, 1.7e-6 above the live #1 (*printed*) | **0 δ** |
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

1. **Switch the main effort to third-autocorrelation**, provided Arthur accepts a result built on MILO's published
   copy with credit. If he does not, do not switch: from the live #1 alone this problem gave 0 δ, no better than the
   current one, and kakeya-needle-128 becomes the alternative to study.
2. **WS1 first** (about an hour):
   - Freeze the problem: pinned verifier (sha256 `b8288d59…`), certificate = live #1 (solution 2617), independent
     verifier = `selection/scout_third_autocorr_exact.py` (3 s for 25,600 values, 11 s for 65,536).
   - Fix the record to beat as the lower of the live #1 and MILO's claimed 1.4488860142647635, so the target is
     ≤ 1.4488760142647635. Candidate C7 passes it by 5.1 δ.
   - Arbitrate the candidates with `verify_claim.py`, re-reading the live #1 just before.
3. **Attack, to widen the margin** (WS2 and WS3; all of it untested beyond section 3.2):
   - Many cooling paths from the published copy: starting temperature between 1e-3 and 1e-1, several noise seeds.
     One path is 5 to 7 minutes on one core, so 100 paths are about 10 core-hours.
   - Chains of re-melt hops from the best candidates (C7 is one hop; the `chain` mode runs them in parallel).
     The ×2 resolution step gained little, so it comes last.
   - A sequential minimax LP polish at the end, as the erdos threads describe [t281] (untrusted data).
   - Compute: a CPU fleet on Modal, a few euros. No GPU needed: one evaluation is an FFT of 51,200 to 262,144
     points.
4. **Keep** the two running attacks on second-autocorrelation; add nothing to them.

Odds before Sunday 14:00 PT, as estimates:

| Outcome | Estimate | Why |
|---|---|---|
| A candidate at least δ below the live #1 passes WS1's arbitration | about 90 % | two verifiers already agree, margin 28 δ; the risks are a new #1 from another team and the freeze taking time |
| It also stays at least δ below MILO's claimed value | about 80 % | margins of 1.3 to 6.1 δ on seven candidates; the risk is a better result, published or not, by MILO or others before Sunday |
| The same from the live #1, without MILO's copy | under 10 % | 0 δ in our runs; MILO needed 11.8 days |
| A record on the current main problem | not estimated here | see `docs/knockout_v2_brief.md` |

## 6. What remains unverified

- MILO's own value: their construction is not public. We reproduced the basin, not their number.
- The server's score of the candidates: nothing was submitted, and the server's numpy version is not pinned. The
  margin is far above float rounding (1e-15).
- WS1 arbitration: `verify_claim.py` has not seen these candidates.
- Whether a MILO-derived result is acceptable for Track 1 and for EinsteinArena.
- Rows marked *printed* in section 3.2 were not saved and not re-scored by the pinned verifier.
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
.venv/bin/python selection/scout_decide.py                   # -> selection/problem_scout.jsonl
```

Runs are time-budgeted, so a replay on another machine gives a nearby score, not the same one. The stored
candidates are the evidence.
