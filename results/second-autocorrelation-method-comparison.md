# Second autocorrelation: method comparison run log

Date: 2026-11-14  
Problem: `second-autocorrelation-inequality` (EinsteinArena problem 3)  
Purpose: compare the previously measured generic probe with the current L4 evolutionary research runner, and record whether a valid head-to-head was possible.

## Result at a glance

**No valid head-to-head comparison was possible.** The previous generic probe has a completed 600-second measurement. The current L4 runner was launched against the same problem, but all three jobs crashed before search because the only supplied optimizer (`optimizers/1a_tuned.py`) declares `PROBLEM = "1a"`. There is no `optimizers/second-autocorrelation-inequality_*.py` implementation yet. The L4 run returned no candidate or score.

This is an integration failure, not evidence that the new search method is worse. The new method has not yet been tested on this problem.

## Previous method: measured result

Source: `selection/registry_stage3.jsonl`, experiment `ws1-stage3-second-autocorrelation-inequality-1001`.

- Method: generic local perturbation + greedy acceptance + polish; started from the current #1; doubled the step-function resolution in the second half.
- Budget: 600 seconds; seed 1001; 1,814 evaluations.
- Strictly improving mutations: 681, or 67.63 per minute.
- Start score: 0.9636386679928137.
- Best score: 0.963638679229496.
- Gain over the recorded #1: 1.1236683e-8.
- Record threshold in the frozen snapshot: 0.963648667992813 (record + delta, delta = 1e-5).
- Outcome: `headroom_below_delta`; no record improvement. The gain was about 890 times smaller than delta.
- Verification: the best was re-scored by the pinned verifier in a fresh process. No independent exact check was recorded for this probe candidate.

The move rate describes accepted incremental score changes. It does not measure progress toward a record by itself.

## Current method: attempted L4 run

Command:

```text
python -m evolution --problem second-autocorrelation-inequality --optimizer optimizers/1a_tuned.py --rounds 1 --islands 1 --seeds 3 --budget 1 --workers 1 --llm none --store /private/tmp/evo-c2-test.jsonl --certs /private/tmp/evo-c2-certs
```

Observed:

- The target problem loaded and reported the frozen snapshot score 0.963638667992813.
- One cold-start hill-sampling spec was planned.
- All 3 seed jobs failed in 0.055 seconds with `ValueError: optimizer is for 1a, job is for second-autocorrelation-inequality`.
- Scores: none. Valid candidates: none. New certificates: none.
- The experiment store and certificate path were under `/private/tmp`; the project’s existing experiment history was not changed by this attempt.

The first CLI attempt without an explicit optimizer failed even earlier because the default path `optimizers/second-autocorrelation-inequality_tuned.py` does not exist.

## Checks run

- `python -m pytest -q tests/test_10_evolution.py tests/test_9_truth_execution.py tests/test_ws1_frozen_problems.py`: **35 passed, 1 skipped**. These cover synthetic L4 behavior, execution/truth handling, and frozen problem artifacts; they do not provide a real L4 optimization result on this problem.
- The pinned problem definition and independent verifier are present under `problems/second-autocorrelation-inequality/`.
- `evolution` can load the problem, but the L4 job worker requires an optimizer file whose `PROBLEM` exactly matches the problem slug.

## Interpretation and next experiment

The previous method demonstrably made many tiny local gains but remained far below the required record margin. The current L4 architecture cannot yet be judged on this target: it lacks a compatible optimizer, so it did no optimization. Its synthetic tests passing show that its selection machinery works in controlled fixtures, not that its research strategy improves this mathematical objective.

For a fair comparison, add a target-specific optimizer with `PROBLEM = "second-autocorrelation-inequality"`, seeded from the verified incumbent certificate, and expose its search controls as `CONFIG` / `SEARCH_SPACE`. Then run both methods from the same incumbent, on the same machine, with matched wall-clock/CPU budgets and at least three paired seeds. Record per-seed pinned scores, independently check the best candidate, and compare best verified gain, mean gain, accepted gains per minute, evaluations per second, and distance to the live record threshold. Re-read the live leaderboard immediately before the run; this file's numeric record is the frozen 2026-11-14 snapshot, not a permanent current record.

---

## Follow-up (2026-11-14, evening): target-specific optimizer and a matched run

### Why the first attempt failed, and what was fixed

The L4 run crashed for one reason: the only optimizer it was given (`optimizers/1a_tuned.py`) declares
`PROBLEM = "1a"`, and no optimizer existed for this problem. Fixes:

- **Target-specific optimizer:** `optimizers/second-autocorrelation-inequality_blockshift.py` (below).
- **Baseline as a file:** `optimizers/second-autocorrelation-inequality_probe.py` is a faithful port of
  WS1's stage-3 probe (same moves, greedy acceptance, polish phase). It runs at the submittable resolution
  only, and it scores on arrays, which gives it slightly more evaluations per second than the original.
- **The L4 CLI no longer crashes on a mismatch.** `python -m evolution` picks `optimizers/<problem>_*.py`
  when `_tuned` is missing. It refuses to start, with a clear message, when the optimizer's `PROBLEM`
  doesn't match. It used to launch jobs that all crashed.
- **Record rule:** `problems.beats_record` now also requires the platform threshold δ (1e-5 here). Before
  this, a +1e-8 candidate would have been logged as `beats_record: true`.

### Why generic search stalls on this problem

The incumbent is an equioscillating optimum. 384,530 points of f*f lie within 1e-7 of its max (only 3 lie
within 1e-9).
- **Dense updates fail.** Even 1e-9 per coordinate raises some plateau point, and the score drops. Measured
  over 60 s each:
  - monotone gradient steps with a soft max at 3 temperatures, additive and multiplicative: at most +5e-16;
  - Adam on the exact subgradient: always below the start.
- **Sparse block moves still win,** as the probe showed, but the probe spends a full 0.23 s verifier
  evaluation on each random try.

### The method: exact incremental block moves with an optimal step

Adding a constant D to f on a block [a, a+b) changes f*f to exactly `c + D·U + D²·T`.
- `U` is twice a sliding-window sum of f: O(n) with prefix sums.
- `T` is a small triangle.

So for a given block:
- The L2 term is an exact quartic in D.
- The L1 term is `(s + bD)²/(m+1)`.
- The max is checked only on the near-top points. A proven bound says no other point can overtake; when the
  bound fails, the full max is computed.

Each proposal scores a log-spaced grid of D (both signs) exactly and refines the best D by golden-section
search, instead of trying one random D. Block sizes are 2^k with k uniform in 0..20. Half the blocks are
placed near the max points of f*f.

Never worse than the start:
- Every 30 s and at the end, f*f is recomputed and scored with the pinned verifier's exact arithmetic.
- Only a strictly better checkpoint is kept; otherwise the search reverts.

Checks:
- Incremental log-score = pinned verifier to < 1e-12 (`tests/test_11_target_optimizer.py`).
- Default block size tuned on seed 1 only. That seed is not an evaluation seed.

### Matched run

Command: `compare.py --problem second-autocorrelation-inequality --optimizers ..._probe.py ..._blockshift.py
--seeds 101 202 303 --budget 300 --incumbent-file <live #1> --exact`

Setup:
- **Incumbent:** the **live** #1, MLAI_YONSEI = **0.9636615620620852**. It was re-read just before the run and
  re-verified locally to the exact leaderboard value. It has moved by +2.29e-5 since the frozen snapshot.
- **Matched conditions:** same incumbent certificate, same seeds, 300 s per job, all 6 jobs in one batch on
  the same machine (8 cores), so both methods share the same load.
- **Verification:** every candidate re-scored by the frozen verifier in the parent process.

| seed | probe (previous method) | blockshift (new) |
|---|---|---|
| 101 | +8.33e-10 | +1.319e-08 |
| 202 | +2.75e-09 | +1.537e-08 |
| 303 | +3.27e-09 | +1.734e-08 |
| mean | +2.29e-09 | **+1.530e-08** |

- **Crashes:** none (0 of 6 jobs).
- **Blockshift wins every pair.** Paired difference +1.30e-8 (SE 5.3e-10), about 6.7× the probe's mean gain.
- **It also beats the proven bar.** The probe's measured gain at this resolution was +8.97e-9 in about
  300 s, on the older incumbent. Blockshift's worst seed (+1.32e-8) is above it.
- **Best candidate:** `second-autocorrelation-inequality/6645ff5969bd` = **0.9636615794004458**, +1.73e-8
  above the live #1.
- **Exact check:** the independent exact verifier gives 0.963661579400446 (difference +1.1e-16).
- **L4 runner:** one round on top of that best (resumed from the certificate, `hill_sampling`, 3 seeds × 30 s)
  reached **0.9636615795830543** (`/31cc89856232`), with no crash.

### What this is not

- **Not a record.** A record needs ≥ 1e-5 over the live #1; we are at +1.7e-8, which is 577× too small.
  Nothing was submitted.
- **Rate of gain:** at roughly 1.5e-8 per 300 s per process, polishing alone will not reach δ. The gap needs
  a structural change (a different basin, or new constructions), not more CPU on local polishing.
- **Contention cost:** under 6-way parallel load blockshift ran about 1.3 proposals/s, against about 12/s
  alone. Large blocks often trigger a full-max fallback, which is memory-bound. Making that path cheaper is
  the next speed-up.
- **`fresh_seeds` doesn't help here:** its cold restarts begin from the older frozen snapshot, so on this
  problem it wastes its slot.
