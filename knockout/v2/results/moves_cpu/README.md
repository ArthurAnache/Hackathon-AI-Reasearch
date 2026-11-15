# Moves on CPU: can a hand-written move and a polish leave the basin of the #1?

Loop 1 of Knockout v2, 2026-11-14 (20:34 to 22:35 PT), `second-autocorrelation-inequality`, n = 1,999,999.
Code: `knockout/v2/`. Status labels: **verified** (measured here), **hypothesis**, **limitation**.

**Answer: no, not at these budgets (verified).** The tables hold 257 episodes (208 on Modal, 49 on local cores, 645,000 evaluations): 18 move families, at strengths that lower the score by 1e-6 to 1e-2 right after the move (down to 0.48 for the two kicks), followed by the standard polish or by an anneal with 300 to 12,000 evaluations, from solution 2782 and from the live #1 (solution 2783).

- Among the 190 episodes with a move, none ends above "polish alone" from the same start with the same budget. The best values are -2.4e-9 from 2782 (`block_noise`, sigma 3e-6) and -2.4e-9 from 2783 (`neighbour_transfer`, share 1e-5): moves mild enough for the polisher to undo them.
- Among the 53 episodes without a move but with another schedule (reheats, holds, multipliers), the only positive value is +2.5e-10 (`f:p0` at 300 evaluations).
- The best exact score is 0.9636616198579508, +5.78e-8 over the live #1 (0.9636615620620852), where a record needs +1e-5. That is the level where the polisher saturates: `mm8` at 12,000 evaluations, `p0` at 6,000 (+5.77e-8). Arbiter: `VALID_BELOW_RECORD` for the three candidates checked. No record is claimed and nothing was submitted.

Two findings matter more than the tables. Every smoothed version of the problem that was tried (beta 1e6 to 1e10) is already stationary at the leader's solutions, so nothing local is left to gain. And the leader's four solutions of the day show a steady +3e-6 per hour, obtained by re-drawing the fine structure under a fixed envelope, which no move here followed by a CPU-sized polish reproduces.

## What was built

| File | Role |
|---|---|
| `objective.py` | exact score (closed form), smoothed value and gradient at a given beta, optional multipliers, evaluation counter |
| `polish.py` | deterministic polisher driven by an evaluation budget; schedules of `(beta, multipliers)` stages; returns the best exact point seen |
| `moves.py` | 18 move families `move(f, rng, strength) -> f'` |
| `episode.py` | one episode (start, move, strength, seed, schedule, budget) and its value against "polish alone"; outcome classes |
| `plans.py` | the episode lists of each experiment |
| `modal_fleet.py` | one-core workers on Modal, solutions on the Volume `knockout-v2`, `launch` / `finalize` / `promote` |
| `local_fleet.py` | the same episodes on local cores |
| `report.py` | the tables below, from the episode tables |
| `trajectory.py` | comparison of the leader's successive public solutions |
| `export.py` | stored solution to candidate file for the arbiter |

Tests: `.venv/bin/python -m pytest tests/test_knockout_v2_moves.py -q` (19 tests, 2 s).

Definitions:
- **Budget**: objective-and-gradient evaluations (4 FFTs of 4,000,000 points each), move included.
- **Value** of an episode: its final score minus the score of "polish alone" (no move, schedule `p0`) from the same
  start with the same budget. Every score in the tables is from the frozen verifier (`load(slug).score`), computed
  by the worker.
- **Classes**: `improved` (value > 1e-7), `undone` (not improved, and the relative L1 distance to the polished start
  is below 1e-6), `worse_basin` (the rest). The 1e-6 threshold is tight: an anneal without any move already ends
  1.9e-6 away from the polished start with the same score to 4e-9, and is labelled `worse_basin`. Read the classes
  together with the value.
- **Schedules**: `p0` = beta 1e9, 3e9, 1e10 (the standard polish). `a7`, `a6`, `a5` = anneals from beta 1e7, 1e6,
  1e5 up to 3e9. `h5` to `h8` = half of the budget held at beta 1e5 to 1e8, then the anneal. `mm6` to `mm8` =
  exponential method of multipliers at beta 1e6 to 1e8. `f:<name>` = L-BFGS-B on f with the bound f >= 0 instead
  of f = w^2. The budget is shared equally between the stages.

## Runs

Start solutions: `sol2782` = EinsteinArena solution 2782 (the frozen certificate, 0.9636386679928137), `sol2783` =
solution 2783 (live #1 on 2026-11-14, 0.9636615620620852). Both scores re-computed here with the frozen verifier.

| Run | Where | Plan | Episodes finished | Code |
|---|---|---|---|---|
| `20261115T044436Z-smoke` | Modal | 12 episodes of 100 evaluations from `sol2783` (plumbing, timing) | 12 | `8adf3b2` minus the strength calibration |
| `20261115T044954Z-e1` | Modal | first E1 plan, 288 episodes | 0, stopped after 5 minutes: it took 91 of the 100 containers of the workspace | `8adf3b2` |
| `20261115T045519Z-e1` | Modal | **E1**: every family from `sol2782` (158 episodes, 1,500 to 12,000 evaluations) | 151 of 158 | `8adf3b2` |
| `20261115T045015Z-e2_reheat` | Modal | no move, 6 schedules x 4 budgets, from `sol2783` | 21 of 28 | `8adf3b2` |
| `20261115T045248Z-holds` | Modal | holds at beta 1e6, 1e7, 1e8 from both starts, 6,000 and 12,000 evaluations | 9 of 16 | `f8802bf` |
| `20261115T050125Z-deep_holds` | Modal | holds at beta 1e5 from both starts | 1 of 4 | `27a361d` |
| `20261115T053146Z-local_moves` | Modal | the three envelope-preserving families from both starts | 21 of 21 | `af99472` |
| `20261115T060359Z-kicks` | Modal | `window_kick` and `sqrt_kick` from `sol2783`, 1,500 evaluations | 5 of 7 | this commit |
| `e2_moves_local` | 5 local cores | **E2**: every family from `sol2783`, two mildest strengths, 1,500 evaluations | 25 of 25 | `48e31ea` |
| `e1_extra_local` | 3 local cores | `envelope` and `clear_band` from `sol2782`, 1,500 evaluations | 9 of 9 | `48e31ea` |
| `from2416_local` | 3 local cores | no move, `p0`, `a7`, `a6` from solution 2416, 1,500 evaluations | 3 of 3 | `b4b0d91` |
| `pilot1`, `pilot2` | local | 8 episodes of 600 evaluations from `sol2782`; 4 of 300 from `sol2783` with `f:` schedules | 8, 4 | `8adf3b2`, `56418f6` |

Four more Modal launches (`direct`, `e2_moves`, `e1_extra`, `from_2416`, the last one twice) were stopped within
minutes to give containers back to the other agents; none of their episodes finished. Local runs replaced three
of them.

Tables: `modal_all.jsonl` (every finished Modal episode but the smoke run, which is in `modal_smoke.jsonl`), `local_*.jsonl` (local runs), `tables_modal.md` and
`tables_local.md` (the full output of `knockout.v2.report`, with the per-strength tables and the hold traces),
`trajectory_tables.md` (output of `knockout.v2.trajectory`).

## Polish alone (the baseline of every value)

| start | budget | score after polish alone | gain over the start |
|---|---|---|---|
| sol2782 | 1500 | 0.9636387212361036 | +5.32e-08 |
| sol2782 | 3000 | 0.9636387221731777 | +5.42e-08 |
| sol2782 | 6000 | 0.9636387228358014 | +5.48e-08 |
| sol2782 | 12000 | 0.9636387233822282 | +5.54e-08 |
| sol2783 | 1500 | 0.9636616182071287 | +5.61e-08 |
| sol2783 | 3000 | 0.9636616191218205 | +5.71e-08 |
| sol2783 | 6000 | 0.9636616197880807 | +5.77e-08 |

The gain of the standard polish stops near +5.8e-8 from both starts (**verified**). It is not the repair of the
integer rounding of the public solutions (**verified**: rounding the polished solution back to integers with
maximum 1e9, the format of solution 2783, keeps +5.0e-8 of the +5.6e-8; adding uniform noise of half a unit to
solution 2783 costs 2e-10 to 1e-9).

## E1: moves from solution 2782

A basin 2.3e-5 above 2782 is known to exist (solution 2783). Question: does a move followed by a polish land above
"polish alone" by more than 1e-6?

| family | episodes | min value | median value | max value | above +1e-6 | improved | undone | worse_basin | median moved score | median dist to polished start |
|---|---|---|---|---|---|---|---|---|---|---|
| block_noise | 3 | -7.47e-06 | -7.34e-08 | -2.38e-09 | 0 | 0 | 0 | 3 | 0.963615 | 1.62e-05 |
| jitter_runs | 3 | -1.46e-03 | -1.18e-03 | -5.87e-05 | 0 | 0 | 0 | 3 | 0.960233 | 1.86e-03 |
| lognormal | 13 | -1.16e-03 | -1.16e-06 | -2.52e-09 | 0 | 0 | 0 | 13 | 0.963574 | 2.39e-05 |
| merge_runs | 13 | -2.14e-03 | -9.61e-05 | -1.94e-06 | 0 | 0 | 0 | 13 | 0.961650 | 2.64e-04 |
| neighbour_transfer | 3 | -5.20e-05 | -6.55e-07 | -1.01e-07 | 0 | 0 | 0 | 3 | 0.963558 | 2.41e-05 |
| open_boundary | 12 | -1.57e-03 | -1.14e-04 | -3.29e-09 | 0 | 0 | 0 | 12 | 0.963195 | 2.41e-04 |
| open_gradient | 12 | -2.30e-04 | -4.25e-06 | -2.15e-08 | 0 | 0 | 0 | 12 | 0.963518 | 4.97e-05 |
| region_noise | 13 | -3.93e-03 | -5.94e-05 | -5.52e-09 | 0 | 0 | 0 | 13 | 0.963373 | 2.64e-04 |
| reheat | 24 | -5.48e-08 | -1.04e-08 | -4.38e-10 | 0 | 0 | 7 | 17 | 0.963639 | 1.15e-06 |
| shift_runs | 13 | -1.35e-03 | -1.35e-04 | -5.36e-06 | 0 | 0 | 0 | 13 | 0.961997 | 4.90e-04 |
| split_runs | 13 | -3.35e-03 | -7.51e-05 | -6.25e-07 | 0 | 0 | 0 | 13 | 0.962090 | 2.61e-04 |
| swap_support | 13 | -4.84e-04 | -1.81e-05 | -2.13e-06 | 0 | 0 | 0 | 13 | 0.963503 | 1.48e-04 |
| threshold | 13 | -5.22e-04 | -3.55e-06 | -1.01e-08 | 0 | 0 | 0 | 13 | 0.963619 | 8.14e-05 |
| transfer_mass | 13 | -2.82e-04 | -2.02e-06 | -2.52e-09 | 0 | 0 | 1 | 12 | 0.963608 | 8.81e-06 |
| local: clear_band | 4 | -7.70e-04 | -2.09e-04 | -1.56e-06 | 0 | 0 | 0 | 4 | 0.963387 | 3.70e-04 |
| local: envelope | 4 | -1.62e-04 | -3.45e-05 | -5.37e-09 | 0 | 0 | 0 | 4 | 0.963180 | 1.47e-04 |
| local: lognormal | 4 | -1.23e-03 | -6.16e-04 | -1.78e-06 | 0 | 0 | 0 | 4 | 0.962117 | 1.64e-03 |

**No (verified).** From solution 2782, 149 episodes with a move (15 families) and 27 without: no positive value. The best move episode is -2.4e-9 (`block_noise`, sigma 3e-6, 1,500 evaluations). Rows marked `local:` come from the local runs. Values go down with the strength in every family but `envelope`, where strength 3 ends above strength 1 and both end far below (per-strength
tables in `tables_modal.md`); the mildest strengths end within 1e-8 of polish alone, on the same point up to a
relative L1 distance of 1e-6 to 1e-5. The anneal `a7` recovers better than `p0` after a move (for instance
log-normal noise of sigma 3e-5 at 1,500 evaluations: -6.4e-8 with `a7`, -1.2e-6 with `p0`), and more budget brings
the result closer to polish alone without passing it.

Why (**verified** measurements, the reading is a **hypothesis**): the score is extremely sensitive to anything that
is not coordinated across cells. Right after the move, log-normal noise of sigma 3e-5 costs 2.8e-5 and sigma 3e-3
costs 2.8e-3 (about one unit of score per unit of relative L1 distance); splitting 14 runs costs 1.2e-2; opening
18,000 boundary cells at 1 % of their neighbour costs 8.5e-3. The autoconvolution of these solutions is within 1e-7
of its maximum on about 380,000 lags for 485,000 free cells, so almost every direction pushes some lag above the
maximum at first order. The polisher then needs thousands of evaluations to flatten the plateau again, and it
flattens it back to where it was.

## No move, other schedules (reheats, holds, multipliers)

From solution 2782:

| schedule | 1500 evaluations | 3000 evaluations | 6000 evaluations | 12000 evaluations |
|---|---|---|---|---|
| a5 | +6.77e-15 (value -5.32e-08, ends at -1.52e-06) | -4.44e-16 (value -5.42e-08, ends at -1.52e-06) | -4.44e-16 (value -5.48e-08, ends at -1.37e-06) | not run |
| a6 | +6.77e-15 (value -5.32e-08, ends at -6.55e-08) | +6.77e-15 (value -5.42e-08, ends at -7.74e-08) | +6.77e-15 (value -5.48e-08, ends at -6.26e-08) | not run |
| a7 | +4.91e-08 (value -4.11e-09, ends at +4.63e-08) | +5.05e-08 (value -3.68e-09, ends at +4.73e-08) | +5.19e-08 (value -2.94e-09, ends at +4.88e-08) | not run |
| h5 | not run | not run | +6.77e-15 (value -5.48e-08, ends at -1.22e-06) | not run |
| h6 | not run | not run | +6.77e-15 (value -5.48e-08, ends at -9.53e-08) | not run |
| h7 | not run | not run | +4.44e-08 (value -1.04e-08, ends at +4.14e-08) | not run |
| h8 | not run | not run | +5.37e-08 (value -1.16e-09, ends at +5.07e-08) | +5.47e-08 (value -6.55e-10, ends at +5.17e-08) |
| mm6 | +6.77e-15 (value -5.32e-08, ends at -2.85e-07) | +6.77e-15 (value -5.42e-08, ends at -3.40e-07) | +6.77e-15 (value -5.48e-08, ends at -5.09e-07) | not run |
| mm7 | +4.29e-08 (value -1.04e-08, ends at +3.50e-08) | +4.89e-08 (value -5.28e-09, ends at +4.41e-08) | +5.12e-08 (value -3.64e-09, ends at +4.68e-08) | not run |
| mm8 | +5.21e-08 (value -1.13e-09, ends at +4.79e-08) | +5.37e-08 (value -5.10e-10, ends at +5.05e-08) | +5.42e-08 (value -6.09e-10, ends at +5.09e-08) | +5.50e-08 (value -4.38e-10, ends at +5.21e-08) |
| p0 | +5.32e-08 (value +0.00e+00, ends at +5.21e-08) | +5.42e-08 (value +0.00e+00, ends at +5.31e-08) | +5.48e-08 (value +0.00e+00, ends at +5.37e-08) | +5.54e-08 (value +0.00e+00, ends at +5.43e-08) |

From solution 2783:

| schedule | 1500 evaluations | 3000 evaluations | 6000 evaluations |
|---|---|---|---|
| a5 | +4.44e-15 (value -5.61e-08, ends at -1.42e-06) | +4.44e-15 (value -5.71e-08, ends at -1.30e-06) | +4.44e-15 (value -5.77e-08, ends at -1.28e-06) |
| a6 | +4.44e-15 (value -5.61e-08, ends at -1.15e-07) | +3.44e-15 (value -5.71e-08, ends at -7.41e-08) | +3.44e-15 (value -5.77e-08, ends at -3.89e-08) |
| a7 | +5.10e-08 (value -5.13e-09, ends at +4.77e-08) | +5.29e-08 (value -4.14e-09, ends at +4.96e-08) | +5.47e-08 (value -3.04e-09, ends at +5.13e-08) |
| h6 | not run | not run | +3.44e-15 (value -5.77e-08, ends at -6.82e-08) |
| h7 | not run | not run | +4.74e-08 (value -1.03e-08, ends at +4.41e-08) |
| h8 | not run | not run | +5.68e-08 (value -9.11e-10, ends at +5.36e-08) |
| mm6 | +3.44e-15 (value -5.61e-08, ends at -3.16e-07) | +4.44e-15 (value -5.71e-08, ends at -4.99e-07) | +3.44e-15 (value -5.77e-08, ends at -3.97e-07) |
| mm7 | +4.52e-08 (value -1.09e-08, ends at +3.56e-08) | +4.94e-08 (value -7.68e-09, ends at +4.29e-08) | +5.42e-08 (value -3.53e-09, ends at +4.96e-08) |
| mm8 | +5.25e-08 (value -3.60e-09, ends at +4.82e-08) | +5.60e-08 (value -1.09e-09, ends at +5.11e-08) | +5.74e-08 (value -3.75e-10, ends at +5.42e-08) |
| p0 | +5.61e-08 (value +0.00e+00, ends at +5.49e-08) | +5.71e-08 (value +0.00e+00, ends at +5.59e-08) | +5.77e-08 (value +0.00e+00, ends at +5.65e-08) |

Each cell: gain over the start of the answer returned (best exact point seen), then its value against polish alone,
then where the schedule itself ended.

- No schedule beats `p0` at any budget (**verified**). Schedules that stay at beta >= 1e7 (`a7`, `h7`, `h8`,
  `mm7`, `mm8`) come back to the same point from below, within 1e-8. Schedules that go down to beta 1e6 or 1e5
  (`a6`, `a5`, `h6`, `h5`, `mm6`) end below their start. `a5` and `h5` end 1.2e-6 to 1.5e-6 below their start at every budget tried (1,500 to 6,000 evaluations): a visit to beta 1e5 leads to a slightly worse arrangement, and a longer sharpening does not bring it back. `a6`, `h6` and `mm6` end 4e-8 to 5e-7 below their start.
- The smoothed objective has almost nothing left to gain near these solutions (**verified**, hold traces in
  `tables_modal.md`). Total gain of -log C_beta during a hold from 2783: 5.9e-7 at beta 1e6 (2,769 evaluations),
  5.5e-8 at 1e7 (3,272), 1.2e-9 at 1e8 (2,931, where L-BFGS stops by itself). After the first stage the gain per
  evaluation is about 1e-11 at 1e6, 5e-13 at 1e7 and 2e-14 at 1e8. Solution 2782 gives the same numbers, although
  a solution 2.3e-5 better exists: local descent on the smoothed objective at beta >= 1e6 does not see that gain.
- The multiplier method removes part of the LogSumExp bias (at beta 1e6 the exact score goes from -4.4e-6 to
  -1.1e-6 below the start in three updates) and then oscillates. It does not end above `p0`.

## E2: moves from solution 2783 (the live #1)

| family | episodes | min value | median value | max value | above +1e-6 | improved | undone | worse_basin | median moved score | median dist to polished start |
|---|---|---|---|---|---|---|---|---|---|---|
| block_noise | 4 | -7.53e-06 | -1.56e-08 | -2.79e-09 | 0 | 0 | 0 | 4 | 0.963648 | 7.74e-06 |
| jitter_runs | 4 | -4.73e-03 | -6.30e-05 | -1.96e-06 | 0 | 0 | 0 | 4 | 0.961577 | 2.82e-04 |
| neighbour_transfer | 4 | -1.70e-04 | -1.82e-06 | -2.41e-09 | 0 | 0 | 0 | 4 | 0.963626 | 1.81e-05 |
| reheat | 21 | -5.77e-08 | -1.09e-08 | -3.75e-10 | 0 | 0 | 8 | 13 | 0.963662 | 1.11e-06 |
| sqrt_kick | 2 | -1.32e-02 | -7.33e-03 | -1.44e-03 | 0 | 0 | 0 | 2 | 0.930133 | 1.12e-01 |
| window_kick | 2 | -8.96e-03 | -4.82e-03 | -6.75e-04 | 0 | 0 | 0 | 2 | 0.760077 | 2.74e-02 |
| window_kick_d8 | 1 | -1.57e-02 | -1.57e-02 | -1.57e-02 | 0 | 0 | 0 | 1 | 0.481687 | 9.12e-02 |
| local: clear_band | 2 | -5.23e-05 | -2.69e-05 | -1.48e-06 | 0 | 0 | 0 | 2 | 0.963617 | 1.05e-04 |
| local: envelope | 2 | -3.28e-06 | -1.64e-06 | -6.59e-09 | 0 | 0 | 0 | 2 | 0.963616 | 1.88e-05 |
| local: lognormal | 2 | -3.66e-08 | -1.98e-08 | -3.05e-09 | 0 | 0 | 0 | 2 | 0.963643 | 7.70e-06 |
| local: merge_runs | 2 | -2.42e-04 | -1.31e-04 | -1.89e-05 | 0 | 0 | 0 | 2 | 0.961275 | 3.43e-04 |
| local: open_boundary | 2 | -7.71e-06 | -3.86e-06 | -1.34e-08 | 0 | 0 | 0 | 2 | 0.963612 | 2.45e-05 |
| local: open_gradient | 2 | -1.25e-06 | -6.52e-07 | -5.59e-08 | 0 | 0 | 0 | 2 | 0.963571 | 2.09e-05 |
| local: region_noise | 2 | -3.14e-05 | -1.57e-05 | -2.66e-08 | 0 | 0 | 0 | 2 | 0.963559 | 1.29e-04 |
| local: shift_runs | 2 | -3.14e-05 | -3.13e-05 | -3.12e-05 | 0 | 0 | 0 | 2 | 0.963142 | 1.34e-04 |
| local: split_runs | 2 | -5.99e-04 | -5.67e-04 | -5.35e-04 | 0 | 0 | 0 | 2 | 0.951667 | 1.59e-03 |
| local: swap_support | 2 | -1.50e-05 | -9.37e-06 | -3.74e-06 | 0 | 0 | 0 | 2 | 0.963578 | 5.32e-05 |
| local: threshold | 2 | -4.90e-07 | -2.55e-07 | -2.01e-08 | 0 | 0 | 0 | 2 | 0.963651 | 2.20e-05 |
| local: transfer_mass | 2 | -3.76e-09 | -3.14e-09 | -2.51e-09 | 0 | 0 | 0 | 2 | 0.963645 | 1.50e-06 |

From solution 2783, 41 episodes with a move (17 families) and 24 without. No move episode has a positive value; the best is -2.4e-9 (`neighbour_transfer`, share 1e-5, 3,000 evaluations). Rows marked `local:` are the local run of E2 (every family at its two mildest strengths); the other rows are the Modal runs `local_moves` and `kicks`.

- The three envelope-preserving families (`block_noise`, `jitter_runs`, `neighbour_transfer`), which re-draw the mass inside small neighbourhoods as one step of the leader does, behave like the others: best values -2.8e-9, -2.0e-6 and -2.4e-9 from 2783.
- The two kicks at the suggested sizes (15 % of the run fragments moved by up to 2 or 8 cells; sigma 0.15 in square-root coordinates) lower the score to 0.57, 0.48 and 0.90, and end 9e-3, 1.6e-2 and 1.3e-2 below polish alone after 1,500 evaluations. At 0.1 % of the fragments and sigma 0.01 they still end 7e-4 and 1.4e-3 below.
- Calibration on solution 2416 (local, 1,500 evaluations): polish alone gains +4.3e-8 and `a7` +3.8e-8. The leader went from that solution to +2.48e-5 (solution 2780): its method does there what this polisher cannot.

## The leader's trajectory (`trajectory_tables.md`)

`get_best_solutions(3, limit=5)` returns the four solutions MLAI_YONSEI submitted on 2026-11-14 (2780 at 07:59 UTC,
2781 at 12:27, 2782 at 15:35, 2783 at 22:41) and solution 2416 (ClaudeExplorer, 2026-07-22, 0.963588), all with
n = 1,999,999. **Verified** by reading the five solutions:

- The leader gains +1.29e-5, +1.29e-5 and +2.29e-5 in 4.5 h, 3.1 h and 7.1 h: about 3e-6 per hour, submitted each
  time the gain passes the 1e-5 threshold. Solution 2780 is 0.20 away (relative L1) from the public solution 2416,
  closer than to anything else: the series is a warm start from it.
- Each step moves 15 % to 23 % of the mass cell by cell, but only 0.025 % to 0.044 % between blocks of 65,536 cells.
  The envelope is kept and the mass is re-drawn inside neighbourhoods of a few hundred cells or fewer. This agrees
  with the coordinator's independent measurement.
- Cell by cell the steps are almost uncorrelated (cosine between successive displacements 0.09 to 0.18). The small
  displacement of the envelope keeps its direction (cosine 0.90 to 0.93 for blocks of 65,536 cells): at every step
  the first 5 % of the grid gains mass (0.00830 to 0.00857 of the total over the five solutions) and the sparse
  band between 65 % and 90 % loses mass (0.00061 to 0.00054).
- Following that drift does not help (**verified**): at solution 2783 the smoothed objective decreases along the
  envelope drift for beta 1e5 to 1e7; along the steepest smooth multiplicative field (the gradient restricted to
  hat functions on blocks of 1,024 to 65,536 cells) the best gain of the smoothed objective is 3e-7 at beta 1e6,
  3e-8 at 1e7 and 7e-10 at 1e8, with a lower exact score; and the `envelope` move ends below polish alone at every
  strength.
- From solution 2782 the straight direction towards 2783 raises the score for steps below 1e-6 of the way
  (+1.1e-5 per unit) and then lowers it (-1.1e-2 per unit): the path between two successive solutions of the
  leader is not a straight line, and the two are separate local optima for us.

**Hypothesis:** the leader runs one long optimization that keeps the envelope and re-draws the fine structure,
with far more evaluations than a CPU core gives (one step of the leader in 3 to 7 hours; one evaluation costs
0.4 s on a Modal core). Nothing here identifies its method.

## Checks suggested during the run (coordinator, second model)

All **verified** by local scripts on the public solutions (not in the repository; the numbers are below).

- **Support comparison at explicit thresholds** (2782 against 2783, cells above t times the maximum). The spread
  of the log-ratio on common cells is dominated by negligible cells: its standard deviation is 2.1 over all cells,
  1.0 for cells above 1e-6 of the maximum and 0.33 above 1e-3 (mass-weighted: 0.90, 0.52, 0.28). The quartiles of
  the ratio barely move (0.79 and 1.23 over all cells, 0.86 and 1.15 above 1e-3). The support change is real at
  every threshold: above 1e-3 of the maximum, 38,398 cells of 2782 (3.7 % of the mass) are below it in 2783 and
  35,909 cells come in.
- **The two terms of 1 - C.** With z = c / max c, 1 - C = [sum z (1 - z) + (1/6) sum (z_{k+1} - z_k)^2] / sum z
  (checked to 1e-13 on the five solutions). From 2416 to 2783 the first term goes from 0.0363988 to 0.0363202 and
  the roughness term goes up from 1.31e-5 to 1.83e-5: the leader gains by filling the plateau (sum z / (2n - 1)
  from 0.606321 to 0.606583) and pays a little roughness for it.
- **Reduced gradient with multipliers.** The multipliers were not fitted by a separate solve; they are the
  LogSumExp weights at a point that L-BFGS brought close to stationarity (700 evaluations from 2783 at beta 1e8,
  then at 1e7). There the gradient on the positive cells is down to 1.5e-6 (median) and 1.6e-5 (maximum). Of the
  1,517,283 zero cells, 27,724 have a positive gradient at beta 1e8 (36,340 at 1e7), 16,956 above 1e-6 (24,331),
  none above 1e-5, and the largest (7e-6) is below the residual on the positive cells. So at this accuracy no
  zero cell asks for mass in a way that stands out. Opening the 1,000 hottest ones raises the exact score linearly
  by 4e-9 per 1e-6 of the mean cell, while the smoothed objective goes down: a gain of the size of the plateau's
  roughness, not a new direction. At the sharply polished solution the LogSumExp gradient is not a valid set of
  multipliers (240,000 zero cells look positive, all the hottest in the last 5 % of the grid) and opening them
  lowers the exact score at first order (-3.5e-8 per 1e-6 of the mean cell for 1,000 cells). Two more facts point
  the same way: the polisher on f with bounds opens 195,000 cells and ends within 3e-10 of `p0`, and the
  `open_gradient` family ends below polish alone at every strength.
- **Kicks**: see E2 above.

## Arbiter

`selection/verify_claim.py`, run locally on downloaded solutions; verdicts are in `selection/claims.jsonl`.

| Candidate (run / episode) | Pinned verifier | Independent verifier | Gain over the live #1 | Verdict |
|---|---|---|---|---|
| `holds` / `sol2783-reheat-s0-r0-p0-b6000` (polish alone) | 0.9636616197880807 | 0.9636616197880808 | 5.773e-08 | `VALID_BELOW_RECORD` |
| `e2_reheat` / `sol2783-reheat-s0-r0-mm8-b6000` | 0.9636616194133927 | 0.9636616194133927 | 5.735e-08 | `VALID_BELOW_RECORD` |
| `e2_reheat` / `sol2783-reheat-s0-r0-mm8-b12000` | 0.9636616198579508 | 0.9636616198579508 | 5.780e-08 | `VALID_BELOW_RECORD` |

Live #1 read by the arbiter at 22:21 PT: 0.9636615620620852 by MLAI_YONSEI, delta 1e-5, target 0.9636715620620852.

No candidate beats the record. Nothing was submitted.

## Modal: timing, cost, limits

- One evaluation takes 0.27 to 0.87 s on a Modal core depending on the container (mean 0.45 s), 0.17 s on an idle
  local core (Apple silicon) and 0.35 s or more when 8 local cores run episodes at once (**verified**).
- Peak memory of one episode: 1.3 GiB. Workers reserve 1 core and 2 GiB.
- Estimated spend: 4.56 USD for the 208 finished Modal episodes of the tables (72.3 worker-hours at 1 core and 2 GiB, at 0.0000131 USD per core-second and 0.00000222 USD per GiB-second); about 0.7 USD for the launches stopped early (0.46 for the first E1 launch); about 0.3 USD for the drivers, the finalize calls and the smoke run; about 2.3 USD for the 27 episodes that were still running when the tables were collected (mostly 12,000 evaluations each). About 7.9 USD in total, of a 30 USD budget. Modal's dashboard has the real bill.
- **Limitation:** the workspace runs at most 100 containers in total. The first E1 launch (288 episodes) took 91
  of them, and this fleet left the GPU agent's apps without a container from 21:16 to 21:22 PT (observed) and
  probably from 20:51 to 20:55. `KO_V2_MAX_WORKERS` (default 40) now caps one launch; several launches still add
  up, so check `modal container list` before launching.

## Verified, hypothesis, limitation

Verified:
- every score in the tables (frozen verifier, in the worker); the three arbiter verdicts;
- no episode with a move ends above polish alone, from 2782 or from 2783, for any of the 18 families, any
  strength, any schedule, any budget up to 6,000 evaluations (only 5 episodes of 12,000 evaluations had finished
  when the tables were collected, none of them with a move);
- polish alone saturates near +5.8e-8, and this is not rounding repair;
- the smoothed objective is stationary near both solutions for beta >= 1e6, up to the small gains listed above;
- the facts about the leader's five public solutions.

Hypotheses:
- Solutions 2782 and 2783 are true local optima of the non-smooth problem, of the "vertex" kind: about 380,000
  lags are active for 485,000 cells, so every first-order direction is blocked and a better point needs a
  coordinated change of many cells and of the active set.
- The leader's method is one long optimization at far more evaluations than these episodes.

Limitations:
- Budgets stop at 12,000 evaluations per episode (about 85 minutes on a Modal core). An episode cannot tell what
  10^5 to 10^6 evaluations after a move would give.
- One seed per (family, strength, schedule, budget) in most cells: the tables show no variance. The polisher is
  deterministic and the values are monotone in the strength, which is why more seeds were not bought.
- The `undone` threshold (1e-6) is tighter than the path dependence of the polisher, so mild moves that end on the
  polished start to 1e-5 are labelled `worse_basin`.
- The `f:` schedules (polisher on f with bounds) were only piloted locally at 300 evaluations (`f:p0` equals `p0`
  to 3e-10 and opens 195,000 cells); their Modal run was stopped for the container limit.
- `envelope` and `clear_band` ran at 1,500 evaluations only. The multiplier schedules were not tuned.
- Scores of the same episode differ by about 3e-12 between a Modal container and the local machine (different
  CPUs, same numpy and scipy): values below 1e-10 mean nothing.
- Runs were launched from the working tree; the "Code" column gives the commit that holds that tree.

## What to try next

1. **Match the leader's regime before looking for a shortcut.** Its steps keep the envelope and re-draw the fine
   structure at a steady 3e-6 per hour. On GPU, run one chain for 10^5 to 10^6 evaluations and score sharpened
   snapshots along the way. The holds above say that staying at beta >= 1e6 with L-BFGS gains nothing near these
   solutions, so a chain that only does that will stall where the episodes stalled: it has to go lower in beta
   for much longer, or use another optimizer, and the sharpening after a visit to beta 1e5 costs about 1.4e-6 that
   must be earned back.
2. **Expect the target to move.** At 3e-6 per hour the leader passes the 1e-5 threshold again every 3 to 4 hours;
   its last submission was at 22:41 UTC on 2026-11-14. A record needs the live #1 at that time plus 1e-5.
3. **A solver for the non-smooth problem**, instead of LogSumExp at high beta: sequential linear programming on
   the exact maximum with a trust region (the linearized problem has the cells as variables and the lags as
   constraints; a first-order LP solver needs only the same FFTs). This is the textbook way off a vertex.
4. Moves are worth another look only with a much longer polish behind them (GPU), and then the envelope-preserving
   ones first (`block_noise`, `jitter_runs`, `neighbour_transfer`), since they are what a step of the leader looks
   like.

## How to reproduce

```bash
# start solutions as float64 .npy on the Volume (2782 = frozen certificate, 2783 = live #1 on 2026-11-14)
.venv/bin/python -m knockout.v2.trajectory --fetch starts/            # saves starts/sol<id>.npy (read-only GET)
.venv/bin/modal volume put knockout-v2 starts/sol2783.npy starts/sol2783.npy     # same for 2782, 2780

KO_V2_MAX_WORKERS=50 .venv/bin/modal run --detach -m knockout.v2.modal_fleet::launch --plan e1 --max-usd 5
.venv/bin/modal run -m knockout.v2.modal_fleet::finalize --run-id <run_id>[,<run_id>...] --name <table name>
.venv/bin/python -m knockout.v2.report knockout/v2/results/moves_cpu/<table name>.jsonl --holds

# the same plan on local cores
.venv/bin/python -m knockout.v2.local_fleet --plan smoke --starts starts/ --workers 6

# arbiter on one stored solution
.venv/bin/modal volume get knockout-v2 <run_id>/solutions/<episode_id>.npy best.npy
.venv/bin/python -m knockout.v2.export best.npy best.json
.venv/bin/python selection/verify_claim.py second-autocorrelation-inequality best.json
```
