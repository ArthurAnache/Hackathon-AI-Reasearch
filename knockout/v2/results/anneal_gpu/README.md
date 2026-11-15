# Deep annealing chains on a GPU — `second-autocorrelation-inequality`

Loop 1 of the v2 improvement effort, 2026-11-14 (20:35 to 22:30 PT). Code: `knockout/v2/gpu/`. Branch:
`feature/ws2-v2-anneal-gpu`.

**Question.** From the live #1 (EinsteinArena solution 2783) or from the previous #1 (solution 2782, the frozen
certificate), does a long "reheat then cool" optimization land in a better basin? We know a basin 2.3e-5 above
2782 exists, because 2783 is in it.

**Answer: no.** 108 chains were run from the two records (96 of 30,000 evaluations, 12 of 20,000): reheats from
β = 1e7 down to 1e3, zero cells opened by a floor or by their derivative, cell kicks (free, keeping block
masses, additive in √f), block means, and starts moved along the step between the two records. None ends above
the plain polish of its start, which gives +5.9e-8 from solution 2783 and +5.6e-8 from solution 2782 and is the
exact local maximum. The mild reheats (β = 1e7 and 1e6) come back to that polish within 1.3e-8; every other
chain ends below the start itself. The best exact score is **0.9636616210553628**, 5.899e-8 above the live #1;
the arbiter says `VALID_BELOW_RECORD`; a record needs 1e-5. **No record is claimed.**

Status labels: **verified** (measured here, numbers below), **hypothesis**, **limitation**. Every score is the
frozen verifier's float64 score computed on CPU; a GPU number is never a score.

## What was built

| File | Content |
|---|---|
| `knockout/v2/gpu/objective.py` | The smoothed objective and its gradient for a batch of chains. Same formula as the frozen verifier except f = w² and a LogSumExp maximum. Written against an array namespace, so numpy (tests) and torch (GPU) run the same lines |
| `knockout/v2/gpu/lbfgs.py` | L-BFGS for a batch of independent rows, Armijo backtracking, one useful evaluation per row and per batched call |
| `knockout/v2/gpu/chain.py` | The annealing loop: a list of stages (β, evaluations) per chain, the best exact point seen, optional opening of closed cells |
| `knockout/v2/gpu/schedule.py` | Schedules, chain specifications, the plans (`deep16`, `kick16`, `reopen16`, `directed16`, `sqrtkick12`), ids, records, cost estimate |
| `knockout/v2/gpu/starts.py` | Changes of a warm start before a chain: cell kick (free, keeping block masses, or additive in √f), block means, a move along the step between the two records |
| `knockout/v2/gpu/worker.py` | What runs in the container: warm starts, the E0 benchmark, a batch of chains, basin hopping |
| `knockout/v2/gpu/cpu_score.py` | CPU re-scoring of a stored solution with the frozen verifier, in a separate process |
| `knockout/v2/gpu/modal_app.py` | Modal app: `gpu="H100"`, 90-minute timeout, at most 2 containers, Volume `knockout-v2-gpu` |
| `knockout/v2/gpu/export.py`, `report.py` | A stored `.npy` solution to the arbiter's JSON; the chain table below |
| `requirements-gpu.txt` | `torch==2.14.1`, installed in the Modal image only |
| `tests/test_knockout_v2_gpu.py` | 13 tests, about 1 s, no torch: schedules, records, objective against the seed program and finite differences, batched L-BFGS, loop bookkeeping, start changes |

Built but not run, for lack of time (limitation): the basin-hopping job (`--job hop`: small kicks, a short polish,
keep the best row, repeat), which has never been executed, and the plans `coarse16`, `blockkick16` and
`blockkick8`, for which the unit tests only check a consistent budget.

Two choices that differ from the seed program (`knockout/seeds/second_autocorrelation.py`):

- **β has one meaning along a chain.** The objective works on g = κ f / Σf with κ fixed so that max(g∗g) = 1 at the
  chain's start. It is then exactly invariant under f → a f. The plain LogSumExp is not: scaling f up raises the
  effective β, which an optimizer can exploit during a long hot stage.
- **Restarts keep searching downwards for a step length.** At β = 1e9 the smoothed maximum is smooth only over
  relative steps of about 1e-9. A first version restarted every line search from 1/|g| and never accepted a step
  (0 of 159 evaluations, measured locally).

## E0 — measure first (verified)

One H100 (`NVIDIA H100 80GB HBM3`), torch 2.14.1+cu130, CUDA 13.0, float64, n = 1,999,999, FFT length 4,000,000
(zero-padded, linear convolution). Raw numbers: `bench/20261115T045027Z.json`.

| Measurement | Value |
|---|---|
| torch objective minus numpy objective on solution 2783, β = 1e5, 1e7, 1e9, 1e11 | at most 8.0e-16 |
| Gradient, largest difference over largest entry, same four β | 1.3e-11, 1.1e-9, 6.2e-8, 4.6e-15 |
| GPU exact score minus the frozen verifier's score | −2.2e-16 |
| One value-and-gradient evaluation, 1 chain, GPU | 1.14 ms (875 evaluations per second) |
| Same, 16 chains in one tensor | 16.6 ms (965 chain-evaluations per second) |
| Same on a CPU core of that container (seed program, scipy.fft) | 0.295 s (3.4 evaluations per second) |
| GPU against one CPU core | 258 times faster |
| One full optimizer step (trial point, evaluation, L-BFGS update, history 8), 1 / 8 / 16 chains | 4.1 / 17.8 / 31.5 ms |
| Full step in the real runs, 16 chains, stage-end saves included | 33.4 to 33.8 ms (30,000 steps in 1,003 to 1,015 s) |
| FFT length 4,194,304 (power of two) instead of 4,000,000 | 1 to 6 % faster, not used |
| Peak GPU memory, 16 chains | 9.8 GiB of 80 |

Cost (Modal list prices read on 2026-11-14: H100 0.001097 USD/s, CPU core 0.0000131 USD/s, memory 0.00000222
USD/GiB/s; the container is one H100, 6 cores and 32 GiB, so 0.001247 USD/s or 4.49 USD per hour):

| | USD per 1,000 evaluations |
|---|---|
| Raw evaluation on the GPU, 1 chain | 0.0014 |
| Chain evaluation in a 16-chain run, optimizer and storage included (measured) | 0.0026 |
| One CPU core on Modal (0.295 s per evaluation, core price only) | 0.0039, and 258 times slower in wall clock |

Two things this changes for the team:

- **One chain already saturates the GPU.** Batching 16 chains gives 10 % more raw throughput, not 16 times. Batching
  is still right (the per-step fixed costs are shared: 509 against 246 chain-evaluations per second), but a
  second GPU doubles the throughput and a bigger batch does not.
- **Evaluations are no longer the scarce resource.** 16 chains of 30,000 evaluations cost about 1.3 USD and 17
  minutes. Waiting for an H100 took 4 to 7 minutes on three launches out of nine.

## E1 and E2 — deep reheat, then slow cooling (verified)

Plan `deep16`, 16 chains per run, 30,000 evaluations per chain, one run from solution 2782 (E1) and one from
solution 2783 (E2). Every chain cools to β = 1e9 with 3 stages per decade on 75 % of its budget, then polishes at
3e9, 1e10, 3e10 and 1e11. "Floor" opens the zero cells before the chain (value = floor × mean non-zero value).

Final exact score minus the start, by starting β (support fixed):

| Starting β | From 2782 (E1) | From 2783 (E2) | Mass moved from 2782 (L1, E1) |
|---|---|---|---|
| 1e9, no reheat (control) | +5.605e-8 | +5.899e-8 | 0.0000024 |
| 1e7 | +5.607e-8 | +5.871e-8 | 0.000014 |
| 1e6 | +4.322e-8 | +4.712e-8 | 0.00011 |
| 1e5 | −7.206e-8 | −9.102e-8 | 0.00089 |
| 1e4 | −1.349e-6 | −1.542e-6 | 0.0077 |
| 1e3 | −2.189e-5 | −2.124e-5 | 0.081 |

- **E1's question: does any chain end above 2782 by more than 1e-6? No.** The best chains from 2782 end at
  +5.6e-8, which is the polish. The basin 2.3e-5 above 2782 was not reached by any of the 15 reheated chains.
- **The deeper the reheat, the lower the chain ends**, in the same way from both starts. A reheat at β ≥ 1e7
  comes back to the same point. A reheat at 1e5 or below ends in another local maximum, lower than the start.
  Each decade of starting β moves about 8 to 10 times more mass and loses about 10 times more score.
- A reheat does move mass, but far less than the jump between the two records, and not towards it: the chain
  that started at β = 1e3 ends at L1 distance 0.081 from 2782 and 0.208 from 2783 (2782 and 2783 are 0.225
  apart).
- **Opening the zero cells with a floor never helped.** A floor of 1e-6 ends like the fixed-support chain of the
  same depth. A floor of 1e-3 ends like it after a deep reheat (β ≤ 1e4) and 5e-6 to 3e-4 lower after a mild one
  (β = 1e5, 1e6); a floor of 1e-2 ends 3.6e-4 lower. With f = w² a small cell moves slowly in both directions:
  after 30,000 evaluations the chain `b4-open3` of E1 still has 85,523 formerly-zero cells above 1e-3 of the mean
  cell value, and its L1 distance to the start is 0.0065 (0.0077 without floor). So the loss looks like leftover
  mass, not another basin (hypothesis for the other floored chains, which were not inspected).
- Four fast cycles are worse than one slow cycle with the same budget (−6e-6 to −8e-6).

## Round 2 — stochastic reheat and support moves (verified)

Two runs of 16 chains, 30,000 evaluations each.

**Kicks, from solution 2782** (plan `kick16`). A kick multiplies every cell by an independent log-normal factor of
width σ; σ = 0.3 moves 23.5 % of the mass, as much as the step between the two records (22.5 %), and takes the
exact score down to about 0.74 (both measured on solution 2783). Then one slow cooling.

| Kick width σ | Cooling from β | Seeds | Final minus start |
|---|---|---|---|
| 0.3 | 1e4 | 2 | −2.14e-5, −1.98e-5 |
| 0.03 | 1e5 | 1 | −1.17e-4 |
| 0.1 | 1e5 | 3 | −1.72e-4, −1.55e-4, −2.5e-5 |
| 0.3 | 1e5 | 3 | −5.01e-4, −4.89e-4, −7.7e-5 |
| 1.0 | 1e5 | 2 | −3.36e-3, −1.37e-3 |
| 0.3 | 1e6 | 2 | −6.94e-3, −6.72e-3 |
| 0.1 and 0.3, with a floor of 1e-3 | 1e5 | 1 each | −2.12e-4, −2.82e-4 |

- **No kicked chain ends above 2782.** The best is 2.0e-5 below it.
- **A kicked chain needs a hot start.** From β = 1e4 it comes back to within 2e-5; from 1e5 it stays 2.5e-5 to
  5e-4 below, from 1e6 it stays 7e-3 below.
- **The chain is pulled back to the solution it was kicked from.** The two σ = 0.3 chains cooled from 1e4 are at
  L1 distance 0.060 and 0.058 from 2782 at β = 2e7 (0.235 right after the kick), 0.063 from each other and 0.222
  from 2783. They are not a new basin: they are solution 2782 with 6 % of the mass left out of place.
- Loss and distance to the start go together, kicked or not: 0.0009 → 1.3e-7, 0.0077 → 1.4e-6, 0.060 → 2.1e-5,
  0.081 → 2.2e-5 (losses against the polished start). Around a record, every point our annealer reaches is lower
  in proportion to how far it stopped from the record (hypothesis: a cone-shaped neighbourhood; four points).

**Opening cells by their derivative, from solution 2783** (plan `reopen16`). While β ≤ 3e7, every 500 evaluations,
the closed cells with a positive derivative of the smoothed objective are opened (the strongest at η times the
mean cell value).

| Starting β | η = 1e-4 | η = 1e-3 | η = 1e-2 | Same reheat, support fixed (E2) |
|---|---|---|---|---|
| 1e7 | −2.18e-6 | −2.14e-4 | −5.02e-3 | +5.87e-8 |
| 1e6 | −2.02e-5 | −2.78e-4 | −1.40e-3 | +4.71e-8 |
| 1e5 | −1.88e-7 | −7.69e-7 | −4.13e-3 | −9.10e-8 |
| 1e4 | not run | −2.05e-6 | −4.55e-3 | −1.54e-6 |

- **Every chain with reopening ends below the same chain without it**, and the loss grows with η.
- The criterion opens far too many cells: 290,000 to 1,600,000 openings per chain, for 1,517,283 zero cells. At a
  hot β most zero cells have a positive smoothed derivative, so this is close to the floor of E1.
- Three kicked chains from 2783 (σ = 0.03, 0.1, 0.3, cooled from 1e5) end at −1.18e-4, −1.34e-4, −5.48e-4, like
  the same kicks from 2782; the σ = 0.1 kick with reopening (η = 1e-3) ends at −1.57e-4.

## Round 3 — kicks that keep the envelope, random and directed (verified)

Plan `directed16`, 16 chains of 30,000 evaluations, once from each record. It follows the coordinator's
measurement that the two records have the same mass per block of 1,000 cells and differ inside neighbourhoods of
about 100 cells or fewer. Final exact score minus the start:

| Change of the start, then cooling from β = 1e4 | From 2783 | From 2782 |
|---|---|---|
| None (same reheat, E1 and E2) | −1.54e-6 | −1.35e-6 |
| Kick σ = 0.3 keeping the mass of every 100 cells, 2 seeds | −1.93e-5, −1.97e-5 | −1.95e-5, −2.24e-5 |
| Kick σ = 0.3 keeping the mass of every 1,000 cells | −1.93e-5 | −2.16e-5 |
| Kick σ = 0.3, block masses free (round 2) | not run | −2.14e-5, −1.98e-5 |
| Kick σ = 0.1 keeping the mass of every 100 cells | −2.82e-6 | −3.41e-6 |
| Block means over 4 cells, inside the support | −5.61e-5 | −4.67e-5 |
| Moved by +0.05 step "2782 → 2783" | −1.21e-6 | −1.70e-6 |
| Moved by +0.1 | −2.43e-6 | −2.77e-6 |
| Moved by +0.2 | −5.30e-6 | −4.89e-6 |
| Moved by +0.5 | −2.49e-5 | −2.91e-6 |
| Moved by −0.5 | −2.59e-5 | −2.15e-5 |

From 2783 a positive move goes beyond it and −0.5 is the midpoint between the records; from 2782, +0.5 is that
same midpoint and −0.5 goes away from 2783. Moved starts cooled from β = 1e7 instead of 1e4 end 6e-6 to 1.5e-3
below their start: at that β the chain cannot digest the move.

- **Keeping block masses changes nothing.** A σ = 0.3 kick ends 1.9e-5 to 2.2e-5 below its start whether the
  masses of 100-cell blocks, of 1,000-cell blocks or of no blocks are kept. The envelope comes back by itself:
  the L1 distance between the 1,024-cell block masses of a chain's end and those of solution 2782 is at most
  0.005 on every end inspected from that start, free kicks included.
- **No chain from 2782 ends above 2782, even when it starts part of the way to 2783.** The chain started at the
  midpoint ends 2.9e-6 below 2782, that is 2.6e-5 below 2783.
- **Chains started on the line between the records stay where they are put.** The midpoint chain ends at L1
  distance 0.128 from 2782 and 0.110 from 2783; the chain started 20 % of the way ends 0.042 from 2782. A random
  kick of the same size is pulled back (round 2). So the line between the two records is made of local maxima of
  almost equal height, 1.7e-6 to 4.9e-6 below 2782 on the first half (the reheat alone costs 1.35e-6), and it
  rises to 2783 on the second half.
- **Solution 2783 is close to the top of that line.** Going 5 % beyond it ends 3.3e-7 above the same reheat
  without the move (−1.21e-6 against −1.54e-6), 10 % beyond ends 8.8e-7 below it, 50 % beyond 2.3e-5 below. The
  gain of 3.3e-7 does not pay for the reheat (1.6e-6), and a move without a reheat cannot be digested.

## Round 4 — the kick suggested by a second model (verified)

Near the end of the time box the coordinator relayed five suggestions from a second model (gpt-6-astra), marked
as unverified. Plan `sqrtkick12`: 12 chains of 20,000 evaluations from solution 2783. The kick is additive in √f
on windows of H cells, zero cells included, and the mass of every window is restored:
f_i ← (√f_i + σ √μ_b ξ_i)², μ_b the mean value of the window. Final exact score minus the start:

| Kick width σ | Window H | Our cooling from β = 1e4, then polish | Suggested schedule β = 1e7, 1e8, 1e9, 1e10 |
|---|---|---|---|
| 0.15 | 16 | −5.53e-5 | −1.21e-2 |
| 0.15 | 64 | −5.95e-5 | −1.36e-2 |
| 0.15 | 128 | not run | −1.44e-2 |
| 0.35 | 16 | −3.74e-4 | not run |
| 0.35 | 32 | −3.80e-4 | not run |
| 0.35 | 64 | −4.09e-4 | −2.67e-2 |
| 0.35 | 128 | −4.15e-4 | not run |
| 0.70 | 16 | −8.13e-4 | not run |
| 0.70 | 64 | −9.49e-4 | not run |

- **This kick does worse than the multiplicative one**: 5.5e-5 to 9.5e-4 below the start, against 2.8e-6
  (σ = 0.1) and 1.9e-5 (σ = 0.3) for the multiplicative kick that keeps block masses. It opens the zero cells of
  every window that holds mass, and opened cells are what E1 and round 2 already found costly with f = w². The
  budget is also shorter here (20,000 evaluations against 30,000), so the two are not strictly comparable.
- **The suggested schedule cannot digest a kick**: a chain that starts at β = 1e7 stays 1.2e-2 to 2.7e-2 below
  its start. This agrees with rounds 2 and 3 (kicks and moved starts cooled from 1e6 or 1e7).
- The window size matters little (16 to 128 cells: within 10 % of each other at equal σ).

What was tried of the five suggestions:

| # | Suggestion | Tried | Outcome |
|---|---|---|---|
| 1 | Additive kick in √f on windows of 16 to 128 cells, zero cells included, window masses restored | Yes, 12 chains | Every chain ends below its start (table above) |
| 2 | Hold the envelope during the first part (block masses fixed through the parametrization, empty band and end block frozen, then released) | No | It needs a new parametrization and its gradient; there was no time to build and test it. What we measured instead: the envelope does not move in any chain (block-mass distance to the start at most 0.005), so we expect little from the constraint (hypothesis) |
| 3 | Schedule β = 1e7, 1e8, 1e9, 1e10 on 2,000 / 4,000 / 6,000 / 8,000 evaluations, L-BFGS history 20, scale reset at each stage, best exact score kept anywhere | Partly | The β schedule ran on 4 chains, with history 8 and the scale fixed at the chain's start: it fails on kicked starts. All chains of this work already keep the best exact point seen at every accepted step. History 20 was not tried |
| 4 | Log the two terms of 1 − C | Yes | The identity holds exactly; the gain between the records is all in the plateau term (next section) |
| 5 | Stationarity check with multipliers on the maximizing lags | Partly | Done with the softmax weights of the smoothed objective as multipliers, at polished points: 36,783 zero cells of solution 2783 have a positive reduced derivative at β = 1e9. Opening them loses (round 2). An exact fit of the multipliers was not done |

## Where the gain between the two records sits (verified, local numpy)

With z = c / max c (zeros outside the support), the identity
1 − C = [ Σ z (1 − z) + (1/6) Σ (z_{k+1} − z_k)² ] / Σ z holds exactly (checked to the last printed digit on
eight solutions). The first term is small when the autoconvolution is near 0 or near its ceiling, the second
when it is smooth. The coordinator relayed it from a second model.

| Solution | 1 − C | Plateau term | Jaggedness term | Lags within 1e-7 of the maximum |
|---|---|---|---|---|
| 2782 | 3.636133e-2 | 3.634441e-2 | 1.692e-5 | 384,530 |
| 2783 | 3.633844e-2 | 3.632015e-2 | 1.829e-5 | 373,674 |
| 2783 polished (best point here) | 3.633838e-2 | 3.632009e-2 | 1.829e-5 | 414,077 |
| 2782 after a reheat at β = 1e4 | 3.636268e-2 | 3.634576e-2 | 1.692e-5 | 305,027 |
| 2782 after a reheat at β = 1e3 | 3.638323e-2 | 3.636633e-2 | 1.689e-5 | 277,972 |
| 2782 kicked (σ = 0.3), cooled from 1e4, at β = 2e7 | 3.638300e-2 | 3.636807e-2 | 1.493e-5 | 288 |
| Midpoint of the records, cooled from 1e4 | 3.636424e-2 | 3.634723e-2 | 1.701e-5 | 292,366 |

- The jump from 2782 to 2783 is **all in the plateau term** (−2.43e-5) and costs 1.4e-6 of jaggedness. The
  jaggedness term is 0.05 % of 1 − C: there is little left to gain there.
- Every chain that ends below its start loses in the plateau term. The kicked chain is smoother than its start
  (jaggedness term −2.0e-6) and its plateau term is worse by 2.4e-5.

## All chains

"Lowest stage end" is the lowest exact score among the points stored at stage ends, so it shows how deep the
reheat went. "Final" is the point at the end of the schedule. Scores are the frozen verifier's, on CPU.
`frozen` = solution 2782 (0.9636386679928139 in the container), `live` = solution 2783 (0.9636615620620850).

Chain names: `bN` starts at β = 1eN; `-fixed` keeps the support, `-openN` floors the zero cells at 1e-N,
`-reopenη` opens cells by their derivative; `kσ` is a log-normal kick of width σ (`-blkM` keeps the mass of every
M cells, `-sK` is the seed); `xT` moves the start by T steps "2782 → 2783"; `cKm` takes block means over K cells
inside the support; `qσ-wH` is the additive kick in √f on windows of H cells (`-adv`: the suggested schedule
1e7 → 1e10). The "Kick" column is σ for both kinds of kick. The full specification of every chain is in its
run folder (`spec.json`, and `config` in each line of `chains.jsonl`).

| Run | Chain | Start | beta start -> end (stages) | Floor | Kick | Reopen | Evaluations | Lowest stage end vs start | Final exact score | Final vs start | Final vs live #1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `20261115T045124Z` | polish | live | 1e+09 -> 3e+09 (2) | 0 | 0 | 0 | 400 | +4.945e-08 | 0.9636616154097396 | +5.335e-08 | +5.335e-08 |
| `20261115T045124Z` | mild | live | 1e+07 -> 1e+09 (2) | 0 | 0 | 0 | 400 | -3.709e-07 | 0.9636615624938339 | +4.317e-10 | +4.317e-10 |
| `20261115T045632Z` | control | frozen | 1e+09 -> 1e+11 (5) | 0 | 0 | 0 | 30000 | +5.136e-08 | 0.9636387240428587 | +5.605e-08 | -2.284e-05 |
| `20261115T045632Z` | b7-fixed | frozen | 1e+07 -> 1e+11 (11) | 0 | 0 | 0 | 30000 | -3.514e-07 | 0.9636387240677677 | +5.607e-08 | -2.284e-05 |
| `20261115T045632Z` | b6-fixed | frozen | 1e+06 -> 1e+11 (14) | 0 | 0 | 0 | 30000 | -3.851e-06 | 0.9636387112088963 | +4.322e-08 | -2.285e-05 |
| `20261115T045632Z` | b5-fixed | frozen | 1e+05 -> 1e+11 (17) | 0 | 0 | 0 | 30000 | -3.826e-05 | 0.9636385959372054 | -7.206e-08 | -2.297e-05 |
| `20261115T045632Z` | b4-fixed | frozen | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -3.439e-04 | 0.9636373184977484 | -1.349e-06 | -2.424e-05 |
| `20261115T045632Z` | b3-fixed | frozen | 1e+03 -> 1e+11 (23) | 0 | 0 | 0 | 30000 | -2.302e-03 | 0.9636167739504681 | -2.189e-05 | -4.479e-05 |
| `20261115T045632Z` | b6-open3 | frozen | 1e+06 -> 1e+11 (14) | 0.001 | 0 | 0 | 30000 | -1.359e-03 | 0.9633469710605802 | -2.917e-04 | -3.146e-04 |
| `20261115T045632Z` | b5-open3 | frozen | 1e+05 -> 1e+11 (17) | 0.001 | 0 | 0 | 30000 | -1.198e-04 | 0.9636331782072820 | -5.490e-06 | -2.838e-05 |
| `20261115T045632Z` | b4-open3 | frozen | 1e+04 -> 1e+11 (20) | 0.001 | 0 | 0 | 30000 | -3.409e-04 | 0.9636373325476384 | -1.335e-06 | -2.423e-05 |
| `20261115T045632Z` | b3-open3 | frozen | 1e+03 -> 1e+11 (23) | 0.001 | 0 | 0 | 30000 | -2.295e-03 | 0.9636178287971071 | -2.084e-05 | -4.373e-05 |
| `20261115T045632Z` | b6-open6 | frozen | 1e+06 -> 1e+11 (14) | 1e-06 | 0 | 0 | 30000 | -4.056e-06 | 0.9636386523440273 | -1.565e-08 | -2.291e-05 |
| `20261115T045632Z` | b5-open6 | frozen | 1e+05 -> 1e+11 (17) | 1e-06 | 0 | 0 | 30000 | -3.831e-05 | 0.9636385927107084 | -7.528e-08 | -2.297e-05 |
| `20261115T045632Z` | b4-open6 | frozen | 1e+04 -> 1e+11 (20) | 1e-06 | 0 | 0 | 30000 | -3.416e-04 | 0.9636372826257050 | -1.385e-06 | -2.428e-05 |
| `20261115T045632Z` | b5-open2 | frozen | 1e+05 -> 1e+11 (17) | 0.01 | 0 | 0 | 30000 | -7.525e-03 | 0.9632805740546782 | -3.581e-04 | -3.810e-04 |
| `20261115T045632Z` | b5-open3-4cycles | frozen | 1e+05 -> 1e+11 (52) | 0.001 | 0 | 0 | 30000 | -1.295e-03 | 0.9636309084230563 | -7.760e-06 | -3.065e-05 |
| `20261115T045632Z` | b4-open3-4cycles | frozen | 1e+04 -> 1e+11 (60) | 0.001 | 0 | 0 | 30000 | -3.591e-04 | 0.9636326472339140 | -6.021e-06 | -2.891e-05 |
| `20261115T045645Z` | control | live | 1e+09 -> 1e+11 (5) | 0 | 0 | 0 | 30000 | +5.406e-08 | 0.9636616210549184 | +5.899e-08 | +5.899e-08 |
| `20261115T045645Z` | b7-fixed | live | 1e+07 -> 1e+11 (11) | 0 | 0 | 0 | 30000 | -3.807e-07 | 0.9636616207704790 | +5.871e-08 | +5.871e-08 |
| `20261115T045645Z` | b6-fixed | live | 1e+06 -> 1e+11 (14) | 0 | 0 | 0 | 30000 | -4.470e-06 | 0.9636616091816316 | +4.712e-08 | +4.712e-08 |
| `20261115T045645Z` | b5-fixed | live | 1e+05 -> 1e+11 (17) | 0 | 0 | 0 | 30000 | -4.443e-05 | 0.9636614710371169 | -9.102e-08 | -9.102e-08 |
| `20261115T045645Z` | b4-fixed | live | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -3.602e-04 | 0.9636600203788652 | -1.542e-06 | -1.542e-06 |
| `20261115T045645Z` | b3-fixed | live | 1e+03 -> 1e+11 (23) | 0 | 0 | 0 | 30000 | -2.276e-03 | 0.9636403207414553 | -2.124e-05 | -2.124e-05 |
| `20261115T045645Z` | b6-open3 | live | 1e+06 -> 1e+11 (14) | 0.001 | 0 | 0 | 30000 | -1.370e-03 | 0.9633515965434708 | -3.100e-04 | -3.100e-04 |
| `20261115T045645Z` | b5-open3 | live | 1e+05 -> 1e+11 (17) | 0.001 | 0 | 0 | 30000 | -1.464e-04 | 0.9636548265324779 | -6.736e-06 | -6.736e-06 |
| `20261115T045645Z` | b4-open3 | live | 1e+04 -> 1e+11 (20) | 0.001 | 0 | 0 | 30000 | -3.612e-04 | 0.9636601200588910 | -1.442e-06 | -1.442e-06 |
| `20261115T045645Z` | b3-open3 | live | 1e+03 -> 1e+11 (23) | 0.001 | 0 | 0 | 30000 | -2.270e-03 | 0.9636426670636111 | -1.889e-05 | -1.889e-05 |
| `20261115T045645Z` | b6-open6 | live | 1e+06 -> 1e+11 (14) | 1e-06 | 0 | 0 | 30000 | -4.670e-06 | 0.9636615508169477 | -1.125e-08 | -1.125e-08 |
| `20261115T045645Z` | b5-open6 | live | 1e+05 -> 1e+11 (17) | 1e-06 | 0 | 0 | 30000 | -4.424e-05 | 0.9636614924233886 | -6.964e-08 | -6.964e-08 |
| `20261115T045645Z` | b4-open6 | live | 1e+04 -> 1e+11 (20) | 1e-06 | 0 | 0 | 30000 | -3.607e-04 | 0.9636601347561392 | -1.427e-06 | -1.427e-06 |
| `20261115T045645Z` | b5-open2 | live | 1e+05 -> 1e+11 (17) | 0.01 | 0 | 0 | 30000 | -7.602e-03 | 0.9632820585117355 | -3.795e-04 | -3.795e-04 |
| `20261115T045645Z` | b5-open3-4cycles | live | 1e+05 -> 1e+11 (52) | 0.001 | 0 | 0 | 30000 | -1.190e-03 | 0.9636540937311685 | -7.468e-06 | -7.468e-06 |
| `20261115T045645Z` | b4-open3-4cycles | live | 1e+04 -> 1e+11 (60) | 0.001 | 0 | 0 | 30000 | -3.970e-04 | 0.9636549189148506 | -6.643e-06 | -6.643e-06 |
| `20261115T051619Z` | control | live | 1e+09 -> 1e+11 (5) | 0 | 0 | 0 | 30000 | +5.406e-08 | 0.9636616210549184 | +5.899e-08 | +5.899e-08 |
| `20261115T051619Z` | b7-reopen0.0001 | live | 1e+07 -> 1e+11 (11) | 0 | 0 | 0.0001 | 30000 | -3.081e-06 | 0.9636593792358931 | -2.183e-06 | -2.183e-06 |
| `20261115T051619Z` | b7-reopen0.001 | live | 1e+07 -> 1e+11 (11) | 0 | 0 | 0.001 | 30000 | -2.861e-04 | 0.9634472464905306 | -2.143e-04 | -2.143e-04 |
| `20261115T051619Z` | b7-reopen0.01 | live | 1e+07 -> 1e+11 (11) | 0 | 0 | 0.01 | 30000 | -6.032e-03 | 0.9586415188309043 | -5.020e-03 | -5.020e-03 |
| `20261115T051619Z` | b6-reopen0.0001 | live | 1e+06 -> 1e+11 (14) | 0 | 0 | 0.0001 | 30000 | -3.942e-05 | 0.9636413397159977 | -2.022e-05 | -2.022e-05 |
| `20261115T051619Z` | b6-reopen0.001 | live | 1e+06 -> 1e+11 (14) | 0 | 0 | 0.001 | 30000 | -3.690e-04 | 0.9633838759604733 | -2.777e-04 | -2.777e-04 |
| `20261115T051619Z` | b6-reopen0.01 | live | 1e+06 -> 1e+11 (14) | 0 | 0 | 0.01 | 30000 | -1.757e-03 | 0.9622626439000300 | -1.399e-03 | -1.399e-03 |
| `20261115T051619Z` | b5-reopen0.0001 | live | 1e+05 -> 1e+11 (17) | 0 | 0 | 0.0001 | 30000 | -4.435e-05 | 0.9636613741817647 | -1.879e-07 | -1.879e-07 |
| `20261115T051619Z` | b5-reopen0.001 | live | 1e+05 -> 1e+11 (17) | 0 | 0 | 0.001 | 30000 | -4.672e-05 | 0.9636607934088194 | -7.687e-07 | -7.687e-07 |
| `20261115T051619Z` | b5-reopen0.01 | live | 1e+05 -> 1e+11 (17) | 0 | 0 | 0.01 | 30000 | -4.729e-03 | 0.9595318732925487 | -4.130e-03 | -4.130e-03 |
| `20261115T051619Z` | b4-reopen0.001 | live | 1e+04 -> 1e+11 (20) | 0 | 0 | 0.001 | 30000 | -3.630e-04 | 0.9636595088430696 | -2.053e-06 | -2.053e-06 |
| `20261115T051619Z` | b4-reopen0.01 | live | 1e+04 -> 1e+11 (20) | 0 | 0 | 0.01 | 30000 | -5.261e-03 | 0.9591106847072890 | -4.551e-03 | -4.551e-03 |
| `20261115T051619Z` | k0.03-b5-s1 | live | 1e+05 -> 1e+11 (17) | 0 | 0.03 | 0 | 30000 | -7.357e-04 | 0.9635434365662917 | -1.181e-04 | -1.181e-04 |
| `20261115T051619Z` | k0.1-b5-s1 | live | 1e+05 -> 1e+11 (17) | 0 | 0.1 | 0 | 30000 | -1.313e-03 | 0.9635280680124766 | -1.335e-04 | -1.335e-04 |
| `20261115T051619Z` | k0.3-b5-s1 | live | 1e+05 -> 1e+11 (17) | 0 | 0.3 | 0 | 30000 | -4.432e-03 | 0.9631136883753209 | -5.479e-04 | -5.479e-04 |
| `20261115T051619Z` | k0.1-b5-s1-reopen0.001 | live | 1e+05 -> 1e+11 (17) | 0 | 0.1 | 0.001 | 30000 | -1.508e-03 | 0.9635046308490931 | -1.569e-04 | -1.569e-04 |
| `20261115T051632Z` | control | frozen | 1e+09 -> 1e+11 (5) | 0 | 0 | 0 | 30000 | +5.136e-08 | 0.9636387240428587 | +5.605e-08 | -2.284e-05 |
| `20261115T051632Z` | k0.03-b5-s1 | frozen | 1e+05 -> 1e+11 (17) | 0 | 0.03 | 0 | 30000 | -6.759e-04 | 0.9635217467717215 | -1.169e-04 | -1.398e-04 |
| `20261115T051632Z` | k0.1-b5-s1 | frozen | 1e+05 -> 1e+11 (17) | 0 | 0.1 | 0 | 30000 | -1.148e-03 | 0.9634669754466021 | -1.717e-04 | -1.946e-04 |
| `20261115T051632Z` | k0.1-b5-s2 | frozen | 1e+05 -> 1e+11 (17) | 0 | 0.1 | 0 | 30000 | -9.364e-04 | 0.9634837365703114 | -1.549e-04 | -1.778e-04 |
| `20261115T051632Z` | k0.1-b5-s3 | frozen | 1e+05 -> 1e+11 (17) | 0 | 0.1 | 0 | 30000 | -5.745e-04 | 0.9636135603589234 | -2.511e-05 | -4.800e-05 |
| `20261115T051632Z` | k0.3-b5-s1 | frozen | 1e+05 -> 1e+11 (17) | 0 | 0.3 | 0 | 30000 | -4.107e-03 | 0.9631380124383819 | -5.007e-04 | -5.235e-04 |
| `20261115T051632Z` | k0.3-b5-s2 | frozen | 1e+05 -> 1e+11 (17) | 0 | 0.3 | 0 | 30000 | -4.343e-03 | 0.9631499681981687 | -4.887e-04 | -5.116e-04 |
| `20261115T051632Z` | k0.3-b5-s3 | frozen | 1e+05 -> 1e+11 (17) | 0 | 0.3 | 0 | 30000 | -8.318e-04 | 0.9635620937114924 | -7.657e-05 | -9.947e-05 |
| `20261115T051632Z` | k1-b5-s1 | frozen | 1e+05 -> 1e+11 (17) | 0 | 1 | 0 | 30000 | -3.100e-02 | 0.9602792010794957 | -3.359e-03 | -3.382e-03 |
| `20261115T051632Z` | k1-b5-s2 | frozen | 1e+05 -> 1e+11 (17) | 0 | 1 | 0 | 30000 | -1.836e-02 | 0.9622651419685506 | -1.374e-03 | -1.396e-03 |
| `20261115T051632Z` | k0.3-b4-s1 | frozen | 1e+04 -> 1e+11 (20) | 0 | 0.3 | 0 | 30000 | -5.442e-04 | 0.9636172826474032 | -2.139e-05 | -4.428e-05 |
| `20261115T051632Z` | k0.3-b4-s2 | frozen | 1e+04 -> 1e+11 (20) | 0 | 0.3 | 0 | 30000 | -5.405e-04 | 0.9636188881760632 | -1.978e-05 | -4.267e-05 |
| `20261115T051632Z` | k0.3-b6-s1 | frozen | 1e+06 -> 1e+11 (14) | 0 | 0.3 | 0 | 30000 | -1.253e-02 | 0.9567027902853857 | -6.936e-03 | -6.959e-03 |
| `20261115T051632Z` | k0.3-b6-s2 | frozen | 1e+06 -> 1e+11 (14) | 0 | 0.3 | 0 | 30000 | -1.253e-02 | 0.9569237365102949 | -6.715e-03 | -6.738e-03 |
| `20261115T051632Z` | k0.1-b5-s4-open3 | frozen | 1e+05 -> 1e+11 (17) | 0.001 | 0.1 | 0 | 30000 | -1.481e-03 | 0.9634266242849112 | -2.120e-04 | -2.349e-04 |
| `20261115T051632Z` | k0.3-b5-s4-open3 | frozen | 1e+05 -> 1e+11 (17) | 0.001 | 0.3 | 0 | 30000 | -2.847e-03 | 0.9633570760944672 | -2.816e-04 | -3.045e-04 |
| `20261115T054039Z` | control | live | 1e+09 -> 1e+11 (5) | 0 | 0 | 0 | 30000 | +5.406e-08 | 0.9636616210549184 | +5.899e-08 | +5.899e-08 |
| `20261115T054039Z` | x0.02-b7 | live | 1e+07 -> 1e+11 (11) | 0 | 0 | 0 | 30000 | -3.497e-04 | 0.9635069718521816 | -1.546e-04 | -1.546e-04 |
| `20261115T054039Z` | x0.05-b7 | live | 1e+07 -> 1e+11 (11) | 0 | 0 | 0 | 30000 | -1.558e-03 | 0.9626603552638219 | -1.001e-03 | -1.001e-03 |
| `20261115T054039Z` | x0.1-b7 | live | 1e+07 -> 1e+11 (11) | 0 | 0 | 0 | 30000 | -1.874e-03 | 0.9624965722462610 | -1.165e-03 | -1.165e-03 |
| `20261115T054039Z` | x0.2-b7 | live | 1e+07 -> 1e+11 (11) | 0 | 0 | 0 | 30000 | -2.466e-03 | 0.9621274323574859 | -1.534e-03 | -1.534e-03 |
| `20261115T054039Z` | x0.05-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -3.587e-04 | 0.9636603537676949 | -1.208e-06 | -1.208e-06 |
| `20261115T054039Z` | x0.1-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -3.653e-04 | 0.9636591361722593 | -2.426e-06 | -2.426e-06 |
| `20261115T054039Z` | x0.2-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -3.729e-04 | 0.9636562577576823 | -5.304e-06 | -5.304e-06 |
| `20261115T054039Z` | x0.5-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -4.363e-04 | 0.9636366946382890 | -2.487e-05 | -2.487e-05 |
| `20261115T054039Z` | x-0.1-b7 | live | 1e+07 -> 1e+11 (11) | 0 | 0 | 0 | 30000 | -2.255e-04 | 0.9635664922629384 | -9.507e-05 | -9.507e-05 |
| `20261115T054039Z` | x-0.5-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -3.859e-04 | 0.9636356290334231 | -2.593e-05 | -2.593e-05 |
| `20261115T054039Z` | k0.3-blk100-b4-s1 | live | 1e+04 -> 1e+11 (20) | 0 | 0.3 | 0 | 30000 | -5.780e-04 | 0.9636423004381401 | -1.926e-05 | -1.926e-05 |
| `20261115T054039Z` | k0.3-blk100-b4-s2 | live | 1e+04 -> 1e+11 (20) | 0 | 0.3 | 0 | 30000 | -5.685e-04 | 0.9636418789506843 | -1.968e-05 | -1.968e-05 |
| `20261115T054039Z` | k0.3-blk1000-b4-s1 | live | 1e+04 -> 1e+11 (20) | 0 | 0.3 | 0 | 30000 | -5.692e-04 | 0.9636422921224304 | -1.927e-05 | -1.927e-05 |
| `20261115T054039Z` | k0.1-blk100-b4-s1 | live | 1e+04 -> 1e+11 (20) | 0 | 0.1 | 0 | 30000 | -4.065e-04 | 0.9636587460095334 | -2.816e-06 | -2.816e-06 |
| `20261115T054039Z` | c4m-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -5.289e-04 | 0.9636054894221159 | -5.607e-05 | -5.607e-05 |
| `20261115T054109Z` | control | frozen | 1e+09 -> 1e+11 (5) | 0 | 0 | 0 | 30000 | +5.136e-08 | 0.9636387240428587 | +5.605e-08 | -2.284e-05 |
| `20261115T054109Z` | x0.02-b7 | frozen | 1e+07 -> 1e+11 (11) | 0 | 0 | 0 | 30000 | -2.692e-05 | 0.9636329236976166 | -5.744e-06 | -2.864e-05 |
| `20261115T054109Z` | x0.05-b7 | frozen | 1e+07 -> 1e+11 (11) | 0 | 0 | 0 | 30000 | -7.286e-05 | 0.9636093908059671 | -2.928e-05 | -5.217e-05 |
| `20261115T054109Z` | x0.1-b7 | frozen | 1e+07 -> 1e+11 (11) | 0 | 0 | 0 | 30000 | -2.175e-04 | 0.9635500486413552 | -8.862e-05 | -1.115e-04 |
| `20261115T054109Z` | x0.2-b7 | frozen | 1e+07 -> 1e+11 (11) | 0 | 0 | 0 | 30000 | -5.715e-04 | 0.9633533871994182 | -2.853e-04 | -3.082e-04 |
| `20261115T054109Z` | x0.05-b4 | frozen | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -3.425e-04 | 0.9636369655323588 | -1.702e-06 | -2.460e-05 |
| `20261115T054109Z` | x0.1-b4 | frozen | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -3.443e-04 | 0.9636359027894024 | -2.765e-06 | -2.566e-05 |
| `20261115T054109Z` | x0.2-b4 | frozen | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -3.493e-04 | 0.9636337750150373 | -4.893e-06 | -2.779e-05 |
| `20261115T054109Z` | x0.5-b4 | frozen | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -3.572e-04 | 0.9636357586275861 | -2.909e-06 | -2.580e-05 |
| `20261115T054109Z` | x-0.1-b7 | frozen | 1e+07 -> 1e+11 (11) | 0 | 0 | 0 | 30000 | -4.407e-04 | 0.9634843677597876 | -1.543e-04 | -1.772e-04 |
| `20261115T054109Z` | x-0.5-b4 | frozen | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -3.843e-04 | 0.9636171448891067 | -2.152e-05 | -4.442e-05 |
| `20261115T054109Z` | k0.3-blk100-b4-s1 | frozen | 1e+04 -> 1e+11 (20) | 0 | 0.3 | 0 | 30000 | -5.276e-04 | 0.9636191446358571 | -1.952e-05 | -4.242e-05 |
| `20261115T054109Z` | k0.3-blk100-b4-s2 | frozen | 1e+04 -> 1e+11 (20) | 0 | 0.3 | 0 | 30000 | -5.519e-04 | 0.9636163198578651 | -2.235e-05 | -4.524e-05 |
| `20261115T054109Z` | k0.3-blk1000-b4-s1 | frozen | 1e+04 -> 1e+11 (20) | 0 | 0.3 | 0 | 30000 | -5.582e-04 | 0.9636170873091221 | -2.158e-05 | -4.447e-05 |
| `20261115T054109Z` | k0.1-blk100-b4-s1 | frozen | 1e+04 -> 1e+11 (20) | 0 | 0.1 | 0 | 30000 | -3.900e-04 | 0.9636352589804619 | -3.409e-06 | -2.630e-05 |
| `20261115T054109Z` | c4m-b4 | frozen | 1e+04 -> 1e+11 (20) | 0 | 0 | 0 | 30000 | -5.702e-04 | 0.9635919290763237 | -4.674e-05 | -6.963e-05 |
| `20261115T060342Z` | q0.15-w16-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0.15 | 0 | 20000 | -8.216e-04 | 0.9636062888374070 | -5.527e-05 | -5.527e-05 |
| `20261115T060342Z` | q0.35-w16-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0.35 | 0 | 20000 | -2.138e-03 | 0.9632875059116439 | -3.741e-04 | -3.741e-04 |
| `20261115T060342Z` | q0.7-w16-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0.7 | 0 | 20000 | -4.411e-03 | 0.9628486671897347 | -8.129e-04 | -8.129e-04 |
| `20261115T060342Z` | q0.15-w64-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0.15 | 0 | 20000 | -8.490e-04 | 0.9636020833334956 | -5.948e-05 | -5.948e-05 |
| `20261115T060342Z` | q0.35-w64-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0.35 | 0 | 20000 | -2.592e-03 | 0.9632525611790811 | -4.090e-04 | -4.090e-04 |
| `20261115T060342Z` | q0.7-w64-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0.7 | 0 | 20000 | -5.524e-03 | 0.9627130867988349 | -9.485e-04 | -9.485e-04 |
| `20261115T060342Z` | q0.35-w32-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0.35 | 0 | 20000 | -2.266e-03 | 0.9632816886418077 | -3.799e-04 | -3.799e-04 |
| `20261115T060342Z` | q0.35-w128-b4 | live | 1e+04 -> 1e+11 (20) | 0 | 0.35 | 0 | 20000 | -2.682e-03 | 0.9632465548169711 | -4.150e-04 | -4.150e-04 |
| `20261115T060342Z` | q0.15-w64-adv | live | 1e+07 -> 1e+10 (4) | 0 | 0.15 | 0 | 20000 | -1.439e-02 | 0.9500219492529620 | -1.364e-02 | -1.364e-02 |
| `20261115T060342Z` | q0.35-w64-adv | live | 1e+07 -> 1e+10 (4) | 0 | 0.35 | 0 | 20000 | -2.795e-02 | 0.9369211814541303 | -2.674e-02 | -2.674e-02 |
| `20261115T060342Z` | q0.15-w16-adv | live | 1e+07 -> 1e+10 (4) | 0 | 0.15 | 0 | 20000 | -1.250e-02 | 0.9516160309220182 | -1.205e-02 | -1.205e-02 |
| `20261115T060342Z` | q0.15-w128-adv | live | 1e+07 -> 1e+10 (4) | 0 | 0.15 | 0 | 20000 | -1.507e-02 | 0.9492625252366103 | -1.440e-02 | -1.440e-02 |

## What solutions 2782 and 2783 tell us (verified, local numpy)

- **The LogSumExp temperature does not connect the two basins.** On the straight segment between them (in
  w = √f, both normalised to unit mass) the exact score falls from 0.96364 to 0.93541 at the midpoint, and the
  smoothed objective keeps the same barrier at every β: 0.95070 → 0.92627 → 0.95072 at β = 1e3, 0.96237 →
  0.93491 → 0.96239 at β = 1e4. Solution 2783 is above 2782 by 1.8e-5 to 2.3e-5 at every β from 1e3 to 1e9.
- **They have the same envelope and differ like cell-level noise.** Mass-normalised L1 distance by block size:

  | Block (cells) | 1 | 4 | 16 | 64 | 256 | 1,024 | 4,096 |
  |---|---|---|---|---|---|---|---|
  | L1 distance | 0.225 | 0.145 | 0.089 | 0.046 | 0.021 | 0.008 | 0.003 |

  The distance shrinks by a factor 0.4 to 0.65 each time the block is four times larger (independent cell-level
  noise would give 0.5). The log-ratio of adjacent common cells has correlation 0.20.
- **The support change is small in mass.** 23,191 cells appear and 25,554 vanish, but they hold 0.77 % and
  0.70 % of the mass. Of the 20,453 runs of 2782, 1,769 are identical in 2783, 15,301 have a moved edge and 3,383
  have no overlap with any run of 2783.
- **A fixed-support polish reaches the exact local maximum.** The polished score of the control chains rises by
  3.4e-9, 1.1e-9, 3.4e-10 and 1.0e-10 over the four last stages (β × 3.16 each): a geometric series, the
  LogSumExp bias going away. Its limit is the start plus 5.90e-8 for solution 2783 and plus 5.61e-8 for
  solution 2782. The "+5.4e-8, then saturation" of v1 is this local maximum, not a stalled optimizer.
- **Zero cells and first-order conditions.** At a point polished at β = 1e9, 36,783 of the 1,517,283 zero cells
  of solution 2783 have a positive derivative of the smoothed objective (14,479 at a point polished at β = 1e5).
  Opening them at β = 1e9 costs exact score: 2e-5 for the strongest cell at 1e-3 of the mean cell value, and the
  polish gets back about 1e-6 per 50 evaluations. The smoothed derivative is valid over relative steps of about
  1/β, so it is not a usable direction for the exact score.

## Arbiter verdicts

`selection/verify_claim.py`, run locally on the downloaded solutions (pinned verifier in a clean process, the
independent exact verifier, a fresh read of the live #1). The lines are in `selection/claims.jsonl`
(`claimed_by: ws2-v2-anneal-gpu`). Nothing was submitted.

| Candidate (id on the Volume) | Chain | Verdict | Pinned score | Independent score | `beats_record_by` |
|---|---|---|---|---|---|
| `20261115T045645Z-e2-live-69b1817/control/best` | polish only, from solution 2783 | `VALID_BELOW_RECORD` | 0.9636616210553628 | 0.9636616210553631 | 5.899e-08 (needs ≥ 1.0e-05) |
| `20261115T045645Z-e2-live-69b1817/b7-fixed/best` | reheat at β = 1e7, from solution 2783 | `VALID_BELOW_RECORD` | 0.9636616207710272 | 0.9636616207710269 | 5.871e-08 (needs ≥ 1.0e-05) |

Both verdicts read the live #1 as 0.9636615620620852 by MLAI_YONSEI, δ = 1e-05, target 0.9636715620620852; the
gap between the two verifiers is 3.3e-16 (tolerance 1e-12). The first candidate is 4.9e-9 above the best point
of Knockout Evolution v1 (0.9636616161700385).

No verdict is `beats_record`. No record is claimed.

## Modal spend (estimate)

Budget: 35 USD. The Modal bill was not read; this is wall time × the list prices above (0.001247 USD/s for the
GPU container). "Billed (estimate)" is the time the worker ran plus about 65 s to start and to commit the
Volume, the overhead measured on E1 and E2, which started at once. "Upper bound" is the whole life of the app,
which includes the wait for an H100 (not billed, as far as we know).

| Launch | Chains × evaluations | Worker wall | Billed (estimate) | Upper bound |
|---|---|---|---|---|
| `prepare` (CPU, 2 cores) | — | 70 s | 0.01 USD | 0.01 USD |
| E0 bench | — | under 30 s | 0.04 USD | 0.04 USD |
| Smoke run | 2 × 400 | 18 s | 0.10 USD | 0.32 USD |
| E1 | 16 × 30,000 | 1,015 s | 1.35 USD | 1.35 USD |
| E2 | 16 × 30,000 | 1,025 s | 1.35 USD | 1.35 USD |
| Round 2, reopen | 16 × 30,000 | 1,022 s | 1.36 USD | 1.80 USD |
| Round 2, kicks | 16 × 30,000 | 1,017 s | 1.35 USD | 1.80 USD |
| Round 3, from solution 2783 | 16 × 30,000 | 1,017 s | 1.35 USD | 1.35 USD |
| Round 3, from solution 2782 | 16 × 30,000 | 1,020 s | 1.35 USD | 1.35 USD |
| Round 4 | 12 × 20,000 | 524 s | 0.68 USD | 0.68 USD |
| **Total** | | | **8.9 USD** | **10.0 USD** |

No app is left running.

## Verified, hypothesis, limitation

**Verified** (measured here, numbers in this file and in the run folders)

- The E0 numbers, on one H100, in float64.
- Every score of the tables: the frozen verifier on CPU inside the container (numpy 2.5.4, scipy 1.18.1). The GPU
  exact score agrees with it within 2e-15 on all 1,903 stored points, none of which the verifier
  rejected. The two best candidates were re-verified locally by the arbiter.
- No reheated, floored, reopened, kicked or moved chain ends more than 3e-11 above the polish-only chain from
  the same start (E1's reheat at β = 1e7 ends 2.5e-11 above it; every other chain ends below it). The best
  exact score is the polish of solution 2783: 0.9636616210553628, 5.899e-8 above the live #1, where a record
  needs 1e-5.
- The deeper the reheat, the lower the end, in the same way from both starts.
- The polish limit of the two basins (start + 5.90e-8 and start + 5.61e-8).
- The control chain gave the same final score to the last digit in three runs from each start,
  although GPU kernels are not guaranteed to be deterministic.
- The comparison of solutions 2782 and 2783 (same envelope, cell-level differences, small support change, a
  barrier at every β). The coordinator measured the block-size comparison independently with other block sizes
  (0.0086 at 1,000 cells, 0.036 at 100, 0.081 at 20); the two sets of numbers agree.

**Hypothesis** (not tested, or tested on too few points)

- Around a record, the end score falls in proportion to the distance at which the chain stops (four points).
- Solution 2783 was not reached from 2782 by a local move of the families tried here. Its fine pattern was
  probably derived again from a coarser description or by a long independent run. We have no evidence of the
  other team's method.
- The losses of the floored chains are leftover mass (one chain inspected).
- More evaluations per chain would shrink the deficit of the kicked chains without changing its sign: no chain
  was run beyond 30,000 evaluations.

**Limitation**

- f = w² only. A zero cell cannot open by itself and a small cell moves slowly. A bound-constrained optimizer on
  f was not tried. The reopening rule uses the derivative of the smoothed objective, which is not a valid
  direction for the exact score at high β.
- One cooling speed (3 stages per decade, 30,000 evaluations). One run per schedule; the two starts act as the
  replication. Kicks: at most 3 seeds per setting.
- The spend is an estimate, not the Modal bill.
- The basin-hopping job and three plans were written and never run.
- Source commits of the runs, all on this branch: `5e16a2a` (smoke), `69b1817` (E1, E2), `bf01439` (round 2),
  `e25ca83` (round 3). Round 4 ran from the working tree on top of `e25ca83` (logged as `e25ca83-wip`); that
  tree is the commit that adds this file, and it differs from `e25ca83` by the √f kick and its plan only.
  The Volume `knockout-v2-gpu` holds every stored solution (about 5.5 GB per 16-chain run).

## Reproduce

```bash
PY=.venv/bin/python; MODAL=.venv/bin/modal
$PY -m pytest tests/test_knockout_v2_gpu.py -q                                    # 13 tests, no torch
$MODAL run -m knockout.v2.gpu.modal_app::launch --job prepare                     # warm starts to the Volume
$MODAL run -m knockout.v2.gpu.modal_app::launch --job bench                       # E0
$MODAL run --detach -m knockout.v2.gpu.modal_app::launch --job anneal --plan deep16 --start frozen \
    --evals 30000 --s-per-eval 0.034 --max-usd 4 --source-commit $(git rev-parse --short HEAD) --tag e1
$MODAL volume get knockout-v2-gpu <run_id>/summary.json knockout/v2/results/anneal_gpu/<run_id>/summary.json
$PY -m knockout.v2.gpu.report knockout/v2/results/anneal_gpu                      # the chain table
$MODAL volume get knockout-v2-gpu <run_id>/<chain>/best.npy runs/anneal_gpu/best.npy
$PY -m knockout.v2.gpu.export runs/anneal_gpu/best.npy runs/anneal_gpu/best.json
$PY selection/verify_claim.py second-autocorrelation-inequality runs/anneal_gpu/best.json
```

`--max-usd` is a hard wall-clock limit inside the container (the chains stop and store where they are). Each
`modal run` is its own app, so the cap of 2 GPU containers holds per launch: do not start a third run while two
are active. GPU kernels are not bit-deterministic: a result is reproduced from its stored solution, not by
re-running the seed.

## What to try next

1. **Stop perturbing the incumbent at random with this annealer.** β reheats, opened cells and random kicks,
   about 90 chains from two records, give the same picture: the chain is pulled back towards the record it
   started from and ends below it by an amount that grows with the distance left.
2. **Search along differences of good solutions** (hypothesis, the one positive hint of this work). The line
   between the two records is the only direction in which a chain stays where it is put and loses almost
   nothing, and solution 2783 sits within 5 % of its top. With one such direction there is nothing more to gain;
   with several good solutions (from the coarse-to-fine resynthesis, or from other teams' entries on the
   leaderboard) each pair gives a line, and one annealed chain per point of the line costs 0.08 USD.
   `ChainSpec.along` and `starts.along_records` do it for the two stored records; they need a small change to
   take any two stored solutions. A milder reheat than β = 1e4 has to be found first, since that reheat alone
   costs 1.6e-6 (cooling from 1e5 costs 1.5e-7 and from 1e6 1.2e-8, but moved starts were only tried from 1e4
   and 1e7).
3. **Use the GPU as the polisher of candidates that come from elsewhere.** A full anneal of one 2M-cell candidate
   (30,000 evaluations) costs about 0.08 USD in a 16-chain batch and 17 minutes. `--start vol:<path>.npy` takes
   any stored solution of the same length, so the coarse-to-fine resynthesis (another branch, built on this one)
   can hand its candidates to `gpu_job` and get exact scores back. That is breadth over starting structures, not
   depth from one start.
4. **Replace f = w² by a bound-constrained optimizer on f** (projected L-BFGS), so that cells open and close at
   their natural rate. Every chain that opened cells here paid for leftover mass.
5. **Slower cooling for kicked chains, only if points 2 and 3 give nothing.** The kicked chains cooled from
   β = 1e4 were still 2e-5 below their start after 30,000 evaluations; 8 chains of 60,000 evaluations cost the
   same wall time as 16 of 30,000 (`blockkick8`, not run). Hypothesis: the deficit shrinks, its sign does not
   change.
6. **Add GPU fallbacks.** `gpu=["H100", "A100-80GB"]` would have avoided the 4 to 7 minute waits; float64
   throughput on the A100 has to be measured first (WS3 rule 11).
