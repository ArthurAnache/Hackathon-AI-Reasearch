# Run report — 20261115T021632Z-second-autocorrelation-inequality-26d8f15

## Summary

- Problem: `second-autocorrelation-inequality`; editor: `openai:gpt-6.1-sol`; stop reason: budget.
- Record to beat (read at run time): 0.963661562062. Warm start: 0.963661562062.
- Best verified score of the run: **0.963661613998** (arm `main`, experiment `exp_000035`, solution `solutions/main/exp_000035.json`).
- Arm `control`: best 0.963661613713, 2 steps, spend 0.606434565058.
- Arm `main`: best 0.963661613998, 2 steps, spend 1.40387871976.
- Budget: 856.0 s of 900.0 s; 2.0103 USD of 5.0 USD (estimates: Modal compute + LLM).
- Promotion gate: no `record_claim`. **This run does not beat the record.**
- Experiments by arm and status: control/discard: 14, control/keep: 4, main/discard: 23, main/keep: 15
- LLM calls: 52 (0 failed), 1.914 USD estimated; by role: program: 30, program_edit: 16, program_reformulate: 2, review_diff: 2, seed_genes: 1, strategy_patch: 1
- Programs rejected before evaluation: near duplicate: 4

## Control arm versus main arm at equal cumulative spend

Spend unit: USD. Each cell is the best verified score the arm had reached at that spend.

| Spend | Control (Hill Sampling) | Main (genes) | Main − control |
|---|---|---|---|
| 0.0758 | 0.963661611733 | 0.963661611666 | -6.696e-11 |
| 0.1516 | 0.963661611733 | 0.963661611666 | -6.696e-11 |
| 0.2274 | 0.963661611733 | 0.963661611666 | -6.696e-11 |
| 0.3032 | 0.963661611733 | 0.963661611666 | -6.696e-11 |
| 0.379 | 0.96366161327 | 0.963661611666 | -1.604e-09 |
| 0.4548 | 0.96366161327 | 0.963661611666 | -1.604e-09 |
| 0.5306 | 0.96366161327 | 0.963661611666 | -1.604e-09 |
| 0.6064 | 0.963661613713 | 0.963661611666 | -2.048e-09 |

Control spent 0.6064 in total and the main arm 1.404; the table stops at the smaller of the two.

## Genealogy of the best candidate of the main arm

0. `m00004` (island I1, lineage L001, operator `refine`, model `gpt-6.1-sol`), first score 0.963661612639 in `exp_000015`: Increasing the initial smoothing sharpness from 1e9 to 1e11 will reduce plateau-induced smoothing bias and produce a positive exact-score gain on the order of 1e-9 or larger.
  - `g0003` (numeric, refuted, posterior mean 0.29): Use constraint generation for minimax subproblems: initially include convolution lags within 1e-7 of the maximum, solve the restricted problem, then scan every lag using FFT convolution. Add violated constraints and repeat. Retain previously binding lags across iterations to prevent cycling between competing peaks. — gate: 0 wins, 3 losses in 3 pairs, p = 0.062 → refuted
1. `m00001` (island I1, lineage L001, operator `seed`, model `None`), first score 0.963661611666 in `exp_000002`: baseline: the seed program as written
  - no gene active at this step

## Gene table

| Gene | Island | Type | Status | Origin | Used | Posterior mean | Text |
|---|---|---|---|---|---|---|---|
| g0001 | I1 | formulation | validated | seed | 0 | 0.50 | Canonical formulation: `construct` returns the candidate directly in the verifier's own format. |
| g0002 | I1 | formulation | candidate | seed | 0 | 0.50 | Replace smooth-max ascent with sequential minimax optimization. Set c=f*f and Q=(2 sum c²+sum adjacent products)/3. Linearize log Q−2 log(sum f)−log t, enforce c+2(f*δf)≤t and f+δf≥0, and impose a trust region. Accept steps only when the exact verifier score improves. |
| g0003 | I1 | numeric | refuted | seed | 0 | 0.29 | Use constraint generation for minimax subproblems: initially include convolution lags within 1e-7 of the maximum, solve the restricted problem, then scan every lag using FFT convolution. Add violated constraints and repeat. Retain previously binding lags across iterations to prevent cycling between competing peaks. |
| g0004 | I1 | structure | candidate | seed | 0 | 0.50 | Optimize one multiplicative weight per existing nonzero run before optimizing individual cells. Preserve each run's internal profile and all zero gaps. Compute run-weight derivatives by summing cell derivatives weighted by the incumbent values, and periodically release only runs showing the largest predicted gains. |
| g0005 | I1 | formulation | candidate | seed | 0 | 0.50 | Optimize selected amplitudes directly with nonnegative bounds instead of squaring unconstrained variables. Include currently zero cells adjacent to run boundaries, whose derivatives vanish under the squared parameterization. Restrict newly activated mass with a small aggregate budget, then increase that budget only after exact-score improvements. |
| g0006 | I1 | numeric | candidate | seed | 0 | 0.50 | Choose LogSumExp sharpness from its observed approximation error rather than a fixed schedule. After normalizing the convolution maximum to one, increase sharpness until log(sum exp(beta*(c−max c)))/beta is below 1e-8. Compare exact scores after every accepted iteration and restart from the best exact point after deterioration. |
| g0007 | I1 | structure | candidate | seed | 0 | 0.50 | Search coordinated packet translations rather than isolated support moves. Select two to eight nonzero runs, propose opposing shifts that preserve their internal amplitudes and avoid overlap, and jointly reoptimize their weights. Rank proposals by predicted changes at near-maximal convolution lags before spending an exact FFT evaluation. |
| g0008 | I1 | structure | candidate | seed | 0 | 0.50 | Build a multiresolution amplitude basis within each nonzero run: begin with one scaling coefficient, then add zero-sum left-versus-right contrasts recursively. Refine runs with the largest projected ascent derivatives. Preserve the original grid and zero gaps throughout; do not interpolate or resample the incumbent. |
| g0009 | I1 | numeric | candidate | seed | 0 | 0.50 | Apply Hessian-vector products to the active-constraint tangent space to seek collective escape directions. Approximate products by gradient differences, constrain first-order changes of binding convolution peaks, and test both signs of leading positive-curvature directions. Use geometrically spaced step sizes and retain only exact-score gains. |
| g0010 | I1 | formulation | candidate | seed | 0 | 0.50 | Estimate nonnegative multipliers on near-maximal convolution lags by minimizing the projected stationarity residual. Use the resulting Lagrangian gradient to rank inactive cells and entire runs, rather than using one arbitrary maximum lag. Test mass transfers between high- and low-reduced-gradient groups with exact-score line searches. |
| g0011 | I1 | structure | candidate | seed | 0 | 0.50 | Run basin-hopping with support-preserving, correlated perturbations: multiply groups of neighboring runs by reciprocal random factors, renormalize, and locally polish each proposal. Sweep perturbation strengths from 1e-4 to 1e-1 in log amplitude. Maintain a separate best-exact archive even when temporarily accepting worse basins. |
| g0012 | I2 | formulation | validated | diff | 0 | 0.50 | Represent the density losslessly by its nonzero indices and weights. Optimize relative weights of spatial/run-width packet classes using a linearized convolution-maximum epigraph, adding violated lag constraints. Backtrack each proposed direction against the frozen score; preserve the original density unless an exactly rescored improvement is found. |
| g0013 | I1 | formulation | candidate | strategist | 0 | 0.50 | Combine lossless sparse support storage with direct nonnegative packet amplitudes. Include zero cells bordering occupied runs as activation variables. Normalize total mass, jointly optimize packet weights and boundary activations, and backtrack against the exact convolution score. Preserve the original grid and retain an untouched incumbent. |
| g0014 | I1 | numeric | candidate | strategist | 0 | 0.50 | Estimate nonnegative multipliers over competing convolution peaks by minimizing the nonnegativity-projected stationarity residual. Use the resulting reduced gradient to rank boundary activations and packet mass transfers. Validate rankings by directional finite differences, then test geometrically spaced steps against the exact score rather than trusting a single-peak gradient. |
| g0015 | I1 | structure | candidate | strategist | 0 | 0.50 | Alternate packet-weight optimization with coordinated translations of several occupied runs, preserving each run's internal profile. Test shifts at multiple spatial scales, reject overlaps, and repolish weights after each translation. Keep a separate exploratory state and exact-best archive so temporary losses can cross support barriers without replacing the incumbent. |
| g0016 | I3 | formulation | validated | diff | 0 | 0.50 | Represent a candidate as baseline samples plus disjoint packet-fission operations. Each operation preserves a packet’s ordered values but splits it into several teeth, borrowing bounded space from neighboring zero gaps. The adapter expands this structural representation into the verifier’s dense nonnegative array; canonical candidates remain accepted unchanged. |

By status: candidate: 12, refuted: 1, validated: 3. A gene is `validated` or `refuted` only by its knock-out pairs (see the genealogy for the evidence format).

## Strategy patches and their outcomes

- `evt_000109` at 54% of the budget: `{"operator_priors": {"refine": [2.0, 1.0], "restructure": [1.0, 1.0], "hybridize": [2.0, 1.0], "reformulate": [1.0, 1.0], "fresh": [1.0, 3.0]}, "selection.parent_power": 0.6, "islands.budget_shares": {"I1": 0.55, "I2": 0.45}, "seed_genes": [{"type": "formulation", "text": "Combine lossless sparse support storage with direct nonnegative packet amplitudes. Include zero cells bordering occupied runs as activation variables. Normalize total mass, jointly optimize packet weights and boundary activations, and backtrack against the exact convolution score. Preserve the original grid and retain an untouched incumbent."}, {"type": "numeric", "text": "Estimate nonnegative multipliers over competing convolution peaks by minimizing the nonnegativity-projected stationarity residual. Use the resulting reduced gradient to rank boundary activations and packet mass transfers. Validate rankings by directional finite differences, then test geometrically spaced steps against the exact score rather than trusting a single-peak gradient."}, {"type": "structure", "text": "Alternate packet-weight optimization with coordinated translations of several occupied runs, preserving each run's internal profile. Test shifts at multiple spatial scales, reject overlaps, and repolish weights after each translation. Keep a separate exploratory state and exact-best archive so temporary losses can cross support barriers without replacing the incumbent."}]}`
  - Rationale (LLM): Recent gains are roughly four orders of magnitude below the required record increment. Refinement has been productive historically, so retain it, but diversify its parents and give the untested sparse formulation on I2 meaningful budget. Favor hybridizing existing methods over fresh restarts, whose observed results are weak. Test changes that address frozen support and nonsmooth peak competition without repeating the refuted constraint-generation scheme. Preserve the exact incumbent and original grid throughout.
  - Rejected entries: ['fidelity budgets must stay ordered f0 <= f1 <= f2; the fidelity keys were dropped']
  - Outcome: reverted: progress 1.059e-09 after vs 1.273e-09 before
