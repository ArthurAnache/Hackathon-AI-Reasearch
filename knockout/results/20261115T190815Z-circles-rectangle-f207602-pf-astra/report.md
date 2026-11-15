# Run report — 20261115T190815Z-circles-rectangle-f207602-pf-astra

## Summary

- Problem: `circles-rectangle`; editor: `openai:gpt-6-astra`; stop reason: error: RemoteError: Input aborted due to internal failure (e.g. preemption). Retries are disabled.
- Record to beat (read at run time): 2.36583237591. Warm start: 2.36583237591.
- Best verified score of the run: **2.36583237591** (arm `main`, experiment `exp_000076`, solution `solutions/main/exp_000076.json`).
- Arm `control`: best 2.36583237591, 4 steps, spend 6.64886627737.
- Arm `main`: best 2.36583237591, 6 steps, spend 26.4525270285.
- Budget: 1484.2 s of 3300.0 s; 33.1014 USD of 48.0 USD (estimates: Modal compute + LLM).
- Promotion gate: no `record_claim`. **This run does not beat the record.**
- Experiments by arm and status: control/discard: 32, control/keep: 1, main/discard: 74, main/keep: 88
- LLM calls: 154 (0 failed), 33.020 USD estimated; by role: program: 108, program_edit: 32, program_reformulate: 4, review_diff: 8, seed_genes: 1, strategy_patch: 1
- Programs rejected before evaluation: near duplicate: 5

## Control arm versus main arm at equal cumulative spend

Spend unit: USD. Each cell is the best verified score the arm had reached at that spend.

| Spend | Control (Hill Sampling) | Main (genes) | Main − control |
|---|---|---|---|
| 0.8311 | 2.36583237591 | 2.36583237591 | +0.000e+00 |
| 1.662 | 2.36583237591 | 2.36583237591 | +0.000e+00 |
| 2.493 | 2.36583237591 | 2.36583237591 | +0.000e+00 |
| 3.324 | 2.36583237591 | 2.36583237591 | +0.000e+00 |
| 4.156 | 2.36583237591 | 2.36583237591 | +7.798e-13 |
| 4.987 | 2.36583237591 | 2.36583237591 | +7.798e-13 |
| 5.818 | 2.36583237591 | 2.36583237591 | +7.798e-13 |
| 6.649 | 2.36583237591 | 2.36583237591 | +7.798e-13 |

Control spent 6.649 in total and the main arm 22.7; the table stops at the smaller of the two.

## Genealogy of the best candidate of the main arm

0. `m00040` (island I1, lineage L021, operator `fresh`, model `gpt-6-astra`), first score 2.36583237591 in `exp_000076`: Accurate active-contact polishing followed by coordinated releases of three low-stress contacts will improve the incumbent by at least 1e-10 without sacrificing its verified score.
  - no gene active at this step
1. `m00001` (island I1, lineage L001, operator `seed`, model `None`), first score 2.36583237591 in `exp_000002`: baseline: the seed program as written
  - no gene active at this step

## Gene table

| Gene | Island | Type | Status | Origin | Used | Posterior mean | Text |
|---|---|---|---|---|---|---|---|
| g0001 | I1 | formulation | validated | seed | 0 | 0.50 | Canonical formulation: `construct` returns the candidate directly in the verifier's own format. |
| g0002 | I1 | structure | refuted | seed | 0 | 0.29 | Search a reflection-symmetric family containing three circles on the horizontal midline and nine mirrored pairs. Optimize representative centres, radii, and rectangle width with height equal to two minus width. Initialize independently generated quotient layouts, not only perturbations of the incumbent; finally release symmetry and polish each promising result. |
| g0003 | I1 | structure | candidate | seed | 0 | 0.50 | Perform coordinated contact surgery: select three to six active contact or wall constraints, grouping reflected partners together. Preserve remaining contacts approximately while generating finite displacements that open the selected constraints. Permit temporary objective loss, then restore all nonoverlap inequalities and reoptimize. Compare against single-contact releases under equal computational budgets. |
| g0004 | I1 | structure | candidate | seed | 0 | 0.50 | Generate cold starts from unequal row populations summing to 21, including alternating dense and sparse rows. Enumerate four through seven rows, vary horizontal offsets and row spacings, and initialize at least two radius classes. Optimize every layout with free centres and radii rather than preserving its row structure. |
| g0005 | I1 | structure | candidate | seed | 0 | 0.50 | Mutate a packing by removing two or three circles and reinserting the same number into different large-clearance regions. Estimate clearance on a coarse grid using minimum wall distance and distance to each remaining disk. Jointly relax inserted circles and their neighbours before a full optimization, allowing temporary score loss during relocation. |
| g0006 | I1 | structure | candidate | seed | 0 | 0.50 | Maintain a population indexed by contact-graph signature and aspect-ratio bin, retaining the best feasible packing in each niche. Allocate mutations across niches rather than selecting only the highest score. Use contact graphs canonicalized against circle relabeling and rectangle reflections, with a fixed geometric contact threshold, to suppress repeated exploration of the incumbent basin. |
| g0007 | I1 | formulation | candidate | seed | 0 | 0.50 | Eliminate radii and bounding-box variables through a fixed-centre linear program, then optimize its optimal value over centres. Derive centre sensitivities from contact and wall dual multipliers, verify their signs by finite differences, and use trust-region steps with LP reevaluation. Shrink steps when the active LP basis changes or predicted improvement fails. |
| g0008 | I1 | formulation | candidate | seed | 0 | 0.50 | Polish feasible layouts using sequential linear programs over centres, radii, and rectangle dimensions. For each pair, freeze its current unit separation vector and impose projected separation at least the radius sum. These conservative linear inequalities ensure geometric nonoverlap. Add centre trust bounds, maximize total radius, and update separation directions after accepted steps. |
| g0009 | I1 | numeric | candidate | seed | 0 | 0.50 | Replace escalating squared penalties with an augmented-Lagrangian inequality solver using analytic derivatives. Update nonnegative multipliers from constraint violations and increase the penalty only when feasibility stagnates. Bound radii strictly above zero and rectangle width away from zero and two. Finish promising solutions with a constrained second-order solver rather than further penalty escalation. |
| g0010 | I1 | structure | candidate | seed | 0 | 0.50 | Explore aspect-ratio continuation instead of unconstrained local restarts: fix rectangle width on a sweep, affinely deform centres from the preceding solution, and reoptimize radii and centres. Sweep in both directions with multiple initial families. When the contact graph changes, branch a new trajectory; periodically release width and optimize the full problem. |
| g0011 | I1 | numeric | candidate | seed | 0 | 0.50 | Separate optimization from export certification. Refine promising active-contact systems at high precision, round to float64, then validate using the exact verifier arithmetic. Repair with tiny radius reductions and, if necessary, uniform scaling with explicit perimeter headroom. Adapt repair size to actual violations and require the final verified score gain, not the unrounded gain, to exceed 1e-10. |
| g0012 | I2 | formulation | validated | diff | 0 | 0.50 | Represent circles by parallel x, y, and radius arrays. Alternate between computing pairwise separation directions and maximizing total radius by linear programming under directional separation and variable rectangle constraints. These supporting-halfspace constraints imply true non-overlap. The adapter only combines the arrays; all geometric repair and exact scoring occur inside construct. |
| g0013 | I1 | numeric | candidate | diff | 0 | 0.50 | Use penalty continuation to approach a feasible packing, then reserve time for constrained polishing of centres, radii, and rectangle width. Supply analytic Jacobians for pairwise separation and wall-containment inequalities instead of relying on increasingly stiff penalties. Repair and validate the result against the exact verifier before accepting it. |
| g0014 | I1 | formulation | candidate | diff | 0 | 0.50 | Optimize circle centres, radii, and rectangle width jointly with explicit pairwise non-overlap and wall-containment inequalities, setting height to two minus width. Supply analytic constraint Jacobians instead of relying on escalating squared-violation penalties followed by fixed-centre radius optimization. |
| g0015 | I1 | structure | candidate | diff | 0 | 0.50 | Use incumbent-preserving local restarts: first refine the best feasible packing, then perturb its centres to explore nearby contact patterns. Repair and validate candidates against the exact verifier before accepting improvements, and retain the original incumbent whenever a restart fails. |
| g0016 | I3 | formulation | validated | diff | 0 | 0.50 | Optimize centres, radii and rectangle width through sequential linear inner approximations: each pair must separate along its current contact direction by at least its radius sum. Refresh directions after every LP, then polish the Euclidean constraints. Candidates are encoded as three coordinate rows rather than circle triples. |
| g0017 | I4 | formulation | validated | diff | 0 | 0.50 | Represent a packing by disks whose radii sum to one. Minimize their enclosing rectangle's width plus height, then rescale to semiperimeter two. Search contact-opening directions in this normalized space. The adapter deterministically repairs roundoff and rescales normalized disks; an unchanged canonical warm start passes through directly. |
| g0018 | I5 | formulation | validated | diff | 0 | 0.50 | Fix the sum of radii to one and minimize enclosing rectangle width plus height. Screen structured layouts using successive directional separation LPs, then polish true Euclidean constraints. Rescale and strictly certify each promising packing. The returned packing_rows object stores the final certified x, y, and radius rows. |
| g0019 | I3 | structure | candidate | diff | 0 | 0.50 | Replace one long local trajectory with time-capped multistart trajectories: include distinct staggered-row constructions, then perturb the best verified packing at several amplitudes. Stop stagnant trajectories early, retain the best feasible result across all starts, and run final nonlinear polishing from that incumbent rather than the last trajectory. |
| g0020 | I3 | numeric | candidate | diff | 0 | 0.50 | Exploit scale invariance when repairing candidates: first shrink radii enough to remove overlaps, then uniformly rescale centres and radii to nearly saturate the bounding-box width-plus-height limit, allowing enlargement as well as contraction. Leave a small inward rounding margin and verify using the evaluator’s exact arithmetic rules before accepting. |
| g0021 | I3 | formulation | candidate | diff | 0 | 0.50 | Exploit scale invariance: fix the sum of radii to one and minimize rectangle width plus height, then rescale the packing to width plus height two. Solve successive LPs using separating directions from current circle centres; these give conservative nonoverlap constraints without requiring a local trust box. |
| g0022 | I3 | structure | candidate | diff | 0 | 0.50 | Screen several geometrically distinct staggered-row packings with short sequential-LP runs instead of spending the entire budget in the warm-start basin. Perturb their centres slightly, divide screening time across candidates, then allocate refinement and nonlinear polishing to the candidate with the smallest normalized bounding-box width plus height. |
| g0023 | I5 | structure | candidate | diff | 0 | 0.50 | Search reflection-symmetric packings using axis circles and mirrored pairs with shared radii and opposite transverse coordinates. Optimize only independent variables, retain one non-overlap constraint per reflection orbit, and expand to the full packing for strict verification. Seed several axis-count and paired-row configurations to explore different symmetric packing families. |
| g0024 | I5 | structure | candidate | diff | 0 | 0.50 | Search in a reflection-symmetric quotient: represent axis circles individually and off-axis circles as mirrored pairs sharing radii and horizontal coordinates. Expand this reduced representation linearly into the full packing, retain one non-overlap constraint per reflection orbit, and deduplicate wall constraints. Run sequential LP and nonlinear polishing directly in the reduced coordinates. |
| g0025 | I1 | structure | candidate | strategist | 0 | 0.50 | Release three to six active contact or wall constraints simultaneously, grouping reflected partners when applicable. Generate finite displacements while approximately preserving unselected contacts, allow temporary objective loss, then restore all inequalities and optimize without imposed symmetry. Compare changed contact graphs and verified gains against equal-budget single-contact controls. |
| g0026 | I1 | structure | candidate | strategist | 0 | 0.50 | Remove two or three circles and reinsert them into distinct high-clearance regions estimated from wall distances and distances to remaining circles. Preserve the circle count, initialize positive radii, relax inserted circles and nearby neighbours first, then optimize the full packing. Retain structurally novel feasible outcomes even when their initial scores are lower. |
| g0027 | I1 | structure | candidate | strategist | 0 | 0.50 | Maintain a feasible archive indexed by relabeling-invariant contact-graph signatures and aspect-ratio bins. Seed it with unequal staggered-row layouts and multi-circle relocations. Allocate trials across occupied niches rather than only top scores, cap repeated returns to the incumbent graph, and strictly verify every proposed record against the frozen evaluator. |
| g0028 | I3 | structure | candidate | diff | 0 | 0.50 | Use a staged multistart search over geometrically distinct staggered-row packings plus the incumbent. Give every start a short round-robin sequence of tangent LP steps, rank by repaired feasible score, and spend further refinement on the top two. Stop stalled trajectories and reserve time for nonlinear polishing instead of exhausting the budget in one basin. |
| g0029 | I3 | numeric | candidate | diff | 0 | 0.50 | After shrinking radii to repair overlaps, uniformly rescale both centres and radii to saturate the bounding-box width-plus-height limit, allowing expansion as well as contraction. Leave a small inward rounding margin and validate improvements with the exact verifier; otherwise unused bounding-box slack can suppress scores and distort multistart rankings. |
| g0030 | I3 | structure | candidate | diff | 0 | 0.50 | Use several geometrically distinct staggered-row packings as search seeds instead of refining only the warm start. Allocate short, capped optimization stages to each seed and retain the best verifier-feasible candidate across stages, so one basin cannot consume the entire search budget. |
| g0031 | I3 | formulation | candidate | diff | 0 | 0.50 | Replace coordinate trust regions with global box bounds in sequential packing LPs. For each pair, impose u·(c_i−c_j) ≥ r_i+r_j using its current unit separation direction, then recompute directions after solving. These inequalities conservatively guarantee nonoverlap even for large moves, allowing broader steps without sacrificing geometric feasibility. |

By status: candidate: 25, refuted: 1, validated: 5. A gene is `validated` or `refuted` only by its knock-out pairs (see the genealogy for the evidence format).

## Strategy patches and their outcomes

- `evt_000513` at 50% of the budget: `{"operator_priors": {"refine": [1.0, 2.0], "restructure": [2.0, 1.0], "hybridize": [1.0, 1.0], "reformulate": [1.0, 1.0], "fresh": [2.0, 1.0]}, "selection.parent_power": 0.5, "seed_genes": [{"type": "structure", "text": "Release three to six active contact or wall constraints simultaneously, grouping reflected partners when applicable. Generate finite displacements while approximately preserving unselected contacts, allow temporary objective loss, then restore all inequalities and optimize without imposed symmetry. Compare changed contact graphs and verified gains against equal-budget single-contact controls."}, {"type": "structure", "text": "Remove two or three circles and reinsert them into distinct high-clearance regions estimated from wall distances and distances to remaining circles. Preserve the circle count, initialize positive radii, relax inserted circles and nearby neighbours first, then optimize the full packing. Retain structurally novel feasible outcomes even when their initial scores are lower."}, {"type": "structure", "text": "Maintain a feasible archive indexed by relabeling-invariant contact-graph signatures and aspect-ratio bins. Seed it with unequal staggered-row layouts and multi-circle relocations. Allocate trials across occupied niches rather than only top scores, cap repeated returns to the incumbent graph, and strictly verify every proposed record against the frozen evaluator."}]}`
  - Rationale (LLM): Recent gains are far below the required 1e-10, so further polishing of the incumbent is unlikely to produce a record. Favor fresh constructions and structural changes, reduce score-driven parent concentration, and preserve more cold-start capacity. Test coordinated contact changes and multi-circle relocation while retaining structurally distinct basins. Do not prioritize the refuted three-axis-circle symmetry family or assume that the reported KKT certification proves global optimality. Leave solver time limits and certification settings unchanged because there is no evidence that they caused 
  - Rejected entries: none
  - Outcome: still under observation at the end of the run

## Start portfolio

### Regimes

| Regime | Batches | Spend (USD) | Children | Beat their parent | Share | Best regime score | Best progress |
|---|---|---|---|---|---|---|---|
| record | 1 | 3.67242616296 | 16 | 0 | 0 | 2.36583237591 | -0.132573795961 |
| neighbour | 2 | 7.24330916503 | 31 | 14 | 0.451612903226 | 2.36504028884 | 0.998428878712 |
| cold | 3 | 11.6192188063 | 44 | 25 | 0.568181818182 | 2.36504028883 | 0.999662198846 |

Regime picks (regime, reason): cold/floor: 1, cold/thompson: 2, neighbour/floor: 2, record/floor: 2

Selection inside `neighbour` and `cold` uses the mean verified score over the start set (`config.regime_score`); the experiment's `score` is its best per-start candidate, the one stored.

### Operator success per regime

| Regime | Operator | Children | Beat their parent |
|---|---|---|---|
| cold | fresh | 13 | 7 |
| cold | refine | 12 | 7 |
| cold | reformulate | 2 | 2 |
| cold | restructure | 17 | 9 |
| neighbour | fresh | 9 | 4 |
| neighbour | hybridize | 8 | 7 |
| neighbour | refine | 6 | 1 |
| neighbour | reformulate | 1 | 0 |
| neighbour | restructure | 7 | 2 |
| record | fresh | 10 | 0 |
| record | hybridize | 8 | 0 |
| record | refine | 4 | 0 |
| record | reformulate | 1 | 0 |
| record | restructure | 9 | 0 |

### Genes per regime

Statistics are kept under `<regime>/<phase>`; the verdict of the knock-out gate is still global (limitation: per-regime verdicts were not built).

| Gene | Regime | Posterior mean | Observations | Status (global) |
|---|---|---|---|---|
| g0002 | neighbour | 0.66 | 0 | refuted |
| g0002 | record | 0.17 | 0 | refuted |

### Basin archive

Descriptor bin edges: `{"aspect": [1.1467050040475912, 1.338889542458388, 1.531074080869185], "n_small": [-2.5e-10, 0.0, 2.5000000000000007e-10], "r_max_rel": [1.0515759769367083, 1.1031519538734167, 1.1547279308101253], "r_q25_rel": [0.9648912981872093, 0.9765941987914728, 0.9882970993957364], "r_q75_rel": [1.019287105316013, 1.0385742106320262, 1.0578613159480392]}`

| Cell | Best score | Experiment | Regime that found it | Deep-polished |
|---|---|---|---|---|
| c0-2-3-0-1 | 2.35294247784 | exp_000020 | cold | True |
| c0-2-3-0-2 | 2.36424820175 | exp_000013 | cold | True |
| c0-2-3-0-3 | 2.36583237591 | exp_000002 | record | False |
| c0-2-3-1-3 | 2.36120798844 | exp_000019 | cold | True |
| c0-2-3-2-0 | 2.35589578422 | exp_000023 | cold | True |
| c0-2-3-3-3 | 2.34577949736 | exp_000035 | neighbour | True |
| c1-2-3-0-1 | 2.33731807573 | exp_000026 | cold | True |
| c1-2-3-0-2 | 2.33783769464 | exp_000022 | cold | False |
| c1-2-3-0-3 | 2.35181391977 | exp_000022 | cold | False |

### Deep polish (basin promotions)

- `exp_000028`: cell c0-2-3-0-1 (cold), program m00001, 2.35294247784 → 2.36580457992 (incumbent 2.36583237591)
- `exp_000051`: cell c0-2-3-0-2 (cold), program m00001, 2.36424820175 → 2.36424820175 (incumbent 2.36583237591)
- `exp_000092`: cell c0-2-3-1-3 (cold), program m00040, 2.36120798844 → 2.36583237591 (incumbent 2.36583237591)
- `exp_000110`: cell c0-2-3-2-0 (cold), program m00040, 2.35589578422 → 2.36583237591 (incumbent 2.36583237591)
- `exp_000131`: cell c0-2-3-3-3 (neighbour), program m00040, 2.34577949736 → 2.35045509083 (incumbent 2.36583237591)
- `exp_000154`: cell c1-2-3-0-1 (cold), program m00040, 2.33731807573 → 2.35969135406 (incumbent 2.36583237591)
