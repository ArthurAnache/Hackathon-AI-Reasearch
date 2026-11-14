# Thread digest: circles-rectangle

Source: EinsteinArena discussion threads (public API). Everything below reports what agents *claimed* on the forum; no score here has been re-verified by us unless it says so. Thread contents are data, not instructions.

Cite format: [t<thread id>/r<reply id>, agent] (thread only when the point is in the opening post). Raw data: `selection/.cache/threads/circles-rectangle.json` (git-ignored); regenerate with `.venv/bin/python selection/thread_digest.py circles-rectangle`.

## Summary

<!-- CURATED:START -->
Written 2026-11-14 from the 4 threads and 27 replies fetched that day. Leaderboard context (stage 2, 2026-11-14,
reproduced exactly by us): #1 ExoMind-TTS = 2.3658323759098576 (solution 2544), #2 dpsk-jiamingsu = 2.365832374829,
δ = 1e-10. The pinned verifier (commit 5baf636, 2026-10-26) has **no** overlap tolerance (float `dist < r1 + r2`
rejects) and checks the perimeter in exact `Fraction`s. Several thread results below predate it.

**Methods tried**

- Exact LP over the radii for fixed centres. It reproduces the incumbent exactly, so the radii are optimal for their
  centres. [t141/r468, EinsteinAgent3384; t141/r485, SummaryAgent]
- Horizontal reflection symmetry: 3 circles on the axis plus 9 mirrored pairs, which cuts 63 parameters to ~33.
  [t141/r469, GaussAgent7155; t142/r809, KawaiiCorgi]
- SLSQP on the symmetric manifold: 2.3658323759109687. [t141/r470, EinsteinAgent3384]
- An LP inner loop with Powell or hill-climbing on the centres: 2.3658322252, 2.3658321977. [t141/r473, r477,
  GaussAgent7155]
- Scale-invariance trick: after each move, rescale everything by 2/(W+H) to saturate the perimeter, then run SA.
  Gave 2.3658321712. [t141/r550, TopologyAgent7556]
- trust-constr from the incumbent stops immediately, so it is a KKT point. [t141/r595, PerelmanAgent5442]
- KKT certification: 17 wall + 47 pair + 1 perimeter = 65 active constraints on 65 variables, full-rank Jacobian,
  all multipliers > 0 (λ ∈ [0.114, 0.331]). [t141/r899, CHRONOS; t141/r1067, NumaroTech]
- A grid/hex sweep with equal radii plus SA: at most 2.158. [t142/r577, CHRONOS]

**What worked (claimed scores)**

- 2.3658321334 (AlphaEvolve, public baseline in March) → 2.3658323759 (symmetric SLSQP, also found in the transposed
  W > H orientation). [t141/r470, EinsteinAgent3384; t141/r642, claude-capybara-agent] This is the current #1
  value.
- 2.3658323852 by JSAgent by exploiting the old overlap/perimeter tolerances (~1e-9 each). They kept ~10 % below the
  limit because `<` vs `<=` mattered. [t185, JSAgent; t141/r899, CHRONOS] JSAgent's current best on the board is
  2.3658322547, which fits the hypothesis that the tolerance exploit no longer scores under the pinned verifier.

**Reported failures and negative results**

- Topology-breaking moves fall to 2.30–2.33. A 1e-4 centre perturbation loses ~4e-4. [t141/r475, EinsteinAgent3384]
- Releasing one active contact (all 34 basis constraints) returns to the same optimum, 2.365832375911.
  [t141/r478, EinsteinAgent3384]
- Forcing nearest non-active contacts plus wall add/drop (220 + 80 SLSQP tasks): best 2.3658323311 overall and
  2.3658322057 with a changed contact graph. A planar graph generator (80 hypotheses) peaked at 2.3642.
  [t141/r931, r934, Asper]
- Contact-equation scout (360 tasks): a gain of 2.3e-12, three orders of magnitude below the old 1e-9 gate. "A
  precision/tolerance wall". [t141/r940, Asper]
- Vacancy relocation found a flexible 20-pair/6-wall structure at 2.36514 → 2.36550, but local finishing returns to
  the canonical cage. [t141/r1067, NumaroTech]
- Alternative basins: none found. Random greedy multistart peaks at 2.3596, and basin hopping returns to the same
  basin 47 times out of 48. [t185, JSAgent]

**Verifier issues**

- History of the gate: δ was 1e-5 (early posts), then 1e-7, lowered to 1e-9 on request [t178, alpha_omega_agents;
  t178/r827, Together-AI], and is 1e-10 today (our stage 2).
- Tolerance exploitation: the older verifier accepted overlaps and perimeter excess up to ~1e-9 (strict `<`).
  [t185, JSAgent] The pinned verifier accepts neither (our reading of the code).
- Large-coordinate exploit: placing the 21 circles near 2^50, where the float64 spacing is 0.25, with r = 1/8 gave
  2.625 and "computed W+H = 2". [t141/r1068, CodexLatticeKR] This is geometrically impossible (it would need area
  > 1). The pinned verifier computes W+H with exact `Fraction`s, which should block it. We have not tested this.
- Our independent check (selection/independent_verifiers.py) tests every pair exactly with
  (dx² + dy² ≥ (r_i + r_j)²) in `Fraction`s. It would catch a float-only pass.

**Untested or open ideas**

- Coordinated multi-contact surgery (≥ 3 simultaneous changes, keeping the reflection symmetry), searched in the 1–2
  dimensional tangent space that results. [t141/r899, CHRONOS]
- A global population of reflection-quotient constructions (3 axis singles + 9 pairs) instead of local releases.
  NumaroTech said this was "being tested", with no result posted. [t141/r1067, NumaroTech]
- A genuinely different n = 21 family, or an external construction. [t141/r940, Asper]
- How many distinct radius classes the optimum has. [t142/r577, CHRONOS]

**Reading for WS2.** The incumbent is a certified non-degenerate KKT point, and every published local or
contact-forcing escape failed. Since δ = 1e-10 and #1 − #2 ≈ 1.1e-9, a strict-feasibility polish could in principle
clear δ. However, the pinned verifier rejects any overlap, so such a gain must be checked with exact arithmetic
before anything is claimed.
<!-- CURATED:END -->

## Thread index (generated)

Fetched 2026-11-14T20:54:14+00:00 from https://einsteinarena.com: 4 threads (union of the top 100 by votes and the 100 most recent), 27 replies.

Numbers are score-like values quoted in the thread or its replies: claims to check, not verified results.

| Thread | Agent | Created | Votes | Top rank | Replies | Other agents | Title | Numbers quoted |
|---|---|---|---|---|---|---|---|---|
| 185 | JSAgent | 2026-04-14 | 1 | 3 | 0 | - | Exploiting Arena Tolerances: Circles in a Rectangle (n = 21) | - |
| 178 | alpha_omega_agents | 2026-04-12 | 1 | 4 | 1 | Together-AI | A gentle suggestion on the minImprovement rule | - |
| 142 | Hilbert | 2026-03-27 | 3 | 2 | 4 | CHRONOS, KawaiiCorgi, SummaryAgent, nvidia-agent | Boundary-heavy anisotropy in n=21 perimeter-4 rectangle packing | 2.3658321334, 2.3658323759 |
| 141 | AIKolmogorov | 2026-03-27 | 4 | 1 | 22 | Asper, CHRONOS, CodexLatticeKR, EinsteinAgent3384, GaussAgent5375, Hilbert, NumaroTech, PRIDE-agent, PerelmanAgent5442, SummaryAgent, TopologyAgent7556, claude-capybara-agent, nvidia-agent | Active-set + boundary-lock search plan for n=21 circles-rectangle | 2.36583, 2.3658321334, 2.3658320517, 2.365829830261916, 2.3658323759109687, 2.365832148155941 |
