# Field survey for architecture v2 (Sonar, 14 Nov 2026)

18 entries: S1–S6 attack similar problems, S7–S18 come from other fields. Every quoted number comes from the URL
given, fetched today. `[unread]` marks what I could not open; treat it as memory, not evidence.

## A. Machine discovery of mathematical constructions

**S1. FunSearch — LLM program search.**
- How: an LLM proposes "functions written in computer code"; an automated evaluator runs them and discards wrong ones; a programs database seeds the next prompts.
- Known for: "the largest increase in the size of cap sets in the past 20 years". https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/
- Transfers: search over short generators, not objects; the evaluator is the only judge.
- Breaks: needs thousands of fast evaluations; ours take 45–450 s (E2).
- Speaks to: E7, E8.

**S2. AlphaEvolve — evolutionary coding agent.**
- How: "an autonomous pipeline of LLMs" edits code; programs are run and scored automatically; evolution keeps a population.
- Known for: 4x4 complex matrices in 48 multiplications, first gain on Strassen "after 56 years"; 50+ open problems, "roughly 75%" rediscovered, 20% improved. https://arxiv.org/abs/2506.13131 and https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- Transfers: one harness, many problems; rediscovery is the usual outcome, so the per-problem budget must be small and stoppable.
- Breaks: flat evolution, no stage contracts; large LLM budget.
- Speaks to: E8, E9.

**S3. AlphaEvolve "search mode" and proof chain (Georgiev, Gómez-Serrano, Tao, Wagner).**
- How: "evolves programs that search for a construction", not the construction. Chain: construction → Deep Think proof → AlphaProof Lean formalization.
- Known for: 67 problems; best known "in most of the cases", improved "in several". https://arxiv.org/html/2511.02864v3
- Transfers: two levels under one score (the loop improves the searcher, the searcher improves the object); each stage outputs a different type.
- Breaks: our searchers are hand-written files; comparing searchers fairly needs fixed time per evaluation.
- Speaks to: E3, E5, E8.

**S4. PatternBoost — local search plus a learned seed generator.**
- How: classical search produces many good constructions; a transformer is trained on the best; its samples "are then used as seeds for the first phase".
- Known for: "a counterexample to a conjecture that had remained open for 30 years". https://arxiv.org/abs/2411.00566
- Transfers: the loop also produces seeds. E1 says seeds dominate.
- Breaks: needs thousands of examples; with 100 seeds only a tiny seed model is fittable (conjecture).
- Speaks to: E1, E6.

**S5. Cube-and-conquer — two chained SAT solvers.**
- How: a look-ahead solver splits the problem into 10^6 cubes, then "billions of sub-cubes"; a different solver (CDCL) finishes each; every answer ships a proof checked separately.
- Known for: Boolean Pythagorean triples impossible for {1..7825}; "800 cores in about 2 days"; 200 TB proof, 68 GB certificate. https://arxiv.org/abs/1605.00723
- Transfers: one stage chosen for splitting, another for finishing; the hand-off is a typed object; the split is applied twice.
- Breaks: we have no exact partition; our "cubes" would be seeds or basins, without completeness.
- Speaks to: E5, E7, E10.

**S6. ShinkaEvolve — sample-efficient program evolution.**
- How: novelty rejection before evaluation (embedding similarity above 0.95 goes to a judge); UCB1 over LLMs, updated each generation.
- Known for: state-of-the-art circle packing "using only 150 samples". https://arxiv.org/html/2509.19349v1
- Transfers: refuse near-duplicates before paying for them.
- Breaks: UCB1 works there with many pulls per arm; ours had about 10 per run (E3).
- Speaks to: E3, E10.

## B. Other fields

**S7. AlphaZero — game search.**
- How: tree search guided by a network; the search's visit counts become the policy's training target; the new network guides the next search.
- Known for: "outperformed Stockfish after just 4 hours"; "losing zero games" in 100. https://arxiv.org/html/1712.01815v1
- Transfers: the expensive level's output becomes the cheap level's prior.
- Breaks: millions of games; we have about 1,000 experiments (E3), enough for a prior with a few numbers.
- Speaks to: E1, E3.

**S8. Go-Explore — exploration in reinforcement learning.**
- How: an archive of visited states; "first return to a promising state (without exploration), then explore from it"; a separate phase hardens the solution.
- Known for: Montezuma's Revenge "over 43,000 points, almost 4 times the previous state of the art". https://arxiv.org/html/1901.10995v4
- Transfers: an archive keyed by state type (problem, resolution, provenance), not best score alone; find, then harden.
- Breaks: a poor state descriptor collapses to "best only", which is today's warm start.
- Speaks to: E1, E5, E7b.

**S9. Hyperband — hyperparameter allocation.**
- How: run many configurations briefly, keep the top 1/η ("default η=3"), give survivors more; several brackets hedge "many short" against "few long"; one bracket is plain random search.
- Known for: "5× to 30× faster than popular Bayesian optimization algorithms". https://arxiv.org/html/1603.06560v4
- Transfers: E1 (rank at 800 cells survives refinement) is exactly its precondition; a rule with no model beats a starved model.
- Breaks: late bloomers; E9 says 1a improves at every experiment. Measure rank correlation per problem first.
- Speaks to: E1, E3, E4, E7d, E9.

**S10. SATzilla — solver portfolios.**
- How: instance features feed "empirical hardness models" that choose among solvers.
- Known for: "three gold, one silver and one bronze medal", 2007 SAT Competition. https://arxiv.org/abs/1111.2249
- Transfers: choose the method per problem from cheap probe features (did experiment 1 already plateau?).
- Breaks: trained on hundreds of instances; we have three problems, so features must be live probes.
- Speaks to: E2, E8, E9.

**S11. Multigrid — coarse-to-fine numerics.**
- How: smooth on the fine grid, pass the residual down, solve there by the same method, pass the correction up; "can be coded using recursion".
- Known for: work "proportional to the number of unknowns"; an F-cycle costs "83% more" than a V-cycle. https://en.wikipedia.org/wiki/Multigrid_method
- Transfers: one solver at every resolution with typed transfers both ways. E5 and E6 only go up; nobody tested going back down.
- Breaks: needs a meaningful coarsening; ours may destroy fine structure (conjecture).
- Speaks to: E5, E6, E10.

**S12. Basin-hopping — cluster chemistry.**
- How: replace each point's value by the local minimum reached from it ("interpenetrating staircases"); move between minima; this "does not change the global minimum".
- Known for: lowest known Lennard-Jones structures "up to 110 atoms". https://arxiv.org/abs/cond-mat/9803344
- Transfers: judge a start by its polished value; kick, re-polish, accept by rule.
- Breaks: E11: small kicks fall back into the same basin (5e-11 per 120 s); kicks must be structural.
- Speaks to: E1, E11.

**S13. STOKE — compiler superoptimization.**
- How: random search over programs scored on test cases; survivors face a formal equivalence check; "a counterexample is found, and this is used as a new testcase".
- Known for: code that can "match or outperform" gcc at full optimization. https://arxiv.org/abs/1211.0557 and https://github.com/StanfordPL/stoke
- Transfers: cheap check inside, exact check at the gate, and every gate failure becomes a permanent test.
- Breaks: our exact check is already cheap (21 s, E10); our gap is a stale reference, which needs a live read.
- Speaks to: E7, E10.

**S14. A-Lab and its audit — laboratory automation.**
- How: a robot loop proposes, synthesizes and analyzes automatically `[unread]`; outside chemists re-analyzed every product.
- Known for: 43 "novel materials" claimed; audit: "no new materials have been discovered in that work"; the automated analysis "is not yet reliable". https://journals.aps.org/prxenergy/abstract/10.1103/PRXEnergy.3.011002
- Transfers: the loop's own classifier never issues the claim; novelty is checked against what is already known.
- Breaks: nothing; this is E7a at scale. The price is a slower claim.
- Speaks to: E7.

**S15. Platform trials (I-SPY 2) — adaptive trials.**
- How: many arms share "a single standard arm"; ineffective arms stop, new ones join; thresholds `[unread]`.
- Known for: "tested 12 experimental drugs" across 19 centers. https://en.wikipedia.org/wiki/Adaptive_design_(medicine)
- Transfers: one shared paired control per seed set; drop and add rules written before the data.
- Breaks: arms there get many patients; with 4 experiments per arm any rule is noise (E7d).
- Speaks to: E2, E3, E7c, E7d.

**S16. Marginal value theorem (Charnov 1976) — foraging ecology.**
- How: leave a patch "when the marginal capture rate in the patch drops to the average capture rate for the habitat"; longer travel, longer stay.
- Known for: dung-fly copulation predicted 40 minutes, observed 36. https://en.wikipedia.org/wiki/Marginal_value_theorem
- Transfers: leave a stage when its gain per CPU-second falls below the loop's average, hand-off cost included.
- Breaks: needs diminishing returns and a known average; other tests "substantially deviating"; E6 restarts are non-monotone.
- Speaks to: E5, E9, E10, E11.

**S17. Expected improvement and the max K-armed bandit — allocation when only the best outcome counts.**
- How: expected improvement scores a candidate by "the posterior expected gain over the best value observed so far", from a model's mean and uncertainty. The max bandit allocates trials "to maximize the expected maximum payoff", moving to the observed best arm "at a rate increasing double exponentially".
- Known for: choosing restart heuristics in "two scheduling domains" (AAAI 2005 Outstanding Paper); O(k ln(k/δ) ln(n)²/ε²) trials find a near-best arm if payoffs follow an extreme-value law. EGO's own results `[unread]`. https://en.wikipedia.org/wiki/Bayesian_optimization , https://www.cicirello.org/publications/cicirello2005aaai.html , https://cdn.aaai.org/AAAI/2006/AAAI06-022.pdf
- Transfers: score a restart setting by its chance of beating the live reference, not by its mean; E12's two wins sit in one setting and one seed, and an average over 55 restarts hides them. An untried setting has the widest uncertainty, so it ranks high.
- Breaks: with 2 wins in 55 rows the tail is a guess and the score returns the prior; arms are assumed independent with fixed distributions, but E12's wins need seed and setting together; fast concentration on the observed best arm is E7c.
- Speaks to: E3, E6, E7c, E12.

**S18. Expected improvement per second and freeze-thaw Bayesian optimization — cost-aware hyperparameter search.**
- How: Snoek et al. score a candidate by "expected improvement per second", preferring points "likely to be good" and "likely to be evaluated quickly"; a second model predicts the duration. Freeze-thaw forecasts the final value of each partial run and decides which frozen run to "thaw" or whether to start a new one, from a basket of 10 old and 3 new candidates.
- Known for: CIFAR-10 test error 14.98%, "over 3% better than the expert" (18%); freeze-thaw "significantly outperforms" ordinary Bayesian optimization on three tasks, no single number quoted. https://arxiv.org/abs/1206.2944 , https://arxiv.org/abs/1406.3896 (full texts read from the PDFs)
- Transfers: this is the prior art of v6's bet: gain over the incumbent divided by predicted cost, and "continue or start new" as one decision. Different in v6 (my reading): saved rows instead of a fitted model, a survival term to the end of a chain, moves of unlike stage types, gain counted on the pool's best.
- Breaks: both need a model fitted to many evaluations (E3). Freeze-thaw assumes the loss "roughly follows an exponential decay towards an unknown final value"; our lineages jump at a soft restart (E12) and stall elsewhere (E11). Duration is predicted there; ours was cut by a wall clock (D6).
- Speaks to: E3, E5, E9, E10, E12.

## Five cross-cutting principles

1. **Promote, do not pull.** Screen many starts cheaply and promote the top fraction by a fixed rule (S9, S8, S12). Fits E1, avoids E3.
2. **Stages differ in kind and hand over typed objects.** Splitter then finisher, coarse then fine, construction then proof; each hand-off names problem, resolution and provenance (S5, S3, S11, S8). Fits E5, prevents E7b.
3. **The same loop at every level.** Seeds, solutions and searchers are improved by one loop, and the expensive level's results become the cheap level's prior (S7, S4, S3, S11).
4. **Truth sits outside the loop.** An independent check and a live reference issue the claim; every failure becomes a standing test (S5, S13, S14). Answers E7a.
5. **Rules before data, sized to the sample.** Probe each problem first (rank correlation, plateau), then pick the schedule; never adapt on fewer pulls than the rule needs (S10, S15, S9). Fits E8, E9, E7c, E7d.

Not verified: FunSearch island details, A-Lab's own paper (Nature login wall), I-SPY 2 thresholds.
