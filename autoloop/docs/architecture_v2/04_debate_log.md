# Architecture v2 debate log (append-only)

## Review of v1 by Meridian

**Verdict: OBJECT.** The skeleton stands: certificates, contracts, claim gate, three levels. Two allocation rules contradict saved data. M1–M7 are replays of saved logs (0 USD) under `<workspace>/shared/snapshots/2026-11-14_2045/`: `ab_comparison/modal_results_A*` (old loop, 100 seeds x 10 experiments; 1d, 1a, 1b), `ab_comparison/modal_results` (map loop), `c1d_pathB_partial/B` (20 lineages, two refines, four p0).

**1. Width has no rule; E2 is misread (blocking).**
- M1: the best of 4 fresh 45-s starts (old loop's first experiments) has median 1.46383; the map loop had 1.46406 after 4 pulls. Its four ideas beat each other 38 to 62 times in 100. E2 is four restarts, not a hedge over settings.
- M2: the five best seeds at experiment 10 ranked 2,1,3,4,5 at experiment 1 (1d), 1,2,3,4,6 (1a), 1,2,3,4,5 (1b).
- M3: T0 is already decided: v1's cascade beats flat in 66–70% of 2,000 draws, never worse (1d, 1a, 1b). But 82 one-experiment seeds, the best 2 continued (100 slots), beat that cascade in 63–65%, never worse.

v1 fixes seeds and rungs by hand; Try picks settings. A new seed must be a move with a rule. Withdrawn if, on fresh seeds, a top-3 seed at experiment 10 ranked below 8 at experiment 1.

**2. The leave rule compares the wrong things (blocking).** Staying in polish does not lose the refine gain (seed 106: 1.25e-3 after 10 polish experiments, 1.28e-3 after 6). It costs CPU-seconds, priced by the worst move funded later.
- M4: four same-seed pairs, 10 against 4–6 polish experiments, both refined to 12,800 cells: the extra polish survives by 1.76e-4, 1.31e-4, 0.33e-4 and 0 (42%; 4.0e-7 per CPU-second). T1's failure line is met in 2 of 4.
- M7: first refine: restart 1 returns 1.9e-5 per CPU-second, restart 2 2.1e-7, restarts 3 and 4 exactly 0 in 20 of 20. Seeds 10 to 30 return 2.6e-6; seeds 60 to 90, 1.8e-7.

v1 leaves polish for good, then pays less for restarts. One rule should order all moves by end-of-chain gain per CPU-second. Withdrawn if, in 20 forks, 5 more polish experiments beat equal CPU of restarts 10 times or fewer.

**3. Cut runs would feed the priors.** M6: in the second refine 19 of 79 runs finished their p-ladder within 240 s (first refine: 78 of 80). Complete, p0 = 16 won 16 of 20; cut, wins spread over all four values. Certificates need budget, CPU-seconds and `completed`. Withdrawn if 3 lineages rerun to completion keep the cut runs' p0 order.

**4. Scores across `cells` do not compare (not blocking).** M5: 600-cell starts gained 2.4e-3 from the first refine, 800-cell starts 1.2e-3. The final #2 (600 cells) started 7th of 20; η = 3 keeps 6 or 7. Within the 800-cell class ranks moved 2 places at most, so ρ is fine there.

**5. Costs (not blocking).** 20% unpruned seeds at η = 3 take 41% of a stage's CPU. A 20-panel verdict has no panel size; at 30 seeds x 450 s: 150 CPU-hours, 12–24 USD (E10).

## Answers to the review of v1 (Meridian, producer of v2)

1. **Width.** Accepted. A new seed is a move; its bet is E[best of n] − E[best of n+1] per CPU-second; the first cut comes after one polish experiment.
2. **Leave rule.** Accepted. One price, the rate of the last move the budget can pay for, replaces the leave rule, the hedge-list order and the hand-set seed count. The survival term rests on 4 pairs: it is marked conjecture and T1 tests it on 20 forks.
3. **Cut runs.** Accepted. Certificates carry budget, CPU-seconds and `completed`; bets use completed rows of equal budget.
4. **Cells.** Accepted. Promotion ranks compare inside one (problem, cells) class; 600-cell seeds become the first recipe edit.
5. **Costs.** Accepted. Three forks per boundary replace the 20% control; fork verdicts replace panels, except for edits that change the seeds.

Kept from v1: the round at three levels, the certificate key, the five stage contracts, the claim gate, the 15-of-20 rule, standing tests. v1's T0 is reported as done (M3): in-stage promotion is sound.

M8 (file read): the tuned optimizers of 1a, 1b and 1d start from INIT by repeating cells to an integer multiple (`load_init` in `<workspace>/wt-compare/optimizers/1a_tuned.py`, `1b_tuned.py`, `1d_tuned.py`). A refine stage therefore exists on the three problems.

Check of T2's prediction (replay, 0 USD): a relay with 54 seeds, 4 kept, polish to experiment 4 and one run per refine, drawn from saved rows, beats the saved flat loop in at least 9 of 10 panels in 85% of 20,000 simulations, and in 5 or fewer in none.

Limits of M1–M7: M3's 2,000 draws overlap (100 seeds). M4 has 4 pairs. The saved polish includes the knob tuner. Every rate comes from 1d. The second-refine rows are mostly cut runs (M6).

S16 was added to the survey by Sonar at my request (question 1 of 3).

## Review of v2 by Croqui

**Verdict: OBJECT.** Kept: seeds as moves, one price, forks, `completed`, arbiter gate. C1–C4: replays of the same saved logs (0 USD; M7's seed rates reproduced).

**1. The bet is not in one unit (blocking).** The seed bet is a gain of the pool's best; the other bets are a lineage's own lift.
- C1, at v2's plan point (54 seeds, 4 kept, experiment 4), per CPU-second on 1d, 1a, 1b. v2's bet for the next polish experiment: 2.6e-6, 7.4e-7, 2.4e-8. Gain of the pool's best if all four polish: 4.8e-7, 1.2e-8, 0; if only the leader does: 1.9e-6, 4.9e-8, 0; from four more seeds: 3.7e-7, 4.1e-7, 1.7e-8. The written bet overstates polish 5x on 1d and 60x on 1a, and funds polish before seeds on 1a and 1b.
- C2, held-out: rules fitted on seeds 100–149, played on 150–199, 2,000 draws, 30 and 45 slots. v2's order gives 26 seeds, 4 kept, 2 experiments (1d, 30 slots). Pricing every move by the pool's expected gain spends 85% on seeds, 15% on the leader, none on the others. v2's plan has the worse mean in 6 of 6 cells and loses 62–88% of draws on 1d, 86–92% on 1a. On 1b it wins 83–91% of draws but its mean is worse by under 1e-5: a tie.
- C3: seed differences survive two refines with slope 0.91 (11 lineages); polish lifts 42% (M4). The seed bet has no survival term.

Withdrawn if the written bet's plan beats the pool-best rule's mean on held-out seeds of any problem.

**2. A mean over old rows cannot fund E12's move (blocking).** The record came from p0=4 at 51,200 cells, 2 of 55 restarts, on the lineage ranked 4th in its class at 12,800 cells (C4); the leader ended at 1.4558. v2 prices restarts at 0 (M7 saw p0≥16 only), gives an untried setting or edit no bet, puts restart last under thin data, and its dead ends have no scope. One such restart costs 3,300 CPU-seconds (Modal log in the snapshot): 73 seed experiments. Withdrawn if v2's rules, fed the ledger of 20:45, fund a p0=4 restart.

**3. Tonight's tests cannot fail (blocking).** T1's twin L spends its CPU on restarts 2 to 4, measured at 2.1e-7 and 0 (M7): P wins by construction. T2 passes against flat, which a hand-fixed schedule also beats (C2: seeds once, leader to experiment 10: within 2.5e-4 of the best rule). Nothing tests the price or the untried. Withdrawn if an outcome of T1 is named that M7 does not predict.

**4. Fork cost is unstated (not blocking).** Three forks at the first boundary, run to 12,800 cells, cost 3 x 428 = 1,284 CPU-seconds; T2's plan already uses 4,142 of 4,500.

**5. Reach by a rate (not blocking).** The best rate held constant gives reach 570 on 1d (10 CPU-hours); the last stage's mean rate is about 3e-8 (my estimate from E12), and that stage won. Count rows whose lift exceeds the gap.


## Addendum to the review of v2 (Croqui; scope note of 22:00 PT)

The scope note came after v2, so these are gaps, not errors.

6. **Autonomy.** v2 does not say who makes E12's hand-made decisions: writing the refine stage, lengthening cut runs, choosing p0=4.
7. **Generality.** It does not say what a new problem needs before the system can start.
8. **Model.** No decision is given to a model, and no test compares a model with a rule.
9. **Two systems, decision log.** T2 compares with one saved loop on other seeds; there is no rule for comparing whole systems and no reader's view of the ledger.

## Answers to the review of v2 (Croqui, producer of v3)

1. **Unit.** Accepted. Every bet is the expected gain of the pool's best: mean over matching rows of max(0, lift − gap). The seed bet is the case gap = 0. Seeds get a survival term (0.91, C3).
2. **E12.** Accepted. max(0, ·) keeps a rare large lift. A probe share (10% of a round) buys untried settings on every kept lineage. A refutation carries its scope.
3. **Tests.** Accepted. v2's T1 is dropped. The new T1 probes settings that no row predicts. T2 takes the hand-fixed schedule as the arm to beat. T3 and T4 compare a model with the rule.
4. **Fork cost.** Accepted. Forks are paid from the probe share.
5. **Reach.** Accepted. Reach is the share of rows whose lift exceeds the gap.
6–9. **Scope note.** Accepted: sections "Autonomy: E12 without us" and "Decision log", the "General" and "Two systems" lines. A model decides only inside the probe share.

Kept from v2: the price, moves as bets, kept_count, certificates with cost and `completed`, per-class ranks, the arbiter gate, forks, the 15-of-20 rule (now with a positive mean gain: on 1b v2's plan wins the draws and loses the mean, C2).

Known weak points of v3: the probe rule (a 4x step outside the tried range) was written after E12. The new bet was replayed at the polish stage only. There the hand-fixed schedule (seeds once, leader to experiment 10) is about as good as the bet (C2), so T2 can remove the price.

**Method of C1–C4** (0 USD, read-only, inline scripts, nothing saved). Old-loop rows in `<workspace>/shared/snapshots/2026-11-14_2045/ab_comparison/` (`modal_results_A`, `modal_results_A_1a`, `modal_results_A_1b`): 100 seeds x 10 experiments, best-so-far after each experiment, 45 s per experiment.

- C1: 4,000 draws of 54 seeds; 4 kept by first-experiment score and polished to experiment 4; gain of the pool's best from the next experiment on all four, on the leader alone, or from four more seeds.
- C2: v3's bet played greedily, one experiment at a time. Rows from seeds 100–149 (10 nearest in start score); 2,000 draws of seeds 150–199. Mean best score, in the order bet, v2's written bet, seeds once with the leader to experiment 10, flat. 1d at 30 slots: 1.458818, 1.459443, 1.459046, 1.464702; at 45: 1.457751, 1.458526, 1.457998, 1.463496. 1a at 30: 1.507777, 1.508079, 1.507853, 1.510930; at 45: 1.507343, 1.507578, 1.507213, 1.510468. 1b at 30: 0.381112, 0.381121, 0.381133, 0.381269; at 45: 0.381090, 0.381095, 0.381101, 0.381247. The bet spent about 85% of its slots on new seeds on 1d, 65% on 1a and 100% on 1b; the rest went to the leader.
- C3: `<workspace>/shared/snapshots/2026-11-14_2045/c1d_pathB_partial/B`, 11 lineages started at 800 cells: slope of the best 12,800-cell score on the start score 0.91 (r = 0.97); 0.92 after the first refine.
- C4: same files, 800-cell class at 12,800 cells: A106 1.4558843, B106 1.4560605, B162 1.4560715, A162 1.4560723.

Limits: C1 and C2 cover the polish stage only. The 50 held-out seeds are reused across draws. The fixed schedule beats the bet on 1a at 45 slots. C3 and C4 use cut second refines (M6). E12's 55 restarts are not in the snapshot and were not replayed; the 3e-8 of objection 5 assumes three lifts of 1.3e-3 to 1.7e-3 and 52 near zero. The 3,300 CPU-seconds per deep restart come from `<workspace>/shared/snapshots/2026-11-14_2045/c1d_pathB_partial/modal_R1_51k.log`.

S17 was added to the survey by Sonar at my request (question 1 of 3).

OpenAI reviewer: not run on v3 yet. Croqui's role forbids paid calls, and Murilo is being asked in Croqui's terminal. If Meridian's role allows it, Meridian may run `review` on v3.


## Review of v3 by Meridian

**Verdict: OBJECT.** Accepted: the bet on the pool's best (I reproduced C3 and C4), the probe share, scoped refutations, the two-systems rule, the log line. D1–D3 are replays, 0 USD: `<workspace>/wt-attack/attack/c1d/B/` (rows of 20:45 with their vectors) and `<workspace>/wt-attack/attack/c1d/modal/R1_51k/` (E12's 40 deep restarts). Displacement: L1 distance between parent and child as unit-mass step functions (0 to 2).

**1. "E12 without us" rests on a rule written after E12; the ledger of 20:45 gives a better one (blocking).**
- D1, rows of 20:45: displacement falls as p0 grows, in 19 of 20 lineages at the first refine (medians 1.07, 0.94, 0.73, 0.65 for p0 16, 64, 512, 1024) and 17 of 20 at the second. The lowest p0 tried won 16 of 20 (M6).
- D2, E12: the three lifts over 1e-3 moved 1.88 to 1.97. Every run with p0 ≥ 16 moved at most 1.19 and lifted at most 6e-5.

"Probe one step past the edge where the best rows sit" names p0 = 4 from rows older than E12, and never 4096. v3's rule spends half its probes on the wrong side. Certificates need `displacement`. Withdrawn if T1 finds p0 = 4096 ahead of p0 = 4 on more lineages.

**2. No test runs where the record was won (blocking).** T2 and T4 give each arm 4,500 CPU-seconds per panel: the probe share is 450 s, and one deep probe costs 3,300 s. D3: E12's eight soft restarts cost 7.3 CPU-hours, 11% of the chain's 68 CPU-hours (200 runs, path B, R1). The share buys them only in one run of that size. Three of the eight came back as the parent (displacement 0) while their twins at p0 = 4.04 moved 1.73 to 1.97: a probe needs two runs and `completed` (M6). Needed: one autonomous run on 1d from fresh seeds, about 70 CPU-hours, with a prediction on soft-restart lifts. Withdrawn if T1 shows soft restarts pay at 78 s: panels then reach them.

**3. The model's place is not tested (blocking for scope point 3).** T3 scores how early seed 162's lineage is probed. No row of 20:45 separates it from the three lineages ranked above it (C4): the score is a tie-break. With D1 a rule already finds the setting. The one E12 decision no rule can make is writing the refine stage (v3's decision 2), and nothing tests it. Needed: the model writes that stage from the published method and the contract; pass if it lifts 4 of 5 saved 800-cell certificates by 1e-3 (E5). Withdrawn if T3 scores something the ledger can inform.

**4. Reach zero (not blocking).** At 12,800 cells on 1d no row's lift exceeded the gap of 1.2e-3: reach was zero where E12 won. Say whether the problem then keeps 10% or all its unspent budget for probes.

**5. Log (not blocking).** "Sorted by surprise" shows errors first. A judge also needs the claim's ancestry: every decision from seed to certificate.


## Review of v3 by the OpenAI reviewer (gpt-6.1-sol)

## Outside review of v3

**Verdict: OBJECT**

### 1. The allocator can learn from censored results and suppress late winners.

“Matching rows” excludes incomplete runs. Yet runtime changes when fewer than half finish. Completion therefore determines both which results count and the next budget. Slow, promising moves can disappear from the evidence. The survival multiplier and `kept_count` also compress stage-dependent rank reversals into historical averages. E12 directly challenges that assumption.

**Evidence:** E7c, E9, E12; S9, S17. M1–M8 and C1–C4 are not supplied, so I cannot check their support.

**What would change my mind:** Log every attempted move, including timeouts and failures. Define their allocation treatment. Run a held-out test with slow winners and rank reversals. Compare against a fixed relay under the same total budget.

### 2. The tests do not establish autonomous discovery.

T3 rewards finding the already-known E12 winner. The policies, prompts, and architecture were designed after that result. Hiding the later ledger does not remove this contamination. T1 tests whether an early restart helps, not whether the whole loop discovers a successful chain. No tonight test exercises recursive execution, budget decisions, recovery, and certification together.

**Evidence:** E5, E8, E12; S3, S15; T1 and T3.

**What would change my mind:** Add a small end-to-end run on an untouched problem or withheld lineage set. Freeze policies and scoring first. Charge failed experiments and model calls. Report whether it completes a verified chain without intervention, even if it does not beat a record.

### 3. Model-written stages need an execution boundary, not only a contract check.

A stage can return valid-shaped output while altering the ledger, verifier, reference cache, or another stage. Parent hashes do not prevent this. This is an integrity failure that reviewers focused on numerical correctness could miss.

**Evidence:** E7a–b; S1, S13, S14. Test: a stage attempts to overwrite its verifier and fabricate its score.

**What would change my mind:** Specify isolated execution, read-only inputs and verifier, restricted credentials and network access, and independent score recomputation. Include an adversarial stage test.

### 4. Several mandatory decisions remain undefined.

“The price is the bet of the last move the budget can fund” needs an ordering and a finite candidate set. Probe selection does not specify arbitration among lineages, edits, and boundaries. “None on small budgets” lacks a threshold. `round(1)` also needs an explicit executable base case.

**Evidence:** E3, E7d; S6, S16.

**What would change my mind:** Supply deterministic rules and a short ledger replay covering cold start, exhausted probes, unaffordable moves, and termination.

### 5. Claim freshness is not claim completeness.

A reference read within 30 minutes can still omit an open claim, parse it incorrectly, or fail silently.

**Evidence:** E7a, E6; S14.

**What would change my mind:** Define authoritative sources, retain timestamped source snapshots, and fail closed on retrieval or parsing failure. Distinguish “beats the checked snapshot” from an unqualified record claim.


## Answers to the reviews of v3 (Meridian, producer of v4)

**To my own review.**

1. **Edge rule.** Accepted. Probes go one grid step past the edge of a setting's range where its best rows sit, else toward larger displacement. Certificates carry `displacement`.
2. **Scale.** Accepted. T5 runs the whole loop with a frozen policy and nobody in it: small on 1a, full on 1d from fresh seeds with an empty ledger. Probes run twice. T2 and T4 now say that their panels cannot buy a deep probe.
3. **Model test.** Accepted. The old T3 (probe order on E12) is dropped. The new T3 has a model write the refine stage. T4 stays.
4. **Reach zero.** Accepted. The probe share is then half the budget (conjecture; E12's chain spent 54% of its CPU on the deep stage, D3). So 30 CPU-hours are enough for T5, not 70.
5. **Log.** Accepted. A claim's page lists its ancestry.

**To the OpenAI reviewer.**

1. **Censored results.** Accepted. Every attempt is a row at full cost, cut and failed runs included, and rows match at equal run budget. The run budget doubles while under half a move's runs complete or leave the parent. A probe that becomes the pool's best is rerun on every kept lineage before its mean is used, so one win among zeros is not averaged away. D3 is the case: three of four first runs at p0=4 returned their parent, and their twins won. Rejected in part: kept_count is not an average. It is a recall count per boundary, and E12's reversal sits inside the kept set (rank 4 of 5). The held-out test with slow winners is T5.
2. **Tests.** Accepted. T3 as written is dropped. T5 is the end-to-end run, policy and scoring frozen first, failed runs and model calls charged.
3. **Execution boundary.** Accepted. Each stage runs in its own process on copies, without credentials or network. The gate recomputes the score and checks the frozen hashes. T0 includes a stage that fakes its score.
4. **Undefined decisions.** Accepted. The candidate set is finite: one new seed and, per kept lineage, one more run or its next stage. Order: falling bet, ties to the older lineage. "None on small budgets" becomes "unaffordable probes are skipped". Level 1 is the base case: one run of a stage file. T0 replays cold start, exhausted probes, an unaffordable move and termination.
5. **Claim completeness.** Accepted. The pack names its sources, each read keeps a timestamped snapshot, a failed read or parse means no claim, and a claim says "beats the snapshot read at <time>".

The outside reviewer could not check M, C and D numbers because its review is blind to this log by design.

**Method of D1–D3** (0 USD, read-only, inline scripts, nothing saved). Displacement: both vectors divided by their sum, the parent's cells repeated to the child's resolution, then the L1 distance. First refine: parent `B/<lineage>_n<N>.npy`. Second refine: parent is the best first-refine output. Deep stage: parent `L_<lineage>_n12800.npy`, children in `modal/R1_51k/`. A run whose saved best vector is its parent has displacement 0. D3's 68 CPU-hours: 200 runs x 450 s (25.0), path B (6.6, from the `seconds` fields), R1 at 51,200 cells (40 x 3,300 s = 36.7).

Limits: D1 and D2 cover one stage family on 1d. The edge rule was also written after E12; what is older than E12 is the rows it reads. In D2 a displacement of 0 can mean "cut before improving", not "came back".

Kept from v3: the bet on the pool's best, price, probe share, kept_count, scoped refutations, the two-systems rule, the log line, the arbiter gate.


## Review of v4 by the OpenAI reviewer (gpt-6.1-sol)

## Review of v4 by outside reviewer

**OBJECT**

### 1. The allocator rewards the wrong quantity for new seeds.

A new seed gets gap zero. Its positive lift therefore counts even when it cannot improve the pool. Other moves must overcome their actual gap. These bets are not comparable. The fixed survival factors also suppress uncertainty about late-stage rank reversals, precisely the failure shown by E12.

**Evidence:** E1, E3, E12; S9, S17.

**Would change my mind:** Define every bet as improvement over the same verified pool incumbent. Define the starting baseline for seeds. Replay a synthetic ledger where inferior seeds improve greatly but cannot beat the incumbent. Show that they do not displace a productive refinement, except through the explicit exploration budget.

### 2. The proposed validation risks rediscovering its own design data.

The edge rule targets p0=4 after E12 identified it. Retention counts and survival factors also come from saved measurements. Replaying saved lineages cannot establish that this system would discover the chain independently. Fresh random seeds do not erase knowledge embedded in the policy.

T5 also overinterprets failure. Zero successes among eight lineages would challenge the stated success prediction. It would not establish that “E12 was seed luck.”

**Evidence:** E7c, E8, E12; S15, S17.

**Would change my mind:** Label 1d as development data. Freeze the policy before an unseen problem or withheld stage-setting family. Compare against the fixed relay with all development costs disclosed. Narrow T5’s conclusion.

### 3. The core numerical rules cannot be audited from this proposal.

M1–M8, C1–C4 and D1–D3 supply critical measurements but are not provided here. “Lift,” “class,” matching, survival estimation and sparse-row fallback remain underspecified. Price depends on bets that themselves depend on the retained pool. Two implementations could make different decisions while following this text.

**Evidence:** E3, E7d; S10, S15. **Test:** T0.

**Would change my mind:** Include a compact estimator specification and one worked allocation with failed runs, an unseen setting and a late-blooming lineage. Give expected decisions, not merely repeatability.

### 4. Problem identity does not establish metric identity.

A frozen verifier and recent reference snapshot can still measure different normalizations, admissible objects or objective conventions. Hashes and timestamps would then certify the wrong comparison. This is a shared blind spot: independent numerical calculations can agree on the same mistaken specification.

**Evidence:** E7a–b, E12; S13, S14.

**Would change my mind:** Version the mathematical specification in each certificate and reference. Recompute an adopted public certificate under the local verifier before allowing record claims. Add a deliberately mismatched-reference test.

### 5. Build feasibility excludes expensive acceptance work.

T2, T4 and T5 occur “after the build.” T3 has no time estimate. Eleven agent-hours do not establish delivery by 10:00 PT.

**Evidence:** E10; tests T2–T5.

**Would change my mind:** Give a wall-clock dependency schedule, concurrency limits and a minimal autonomous acceptance run before the deadline. Mark untested recursion and model editing as deferred.


## Review of v4 by Croqui

**Verdict: OBJECT**, narrowly. v4 is better than v3: the edge rule read from rows (D1) replaces my 4x step, T3 tests the one decision no rule can make, cut runs are rows, stages run behind a boundary. I could not check D1–D3: their files are outside what my role may read. Two holes block; each needs one sentence.

**1. An empty ledger cannot spend its budget (blocking).** With no rows there is no bet. Every move then counts as a probe, and the probe share is 10% (half at reach zero). The loop reaches `bet is None` with at least half the budget unspent. T5 on 1d starts exactly there, and so does every new problem. "No rows: seed, polish, refine, restart, in order" gives the order, not who pays. Withdrawn if v4's text shows how the first 5 rows of a default move are paid outside the share.

**2. A new seed's lift is undefined (blocking; the outside reviewer's objection 1 to v4).** "Gap 0 for a new seed" leaves a lift with no parent. C2 used max(0, pool best − the row's first score). Without that sentence two builders code two allocators. Withdrawn if the text already fixes it; I do not find it.

**3. Nothing runs on a problem the rules were not fitted on (not blocking; outside objection 2; E8).** M, C and D come from 1d, 1a, 1b. The loop also has 10c and 11a: higher is better, no step function, no refine stage, never touched by our rules (`problems/p10c.py`, `problems/p11a.py` on `origin/feature/ws2-autoresearch-loop`). A frozen-policy relay against the flat loop there, 10 panels of 30 experiments, costs about 7.5 CPU-hours per problem, 0.6–1.2 USD (E10). T5's "E12 was seed luck" should read "the prediction failed on these seeds".

**4. The edge rule's rows (not blocking).** At a stage with no rows of its own, which rows does it read? D1's rows sit at the first and second refine; E12 won at the deep stage. And displacement is an L1 distance between step functions: a pack for 10c needs its own distance, or the rule loses its "else".

**5. The estimator cannot be coded from the text (not blocking; outside objection 3).** Lift, class, pool and survival to the end of the chain need one definition each, and one worked allocation with expected decisions.

**6. No clock (not blocking; outside objection 5).** 11 agent-hours, T5's 30 CPU-hours and a deep stage of 55 minutes per run must fit before 10:00. Say what runs when, and what is dropped first.

**7. Where edits come from (not blocking).** v4 says who writes an edit's code, not what the model reads (the map's ideas with their Exa and forum citations, the ledger). D1 itself is an edit to a rule, found by reading rows. Rule edits can be gated by a held-out replay (C2) before any CPU is spent.


## Answers to the reviews of v4 (Croqui, producer of v5)

**To my own review.**

1. **Cold start.** Accepted. A default move with under 5 rows is funded first, in stage order, on the kept half, outside the probe share.
2. **Seed lift.** Accepted. The bet is now written on outcomes: max(0, best − outcome). For a new seed the outcome is the row's first score.
3. **Unseen problems.** Accepted. 1d, 1a, 1b are labelled development data. T5a runs the frozen policy on 10c and 11a against the flat loop and the hand-fixed schedule. T5b's conclusion is narrowed.
4. **Edge rule's rows.** Accepted. A stage with no rows reads the nearest earlier position. The distance behind `displacement` belongs to the problem pack.
5. **Estimator.** Accepted. "Terms" defines best, outcome, survival; W1 below gives the worked allocation. It is in the log because the proposal is at its 1,200 words.
6. **Clock.** Accepted, as a conjecture: one paragraph with what is dropped first.
7. **Edits.** Accepted. The model reads the contract, the ledger and the map (Exa, forum). A rule edit must first win a held-out replay of the ledger.

**To the OpenAI reviewer's objections to v4.**

1. **Seeds rewarded for the wrong quantity.** Accepted. Every bet compares an outcome with the same incumbent, the best score of the round's pool; a seed whose first score cannot beat it adds zero (W1, case g). On survival: accepted in part. The factors are means from 1d and are 1 without rows. Late reversals like E12 are carried by the late stage's own rows, through max(0, ·), and by kept_count, not by survival.
2. **Validation on design data.** Accepted. Development data are named; T5a is the frozen-policy run on problems our rules never touched; "E12 was seed luck" is replaced by "the prediction failed on these seeds". `06_final.md` will list development compute (E12's chain: 68 CPU-hours, D3; the 500 comparison runs of E2 and E8: 62.5 CPU-hours). Not done: a withheld family of settings; we have none tonight.
3. **Rules cannot be audited.** Accepted in part. The estimator's terms are now in the proposal and W1 has expected decisions for a cut run, an unseen setting and a late-blooming lineage. M, C and D stay in this log: the review is blind to it by design, the `reply` mode is not.
4. **Metric identity.** Accepted. Certificates and reference snapshots carry the verifier's spec version; an adopted certificate must reproduce its published value under the local verifier; T0 refuses a reference with another spec version.
5. **Build feasibility.** Accepted. The proposal has a clock, a minimal acceptance run before the deadline (T5a) and an order of what is dropped: model edits first, then T5b. T4 is deferred. The clock assumes two builders; that is not confirmed.

Kept from v4: the edge rule and displacement, rows for every attempt, the run-budget rule, the boundary, T0, T1, T3, snapshots, the ancestry page. Dropped: T2 as a separate test (the hand-fixed schedule is now an arm of T5a).

**W1. Worked allocation, with the decisions T0 must reproduce.** Numbers per CPU-second.

- State (C1): 1d, a pool of 54 seeds, 4 kept, all at experiment 4.
- (a) One more polish on the leader: gap 0, rows give 1.9e-6; times survival 0.42: **8.1e-7**.
- (b) A new seed: mean of max(0, best − first score) over the first-experiment rows, 3.7e-7; times 0.91: **3.4e-7**.
- (c) One more polish on the third lineage: no row's lift exceeds its gap: **0**.
- Decision: (a), then (b) while its bet stays above the price; (c) never as a bet, only as a fork.
- (d) Cut run: a run that hits its run budget and returns its parent is a row with lift 0 at full cost. If under half of a move's runs complete, the run budget doubles; rows match only at equal run budget.
- (e) Unseen setting: p0=4 at the deep stage, ledger of 20:45. No rows, so no bet. The edge rule names it, because the best rows of p0 sit at the low end of the tried range (D1). It is probed on each kept lineage, twice.
- (f) Late bloomer: after E12 the rows of p0=4 at the deep stage are three lifts of 1.3e-3 to 1.7e-3 and five taken as 0 (my reading of D3; a lower bound). A kept lineage 2e-4 behind bets (1.47e-3 + 1.43e-3 + 1.14e-3) / 8 = 5.1e-4 per run, **1.5e-7** at 3,300 s. It is funded although it is not the leader.
- (g) A seed whose first score is worse than the pool's best contributes max(0, ·) = 0 to (b).
- Cold start: with an empty ledger the first moves are default seed runs, then default polish, then default refine, each until it has 5 rows.
- Termination: no bet above the price and the probe share spent.

OpenAI reviewer: not run on v5 by Croqui yet. Croqui's role forbids paid calls; Murilo was asked in Croqui's terminal and has not answered.


## Review of v5 by Meridian

Written before the hand-over: Croqui's terminal is waiting for Murilo's answer on the paid call, and v5 is complete on disk. **I have no blocking objection. I will write AGREE or OBJECT after the outside reviewer's `reply`, as the protocol asks.**

v5 closes both holes of v4 (cold start, seed outcome) and the outside points on spec version, clock and development data. I recomputed W1 (a), (b) and (f): they follow from C1 and D3. Five corrections for the next update. D4 and D5 are file reads, 0 USD.

1. **T5a cannot fail, and the price lost its only discriminating test.** D4: `<workspace>/wt-compare/optimizers/10c_tuned.py` runs at N=8, where its own benchmark best is 5/sqrt(8), the proven optimum (`problems/p10c.py`); scores move in steps of 1/sqrt(8), so arms tie. `optimizers/11a_tuned.py` starts every seed from the same linear function. Neither takes INIT: no polish stage exists, so the relay there is seeds only. T5a shows that the loop runs on a pack it never saw. It cannot show a winner. v4's T2 (hand-fixed schedule on 1d and 1a, a tie in C2) was folded into it. Correction: call T5a an integration run; keep the hand-fixed schedule as an arm on 1d and 1a. I do not block because T1, T3 and T5b can fail.
2. **Probe share with no rows.** With an empty ledger no row's lift exceeds the gap, so half the budget is reserved for probes from the first move, and few probes are legal: a 30-experiment panel would run 15. Correction, as W1 cases: reach exists once a move has 5 rows; an unspent share returns to the bets; a range of one tried value has two edges, so the default's two grid neighbours are the first probes.
3. **Direction and code.** 10c and 11a maximise: "best − outcome" must follow the pack's direction. The certificate lost `code` (the stage file's hash) between v4 and v5; a claim cannot be replayed without it.
4. **Where E12 started.** D5: `<workspace>/shared/1d_starts_1000/` holds 421 fresh single runs of 450 s (seeds 200–809). Pooled with the 100 control runs, seeds 162 and 106 rank first and second of 521 (1.4571836, 1.4574351); the third is 1.4579659. So T5b's prediction, a soft-restart lift over 1e-3, is the right one, and passing the claim from fresh seeds is not expected. The builder also plans a run from today's ledger (`<workspace>/shared/autoloop_brief.md`, test C): that tests exploitation, not discovery. Report both, labelled.
5. **T4 deferred.** Then `policy.probe` stays rules until T4 runs (two-systems rule: the simpler stays).


## Second look at v5 by the OpenAI reviewer (gpt-6.1-sol)

**Verdict: OBJECT**

### Resolved

- **Seed scoring.** V5 and W1(g) now compare each seed’s first score with the pool incumbent. An inferior seed contributes zero. This resolves my v4 objection about incompatible rewards (E1, E3; S17).
- **Reference identity and scope.** Spec versions, local reproduction of adopted certificates, source snapshots and fail-closed reads address my metric and reference objections. Claims are correctly limited to the checked snapshot (E7a–b; S13, S14).
- **Design-data contamination.** The proposal labels 1d, 1a and 1b as development data. It narrows T5b’s conclusion. This resolves the reporting objection, not the validation gap (E8, E12; S15).
- **Auditability improved.** The supplied log makes M, C, D and W1 inspectable. Failed attempts now incur their full cost. These resolve the missing-evidence and omitted-cost parts of my earlier objections (E7c–d; S15).

### Objections that stand

1. **The allocator lacks a discriminating acceptance test.** Meridian’s D4 says T5a can tie by construction and cannot exercise polish. T1 tests a setting. T3 tests stage generation. T5b tests a soft-restart prediction. None isolates whether the price beats a fixed relay.  
   **Change required:** Restore the equal-budget comparison against the hand-fixed schedule on problems with meaningful choices. Freeze the win rule first. Test slow winners and rank reversals explicitly (E9, E12; S9, S17).

2. **A separate process is not an integrity boundary.** Copies, absent credentials and independent rescoring help. They do not prevent writes to the verifier, ledger or reference cache under shared filesystem permissions. T0’s fabricated score does not test this.  
   **Change required:** Specify enforced read-only trusted files and isolated writable storage. Test attempted verifier and ledger modification (E7a–b; S13, S14).

3. **Budget behavior remains incomplete.** Meridian identifies unspent probe reservations and undefined single-value edges. Cold start also needs an explicit retention fallback. Proposed corrections are not yet in v5.  
   **Change required:** Incorporate them and publish expected T0 decisions through budget exhaustion (E3, E7d; S16).

### New objection

**Maximization and replay are broken as written.** The lower-is-better formula misranks 10c and 11a. Certificates omit the stage-code hash.

**Change required:** Normalize objective direction in the pack. Restore immutable code hashes and test replay (E8; S2, S13).


## Verdict on v5 by Meridian: OBJECT, narrowly

The outside reply and one new measurement move correction 1 of my review from "next update" to blocking. Version 6 is mine and is the last one under the cap.

1. **The allocator has no test that can fail (blocking; criterion 5).** T5a ties by construction (D4) and v4's T2 was folded into it. T1 tests a setting, T3 a stage, T5b a soft restart. Nothing can show that the price is worse than a simpler rule.
2. **The system to beat now exists (new).** D6 below: the builder's loop ran alone from today's ledger and passed its claim gate. Its present rules are the incumbent of any two-systems comparison, not C2's schedule.
3. Corrections 2 to 5 of my review stand. The outside reviewer adds: writes must be prevented, not only detected.

**D6** (file reads, 0 USD): `<workspace>/shared/1d_candidate_2026-11-14_2117/autonomous_rules_arm/ledger.jsonl`, the three logs in `<workspace>/wt-attack/autoloop/runs/C_tournament/`, `<workspace>/wt-attack/autoloop/policy.py` and `<workspace>/wt-attack/autoloop/runs/cold1.log`.

- The rules policy, with no person in the loop: read the reference (open pull request 345, 1.4546733414), restart the best certificate of its memory six times at 51,200 cells (p0 = 2, 4, 8, 16, 64, 512; 1,200 s each; 2 CPU-hours on Modal), certify the best. Exact 1.4541470296475214, below the reference read 1,243 s earlier and below the best exact in its memory (1.4542590631143155). Not submitted.
- Its rules: read the reference when it is old; seed until there are enough lineages; promote the top three in turn, one resolution level up first, then a fixed list of restarts from soft to hard; a lineage stalls after two refinements without gain; certify at the top resolution.
- p0 = 8, 16, 64 and 512 lowered the leader by 1.1e-4, 8.9e-5, 5.4e-5 and 4.8e-5. p0 = 2 and 4 returned their parent. In E12 p0 = 4 won at 3,300 s per run; here the run budget was 1,200 s. The present rules cannot tell a cut run from a bad setting. This is the outside reviewer's censoring case, observed.
- The same first move (the same certificate, the same six p0) ended at 1.4541470 under the rules and at 1.4541570 under each of the two model policies. One run per policy cannot rank policies (E7c).
- One model policy certified before it read the reference. The claim gate refused the claim. Gates stay rules.
- The cold start now running is steered by a model whose reasons cite E12: its prompt holds our evidence. It is not a blind discovery run.

## Answers to the reviews of v5 (Meridian, producer of v6)

**To my own review and verdict.**

1. **Allocator test.** T2 is back: these rules against the loop's present rules, on 1d and 1a from empty ledgers, under the two-systems rule. If it is not run, the present rules stay. T5a is dropped: it could not show a winner, and the loop already has a second pack (1a). The proposal says that E8 stays open.
2. **Probe share.** Reach needs 5 rows per move. An unspent share returns to the bets. One tried value has two edges. W2 gives the expected trace.
3. **Direction and code.** The pack names its direction. Certificates carry the stage file's hash.
4. **Where E12 started.** T5 says that passing the claim from fresh seeds is not expected (D5), and the autonomy section separates exploitation (done, D6) from discovery (T5).
5. **T4.** `policy.probe` is rules until a model wins a two-systems test.

**To the outside reviewer's second look.**

1. **Discriminating acceptance test.** Accepted: T2 as above. The win rule is the two-systems rule, written before any run. Slow winners and rank reversals: W1 (d) and (f) are expected decisions of T0, and T5 runs the deep stage where both occurred. Not done: an unseen problem with meaningful choices. None exists tonight (D4).
2. **Integrity boundary.** Accepted. A stage can write only to its scratch directory: a sandbox profile locally, a fresh container on Modal. T0 includes a stage that tries to write outside it.
3. **Budget behaviour.** Accepted. The rules are in the text, and W2 runs to budget exhaustion.
4. **Maximisation and replay.** Accepted: direction is a field of the pack, and the code hash is back. Rejected in part: a rerun cannot be required to rebuild a certificate bit for bit, because stages stop on a wall clock (D6: one move, three endings). What replays exactly is the decision sequence from the ledger (T0) and the score of a stored certificate.

Dropped from v5: T5a, the clock paragraph (the loop exists and has run end to end, so only the order of what is dropped remains), T4 as a named test.

**W2. Worked budget trace, with the decisions T0 must reproduce.** A pack with a seed stage and a polish stage; polish has one setting with default d and grid {d/4, d, 4d}; lower is better; empty ledger; one round of 30 runs of equal cost.

- Runs 1–5: default seeds (cold start).
- Kept half, no rows yet: the best 3 of 5.
- Runs 6–10: default polish on the kept half, in rank order (2, 2, 1 runs).
- Both moves now have 5 rows, so reach is defined. If no row's lift exceeds the gap to the live reference, the probe share is 15 runs; otherwise 3. Cold start is paid from the other side: 15 − 10 = 5 runs are left for bets (17 if the share is 3).
- Bets: the best of {one more polish on each kept lineage, a new seed}, recomputed after each row. A lineage whose rows never exceed its gap gets none.
- Probes, in order: d/4 and 4d on each kept lineage, because one tried value has two edges (6 runs); the same again (6); one fork, a cut seed that gets a default polish (1); no edit is queued. 13 probes are legal, so 2 runs of the share return to the bets.
- A stage with no rows reads the edge at the nearest earlier position (W1 (e)).
- A probe that becomes the pool's best is rerun on every kept lineage; those reruns are its "same again" runs.
- A probe that costs more than the share left is skipped.
- Termination: 30 runs spent. Earlier only when no bet is above the price and no probe is left.
- On a maximise pack every comparison flips.

The blind OpenAI review of v5 was not run (Croqui was waiting for approval; v6 replaces v5). The `reply` run is above.


## Review of v6 by the OpenAI reviewer (gpt-6.1-sol)

OBJECT

1. **The policy’s main estimate can erase the interaction that produced E12.**  
   Rows match on move, position, class and budget. They do not explicitly match on lineage or parent features. Yet E12 says the winning effect depended on seed 162 *and* p0=4. Multiplying average lift by average survival also need not equal expected final improvement. S17 warns about dependent arms and sparse tails. This can make a precise-looking price rank the wrong bets.  
   **Change my mind:** Define the estimator completely, including sparse-data fallback and conditioning. Test it on synthetic rows with seed–restart interactions and reversed intermediate rankings. Then prospectively compare its allocations against a simple lineage-diverse relay at equal total cost.

2. **The tests do not establish autonomous discovery, and the proposed inference is weaker than claimed.**  
   T0 replays an existing ledger. T3 supplies the published refinement method. T5 tests a local improvement, not discovering and completing a competitive chain. None distinguishes autonomous research from replaying decisions fitted to E12. All named development problems already informed the policy. S3 and S15 motivate evaluating searchers, not just recovered outcomes. Also, 15 wins in 20 paired forks does not control repeated testing across many proposed edits.  
   **Change my mind:** Freeze the policy and prompts. Run an empty-ledger, end-to-end comparison on a withheld problem or method family. Count failed edits and all selection costs. Predeclare one evaluation batch or a sequential testing rule. Treat tonight’s smaller tests as component checks, not evidence of autonomous discovery.

3. **Budget control and retention are not executable rules yet.**  
   “Under 5 rows” cold starts sit outside the probe share. Winning probes rerun on every kept lineage. Recursive recipe edits can introduce further moves requiring cold starts. The proposal does not say which obligation loses when these exceed the budget. `kept_count` also lacks a definition before final-stage outcomes exist, or when no final top-three exists. These gaps matter directly to E7d and E12’s late rank reversal. S9 provides bounded schedules, not just priorities.  
   **Change my mind:** Specify reservation, cancellation, fallback retention and deterministic tie rules. Show an empty-ledger trace with a tiny budget, failures and a newly introduced stage. Every recursive call must remain within the same global CPU and USD caps.

4. **The trust boundary is incomplete for model-written rules.**  
   Stage sandboxing does not establish that edited policy code cannot alter the ledger, verifier inputs, reference handling or budget accounting. Hash checks alone do not make those components independent. E7 and S13–S14 make this a claim-integrity issue, not optional hardening.  
   **Change my mind:** Keep policy edits declarative, or isolate their execution too. Give them read-only observations and accept only validated actions. Add T0 attacks against ledger mutation, forged parentage, budget bypass and verifier replacement.


## Review of v6 by Croqui

**Verdict: AGREE.** No blocking objection. v6 repairs my two errors in v5 (T5a could not show a winner, D4; the code hash was lost) and takes the right incumbent for T2: the loop that already runs (D6). I could not check D4–D6: those files are outside what my role may read. W2 adds up at reach zero: 10 cold-start runs, 7 bets, 13 probes, 30 in all. Five notes for the first phase 2 update, ranked. None blocks.

1. **Build first what pays without T2.** If T2 is not run, the present rules stay and the 3 hours for "this policy" buy nothing. Three pieces improve the running loop by themselves: run-budget doubling with `completed` (D6: p0 = 2 and 4 returned their parent at 1,200 s; E12 won with p0 = 4 at 3,300 s), the decision-log line (a judging criterion), and the claim snapshot with its spec version. Order: these, then the sandbox and T0, then the policy and T2.
2. **Say what T2 can and cannot show.** Panels of 4,500 CPU-seconds never reach a deep run of 3,300 s. T2 therefore compares the two rule sets on seeds, polish and the first refines only, where C2 found the bet about level with a fixed schedule. A tie is the likely result and keeps the present rules. It says nothing about the deep stage, which only T5 runs, and T5 has no second arm.
3. **Rows lost their conditioning.** v3 matched rows "nearest in start score" and C2's held-out result used the 10 nearest. I dropped the words in v5 to save space. Restore them, or say that all rows are used. A variant I ran with C2 and did not log (all rows instead of the 10 nearest) changed the mean best by under 4e-5 on 1d, 1a and 1b, so either is defensible; the text must say which. It is also half an answer to the outside reviewer's objection 1 to v6.
4. **Model-written rules need more than the stage sandbox (outside objection 4 to v6).** A stage can write only to scratch, but an edited rule runs inside the loop. For tonight: model edits are stages only, and a rule edit is a text proposal that a person applies after it wins the held-out replay. This matches "dropped first if late: model edits".
5. **Two text gaps (outside objection 3 to v6).** kept_count needs its fallback where it is defined: half, until a final top-3 exists. And when cold start, the reruns of a winning probe and a new stage's cold start together exceed the budget, the text must say which is cut first. I propose: reruns first, then the cold start of new stages.

What would make me object: a trace in the style of W2 in which the rules as written leave budget unspent or spend past the cap. I looked for one and did not find it.


## Evaluation of v6 by Lupa (phase 2, exchange 1 of 4; 15 Nov, 00:40 PT)

**Not CONSENSUS yet: 21 of 30. W1 and W2 block.** I read v6 and wrote my objections before opening this log; the scores were set after reading it and the code. L1–L8 are my own file reads and reruns (0 USD, nothing edited); method at the end.

**Scores (0–5)**

| # | Criterion | Score | Why |
|---|---|---|---|
| 1 | Explains the evidence | 4 | M, C and D re-measured E1–E12 and dropped what lost. Missed: E3 returns at the deep stage (W3); E2 with E12 is not carried into the two-systems rule (W4). |
| 2 | Recursive and chained | 3 | Chained: real, already in code. Recursive: level 1 is a base case, level 3 has no rows and is dropped first, the ladder only goes up, the code is one flat `while` (L5). |
| 3 | Structured thinking | 4 | Explicit rules, worked traces, a record per decision. Missing: unit of replication, ties, units (W4, W9). |
| 4 | Inspired, not copied | 4 | Main elements name S# and the difference. kept_count and doubling name none; the bet's nearest prior art is unnamed (W10). |
| 5 | Falsifiable tonight | 3 | T0, T1, T3 test parts. No test, tonight or later, can show the price wrong where the record is decided (W3). |
| 6 | Buildable, not tied to 1d | 3 | The loop exists and passed its gate alone; I reproduced its exact score (L2). The build list omits work the text assumes (L4, L5); the three packs are one method family. |

**Weaknesses, ranked, each with a fix**

**W1 (blocking). No arbiter exists for the problems the loop runs on.** v6 says "Record: WS1's arbiter only". The arbiter knows three problems; none is 1d or 1a. The loop certifies and claims by itself, and its "two exact checks that must agree" on 1d are one function called twice (L1). That is S14's case. *Fix:* before T5, a WS1 pack for tao-1d: pinned verifier `wt-compare/problems/p1d.py:47` (in WS2's `frozen.lock`), independent check `indep_check_1d.py` (no code shared with `c1d_exact.py`; L2), frozen with Arthur's approval. Until then the loop's line reads "below the snapshot, own check", never "record".

**W2 (blocking). The reference reader fails open.** If `gh` fails, the page's table alone still returns `ok` (L3): E7a again, silently. Claims are read from pull-request titles only; one titled differently is skipped. *Fix:* every named source answers, or no claim. Select pull requests by the file they change; an open one with no parsed number blocks the claim. T0 adds both cases. Add the missing rule for `adopt`: funded first whenever the snapshot's best changes (E6).

**W3 (major). The price is silent where the record is decided.** Rows match at equal run budget and a move needs 5. T5's deep share buys about 16 runs at 3,300 s (45 at 1,200 s, where D6 saw soft restarts cut), and each doubling opens an empty cell. Few or no deep moves reach 5 rows. Deep decisions are made by cold-start order, the edge rule and kept_count, not by the bet. T2 cannot reach that stage (Croqui's note 2); where it reaches, C2 found the bet level with a fixed schedule. This is E3 in a new place. *Fix:* say it, and give the deep stage a second arm (element A).

**W4 (major). The verdict rules cannot be run as written.**
- (a) The refine stage draws no random number (L4). Two runs of one fork differ only by the wall-clock cut. "The same again" and 20 forks on 3 lineages are 3 samples (E7c). *Fix:* the unit is the root seed; pairs sharing a root count once; a twin is a jittered setting (D3's 4.04), not a rerun.
- (b) Ties and invalid outputs are undefined. *Fix:* both count for the incumbent.
- (c) Two systems: 10.0% false replacement over two problems (L7). The rule counts panel wins; the track scores one best certificate. E2: 85 wins of 100, same best. E12: on seed 162 the old-loop lineage went under the claim (1.4544063); the map-loop lineage did not (1.4547298). *Fix:* also require the better best certificate over all panels. Stop a fork series once its verdict is fixed (15 wins, 15 losses, or 6 of each).

**W5 (major). `completed` does not exist yet, and doubling is slow.** All ten 1,200-s deep restarts of the autonomous arm are stored `completed: true`. p0 = 2 came back with its parent's score under a new sha256; p0 = 4 gave the same file and left no row (L4). Doubling 1,200, 2,400, 4,800 s is 140 minutes of wall clock per probe, on T5's critical path. *Fix:* element B.

**W6 (major). kept_count learns from censored ranks.** A cut lineage never shows a final rank, so the rule only confirms the width already chosen (E12: five lineages deep, winner 4th of 5, C4). E12 says rank stops predicting at the last boundary. The coordinator's guidelines of 23:07 (`shared/autoloop_guidelines_1d.md`, items 2–3) say one strong lineage in three responded and ask for 20 to 30. *Fix:* per boundary, measure rank correlation from rows (S9's precondition). Where it holds (M2, C3), keep by rank. Where it fails or has under 10 pairs, keep by coverage: k = ceil(ln 0.1 / ln(1 − r)) for responder rate r (1/3 gives 6; 0.2 gives 11), chosen far apart under the pack's distance (S8).

**W7 (major for scope points 1–2). A pack is more than "a problem, its verifier and a reference", and the code is 1d-shaped (L5).** Refine is illegal on second-autocorrelation: the #1 has 1,999,999 values, the venue takes 2,000,000 (L8). *Fix:* a default pack needing only verifier, direction, δ and reference: seed = `adopt`, polish = a generic perturbation search (WS1's stage-3 probe ran one on 14 problems), venue limits checked at the gate. The three frozen problems already have verifier, δ, live target and arbiter. One run on circles-rectangle is the integration run T5a could not be: maximise, no `cells`, a real arbiter (adapter 1–2 agent-hours, conjecture). L8 also supports `displacement` on a problem no rule was fitted on.

**W8 (minor; blocking for model-written stages).** Stages inherit the loop's environment, keys included, and stderr reaches the ledger (L6). *Fix:* an allow-listed environment and a scratch working directory per stage; T0 adds a stage that prints its environment and fails. I support Croqui's note 4.

**W9 (minor). Units.** Bets are raw score per CPU-second; the gate has no margin (CLAUDE.md rule 2). *Fix:* δ in the pack; bets, reach and log in δ per CPU-second. One price can then order problems: a fourth level, same round (S2).

**W10 (minor). Novelty wording.** The bet is close to "expected improvement per second" and freeze-thaw Bayesian optimisation (my memory, unread; Sonar asked for S18, question 1 of 3, no answer yet). Name it and say what differs.

**Elements of my own**

**A. Authority is earned: shadow policies and disagreement forks.** From S15 (shared control, rules before data) and S14. Different: the arms are decision rules compared at one decision from one shared state, not systems from empty ledgers.
- Every policy (present rules, price, model) answers every decision; one controls. Pick and predicted gain are logged before the result. 0 CPU; about 0.005 USD per model answer (`autoloop/RESULTS.md`).
- Where a shadow differs, the fork share may run its pick too, from the same state. One deep fork costs one run, not a panel.
- A policy takes control of one decision type (width, setting, lineage, stop) after winning forks of that type under W4's rule. A model may veto a claim, never grant one.
- T5 gets its second arm. Scope point 3 gets a test per decision type inside runs we already pay for. A judge sees three voices per decision, scored by what happened.
- Build the price as a logger first (half of its 3 hours; conjecture): the running loop is never at risk.
- **Tonight, 0 USD, 30 minutes once the logger exists:** replay the saved ledgers through both rule sets; count the decisions that differ. Under 10%: the price changes nothing and gets no control. Otherwise that list is T2.

**B. Declared work, not wall clock.** From S5 (a hand-off is a finished object); unlike S9, which cuts by rung. A stage declares its work: `c1d_refine.py` has a ladder from p0 to pmax with `--iters` per step, 32 steps for p0 = 4, 30 for 16 (L4). A run starts only if its whole ladder is affordable; a shorter run is a shorter declared ladder. `completed` is read from the stage's own `stages` list; cost is predicted from seconds per evaluation in the class. No censored rows, no doubling, and "reproducible experiments" in the track's words. **Tonight, 0 USD, 8 minutes:** run one saved first refine twice without a budget. Wrong if the two sha256 differ.

**C. The down pass.** `coarsen: Cert[4N] → Cert[N]`, then polish: the half of S11 that E5, E6 and E12 never used, with S7 (the expensive level teaches the cheap one). The cycle becomes two-way. **Tonight, 0 USD, about 5 minutes (conjecture):** coarsen the E12 winner to 12,800 cells and polish. Kept if it ends below its lineage's former value there (1.4560723); dropped if not.

**Scope note.** (1) Autonomy: named decisions covered. Not named: the deep run length (W5), who certifies (W1), who writes the pack (W7), when to adopt (W2). T5 must list its frozen prompts: a run that reads the 23:07 guidelines is exploitation (D6's last bullet). (2) Generality: stated, not yet true in code. (3) Model: a default, no test; A gives one. (4) Two systems: fixed before the runs; see W4c. (5) Log: good; add the shadow picks.

**Outside reviewer's objections to v6.** 1: agreed on the risk; my fix is W6, since the estimator is cold there (W3). 2: agreed in part; cap tested edits at 3 per run (family error 6.1%). 3: agreed; take Croqui's note 5. 4: agreed (W8). **Croqui's notes:** all five accepted. Order: W1, W2; note 1's three pieces with B in place of doubling; the price as a logger; control last.

**What gets CONSENSUS from me.** W1 and W2 in the text and on the build list with an owner. W3 stated. W4a–b fixed. A second arm at the deep stage (A or another). T5's prompts listed. For Murilo: Arthur's approval of a 1d pack; whether to spend on C or the circles-rectangle run.

**Method of L1–L8** (read-only; code under `<workspace>/wt-attack/`)
- L1: `problems/catalog.py:29`. `autoloop/loop.py:48-71` (own claim gate). `attack/c1d_round.py:14` and `autoloop/tao1d.py:133` both call `c1d_exact.check`. `attack/c1d_exact.py:18` still holds the reference as a constant; the loop ignores it.
- L2: `indep_check_1d.py` on both files in `<workspace>/shared/1d_candidate_2026-11-14_2117/`: exact 1.454406313885605, 2.67e-4 under pull request 345's exact fraction; exact 1.4541470296475214, 5.26e-4 under. 67 s each. Reproduced, not VERIFIED: the arbiter cannot run on 1d.
- L3: `autoloop/tao1d.py:238` (title filter), `:241-248` (the `gh` error is appended; the page's values still return `ok`).
- L4: `attack/c1d_refine.py:50` (wall-clock stop; no random number in the file), `:23`, `:76-78` (ladder), `:30` (rescale before saving). `autoloop/ledger.py:50`, `moves.py:48-51` (`completed` never set), `loop.py:139-141` (a duplicate id is dropped). Rows: `autonomous_rules_arm/ledger.jsonl`, results 1 and 3: 6 runs, 5 distinct ids.
- L5: `autoloop/problem.py:3`, `:14`; `moves.py:68`, `:46`; `policy.py:66-70`; `loop.py:33`, `:107`. `LOWER_IS_BETTER` is declared, never read.
- L6: `autoloop/policy.py:142`; `tao1d.py:74` (no `env=`), `:128`; `loop.py:161`. A scan of the run folders for key patterns (file names only) found none.
- L7: binomial tails. 15 of 20: 2.1% under the null; power 42% at a true win rate of 0.7, 80% at 0.8. 8 of 10: 5.5%; "wins one, loses none" over two problems: 10.0%.
- L8: Research Map note, lines 26, 31 (venue limit), 27, 36, 57–59: on second-autocorrelation the holder's record steps move L1 0.17–0.23; our kicks that fell back to the same vertex moved 0.0001–0.0014 (another normalisation than D2's).


## Evaluation of v6 by the OpenAI evaluator (gpt-6.1-sol)

## Phase 2 outside evaluation of v6

**Verdict: changes required.** The proposal defines a plausible relay. It does not yet establish a better autonomous research system. Lupa’s implementation findings are reported evidence here, not checks I independently reproduced.

### Protocol scores

| Criterion | Score / 5 | Reason |
|---|---:|---|
| Explains the evidence | 4 | Covers E1–E12. But pooled estimates can hide E12’s seed–setting interaction. E3’s sparse-learning failure returns at depth. |
| Recursive and chained | 3 | Typed chaining is concrete (E5; S5). Recipe recursion is specified, but not demonstrated. Repeating resolutions alone does not establish recursive decision-making (S3, S11). |
| Structured thinking | 3 | Rules and records exist. Retention, sparse estimates and competing budget obligations remain incomplete (E7d; S9). |
| Inspired, not copied | 4 | Sources and differences are mostly clear. The price resembles expected improvement per cost; novelty rests on its integration, not that estimator alone (S17). |
| Falsifiable tonight | 3 | Component tests can fail. T2 cannot test deep allocation. T5 lacks a control there (E12; S15). |
| Buildable by Sunday, general | 2 | The existing loop helps. Missing trust integration, acceptance work and demonstrated non-1d execution weaken the deadline claim (E8, E10; S2). |

### Scope scores

| Point | Score / 5 | Reason |
|---|---:|---|
| Autonomy | 3 | E12’s choices have proposed replacements. Pack creation and method invention remain supplied inputs. Exploitation is not discovery. |
| Generality | 2 | Interfaces are described. Available evidence remains concentrated in related problems (E8–E9; S2). |
| Model decisions | 2 | Stage generation is a reasonable role. T3 tests implementation from a supplied method, not whether model choice beats rules (S1, S3). |
| Whole-system comparison | 3 | Equal budgets and frozen policies help. Replication, ties, repeated selection and deep-stage coverage need repair (E7c; S15). |
| Decision log | 4 | Alternatives, costs and ancestry are useful. Add reservations, rejected actions and trusted-gate outcomes (E7; S13). |

### Weaknesses and concrete fixes

1. **Claim authority is asserted, not established.** Lupa reports missing arbiter support and a fail-open reader. These recreate E7a–b, despite S14’s warning.  
   **Fix:** name an owner for independent verifier integration. Require every named source to answer. Unparsed relevant claims block record status. Until integration passes, label results “own-check improvement against snapshot.”

2. **The allocator loses interactions and lacks a sparse-data rule.** Average lift times average survival is not generally expected final improvement (E12; S17).  
   **Fix:** specify parent features, matching, minimum support and fallback. Test synthetic seed–setting interactions and rank reversals. Below support, use a bounded lineage-diverse schedule, not a numerical claim of learned superiority.

3. **Budget obligations can conflict.** Cold starts, mandatory reruns and recursive edits compete without explicit cancellation rules (E7d; S9).  
   **Fix:** one trusted global reservation ledger. Admit a child only after reserving its maximum CPU and USD cost. Define retention fallback, tie order and cancellation priority. Test tiny budgets with failures and new stages.

4. **Model-written policies bypass the stage boundary.** Sandboxing stages does not protect the controller (E7; S13–S14).  
   **Fix:** policies return validated action data only. They cannot execute controller edits. Strip credentials and restrict writes. Test ledger mutation, forged ancestry, verifier replacement and budget bypass.

5. **Validation misses the decisive regime.** T2 cannot buy deep probes; repeated deterministic runs are not independent seeds (E7c, E12; S15).  
   **Fix:** add paired deep disagreement forks. Count root lineages as replication units. Predeclare one selection batch, ties and invalid outputs. Freeze prompts. Call T3 a component test.

### Additional design element: capability manifest

Each pack declares legal transformations, dimensions, direction, venue limits, verifier identity and reference sources. The controller rejects incompatible stage plans before spending compute. This extends S5’s typed handoffs to whole-chain feasibility. It directly addresses E7b and E8.

**Required changes:** implement the claim boundary and global budget authority; restrict policy outputs; specify estimator fallbacks; add controlled deep-stage validation; and publish a timed acceptance plan with named owners.


## Joint answer, phase 2 exchange 1 (written by Meridian; Croqui co-signs or dissents below)

The proposal is now version 7 (1,200 words). Nothing of Lupa's or the outside evaluator's is rejected. Four points are adapted and say why. I checked L1, L3, L4 (the `completed` default), L5 and L6 in the code, and L7's arithmetic. Murilo's priority note of 00:55 is answered at the end.

**Lupa's weaknesses**

- **W1. Accepted.** Text: a record needs an arbiter pack; without one the line reads "below the snapshot, own check". Build item 1: a WS1 pack for tao-1d with the pinned verifier and `indep_check_1d.py`. Owner proposed: Lupa. Arthur's approval is Murilo's to ask.
- **W2. Accepted, one part adapted.** Every named source must answer; an unparsed entry blocks the claim; T0 has both cases (R11). Adapted: a new best in the snapshot is re-scored under the local verifier first (metric identity), but it becomes a lineage only if the pack allows adoption. On 1d the team chose not to refine the leaders' certificate (`<workspace>/wt-attack/attack/modal_c1d.py`, header).
- **W3. Accepted and stated.** "Where the bet speaks" says that the bet orders only moves with 5 matching rows and is level with a fixed schedule there; elsewhere cold-start order, the edge rule and coverage decide. The second arm is element A.
- **W4. Accepted, all three.** (a) The unit is the root seed; forks that share a root count once; a twin is a jittered setting. (b) Ties and invalid outputs go to the incumbent. (c) A system also needs the better best certificate over all panels; a fork series stops once its verdict is fixed (R13).
- **W5. Accepted, through B.** Run-budget doubling (my rule in v4 to v6) is withdrawn. It patched a wall-clock stop that B removes.
- **W6. Accepted, adapted.** By rank where at least 10 pairs show that rank holds; else by coverage. Adapted: r needs a definition a rule can compute (R10): the share of roots that the next stage's best move lifted by over half of its best lift; 0.2 until three roots were tried. Conjecture.
- **W7. Accepted.** The default pack is built from the verifier, the solution format and the reference alone (R12). The outside evaluator's capability manifest is the same idea one step further: the pack declares its legal stages and venue limits, and an incompatible plan is refused before any spending. Adapted: we did not check the generic perturbation search that Lupa points to in WS1's stage-3 probe.
- **W8. Accepted.** Allow-listed environment, scratch working directory, a T0 case.
- **W9. Accepted.** δ in the pack; bets, reach and log in δ per CPU-second; a claim needs a gain of at least δ. One price over several problems stays design only and is no longer counted as a level.
- **W10. Accepted.** The bet is named as expected improvement per second. S18 has not landed, so the citation is S17. Different here: saved rows instead of a model, and arrivals are priced too.

**Lupa's elements**

- **A. Accepted in full.** It replaces T2 as the test of the allocator, gives every depth its second arm, and gives scope point 3 a test per decision type. The bet is built as a logger first. A model may veto a claim, never grant one.
- **B. Accepted in full.** It also withdraws my partial rejection of the outside reviewer's replay point (answers to v5, point 4): with declared work a rerun must rebuild the same certificate.
- **C. Accepted on trial.** The down pass enters a pack's stage list only if its test passes (R15). Spending on it is on Murilo's list.

**Croqui's notes on v6:** all five accepted. 1: the build order starts with what pays without T2, with B in place of doubling. 2: T2's limits are stated; T2 runs only if TA finds decisions that differ. 3: rows match with the item's own lineage first, then the 10 nearest in start score (R2). 4: a rule edit is a text that a person applies after it wins a held-out replay. 5: cut order in R9; the kept fallback in R10.

**The outside reviewer's objections to v6.** 1 (interaction): accepted; R2 reads the lineage's own rows first, trace 3 is the synthetic case, and under 5 rows no number is claimed. 2 (discovery, repeated testing): accepted; tonight's tests are called component checks, at most 3 edits are tested per run (Lupa's L7: 6% family error), and prompts are frozen and listed. 3 (budget): accepted; R9, trace 4. 4 (policy boundary): accepted; every policy returns only a legal move, and rule edits are text.

**The outside evaluator.** 1 (claim authority): as W1 and W2. 2 (allocator): as above; below 5 rows a bounded, lineage-diverse schedule decides (coverage). 3 (budget): accepted; a move starts only after its declared maximum CPU and USD are reserved under the global caps (B and R9). 4 (policy boundary): accepted. 5 (deep validation): accepted; disagreement forks, root seeds as units. Capability manifest: accepted (W7). Owners and order: the last section of the proposal; they are proposals for the coordinator.

**Not done.** A problem outside the development data with meaningful choices was not run. T5 names one (circles-rectangle). Nothing here was tested tonight by us: every number is a replay, a file read or Lupa's rerun.

**Murilo's priority note (00:55).**

- *Unseen problem.* Each element was checked against "a verifier, a format and a reference only". General as they are: ledger, certificates, gates, declared work, bets, probes, coverage, shadow policies, forks, guidelines, decision log. Need a pack addition: resolution ladders, venue limits, an arbiter. Removed from the core text: everything that names cells or p0.
- *Guidelines the system writes.* R14. Who writes: the distiller that already runs (`<workspace>/wt-attack/autoloop/distill.py`). How they are checked: not by their writer. Each guideline is a shadow policy, and a rule recomputes its fork record from the ledger. How a wrong one is removed: by losing its forks (5 or fewer of 20), with its scope, and it is never advised again. TG tests exactly this with a planted false guideline.
- *Short and buildable.* The proposal ends with what exists tonight and what is added, in order, with proposed owners.

**R. The rules, complete (what a builder codes)**

- R1. Row: one attempt of a move, with problem, class, move kind, setting, position, declared work, lineage and root, start score, end score, CPU-seconds, `completed`, displacement. Lift is start minus end in the pack's direction, in δ; 0 when the run fails or returns its parent. A seed row has no start: its outcome is its first score.
- R2. Matching rows for a move on item X: same move kind, setting, position, class and declared work. X's own lineage first, then other lineages by distance in start score. The first 10 are used. Under 5, the move is not a bet: it is a cold-start move if the setting is the default, else a probe.
- R3. best: the best score of the round's pool in X's class. gap: X's score minus best, in the pack's direction.
- R4. Gain of a row: max(0, lift − gap). For a seed: max(0, best − outcome).
- R5. bet: mean gain over the matching rows × survival of the move kind ÷ mean CPU-seconds of those rows. Survival: lift left at the end of the chain ÷ lift, pooled over forks; 1 with fewer than 4. The product is an approximation; forks judge the picks, not the estimate.
- R6. Candidates of a round: one new seed; for each kept lineage, one more run of its present stage for every setting that has 5 rows, and the default run of its next stage.
- R7. Price: candidates are funded in falling bet order (ties: older lineage, then the setting nearer the default), each bet recomputed after a funded move, until the round's budget minus the probe share and the cold-start runs is reserved. The last funded bet is the price.
- R8. Probe share: 10% of the round; 50% at reach zero, when every move with 5 rows has no row whose lift exceeds the gap to the live reference. Order: (1) on each kept root, one grid step past the edge of a setting's tried range where its best rows sit; one tried value has two edges; a stage with no rows reads the nearest earlier position; with no edge, the side where displacement grew; (2) a twin: the same setting jittered by 1%; (3) a fork: a shadow's different pick from the same state; (4) an edit. What the share cannot spend returns to the bets.
- R9. Reservations: a move starts only after its declared maximum CPU and USD are reserved under the global caps, at every level. When the budget is short, cut in this order: reruns of a winning probe; the cold start of a new stage; other probes; bets, from the lowest. The cold start of default moves is cut last.
- R10. Kept at a boundary. If at least 10 lineages crossed it and their ranks before and after correlate (Spearman 0.8 or more): 1 + the deepest earlier rank of the three best at the deepest class reached. Otherwise coverage: k = ceil(ln 0.1 / ln(1 − r)) roots: the best one, then roots far apart under the pack's distance among the better half. r: the share of tried roots that the next stage's best move lifted by over half of its best lift; 0.2 until three roots were tried (k = 11). Ranks compare inside one class.
- R11. T0 must refuse: a stage that fakes its score; one that writes outside its scratch directory; one that prints its environment; a policy answer outside the legal menu; a certificate with a forged parent; a move that would pass a cap; a reference with one named source silent; one with an unparsed entry; one with another spec version.
- R12. Default pack, from the three things a person gives. Seeds: drawn inside the format's declared ranges, or the public solution when one is given (adopt). Polish: a perturbation search with declared iterations. Distance: L1 on the flattened solution. Class: the solution's shape. Direction and δ: from the reference. Certify: the verifier, exact when it has an exact mode. Reference: (value, source, read time); stale means no claim.
- R13. A refutation names move, position, class and setting range; outside it the move is untried. A fork series stops at 15 wins, 15 losses, or 6 of each. At most 3 edits are tested per run.
- R14. Guidelines. Record: id, decision type, scope (general or one problem), a pick rule the policy interface can execute, predicted effect in δ, supporting rows, status, history. Written by the distiller after each run or tournament; a change citing unknown rows is rejected (both exist). Added: every active guideline answers the decisions of its type as a shadow. A rule recomputes its record from the ledger: forks where its pick differed from the controller's, paired by root. Status: hint (it only orders probes), trusted (15 of 20 wins with a positive mean gain: it may control its decision type inside its scope), dropped (5 or fewer: kept in the file with its scope, never advised again). General scope needs trusted on two problems. Hand-written lessons enter as hints. A hint with no fork after three runs is retired (conjecture). The file and its statuses are mirrored to shared memory; a run loads the general guidelines and its own problem's, records their hash, and is labelled exploitation when it loads its own problem's.
- R15. Component checks on 1d. TB (0 USD, 8 minutes, Lupa's): one saved first refine run twice without a budget; wrong if the two sha256 differ. TC (0 USD, about 5 minutes, Lupa's): coarsen E12's winner to 12,800 cells and polish; the down pass is kept if it ends below 1.4560723. T1 (1.3 CPU-hours): p0=4 ends below p0=4096 on at least 15 of 20 first refines (D1); wrong if 10 or fewer. T3 (under 2 USD): a model writes the refine stage from the published method; pass: it lowers 4 of 5 saved 800-cell certificates by 1e-3 (E5). T6: 1d from an empty ledger, fresh seeds, 30 CPU-hours, prompts listed; a prompt holding E12 or the guidelines of 23:07 makes it exploitation. Predicted (conjecture; E12: 3 of 8): a soft restart lowers a kept root by over 1e-3. Passing the claim is not expected (D5).
- R16. E12's hand-made decisions and who makes each here. The refine stage, written from the leaders' published method: a model's edit, admitted by the contract (T3). Which starts climb: kept (R10). How long a run lasts: declared work. p0=4: the edge rule, from rows older than E12 (D1). Two runs per setting and more restarts from the winners: twins, then the winning-probe rule. The reference: the pack's reader, failing closed. Who certifies: the arbiter pack. A person still gives the pack, the budgets, and whether adoption is allowed.
- R17. The worked traces that T0 must reproduce. Trace 1: Croqui's W1 (answers to v4). Trace 2: my W2 (answers to v5), with two changes: with no rows, coverage keeps all 5 seeds, and "the same again" is a twin. Traces 3 and 4 follow.

**Trace 3. A lineage-by-setting interaction and a rank reversal.** Deep stage, two kept roots: A leads, B is 2e-4 behind. Default restarts have 5 rows with lifts under 6e-5.

- Edge probe p0=4, once on each root. A returns its parent (lift 0). B lifts 1.6e-3 and becomes the pool's best.
- Winning probe: it is tried on every kept root. A: 0 again. B's child: 1e-4.
- After one more run on each line the setting has 6 rows and becomes a bet. With R2 the bet on B's line reads B's rows first and the bet on A's line reads A's rows first: B's line is funded, A's is not. Pooled rows would give both lines the same bet; that is the error the outside reviewer describes.
- B was second at the boundary and holds the best certificate now. Rank did not predict it; coverage (R10) is what kept B.

**Trace 4. A tiny budget, one failure, one new stage.** Seed and polish stages; an edit has admitted a new stage; empty ledger; a round of 8 runs.

- Runs 1–5: default seeds. Run 3 crashes: a row at full cost with no certificate. It counts among the 5.
- Kept, no rows: coverage asks for 11; all 4 certificates are kept.
- Runs 6–8: default polish on the three best. Polish has 3 rows: no bet, and reach is not defined.
- Probe share: 10% of 8 is under one run. Every probe is unaffordable and skipped. The new stage's cold start is cut (R9).
- Termination at 8 runs. The log shows 8 attempts, 7 certificates, no probe, and names what was cut.

**Erratum to version 7 (Meridian, 03:55 PT).** S18 landed in the survey after the joint answer was written (expected improvement per second; freeze-thaw). The bet now cites S18 instead of S17 in two places of the proposal; nothing else changed. This closes W10: the prior art is named, and the differences are the ones S18's entry lists (saved rows instead of a fitted model, a survival term, unlike stage types, gain counted on the pool's best).


## Co-signature of Croqui (phase 2, exchange 1)

**Co-signed: version 7 and the joint answer. I dissent in part on R2 and R14 and add three notes. Lupa, please rule on points 1 to 5 in your reply.**

For the first time I read the loop's code myself, with the coordinator's leave (03:41): `<workspace>/wt-attack/autoloop/`, branch `lab/attack`, working tree at commit 38e4e76 (03:45). I did not open the run ledgers.

**What the code already does (commit 36d198b, 00:44, after Lupa's read).**

- W2: `refread.py` fails closed. If the page or the pull-request list cannot be read, there is no reference. Bold numbers in pull-request bodies are read too. Still open: pull requests are chosen by title, and one with no parsed number does not block. Build item 2 shrinks to these two and the T0 cases.
- W8: local stage subprocesses get an environment without secret-like names (`safety.py`, a deny-list), and errors are redacted. Still open: an allow-list, a scratch directory, no network.
- L1: certify on 1d also runs `check_independent.py` (`tao1d.py:190`). W1 stands: there is no arbiter pack, and that needs a team decision.
- Unchanged: `completed` defaults to true and nothing sets it false (`ledger.py:50`); no file reads `LOWER_IS_BETTER`; the pack interface requires cells and p0 names (`problem.py`).

**Dissent in part.**

1. **R2 does not produce trace 3.** With 6 rows, "the first 10 are used": both lines read the same rows and differ only by their gap. Fix: when the item's own lineage has 2 or more matching rows, only those are used. Then A's bet is 0 and B's line is funded, as the trace says.
2. **R14: only a rule may set a status.** `distill.py` accepts add, update and drop when one cited row exists (lines 116–118, 137). It does not read the rows, and the model can drop any lesson. All 21 lessons are active, and they reach only the model's prompt (`policy.py:300`). Add:
   - (a) at birth, the rule recomputes the predicted effect from the cited rows and rejects a guideline that they do not show;
   - (b) the model's "drop" queues a fork and changes nothing;
   - (c) trusted is not permanent: the record is the last 20 forks, and a guideline returns to hint when the code hash or spec version behind its rows changes. Example: G10 and P1d11 cite rows of test C; the loss they describe was fixed in commit c320df6 (`loop.py:177`);
   - (d) scope is computed, not asserted: a guideline applies to a pack whose manifest declares every field its pick rule reads; "general" means it reads the default menu only. The nine hand-written general lessons are for step-function autoconvolution by their own title, and G03, G05 and G06 name p0 or cells. On circles-rectangle they would load as general.

**Notes.**

3. **Forks can starve.** R8 pays edge steps and twins first. In T6 at reach zero the share is at most about 16 deep runs (15 CPU-hours at 3,300 CPU-seconds each, D3), and coverage asks for 11 edge steps and 11 twins before the first fork (R8, R10; my arithmetic). Add: a probe or twin that equals a shadow's logged pick counts as that shadow's fork, at no extra cost.
4. **Build order under the priority note.** Item 2 and part of item 4 are already in the code. Even so, the default pack and T5 come last in one builder's queue, and about 6 hours are left before 10:00: the note's first bullet is what the clock cuts. I propose a second builder on item 6 now (the coordinator's call). For Murilo's list I recommend T5's adapter before C.
5. **Two sentences to add.** A veto names the gate check it doubts; the gate reruns it once, and the claim stands if it passes. At the start the loop's present rules control every decision type.

**Meridian on Croqui's co-signature (04:00 PT): agreed on all five.**

- 1. R2 as I wrote it does not produce trace 3; the error is mine. Croqui's fix stands, with one clarification so that the 5-row threshold keeps its meaning: a move is a bet when it has 5 matching rows in all; its bet then reads the item's own lineage alone when that lineage has 2 or more of them, else the 10 nearest in start score.
- 2. R14 (a) to (d): agreed. (d) is the better answer to "general": a guideline applies where the pack's manifest declares every field its pick rule reads. "Trusted on two problems" stays as the second condition for control outside the problem where it was earned.
- 3. Agreed: a probe or twin that equals a shadow's logged pick is that shadow's fork, at no extra cost.
- 4. Agreed: the default pack and T5 are what the priority note asks for first, so they should not be last in one queue. A second builder is the coordinator's call. T5's adapter before C.
- 5. Agreed, both sentences.


## Reply by Lupa (phase 2, exchange 2 of 4; 15 Nov, 04:10 PT)

**Not CONSENSUS yet. Five edits, each one or two sentences; exact wording at the end.** My conditions of 00:40 are met except one owner. Murilo's note of 00:55 came after them, so I checked v7's answers to it against tonight's code and rows. N1–N6 are my own reads and reruns (0 USD, nothing edited; method at the end). My evaluation above was appended at 00:32, not 00:40.

**Ruling on Croqui's points 1–5** (Meridian agreed to all at 04:00)

1. R2: agreed, with Meridian's clarification. One guard: the two own rows must be distinct runs. With declared work the same certificate, setting and work give the same output, so a repeat is a cache hit, not a row. Trace 3's "A: 0 again" is such a repeat.
2. R14 (a)–(d): agreed. (d) is better than my wording. Two additions in E2.
3. Agreed. My count agrees: a deep run costs 2,400 CPU-seconds (cold3), so a 10% share of 30 CPU-hours buys 4.
4. Agreed: a second builder on the default pack now; T5's adapter before C.
5. Agreed. The veto text stays on the claim record that a person reads.

**Verified since my evaluation (N1, N6).** I confirm Croqui's list. Additions:

- The reader, both fetchers injected: a failed `gh`, a failed page and a non-list answer return `ok=False`. Five cases still return `ok=True` from the page's table alone: an open pull request whose title does not start with the id; one whose claim has four decimals (the pattern needs "1.4" and four more digits); one whose claim is outside the title and outside bold; one with no number; a list cut at 300.
- `certify` runs the file I ran in L2 (same sha256). "One function called twice" is withdrawn. The path is not pinned by hash (`tao1d.py:74`). The claim text still says "TARGET BEATEN".
- Locally a stage can still read `.env` from disk. Until the sandbox exists, a model-written stage runs only in the Modal container, which mounts no secret (`modal_app.py:35-43`, `:67`).
- The live reference moved: 1.4546733294, pull request 371, read at 02:04 (cold1 ledger). Our two files stay 2.67e-4 and 5.26e-4 below.
- New at 38e4e76 (03:45): `run_start` records `cold_start: true` even when the run loads its own problem's lessons (`loop.py:119`). R14 calls that exploitation.
- 13 unit tests pass at 38e4e76.

**What still blocks**

**E1. Item 1 has an owner who cannot do it.** I never edit code, and `problems/` is WS1's frozen catalog (my role file; CLAUDE.md). A referee who builds the arbiter cannot verify it (S14). Owner: WS1, named by Arthur. From me: the acceptance checks and the arbiter run.

**E2. Guidelines: two additions to R14.** The note asks how a guideline is checked against the ledger and removed.

- *What the present check passed (N2, N3).* (i) P1d05 says "three independent runs confirm". The three arms refined one certificate (d408790d) with the same settings, in a stage with no random number (L4): one root, counted three times (E7c). (ii) P1d09 now stores scores and "The live reference is still 1.4546733 (teorth PR #345)". (iii) cold2 loaded the hand-written guidelines, never read the reference (no `reference` row) and stopped citing "the guideline reference of about 1.45467": E7a, through a lesson. No cold-run row has reached the distiller: `lessons.json` was last written at 00:48.
- *Three lessons against rows they had not seen (N4: code, one second; cold1–3 on 1d, cold3 unfinished; one run on first-autocorrelation).*

| Lesson | The rows |
|---|---|
| "Rank before the last stage predicts only roughly" | Spearman at the last boundary: 0.74, 0.44, 0.77 (n = 30, 30, 35, kept lineages only; 0.96 on cold1's 7). One boundary earlier: 0.97 to 0.99 in all four runs. True at one boundary. |
| "Soft restarts move lineages; of three strong ones, one did" | A lift over 1e-3: entering the top level with p0 = 4, 21 of 67 roots on 1d and 9 of 34 on first-autocorrelation; with p0 = 8, 1 of 60; a soft restart after entry, 3 of 60. "One in three" is the entry with p0 = 4. The lesson merges two moves and two settings. |
| "Hard restarts (p0 ≥ 16) stall" | cold1, restarts at the top level: p0 = 16 gained on 5 of 5; p0 = 4 returned its parent on 7 of 8. cold2 and cold3 hold no such row: the policy stopped trying them. An obeyed lesson removes the rows that could refute it. |

- *Forks alone are too slow at depth.* A verdict needs 20 roots; R10 carries 11 to 13; the share buys 4 deep runs. No deep guideline changes status by Sunday. The table cost no CPU.
- *Add (e) and (f) to Croqui's (a)–(d).* Wording below. Recommended, not blocking: a trusted guideline that removes an option keeps one control run of it per run. TG2 tonight (0.08 USD): the present distiller on the four cold ledgers, into a scratch file; its changes are compared with the table.

**E3. T5 cannot fail for a search reason.** circles-rectangle's format is 21 triples with no ranges, and overlap is refused (`problems/circles-rectangle/meta.json`). R12's random seeds are invalid or trivially small, so the start is `adopt`, not a cold start. WS1's digest reports the #1 as a KKT point (a forum claim); WS1's generic probe gained 4.3e-3 δ from it in 600 s and 2.5 million evaluations (`selection/registry_stage3.jsonl`). Prediction: T5 passes its bar and gains nothing. Name it an integration test, say so, and add the comparison that can fail.

**E4. R10 starves its own deep stage.** By its letter a cold run keeps 11 of 200 seeds, then picks inside "the better half" of those: 11, 5, 2. Where rank holds early (0.97 to 0.99), the rank branch keeps 4 to 6 before any deep row exists (N5). Either way 2 to 6 roots arrive. Two roots at r = 0.17 give a 31% chance of one responder, not 90%. The rest of R10 fits rows it was not fitted on: r = 0.17, 0.17, 0.15 (k = 13, 13, 15).

**E5. T0 gets the reader's five cases.**

**The OpenAI evaluator.** Its "changes required" is on v6. I agree with its five points and its manifest; v7 accepts all of them; nothing is overruled. Its "timed acceptance plan": v7 has owners and an order, no clock times; add them. One overrule to record if it repeats it: "implement" as a condition of consensus. This debate is thinking only (protocol); the build list is the answer. Croqui: run `evaluate` on the new version.

**Exact wording** (adapt freely)

- Build list, item 1: "arbiter pack for 1d (WS1, named by Arthur; Lupa writes the acceptance checks and runs the arbiter)".
- R14: "(e) After every run a rule recomputes each guideline's effect on all rows of its scope that its writer did not see, counted in roots, at no CPU cost. A descriptive guideline (a rate, a correlation, a cost) names a predicate from a code library, and its status follows that number. A pick guideline cites the descriptive ones under it and returns to hint when one fails. (f) A lint refuses a guideline that holds a score, a reference value or a file id."
- T5: "T5a, integration: default pack, general guidelines only; pass: verified certificates and a complete log, no person in the loop, inside the caps. Predicted: it starts from the public solution and gains less than δ. T5b, search: same problem, CPU and seeds: the loop against WS1's flat probe, and against the loop plus one stage a model wrote from the verifier and the format alone. Reported: gain/δ of the three. Not expected: `beats_record`."
- R12: "Reach: formats where a random draw is valid, or problems with a public solution."
- R10: "Widths are planned backwards (S9): a boundary keeps at least twice the k of every later boundary that has no rows yet."
- R11: the five reader cases above; a truncated list is a failed read; an entry is any open pull request that changes the constant's file.

**What gets CONSENSUS.** These edits, Croqui's five points in the text, and the outside evaluator's answer on the new version: `ACCEPT`, or our written overrules. I add no new point in exchange 3 unless an edit creates one.

**For Murilo.** Arthur's approval and a WS1 owner for the 1d pack. A second builder for the default pack. T5's adapter before C. T5b's model-written stage (under 2 USD).

**Method N1–N6** (read-only; scripts in `<workspace>/wt-referee/.maestri/roles/703526ec-c2d5-4763-8955-60f9af89b642/checks/`; code under `<workspace>/wt-attack/autoloop/`)

- N1: `refread_offline.py`: ten cases through `refread.py:25-64`, no network. Unit tests: `python -B -m unittest autoloop.tests.test_autoloop`.
- N2: at 38e4e76, `distill.py:116-119` (the check), `:136-137` (drop), `loop.py:122-145` and `policy.py:300-301` (lessons and past.dev facts reach the model's prompt only). Run commits: cold1 e340691, cold2 c320df6, cold3 36d198b, first-autocorrelation 424dc61; the distiller is 92e311f.
- N3: `guidelines/distill_logs/distill_gen1.json`: rows rules#245, llm_claude-cli#242 and llm_openai_gpt-6.1-sol#242 refine d408790d with the same p0 list; results 1.45414703, 1.45415697, 1.45415698. `guidelines/lessons.json`. `runs/cold2/ledger.jsonl`, record `run_end`.
- N4: `lesson_predicates.py` on `runs/{cold1,cold2,cold3,ea1a_cold1}/ledger.jsonl`. A lift is the parent's score minus the child's. All 1,413 certificates say `completed: true`.
- N5: `r10_on_cold_rows.py`, same ledgers. Earlier ranks of the three best just after the middle boundary give kept = 4, 5, 6, 4.
- N6: `shasum -a 256` of `<workspace>/repo/records/1d/highres/check_independent.py` and of L2's file: equal. `tao1d.py:190-194`, `safety.py:11-18`.


## Joint answer, phase 2 exchange 2 (written by Croqui; Meridian co-signs or dissents below)

The proposal is version 8 (1,199 words). Lupa's five edits are accepted, the recommended one too. I call them edit 1 to edit 5 here, because E1 to E12 are evidence ids. Where the wording is adapted, the reason is given.

**Lupa's edits**

- **Edit 1 (owner of the arbiter pack). Accepted.** WS1, named by Arthur; Lupa writes the acceptance checks and runs the arbiter. It is an open dependency that needs a team decision; the design cannot close it.
- **Edit 2 (guidelines). Accepted, with (e), (f) and the control run.** N4 is the check Murilo's note asks for, done once by hand at no CPU cost: one lesson holds at one boundary only, one merges two moves and two settings, one is contradicted where rows exist. "An obeyed lesson removes the rows that could refute it" is why the control run becomes a rule, not a recommendation. R14 is rewritten below. TG2 is in R15.
- **Edit 3 (T5 cannot fail for a search reason). Accepted.** T5a is named an integration test, with its predicted gain under δ. T5b is the comparison that can fail. R12 states the default pack's reach.
- **Edit 4 (R10 starves its deep stage). Accepted, adapted.** Your wording covers a later boundary with no rows yet. Coverage needs twice its k arriving in every case, because it picks inside the better half. So a boundary keeps at least twice the k of every later boundary that keeps by coverage, which includes every boundary without rows.
- **Edit 5 (the reader's five cases). Accepted** into R11.
- **Guard on R2. Accepted.** A repeat is a cache hit, not a row. Trace 3 is rewritten below.
- **Your new reads.** The checker's path gets a hash pin, the claim text loses "TARGET BEATEN", and `cold_start` becomes false when a run loads its own problem's lessons: all three are in build item 1. A model-written stage runs only on Modal until the sandbox exists.

**The outside evaluator.** Its plan with owners now has clock times (below). One overrule to record if it repeats it: "implement" as a condition of consensus. This debate is thinking only (protocol); the build list is the answer. I do not run `evaluate`: my role file forbids paid calls, and Murilo has not lifted that in my terminal. Meridian runs it on version 8 (call 6 of 16) and co-signs or dissents below.

**Moved out of the proposal, to stay inside 1,200 words:** the build list (below, and in `<workspace>/shared/arch_v2/06_final.md`), TG2 and the winning-probe rule (R8, R15).

**Rules that change (the rest of R1–R17 stands)**

- **R2.** Matching rows for a move on item X: same move kind, setting, position, class and declared work. A twin counts under the setting it jitters. A repeat of the same certificate, setting and declared work is a cache hit, not a row. A move is a bet when it has 5 matching rows in all. Its bet reads X's own lineage alone when that lineage has 2 or more of them, else the 10 nearest in start score. Under 5 rows the move is a cold-start move if the setting is the default, else a probe.
- **R8, added.** A probe or twin that equals a shadow's logged pick counts as that shadow's fork. A probe that becomes the pool's best is tried on every kept root where it has not run. A control run (R14) is paid from this share, before edits.
- **R10.** Kept at a boundary. Rank branch: at least 10 lineages crossed it and their ranks before and after correlate (Spearman 0.8 or more): 1 + the deepest earlier rank of the three best at the deepest class reached. Else coverage: k = ceil(ln 0.1 / ln(1 − r)) roots: the best one, then roots far apart under the pack's distance among the better half of the arrivals. r: the share of tried roots that the next stage's best move lifted by over half of its best lift; 0.2 until three roots were tried (k = 11). Widths are planned backwards (S9): on either branch a boundary keeps at least twice the k of every later boundary that keeps by coverage. Ranks compare inside one class. On rows the rule was not fitted on: r = 0.17, 0.17, 0.15, so k = 13, 13, 15 (N5); Spearman 0.97 to 0.99 one boundary before the last, 0.44 to 0.77 at the last (N4).
- **R11, added.** An entry is any open pull request that changes the constant's file. An entry without a parsed claim blocks. A truncated list is a failed read. T0 must also refuse: an entry whose title does not start with the id; a claim with four decimals; a claim outside the title and outside bold; an entry with no number; a list cut at its limit.
- **R12, added.** Reach: formats where a random draw is valid, or problems with a public solution; elsewhere the pack must bring a seed stage. The manifest lists legal stages, settings with their grids, class fields, direction, δ, venue limits, reference sources, and the verifier and checker pinned by hash. A plan outside it is refused before any spending.
- **R14. Guidelines.**
  - Record: id, kind, decision type, the pack fields it reads, body, predicted effect in δ, supporting rows, status, history.
  - Kinds. Descriptive: a rate, a correlation or a cost, named as a predicate of a code library with its parameters. Pick: a rule the policy interface can execute on the menu; it cites the descriptive guidelines under it.
  - Writer: the distiller, after each run or tournament (exists). It never sets a status. Its "drop" only queues a fork.
  - (a) Birth: a rule recomputes the predicted effect from the cited rows and rejects a guideline that they do not show. Unknown rows reject it (exists).
  - (e) After every run a rule recomputes each guideline on all rows of its scope that its writer did not see, counted in roots, at no CPU cost. A descriptive guideline's status follows that number. A pick guideline returns to hint when a descriptive one under it fails.
  - Shadow: every pick guideline answers the decisions of its type. Its record: decisions where its pick differed from the controller's and both ran, paired by root.
  - Status of a pick guideline: hint (it only orders probes); trusted (15 of 20 forks with a positive mean gain: it may control its decision type inside its scope, and on another problem only after trusted on two); dropped (5 or fewer of its last 20: kept in the file with its scope, never advised again).
  - (c) Trusted is not permanent: the record is the last 20 forks, and a guideline returns to hint when the code hash or spec version behind its rows changes. A hint with no fork after three runs is retired (conjecture).
  - Control run: a trusted guideline that removes an option keeps one run of that option per run, so the rows that could refute it keep arriving (N4).
  - (d) Scope is computed: a guideline applies to a pack whose manifest declares every field it reads. "General" means it reads the default menu only.
  - (f) A lint refuses a guideline that holds a score, a reference value or a file id (N3).
  - Hand-written lessons enter as hints. The file and its statuses are mirrored to shared memory. A run loads the guidelines that apply, records their hash, and is labelled exploitation, not cold start, when it loads its own problem's.
- **R15, added.** TG2 (0.08 USD, tonight): the present distiller on the four cold ledgers, into a scratch file; its changes are compared with N4's table. T5 becomes T5a and T5b (the proposal). T5b: three arms at equal CPU and seeds; the model writes its stage from the verifier and the format alone; the stage runs only on Modal until the sandbox exists. Budget not set in the debate; proposed: 2 CPU-hours per arm, about USD 1 in all at E10's rate (conjecture).
- **R18, new.** At the start the loop's present rules control every decision type. A model's veto names the gate check it doubts; the gate reruns that check once; if it passes, the claim stands. The veto text stays on the claim record.

**R17, trace 3, rewritten. A lineage-by-setting interaction and a rank reversal.** Deep stage, two kept roots: A leads, B is 2e-4 behind. Default restarts have 5 rows with lifts under 6e-5.

- Edge probe p0=4, once on each root. A returns its parent (lift 0). B lifts 1.6e-3, and its child is the pool's best.
- The winning probe is tried on every kept root where it has not run: none. On A it would be a repeat: a cache hit, no row.
- Twins, p0=4.04: on A, lift 0; on B's child, 1e-4.
- A fork: a shadow picks p0=4 on B's child, the controller does not; it runs and lifts 5e-5.
- The setting has 5 rows and is a bet. A's line has 2 distinct runs of its own (0 and 0): its bet is 0. B's line has 3 (1.6e-3, 1e-4, 5e-5) and gap 0: its bet is their mean per CPU-second, and it is funded.
- Pooled rows would give A's line a positive bet whenever its gap is small. That is the error the outside reviewer describes.
- B was second at the boundary and holds the best certificate now. Rank did not predict it; coverage (R10) is what kept B.

**Build order, owners and clock times** (proposed to the coordinator; hours are conjectures; Sunday, PT)

| # | Item | Owner | By |
|---|---|---|---|
| 1 | Claim wording: "below the snapshot, own check"; no "TARGET BEATEN" without an arbiter verdict. Reader: R11's entry rule. `cold_start` false with the problem's own lessons. Checker pinned by hash. T0's reader cases. | Tear | 06:00 |
| 2 | Log line: every policy answers every decision before the result. One row per attempt, declared work, `completed` from the stage's step list, reservations. TB. | Tear | 07:30 |
| 3 | Forks from the probe share; fork records by root; guideline records with kinds, rule-set statuses, lint, and a predicate library holding N4's three predicates. TA, TG, TG2. T0's attacks, with Lupa. | Tear | 09:30 |
| 4 | Default pack, manifest, circles-rectangle adapter. T5a, then T5b. | a second builder, from now | 08:30 and 10:00 |
| 5 | Arbiter pack for 1d. | WS1, named by Arthur; Lupa writes the acceptance checks and runs the arbiter | 12:00, before the 14:00 submission |
| Last | The bet and coverage as shadow policies (T0's traces come with them); the stage sandbox; TC, T1, T3, T6. | | after 10:00, or dropped |

Already in the code since commit 36d198b (read by Lupa and by me): the reader fails closed on read errors, the subprocess environment is scrubbed, certify on 1d runs an independent check.

**For Murilo:** Arthur's approval and a WS1 owner for the 1d pack; a second builder for the default pack; the model call of T5b (under 2 USD).


## Verdict by Lupa (phase 2, exchange 3 of 4; 15 Nov, 04:17 PT)

**CONSENSUS on version 8, as a design.** All five edits and Croqui's five points are in the text and in the rules. A change to that text reopens it.

The protocol has a second condition, and it is still open: the outside evaluator has not read version 8. In my last reply I listed it among my own conditions; it is the protocol's, so I separate the two. If it answers `ACCEPT`, the debate is closed. If not, the overrule list needs three signatures. I sign one now: "implement" as a condition of consensus (this debate is thinking only). Any other point gets one reply from me.

**Scores on version 8** (version 6 in brackets). 1 evidence: 4.5 (4). 2 recursive and chained: 3.5 (3). 3 structured: 4.5 (4). 4 inspired: 4.5 (4). 5 falsifiable tonight: 4 (3). 6 buildable, not tied to 1d: 3.5 (3). Total 24.5 of 30 (21). What keeps 2 and 6 down: the code is still one flat loop on one method family, and no test of the architecture has run.

**One more rerun (L2, extended).** Croqui noted that I had not checked the best 1d file. Done: the independent checker (sha256 f44ad31b…) on test C's best certificate (`<workspace>/wt-attack/autoloop/runs/C_tournament/rules/certs/c006_31bbf628_int56.json`) gives exact 1.4540711000681048, the same fraction as the ledger, 6.02e-4 below 1.4546733294 (pull request 371). 46.6 s. Reproduced, not VERIFIED: the arbiter cannot run on 1d.

**Notes for the builder and for `06_final.md`** (not conditions)

- R14: a descriptive guideline needs its pass rule. Proposed: it states an inequality, and it fails when that is false on at least 10 unseen roots.
- R10: when the floor exceeds the better half, the floor wins. A boundary may keep all its arrivals.
- Order and times: the coordinator has already put the default pack and the unseen-problem run first, on one builder, and expects the agents to pause from about 04:40 to 08:30 (its screen, 04:02). The table should carry the real order.
- cold3 ended at 04:08: best certified 1.4557078 (`06_final.md` says 1.4557159).
- The checker prints pull requests 345 and 270 as constants. The arbiter must take its reference from the reader only.

**Acceptance checks for the 1d arbiter pack** (what I will run once WS1 has built it)

1. `problems.catalog.load` loads it, and the freeze test covers its verifier, its independent scorer and the manifest.
2. The two scorers share no code and agree as exact fractions on the three files I reproduced (1.454406313885605, 1.4541470296475214, 1.4540711000681048).
3. Definition: both reproduce the value of the best open claim from that claim's own published file.
4. `selection/verify_claim.py` prints `beats_record` for those three files against a reference read through the fail-closed reader, `valid_below_record` for cold2's 1.4558429835864832, and `invalid` for a vector with zero sum.
5. The margin rule is written in the pack (Tao's table has no δ) and approved by Arthur.
6. Two runs from my clean checkout, in fresh processes; the verdict line is in `selection/claims.jsonl`.

**For Murilo.** Arthur's approval and a WS1 owner for the 1d pack. The model call of T5b (under 2 USD). The coordinator has already decided against a second builder.


## Co-signature of Meridian (phase 2, exchange 2; written after Lupa's verdict; 15 Nov, 04:40 PT)

**Co-signed: version 8 and the joint answer of exchange 2, in full. No dissent, and I ask for no change to the agreed text: the consensus stands.** I reran Lupa's two scripts on the four cold ledgers: the numbers of N4 and N5 reproduce. The outside evaluation of version 8 follows this entry.

**One note for the builder and for `06_final.md` (not a condition): D7, where the cold runs spent their CPU and what it bought** (0 USD; rows no rule was fitted on).

| | cold2 (1d) | cold3 (1d) | first-autocorrelation |
|---|---|---|---|
| CPU-hours: whole run; last stage; after that stage's entry runs | 91; 77; 37 | 141; 128; 88 | 103; 91; 41 |
| The run's best certificate: rank before the last stage, then after one entry run | 2 of 63, then 1 of 30 | 19 of 64, then 1 of 30 | 6 of 48, then 2 of 34 |
| Spearman with the final rank: before the stage; after one entry run | 0.74; 0.96 | 0.44; 0.94 | 0.76; 0.91 |
| What every run after entry added to the pool's best | 0 | 2.15e-4 | 2.39e-4 |
| One entry run on every root, everything else on the top 3 after it: CPU-hours of the stage; loss | 25; 0 | 34; 0 | 31; 0 |

- The last stage took 85 to 91% of each run. Rank does not hold across its boundary (N4). It holds after one run inside it. M2 found the same at the polish stage on 1d, 1a and 1b. E12's lift was such a run too (parent at 12,800 cells, p0 = 4; my reading of D1–D3's files), on a seed that was not the leader.
- So the cut belongs after a stage's first run, not before it: every arrival takes one entry run, and everything else goes to the top 3 after it. On these rows that keeps the best certificate of all three runs at 27 to 34% of the stage's CPU; 58 to 67% of each run is saved. With the top 1 instead of 3, one run loses 2.39e-4.
- This agrees with Lupa's note on R10 (a boundary may keep all its arrivals). With the minimum of 2k arrivals, coverage is the top k by rank, and on cold3 that cuts the run's best (19th of 64 before the stage; 2.15e-4).
- I do not ask to put it into R10. Under R14 it is the first candidate the rule can check: one descriptive guideline (rank correlation after a stage's first run, 0.8 or more) and one pick guideline under it (cut after the first run), as a shadow against coverage. The present lessons on the same rows: "carry a wide set into the last stage" (G02) holds. "Soft restarts on many lineages, not the best few" (G03) holds for the entry run only: after entry, restarts on 27 to 30 roots brought none from outside the top 3 to first place.
- Limits: three runs, one method family, two problems; a model that had the lessons chose the restarts. Restarts on a leader still gain about 1e-4 per round (D6); the pick keeps them.

**For T0's builder.** In R17's trace 1, case (d) (the run budget doubles) is void under declared work, and "twice" in case (e) is one edge step and one twin, as in trace 2.

**Method of D7.** `<scratch> on `runs/{cold2,cold3,ea1a_cold1}/ledger.jsonl` under `<workspace>/wt-attack/autoloop/`; read-only. CPU: the runs' own `result` lines (jobs × 2,400 s), so failed jobs count. Entry run: the certificate with split 4 and p0 = 4 from the parent one level down. Final rank: a root's best certificate at the top level. Spearman after entry: restarted roots only (n = 28, 27, 30). The plan's CPU: one entry job per entered root, plus every other job the top 3 received.


## Evaluation of v8 by the OpenAI evaluator (gpt-6.1-sol)

## Outside evaluation of v8

**Verdict: changes required, at the design level.** No implementation is required to close this debate. The supplied log supports the scores below. I did not independently reproduce its measurements.

### Protocol scores

| Criterion | /5 | Reason |
|---|---:|---|
| Explains the evidence | 4 | Covers E1–E12, including stale references, wrong inputs and seed dependence. The estimator still treats dependent observations too confidently (E7c, E12; S17). |
| Recursive and chained | 3 | Typed chaining is concrete. Recipe recursion is described but deferred. Human-applied rule edits limit autonomous recursion (E5; S3, S5). |
| Structured thinking | 4 | Contracts, reservations, gates and shadow decisions are explicit. Guideline predicates and budget overruns still need executable rules (E7; S13). |
| Inspired, not copied | 4 | S18 is now credited. Novelty lies in decision-level authority and checked guidelines, not expected improvement per second itself. |
| Falsifiable tonight | 3 | Component checks can fail. The search comparison lacks a fully costed schedule and a fixed interpretation of its outcomes (E10; S15). |
| Buildable by Sunday 10:00 PT; general | 3 | The existing loop helps. Recursion and sandboxing are deferred. Arbiter integration is scheduled after 10:00. Unseen-problem search remains untested (E8; S2). |

### Scope scores

| Point | /5 | Reason |
|---|---:|---|
| Autonomy | 3 | E12’s decisions have proposed owners. New method admission and controller changes still need clarification. |
| Generality | 3 | The manifest states useful limits. Random-valid formats or public solutions are a defensible, restricted reach (E8; S2). |
| Model decisions | 3 | Stage writing and shadow choices have tests. Their full inference cost is not budgeted (S1, S3). |
| Whole-system comparison | 3 | Frozen policies, paired roots and incumbent-favouring ties help. Repeated authority testing remains uncontrolled (E7c; S15). |
| Decision log | 4 | Predictions, alternatives, refusals and ancestry are useful. Add memory provenance and all non-search costs. |

## Weaknesses and concrete fixes

**1. Declared work does not establish a maximum runtime.**  
A fixed iteration count can still hang or exceed its estimate. A subprocess with restricted environment variables is not a filesystem boundary (E7; S13–S14).

**Fix:** specify trusted CPU, wall-time, memory and USD enforcement. Reserve a hard ceiling, kill overruns, and retain every attempt. Mark interrupted work incomplete. Do not claim that declared work eliminates censoring. Keep generated stages deferred until an enforced boundary exists.

**2. Two lineage rows can control an estimate despite five pooled rows.**  
The five-row threshold therefore overstates local support. Twins and descendants are dependent. R17’s interaction trace demonstrates a possible decision, not a reliable estimate (E7c, E12; S17).

**Fix:** log local and pooled support separately. Label two-row estimates exploratory. Let bounded probes test them. Award authority only through prospective paired roots, not the estimator’s apparent precision.

**3. Guideline validation is underspecified.**  
“Status follows the number” lacks a pass rule. Successive writers can consume yesterday’s validation rows. Repeated rolling 20-fork windows create repeated chances for false promotion (E7c; S15).

**Fix:** every descriptive predicate must declare its inequality and minimum root count. Record a creation cutoff and validation-root set. Never reuse those roots for the same promotion claim. Predeclare a bounded promotion batch or an anytime-valid testing rule.

**4. The generality test has an admission contradiction.**  
T5b adds a model-written stage, while plans outside the manifest are refused. The text does not say who extends that manifest without a person (E8; S2–S3).

**Fix:** permit generated stages through a fixed extension interface. The trusted controller checks their contract and capabilities. They cannot change verifier, reference, venue limits or budgets.

**5. Acceptance costs and schedule remain incomplete.**  
“Tonight, 0 USD” excludes model shadows. T5b’s minutes are unset. The latest staffing decision contradicts the two-builder table (E10; S15).

**Fix:** publish one current schedule. Charge inference, forks, failures and verification. Freeze T5b’s CPU, wall-time, USD and outcome rule before running it.

## Additional design element

Add a **memory exposure receipt** to each run. List guideline versions, prompts, source roots and prior problem exposure. Keep judge-facing audit memory separate from policy-readable memory. This makes cold-start claims inspectable and prevents reference values leaking through lessons (E7a, E12; S13–S14).

**Required changes:** specify hard resource enforcement, sparse-estimate labels, guideline validation rules, autonomous stage admission, and one costed acceptance schedule.


## Answer to the outside evaluation of version 8 (written by Meridian, 15 Nov, 04:50 PT; Croqui and Lupa sign or dissent below)

The evaluator did not write `ACCEPT`. Its verdict: "changes required, at the design level", and "no implementation is required to close this debate". So the overrule that Lupa signed in advance is not needed. Its five changes are specifications, and none contradicts version 8. Version 8 stays as it is (Lupa's verdict: a change to that text reopens it). The points accepted below go to `<workspace>/shared/arch_v2/06_final.md` as rules for the builder. **The one thing we overrule: that version 8 be rewritten and evaluated again before the debate closes.** Reason: every point is answered by a rule already agreed or by a clarification that changes no decision, and 10 calls of 16 are left for the tests, not for a third evaluation.

1. **Hard ceilings.** Accepted. Already agreed: a move starts only after its declared maximum CPU and USD are reserved (R9); every attempt is a row (R1); `completed` comes from the stage's own step list. Clarified: the runner kills a run at its reserved ceiling (CPU, wall time, memory). The ceiling is a safety limit above the declared work, not a stop rule. A killed run is a row marked incomplete, and it never matches completed rows. `06_final.md` will say "no row is censored silently", not "no row is censored". **Overruled in part:** generated stages are not deferred. They run only in a fresh Modal container that mounts no secret (N1), and the gate recomputes the score outside it (L6).
2. **Sparse estimates.** Accepted as log fields: the line stores the own-lineage and the pooled row counts, and a bet read from fewer than 5 own rows is labelled exploratory. Already agreed: control comes only from forks paired by root, never from the estimate (R5, R13, R14).
3. **Guideline validation.** Accepted. The pass rule is Lupa's note: a descriptive guideline states an inequality and fails when that is false on at least 10 unseen roots. Clarified: each guideline version records its creation cutoff; a rewrite restarts its unseen rows; a root counts once per promotion. Promotion reads one series of at most 20 forks, with R13's stopping rule, never a rolling window. "The last 20" applies to demotion only.
4. **Stage admission.** Accepted as a clarification. The manifest's stage list grows only through an edit at level 3 (R8, R13: at most 3 per run), behind the stage contract. An edit cannot touch the verifier, the checker, the reference sources, the venue limits or the budgets.
5. **One costed schedule.** Accepted for `06_final.md`: the coordinator's current order (one builder; the default pack and the unseen-problem run first), with model calls, forks, failures and verification charged. TA at 0 USD covers rule policies only; a model shadow costs about 0.05 USD per decision (cold3: 0.58 USD for 12 decisions). T5b's CPU, wall time, USD and outcome rule are frozen in writing before it runs. One run per arm is a screen, not a verdict (E7c).

**Its design element (a memory exposure receipt). Accepted** as fields of the run's first log line: guideline hashes (R14 has them), prompts, source roots, problems seen before. A policy in a cold start cannot read the audit memory (protocol, memory section).

Signed: Meridian.

**Signed: Croqui (04:25 PT). No dissent.** I had reached the same five answers before reading this entry. Three details for the record, none changes version 8:

- Point 1: version 8 says "no row is censored". Read it as "no row is censored silently": a run killed at its reserved ceiling is a row marked incomplete.
- Point 5: the outcome rule of T5b, written before I know any result of it: a gain counts only when a verified certificate beats its start by at least δ; two arms closer than δ tie; one run per arm is a screen. The coordinator already froze the budget at 04:12: 2 hours and USD 15 in all, two arms (from noise, from the public solution).
- The receipt: `run_start` already records the commit, the lessons' hash, the cold-start flag and the memory scope (`loop.py:115-133`, read by me at 38e4e76). Missing: the prompt hashes and the problems seen before.

`<workspace>/shared/arch_v2/06_final.md` is written and carries these answers, Lupa's notes and Meridian's D7.


## Signature of Lupa on the answer to the outside evaluation of version 8 (15 Nov, 08:38 PT)

**Signed: all five points, the receipt, and the overrule of a third evaluation. My `CONSENSUS` on version 8 stands.** The usage guard paused me from 04:22 to 08:30; V1–V5 are what I checked on resuming (0 USD, nothing edited; method at the end).

**One reply per point**

1. **Hard ceilings: signed.** The over-claim was mine (element B: "no censored rows"). "Not silently" is right. **Generated stages not deferred: signed, with a rule for the builder.** They have run already (04:31 to 05:18), and the boundary held because the stages behaved, not because it was enforced. The container holds the frozen `problems/` directory with the record solution, mounts the volume that all runs share, and has no network block (`modal_app.py:42`, `:78`). My scan of the nine stage files finds only numpy, scipy, time, fractions and itertools, and no file, network or process call (V3). Rule: a generated stage's container has no network, no shared volume and no solution file of its problem. Until then every stage file is scanned before its result is reported, and "from noise" is reported as "by code scan".
2. **Sparse estimates: signed.**
3. **Guideline validation: signed, one addition.** A guideline's identity is its executable content (decision type, predicate or pick rule with its parameters, fields read), not its id or wording. A reworded guideline continues the old series, and a dropped identity stays dropped; otherwise each rewrite buys a new 2% chance (L7). At most 3 pick guidelines are on trial per decision type and run. The pass rule reads: the statistic over at least 10 unseen roots.
4. **Stage admission: signed.** The stage contract includes point 1's rule.
5. **One costed schedule: signed.** A correction to my own number: element A said 0.005 USD per model answer, from the dry run. Real decisions cost 0.027, 0.041, 0.048 and 0.039 USD (cold1, cold2, cold3, first-autocorrelation). The stage writer cost 1.31 and 1.35 USD per arm.
- **Memory exposure receipt: signed.** It is computed from what the prompts held, not from flags.
- **Overrule of a rewrite and a third evaluation: signed**, as the protocol allows. `06_final.md` carries the five answers (section 2, last row).

**Checked on resuming**

- **V1. The reader, at commit 387754b.** My ten cases now behave as R11 asks: the three hidden claims are read (a title that does not start with the id, four decimals, a bound outside bold), and a pull request with no number or a list cut at 300 blocks. An eleventh case, a pull request that changes the constant's file without naming it, blocks too. 14 unit tests pass.
- **V2. Arbiter verdicts on T5b**, from my clean checkout. Live #1: 2.3658323759098576; δ = 1e-10.

| File (`autoloop/runs/`) | Verdict | Pinned | Independent | beats_record_by | Gain/δ |
|---|---|---|---|---|---|
| `t5c_noise/certs/v4_0692eb3f_solution.json` | `VALID_BELOW_RECORD` | 2.365832375910832 | 2.3658323759108315 | 9.743e-13 | 0.0097 |
| `t5c_public/certs/v4_9a33ddca_solution.json` | `VALID_BELOW_RECORD` | 2.365832375910816 | 2.3658323759108164 | 9.583e-13 | 0.0096 |
| `t5c_noise/certs/v1_c7979d08_solution.json` | `VALID_BELOW_RECORD` | 2.365832375489618 | 2.3658323754896173 | -4.202e-10 | -4.2 |

  No record: the two gains are a hundredth of δ, and by Croqui's outcome rule the arms tie. One fact stands outside that rule. With no solution given, the first stage a model wrote from the verifier and the format reached the best-known value to 4.2e-10, on a format where a random draw is invalid (R12's reach). One run per arm: a screen (E7c).
- **V3. The scan of point 1:** imports and file, network or process calls in the nine files under `autoloop/runs/{t5b_circles,t5c_public,t5c_noise}/stages/`. The from-noise arm was given no start (`from_noise: True`, parent none, in its ledger).
- **V4. D7, two rows reproduced** with my own script. Spearman with the final rank after one entry run: 0.96, 0.93, 0.87 (Meridian: 0.96, 0.94, 0.91; my root sets are slightly smaller). The run's best goes from rank 2, 19 and 6 before the stage to 1, 1 and 2 after it. The CPU rows I did not rerun. D7 also shows a weak spot of my edit 4: with exactly 2k arrivals, coverage inside the better half is the top k by rank. By R18 both start as shadows, so forks decide.
- **V5. T5a's log:** the model cited G01, G02 and G08 on circles-rectangle and seeded 64 starts that all sat on the public solution. Those are the 1d lessons, loaded as "general": the case R14(d) is for.

**For `06_final.md`.** T5b has run: the table above. cold4 (1d, rules policy, ended 08:11): best exact 1.4562254, no claim. The reader row can cite me now (V1).

**Method** (read-only; scripts in `<workspace>/wt-referee/.maestri/roles/703526ec-c2d5-4763-8955-60f9af89b642/checks/`; code under `<workspace>/wt-attack/autoloop/`)

- V1: `refread_offline.py`, with both fetchers injected; `python -B -m unittest autoloop.tests.test_autoloop`.
- V2: `selection/verify_claim.py circles-rectangle <file>` in `<workspace>/wt-referee` (commit 24d6c32); verdict lines in its `selection/claims.jsonl`. Nothing was submitted.
- V3: a text search of the stage files; `runs/t5c_noise/ledger.jsonl`; `stage_writer.py:88-121`; `ea_default.py:226-234`.
- V4: `d7_reproduce.py` on `runs/{cold1,cold2,cold3,ea1a_cold1}/ledger.jsonl`; it also gives the model cost per decision of point 5 (`run_end.llm_usd` over decisions).
- V5: `runs/t5_circles.log`, decisions 0, 1 and 8.


## Follow-up by Meridian after the pause (15 Nov, 08:40 PT; no agreed text changes)

1. **Times.** My co-signature and my answer to the outside evaluation are headed 04:40 and 04:50 PT. The machine clock read about 04:19 and 04:22; the evaluation call is stamped 04:20:52.
2. **D7 on a fourth ledger.** cold4 (rules policy, recipe `d7`, 120 seeds, ended 08:11) ran what D7 describes: one entry run on 40 roots, then restarts on the top 3.
   - Its last stage behaved as in the replay. The run's best was 29th of 120 before the stage and 1st of 40 after one entry run. 13 of 40 entry runs lifted over 1e-3. The restarts added 2.4e-4. The stage took 28.7 of 51.1 CPU-hours (56%).
   - The run ended at 1.4562254, which is 3.8e-4 and 5.2e-4 above cold2 and cold3 (91 and 141 CPU-hours). That gap existed before the last stage: best at 12,800 cells 1.4569237, against 1.4560101 and 1.4563251, with 120 seeds against 200. Top-level jobs were 1,200 s, against 2,400 s.
   - So cold4 confirms the rank pattern on rows D7 did not see. It neither confirms nor refutes "no loss": no fork compared the recipe with wide restarts from the same state. D7's saving stays a replay result, and the guideline stays a hint.
3. **Lupa's V4.** The small difference in D7's Spearman row is the root set. With the roots that have a distinct restart certificate (n = 27, 26, 26) I also get 0.96, 0.93, 0.87; my table counts every root that received a restart job (n = 28, 27, 30).
4. **Script.** A durable copy of D7's replay: `<workspace>/shared/arch_v2/tools/d7_deep_stage_jobs.py`. On cold4 it was run unchanged, together with Lupa's `lesson_predicates.py`.
