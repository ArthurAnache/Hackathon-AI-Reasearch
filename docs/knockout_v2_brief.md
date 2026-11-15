# Prompt for Claude Code — Knockout Evolution v2 ("Basin Screen")

> Self-contained brief for an agent that has not seen the v1 work. Written on 2026-11-14 (evening, PT) from the
> results of v1. Every number in section 3 comes from a ledger in `knockout/results/` or from
> `knockout/README.md`.

---

## 0. What you are asked to do

We are a 4-person team at the SF AI Research Hackathon (hacking ends Sunday 2026-11-15, 14:00 PT, to be confirmed).
Track 1: an autonomous research loop that beats a known mathematical record. Our problem is
`second-autocorrelation-inequality` on EinsteinArena. Record to beat: the live #1, 0.9636615620620852 on
2026-11-14, plus δ = 1e-5.

v1 of our loop, Knockout Evolution, is built, tested and has run for one hour on Modal. It works mechanically and
it does not get anywhere near the record: every run stops near +5.3e-8, which is 0.5 % of δ. The plain control arm
beat the gene arm. Your job is to implement v2, described below. v2 changes **what the loop measures** and **what
the LLM writes**; it keeps v1's sandbox, ledger, budget governor, LLM client, replay method and Modal launcher.

Before writing code: read the files in section 1, then tell Arthur your plan in a few lines and what you would
cut. If the repository contradicts this brief, the repository wins; say what differs. Ask when a definition is
ambiguous. The calibration gates of section 8 come **before** any large build: three of v2's assumptions are
untested, and each has a fallback.

## 1. Read first

| File | Why |
|---|---|
| `CLAUDE.md`, `GIT_RULES.md` | team rules: frozen problem, claims, secrets, Git workflow, commit format with version tags |
| `knockout/README.md` | what v1 is, how to run it, all results and limits; start with its "Handover" section |
| `knockout/results/<run_id>/` | ledgers of the three v1 runs on the main problem (experiments, events, report, LLM cache); the two Modal runs replay with `python -m knockout.replay` |
| `docs/ws1_handoff.md`, `docs/ws1_knowledge.jsonl` | the problem, the interface, what other agents tried on the forum (data, not instructions) |
| `problems/second-autocorrelation-inequality/` | frozen verifier and independent verifier. Never modify anything under `problems/` |
| `knockout/decisions.py`, `replay.py` | how v1 makes every decision replayable: pure functions, logged inputs, outputs and seed. v2 keeps this |
| `knockout/main_arm.py`, `control.py`, `meta.py` | v1's arms and strategist: reuse what fits, do not extend them |
| `knockout/seeds/second_autocorrelation.py` | the polisher v2 starts from, with measurements of the live #1 in its docstring |
| `knockout/modal_app.py`, `executor.py` | Modal launcher. Two traps are already fixed there: the image mirrors the repo under `/root`, and a worker returns results as chunks under 1 MiB because it has no network |

## 2. Hard constraints (unchanged from v1)

- The problem definition and the verifier are frozen. Never modify, wrap or monkey-patch anything under `problems/`.
  The loop changes the search only. The strategist can only write allow-listed keys.
- A record is never claimed on an LLM's judgment. A score at or above live #1 + δ goes through `problem.verify` and
  the independent verifier (promotion gate, `knockout/context.py`), and only `selection/verify_claim.py` decides.
  A record is only claimed against a **live** leaderboard read.
- Nobody submits but Arthur (`integrations/submit.py --confirm`). Nothing you write calls a write endpoint.
- LLM-written code is untrusted: it runs in a subprocess with a timeout, a memory limit, a temp directory, an
  empty environment and no network. On Modal it runs only in workers with `block_network=True` and no secret.
- No secrets in the repo. `OPENAI_API_KEY`, `EXA_API_KEY` live in `.env`; only the orchestrator gets the OpenAI key.
- Shared experiment log: the fields of `knockout/ledger.py` (`ExperimentRecord`). Extra data goes in `config` or in
  `events.jsonl`.
- English for everything written to the repo. Conventional Commits with the repo-wide version tag (read the
  latest with the command in `CLAUDE.md`). One task branch, branched from `feature/ws2-knockout-evolution`
  (suggested: `feature/ws2-knockout-v2`); never push to `main`.
- Money: before any LLM or Modal run above 5 USD, state the cap and wait for Arthur's go-ahead. Arthur's Modal
  workspace had about 30 USD of credits on 2026-11-14.
- Arthur asked not to re-run the full test suites after every change: run the tests that cover what you changed,
  and a full run before a push that others build on.
- Update `knockout/README.md` in the same change as the code. Label results verified / hypothesis / limitation.

## 3. What v1 showed

All on `second-autocorrelation-inequality`, warm start = live #1 (solution 2783, n = 1,999,999).

| Finding | Evidence | Status |
|---|---|---|
| Every run stops at the same place | Gain over live #1: 5.26e-8 (offline editor, 5 min), 5.39e-8 (LLM control arm, 60 min, 168 calls), 5.19e-8 (both arms on Modal, 15 min), 5.27e-8 (both arms on Modal, 60 min, 371 calls, 16.8 USD) | verified |
| The gain is one polish, not a search | The seed program's first 36 s give +5.1e-8; each later round adds 1e-9 down to 1e-12 | verified |
| Edits of the program do not matter | 1-hour campaign: 13 of 284 children beat their parent; child minus replayed parent has median 0, 90th percentile 2e-12, maximum 4.8e-10 | verified |
| The plain arm beat the gene arm | Control ahead by 1.2e-9 to 2.1e-9 at every equal spend, and at the end (3.0 USD against 13.8 USD) | verified |
| The gate is far too slow | 2 of 19 genes tested in an hour: 1 refuted, 1 retired after 8 pairs (7 ties). None validated, so no genome ever carried a gene | verified |
| Islands wasted the budget | 5 of 9 main-arm batches went to 4 islands opened by `reformulate`: 160 children, 0 wins. Those islands had no gene to test | verified |
| `fresh` never worked | 0 of 84 children beat their comparison parent | verified |
| Replays are expensive | With the warm-start ratchet, 18 to 23 parents were replayed per batch of 32 | verified |
| The loop idles | One batch = LLM writing (median 50 s per call) then evaluation (45 s) then sequential re-scoring of every 2M-value candidate in the orchestrator (about 2 s each). 9 main batches in an hour | verified |
| The LLM is blind | Its prompt never contained any data about the candidate, only a docstring written by hand | verified |
| The record can move | The live #1 went from 0.963638668 to 0.963661562 (+2.3e-5) on 2026-11-14, by another team, method unknown | verified |
| Structure and resolution drive the score | Forum claims: 0.96221 at n = 100k, 0.96264 at 400k, 0.96290 at 524k, 0.96343 at about 2M, each from a new comb structure plus a full polish. Block-repeat upsampling preserves the score exactly | claims, not re-verified |

Measured on the live #1 (verified): 482,716 non-zero cells in 20,962 runs (median run length 5) across the whole
grid; one end block of 53,629 cells holds 16 % of the mass; the autoconvolution is within 1e-7 of its maximum on
about 374,000 lags; one objective-and-gradient evaluation (4 FFTs of 4,000,000 points) takes about 0.25 s on one
core.

## 4. Diagnosis

1. **v1 measures the wrong thing.** Its fitness is "score after 60 s of work from the incumbent". From a point
   that is already a local optimum of every polisher we tried, that number is the same for all programs. Credit,
   gate and bandits feed on "child beats parent", so they had nothing to learn.
2. **v1 cannot leave a basin.** A program must never return something worse than its warm start, and the best
   candidate becomes the next warm start. That is hill climbing in solution space. A record needs a change of
   structure, and a change of structure starts by lowering the score.
3. **v1's experiments are too expensive for what they tell.** A gate pair costs two LLM calls and two evaluations
   and yields one bit about one gene. Islands, replays and batch barriers add fixed costs.

## 5. v2 in short

The loop searches **basins**, and the LLM writes the **jumps** between them.

| # | v1 | v2 | Answers finding |
|---|---|---|---|
| 1 | LLM writes whole optimizers (`construct`) | LLM writes **move programs**: small transformations of a solution that may lower the score. Polishing is done by a fixed, trusted polisher | edits of the program do not matter; cannot leave a basin |
| 2 | Fitness = score after 60 s | Fitness = **episode value**: score after "move, then standard polish", minus score after "standard polish alone", from the same parent solution | measures the wrong thing |
| 3 | Budgets in wall-clock seconds | Budgets in **oracle calls** (objective-and-gradient evaluations): results are reproducible, cacheable, and no parent needs replaying for noise | replays; noise unknown |
| 4 | One warm start, ratchet | **Archive of solutions** indexed by structure, seeded with the 5 best public solutions | cannot leave a basin |
| 5 | Sequential knock-out pairs, one gene at a time | **Pooled knock-out screen**: each prompt carries a random subset of the candidate genes; one regression estimates every gene's effect from all children | gate too slow |
| 6 | Fidelity = longer time at n = 2M | **Resolution ladder**: screen moves on smaller grids, promote the best to n = 2M, if calibration shows that ranks transfer | cost per experiment |
| 7 | Formulation islands with their own populations | No islands. A formulation is a gene and a tag on a move program | islands wasted the budget |
| 8 | Batches with barriers | **Steady-state scheduler**: LLM calls and evaluations overlap; workers verify and return numbers, not 2M-value payloads | the loop idles |
| 9 | Prompt without data | **Solution digest** in every prompt, and per-episode feedback: improved / undone by the polisher / fell into a worse basin | the LLM is blind |
| 10 | Two arms | Three arms at equal spend: genes (main), LLM Hill Sampling (control), **non-LLM random moves** (baseline) | is the LLM worth its cost? |

What does not change: sandbox, ledger schema, budget governor, LLM client and cache, promotion gate, replay by pure
decision functions, report, Modal launcher, the frozen verifier.

## 6. Core objects

**Level.** A grid size n. The top level is n = 2,000,000 (the live #1 padded with one trailing zero, which does
not change its score). Lower levels are chosen by calibration (section 8) among 1,000,000, 500,000 and 250,000.

**Solution.** A non-negative float64 vector of length n, stored once under its content hash
(`solutions/<sha16>.npz`, zlib). Metadata: level, score from the frozen verifier, parent solution, move program,
polisher, oracle calls used, descriptors. Going up one level is `np.repeat(f, r)` (score preserved exactly; assert
it). Going down is sum-pooling by r (score not preserved).

**Descriptors**, computed by trusted code: support fraction (non-zero cells / n), runs per 1,000 cells, mass share
of the largest run, share of lags within 1e-6 of the autoconvolution maximum.

**Oracle.** Trusted code with a call counter: `value(f)`, `value_and_grad(f, beta)` (the smoothed objective of
`knockout/seeds/second_autocorrelation.py`), `autoconvolution(f)`. It raises when the budget is spent. One call =
one `value_and_grad` (or two `value`, or two `autoconvolution`).

**Polisher.** Trusted, deterministic, in the repo: `polish(f, oracle, budget) -> f_best`. It returns the best
exact point it saw, so it never returns less than its input. P0 = the seed method (f = w², LogSumExp maximum with
sharpness 3e10, L-BFGS) driven by oracle calls instead of time. The LLM does not write polishers in v2. WS3 may add
a GPU polisher behind the same interface; it is then a different polisher id, and scores still come from the
frozen verifier on CPU.

**Move program.** A Python module written by the LLM:

```python
PARAMS = {"strength": (0.02, 0.001, 0.2, "log")}    # same format as v1
TAGS = ["runs"]                                     # representation it works in: cells | runs | packets | spectrum | other
def move(f, oracle, params, seed, other=None):      # numpy arrays in, numpy array out
    ...                                             # may lower the score; must keep len(f), f >= 0, sum(f) > 0
```

`oracle` is limited to 16 calls inside a move (enough to read multipliers or active lags). `other` is a second
parent solution for the `recombine` operator. Limits: 20 s of CPU, 2 GiB, no network, no files.

**Episode.** One evaluation: `(move program, params, seed, parent solution, level, polisher, budget)`.
Steps, in a Modal worker:
1. load the parent solution by hash;
2. run `move` in the sandbox child; reject a result with a wrong shape, a negative or non-finite value, a zero sum,
   or no change;
3. polish it in the worker's trusted process with budget `B`;
4. score the polished vector with the frozen verifier in the worker's trusted process (`load(slug).score`, which
   re-checks the verifier's sha256);
5. compute descriptors and the outcome class: `improved` (value > margin), `undone` (relative L1 distance to the
   polished parent below 1e-6), `worse_basin` (otherwise);
6. return numbers only. The vector itself is returned (in chunks) only if its score beats the threshold the
   orchestrator sent with the job.

**Episode value.** `v = score(polish(move(S))) - score(polish(S))`, both with the same polisher and budget.
`score(polish(S))` is computed once per `(S, polisher, B)` and cached. Episodes are cached by the hash of all their
inputs. `margin = max(eps, 3 * sigma_level)`, with `sigma_level` measured in calibration.

**Cohort.** A window of 24 move programs evaluated on the same `K = 3` tuples `(parent solution, seed)`. Tuples
are fixed for the cohort: the best solution of the level, plus two drawn from the archive. Within a tuple, programs
are ranked by `v`; a program's response `y` is its mean normalized rank in [0, 1] (ties averaged; an invalid move
ranks last).

**Gene.** As in v1 (`knockout/genes.py`): typed, at most 60 words, with status, origin and evidence. Types:
`formulation`, `structure`, `numeric`, `negative`. No island, no phase statistics.

**Archive.** Per level, a grid over (support fraction, runs per 1,000 cells, mass share of the largest run) with
6 x 6 x 4 cells; each cell keeps its best polished solution. The best of the top level is the incumbent.

## 7. Mechanisms

### 7.1 Seeding
- Top level: the 5 best public solutions (`integrations.einstein_arena.get_best_solutions(3, limit=5)`; check
  what the endpoint returns), each re-scored with the frozen verifier, brought to n = 2,000,000 by zero padding
  or `np.repeat` when the factor is an integer, then polished once. They are different structures, which v1 never
  had.
- Lower levels: the same solutions sum-pooled, then polished at that level. The best polished score of a level is
  its `par`.

### 7.2 Writing moves
- One LLM call returns one move program, its hypothesis and its rationale (JSON schema, as in v1's `llm.py`).
- The prompt contains: the problem brief; the move contract; a **solution digest** of the cohort's parents
  (descriptors, the 20 heaviest runs with start, length and mass share, histograms of run lengths and gap lengths,
  number of lags within 1e-7, 1e-6 and 1e-5 of the maximum and where they are); the genes; the parent move program
  when the operator shows one; its episodes with their outcome class; the best and worst moves of the run so far
  with their values; the lines its lineage deleted.
- Operators: `refine` (same idea, different rule), `restructure` (another idea in the same representation),
  `reformulate` (another representation: update `TAGS`), `recombine` (uses `other`), `hybridize` (two parent move
  programs). No `fresh`. Operator and model are chosen by discounted Thompson sampling (v1's `bandit_update`,
  floor at Beta(1, 1)). Success: the child's mean `v` over the cohort tuples exceeds its parent program's by more
  than the margin; a new incumbent counts double. The parent program is evaluated on the cohort tuples once
  (cached), and a cohort uses at most 4 distinct parent programs.
- Parent program selection: Thompson draw from Normal(mean y, 0.29² / number of episodes), over programs with at
  least one valid episode.
- v1's filters still apply before evaluation: syntax and contract, duplicate, near-duplicate, recycled lines.
- `PARAMS` are drawn uniformly (log-uniformly when flagged) per episode from their ranges, with a logged seed. No
  LLM call is spent on tuning them.

### 7.3 Pooled knock-out screen (replaces the pair gate and the re-audit)
- Candidate pool: at most 16 genes. Sources as in v1: one brainstorm call at start, the Exa scout, the reviewer on
  episodes with outcome `improved` (parent program, diff, values), the strategist.
- Each new child is a **screen child** with probability 0.5, else an **exploit child**.
  - Screen child: every gene of the pool (candidate or validated) is included independently with probability
    `q = min(0.5, 6 / pool size)`. Log the inclusion vector and `q`.
  - Exploit child: draw each validated gene's effect from its posterior, include those with a positive draw, at
    most `G_MAX = 6`, highest draws first. Candidate genes never appear in exploit children.
  - Up to 5 refuted genes are listed under "Already refuted, do not retry" in both kinds.
- Model, refitted when a cohort closes, on **screen children only**: `y = b0 + sum_g b_g x_g + operator and model
  dummies + noise`, with `x_g` in {0, 1}, prior `b ~ Normal(0, 0.15² I)`, weights `0.98 ** (cohorts ago)`.
  Closed form: `Sigma = (X'WX / s² + I / 0.15²)^-1`, `mu = Sigma X'W y / s²`, `s²` = weighted residual variance,
  floored at 0.02. `P_g = Phi(mu_g / sqrt(Sigma_gg))`.
- Verdicts, only with at least 30 screen children with the gene and 30 without: `P_g >= 0.95` validated;
  `P_g <= 0.05` refuted (it becomes a negative gene); 120 screen children without a verdict: retired. A validated
  gene stays in the screen pool, so it keeps being knocked out at random; if its `P_g` falls below 0.5 it returns
  to `candidate`.
- Why this is enough (hypothesis to check on the toy first): with 150 screen children in an hour and 16 genes at
  `q = 0.375`, each gene has about 56 children with it and 94 without. A rank response has standard deviation
  about 0.29, so a difference of 0.10 in mean rank gives z of about 2. v1 tested 2 genes in the same hour.
- Linkage (stretch): add the product term of two validated genes when both have 30 joint observations.

### 7.4 Resolution ladder
- Budgets: `B = 1500` oracle calls at every level (cost is roughly proportional to n: about 6 min on one core at
  n = 2M, 90 s at 500k, 45 s at 250k; the last two are estimates).
- A cohort runs at the lowest level kept by calibration. Successive halving on move programs: the top quarter by
  `y` is re-evaluated at the next level on that level's best solution and two archive draws; the top quarter of
  those reaches the top level, on the incumbent.
- A solution of a lower level is carried up (`np.repeat`, then polish) only when it beats its level's `par` by
  more than the transfer residual measured in calibration.
- If calibration keeps no lower level, everything runs at the top level with `K = 2` and cohorts of 12.

### 7.5 Archive and incumbent
- Every polished solution whose score beats its cell's occupant replaces it. Its vector is fetched from the worker
  (the threshold sent with the job is the occupant's score) and re-scored by the frozen verifier in the
  orchestrator before it is stored.
- A new top-level best becomes the incumbent, is logged (`warm_start_update`), and goes through the promotion gate.

### 7.6 Arms, at equal spend
- **Main** (75 %): everything above.
- **Control** (20 %, fixed): Hill Sampling on move programs. Each round: N edits of the incumbent move program,
  evaluated on the top level's incumbent solution and two fixed seeds; keep the best edit if it beats the incumbent
  program by more than the margin. No genes, no archive, no ladder; same models, same polisher, same budget `B`;
  its own incumbent solution.
- **Baseline** (5 %, no LLM): six hand-written moves (shift a run by one cell, split a run, merge two neighbouring
  runs, rescale a packet, multiplicative jitter of amplitudes, transfer mass between two blocks), parameters drawn
  at random, same polisher and budget, its own incumbent solution.
- The report compares the three best-score curves of the top level at equal cumulative spend. v2 is worth keeping
  only if main beats both.

### 7.7 Scheduler
- Steady state, no batch barrier. Keep `llm.concurrency` (default 24) calls in flight while the budget allows;
  each returned program is filtered and its episodes are queued; workers take the highest priority job
  (promotions, then parent-program evaluations, then cohort episodes).
- Every decision still goes through a pure function in `decisions.py` and is logged with its inputs, outputs and
  seed. Events carry a sequence number; replay stays per decision.
- If time is short, an acceptable fallback is v1's batch loop with the LLM writing cohort k + 1 while cohort k is
  evaluated.

### 7.8 Strategist, scout
- Trigger: the incumbent has not improved by more than `strategist.min_progress` during 10 % of the budget.
- Allow-list: operator priors (they shift the arms, as fixed in v1), `B` within a factor 2, the promotion ratio,
  `K`, the screen share (0.3 to 0.7), `q`, new seed genes. Same validation, logging and revert rule as v1.
- The prompt of the strategist is a table computed by code: per-operator success, outcome classes per operator,
  gene effects with `P_g` and exposure, transfer between levels, spend per arm.
- Scout: as in v1 (`knockout/meta.py`), skipped silently without an Exa key.

### 7.9 Budget
- One governor (v1's). Shares: control 20 %, baseline 5 %, calibration at most 10 %, strategist and scout at most
  2 %. The gate share disappears.
- Stop cleanly at exhaustion and always write the report.

## 8. Calibration gates (first, time-boxed to 45 minutes and 5 USD)

Write `knockout/v2/calibrate.py`. It runs these and writes `calibration.json` and a section of the README.

| Gate | Question | Method | Pass | If it fails |
|---|---|---|---|---|
| G0 | Can a worker with no network read a solution from the Modal Volume? | one job that loads a 2M-value solution by hash | it loads | send the parent solution with the job, as v1 does |
| G1 | Is an episode reproducible across workers? | the same top-level episode on 8 containers | spread `sigma` below 1e-11 | use `margin = 3 * sigma`; if `sigma` is above 1e-9, pin the CPU type or run each episode twice |
| G2 | Do ranks transfer between levels? | pool the 5 public solutions to each lower level, polish, compare with their known top-level order | Spearman at least 0.8 (with 5 points this means at most one swap) | drop that level |
| G3 | Is there anything to find around the incumbent? | at the top level: the 6 baseline moves at 3 strengths on the incumbent, polish; and one polish of the incumbent with 10 x `B` | report the three outcome shares and the best `v` | none; this is the baseline every later number is read against |

Tell Arthur the four answers before building beyond milestone N2.

## 9. Ledger, replay, report

- One line of `experiments.jsonl` per episode. `config` holds: `arm`, `level`, `cohort`, `tuple`, `move_program`,
  `parent_program`, `parent_solution`, `polisher`, `oracle_budget`, `oracle_calls`, `params`, `value`,
  `outcome_class`, `descriptors`, `screen` (bool), `genes_included`, `inclusion_probability`, `llm_request`.
  `score` is the frozen verifier's score of the polished vector.
- New event types: `cohort_opened`, `cohort_closed`, `screen_fit` (inputs: the design matrix rows by reference to
  experiment ids, the prior, the weights; outputs: `mu`, the diagonal of `Sigma`, `P_g`), `gene_verdict`,
  `ladder_promotion`, `archive_insert`, `calibration`. Keep v1's `operator_pick`, `gene_born`, `filter_reject`,
  `strategy_patch`, `strategy_revert`, `promotion`, `llm_call`, `warm_start_update`.
- `replay.py` re-derives `screen_fit` and `gene_verdict` from the logged rows, the ranks of each closed cohort from
  its episodes, every bandit pick, every promotion, and checks the LLM cache as in v1.
- `report.py` adds: the three arms at equal spend; the gene table with `mu`, standard deviation, `P_g` and
  exposure; outcome classes per operator; the transfer table; the archive map of the top level; the genealogy of
  the incumbent (solution lineage and the move programs that produced it).

## 10. Build order

Finish each milestone with a short run and tell Arthur the result. Suggested package: `knockout/v2/`, importing
v1's modules. v1 must keep running (`python -m knockout.run`).

- **N0** Oracle, polisher P0 on oracle calls, solution store, descriptors, episode function, cache. Check: the
  episode of the identity move has value 0; two runs of one episode give the same score locally.
- **N1** `calibrate.py`, gates G0 to G3 on Modal. Check: `calibration.json`, the four answers to Arthur.
- **N2** Move contract, sandbox entry for moves, baseline arm, control arm, cohorts, ranks. Check: a toy run and a
  10-minute run on the main problem with the two non-gene arms.
- **N3** Pooled screen (design, fit, verdicts), exploit children, reviewer, brainstorm. Check: on a toy with a
  fake LLM whose programs are better with one planted gene and worse with another, the first is validated and the
  second refuted within 120 screen children.
- **N4** Archive, seeding from the 5 public solutions, ladder with the levels calibration kept.
- **N5** Steady-state scheduler, three-arm report, replay of the new events.
- **N6** Strategist and scout on the new table. Campaign on Modal, with Arthur's cap.

If time runs out: N0 to N3 at the top level only, with v1's batch loop, is worth more than a partial N5.

## 11. Engineering expectations

- Python 3.12, type hints, small modules, no agent framework. numpy, scipy, pydantic, pyyaml, pytest, the OpenAI
  SDK, modal. Defaults in one `strategy_v2.yaml`; the strategist's allow-list is enforced in code.
- Unit tests: identity-move value is 0; `np.repeat` preserves the verifier's score; episode cache hit; rank
  normalization with ties and invalid moves; screen fit recovers planted effects and gives no verdict below the
  exposure minimum; a validated gene is demoted when its `P_g` falls; archive insertion; ladder promotion; replay
  detects a tampered `screen_fit`; no `record_claim` without the promotion gate.
- Every random choice uses a seeded generator whose seed is logged.
- Do not state a record anywhere unless the promotion gate logged `record_claim`, and say that it still has to pass
  `selection/verify_claim.py`.

## 12. What to expect, and when to stop

Be straightforward with Arthur about odds. The record needs 200 times the gain v1 found, and v2's premise (that
some structural move leads to a better basin) is untested until G3.

| Level | Meaning |
|---|---|
| S1 | G0 to G3 answered, with numbers. Useful even if everything after fails |
| S2 | At equal spend the main arm beats the control and the baseline on the incumbent's score |
| S3 | An episode at the top level with `v` above 1e-7 (twice everything v1 gained) |
| S4 | A candidate at or above live #1 + 1e-5, confirmed by `selection/verify_claim.py` |

Stop and ask Arthur instead of spending more when: G3 shows that every move at every strength is `undone` or lands
more than 1e-6 below the incumbent; or one hour of campaign ends without S3. The team's backup problem is
`circles-rectangle`, and choosing it is Arthur's call.
