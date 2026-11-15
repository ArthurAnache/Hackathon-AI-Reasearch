# Prompt for a coding agent — Knockout Evolution "Start Portfolio"

> Self-contained brief for an agent that has not seen this work. Written on 2026-11-15 (PT) by Arthur with Claude
> Code. Base: branch `genetic-method` (= `feature/ws2-knockout-evolution` @ `60e3856`, Knockout Evolution v1,
> milestones M0–M5). If the repository contradicts this brief, the repository wins: say what differs.

---

## 0. Goal, in one paragraph

We are not trying to hand-craft a solution or a method for one problem. We are building the **agentic genetic
structure** that discovers methods, which then find good solutions. v1 of that structure (Knockout Evolution) works
mechanically but never learned anything: every program started from the record solution, every program landed on
the same score, so "child beats parent" was noise and the genes, the knock-out gate and the bandits had no signal.
This brief adds a **start portfolio**: programs are evaluated from three kinds of starting points (the record
basin, its neighbourhood, and cold starts), the loop learns which genes and operators work in each, keeps an
archive of distinct basins it discovers, and deep-polishes any basin that comes close to the record. The agents can
both exploit the record's basin and move away from it to find other records.

Time box: about 3–4 hours of implementation, then short comparison runs. The hackathon ends Sunday 2026-11-15,
14:00 PT (to be confirmed). Build in the order of section 9 and stop at the first milestone that does not fit.

## 1. Read first

| File | Why |
|---|---|
| `CLAUDE.md`, `GIT_RULES.md` | team rules: frozen problems, claims, secrets, Git workflow, commit format with version tag |
| `knockout/README.md` | what v1 is and why it stalled; read "Handover", "Both arms" and "Limitations" |
| `knockout/main_arm.py` | the arm you will modify: `MainArm.step()`, `_pick_island`, `_draft`, `_evaluate_f0`, `_account`, `_credit`, `_record_score`, `_ladder`, `_update_best` |
| `knockout/context.py` | `RunContext.submit(source, budget_s, seed, init=...)` and `finish()`: the one evaluation path, and the promotion gate |
| `knockout/decisions.py` | every decision is a pure function of logged inputs and a seed; `thompson_pick`, `bandit_update`, `gene_posterior`, `phase` |
| `knockout/replay.py`, `knockout/ledger.py` | replay re-derives each decision from its event; the ledger validates event types |
| `knockout/llm_editor.py` | `problem_brief()` and `state_block()`: what the LLM is told (record value, distance to record, forum knowledge) |
| `knockout/run.py` | CLI, `SEEDS`, `load_init`, profiles |
| `knockout/problems_api.py` | `ProblemHandle` over the toy and the frozen catalog |
| `knockout/seeds/toy_circles.py` | seed program of the toy (26 circles in the unit square); the circles-rectangle seed is derived from it |
| `problems/circles-rectangle/` | frozen problem: 21 circles in a rectangle of perimeter 4, maximize the sum of radii. Never modify `problems/` |

## 2. Hard constraints (unchanged)

- The problem definitions and verifiers under `problems/` are frozen. Never modify, wrap or monkey-patch them.
- Fitness always comes from the frozen verifier, run in the trusted process (`ctx.finish`). Never from the
  sandbox's guide value, never from an LLM.
- A record is never claimed by the loop. A score at or above record + δ goes through the promotion gate
  (`context.py`); only `selection/verify_claim.py` decides. Nothing submits (`integrations/submit.py` needs Arthur's
  `--confirm`).
- LLM-written code is untrusted and runs only in the sandbox (timeout, memory limit, temp dir, empty env, no
  network). Trusted code added by this brief (start generation, descriptors, archive) runs in the orchestrator.
- No secrets in the repo, in logs or in Modal images.
- Every new random choice uses a seeded generator; the seed is logged. Every new decision is a pure function in
  `decisions.py`, logged with inputs, outputs and seed, and re-derived by `replay.py`.
- v1 must keep running unchanged when the new option is off (`portfolio.enabled: false`, the default).
- Money: any run with the LLM or on Modal needs Arthur's go-ahead with a stated USD cap. Tests and fake-LLM runs
  are free.
- Git: branch `feature/ws2-start-portfolio` from `genetic-method`. Conventional Commits with the repo-wide version
  tag: read the latest with `git log --all --oneline | grep -oE 'V[0-9]+(\.[0-9]+)?' | head -1` and bump the minor
  (it was V4.92 on 2026-11-15). Stage explicit paths. Never push to `main`.
- Run only the tests that cover what you changed; a full knockout test run before a push.
- Update `knockout/README.md` in the same change as the code (new section "Start portfolio"), with results labelled
  verified / hypothesis / limitation.

## 3. Why v1 learned nothing (the facts this design answers)

| v1 fact (verified in the ledgers) | Consequence | Answer here |
|---|---|---|
| Every program started from the record solution (`--init live`) and could not return worse | all children ≈ parent (median gain 0); hill climbing in one basin | three start regimes, two of them away from the record |
| Fitness = score after F0 seconds from the incumbent | no spread between programs, so credit, gate and bandits had no signal | cold-start fitness: programs differ a lot from weak starts |
| The LLM prompt contained the record value and the remaining gap | pushed every edit toward fine polishing | `prompt.show_record: false`; the LLM sees only its regime and normalized progress |
| Islands opened by `reformulate` drew whole batches and had no genes | 5 of 9 batches wasted | regime bandit with floors; basin archive replaces "a new island per formulation" as the way to diversify |
| Forum knowledge and Exa genes were always in the prompt | bias toward methods others already tried | `prompt.knowledge: on/off` switch, itself an experiment |

## 4. Core objects

### 4.1 Regimes

Three regimes, names fixed: `record`, `neighbour`, `cold`.

| Regime | Start set (what `init` is) | What a child must beat |
|---|---|---|
| `record` | the **incumbent**: the best solution known to the run. At start: the frozen certificate or `--init live` if given; for the toy (no certificate), the best solution found so far by any regime | its parent's score from the same incumbent (v1 behaviour, with the ratchet) |
| `neighbour` | K_n starts drawn from: (a) the incumbent perturbed with a logged strength, (b) entries of the basin archive other than the incumbent's cell | its parent's mean score on the same start set |
| `cold` | K_c fixed weak starts generated once per run by trusted code, with a logged seed | its parent's mean score on the same start set |

Defaults: K_n = 2, K_c = 2 (each start is one sandbox job, so a child costs K evaluations in those regimes).

### 4.2 Start set

```python
@dataclass(frozen=True)
class StartSet:
    id: str                  # e.g. "cold:0", "record:3", "neighbour:7"; changes whenever a member start changes
    regime: str              # record | neighbour | cold
    starts: tuple            # canonical candidates (or None for "no init")
    start_scores: tuple      # frozen-verifier score of each start (None if a start is not a valid candidate)
    origin: tuple            # per start: "incumbent", "perturb:<strength>", "archive:<cell>", "cold:<k>"
```

A start set is immutable. `record` gets a new id when the incumbent moves (this replaces v1's `epoch`); `cold`
keeps one id for the whole run; `neighbour` gets a new id each time it is redrawn (at most once per batch of that
regime).

**Member scores are keyed by start set**, not by epoch: `Member.scores["F0@<start_set_id>"] = mean score`.
`Member.score(fid, epoch)` becomes `score(fid, start_set_id)`. A parent with no score on the current start set is
evaluated on it before its children are compared (v1's replay rule, generalized), and that score is cached on the
member, so a parent is evaluated at most once per start set.

### 4.3 Program fitness per regime

- `record`: as in v1 with the ratchet: score from the incumbent; `fit` = gain over the incumbent's score.
- `neighbour` and `cold`: the program runs once per start (same `construct` seed); its regime score is the **mean
  of the frozen-verifier scores** over the start set. An invalid result on one start counts as that start's score
  (for maximisation, use the start's own score if valid, else the worst score seen on that start set; document the
  rule in code). Comparisons are always between programs on the same start set, so no normalization is needed for
  selection.
- **Normalized progress** (for the prompt and the report only, never for selection):
  `p = (score - mean start score) / (reference - mean start score)`, where `reference` is the record value read by
  trusted code. Clip to [-1, 2]. With `prompt.show_record: false`, the LLM sees `p` and never `reference`.

`Member.fit` (the ranking key for parent selection) is per regime: keep `Member.fit_by_regime: dict[str, float]`
and rank parents with the fit of the regime the batch runs in. A member with no fit in that regime ranks after all
members that have one (it can still be drawn, which forces its evaluation on the start set).

### 4.4 Cold starts (trusted, per problem)

`knockout/starts.py`:

```python
def cold_starts(problem_id: str, k: int, seed: int) -> list   # canonical candidates, valid or not
def perturb(problem_id: str, candidate, strength: float, seed: int)  # canonical candidate
def descriptors(problem_id: str, candidate) -> dict[str, float]
```

| Problem | `cold_starts` | `perturb` | `descriptors` |
|---|---|---|---|
| `toy-circles-26` | 26 random centres in [0.1, 0.9]², radius 1e-3 each | move each centre by N(0, strength), shrink radii until valid | sorted-radii quartiles (4 values), number of circles touching the boundary (within 1e-9) |
| `circles-rectangle` | 21 random centres in a random rectangle of perimeter 4 (aspect ratio drawn in [0.5, 2]), radius 1e-3 | same, keep inside the bounding rectangle | aspect ratio of the bounding box, sorted-radii quartiles, number of circles of radius below 0.5 × median |
| `second-autocorrelation-inequality` (optional) | positive random vector of length 50,000 (log-normal), or a coarse comb | multiply a random 5 % of the runs by exp(N(0, strength)) | support fraction, runs per 1,000 cells, mass share of the largest run |

Strength for `perturb` is drawn log-uniformly in `[portfolio.perturb_min, portfolio.perturb_max]` with a logged
seed. A start that the frozen verifier rejects is allowed (the program must repair it); log its score as None.

### 4.5 Basin archive

`knockout/basins.py`, trusted:

- A basin cell is a discretization of the descriptors (`portfolio.archive_bins` bins per descriptor, bin edges fixed
  at run start from the descriptors of the cold starts and the incumbent; log them in `run_start`).
- Each cell keeps its best valid solution (score from the frozen verifier), with the experiment id, regime and
  program that produced it.
- Every valid candidate of every regime is offered to the archive. Insert if the cell is empty or the candidate
  beats the occupant (`D.improves` with `selection.eps_rel`). Log `basin_insert`.
- `incumbent_cell` = the cell of the incumbent. "Other basins" = every other occupied cell.
- Store solutions as v1 does (`ledger.save_solutions` rules); the archive keeps only the experiment id and the
  path, not the object, when the candidate is large.

### 4.6 Deep polish (promotion of a basin)

When an archive entry outside `incumbent_cell` reaches a normalized score within `portfolio.promote_gap` of the
incumbent (default: `incumbent_score - entry_score <= promote_gap * |incumbent_score - mean cold start score|`,
default `promote_gap = 0.02`), and it has not been deep-polished yet:

1. take the current best `record`-regime program (highest `fit_by_regime["record"]`);
2. run it from that entry with budget `portfolio.deep_s` (default 10 × `fidelity.f0_s`), seed logged;
3. re-score with the frozen verifier (as every evaluation); log a `basin_promotion` event with the entry, the
   program, the score before and after;
4. if the result beats the incumbent: it becomes the new incumbent (new `record` start set id), and the event says
   `new_incumbent_from_other_basin: true`. This is the key success signal of the whole design.

At most `portfolio.max_deep_per_batch` (default 1) per batch; charge it to the main arm's budget.

## 5. Mechanisms

### 5.1 Regime choice (one per batch)

Replace nothing in v1's island logic; add a regime pick **before** the island pick in `MainArm.step()`. All children
of a batch run in the same regime (simpler accounting, one start set per batch).

Pure function in `decisions.py`:

```python
def pick_regime(arms: dict[str, list[float]], spend: dict[str, float], floors: dict[str, float],
                seed: int) -> dict:
    """If some regime's share of the main arm's spend is below its floor, return the regime with the largest
    shortfall (floor - share), ties by name. Otherwise Thompson sampling over the Beta arms.
    Returns {"choice", "reason": "floor" | "thompson", "shares", "draws"}."""
```

- `arms`: Beta(α, β) per regime, start at Beta(1, 1), updated with v1's `bandit_update` (discount
  `bandit_discount`). Success for a batch: at least one child beat its parent in that regime by more than
  `eps_rel`; a child that also improved the archive in a cell other than `incumbent_cell` counts double.
- `spend`: USD spent by the main arm per regime so far (track it in the arm; the governor has only categories).
- `floors` default: `{record: 0.2, neighbour: 0.2, cold: 0.2}`; the remaining 40 % follows the bandit.
- Log `regime_pick` with inputs, outputs and seed. Add the event type to `ledger.py` and a derivation to
  `replay.py`.

### 5.2 Genes and credit per regime

- v1 keeps gene statistics per phase (`Gene.stats[phase] = [a, b, n]`, `D.gene_posterior` pools phases when a
  phase has fewer than `genes.phase_min_obs` observations). With the portfolio on, the key becomes
  `f"{regime}/{phase}"`. Pooling across keys then happens automatically when a key has few observations.
- `_credit` and `build_genome` read and write the key of the batch's regime. A gene can therefore be validated for
  `cold` and refuted for `record`; the genome of a child is drawn from the posterior of its own regime.
- Knock-out gate: a gate pair runs in the regime of its batch and is logged with it. Keep v1's sequential gate
  thresholds. A gate verdict applies to the gene globally in v1; here, store the verdict per regime
  (`Gate.regime`), and let a gene be `validated` in one regime and `candidate` in another: the genome pool of a
  regime uses only genes validated in that regime (plus v1's rule for freshly validated genes). If this is too
  large a change for the time box, keep the global verdict and note it as a limitation.
- Operator and model bandits: keep one set of arms per regime (`operator_arms[regime]`), initialized from
  `operator_priors`. Different regimes plausibly favour different operators (`fresh` may work from cold starts and
  never from the record).

### 5.3 What the LLM is told

In the prompt built by `MainArm._prompt` (and `state_block` if reused):

- A **regime section**, always: which regime, what `init` will be (for `neighbour` and `cold`: "your `construct`
  is called once per start; each start is a canonical candidate that may be invalid or poor; return a valid
  candidate"), the descriptors of the starts, the parent's mean score and normalized progress `p` on this start
  set, and the best `p` seen in this regime.
- `prompt.show_record` (default **false** with the portfolio on): when false, remove the record value, the record
  threshold and "`init` is X away from the record threshold" from the prompt. Keep them in trusted code (promotion
  gate, report). When true, v1 behaviour.
- `prompt.knowledge` (`on` | `off`, default `on`): when off, `problem_brief` omits `docs/ws1_knowledge.jsonl`, the
  Exa scout is not called, and genes with origin `scout` are not created. Log the value in `run_start`.

### 5.4 Start sets in the evaluation path

- `MainArm` holds `self.start_sets: dict[str, StartSet]` (current set per regime) and `self.incumbent`,
  `self.incumbent_score`.
- `_evaluate_f0` submits, for each child and each parent that needs a score on the batch's start set, one job per
  start: `ctx.submit(source, f0, eval_seed, init=start, adapter_source=...)`. Aggregate in the trusted process
  (mean of the per-start frozen-verifier scores, rule of 4.3). Keep the per-start scores and experiment ids in
  `config` (`per_start`).
- One experiment line per (program, start set) with `score` = the best per-start score (the candidate stored is that
  one, so the ledger's `score` always matches a stored candidate) and `config.regime_score` = the mean, which is
  what selection uses. State this choice in the README.
- `record` regime: unchanged from v1 (one start), including the ratchet: when a `record`-regime candidate beats the
  incumbent, it becomes the incumbent and the `record` start set id changes (`warm_start_update` event, as now).
  A candidate from another regime that beats the incumbent also replaces it (log `warm_start_update` with
  `regime`).

### 5.5 Control arm

Unchanged (Hill Sampling from the incumbent, `record` regime only, fixed 20 % of spend). It is the baseline the
portfolio must beat. Do not give it the archive or the portfolio.

## 6. Configuration (`strategy.yaml`)

```yaml
portfolio:
  enabled: false            # v1 behaviour when false
  regimes: [record, neighbour, cold]
  floors: {record: 0.2, neighbour: 0.2, cold: 0.2}
  k_cold: 2
  k_neighbour: 2
  cold_seed: 7
  perturb_min: 0.01         # relative strength, log-uniform
  perturb_max: 0.3
  neighbour_archive_share: 0.5   # probability that a neighbour start is an archive entry rather than a perturbation
  archive_bins: 4
  promote_gap: 0.02
  deep_factor: 10           # deep polish budget = deep_factor * fidelity.f0_s
  max_deep_per_batch: 1
prompt:
  show_record: true         # set to false by the "portfolio" profile
  knowledge: on
```

Add a profile in `run.py`: `"portfolio": {"portfolio.enabled": True, "prompt.show_record": False}`. Add
`portfolio.floors`, `portfolio.promote_gap` and `portfolio.k_cold` to `strategist_allow_list` with range checks
(floors each in [0.1, 0.4], sum ≤ 0.9).

## 7. Ledger, replay, report

- New event types (add to `ledger.py`): `regime_pick`, `start_set` (id, regime, origins, start scores, seed),
  `basin_insert` (cell, experiment id, score, previous occupant), `basin_promotion` (entry, program, budget, score
  before/after, `new_incumbent_from_other_basin`).
- `experiments.jsonl` `config` gains: `regime`, `start_set`, `per_start` (list of {origin, score, experiment}),
  `regime_score`, `progress` (normalized), `cell`.
- `replay.py`: re-derive `regime_pick` from its inputs; check that every `basin_insert` is consistent with the
  scores in the ledger (the inserted score beats the previous occupant); keep all v1 checks.
- `report.py` adds:
  - per regime: spend, batches, children, share of children that beat their parent, best `regime_score` curve;
  - gene table per regime (posterior mean, observations, verdict);
  - operator success per regime;
  - basin archive: occupied cells, best score per cell, which regime found it;
  - every `basin_promotion`, and in bold any `new_incumbent_from_other_basin`;
  - main arm vs control at equal spend on the incumbent score.

## 8. Tests (fast, no network, fake LLM)

Add `tests/test_knockout_portfolio.py`:

1. `pick_regime` returns the most under-floor regime when a floor is violated, and a Thompson choice otherwise;
   same seed, same output.
2. `cold_starts` and `perturb` are deterministic for a seed and return the canonical shape for the toy and for
   circles-rectangle.
3. A member's score is cached per start set: a parent is evaluated once per start set id.
4. Regime score = mean over starts; the ledger `score` equals the stored candidate's frozen-verifier score.
5. Gene stats are written under `"<regime>/<phase>"` and `gene_posterior` pools when a key has few observations.
6. Archive: insert into an empty cell; replace only on a strict improvement; `incumbent_cell` follows the incumbent.
7. Deep polish fires once for an entry within `promote_gap`, and a result above the incumbent replaces it with
   `new_incumbent_from_other_basin: true`.
8. `prompt.show_record: false` removes the record value from the prompt text (assert on the rendered prompt).
9. With `portfolio.enabled: false`, the existing v1 tests pass unchanged.
10. A recorded 60-second toy run with the fake LLM of `tests/test_knockout_m1_m5.py`: ledger validates, report is
    written, replay passes, and all three regimes were picked at least once.

## 9. Build order (stop at the first step that does not fit in the time left)

| Step | Content | Check |
|---|---|---|
| P0 | `StartSet`, scores keyed by start set, `starts.py` for the toy, `pick_regime`, cold and record regimes only, `prompt.show_record` | tests 1–5, 8, 9; toy run with fake LLM |
| P1 | `neighbour` regime (perturbations), per-regime gene keys and operator arms, report per regime | test 10; toy run with fake LLM |
| P2 | `basins.py` (descriptors, archive, `basin_insert`), neighbour starts from the archive | test 6 |
| P3 | deep polish and `basin_promotion` | test 7 |
| P4 | `knockout/seeds/circles_rectangle.py` (from the toy seed: 21 circles, rectangle of perimeter 4, aspect ratio as a parameter; the final step must produce a candidate the frozen verifier accepts, which has **no** overlap tolerance and checks the perimeter in exact Fractions), `starts.py` for circles-rectangle, register it in `run.py` `SEEDS` | seed program scores a valid candidate from a cold start and from the certificate |
| P5 | `prompt.knowledge` switch | prompt test |

P0–P2 is the minimum worth demoing. If time is short, skip P5 and the per-regime gate verdicts (5.2), and say so
in the README.

## 10. Experiments to run after the build (each needs Arthur's go-ahead and a cap)

Suggested cap: 5 USD per run, LLM only, local executor (no Modal needed for circles).

| Run | Problem | Setting | Question |
|---|---|---|---|
| E1 | `toy-circles-26` | `--profile portfolio`, 10 min | Does any gene get validated in some regime? (v1: none in an hour on the main problem) |
| E2 | `circles-rectangle` | v1 (`portfolio.enabled: false`) vs `--profile portfolio`, same cap, warm start = frozen certificate | At equal spend: main arm vs control; number of distinct basins; any `basin_promotion` |
| E3 | `circles-rectangle` | `--profile portfolio` with `prompt.knowledge: off` vs `on` | Does the structure find good methods without external knowledge? |

## 11. Success levels

| Level | Meaning |
|---|---|
| L1 | The portfolio runs end to end on the toy and on circles-rectangle; ledger validates; replay passes |
| L2 | At least one gene validated in at least one regime (v1 never validated a gene on the main problem) |
| L3 | The main arm beats the control arm at equal spend on the incumbent score |
| L4 | A basin found from a `cold` or `neighbour` start, after deep polish, beats the incumbent (`new_incumbent_from_other_basin`) |
| L5 | A candidate at or above live #1 + δ, confirmed by `selection/verify_claim.py` (not expected on circles-rectangle: its record is a certified KKT point) |

Report the level reached in `knockout/README.md`, with the run ids and the numbers, labelled verified / hypothesis /
limitation. Do not state a record anywhere unless the promotion gate logged `record_claim`, and say that it still
has to pass `selection/verify_claim.py`.

## 12. Known risks

- **Cost per child.** `neighbour` and `cold` children cost K evaluations. With K = 2 the main arm evaluates fewer
  children per dollar than v1; the floors keep `record` alive, but compare arms at equal spend, not equal children.
- **Deep polish uses a record-regime program on a different basin.** That program may be tuned to the incumbent's
  structure. If promotions never succeed, try the best program of the regime that found the entry as well, and log
  both.
- **Descriptors decide what "another basin" means.** Poor descriptors merge basins or split one basin into many
  cells. Print the occupied cells at the end of E1 and adjust bins before E2, and say so.
- **Circles-rectangle verifier has no tolerance.** Seeds must shrink radii slightly (as the toy seed does) or
  every candidate is invalid; check this first in P4.
