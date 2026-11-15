# Knockout Evolution (WS2 research loop)

An autonomous research loop. The population does not evolve prompts. It evolves sets of small, typed **insight
genes**, and a gene earns influence only through a controlled knock-out experiment against the frozen verifier.
Every decision is a logged event with evidence behind it.

The full design brief is the team prompt "Knockout Evolution"; this README records what is built, how to run it,
and the results. Status labels: **verified** (reproduced here), **hypothesis**, **limitation**.

## Status

| Milestone | Content | State |
|---|---|---|
| M0 | config, sandbox, tuner, ledger, budget governor, executors (local + Modal), Hill Sampling control arm, push-to-run | **done** (2026-11-14) |
| M0.5 | LLM editor (OpenAI) on the control arm, seed program for the main problem, warm-start ratchet, first runs on `second-autocorrelation-inequality` | **done** (2026-11-14) |
| M1 | genes, prompt rendering, gene birth (diffs, seeds), Thompson genome construction, clade credit, phases | **done** (2026-11-14) |
| M2 | knock-out gate, negative genes, anti-cycling and novelty filters, re-audit | **done** (2026-11-14) |
| M3 | fidelity ladder, promotion gate, `report.py` | **done** (2026-11-14) |
| M4 | operator and model bandits, formulation islands, migration, lineage freezing | **done** (2026-11-14) |
| M5 | strategist, Exa scout, linkage, `replay.py`, Modal executor | **done** (2026-11-14); one 15-minute run completed on Modal |

M1 to M5 were written in one pass on 2026-11-14, at Arthur's request, without the per-milestone toy run the brief
asks for. What was checked instead is listed under "Results". Read the "Limitations" before trusting a mechanism.

## Layout

| File | Role |
|---|---|
| `strategy.yaml` | every default; the strategist may only patch `strategist_allow_list` keys (`config.Strategy.apply_patch`) |
| `problems_api.py` | read-only handle over the toy fixture and the frozen catalog (`problems.catalog.load`); never wraps the verifier |
| `toy.py` | test fixture: 26 circles in the unit square, max sum of radii, exact `Fraction` check |
| `seeds/` | seed candidate programs (`PARAMS` + `construct(params, seed, budget_s, init=None)`) |
| `sandbox.py`, `sandbox_entry.py` | run one untrusted program in a child process: timeout, rlimits, temp dir, empty env, no network |
| `tuner.py` | non-LLM tuner of `PARAMS`: CMA-ES with restarts, random-search fallback; runs inside the sandbox |
| `context.py` | the one evaluation path: sandboxed tuning, then re-scoring in the trusted process with the frozen verifier |
| `executor.py` | `LocalExecutor` (process pool) and `ModalExecutor` (Modal CPU workers) behind one interface |
| `budget.py` | budget governor: wall clock + one USD cap (Modal compute + LLM), fixed control share, gate/meta caps |
| `ledger.py` | `experiments.jsonl` + `events.jsonl`, validated with pydantic; `python -m knockout.ledger validate runs/<id>` |
| `editor.py` | `Editor` protocol and `OfflineParamEditor` (PARAMS mutations, no LLM) |
| `llm_editor.py` | `OpenAIEditor`: one model call = one complete child program + its hypothesis; validated before any evaluation, charged to the governor, logged as `llm_call` / `filter_reject` events |
| `control.py` | Hill Sampling control arm, with the optional warm-start ratchet |
| `decisions.py` | every decision of the loop as a pure function of logged inputs and a logged seed (bandits, genome construction, gate verdict, credit, promotion, freezing, migration, strategist trigger) |
| `genes.py` | `Gene`, `GeneStore` (status local to an island, dedupe), prompt rendering of a genome |
| `filters.py` | anti-cycling and novelty filters: normalized-source hash, token-set Jaccard, recycled deleted lines |
| `llm.py` | the one door to OpenAI: JSON answers, response cache by request hash, cost charged to the governor |
| `main_arm.py` | the main arm: islands, parents, Thompson genomes, knock-out gate, clade credit, fidelity ladder, gene birth from diffs, lineage freezing, migration, re-audit, linkage |
| `meta.py` | Exa scout (literature genes) and the restricted strategist (allow-listed, validated, reverted patches) |
| `report.py` | `python -m knockout.report runs/<id>` writes `report.md` from the ledger |
| `replay.py` | `python -m knockout.replay runs/<id>` re-derives every decision from the events and the LLM cache |
| `run.py` | CLI and `run()`; `modal_app.py` calls the same `run()` on Modal |
| `launch.yaml` | parameters of the run that a `[ko-run]` push launches |

## Run it

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/test_knockout_m0.py -q                 # 16 tests
.venv/bin/python -m knockout.run --problem toy --wall-clock 60          # local, writes runs/<run_id>/
.venv/bin/python -m knockout.ledger validate runs/<run_id>
```

### On the main problem, with the LLM editor

```bash
# OPENAI_API_KEY in .env (see .env.example). The run needs a spend cap.
.venv/bin/python -m knockout.run --problem second-autocorrelation-inequality --profile large-warm-start \
    --init live --editor openai --wall-clock 3600 --max-usd 10
.venv/bin/python selection/verify_claim.py second-autocorrelation-inequality runs/<run_id>/<best_solution_path>
```

- `--init live` starts from the live #1 of EinsteinArena (downloaded, cached under `selection/.cache/live/`, and
  re-scored here with the pinned verifier before the run starts). Without `--init`, the warm start is the frozen
  certificate, which was no longer the #1 on the evening of 2026-11-14 (0.963638668 vs live 0.963661562).
- `--profile large-warm-start` sets: 60 s per evaluation, one `construct` call per evaluation (`tuner.calls=1`),
  6 edits per round, 7 workers, `control.ratchet=true`, `selection.eps_rel=1e-11`, and
  `ledger.save_solutions=selected` (a candidate is 10–40 MB of JSON). `--set` still overrides any of them.
- `--editor offline` runs the same loop without any LLM call (PARAMS mutations only).
- The model and its price estimates are in `strategy.yaml` (`llm.*`).

How one round works with the ratchet:
1. The editor gets the incumbent program, the problem brief (frozen verifier source + `docs/ws1_knowledge.jsonl`,
   marked as claims), the measured distance to the record threshold, and the history of edits with their gains.
   It returns up to N programs, one per operator (`refine`, `restructure`, `reformulate`, `fresh`).
2. The N programs and a replay of the incumbent program all start from the same warm start, in the sandbox.
3. Every returned candidate is re-scored by the frozen verifier in the trusted process.
4. The best edit replaces the incumbent program if it beats the replay by more than `eps_rel`. The best candidate
   of the round (replay included) becomes the warm start of the next round (`warm_start_update` event).

A score from a run is not a claim. Only `selection/verify_claim.py` decides, and nothing here submits.

### Both arms

With `--editor openai` the run has two arms (set `run.arms: control` to keep only the first):

- **control** (`control.py`): Hill Sampling, no genes. It gets a fixed 20 % of the spend (`budget_shares.control`),
  whatever the bandits do, and never shares programs, genes or its warm start with the main arm.
- **main** (`main_arm.py`): one batch per step, on one island:
  1. island pick (Thompson bandit; success = a child beat the island's reference);
  2. for each child: operator and model (Thompson bandits), parent A and parent B (rank power law), genome
     (Thompson draw per validated gene from the statistics of the island's current phase, at most `genes.g_max`,
     linked genes together, incompatible genes never together), plus up to 5 refuted genes shown as "do not retry";
  3. up to `gate.pairs_per_batch` knock-out pairs for the gene under test: two children with the same parent,
     operator, model and base genome, one prompt with the gene and one without;
  4. the LLM writes the programs; the filters drop duplicates, near-duplicates and children that bring back lines
     their lineage deleted;
  5. F0 evaluation of the children and of every parent that has no score from the current warm start, then the
     fidelity ladder (`fidelity.levels`);
  6. credit: `s = 1` if the child beats its parent at F0 from the same warm start. The genes of its genome get
     `a <- gamma a + s`, `b <- gamma b + (1 - s)`, and the genomes of its ancestors at depth 1..3 the same with
     weight `0.5 ** depth`. Posterior of a gene: Beta(1 + a, 1 + b), per phase;
  7. gate verdict after each pair from the third on: P(theta > 0.5) under Beta(1 + wins, 1 + losses) at least 0.90
     validates, at most 0.10 refutes (the gene becomes a negative gene), 8 pairs without verdict retire it;
  8. gene birth: one reviewer call per improving child (parent, diff, both scores) returns 0 to 2 candidate genes;
  9. lineage freezing, migration of validated genes to other islands (as candidates), re-audit of the most used
     validated gene every `gate.reaudit_every` evaluations.
- **Strategist** (`meta.py`): when the main arm's best has not moved by more than `strategist.min_progress` during
  `strategist.stagnation_fraction` of the budget, one LLM call proposes a patch. Only `strategist_allow_list` keys
  pass, each value is range-checked, and the patch is reverted if progress over the next window is lower.
- **Promotion gate** (`context.py`): a score at or above record + margin goes through `problem.verify` and the
  independent verifier. Both arms use it. Only then is `decision = "record_claim"` logged, with the candidate
  stored. A float score that the independent verifier does not confirm is logged as `numerical_artifact`, is
  never used, and becomes a negative gene.

```bash
.venv/bin/python -m pytest tests/test_knockout_m0.py tests/test_knockout_m1_m5.py -q     # 37 tests
.venv/bin/python -m knockout.report runs/<run_id>      # report.md (also written at the end of every run)
.venv/bin/python -m knockout.replay runs/<run_id>      # exits 1 on the first decision that cannot be re-derived
```

### On Modal, from a push

1. Repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (Settings → Secrets and variables → Actions) for
   the Modal workspace that pays, and `OPENAI_API_KEY` when `launch.yaml` has `editor: openai`.
2. Edit `knockout/launch.yaml` (problem, profile, warm start, editor, wall clock, `max_usd` in (0, 500]).
3. Commit with `[ko-run]` in the message and push. `.github/workflows/knockout-run.yml` runs the tests, then
   `modal run --detach -m knockout.modal_app::launch --source-commit <sha>`, and exits. Commits without the tag
   launch nothing. A manual "Run workflow" on GitHub does the same.
4. Results: `modal volume get knockout-runs <run_id> ./runs/` (the run id is printed in the GitHub job log).

The same launch works from a laptop with a configured `~/.modal.toml`:
`.venv/bin/modal run --detach -m knockout.modal_app::launch --source-commit $(git rev-parse --short HEAD)`.

How the Modal run is split:
- **orchestrator** (`orchestrate`, 4 CPU, 16 GiB, network allowed for the leaderboard and LLM calls, Volume
  mounted, the only function that receives `OPENAI_API_KEY`): runs the loop, re-scores every candidate in float64
  with the frozen verifier, writes the ledger. It never runs an LLM-written program;
- **workers** (`evaluate_job`, `modal.cpu` cores, `block_network=True`, no secret, `max_containers` cap,
  explicit timeout): run one untrusted program each, inside a child process with rlimits.

## Ledger format

`experiments.jsonl` uses the union of the formats the repo asks for, so every consumer reads it:
the CLAUDE.md shared fields, WS2's `interfaces.py` `RECORD_FIELDS` (its `JsonlStore` refuses lines without them),
and the CLAUDE.md WS3 rule 9 fields (`run_id`, `source_commit`, `device`, `gpu_type`, `dtype`, `torch_version`,
`cuda_version`, `gpu_objective`, `wall_s`). Extra data sits in `config`. `gpu_objective` holds the sandbox's guide
value (CPU or GPU); it is never used for selection. Solutions are stored as `solutions/<chain>/<experiment_id>.json`,
global id `<run_id>/<chain>/<experiment_id>`.

`events.jsonl` types: the prompt's list plus `run_start`, `run_end` and `control_round` (needed to replay the control
arm), `llm_call` (role, model, tokens, cost, latency, error, request hash), `warm_start_update`, `island_opened`,
`island_replaced` and `linkage`. `operator_pick` has a `kind`: `operator`, `model`, `parent` or `parent_b`.

A decision event stores `inputs`, `outputs` and its `seed`: the arguments and the result of one function of
`decisions.py`. `replay.py` calls that function again on the recorded inputs. It also checks that every LLM answer
is in `llm_cache/`, that every evaluated program is the one in the cached answer, that gate counters match the
logged pairs, and that every patched key is allow-listed. It does not re-run programs.

With the ratchet, each experiment's `config.ref_score` is the score its edit had to beat (the replay of the incumbent
program from the same warm start), and `config.edit_meta` holds the model's rationale and token usage.

## Results

**M0 check, toy problem, local executor, 2026-11-14** (verified: scores are exact `Fraction` checks, re-run on the
saved solution):
- 60 s wall clock, 8 workers, F0 = 2 s, 8 edits per round: baseline 2.62488 → 2.63592 after 22 rounds
  (177 experiments, 3 keeps, 0 crash). Both JSONL files validate.
- The first toy reference (2.635, AlphaEvolve 2025) was below what the loop reached in one minute, so it was raised
  to 2.63598 (values reported after AlphaEvolve; hypothesis, not re-verified). The toy is a fixture, not a target.

**Main problem, `second-autocorrelation-inequality`, local executor, 2026-11-14.** Warm start: the live #1
(solution 2783, MLAI_YONSEI, 0.9636615620620852, re-scored here with the pinned verifier). Record threshold:
0.9636715620620852 (live #1 + δ, δ = 1e-5). Profile `large-warm-start`. All three runs used the control arm only.

| Run | Editor | Wall clock | Rounds | Best score (pinned verifier) | Gain over live #1 | LLM cost (estimate) |
|---|---|---|---|---|---|---|
| `20261115T001337Z-…` | offline (PARAMS mutations) | 5 min | 5 | 0.9636616146696421 | 5.26e-8 | 0 |
| `20261115T001854Z-…` | `gpt-6.1-sol` | 7 min | 4 | 0.9636616161700385 | 5.41e-8 | 0.39 USD |
| `20261115T002756Z-…` | `gpt-6.1-sol` | 60 min | 28 | 0.9636616159351276 | 5.39e-8 | 6.44 USD |

- **No record.** `selection/verify_claim.py` on the best candidates of the two LLM runs: `VALID_BELOW_RECORD`,
  pinned and independent verifiers agree within 5.6e-16 (verified). The gain is 0.54 % of δ.
- The gain comes from the first evaluation (~5.1e-8, the seed program's L-BFGS from the warm start). Later rounds
  add 1e-9 down to 1e-12 each: the search saturates near +5.4e-8 whatever the editor (verified on these runs).
- 60-minute run: 168 LLM calls, 0 failed, 0 program rejected, 0 crash; 5 of 152 edits were kept. The kept edits
  retune the smoothing sharpness (1e9 → 1e11 → 3e10 → 5e10) or the staging. No structural edit (packet weights,
  cutting planes, block reparametrizations) gained more than the incumbent (verified in the ledger).
- Reading (hypothesis): the live #1 is a local optimum of this family of smoothed-minimax polishers on a fixed
  support, as the forum reported for coordinate polishing at 2M. A record needs a change of structure, which the
  control arm did not find in one hour.
- The frozen certificate (solution 2782, 0.963638668) is no longer the live #1; re-freezing is WS1's call.

**Both arms, toy problem, `gpt-6.1-sol`, local executor, 2026-11-14** (`20261115T014358Z-toy-circles-26-…`,
5 min, 0.71 USD estimated):
- control arm 2.62489 → 2.63071 (2 rounds); main arm 2.62489 → 2.63598308 (2 batches). Two batches are far too
  few to compare the arms (limitation).
- 17 genes born (10 from the brainstorm call, 5 from the Exa scout with their URLs, 2 from the strategist); one
  gate got 2 pairs (both losses, no verdict yet); `reformulate` opened 2 islands; the strategist patched 4
  allow-listed keys; `replay.py` re-derived 73 decisions of 11 kinds and matched 18 evaluated programs to the LLM
  cache (verified).
- The promotion gate logged one `record_claim` on the toy: 2.63598308 against the toy reference 2.63598 + 1e-6,
  confirmed by the exact `Fraction` check. **This is not a record**: the toy reference is an unverified value kept
  only so that the gate has something to test, and the toy is a fixture.

**Both arms on Modal, main problem, `gpt-6.1-sol`, 2026-11-14** (`20261115T021632Z-…-26d8f15`, commit `26d8f15`,
`launch.yaml` as committed: 15 min, 5 USD cap, 16 children per batch, up to 32 CPU workers):
- The run ended on its budget after 849 s, with no crash: 56 experiments (38 main, 18 control), 52 LLM calls, none
  failed, about 2.0 USD by the governor's estimate (LLM + Modal compute; the real bill is on the dashboards).
- Workers ran every LLM-written program with the network denied (`unshare -rn` inside a `block_network` container).
- Best verified score: 0.9636616139976134 (main arm), a gain of 5.19e-8 over the live #1. `verify_claim.py`:
  `VALID_BELOW_RECORD`. **No record.**
- At equal spend the control arm was ahead of the main arm by about 2e-9 (0.6 USD each); the main arm finished
  ahead by 3e-10 only after spending 2.3 times more. These differences are at the level of timing noise: this run
  does not show that either arm is better.
- Genes: 13 born, one refuted by its gate (0 wins, 3 losses), one still open (2 wins, 1 loss), none validated;
  2 islands opened; 1 strategist patch, later kept or reverted by rule. `replay.py` re-derived 146 decisions and
  matched 43 programs to the LLM cache.

**Tests** (verified, 2026-11-14): `tests/test_knockout_m0.py` 21 passed, `tests/test_knockout_m1_m5.py` 16 passed
(gate thresholds, credit, genome construction, filters, strategist allow-list, promotion gate, and a recorded
55-second toy run of both arms with a fake LLM: ledger schema, report, replay, 4 kinds of tampering detected),
`tests/test_ws1_*.py` 17 passed, 1 skipped.

## Limitations

- **macOS:** `sandbox-exec` denies network and writes outside the job dir, but `RLIMIT_AS` is not enforced, so the
  memory limit is not guaranteed locally. On Modal (Linux) the memory rlimit applies and the worker function has
  `block_network=True`.
- The untrusted program shares its process with the guide objective, so it could fake the guide. This only misleads
  the tuner: every returned candidate is re-scored by the frozen verifier in the trusted process.
- Modal costs in the governor are estimates from `strategy.yaml` prices (check https://modal.com/pricing); the
  Modal dashboard has the real bill.
- `OfflineParamEditor` only rewrites `PARAMS`. Real edits come from the LLM editor.
- **No run has shown that the main arm beats the control arm.** The 60-minute run on the main problem used the
  control arm only (it predates the main arm).
- The three main-problem runs listed under "Results" predate the LLM cache and the logged decision inputs, so
  `replay.py` rejects them ("no cached response"). Runs made from this commit on can be replayed.
- **Replay is per decision, not a re-simulation.** Each event is re-derived from its own recorded inputs. A wrong
  input that was logged consistently would pass. Evaluations are not re-run (scores come from the ledger).
- **Deviations from the brief**, chosen here and open to review:
  - a validated gene with fewer than `genes.phase_min_obs` observations enters every genome pool of its island
    (the brief takes genes only from the two parents, so a freshly validated gene could never spread);
  - a `fresh` child has no parent in its prompt but is still compared with a drawn parent for credit;
  - with the ratchet, "island improved" means a child beat the best replay of the batch, and a parent is replayed
    whenever the warm start has moved (one extra evaluation per distinct parent and batch);
  - re-audit and migration counters are per island; lineage freezing skips its F2 tuning when
    `fidelity.levels < 3`;
  - bandit arms never decay below Beta(1, 1);
  - `strategist.min_progress` (not in the brief) defines "no improvement" when every batch gains a little;
  - `linkage`, `island_opened`, `island_replaced` are extra event types.
- **Gate on noisy problems.** With `selection.eps_rel` below the timing noise, a pair "win" can be noise; three
  straight noise wins have probability 1/8. Raise `eps_rel` or `gate.min_pairs` before trusting a validated gene
  on the main problem.
- **Modal:** no run has completed on Modal yet. First attempt (2026-11-14 18:03 PT, Arthur's workspace): the
  orchestrator downloaded and scored the live #1, then failed after 7.8 s with `FileNotFoundError:
  /root/problems/second-autocorrelation-inequality/meta.json`. Cause: the image put `problems/`, `integrations/`
  and `docs/` under `/root/repo`, while Modal mounted the `knockout` package at `/root/knockout`. Fix: the image
  now lays everything out under `/root` (verified by the second attempt). Second attempt (18:09 PT): the worker
  ran the seed program, then could not return its result: above 2 MiB Modal sends a return value through external
  blob storage, and the worker has no network (`block_network=True`), hence a DNS error. Fix: the worker yields its
  result as compressed chunks of at most 1 MiB (`executor.pack_result`), which stay on Modal's own channel; the
  network stays blocked. `launch.yaml` stays a 15-minute, 5 USD smoke run until one run ends cleanly.
- A record is only claimed against a live leaderboard read. If the read fails, the frozen catalog serves its
  snapshot (stale since 2026-11-14), the run prints a warning, and `beats_record` stays false.
- The scout needs `EXA_API_KEY`; it is not passed to Modal, so the scout is skipped there.
- Only `second-autocorrelation-inequality` has a seed program. `circles-rectangle` and `ring-loading-15` still
  refuse to start.
- With the ratchet, program fitness is relative to a moving warm start. A program is compared only with the replay
  of the incumbent in the same round; scores of different rounds are not comparable as program fitness.
- Programs stop on wall-clock deadlines, so two runs of the same program differ slightly (measured: ~1e-10 on this
  problem). `selection.eps_rel=1e-11` is below that noise, so some "keep" decisions on programs are noise. The
  candidate scores themselves are exact (pinned verifier).
- LLM costs in the governor are estimates: token usage times `llm.usd_per_mtok_*` (gpt-6.1-sol: 2 / 10 USD per
  million tokens, from third-party price lists, not checked on OpenAI's pricing page).
- The LLM editor sends the problem brief, the incumbent program, hypotheses and error messages of earlier
  programs to OpenAI. It sends no candidate data and no secret. On macOS the sandbox now also denies reads of
  `.env` and of the user's credential directories; on Linux that file route is closed only by not shipping secrets
  in the image (CLAUDE.md WS3 rule 8).
- The Modal path is tested with a fake `.remote` only; it has not run on Modal yet (no token on this machine).

## Design references

Cited from the team brief, **not yet verified** by us: arXiv 2602.16805 (simple baselines vs code evolution),
2609.25510 (Hill Sampling), 2606.10402 (reformulations behind EinsteinArena records), 2605.20086 (what evolutionary
coding agents evolve), 2510.21614 (Huxley-Gödel Machine), 2605.09018 (evolutionary ensemble of agents), 2611.14326
(falsification-gated self-evolution), 2604.18005 (communication density and convergence), 2602.23413 (EvoX).
