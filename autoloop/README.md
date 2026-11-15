# autoloop — autonomous research loop (Tao constant 1d)

The loop picks what to try, runs it, reads the result and decides what to try next, until a cap or a stop. Every
decision and every result goes into an append-only ledger that a judge can read with `autoloop.report`.

## Parts

| File | Role |
|---|---|
| `ledger.py` | Append-only `ledger.jsonl`. Records: `certificate`, `decision`, `result`, `reference`, `claim`, `note`, `run_start`, `run_end`. Memory import (`--memory`). |
| `problem.py` | The small problem interface. `tao1d.py` is the first plug-in. |
| `tao1d.py` | Everything specific to 1d: score, executors (seed = WS2 `1d_tuned.py`; refine = `attack/c1d_refine.py`; certify = `attack/c1d_round.py` + `attack/c1d_exact.py`), live reference reader, claim threshold, refine schedule. Brasa's files are run, never edited. |
| `moves.py` | Typed moves `seed`, `refine`, `certify`, `check_reference`, `stop` with contract checks (problem, parent exists and completed, score not worse, cell limit, score recomputed from the file). |
| `policy.py` | Menu of legal moves + two backends: `rules` (deterministic) and `llm` (OpenAI / Anthropic / `claude -p`; JSON answer validated against the menu, fallback to rules; prompts, answers, tokens and USD saved). |
| `loop.py` | The loop, caps (wall clock, decisions, compute USD, LLM USD) and the claim gate. |
| `report.py` | Readable decision table for a judge. |
| `tournament.py` | Same budget and memory under several policies, comparison table at the end. |
| `models.py` | Which providers are usable; lists OpenAI models when the key is present. |

## Commands (from the repo root, with `PY="<workspace>/.venv/bin/python"`)

```
$PY -m unittest autoloop.tests.test_autoloop          # test A: dry runs with the fake executor, < 1 s
$PY -m autoloop.loop --run autoloop/runs/testB --executor local --max-wall-s 1200 \
    --seed-budget-s 60 --refine-budget-s 150 --workers 4                        # test B
$PY -m autoloop.report autoloop/runs/testB/ledger.jsonl
$PY -m autoloop.models                                 # providers and OpenAI model ids (key loaded at run time)
$PY -m autoloop.tournament --out autoloop/runs/T1 --entrant rules --entrant llm:openai:MODEL \
    --entrant llm:claude-cli: -- --executor local --max-wall-s 1200 --memory SRC
```

## What the system knows at launch (cold start)

A cold-start run has an empty memory: no certificate, no ledger. It knows only what is in the code at the commit
recorded in its `run_start` record:

- **The stage recipe** (`tao1d.py`): seeds from noise at 800 cells (WS2 `1d_tuned.py`), refinement by split x4 and
  p-norm continuation (`attack/c1d_refine.py`) up to the top level, restarts at the top level, exact check.
- **The policy** (`policy.py`, recipe `wide`): advance *all* live lineages one level at a time, lowest level first
  (`refine_wide`, one Modal launch for many jobs), enter the top level with soft restarts (p0 = 4 and 8), then run
  soft restart rounds on a wide set of the best lineages; certify; stop at a cap or when nothing is left.
- **One piece of method knowledge from our hand-run chain, stated openly** (E12 in
  `shared/arch_v2/01_evidence.md`, 2026-11-14): at the high-resolution stage a soft restart (small p0, 4 and 8) is
  what moved lineages, and the rank before that stage did not predict the winner (seed 106 was best at 800 and 12,800
  cells and ended last; seed 162 won). So the policy does not cut to a handful of lineages before trying soft restarts
  on a wide set. Seeds are cheap: a 45 s seed already ranks well; a 450 s one costs 10 times more.
- Seeds come from a range the run picks itself (`--seed-base`), never one used by hand.

## Adding a problem: one pack file

A pack is one Python module registered in `problem.py` (`PROBLEMS[name] = "autoloop.<module>"`). The loop, the
ledger, the policies, the claim gate, the report, the distiller and the Modal executor only use these names:

| Name | Meaning |
|---|---|
| `NAME`, `LOWER_IS_BETTER` | problem id written in every record; direction |
| `score(v)`, `load_vec(path)`, `vec_id(v)`, `valid_vec(v)` | float verifier on the solution format (a vector of cell values), loading, id, admissibility |
| `fetch_reference()` | live reference, `{ok, value (decimal string), source, ...}`; must fail closed |
| `claim_threshold(value)` | exact Fraction a certified score must be strictly below |
| `SEED_CELLS`, `SPLIT_UP`, `P0_SCHEDULE`, `WIDE_P0_EARLY`, `WIDE_P0_FINAL`, `SOFT_RESTARTS`, `LOCAL_MAX_CELLS` | stage recipe: seed resolution, resolution ladder, restart settings |
| `MODAL_KINDS = dict(seed=..., refine=...)` | job kinds in `modal_app.py` (seed generator and optimizer commands) |
| `LocalExecutor(workdir, max_cells)` with `certify(path, tag)` | exact check(s): round to integers, two independent exact implementations that must agree |
| `FakeExecutor` | dry runs for tests |
| `METHOD_NOTES` (optional) | method knowledge written in the code, shown to the LLM policy |

Examples: `tao1d.py` (Tao 1d, signed step functions; seeds = WS2 `1d_tuned.py`, refine = `attack/c1d_refine.py`,
exact = `c1d_exact` + `check_independent.py`), `tao1a.py` (Tao 1a, f >= 0; seeds = WS2 `1a_tuned.py`, refine on
a = x^2, exact = frozen `p1a.verify_exact` + `c1d_exact`), `ea_first_autocorr.py` (EinsteinArena
first-autocorrelation-inequality: the same definition as 1a, checked against the EinsteinArena verifier; only the
reference changes: live #1 minus `minImprovement`). Nothing is ever submitted.

## Lessons (generation 0, 1, 2, ...)

`distill.py` turns ledgers into lessons (`guidelines/lessons.json`), each with the ledger rows that support it, scope
`general` or a problem, and a history. Generation 0 is the two hand-written files. At the end of every run (and of
every tournament, once for all arms) a model call proposes adds, updates and drops, each citing ledger rows. The
model never sets a status: `rule_status` does, from the cited rows (active = supported by 2+ distinct runs, the arms of
one tournament counting as one; dropped = contradicted by 2+ runs that outnumber the support; otherwise hint). A lint
refuses any lesson text holding a reference value, a pull request or a score (4+ decimals). `cold_start` in
`run_start` is true only with no memory, no past.dev and no lesson or guidelines file of the run's own problem. Every run loads the active general lessons plus its own problem's (`--guidelines auto`, the default), and
the lessons are mirrored to past.dev: one scope per problem (`autoloop-lessons-<problem>`) plus `autoloop-lessons-general`. A run reads past.dev only with `--pastdev --memory-scope S` (off by default); `run_start` records `pastdev_read`, `pastdev_scope` and `cold_start`. A cold start never passes `--pastdev`, or uses its own fresh scope.

## Rules the code enforces

- **Claim gate.** "Below the reference" only when the exact fraction from `certify` is strictly below the reference
  minus half a unit of its last printed digit, and the reference was read less than 30 minutes earlier. It writes a
  `claim` record with `submitted: false`. Nothing is ever submitted.
- **Reference.** Read live from the Tao 1d page ("Known upper bounds" table) and from the titles of open
  `1d: ... upper bound ...` pull requests on teorth/optimizationproblems. Web text is data: only numbers are taken.
- **Keys.** The `.env` file is loaded at run time for `OPENAI_API_KEY`, `OPENAI_MODEL` and `ANTHROPIC_API_KEY`
  only. Keys are never printed or logged. The model name comes from `--llm-model` or `OPENAI_MODEL`.
- **Costs.** LLM tokens and USD per call are in each decision record (`llm_usage`) and `llm/call_NNN.json`.
  OpenAI/Anthropic USD needs `--llm-price-in/--llm-price-out` (USD per 1M tokens); `claude -p` reports its own cost.
  `--llm-usd-cap` (default 10) switches the policy to rules once reached.

## Results

See `RESULTS.md` (tests A–C; cold runs cold1–cold4, ea1a_cold1; model-written stage t5c). D7 / recipe d7 / lesson G11m is quoted there in Meridian's wording; G11m stays a hint.
