# Hackathon-AI-Research: autoresearch for mathematical records

[karpathy/autoresearch](https://github.com/karpathy/autoresearch), applied to open problems from Terence Tao's
[optimisation constants list](https://teorth.github.io/optimizationproblems/). An agent repeatedly changes a
search algorithm, runs it for a fixed 60 s with a fixed seed, has the result checked by a frozen verifier,
and keeps the change only if the verified score improved.

## Architecture: 4 layers, one workstream each

```
L4 ORCHESTRATION (WS2)  evolution/orchestrator.py  meta-bandit splits the budget across arms
                        arms = [hill_sampling (control), gene_evolution, formulation, fresh_seeds]; islands + migration
L3 RESEARCH      (WS2)  evolution/  genes -> Thompson crossover · mutation · knock-out gate · graveyard
                        optional LLM structural edit (evolution/llm.py); evidence in (WS4: Exa, past.dev)
L2 EXECUTION     (WS3)  execution/  executor.submit(job) / .map(jobs) -> results
                        LocalExecutor | ModalExecutor (same interface); inner tuner: research/ (no LLM)
L1 TRUTH  (WS1, frozen) problems/ (+ WS1's EinsteinArena problems)  verify / score / verify_exact
                        certificates.py: best solutions, re-verified on load, the only allowed warm starts
SHARED                  experiment store: one schema (interfaces.new_record), append-only, all workers
```

**Survival of the fittest.** Every method (a gene: a config value, a representation change such as
resolution upscaling, or a whole optimizer) is first tested on its own. Winners are bred: children
inherit each parent gene with a probability drawn from that gene's track record (Thompson sampling),
then mutate (nudge a value, add or drop a gene, or ask an LLM for a structural edit). Only the fittest
genomes survive. A gene is *validated* only when a knock-out test (same parent certificate, same seeds,
with versus without the gene) shows it earns its place; otherwise it goes to the graveyard and is never
retried. Islands evolve independently and exchange only validated genes and best certificates.

**Explore, then refine (WS3's Modal lessons, 2026-11-14).** WS3's 88-chain campaign on 1a (1.51376 -> 1.50707)
showed that, with the same config, the seed alone moves the result from 1.508 to 1.521; that the median chain does
not improve; and that new bests come from fresh seeds, not from extending the best chain. So:
- a 4th arm, `fresh_seeds`, makes cold restarts from new seeds with the validated genes. Like the control,
  it always gets a slot;
- it is judged on its **best** seed (a new verified best is a real improvement), while methods
  (genes, configs) are still judged on their mean over seeds with a real margin;
- the bandit rewards an arm that produces a new verified best, so budget follows best-of-N progress;
- `analysis.py --arms` reports each arm's score distribution over seeds (median, best) and its new bests;
- genes shared through past.dev go to one island only, so islands keep their diversity;
- Modal jobs use `cpu=1` (a single-threaded NumPy chain uses 94-96% of one CPU).

Same run before and after (1a, start from WS3's pilot certificate 1.5137599, 2 islands x 4 rounds x 4
candidates x 3 seeds x 10 s, local): **1.513406 without `fresh_seeds`, 1.509403 with it** (gap to the record
0.0107 -> 0.0067). The new best came from a fresh seed (best seed 1.50940; that candidate's mean was
1.51216). This is one run of each, a strong hint rather than a proof. The record (1.502743) is not beaten.
WS3's further finding that 1a_tuned plateaus near 1.507 means the remaining gap needs new algorithms: LLM
operator genes (`--llm`, OpenAI or Claude, spend-capped) and formulation genes. More CPU on the same algorithm
will not close it.

**Fair comparisons.** At least 3 seeds per candidate, the same budget for every arm, and "keep" only when
the mean improvement beats `max(1e-9 relative, 2 x standard error)`. Any verified candidate that beats the
parent still becomes a certificate, so solutions advance even while methods are still being judged.

**Research tools (WS4's adapters, all optional, keys in the git-ignored `.env`).** They run in the parent
process once per round, never inside a job, so Modal containers need no secrets
([evolution/tools.py](evolution/tools.py)):

| Tool | Flag | What it changes in the loop |
|---|---|---|
| Exa | `--exa` | Record audit on round 1, then a literature search whenever every island has gone 2 rounds without a better certificate. Findings go into the LLM editor's prompt (as untrusted data) and into shared memory. |
| past.dev | `--pastdev` | Before a round: genes another agent *validated* come in with a favourable prior and are tested first (they must still pass this island's knock-out gate); genes another agent *buried* are buried here too; configs other agents kept become config genes. After a round: this run's verdicts and best scores are shared (never solutions). `--memory-scope` isolates a comparison. |
| LLM editor | `--llm openai` / `claude` / `auto` | Structural edits of the optimizer (operator genes), informed by the top genes, the graveyard and the Exa findings. `auto` uses `OPENAI_API_KEY` if set, else `ANTHROPIC_API_KEY`. Capped in dollars (see below). |
| Modal | `--executor modal` | Same jobs on Modal containers instead of local processes. |

Every record's `agent` field says which tools were on (e.g. `evolution@laptop-a[exa,pastdev;scope=ab-1]`),
and a candidate whose genome contains a tool-suggested gene cites it in `evidence`. So "tool off" and
"tool on, found nothing" are never confused, and each tool's contribution can be measured with
`analysis.py --arms` on the same problem, budget and seeds.

**LLM spend cap ([evolution/llm.py](evolution/llm.py)).** The LLM editor is the only part of the loop
that costs money; the verifier, the arms, the knock-out gate and the bandit are free CPU.
- Default model: `gpt-6.1-terra` ($1 input / $5 output per 1M tokens, OpenAI pricing page checked
  2026-11-14). `--llm-model gpt-6-luna` is ~10x cheaper. Prices live in `PRICES`; a model without a price is refused.
- Cost of each call = tokens billed in the API's `usage` x price. It is added to a running total in
  `llm_spend.json` (git-ignored, shared by every run on this machine, so the cap holds across restarts).
- Before each call, the worst case (prompt estimate + `--llm-max-tokens` of output) is checked against
  `--llm-budget-usd` (default 10, the TOTAL ever spent, not per run). If it could cross the cap, no call
  is made and the loop continues on the free arms.
- A call is made only for an island that has not improved for `--llm-stall-rounds` rounds (default 2),
  at most `--llm-max-edits` per round across islands (default 2), with a compact prompt (problem line,
  scores, current optimizer, top genes, graveyard one-liners, 3 literature snippets).
- Each record carries `llm_cost_usd` and `llm_usage` (model, tokens, cost, latency) for the call that
  produced the candidate. Calls whose edit was invalid are in `llm_spend.json` with their `error`.
- Limitation: the check-then-call is not atomic across processes; two runs in parallel could overshoot
  by at most one call's worst case (~$0.06 with the defaults). Also set a hard limit in the provider dashboard.

```bash
python -m evolution --problem 1a --islands 2 --rounds 4 --seeds 3 --budget 10 --workers 8      # local
python -m evolution --problem second-autocorrelation-inequality --islands 1 --rounds 3 --budget 10 \
    --llm openai --llm-budget-usd 0.5 --llm-stall-rounds 0 --llm-max-edits 1                   # ~$0.50 LLM smoke test
python -m evolution --problem 1a --exa --pastdev --agent-id laptop-a --memory-scope ab-1       # + tools
modal deploy execution/modal_backend.py && python -m evolution --problem 1a --executor modal   # Modal
python importers.py results/ws2-v1          # WS3 pilot results -> the one store (+ re-verified certificates)
python analysis.py --arms                   # arm comparison;  --csv out.csv for an export
```

## Second autocorrelation (EinsteinArena main problem)

`optimizers/second-autocorrelation-inequality_blockshift.py` is a target-specific optimizer. Its block moves use
exact incremental updates of f*f and choose the optimal step size. In a matched run (same live #1 incumbent,
paired seeds 101/202/303, 300 s each, one batch), it gained +1.53e-8 on average against +2.29e-9 for WS1's
probe. It won on every seed, and its best was +1.73e-8 above the live #1, confirmed by the exact verifier.
That is not a record (δ = 1e-5). Details: [results/second-autocorrelation-method-comparison.md](results/second-autocorrelation-method-comparison.md).
`compare.py` runs any such head-to-head: same incumbent, paired seeds, equal budgets, one batch.

## Layout

| File | Role | Who edits |
|---|---|---|
| `problems/` | verifiers and records for 1a, 1b, 1d, 10c, 11a; WS1's EinsteinArena problems in `problems/<slug>/` | WS1 (frozen) |
| `certificates.py` | certificate store: verified best solutions, the only warm starts | frozen |
| `execution/` | `Job`, `LocalExecutor`, `ModalExecutor` | WS3 |
| `evolution/` | genes, operators, arms, islands, meta-bandit, LLM editor, research tools (Exa, past.dev) | WS2 |
| `importers.py` | external results (e.g. Modal pilot) into the one store | WS2/WS3 |
| `prepare.py` | fixed constants (budget, seed), frozen-file check, CLI (Karpathy's `prepare.py`) | frozen |
| `interfaces.py` | **the shared contract** + the one experiment protocol (`run_experiment`) | frozen |
| `optimizers/` | `<id>_baseline.py` and `<id>_tuned.py` per problem | WS3 |
| `search.py` | the file a coding agent edits (Karpathy's `train.py`), copied from `optimizers/` | the agent |
| `loop.py` | scripted research loop (calls only the contract) | WS2 |
| `research/` | **research feedback framework**: turns results into the next experiment (`FeedbackAgent`) | WS2 |
| `program.md` | instructions for the coding agent | humans |
| `analysis.py` | history, best-so-far, gap to the record (`--json` for the dashboard) | — |
| `compare.py` | matched head-to-head of optimizers: same incumbent, paired seeds, equal budgets | WS2 |

The contract and how each workstream plugs in are described in [INTERFACES.md](INTERFACES.md).

## Two ways to run the loop (both use the same contract and the same log)

```bash
python -m venv .venv --system-site-packages && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                         # 149 tests (all workstreams) -> reports/test_report.md

# A. Karpathy mode: a coding agent edits search.py (algorithm changes)
#    In Claude Code: "Have a look at program.md and let's kick off a new experiment!"
cp optimizers/1d_tuned.py search.py && python search.py "my hypothesis"

# B. Scripted mode: FeedbackAgent tunes CONFIG and refines the best solution (no credentials needed)
python loop.py --problem 1d --optimizer optimizers/1d_tuned.py -n 20
python loop.py --problem 1d --optimizer optimizers/1d_tuned.py --insights   # what the results say so far

python analysis.py                                 # what happened, best so far, gap to record
```

## Research feedback framework (`research/`)

Every proposal is rebuilt from the experiment store, so the agent is stateless and can be resumed:

1. **Space**: the optimizer's numeric `CONFIG` knobs, with ranges from its optional `SEARCH_SPACE`, mapped to [0,1]^d.
2. **Knowledge**: what the results say. Every experiment is tagged `strategy=…; parent=…`, so each one is a
   controlled comparison against the best result it started from. From these tags it derives per-knob effects
   (direction and slope), each strategy's track record, a trust region (grows after a keep, shrinks after a miss),
   failure regions (crash, invalid, over budget), and the set of configs already tried.
3. **Strategies**: pluggable generators of candidate experiments.
   - `coordinate`: per-knob steps with adaptive size and direction.
   - `surrogate`: a quadratic model fitted to all results, maximised inside the trust region.
   - `combine`: applies moves that each helped, together.
   - `explore`: trust-region sampling.
   - `evidence`: config hints from Exa or past.dev.
   - `refine`: warm start from the best solution found so far (`CONFIG["INIT"]`).
4. **FeedbackAgent** (the contract's `ResearchAgent`): picks a strategy by its improvement rate so far (UCB),
   never repeats a config or goes near a failure, writes a hypothesis and a rationale that cite the experiments
   it relies on, and stops after `patience` experiments in a row without a keep.

To add a strategy, write a class with `name` and `propose(knowledge, rng) -> [Candidate]`, then pass it as
`FeedbackAgent(..., strategies=[...])`. To make an optimizer refinable, add `"INIT": None` to its `CONFIG`
(see `optimizers/1d_tuned.py`). With warm start on, results compound across experiments; use `--no-warm-start`
to compare algorithms from scratch.

## Problems and current results (60 s per experiment, laptop CPU)

| id | problem | best known | our best | |
|---|---|---|---|---|
| 1a | Sidon autocorrelation (min) | 1.5027434867 | 1.514054 | not beaten |
| 1b | Erdős minimum overlap (min) | 0.3808585503 | 0.381599 | not beaten |
| 1d | Signed autocorrelation (min) | 1.457854885 | 1.467369 | not beaten |
| 10c | Spencer discrepancy (max) | 5/√8 | 5/√8 | tie (proven optimal at n = 8) |
| 11a | L¹ Poincaré, Hamming cube (max) | √(π/2) | 0.960518 | not beaten |

A record counts only if it beats the best known value by more than 1e-9 (relative), and only after
`python prepare.py verify <id> <certificate_id> --exact`.
