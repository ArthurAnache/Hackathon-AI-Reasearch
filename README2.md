# Hackathon-AI-Research: autoresearch for mathematical records

[karpathy/autoresearch](https://github.com/karpathy/autoresearch), applied to open problems from Terence Tao's
[optimisation constants list](https://teorth.github.io/optimizationproblems/). An agent repeatedly changes a
search algorithm, runs it for a fixed 60 s with a fixed seed, has the result checked by a frozen verifier,
and keeps the change only if the verified score improved.

## Layout

| File | Role | Who edits |
|---|---|---|
| `problems/` | verifiers and records for 1a, 1b, 1d, 10c, 11a | WS1 (frozen) |
| `prepare.py` | fixed constants (budget, seed), frozen-file check, CLI (Karpathy's `prepare.py`) | frozen |
| `interfaces.py` | **the shared contract** + the one experiment protocol (`run_experiment`) | frozen |
| `optimizers/` | `<id>_baseline.py` and `<id>_tuned.py` per problem | WS3 |
| `search.py` | the file a coding agent edits (Karpathy's `train.py`), copied from `optimizers/` | the agent |
| `loop.py` | scripted research loop (calls only the contract) | WS2 |
| `research/` | **research feedback framework**: turns results into the next experiment (`FeedbackAgent`) | WS2 |
| `program.md` | instructions for the coding agent | humans |
| `analysis.py` | history, best-so-far, gap to the record (`--json` for the dashboard) | — |

The contract and how each workstream plugs in are described in [INTERFACES.md](INTERFACES.md).

## Two ways to run the loop (both use the same contract and the same log)

```bash
python -m venv .venv --system-site-packages && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                         # 90 tests -> reports/test_report.md

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
`python prepare.py verify <id> <solution_path> --exact`.
