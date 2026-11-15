# INTERFACES.md: the shared technical contract

Everything is implemented in [interfaces.py](interfaces.py), which is hash-locked. The loop
([loop.py](loop.py)) and the Karpathy-mode entry point (`prepare.run`) call only these functions.

| Contract | Implementation | Owner | Plug in by |
|---|---|---|---|
| `problem.verify(candidate) -> bool` | `Problem.verify` | WS1 | adding `problems/<id>.py` (see below) |
| `problem.score(candidate) -> float` | `Problem.score` (raises `InvalidSolution`) | WS1 | same |
| `optimizer.run(problem, budget_s, seed, config) -> candidate` | `FileOptimizer` (any file in `optimizers/`, or `search.py`) or any class with `run` + `version` | WS3 | adding `optimizers/<id>_<method>.py`, or a class that dispatches to Modal |
| `research_agent.propose(history, evidence) -> experiment_config` | `research.FeedbackAgent` (offline, uses all results so far); a coding agent via program.md | WS2 | a class with `name`, `propose`, `review` (e.g. an LLM agent), or a new strategy in `research/strategies.py` |
| `research_agent.review(experiment_result) -> next_action` | `FeedbackAgent.review` → `"continue"` or `"stop"` | WS2 | same |
| `experiment_store.append(result) -> None` | `JsonlStore` (`experiments.jsonl`, never committed) | WS2 / WS4 | a class with `append` + `read` (e.g. one that also mirrors to past.dev or a dashboard) |
| `evidence_source.gather(problem, history) -> [Evidence]` | `NoEvidence` (default) | WS4 | a class wrapping Exa / past.dev |

The keep/discard rule is **not** part of the agent. `run_experiment()` enforces it: a run is kept only if it is
valid, within budget, the frozen files are intact, and it beats the best kept score for the same problem
and budget by more than `MIN_IMPROVEMENT_REL`.

## Problem module (`problems/<id>.py`, WS1)
```python
ID, NAME, URL, DIRECTION ("minimize" | "maximize"), BEST_KNOWN, BEST_KNOWN_SOURCE, SOLUTION_FORMAT
verify(solution) -> float              # the score; raises InvalidSolution
verify_exact(solution) -> Fraction     # optional exact certificate
```
Register it in `problems/__init__.py`, add `optimizers/<id>_baseline.py` and known-value tests,
then a human runs `python prepare.py lock`.

## Optimizer file (`optimizers/*.py`, `search.py`, WS3)
```python
PROBLEM = "<id>"
CONFIG = {"N_CELLS": 800, "LR": 0.01}            # every tunable knob
def search(budget_s, seed, config=CONFIG) -> candidate
if __name__ == "__main__":
    run(__file__)                                  # from prepare import run
```
Only the standard library and `numpy` are allowed. `optimizer_hash` is the sha256 of the file (12 hex).
Optional: `SEARCH_SPACE = {"KNOB": (low, high, "log" | "linear")}` (ranges for the research agent) and
`"INIT": None` in CONFIG (warm start: the agent passes the path of the best solution so far).

Evidence with a structured hint is acted on directly: put `{"config": {"KNOB": value}}` in `Evidence.content`.

## ExperimentConfig (what `propose` returns)
`hypothesis: str`, `config: dict` (overrides of CONFIG), `optimizer: str | None` (path of another optimizer file), `rationale: str`.

## Evidence (WS4)
`source, content, citation, query, author, retrieved_at, latency_s`. Every experiment stores the evidence it was given.

## Experiment record (one line of `experiments.jsonl`, the ONE shared schema)
Built with `interfaces.new_record(...)` by every writer (Karpathy mode, loop.py, every evolution island and
worker, the importer). Append-only. CSV is an export (`python analysis.py --csv out.csv`), never a source.

| group | fields |
|---|---|
| identity / provenance | `experiment_id, timestamp, problem, agent, arm, island, worker_id, optimizer, optimizer_hash, source_sha, parent_id, genes[]` |
| what and why | `hypothesis, rationale, config, warm_start, evidence[]` |
| protocol | `seeds[], budget_s` |
| outcome | `scores[]` (verified, per seed), `score` (mean), `valid, status (keep/discard/invalid/crash/void), best_before, beats_record, duration_s, error, certificate_path` |
| LLM spend | `llm_cost_usd` (USD of the LLM call that produced this candidate, 0.0 otherwise), `llm_usage` (provider, model, input / cached / output tokens, cost, latency; null otherwise) |

## Certificates (L1, `certificates.py`)
`CertificateStore.put(problem, candidate, score) -> Certificate` (verifies first), `.load(problem, id)` (re-verifies,
refuses edited files), `.best(problem)`. Ids are `<problem>/<sha12>`. **A warm start (`config["INIT"]`) must be a
certificate id**; anything else is rejected.

## Execution (L2, `execution/`, WS3)
```python
job = make_job(problem, optimizer_path, config, seed, budget_s, warm_start=cert_id)   # self-contained, hash-checked
executor.submit(job) -> JobResult        executor.map([jobs]) -> [JobResult]
LocalExecutor(workers=8) | ModalExecutor()          # same interface; `modal deploy execution/modal_backend.py`
```
`JobResult = {job_id, status (ok|crash|timeout), candidate, duration_s, worker_id, error, meta}`. Workers never score:
the parent re-verifies every candidate.

## Evolution (L3 + L4, `evolution/`, WS2)
Genes `{kind: config|formulation|operator, name, params, origin}` live in a pool with a Beta belief and a state
(candidate, validated after a knock-out pass, or graveyard). Arms: `hill_sampling` (control), `gene_evolution`,
`formulation`. The meta-bandit splits candidate slots, islands migrate only validated genes and best certificates
through the stores. `python -m evolution --problem 1a --executor local|modal`.

## Demo / dashboard
`python analysis.py --json`: for each problem, counts, baseline, best, best_so_far, gap_to_record,
beats_record and ties_record, and the full history.
