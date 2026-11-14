# INTERFACES.md: contracts between the parts

| Part | Who edits | Contract |
|---|---|---|
| `problem/` | humans only (frozen, hash-checked) | `verify(solution) -> float`, raises `InvalidSolution`; constants `DIRECTION`, `N_POINTS`, `KNOWN_BEST` |
| `strategy/` | the agent | `search(config: dict, seed: int, budget_s: float) -> solution`; settings in `config.yaml` |
| `runner/` | humans only (frozen, hash-checked) | the loop, CLI, agent prompt, experiment protocol (`runner/config.yaml`) |
| `program.md` | humans | research instructions the agent reads every step |
| `experiments.jsonl` | runner only (append-only) | one row per experiment, schema below |

## Solution format (dummy problem)
```json
{"points": [[0.1, 0.2], [0.5, 0.9], ...]}   // exactly N_POINTS pairs inside [0,1]^2
```

## Agent proposal
```json
{"hypothesis": "...", "rationale_from_history": "...",
 "files": {"strategy/search.py": "<full content>", "strategy/config.yaml": "<full content>"}}
```
Any path outside `editable_paths` (runner/config.yaml) makes the experiment fail with a `ScopeViolation`.

## experiments.jsonl row
| field | meaning |
|---|---|
| `id` | `exp-0000`, `exp-0001`, ... |
| `timestamp` | UTC, ISO 8601 |
| `agent` | `none` (baseline), `heuristic`, or `claude:<model>` |
| `hypothesis` | one falsifiable sentence |
| `rationale_from_history` | which past results motivated it |
| `idea_key` | heuristic agent's dedup key (null for Claude) |
| `edited_files`, `diff` | what changed in `strategy/` |
| `seeds`, `budget_s` | the fixed protocol used |
| `scores` | verified score per seed |
| `score` | aggregate (mean) of verified scores, null on failure |
| `best_before` | best kept score before this experiment |
| `decision` | `baseline`, `keep`, or `reject` |
| `failure_reason` | why it was rejected (crash, timeout, invalid, no improvement), else null |
| `duration_s` | wall-clock time of the evaluation |
| `frozen_hash` | hash of problem/ + runner/ at run time |
| `commit` | git commit holding the strategy that was scored (keep) or HEAD (reject) |

## Swapping in a real problem
Replace `problem/problem.py` keeping the same `verify` contract, update
`strategy/search.py` to emit that solution format, update `program.md`, then run
`python -m runner.cli freeze` and commit `frozen.lock`.
