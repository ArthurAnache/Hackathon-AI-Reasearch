# Hackathon-AI-Research: autoresearch loop for math records

An autonomous research loop (Karpathy "autoresearch" pattern) that tries to
improve a verified mathematical optimisation score. An agent proposes one
hypothesis at a time, edits only the search strategy, and a frozen runner
measures it with fixed seeds and a fixed budget, verifies it independently,
keeps or rejects it via git, and logs every decision.

The problem is a placeholder for now (12 points in a unit square, maximise the
smallest pairwise distance, best known ≈ 0.388730). Swap in a real record
(EinsteinArena, Tao optimisation constants, AlphaEvolve) by replacing
`problem/problem.py`; see INTERFACES.md.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest                      # 10 tests, ~2 s
python -m runner.cli run -n 2         # baseline (first time) + 2 experiments
python -m runner.cli status           # table of every experiment + best score
```

To let Claude choose the experiments instead of the offline playbook:

```bash
cp .env.example .env                  # then paste your ANTHROPIC_API_KEY into .env
python -m runner.cli run -n 5 --agent claude
```

## How to read this code

### The three parts

| Folder | Role | Who may edit |
|---|---|---|
| `problem/` | **The referee.** Defines the problem and `verify(solution) -> score`. The only score we trust. | Humans only. Frozen and hash-checked. |
| `strategy/` | **The player.** `search.py` (the algorithm) and `config.yaml` (its knobs). | The agent. This is all it may change. |
| `runner/` | **The lab protocol.** The loop, the agent, evaluation, git and logging. | Humans only. Frozen and hash-checked. |

### File map

```
problem/problem.py     verify(), InvalidSolution, N_POINTS, DIRECTION, KNOWN_BEST
strategy/search.py     search(config, seed, budget_s) -> {"points": [...]}
strategy/config.yaml   method, n_samples, step_size
runner/config.yaml     experiment protocol: seeds, budget, keep threshold, agent, model
runner/cli.py          `run`, `status`, `freeze` commands
runner/loop.py         run_experiment(): one full iteration of the loop (start here)
runner/agent.py        HeuristicAgent (offline playbook) and ClaudeAgent (API)
runner/evaluate.py     runs the strategy per seed in a subprocess, then calls verify()
runner/_worker.py      the subprocess entry point that imports strategy/search.py
runner/frozen.py       sha256 of problem/ and runner/ vs frozen.lock
runner/gitops.py       commit on keep, checkout/clean strategy/ on reject
runner/history.py      read/append experiments.jsonl, best-so-far
program.md             the research brief the agent reads every step
INTERFACES.md          contracts and the experiments.jsonl schema
experiments.jsonl      the decision log, one JSON row per experiment
frozen.lock            hashes of the frozen files
tests/                 verifier, tamper detection, scope guard, a 2-experiment loop, crash handling
```

### The path one experiment takes (`runner/loop.py: run_experiment`)

1. **Check the frozen files.** `frozen.check()` recomputes hashes of `problem/` and `runner/` and stops if anything changed.
2. **Read history.** `history.read()` loads `experiments.jsonl`; the best kept score is the bar to beat. If there is no baseline yet, it scores the unchanged strategy first.
3. **Propose.** `agent.propose(ctx)` returns a hypothesis, a rationale citing the history, and new contents for files in `strategy/`.
4. **Apply, inside the fence.** `apply_proposal()` refuses any path not in `editable_paths`, writes the files, and re-checks that nothing outside `strategy/` changed.
5. **Run and verify.** `evaluate()` runs the strategy once per fixed seed, each in a fresh process with a hard timeout, and scores each solution with `problem.verify`. Score = mean over seeds.
6. **Decide.** Keep if the score beats the best so far by more than `min_improvement`, and commit `strategy/`. Otherwise (no gain, crash, timeout, invalid solution) revert `strategy/` with git.
7. **Log.** Append a row (hypothesis, rationale, per-seed scores, decision, failure_reason, frozen hash, commit) and commit `experiments.jsonl`.

So `git log` reads as a lab notebook: each kept idea is a commit, and each experiment, kept or not, has a log commit.

### Changing a frozen file on purpose

Edit it, then run `python -m runner.cli freeze` and commit `frozen.lock`. Without that, the loop refuses to run.
