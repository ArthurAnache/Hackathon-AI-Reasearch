# autoresearch for mathematical optimisation

This is an experiment to have the LLM do its own research, adapted from
[karpathy/autoresearch](https://github.com/karpathy/autoresearch). Instead of training
a language model, you improve a **search algorithm** for an open problem from Terence
Tao's [optimisation constants list](https://teorth.github.io/optimizationproblems/).
You are scored by a frozen, independent verifier.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag and a problem.** Propose a tag based on today's date, for example `nov14`.
   List the problems with `python prepare.py list`. The branch `autoresearch/<tag>-<problem>`
   must not already exist, because this is a fresh run.
2. **Create the branch** from the current main: `git checkout -b autoresearch/<tag>-<problem>`.
3. **Read the in-scope files.** The repo is small, so read all of these for full context:
   - `README.md`: repository context.
   - `prepare.py`: fixed constants (time budget, seed) and the CLI. Do not modify.
   - `interfaces.py`: the team contract and the experiment protocol (`run_experiment`). Do not modify.
   - `problems/<problem>.py`: the problem statement, the verifier and the best known value. Do not modify.
   - `optimizers/<problem>_*.py`: the starting optimizers for this problem (`_baseline` is simple, `_tuned` is the best known so far).
4. **Create `search.py`**: `cp optimizers/<problem>_tuned.py search.py`, then `git add search.py && git commit -m "start"`.
5. **Check the frozen files**: `python prepare.py check` must print `frozen: ok`.
6. **Confirm and go.** Confirm that the setup looks good.

Once you get confirmation, start experimenting.

## Experimentation

Each experiment runs the search for a **fixed time budget** (`TIME_BUDGET` in `prepare.py`,
wall clock) with a fixed seed. You launch it with your hypothesis as the argument:
`python search.py "Doubling N_CELLS will lower the score: the run ends far from converged"`.

**What you CAN do:**
- Modify `search.py`. This is the only file you edit. Everything in it is fair game: the algorithm
  (local search, annealing, gradient methods, linear programming, restarts, structured
  initialisations), the representation (number of cells, matrix size, dimension) and the hyperparameters.
  Keep its contract: `PROBLEM`, a `CONFIG` dict of hyperparameters, `search(budget_s, seed, config=CONFIG)`
  returning a candidate, and `run(__file__)` under `if __name__ == "__main__":`.

**What you CANNOT do:**
- Modify `prepare.py`, `problems/` or `frozen.lock`. They hold the problem, the verifier and the protocol.
  The harness checks their hashes and voids any result if they changed.
- Change `PROBLEM` in `search.py`, or change the `run(__file__)` entry point.
- Install new packages. You can only use the standard library and `numpy`.
- Game the verifier. The score is whatever `problems/<problem>.py` says. Your own internal score is only a guide.

**The goal is simple: get the best score.** The direction (lower or higher) is printed in every summary.
The time budget is fixed, so you don't need to worry about runtime. Use the whole budget, but
`search()` must return within it: more than 10% over voids the result, and twice the budget kills the run.

**Simplicity criterion.** All else being equal, simpler is better. A tiny improvement that adds a lot
of hacky code is probably not worth keeping. Removing code and getting an equal or better score is a
great outcome.

**Records.** Every summary prints `beats_record:`. If it ever says `True`, stop and tell the human.
Certify the result first with `python prepare.py verify <problem> <solution_path> --exact`.

**The first run**: your very first run should always establish the baseline, so run `search.py` as is.

## Output format

When the run finishes it prints a summary like this:

```
---
score:          1.4673685487
status:         keep
experiment_id:  exp_0007
problem:        1d
best_before:    1.4681
beats_record:   False
budget_s:       60.0
duration_s:     57.0
seed:           42
algorithm_version: 3f2a9c1d0b4e
solution_path:  solutions/1d/exp_0007.json
direction:      minimize
best_known:     1.457854884921989
decision:       keep
```

Read it with `grep "^score:\|^status:\|^decision:" run.log`. If the run crashed or was invalid, there is no `score:` line,
the `error:` line says why, and the decision is `reset`.

## Logging results

The harness logs every experiment for you: one record per run in `experiments.jsonl` (hypothesis, config,
`algorithm_version`, score, status, ...). Do NOT commit it. `python analysis.py` summarises it: best so far per
problem, keep rate, and the gap to the record.

## The experiment loop

The experiment runs on a dedicated branch, for example `autoresearch/nov14-1d`.

LOOP FOREVER:

1. Look at the git state (the current branch and commit) and at `python analysis.py` (what was tried and how it went).
2. Change `search.py` to try one experimental idea, by editing the code directly.
3. Run `git commit`.
4. Run the experiment: `python search.py "<your hypothesis>" > run.log 2>&1`. Redirect everything; do NOT use tee or let the output flood your context.
5. Read out the result: `grep "^score:\|^status:\|^decision:" run.log`.
6. If the status is `crash`, run `tail -n 50 run.log` to read the error and try to fix it.
   If you can't get it to work after a few attempts, give up on that idea.
7. If the decision is `keep`, "advance" the branch and keep the git commit.
8. If the decision is `reset`, run `git reset --hard HEAD~1` to go back to where you started.

You are a completely autonomous researcher trying things out. If they work, keep them. If they don't,
discard them. Choose each next experiment from what you learned: build on what was kept, don't repeat
what was discarded, and say in the hypothesis why you are trying it.

**Timeout**: each experiment takes about `TIME_BUDGET` seconds plus a few seconds of overhead. The harness kills a
search that runs past twice the budget.

**Crashes**: if a run crashes, use your judgement. If it's something dumb and easy to fix (a typo, a missing
import), fix it and re-run. If the idea itself is fundamentally broken, skip it and move on.

**NEVER STOP**: once the experiment loop has begun (after the initial setup), do NOT pause to ask the human
whether to continue. The human might be away and expects you to keep working until you are manually
stopped. If you run out of ideas, think harder: re-read the problem page linked in `problems/<problem>.py`,
try combining near-misses, try more radical algorithmic changes.
