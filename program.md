# program.md: instructions for the research agent

This file is read by the agent at every step (Karpathy "autoresearch" pattern).
Humans edit this file to steer the research; the agent never edits it.

## Goal
Improve the score returned by `problem.verify` (direction: **maximize**).
Current dummy problem: 12 points in the unit square, maximise the minimum
pairwise distance. Literature best ≈ 0.366025.

## Rules
1. You may only change `strategy/search.py` and `strategy/config.yaml`.
2. `search(config, seed, budget_s)` must return `{"points": [[x, y], ...]}`
   with exactly 12 points inside [0,1]^2, within `budget_s` seconds.
3. Everything is scored by the frozen verifier with fixed seeds and a fixed
   time budget. Self-reported scores are ignored. Do not try to game it.
4. One focused hypothesis per experiment, stated so it can be proven wrong.
5. Read the history. Do not repeat a rejected idea unless you say what is
   different this time.

## Ideas worth trying (not exhaustive)
- Local search / simulated annealing with a cooling schedule
- Repulsion (treat points as charges), then polish
- Random restarts, keep the best within the budget
- Starting from structured layouts (grids, hexagonal patterns)
- Using more of the time budget (runs currently finish well under it)
