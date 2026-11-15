# Problem 1d: upper bound below the best value listed on the Tao page (not the best known)

Problem: signed autocorrelation constant C_1d from the Tao optimization-constants list
(https://teorth.github.io/optimizationproblems/constants/1d.html). A step function with N equal cells gives the
upper bound `2N * max_k |(v*v)_k| / (sum v)^2`; lower is better.

## Result (14 Nov 2026)

| File | Loop | Seed | Cells | Exact score |
|---|---|---|---|---|
| `best_1d_oldloop_seed106.json` | old (tuned optimizer) | 106 | 800 | **1.4572871862906502** |
| `best_1d_oldloop_seed162.json` | old (tuned optimizer) | 162 | 800 | 1.4573206011778774 |
| `best_1d_maploop_seed162.json` | research map | 162 | 800 | 1.4574123218068238 |
| `best_1d_maploop_seed106.json` | research map | 106 | 800 | 1.4575078453660368 |
| `best_1d_maploop_seed152.json` | research map | 152 | 1600 | 1.4576365413545382 |

Reference values on the page, as read on 14 Nov 2026:

- 1.457854884921989, SCiEL, 800 cells. Best value in the page's table, marked with an asterisk (minimal
  verification).
- 1.4630715826075594, 300 cells. Best value in the table replayed by a second party.

All five files score below both.

**This is not the best known bound.** Open, unmerged pull requests on github.com/teorth/optimizationproblems claim
lower values, as read on 14 Nov 2026: 1.4546733414 (#345, 163,840 cells), 1.4546733458 (#270, 163,840 cells),
1.454727664528751 (#312, 81,920 cells), 1.4556234448 (#271, 40,960 cells) and 1.455682936505816 (#269, 10,240
cells). Our files use 800 to 1,600 cells and are 0.0026 above the best of these claims. The verifier's
`beats_record: True` compares against the SCiEL value only.

Each file was checked with the frozen verifier of the WS2 loop:

    python prepare.py verify 1d records/1d/<file> --exact      # prints beats_record: True

The seed-162 map-loop file was also recomputed with exact fractions outside the verifier, with the same value.

## What the result does and does not show

- The result comes from many independent starts (100 seeds in parallel on Modal CPUs, 10 experiments of 45 s
  each), with the WS3 optimizer `optimizers/1d_tuned.py`. Every search starts from random noise.
- **Both loops beat the value listed on the page.** The old loop has the best single result, so that result is
  not due to the research map.
- **The map loop is the more reliable loop.** On the same seeds and budget it wins 85 of 100 paired runs on Modal
  against the old loop (`comparison_modal_100seeds.csv`; median 1.46375 against 1.47274) and 13 of 15 on a laptop
  (`comparison_local_15seeds.csv`, 16 experiments of 30 s).
- **It also beats one long run of the tuned optimizer** (same seeds, 1 experiment of 450 s instead of 10 of 45 s):
  86 of 100 paired runs, median 1.46375 against 1.47263. The single run and the old loop are level (54 of 100).
  The best single values of the three arms are the same to 3e-4 (1.45718, 1.45729, 1.45741): the map improves the
  typical run, not the best one. The map had four runnable ideas, all variants of the tuned optimizer (default,
  1,600 cells, 300 cells with a higher final beta, lower learning rate), and the baseline optimizer marked as a
  dead end, so it spent no experiment on it.
- The values are not on the Tao page. A submission there needs a pull request with a permanent link to the
  certificate, and the entry would carry an asterisk until a second party replays it.

## Reproduce

`modal_seeds.py` (problem 1d, `--arm A` or `--arm B`) and `modal_problem.py` (old loop, any problem of the loop)
expect a folder `modal_src/` next to them: a clean copy of the WS2 loop checkout (`feature/ws2-autoresearch-loop`)
with this branch's `integrations/` on top, without `.env`, caches or old runs, plus `ideas_clean.jsonl` (the idea
list with verdicts cleared) for arm B.

    python -m modal run modal_seeds.py --seed-start 100 --n-seeds 100 --nexp 10 --budget 45 --arm B --out modal_results
    python -m modal run modal_seeds.py --seed-start 100 --n-seeds 100 --nexp 10 --budget 45 --arm A --out modal_results_A

Each launch uses 100 containers with 2 CPUs for about 8 minutes. No GPU and no secrets are used.

The runs were launched and checked with Claude Code; the numbers above are the verifier's output.
