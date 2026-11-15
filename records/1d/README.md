# Problem 1d: results

Problem: signed autocorrelation constant C_1d from the Tao optimization-constants list
(https://teorth.github.io/optimizationproblems/constants/1d.html). A step function with N equal cells gives the
upper bound `2N * max_k |(v*v)_k| / (sum v)^2`; lower is better.

## Autonomous result: 1.4540711000681048 (15 Nov 2026)

| File (`autonomous/`) | Cells | Exact score |
|---|---|---|
| `c006_31bbf628_int56.json` | 51,200 | **1.4540711000681048** = 21267647941510998263770007610303168512/14626277862557668614837367632277156521 |

Found by the autonomous loop in `autoloop/` with its rules policy, run `autoloop/runs/C_tournament/rules`, with no person
in the loop. It started from a frozen memory of 239 certificates we already had (best 1.4542590631143155, the assisted
result of the next section), chose which lineages to continue and how to restart them, ran the exact check and passed
its own claim gate (`autonomous/CLAIM_PASSED.json`: the reference was read 7 s before the claim).

- **Reference.** Best open claim in the Tao repository, re-read on 15 Nov: 1.4546733294 (pull request #371). The claim
  file records the reference at claim time, 1.4546733414 (#345). The file is 6.02e-4 below both.
- **Checked by** `highres/check_independent.py` (same fraction, about 50 s), and reproduced by a second agent from a
  clean checkout.
- **Same tournament, a model as policy.** Claude: 1.454104568666658, claim passed. gpt-6.1-sol: 1.4541162744883922,
  no claim, because it never chose to read the reference.

**What this is not.**

- Not from scratch. Four cold-start runs on 1d (empty memory, lessons only) did not pass: best 1.4557078032949418.
  See `autoloop/RESULTS.md`.
- Not submitted anywhere, and not verified by the team's arbiter (there is no arbiter pack for 1d yet). It reads
  "below the reference snapshot, own check".
- In the run logs under `autoloop/runs/`, local paths were replaced by `<workspace>`; nothing else was edited.

## High-resolution result: below the best open claim (14 Nov 2026, 22:20 PT)

| File (`highres/`) | Cells | Exact score |
|---|---|---|
| `c1d_n51200_1.4542590631143155.json` | 51,200 | **1.4542590631143155** = 132922799643235991523198037060598323200/91402421353029087699033936155957709001 |
| `c1d_n51200_1.454406313885605.json` | 51,200 | 1.454406313885605 = 33230699911294658122227553895833414400/22848291838417025440738871578647258529 |

Best claim we found for 1d: 1.4546733414 (open pull request #345 on github.com/teorth/optimizationproblems, 163,840
cells), re-read on 14 Nov at 22:21 PT. The best file is 4.14e-4 below it. Integer heights, max |v| = 2^56.

**Checked by:** `highres/c1d_exact.py` (Kronecker substitution, a direct sum at the deciding knot, a check modulo a
61-bit prime), the frozen verifier of the WS2 loop (`problems/p1d.py`, same fraction), and
`highres/check_independent.py`, written separately and sharing no code with the first:

    python records/1d/highres/check_independent.py records/1d/highres/c1d_n51200_1.4542590631143155.json   # about 50 s

**How it was obtained.** A chain of three stages on one of our own starts (old loop, seed 162):
800 cells from random noise, 1.4573206 (`best_1d_oldloop_seed162.json`) -> split and refined to 12,800 cells, 1.4560723
(`highres/lineage_seed162_n12800.npy`) -> split to 51,200 cells and restarted with a soft p-norm (p0 = 4), then once
more at p0 = 8. Of 55 restarts over five lineages in the first round, 2 were below the open claim, both from seed 162
at p0 = 4; every restart with p0 >= 16 stayed at 1.4558 to 1.4561.

**Method credit.** The recipe (split the cells, minimise log ||v*v||_p - 2 log |sum v| with L-BFGS while p is doubled,
restart from a low p) is the one described in pull requests #312, #345 and #270 of the Tao repository. The code
(`highres/c1d_refine.py`, `c1d_round.py`, `c1d_exact.py`) and the starting solutions are ours; no certificate of
another team was used.

**What this is not.**

- Not submitted anywhere, so nobody outside the team has replayed it.
- Not found by an autonomous system. Every decision in this chain (which lineages to continue, which restart to
  try, when to stop) was made by Claude Code agents under human supervision. The autonomous loop's result is the section above.
- The reference is the Tao repository (page table and open pull requests). Other venues were not searched.

## 800-cell results (14 Nov 2026, afternoon)

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

**These 800-cell values are not the best known bound.** Open, unmerged pull requests on github.com/teorth/optimizationproblems claim
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
