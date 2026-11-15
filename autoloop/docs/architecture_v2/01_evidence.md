# Evidence pack, 14 Nov 2026 (inputs for the architecture v2 debate)

Written by the coordinator from our own measurements. Cite these as E1..E11. Full report with tables:
`<workspace>/shared/snapshots/2026-11-14_2045/research-map-report.html`.
Current code: `<workspace>/repo/integrations/` (ideas.py, verdicts.py, idea_chooser.py, idea_agent.py,
research_map.py, run_loop.py). The WS2 loop it sits on: `<workspace>/wt-compare/` (loop.py, prepare.py,
optimizers/, problems/).

## What Murilo asked for (translated)

Rethink the architecture of the model itself so that it is **more recursive and chained** and **thinks in a more
structured way**. Take inspiration from benchmarks and systems of other fields: systems that solve similar problems, and
systems that solve completely different ones. **Do not copy what exists; use it as inspiration.** Work iteratively.

## Today's architecture, in five lines

- Unit of search: an **idea** (a claim with a source and a runner = optimizer file + settings), not a config.
- A UCB1 bandit picks an idea for each experiment; never a dead end; one knob tuner (WS2 FeedbackAgent) per idea.
- Every experiment carries `idea=<id>`; verdicts are recomputed from experiments, gain in units of delta.
- One flat loop: N experiments of fixed time, each from scratch or warm-started from the best saved solution.
- No stages: nothing takes the output of one method as the typed input of another.

## Evidence (problem 1d = Tao signed autocorrelation constant, lower is better, unless stated)

- **E1. The start point dominates.** Seeds spread from 1.457 to 1.50 at 800 cells. Seeds 106 and 162 are the top two
  in all three arms. The ranking at 800 cells is roughly kept after refinement to 12,800 cells.
- **E2. Map loop vs old loop vs one long run, 100 paired seeds, equal time (450 s).** The map wins 85 and 86 of 100.
  Medians 1.46375 vs 1.47274 and 1.47263. Best of 100 is the same for the three (1.45741, 1.45729, 1.45718).
- **E3. The chooser did not choose.** Over 1,000 experiments its four runnable ideas got 245, 252, 242, 261 runs
  (10 experiments per run is too few pulls). The gain in E2 is hedging over varied settings (300/800/1,600 cells,
  learning rate), not learning which idea is best.
- **E4. The knob tuner adds nothing measurable here.** Old loop (tuner, 10 x 45 s) vs single run (1 x 450 s): 54 of 100.
- **E5. Chaining beats more of the same.** The 800-cell search stops improving after about 8 experiments. Feeding its
  output to a different method (split cells, p-norm continuation L-BFGS) gives 1.45729 -> 1.45604 (3,200 cells) ->
  1.45588 (12,800 cells) in about 2 CPU-minutes.
- **E6. The public state of the art is a relay.** Each 1d certificate refines the previous team's (10,240 cells
  1.455683 -> 81,920 cells 1.454728 -> 163,840 cells 1.4546733), gains of 1e-3 early and 1e-9 late. The restart
  parameter matters and not monotonically. Best open claim: 1.4546733414. Ours: 1.4558843 at 12,800 cells, in progress.
- **E7. Our failures were process failures.** (a) A "record" was announced against a stale reference: the constant in
  a frozen file, not the open pull requests. (b) A stage consumed solutions of the wrong problem: inputs were selected
  by file pattern and lowest score. (c) A verdict from one seed can flip on seed luck. (d) With 2 ideas and 4
  experiments the exploration cost made the map lose (a teammate's benchmark, 3 seeds: 0 of 3 on 1d).
- **E8. Generality is unknown.** The map was measured at scale on 1d only. The old loop with 100 seeds improved the
  team's earlier values on 1a and 1b without a record (gaps 4.3e-3 and 1.9e-4).
- **E9. Problems differ in what helps.** On 1b the best seed reached its best value in experiment 1 and never moved.
  On 1a every experiment improved a little.
- **E10. Costs.** 100 runs x 450 s on Modal CPUs: 1 to 2 USD. Exact check of a 163,840-cell certificate: 21 s.
  Refinement to 12,800 cells: about 1 minute on a laptop.
- **E11. Second-autocorrelation (EinsteinArena).** From the #1 solution, local moves gain about 5e-11 per 120 s; a
  sequential LP on the active set gains 7 times more with diminishing returns; all are 1e-5 short of a record.

## Constraints

- Submission: Sunday 15 Nov, 14:00 PT. Track: "an autonomous research loop that attempts to improve a verified
  mathematical record, with reproducible experiments and decision logs".
- Anything proposed must name a test that can run tonight, with its cost.

## Added 21:45 PT

- **E12. A chained run of our own passed the best open claim on 1d.** Old-loop seed 162: 800 cells 1.4573206 ->
  12,800 cells 1.4560723 (local) -> 51,200 cells **1.454406313885605** (Modal, exact check by four independent
  calculations; best open claim 1.4546733414, re-read at 21:25 PT). Of 55 restarts at 51,200 and 204,800 cells over
  five lineages, 2 are below the claim, both from seed 162 with the softest restart (p0 = 4): 1.4544063 and 1.4544376.
  The same seed from the map loop reached 1.4547298 at p0 = 4. Every restart with p0 >= 16 stayed at 1.4558 to 1.4561.
  Seed 106, the best at 800 and at 12,800 cells, ended at 1.4558318: **the rank before the last stage did not predict
  the winner; the restart setting of the last stage did.** The leaders' record also descends from a single seed.
  Files: `<workspace>/shared/1d_candidate_2026-11-14_2117/`. Not submitted anywhere.
