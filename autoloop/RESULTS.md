# autoloop results

## Test A — dry run, fake executor (2026-11-14 22:20 PT) — verified

`.venv/bin/python -m unittest autoloop.tests.test_autoloop` → 11 tests OK in < 0.1 s:
- The loop closes: check_reference → seed → refine → certify → claim gate. It stops at the decision cap, the
  wall-clock cap, the compute-USD cap, and by itself when no legal move is left.
- Contracts: a child worse than its parent, an unknown parent, or a cell count above the executor limit are refused.
- LLM policy (test double): valid answers are used, tokens and USD are logged per call, and invalid answers (index
  out of the menu, not JSON, no reason) fall back to rules. The LLM USD cap switches to rules.
- Memory: imported certificates keep `origin: imported`; with enough lineages in memory there is no new seed.
- Tournament (fake executor, 3 decisions per entrant: rules, test double, real `claude -p`) prints the comparison
  table. `claude -p` with its own system prompt and no tools: about USD 0.005 per decision.

## Test B — small real run from empty, local, 4 workers (2026-11-14 22:03–22:19 PT) — verified

```
.venv/bin/python -m autoloop.loop --run autoloop/runs/testB --executor local --max-wall-s 1200 \
    --seed-budget-s 60 --refine-budget-s 150 --workers 4 --min-roots 4 --max-decisions 60
.venv/bin/python -m autoloop.report autoloop/runs/testB/ledger.jsonl
```

- 13 decisions, 954 s wall, 3,668 CPU-seconds, USD 0, no human input.
- Reference read live at test C: 1.4546733414 (open PR #345 on teorth/optimizationproblems). It later moved to 1.4546733294 (PR #371, read 02:04 PT on 15 Nov, cold1 ledger).
- 4 seeds from noise (WS2 `1d_tuned.py`, 60 s, 800 cells): best 1.4642194.
- Rules promoted the top 3 lineages in turn: 800 → 3,200 → 12,800 cells, then harder restarts.
- Best: 1.462353698756413 at 12,800 cells. Exact check passed (c1d_round + c1d_exact). The claim gate wrote "not
  below the reference" (expected at this budget).
- Stopped by itself with 246 s left (a refine needs 270 s).
- Limitation: 20 minutes from noise is far from the record; test C starts from today's certificates.

## Test C — tournament of three policies from a frozen memory (2026-11-14 22:25 – 2026-11-15 00:06 PT) — verified

Same frozen memory for all arms (`runs/C_memory/ledger.jsonl`: 235 certificates copied once from today's files;
best exact 1.4542590631143155 at 51,200 cells), 90 min wall clock, USD 15 Modal each, run in parallel, no human input.
Hybrid executor (Modal CPUs above 12,800 cells, `attack/modal_c1d.py --detach`), restarts of 1,200 s.

| Arm | Best exact | Claims passed | Decisions | Modal USD (est.) | Model USD |
|---|---|---|---|---|---|
| rules | **1.454071100068105** | 2 | 7 | 0.47 | 0 |
| claude -p | 1.454104568666658 | 1 | 6 | 0.47 | 0.13 (subscription, notional) |
| gpt-6.1-sol | 1.454116274488392 | 0 | 5 | 0.47 | 0.12 (assumed price: 15 / 60 USD per 1M tokens) |

- Claims: below the live reference read < 30 min before (1.4546733414, PR #345, at test C; now 1.4546733294, PR #371) and below the memory's best exact.
  The coordinator re-checked the first rules claim (1.4541470296475214) with independent exact code: same fraction.
- gpt-6.1-sol certified before reading the reference, so its claim gate wrote no claim. Limitation of the gate,
  fixed in cold2 (certified results are re-gated after a later reference read).
- Limitation: every Modal call of the third round was cancelled at 23:22–23:25 PT by an external cancellation signal
  (all `revolutionaries-c1d` apps at once); results came from the volume checkpoints, so round 3 is partial.
- Judge logs: `python -m autoloop.report autoloop/runs/C_tournament/<arm>/ledger.jsonl`.

## Cold and model-stage runs (2026-11-15) — verified (ledgers in `runs/<run>/ledger.jsonl`)

No run passed the claim gate. Nothing was submitted.

| Run | Ended (PT) | Best exact | Target (live reference read by the run) | Passed |
|---|---|---|---|---|
| cold1 (1d) | 02:21 | 1.456288886536595 | < 1.4546733294 (PR #371) | no |
| cold2 (1d) | 03:06 | 1.4558429835864832 | < 1.4546733294 | no |
| cold3 (1d) | 04:08 | 1.4557078032949418 | < 1.4546733294 | no |
| ea1a_cold1 (1a) | 03:13 | 1.5066878908660457 | < 1.5026648336 (PR #367) | no |
| cold4 (1d, rules, recipe d7) | 08:11 | 1.4562254276460898 | < 1.4546733294 | no |
| t5c_noise (circles-rectangle, model-written stage; from noise by code inspection, isolation not enforced) | 05:18 | sum 2.3658323759108315 | > 2.3658323759098576 + δ 1e-10 | no (+9.7e-13) |
| t5d_noise_isolated (circles-rectangle, model-written stage, from noise, isolation enforced: Modal block_network=True, no problems/, no volume, no secret; checked in-container on all 32 jobs) | 10:09 | sum 2.365832375910797 | > 2.3658323759098576 + δ 1e-10 | no (+9.4e-13) |
| t5c_public (circles-rectangle, model-written stage from public #1) | 05:18 | sum 2.3658323759108164 | as above | no (+9.6e-13) |

- cold4 label: "no solutions carried over; lessons distilled by the system from its earlier runs on this problem".
  Its ledger records `cold_start=false`, `pastdev_read=false`. 24 decisions, 3.7 h, USD 4.72 Modal; it stopped by
  itself (no legal move left).
- D7 / recipe d7 / lesson G11m (Meridian's wording): in replay on three ledgers the recipe kept the best certificate
  with about a third of the last-stage CPU; one prospective run (cold4) cost 51 CPU-hours against 91 and 141 and ended
  worse (1.4562254), for reasons upstream of that stage (best at 12,800 cells was already 1.4569237 against 1.4560101
  and 1.4563251; 120 seeds against 200; 1,200 s jobs against 2,400 s); not yet tested by a fork.
  G11m stays a hint, not active.
- t5d_noise_isolated: version 4's model call timed out once (10:00 PT); the run was resumed in the same ledger with
  `--resume` (note in the ledger), so all 4 versions ran. USD 0.46 Modal, USD 1.15 notional OpenAI.
