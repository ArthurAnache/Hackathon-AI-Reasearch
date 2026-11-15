# Architecture v2: final document (Croqui, 15 Nov 2026, 04:25 PT)

**Status at 04:25 PT.** Lupa wrote `CONSENSUS` on version 8, as a design: 24.5 of 30 (version 6: 21). The two architects have nothing open between them. The outside evaluator (OpenAI) did not write `ACCEPT` on version 8: "changes required, at the design level", and "no implementation is required to close this debate". Its five points are accepted as rules or clarifications (section 2, last row). One thing is overruled: rewriting version 8 and evaluating it a third time, because no point changes a decision of version 8. That answer is signed by Meridian and by me. Lupa was asked to sign; its signature is not in the log yet.

Precise text: `<workspace>/shared/arch_v2/03_proposal.md` (version 8, unchanged since the verdict) and rules R1–R18 in `<workspace>/shared/arch_v2/04_debate_log.md`. This file is the short, buildable summary. E, S, M, C, D, L, N, W, R: evidence, survey, measurements and rules in those files.

## 1. The architecture

**In one paragraph.** A person gives a verifier, a solution format and a reference to beat. The system keeps a pool of certificates: a solution with its score, its parent and how it was made. Typed moves improve the pool: seed, polish, restart, certify, and stages a pack adds. Several voices answer every decision: the loop's present rules, the bet, learned guidelines, a model. One voice controls each decision type. The others are logged in shadow, and a share of the budget runs their picks when they differ. A voice takes control of a decision type only by winning those forks. Truth stays outside the loop: the verifier recomputes every score, the reference is read live from every named source, and only an independent arbiter makes a record.

**On a problem it has never seen.** With nothing but the verifier, the format and the reference, a default pack supplies seeds (a draw inside the format, or the public solution) and a generic perturbation polish. It reaches formats where a random draw is valid, or problems with a public solution. Resolution ladders, restart settings, venue limits and an arbiter are pack additions declared in a manifest. A model may write such a stage; it enters behind the gate.

### 1.1 Exists and runs tonight

Code: `<workspace>/wt-attack/autoloop/`. Source: **C** = code read by me at commit 38e4e76; **G** = git log, run log or lessons file read by me at 04:19, new code not read; **L** = Lupa; **R** = the builder's report.

| Piece | Source |
|---|---|
| Loop: state from the ledger, menu of legal moves, a policy picks, contracts, result, claim gate, until a cap (decisions, wall clock, compute USD, model USD). One flat loop. | C |
| Ledger, append-only. A decision stores the whole menu, the pick, the reason and the numbers. | C |
| Typed moves: seed, refine, refine_wide, certify, check_reference, stop. A child is not worse than its parent; the score is recomputed from the file. | C |
| Policies: rules, or a model (OpenAI, Anthropic, Claude) whose answer must be a menu item, else rules. | C |
| Claim gate: exact fraction below a reference read under 30 minutes earlier; re-gates after a later read; never submits. | C |
| Reference reader for 1d and 1a: fails closed; the ten cases Lupa wrote pass (commit f927819). | C, L, R |
| Certify on 1d: two independent exact implementations. | C, L |
| Tournament, same memory and budget per policy. Test C: rules exact 1.4540711 (2 claims), Claude 1.4541046 (1), gpt-6.1-sol 1.4541163 (0); USD 0.47 of compute each. | R |
| Lessons: a model adds or updates them after a run. A lint refuses reference values and scores; a rule sets the status from the cited rows (commit 6d803c3). Now 20 hints, 1 dropped. They reach the model's prompt only. | C, G |
| Packs: tao-1d, tao-1a, two EinsteinArena autocorrelation packs (one method family), and a default pack built from verifier, format and reference alone (commit 1a6410d). | C, G |
| Cold starts ran: 1d three times (best certified 1.4557078; reference 1.4546733) and first-autocorrelation once (1.5066879; reference 1.5027435). No claim. The 1d lessons were in the model's prompt. | G |
| Unseen problem ran: the default pack on circles-rectangle, 04:08, 185 s, no person. It started from the public solution and gained 4.4e-14; δ is 1e-10. | G, R |

Missing in the code (C, L): `completed` is never false; the direction flag is not read; the pack interface requires cells and p0; lessons are free text that no rule can execute. No arbiter pack exists for 1d or 1a.

### 1.2 What the architecture adds, in order

Order: a proposal to the coordinator. Hours: conjectures.

| # | Addition | Why | Owner; agent-hours |
|---|---|---|---|
| 0 | Done tonight: default pack and the run on an unseen problem (T5a); lint and rule-set status for lessons; the reader's entry cases. | priority note; N1, N3 | Tear |
| 1 | A stage written by a model for the default pack, run on Modal, kept only if the verifier confirms a gain (T5b). | T5a | Tear; 2 |
| 2 | Claim wording "below the snapshot, own check" until an arbiter verdict exists. Checker pinned by hash. | E7a; W1 | Tear; 0.5 |
| 3 | Log line: every policy answers every decision before the result (TA). | the judged log; base of 5 | Tear; 1 |
| 4 | Honest rows: one row per attempt; a stage declares its work and starts only when all of it is reserved; `completed` from its own step list (TB). | D6, L4 | Tear; 1.5 |
| 5 | Forks: 10% of a round (50% when no known move can reach the reference) runs shadow picks that differ. Unit: the root seed. 15 of 20 wins give control of one decision type. Guideline kinds and records (TG, TG2). | E3, E7c; W3, W4 | Tear; 2 |
| 6 | Arbiter pack for 1d; Lupa's six acceptance checks are in the log. | W1; S14 | WS1, named by Arthur; by 12:00 |
| Last | The bet (expected gain of the pool's best per CPU-second, from saved rows; S18) and coverage (R10) as shadow voices; stage sandbox; the 1d checks of R15. | C2; E12, W6; W8 | after 10:00, or dropped |

### 1.3 Guidelines the system writes

- **Who writes.** A model, the distiller, after each run or tournament. It never sets a status; its "drop" only queues a fork.
- **Two kinds.** Descriptive: a rate, a correlation or a cost, computed by a predicate from a code library. It states an inequality. Pick: a rule the menu can execute, citing the descriptive ones under it.
- **How they are checked.** By a rule. At birth the cited rows must show the predicted effect. After every run each guideline is recomputed on all rows of its scope that its writer did not see, counted in root seeds, at no CPU cost. A pick guideline is also a shadow voice, scored by forks.
- **Status.** Hint: it only orders probes. Trusted, after 15 wins of 20 forks: it may control its decision type; on another problem, only after trusted on two.
- **How a wrong one is removed.** A descriptive guideline fails when its inequality is false on at least 10 unseen roots. A pick guideline is dropped, with its scope, at 5 or fewer wins of its last 20 forks, and never advised again. It returns to hint when a fact under it fails or the code behind its rows changes. A trusted guideline that removes an option keeps one run of that option per run, so the rows that could refute it keep arriving.
- **Where one applies.** A pack whose manifest declares every field the guideline reads. "General" means it reads the default menu only. A lint refuses scores, reference values and file ids. A run that loads its own problem's guidelines is labelled exploitation.

## 2. What changed in each round, and why

| Version | Main change | What forced it |
|---|---|---|
| v1 Croqui | Relay of typed certificates; one round at three levels; fixed rungs; a rule for leaving polish. | Murilo's ask; E5, E6. |
| v2 Meridian | One price; a new seed is a move; survival; kept_count; forks. | Replays M1–M7: E2's gain was four restarts, not a hedge; fixed rungs lost to a simpler schedule. |
| v3 Croqui | The bet is the gain of the pool's best per CPU-second; a probe share for untried moves; autonomy, model, two-systems and log sections. | C1–C4: v2's bet mixed units and overpaid polish. E12: the winner ranked 4th before the last stage. Scope note. |
| v4 Meridian | Edge rule read from rows; every attempt is a row; stage boundary; reference snapshots. | D1–D3: v3's probe rule was written after E12. Outside review: censored rows, no boundary. |
| v5 Croqui | Bet on outcomes; cold start; spec version; replay gate for rule edits. | A seed's lift was undefined; no cold start. |
| v6 Meridian | The allocator is tested against the loop's present rules. | D4: v5's test tied by construction. D6: the builder's loop passed its gate alone. |
| v7 Meridian | Arbiter wording; root seed as unit; declared work replaces run-budget doubling; coverage; default pack; authority earned by forks; guidelines as shadow voices. | Lupa's evaluation (21 of 30; L1–L8); outside evaluation; Murilo's priority note. |
| v8 Croqui | The bet reads the item's own lineage; only a rule sets a guideline's status; two kinds of guideline, checked on rows the writer did not see; lint; control run; scope from the manifest; widths planned backwards; the reader's entry rule; T5 split into integration and search; owners and clock times. | My reading of the code: R2 did not produce its worked trace, and the model could drop any lesson. Lupa's reply (N1–N6): three lessons checked against rows they had not seen; R10 starved the deep stage; T5 could not fail for a search reason. |
| After the verdict | Lupa's notes, not conditions; the proposal is unchanged. A descriptive guideline states an inequality and fails on 10 unseen roots. In R10 the floor wins over the better half, so a boundary may keep all its arrivals. The arbiter takes its reference from the reader only: the independent checker prints two pull requests as constants. | Lupa's verdict. |
| After the outside evaluation of version 8 | Its five points, accepted without changing the proposal. (1) The runner kills a run at its reserved ceiling of CPU, wall time and memory; that row is marked incomplete and never matches completed rows: no row is censored silently. A model-written stage runs only in a fresh Modal container that mounts no secret. (2) The log stores own-lineage and pooled row counts; a bet read from fewer than 5 own rows is labelled exploratory. (3) A guideline version records its creation cutoff; a root counts once per promotion; promotion reads one series of at most 20 forks, and "the last 20" applies to demotion only. (4) The manifest's stage list grows only through an edit, behind the stage contract; an edit cannot touch the verifier, the checker, the reference sources, the venue limits or the budgets. (5) One costed schedule: sections 1.2 and 3. Its extra element, a memory exposure receipt, becomes fields of the run's first log line: guideline hashes, prompts, source roots, problems seen before. | The outside evaluation (call 6 of 16). |

## 3. Experiment plan

One test of this architecture has run: T5a. Every other number in the debate is a replay, a file read or a rerun of saved files.

| Test | What it shows | Cost | State |
|---|---|---|---|
| T5a | Integration on a problem no rule has seen (circles-rectangle: maximise): default pack. Pass: verified certificates, a complete log, no person, inside the caps. Predicted: it starts from the public solution and gains under δ. | USD 1.15 (R) | Ran at 04:08, 185 s, 9 decisions, no person. Gain 4.4e-14 against δ 1e-10, as predicted. It stopped because every lineage sat on the public solution. Not checked by me: the arbiter's verdict on its certificate. |
| T5b | Search, same problem, CPU and seeds: the loop against WS1's flat probe, and against the loop plus one stage a model wrote from the verifier and the format. Reported: gain/δ of the three. A record is not expected. Outcome rule, fixed before the run: a gain counts only when a verified certificate beats its start by at least δ; arms closer than δ tie; one run per arm is a screen. | The coordinator's order: 2 hours, USD 15 in all | The model-written stage is ordered, with two arms: from noise and from the public solution. The flat-probe arm is not ordered. |
| T0 | A replay reproduces the worked traces (R17) and refuses the attacks and reader cases of R11. | 0 USD | Reader cases pass (R). The rest not run. |
| TA | Saved ledgers replayed through every policy. Under 10% of decisions differing: that policy gets no control. | 0 USD for rule policies, 30 minutes after item 3. A model shadow costs about USD 0.05 per decision (Meridian's reading of cold3: USD 0.58 for 12 decisions). | not run |
| TG | A planted false guideline is dropped by the rule, not by the model. | 0 USD (replay) | not run |
| TG2 | The present distiller on the four cold ledgers; its changes are compared with Lupa's table (N4). | USD 0.08 | not run |
| TB | One saved first refine, run twice with declared work, gives the same sha256. | 0 USD, 8 minutes | not run |
| T1 | The softest restart beats the hardest on 15 of 20 first refines (D1). | 1.3 CPU-hours, USD 0.10–0.20 | not run |
| T3 | A model writes the refine stage; it lowers 4 of 5 saved certificates by 1e-3. | under USD 2 | not run |
| TC | Down pass: coarsen E12's winner to 12,800 cells and polish; kept if below 1.4560723. | 0 USD, 5 minutes | not run |
| T6 | 1d from an empty ledger, prompts listed. Predicted: a soft restart lowers a kept root by over 1e-3. Passing the claim is not expected (D5). | 30 CPU-hours, USD 2–5 | Three cold starts ran with the 1d lessons in the prompt: exploitation, no claim. |

**Already in hand, at no CPU cost (Lupa's N4, on the four cold-start ledgers).** Three hand-written lessons were checked against rows they had not seen. "Rank predicts only roughly": true at the last boundary only (Spearman 0.44 to 0.77 there, 0.97 to 0.99 one boundary earlier). "One strong lineage in three responds to a soft restart": it is the entry with the softest setting, 21 of 67 roots on 1d and 9 of 34 on first-autocorrelation; the next setting, 1 of 60. "Hard restarts stall": contradicted where rows exist, and later runs stopped trying them, so an obeyed lesson removed the rows that could refute it.

**Meridian's D7, on three of those ledgers, at no CPU cost.** The last stage took 85 to 91% of each cold run's CPU. Rank does not hold across its boundary, but it holds after one run inside it (Spearman 0.91 to 0.96). One entry run on every root, with everything else on the top 3 after it, would have kept each run's best certificate at about a third of that stage's CPU. It is the first candidate the guideline rule can check: one descriptive guideline (rank correlation after a stage's first run, 0.8 or more) and one pick guideline under it (cut after the first run), as a shadow against coverage. Limits: three runs, one method family, two problems.

## 4. Open

- **Dependency, a team decision:** an arbiter pack for 1d (Arthur's approval, a WS1 owner). The design cannot close it. Until then a 1d result reads "below the snapshot, own check". The fallback is exact 1.4540711000681048 from test C, 6.02e-4 below the reference (1.4546733294, pull request 371). Lupa reproduced it with the independent checker. Reproduced, not verified; not submitted.
- **Not shown:** that the bet beats a simple rule (C2: level on polish); that the loop gains anything outside step-function autoconvolution (T5a gained under δ); discovery from an empty ledger (the cold starts made no claim); that a learned guideline helps on another problem. No deep guideline can change status by forks before Sunday: a verdict needs 20 roots.
- **Lupa's scores on version 8:** evidence 4.5; recursive and chained 3.5; structured 4.5; inspired 4.5; falsifiable tonight 4; buildable and general 3.5. What keeps two of them down: the code is still one flat loop on one method family.
- **Outside evaluator (OpenAI) on version 8:** changes required at the design level; 4, 3, 4, 4, 3, 3 on the six criteria. Its five points are accepted as rules (section 2, last row). Overruled: a third rewrite and evaluation. Lupa's signature on that answer is not in the log yet.
- **Between the architects:** nothing open.

## 5. For Murilo

1. Ask Arthur to approve an arbiter pack for 1d and to name a WS1 owner. Without it the 1d result is "below the reference, own check", not a verified record.

Already decided by the coordinator: one builder, default pack first; the model-written stage for T5b (2 hours, USD 15 in all).

## 6. Development compute

E12's chain: 68 CPU-hours (D3). The 500 comparison runs of E2 and E8: 62.5 CPU-hours. Test C: 18 CPU-hours, USD 1.41 of compute (R). T5a: USD 1.15 (R). Outside reviewer: 6 calls of 16.

Not done by me: `tools/memory_facts.jsonl` and `memory_sync.py`. The sync writes to an outside service, which my role does not allow.
