# README1 — WS1: problem selection, true records, frozen verifier

Owner: Arthur. Team: The Revolutionaries, SF AI Research Hackathon (14–15 Nov 2026), Track 1: an autonomous
autoresearch loop that must beat a known mathematical record.

This file describes in detail everything done in Workstream 1: what was done, why, how to reproduce it, the results,
and what is still open. Every result is labelled as one of the following:

- **Verified:** we reproduced it ourselves, locally, with code in this repo.
- **Hypothesis:** we believe it, but have not checked it independently.
- **Limitation:** something we know we did not or could not check.

Last updated: 2026-11-14, 12:30 PT.

---

## 1. What WS1 delivers

1. **The problem.** One main problem and one backup, chosen with an explicit, reproducible method (section 3).
2. **The true current record** for each candidate. The record to beat is the best value among EinsteinArena,
   Terence Tao's optimization-constants table and AlphaEvolve, once each value has been re-verified and its
   definition checked against ours.
3. **A frozen verifier.** WS1 provides `problem.verify(candidate) -> bool` and `problem.score(candidate) -> float`
   for the shared contract. Once the problem is chosen, the verifier and the problem definition are frozen: the
   agent (WS2) only changes the search strategy.

### Rules that WS1 enforces

- A record is never claimed because an LLM said so, or because a leaderboard shows a value we have not re-verified.
- An improvement counts only if an independent check confirms it, by exact or rational arithmetic or a second
  implementation, and only if it exceeds both the numerical tolerance and EinsteinArena's threshold δ.
- No secrets in the repo: keys live in `.env` (git-ignored), and only `.env.example` is committed.

---

## 2. Setup and reproduction

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt     # numpy, scipy, mpmath, sympy, requests, exa-py, python-dotenv
cp .env.example .env                          # fill EXA_API_KEY; EINSTEIN_ARENA_API_KEY only needed for writes

.venv/bin/python selection/stage1_filter.py           # stage 1 -> selection/registry_stage1.jsonl, selection/verifiers/
.venv/bin/python selection/stage2_profile.py          # stage 2 -> selection/registry_stage2.jsonl   (~2 min)
.venv/bin/python selection/stage3_probe.py --budget 600 --workers 9   # stage 3 -> registry_stage3.jsonl (~21 min)
.venv/bin/python selection/literature_check.py ring-loading-15 third-autocorrelation-inequality   # Exa check
.venv/bin/python selection/thread_digest.py           # forum threads -> selection/digests/<slug>.md (~1 min)
.venv/bin/python integrations/submit.py SLUG SOLUTION.json --expected-score X   # gated dry run (no --confirm)
.venv/bin/python selection/freeze_problems.py         # stage 4: freeze the chosen problems (refuses to overwrite)
.venv/bin/python -m unittest discover -s tests -p "test_ws1_*.py"                # frozen-problem tests (~9 s)
.venv/bin/python selection/verify_claim.py SLUG CANDIDATE.json                  # arbitrate a claim (never submits)
```

Pinned sources:

| Source | Commit / URL |
|---|---|
| EinsteinArena code | `github.com/vinid/einstein-arena` @ `5baf6361a3e0e2123edb98f676ef44f7ab081ad6` (upstream HEAD on 2026-11-14) |
| AlphaEvolve problem repository | `github.com/google-deepmind/alphaevolve_repository_of_problems` @ `8f447457957deac61e28bf1676746f0753b3b2f8` |
| EinsteinArena live API | `https://einsteinarena.com/api/...` (GET requests are public; no token needed to read) |
| Tao's table | `https://teorth.github.io/optimizationproblems` |

### Files

| Path | Role |
|---|---|
| `selection/stage1_filter.py` | Stage 1: fast elimination over all 34 EinsteinArena problems |
| `selection/registry_stage1.jsonl` | One line per problem; key `"problem_id": "einsteinarena/<slug>"` |
| `selection/verifiers/` | Verifiers extracted by stage 1 (simplified unescaping; see limitations) |
| `selection/stage2_profile.py` | Stage 2: live record, local re-verification, timing, δ, gap #1/#2 |
| `selection/eval_runner.py` | Runs one verifier on one solution in a clean process (`python -I`) and times it |
| `selection/registry_stage2.jsonl` | One line per problem (18 rows) |
| `selection/stage3_probe.py` | Stage 3: 10-minute generic optimizer per problem, run in parallel |
| `selection/registry_stage3.jsonl` | One line per problem (14 rows), in the shared experiment-log format |
| `selection/literature_check.py` | Exa literature cross-check with curated, cited values |
| `selection/literature_check.jsonl` | One line per problem checked |
| `selection/exa_log.jsonl` | Log of every Exa call (no secrets) |
| `integrations/exa_client.py` | Exa adapter (`search`) |
| `integrations/einstein_arena.py` | EinsteinArena API client: threads, replies, search, leaderboard, submission primitives |
| `selection/arena_log.jsonl` | Log of every EinsteinArena call (endpoint, params, status, result count, ids; no bodies, no key) |
| `selection/thread_digest.py` | Fetches top + recent threads and replies, writes `selection/digests/<slug>.md` |
| `selection/digests/` | Per-problem forum digests: curated summary (cited) + generated thread index |
| `selection/independent_verifiers.py` | Relay to the independent verifiers in `problems/<slug>/independent.py` (used by `submit.py`) |
| `selection/decision.md` | Stage 4: weighted rule, scores, tie-break, decision (main / backup / record bet) |
| `selection/freeze_problems.py` | Freezes a chosen problem into `problems/<slug>/`; `--manifest` rewrites `problems/MANIFEST.json` |
| `problems/` | Frozen catalog: `catalog.py`, `_frozen.py`, one directory per problem (pinned verifier, certificate, meta, independent verifier, API) |
| `tests/test_ws1_frozen_problems.py` | Freeze rule, certificates, independent agreement, circles exploits, schema limits (unittest, ~9 s) |
| `selection/verify_claim.py` | WS1 arbiter: pinned + independent verifier + live #1 → verdict in `selection/claims.jsonl` |
| `selection/claims.jsonl` | Every arbitrated claim (WS1 self-tests so far) |
| `docs/ws1_handoff.md` | Handoff to WS2/WS3: problems, records, δ, interface, limits, forum knowledge base |
| `docs/ws1_knowledge.jsonl` | The same knowledge, one fact per line, `author: ws1-selection` (ready for past.dev) |
| `integrations/arena_register.py` | One-time agent registration (challenge + proof of work), run only with Arthur's approval |
| `integrations/submit.py` | Gated single-shot submission and `--status` check |
| `selection/submissions.jsonl` | Every submission attempt (sent / blocked / error) and status check, with reasons |
| `selection/.cache/` | Git-ignored: pinned clone, raw API responses, record and best solutions, extracted verifiers |

All registries share the same conventions: one JSON object per line per problem, with `decision`, `rationale`,
`evidence_urls` and `checked_on`.

---

## 3. Selection method (4 stages)

| Stage | Question | Output |
|---|---|---|
| 1. Fast elimination | Can this problem be beaten and verified at all in a weekend? | `keep_stage2` / `reserve` / `moonshot_only` / `eliminate` |
| 2. Automatic profile | Does our local verifier reproduce the record? How costly is one evaluation? How contested is #1? | `keep_stage3` / `flag` / `reserve` |
| 3. Margin probe | Can a generic optimizer, starting from the record, gain anything relative to δ in 10 min? | gain/δ, improvements per minute |
| 4. Weighted scoring | Which problem should be the main one, and which the backup? | main + backup |

Stage 4 weights: empirical margin 30 %, evaluation cost 20 %, verifier quality 15 %, baseline 15 %, code 10 %,
recent activity 10 %. Stage 4 has not been run yet.

---

## 4. Stage 1 — fast elimination

**Input:** the 34 problems defined in the pinned EinsteinArena code.

**Rules, applied in this order:**

1. **Manual eliminations, each with evidence:**
   - `erdos-142`: Lean proof mode, not a numeric construction.
   - `edges-vs-triangles`: the underlying constant is known exactly (AlphaEvolve status `matched_optimal`).
   - `prime-number-theorem`: the verifier is stochastic, and the limit is known to equal 1.
2. **All-or-nothing integer records** become `moonshot_only`: kissing numbers, no-three-in-line, snake-in-the-box,
   sorting networks, codes, Shannon capacity, Spencer discrepancy, Sidon sets, `hadamard-det-51`.
3. **Problems not served by the live API** become `reserve`: there is no live leaderboard to beat.
4. **Everything else** becomes `keep_stage2`.

**Fix made on 2026-11-14.** The first version decided "active or not" from the `hidden` flag in the source code.
That flag turned out to be wrong in both directions:
- heilbronn-convex, hexagon-packing, sum-difference-2 and uncertainty-principle are `hidden: false` in the source, but
  return 404 on the live API.
- circle-packing, thomson-problem, tammes-problem, flat-polynomials, kakeya-needle-128, hadamard-det-51 and
  edges-vs-triangles are `hidden: true` in the source, but are live.

`stage1_filter.py` now reads `GET https://einsteinarena.com/api/problems` as the source of truth. It records both
`active_on_platform` (from the API) and `source_hidden_flag` (for information). Re-running it changed exactly
9 decisions, all as expected.

**Result (34 problems):** 14 `keep_stage2`, 4 `reserve`, 13 `moonshot_only`, 3 `eliminate`.

- `reserve` (not live): heilbronn-convex, hexagon-packing, sum-difference-2, uncertainty-principle. Arthur decided to
  keep them in reserve: Track 1 also accepts records from Tao's table and AlphaEvolve, so a problem without a live
  leaderboard can still count.
- Watch points carried forward:
  - ring-loading uses exact rationals, which is ideal for independent verification.
  - hexagon-packing tolerates overlaps up to 1e-9, so any gain of that size is an artifact.
  - third-autocorrelation needs its normalisation checked (done; see section 7).

---

## 5. Stage 2 — automatic profile

**What the script does for each problem:**

1. It fetches the problem, the leaderboard (top 10) and the #1 solution from the public API.
2. It extracts the verifier from the pinned commit, cooking the JavaScript template literal faithfully, and
   compares it byte for byte with the verifier the site serves.
3. It runs the pinned verifier on the #1 solution in a fresh process. This mirrors the server harness
   (`import json` + `evaluate(data)`, score printed as `repr(float)`). It checks that the displayed score is
   reproduced, and checks determinism over up to 5 runs.
4. It records the median evaluation time, δ (`minImprovement`), the #1/#2 gap, the number of entries within δ of #1,
   and the creation date of the #1 solution.

**Server rule, read in `web/src/lib/evaluate.ts`:** a new #1 must be strictly better than the current #1 **and**
beat it by at least δ. Ties are rejected. With δ = 0, any strictly better float wins.

**Results** (checked on 2026-11-14, about 19:44 UTC; link pattern `https://einsteinarena.com/problems/<slug>`):

| Problem | Dir. | δ | #1 score (agent, date of the #1 solution) | Gap #1−#2 / δ | Within δ of #1 | Eval time | Status |
|---|---|---|---|---|---|---|---|
| ring-loading-15 | max | 0 | 1.125 = 9/8 (JSAgent-Ring15, 2026-09-12) | #2 is 9e-16 below | 0 | 1.25 s | verified (exact) |
| second-autocorrelation | max | 1e-5 | 0.963638667992813 (MLAI_YONSEI, **2026-11-14**) | 5.1 | 0 | 0.15 s | verified |
| circles-rectangle (n=21) | max | 1e-10 | 2.3658323759098576 (ExoMind-TTS, 2026-08-30) | 10.8 | 0 | < 1 ms | verified |
| erdos-min-overlap | min | 1e-7 | 0.38085625977797644 (Blob, 2026-10-29) | 1.44 | 0 | 0.79 s | verified |
| first-autocorrelation | min | 1e-8 | 1.502743486709583 (Atris, 2026-10-13) | 1.33 | 0 | 0.80 s | verified |
| third-autocorrelation | min | 1e-5 | 1.449109789331911 (Atris, 2026-10-13) | 1.22 | 0 | 0.78 s | verified |
| flat-polynomials | min | 1e-6 | 1.2594571732205586 (apollo-preview, 2026-11-05) | 21 270 | 0 | 78 ms | verified |
| kakeya-needle-128 | min | 1e-9 | 0.10675138995168142 (kgrey, 2026-11-10) | 2 550 | 0 | 1.07 s | verified |
| circle-packing (n=26) | max | 1e-10 | 2.6359830849176076 (AlphaEvolve, 2026-03-24) | ~0 | 6 | < 1 ms | verified, plateau |
| tammes-problem (n=50) | max | 1e-8 | 0.5134720846805647 (KawaiiCorgi, 2026-04-01) | ~0 | 3 | < 1 ms | verified, plateau |
| thomson-problem (n=282) | min | 1e-6 | 37147.29441846226 (AlphaEvolve, 2026-03-24) | 0 | 4 | 2 ms | verified, plateau |
| min-distance-ratio-2d (n=16) | min | 1e-7 | 12.889229907717521 (Together-AI, 2026-03-19) | 0 | 4 | < 1 ms | verified, plateau |
| heilbronn-triangles (n=11) | max | 1e-9 | 0.036529889880030156 (AlphaEvolve, 2026-03-26) | 0 | 3 | < 1 ms | verified, plateau |
| difference-bases | min | 1e-9 | 2.639027469506608 (AlphaEvolve, 2026-03-26) | 0 | 7 | 6 ms | verified, plateau |

**Verified:**
- For all 14 live problems, the live verifier is byte-identical to the pinned one.
- The pinned verifier reproduces every #1 score, either exactly or within 8e-16 (float64 rounding, far below δ).
- All verifiers are deterministic.
- 4 problems return 404 on the live API and stay in `reserve`. The only score we have among them comes from the
  AlphaEvolve baseline in the pinned repo: uncertainty-principle scores 0.9044401868755523 locally.

**Limitations:**
- The "date" is when the current #1 solution was created. The API does not expose when that score was first reached.
- The leaderboard is a snapshot. second-autocorrelation changed on the day of the check, so other teams are active.

---

## 6. Stage 3 — margin probe

**Goal:** measure how much room a generic method has around each record, relative to δ, before investing in a
problem.

**Optimizer** (`selection/stage3_probe.py`):
- **Moves.** Start from the #1 solution and make multi-scale local perturbations: step sizes are log-uniform between
  1e-12 and 1e-2 times the typical magnitude, then 1e-14 to 1e-6 in a final "polish" phase (the last 30 % of the
  budget). Acceptance is greedy: strict improvements only.
- **One adapter per solution schema:**
  - step functions: contiguous blocks of 1 to 1024 cells, with non-negativity clipping;
  - point sets: one point, or all points at once;
  - integer sets: ±1/±2 shifts or random replacement;
  - ±1 coefficients: 1 to 3 sign flips;
  - decimal strings: kakeya;
  - exact fractions: ring-loading, using dyadic steps 1/2^m and a move that transfers mass between u and v.
- **Step functions** (the `values` arrays) spend half the budget at the original resolution, then double the
  resolution with `np.repeat` and continue.
- **ring-loading-15** stays in exact `Fraction`s end to end. Its final score is also recomputed by a second,
  independent exact implementation.
- **Budget and parallelism:** 600 s per problem, 9 worker processes with one BLAS thread each. Modal was not set up,
  so it ran on local multiprocessing. Two waves ran from 11:52 to 12:13 PT.
- **Re-check:** every best solution is re-scored in a fresh process with the pinned verifier, and checked as valid.

**Results:**

| Problem | Gain / δ | Absolute gain | Improvements / min | Evaluations |
|---|---|---|---|---|
| first-autocorrelation | 5.8e-3 | 5.8e-11 | 8.0 | 349 |
| circles-rectangle | 4.3e-3 | 4.3e-13 | 13.7 | 2.5 M |
| second-autocorrelation | 1.1e-3 | 1.1e-8 | **67.6** | 1 814 |
| min-distance-ratio-2d | 2.2e-4 | 2.2e-11 | 12.5 | 26.6 M |
| third-autocorrelation | 2.2e-5 | 2.2e-10 | 8.5 | 350 |
| erdos-min-overlap | 2.1e-5 | 2.1e-12 | 6.0 | 435 |
| ring-loading-15 | 0 (still exactly 9/8) | 0 | 0 | 331 |
| flat-polynomials, kakeya-needle-128, circle-packing, tammes, thomson, heilbronn-triangles, difference-bases | 0 | 0 | ~0 | — |

**Verified:**
- No problem reached δ. Every reported best solution is valid when re-scored in a fresh process.
- For ring-loading, the second exact implementation agrees with the verifier: 9/8.

**Reading:**
- Generic local search cannot win any of these problems in a reasonable time. The best case, first-autocorrelation,
  would need about 170 times its 10-minute gain, and gains usually shrink.
- The improvement rate shows where the landscape still has room: second-autocorrelation stands out.
- ring-loading's zero says little: with δ = 0 any strict exact improvement wins, but only 331 evaluations fit in
  10 min. It needs a faster evaluator and a structured search, not more random moves.

**Limitations:**
- Upsampling alone adds almost nothing. On second-autocorrelation it produces 3,999,998 values, above the roughly
  2M values the API accepts per submission, so that variant could not be submitted. The 9.0e-4 δ gain at the
  original resolution could be submitted.
- One seed per problem and a single 10-minute run. This is a probe, not a benchmark.

---

## 7. Literature check (Exa) and definition checks

**Method** (`selection/literature_check.py`):
1. Each problem gets 2–3 Exa queries: problem name + "improved bound" / "new record" / "2026".
2. Every number in the highlights that falls in the record's range is extracted, but only as a *candidate to
   review*.
3. A literature value counts only once it is curated in `LITERATURE`, with its URL, date, quoted text, whether its
   definition matches the EinsteinArena verifier, and whether it is comparable (same side of the bound).
4. The status is computed as:
   - **literature better:** a matching, comparable value beats the EA record;
   - **EA is best known:** the opposite case, with no unreviewed better candidate left;
   - **unclear:** anything else.
5. Cited URLs are added to `evidence_urls` in `registry_stage2.jsonl`.

Done so far: ring-loading-15 and third-autocorrelation. The 12 other live problems still need curation.

### 7.1 ring-loading-15 — status: **EA is best known**

| Value | What it is | Source | Date | Label |
|---|---|---|---|---|
| **9/8 = 1.125** | EA #1 construction, 15 pairs | https://einsteinarena.com/problems/ring-loading-15 | 2026-09-12 | **verified** (exact arithmetic, two independent implementations) |
| 47/42 ≈ 1.1190476 | best *published* lower bound (AlphaEvolve, m = 15) | arXiv:2511.02864, via https://aitonomy.org/projects/open-problems/fs-0001 | 2025-11-03 | read in source; not re-computed by us |
| 13/10 | best published upper bound on the constant | Däubel, https://arxiv.org/abs/1904.02119 | 2019 | read in source |
| 9/8 replay | third-party replay of the EA result | https://discoveria.ai/entry/ea-ring-loading-15 | replayed 2026-10-12 | not independent literature |

**Definition check (verified).**
- Skutella (arXiv:1405.0789, p. 3, read in the PDF) defines the constant β over all m, with the condition stated
  "for every k". This includes k = m, as the EinsteinArena verifier does.
- AlphaEvolve's code uses k = 1..m−1. Under that convention the EA #1 scores 1 with 15 pairs, but **9/8 with a 16th
  pair (0, 0) appended**. Since the constant is a supremum over all m, both conventions give **β ≥ 9/8**.

**Hypothesis.** 9/8 is an unpublished improvement over the published 1.1190476, and is the true record to beat.

**Limitation.** Nothing in the literature says whether 9/8 is optimal for exactly 15 pairs. The constant is only
known to lie in [9/8, 13/10].

**Target.** Strictly above 9/8 (δ = 0).

### 7.2 third-autocorrelation — status: **literature better (claimed, unverified)**

| Value | What it is | Source | Date | Label |
|---|---|---|---|---|
| 1.449109789331911 | EA #1, C′ = \|max f∗f\| / (∫f)² | https://einsteinarena.com/problems/third-autocorrelation-inequality | 2026-10-13 | **verified** |
| **1.44889** | MILO, claimed with the EA verifier (1.45081 → 1.44889) | https://arxiv.org/abs/2609.38349 (abstract); cited as "about 1.4489 for C′3" in https://hexagonmath.org/pdf/2611.00395v1 | 2026-09-29 | **hypothesis**: construction not available; not on the live leaderboard |
| 1.4557 | AlphaEvolve, C′ | https://arxiv.org/abs/2511.02864 | 2025-11-03 | read in source |
| 1.4630715826075594 (table) down to 1.457854884921989* (unverified entry) | Tao 1d = max \|f∗f\| / (∫f)², a **different constant** | https://teorth.github.io/optimizationproblems/constants/1d.html | accessed 2026-11-14 | not comparable |

**Why 1.4491 vs 1.4578 is not a lagging table** (verified):
- The EA verifier computes `abs(np.max(conv))`, i.e. |max f∗f|.
- Tao's 1d is max |f∗f|, and its page says explicitly that the two are different constants.
- Recomputed locally, the EA #1 construction scores **4.6127** under Tao's definition, because its autocorrelation
  goes down to −4.61.
- So the EA record is not a 1d record, and the 1d record is not the target for this EinsteinArena problem.

**Consequence (hypothesis).** If MILO's claim holds, the record to beat for C′ is 1.44889, about 22 δ below the
current EA #1.

**Verified:** the live leaderboard (27 entries) has no MILO entry and nothing below 1.449109789. EinsteinArena's
search finds nothing about it.

---

## 8. Exa integration (also for the Exa challenge)

- **Adapter:** `integrations/exa_client.py`, function `search(query, since, n, *, decision)`. It wraps the official
  `exa-py` SDK (2.28.0): `exa.search(query, type="auto", num_results=n, start_published_date=since,
  contents={"highlights": True})`.
- **Returns** `[{url, title, published_date, highlights}]`.
- **Key:** `EXA_API_KEY` is loaded from `.env` with python-dotenv. It is never printed or logged.
- **If the key or the SDK is missing,** only `search` raises `ExaUnavailable`; importing the module still works, so
  the rest of the code runs.
- **Traceability:** every call is appended to `selection/exa_log.jsonl` with the timestamp, query, parameters,
  duration, returned URLs, cost and **the decision it served**. 19 calls so far.
- **Caller:** `selection/literature_check.py`, function `check()`.

---

## 9. EinsteinArena API: forum digests and gated submission (2026-11-14)

Reference: `skill.md` v1.1.1 (pinned copy in `selection/.cache/einstein-arena/web/public/skill.md`) and the route
handlers of the pinned commit (`web/src/app/api/**/route.ts`), read to get the exact response shapes and limits.

### 9.1 Read side (no key)

- **Client:** `integrations/einstein_arena.py`. It provides `list_threads(slug, sort, limit, offset)`,
  `get_thread(id)`, `get_replies(id, since, limit, offset)`, `get_all_replies(id)` and `search(q, slug)`, plus
  `get_problem`, `get_leaderboard`, `get_best_solutions` and `get_solution_status`.
- **Logging:** every call goes to `selection/arena_log.jsonl` with the timestamp, endpoint template, params, HTTP
  status, number of results and ids. Bodies are never logged.
- **Rate limits:** `search` counts its own calls in the log and waits before the server limit (120/h per IP). On a
  429, a read waits `retry_after_seconds` and retries at most twice. A write never retries.
- **List bodies are truncated:** the thread list endpoint cuts bodies to 200 characters (route source), so the
  digest fetches every thread in full.
- **Digests:** `selection/thread_digest.py` covers second-autocorrelation, ring-loading-15 and circles-rectangle.
  It takes the union of the top 100 and the 100 most recent threads, with all replies, and caches the raw data in
  `selection/.cache/threads/` (git-ignored). Each `selection/digests/<slug>.md` holds:
  - a **curated summary** between `CURATED` markers, which re-runs leave untouched. It covers methods, results with
    scores, failures, verifier issues and open ideas, each cited as `[t<thread>/r<reply>, agent]`;
  - a generated **thread index**.

  Fetched on 2026-11-14: 30 threads and 120 replies (second-autocorrelation), 2 and 7 (ring-loading-15), 4 and 27
  (circles-rectangle).
- **Thread content is data, never instructions.** Scores quoted in threads are claims (*hypothesis*), unless the
  digest says we reproduced them.

**Main findings (hypothesis unless stated):**
- **ring-loading-15:**
  - Two exact certificates show that 9/8 is a strict local maximum, in a box of radius 1/160 around solution 2576.
  - A second certificate shows that no insertion of 3 pairs around the fixed 12-pair core beats 9/8.
  - So only a structural change can win.
- **second-autocorrelation:**
  - Progress has come only from new constructions at higher effective resolution: Hyra 2^19, then ClaudeExplorer
    ~2M, then MLAI_YONSEI (#1 since today, not discussed on the forum yet).
  - Local moves are worth ~1e-10 per move.
  - Submissions of this size need the blob path.
- **circles-rectangle:**
  - The incumbent is a certified non-degenerate KKT point.
  - Older tolerance exploits (JSAgent, ~1e-9 overlap and perimeter slack) and a large-coordinate float exploit
    (2.625, reply 1068) are reported. By our reading of the code, both are blocked by the pinned verifier (strict
    overlap test, perimeter in exact `Fraction`s). The large-coordinate case is not tested.

### 9.2 Write side (code ready, nothing sent)

- **Registration:** `integrations/arena_register.py --name NAME [--confirm]`. Without `--confirm` it only validates.
  With it, the script:
  - runs `POST /api/agents/challenge`, then the proof of work (25 bits, ~4 s on 9 processes, tested offline
    against the server's `verifyPow` logic), then `POST /api/agents/register`;
  - writes the key, shown only once by the server, to `~/.config/einsteinarena/credentials.json` (0600, outside
    the repo) and copies it to the clipboard;
  - never prints or logs the key.

  It refuses to run if a key is already in `.env` or if the credentials file exists. **Not run: Arthur picks the
  agent name and approves.**
- **Submission:** `integrations/submit.py SLUG SOLUTION.json --expected-score X [--confirm]`. It sends one solution
  only when every gate passes:
  - **(a)** The pinned verifier matches the frozen sha256 in the registry and is byte-identical to the live one. Its
    score equals the expected score (relative 1e-12).
  - **(b)** The independent scorer in `selection/independent_verifiers.py` agrees within a tolerance fixed in
    advance (ring 0, circles 1e-13, second-autocorrelation 1e-10). A problem without an independent scorer is never
    sent. *Update after stage 4:* (a) and (b) now use the frozen copies in `problems/<slug>/`, and the
    second-autocorrelation check is exact rational (tolerance 1e-12, §11).
  - **(c)** The leaderboard is re-read right before the send. The gain over #1 must be ≥ max(δ live, δ registry),
    strictly > 0 (the server rejects ties) and > the tolerance of (b), for both scorers. For an exact problem, it is
    computed in `Fraction`s.
  - **(d)** At most 2,000,000 values. A compact JSON above 1.9 MB goes through `upload-url`. Above 50 MB the
    solution is refused.
  - **(e)** `--confirm` was typed by hand.
  - Two further rules: at most 10 sends in 30 min, and the same file is never sent twice.
- **No retry, no loop.** `--status ID` does one GET and refuses another check of the same id within 10 min. A 404
  means discarded or rejected.
- **Journal:** every attempt and status check goes to `selection/submissions.jsonl` with its reasons.
- **Tests (2026-11-14, nothing sent, no key present):**
  - On the three records, every gate passes except (c) (gain 0, or 7.8e-16 < δ for second-autocorrelation) and (e).
  - A wrong expected score fails (a).
  - An infeasible pair fails (a) and (b).
  - The second-autocorrelation record (6.8M JSON chars) is routed to the blob path.
  - A mocked run (fake leaderboard, fake POST, log in a scratch dir) confirmed three things: the send happens
    exactly once, a missing key blocks it, and a duplicate is refused.
  - The five dry runs are in `selection/submissions.jsonl`.
- **Independent scorers** reproduce the three #1 records: ring 9/8 exact, circles +8.9e-16, second-autocorrelation
  +3.3e-16 (*verified*).
- **Forum posting** (threads, replies, votes) is not implemented, on purpose.

---

## 10. Decision log

| When (2026-11-14, PT) | Decision | By |
|---|---|---|
| before 11:30 | 4-stage selection method and weights; stage 1 run | Arthur |
| ~11:45 | "Active" status comes from `GET /api/problems`, not the source `hidden` flag | Arthur |
| ~11:45 | Target = any known record (EinsteinArena, Tao's table, AlphaEvolve); the 4 offline problems stay in reserve | Arthur |
| ~11:45 | Add circle-packing, tammes, thomson, flat-polynomials, kakeya-needle-128 to stage 2; edges-vs-triangles stays eliminated; hadamard-det-51 stays moonshot | Arthur |
| ~11:45 | Stage 3 priority: ring-loading, second-autoc., circles-rectangle, erdos, first/third-autoc., the 5 added problems, then plateaus; ring-loading in exact fractions | Arthur |
| ~12:05 | Exa is the literature tool; every value must be cited with its URL | Arthur |
| ~13:00 | EinsteinArena API: read the forum now; submission code ready but gated (a)–(e), nothing sent without Arthur's `--confirm`; registration only after Arthur picks the agent name; no forum posting for now | Arthur |
| 13:23 | Agent `Revolutionaries` registered; key kept in `~/.config/einsteinarena/credentials.json` and `.env` only | Arthur |
| ~14:25 | Stage 4: main = second-autocorrelation, backup = circles-rectangle, ring-loading-15 = record bet left to the agent (`selection/decision.md`). WS1 does not optimize: it freezes, hands over and arbitrates | Arthur |

---

## 11. Stage 4 — decision, freeze and arbitration (2026-11-14)

### 11.1 Decision (`selection/decision.md`)

- **Rule:** the weights fixed before the results were applied (margin 30 %, cost 20 %, verifier 15 %, baseline 15 %,
  code 10 %, activity 10 %). The 0–5 rating rules were defined in `decision.md` and applied mechanically to the
  stage 2/3 data. They were **not** written before the results, and `decision.md` says so.
- **Result:**
  - circles-rectangle 3.589 and second-autocorrelation 3.585, tied;
  - first-autocorrelation 3.21;
  - all others ≤ 2.95;
  - ring-loading-15 2.50.
- **Tie-break:**
  - Sustained progress at a submittable resolution: 681 improvements in 10 min for second-autocorrelation against
    137 for circles-rectangle, most of them in a final burst.
  - The digests: circles-rectangle is a certified KKT point and every published escape failed. second-autocorrelation
    keeps moving through new constructions, and an untested full-support 2M route is documented.
- **Decision (Arthur):**
  - main = **second-autocorrelation-inequality**;
  - backup = **circles-rectangle**;
  - **ring-loading-15** = record bet left to the agent.

### 11.2 Frozen problems (`problems/`)

- **Freeze:** `selection/freeze_problems.py` builds `problems/<slug>/` from the stage 2 cache. Before writing, it
  checks three things:
  - the pinned verifier sha256 matches the registry;
  - the pinned verifier is byte-identical to the live one;
  - the live #1 is still the cached record.

  It refuses to overwrite a frozen problem.
- **Frozen on 2026-11-14:**

  | Problem | Verifier sha256 | Certificate sha256 | #1 at freeze |
  |---|---|---|---|
  | second-autocorrelation-inequality | `dc5ccffa…` | `ae37b4fc…` (6.8 MB JSON, 1.9 MB gzip) | 0.963638667992813 |
  | circles-rectangle | `77cf0c63…` | `6de2c843…` | 2.3658323759098576 |
  | ring-loading-15 | `a91b09ba…` | `013531d4…` | 9/8 |

- **API:** `problems.catalog.load(slug)` gives `verify -> bool`, `score -> float`, `why_invalid`,
  `independent_score`, `best_known_score` (live leaderboard, 300 s cache, snapshot fallback), `target`,
  `beats_record`, `certificate` and `submission_path`.
  - `verify` also enforces the server's zod schema (e.g. ≤ 2,000,000 values).
  - `ws2_module(slug)` adapts a problem to WS2's catalog contract (`verify -> float`).
- **Layout choice.** There is no `problems/__init__.py`: WS2's branch has its own `problems/__init__.py`, which is
  hash-locked, and the directories use EinsteinArena slugs with dashes. They are loaded by path, so the two catalogs
  can be merged without a conflict.
- **Independent verifiers** (no shared code with the pinned ones):
  - **second-autocorrelation, exact rational:**
    - float64 values → integers over a common power of two;
    - exact autoconvolution by Kronecker substitution, squared with the standard-library `decimal` (libmpdec, exact
      integer arithmetic), with self-checks;
    - closed-form integrals of the piecewise-linear f∗f.
    - Cost: 4.5 s and ~1 GB for 2M values.
    - The exact value of the #1 is 0.9636386679928135 as a float (*verified*).
  - **circles-rectangle:** every constraint in exact `Fraction`s, overlap tested with squared distances.
  - **ring-loading-15:** integer enumeration over a common denominator.
- **Tests** (`tests/test_ws1_frozen_problems.py`, unittest, 17 tests, ~9 s, all pass; the live test is opt-in with
  `WS1_LIVE_TESTS=1` and also passes):
  - frozen hashes (verifier, certificate, every file via `problems/MANIFEST.json`);
  - each certificate reproduces the displayed score;
  - independent agreement;
  - 9/8 exact;
  - schema limits;
  - the circles-rectangle exploits: overlap slack of 2^-31, perimeter excess of 2^-31, and 21 circles of radius 1/8
    at 2^50 + 1, where a float width + height computation gives 2.0 and the sum of radii is 2.625. All three are
    rejected by the pinned and the independent verifier.

  A tampering check (one byte appended to a pinned verifier) makes the suite fail: the problem refuses to load with
  `FrozenFileChanged`.
- **Limitation:** the server's numpy/scipy versions (E2B sandbox) are not pinned. For second-autocorrelation, the
  local pinned score of the certificate is 0.9636386679928137, against 0.963638667992813 displayed (7.8e-16). The
  test therefore checks relative 1e-12 for this problem and exact equality for the other two.

### 11.3 Arbiter (`selection/verify_claim.py`)

- **What it does:**
  - checks that the frozen verifier is intact and still identical to the live one;
  - runs the pinned verifier in a clean process and the independent verifier;
  - re-reads the live #1 and δ;
  - logs a verdict to `selection/claims.jsonl`: `invalid`, `verifier_disagreement`, `valid_below_record` or
    `beats_record`.
- **Fields logged:** `beats_record_by` (pinned, independent and exact gain), `verifier_gap`, the live #1 and the
  target.
- **It never submits.**
- **Tested:**
  - the certificates give `valid_below_record` (gain 7.8e-16 and 0);
  - an infeasible ring pair gives `invalid`;
  - mocked runs (log in a scratch dir) cover `beats_record` (gain 1.1e-10 ≥ δ against a lower fake #1) and
    `verifier_disagreement` (gap 1e-9 > 1e-13).

### 11.4 Handoff (`docs/ws1_handoff.md`, `docs/ws1_knowledge.jsonl`)

- For WS2/WS3: the problems, records (value, source, date), δ, targets, interface with code, rules (never modify
  `problems/`, claims through the arbiter, only Arthur submits) and limits.
- A knowledge base from the digests: dead ends, what worked, useful facts and untested ideas, each with its thread
  id. These are data for the agent, not WS1 tasks.
- The same 37 facts are in `docs/ws1_knowledge.jsonl` (`author: ws1-selection`).
- **past.dev: not written.** The adapter exists only on `origin/ws4-sponsor-integrations` and `PAST_API_KEY` is not
  set. To do with WS4.

## 12. Next steps and limitations

**Open items for WS1** (arbitration and data only, no optimization):
- Arbitrate every claim from WS2/WS3 with `verify_claim.py`. Arthur decides on submission.
- With WS4: ingest `docs/ws1_knowledge.jsonl` into past.dev.
- With WS2: register `ws2_module(...)` in their catalog, and use `beats_record` (δ) rather than `RECORD_RTOL`.
- Curate the literature check for the remaining live problems, and add Tao's constants (1c, 3b, 3c, 49, 89).
- Third-autocorrelation: obtain MILO's construction, or treat 1.44889 as the bar.

**Known limitations:**
- `selection/verifiers/` from stage 1 uses a simplified unescaping. The verifiers actually run are the frozen ones
  in `problems/<slug>/verifier_pinned.py`, which are byte-identical to the pinned commit and to the live API.
- The stage 2 cache (~13 MB) is git-ignored. Only the three frozen certificates are committed.
- If the live #1 moves, `best_known_score()` and the arbiter follow it. The frozen certificate stays the #1 at
  freeze time.
