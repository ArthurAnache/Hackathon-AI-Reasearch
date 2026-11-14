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

## 9. Decision log

| When (2026-11-14, PT) | Decision | By |
|---|---|---|
| before 11:30 | 4-stage selection method and weights; stage 1 run | Arthur |
| ~11:45 | "Active" status comes from `GET /api/problems`, not the source `hidden` flag | Arthur |
| ~11:45 | Target = any known record (EinsteinArena, Tao's table, AlphaEvolve); the 4 offline problems stay in reserve | Arthur |
| ~11:45 | Add circle-packing, tammes, thomson, flat-polynomials, kakeya-needle-128 to stage 2; edges-vs-triangles stays eliminated; hadamard-det-51 stays moonshot | Arthur |
| ~11:45 | Stage 3 priority: ring-loading, second-autoc., circles-rectangle, erdos, first/third-autoc., the 5 added problems, then plateaus; ring-loading in exact fractions | Arthur |
| ~12:05 | Exa is the literature tool; every value must be cited with its URL | Arthur |

---

## 10. Current shortlist and next steps

**Shortlist going into stage 4** (still to confirm with the weighted scoring):

1. **second-autocorrelation:** the highest improvement rate in the probe, an active #1 (changed today), cheap
   evaluation (0.15 s) and a δ of 1e-5 that is reachable in principle.
2. **first-autocorrelation:** the best gain/δ in the probe, with δ = 1e-8.
3. **ring-loading-15:** exact verifier, δ = 0, and a record (9/8) that already beats the literature. Any strict
   exact improvement would be a new lower bound on a classical constant. It needs a faster exact evaluator and a
   structured search.
4. **Backup:** circles-rectangle (evaluation under 1 ms, δ = 1e-10).

**Open items:**
- Run the stage 4 weighted scoring and pick the main problem and the backup.
- Curate the literature check for the 12 remaining live problems.
- Freeze the verifier for the chosen problem behind `problem.verify` / `problem.score`, with an independent second
  implementation (exact or rational where possible).
- Add Tao's constants that are missing from EinsteinArena (1c, 3b, 3c, 49, 89) to the registry.
- Third-autocorrelation: try to obtain MILO's construction, or treat 1.44889 as the bar.

**Known limitations of WS1 so far:**
- `selection/verifiers/` from stage 1 uses a simplified unescaping. The verifiers that are actually run are the
  faithful extractions in `selection/.cache/stage2/<slug>/verifier_pinned.py`.
- The cache, including the record solutions (~13 MB), is git-ignored. Re-running stage 2 rebuilds it.
