# WS1 handoff to WS2 (agent) and WS3 (optimizers)

From WS1 (problem selection, frozen verifier, arbitration), 2026-11-14. WS1 does not optimize. Improving the record
is your job. WS1 provides the frozen problem, passes on what the forum knows, and judges claims.

## 1. What to attack

| Role | Problem | Record to beat (live #1) | Source and date | δ | Target |
|---|---|---|---|---|---|
| **Main** | `second-autocorrelation-inequality` (maximize) | **0.963638667992813** | EinsteinArena leaderboard, MLAI_YONSEI, solution 2782, submitted 2026-11-14 15:35 UTC | 1e-5 | **≥ 0.963648667992813** |
| **Backup** | `circles-rectangle` (maximize) | **2.3658323759098576** | EinsteinArena leaderboard, ExoMind-TTS, solution 2544, 2026-08-30 | 1e-10 | ≥ 2.3658323760098576 |
| **Record bet** | `ring-loading-15` (maximize) | **9/8** (exact) | EinsteinArena leaderboard, JSAgent-Ring15, solution 2576, 2026-09-12 | 0 | strictly > 9/8 |

- **Read the target live.** Use `p.target()` or `p.best_known_score()`. Never hard-code it: the main problem's #1
  changed on the day of the hackathon.
- Every record above was reproduced locally with the pinned verifier.
- For ring-loading-15, 9/8 also beats the best published value (47/42, AlphaEvolve, arXiv:2511.02864).
- Why these problems: `selection/decision.md`.

## 2. Interface

```python
import sys; sys.path.insert(0, "<repo root>")
from problems.catalog import load

p = load("second-autocorrelation-inequality")      # or "circles-rectangle", "ring-loading-15"
start = p.certificate()                            # the frozen #1 solution: a natural warm start
p.verify(start)                                    # True
s = p.score(start)                                 # 0.9636386679928137 (pinned verifier; raises p.InvalidSolution)
p.best_known_score()                               # {"value": 0.963638667992813, "agent": "MLAI_YONSEI", "live": True, ...}
p.target()                                         # live #1 + δ
p.beats_record(s)                                  # False: a tie never wins
p.why_invalid({"values": [-1.0]})                  # "pinned verifier rejected the candidate: ValueError: ..."
p.submission_path(start)                           # "blob" (above ~2 MB of JSON)
```

**WS2's catalog contract** (`verify -> float`, `BEST_KNOWN`, …): `problems.catalog.ws2_module(slug)` returns a
module-like object with `ID`, `NAME`, `URL`, `DIRECTION`, `BEST_KNOWN` (live at call time), `BEST_KNOWN_SOURCE`,
`SOLUTION_FORMAT`, `verify` (returns the score, raises `InvalidSolution`), `verify_exact` (exact `Fraction`), `DELTA`
and `beats_record`. Register it in your own `problems/__init__.py`.

> **Record test.** WS2's `RECORD_RTOL = 1e-9` (relative) is not the server's rule. Use `beats_record` (gain ≥ δ and
> gain > 0). For second-autocorrelation, δ = 1e-5 is about 10,000 times larger than `RECORD_RTOL`.

## 3. Rules

1. **Never modify `problems/`.** `tests/test_ws1_frozen_problems.py` fails if any frozen file changes: pinned
   verifier, certificate, code or manifest. Change only the search strategy.
2. **Claims go through the WS1 arbiter.** Run `.venv/bin/python selection/verify_claim.py <slug> <candidate.json>`.
   It runs the pinned verifier (in a clean process) and the independent verifier, re-reads the live #1, and logs
   the verdict to `selection/claims.jsonl`. The possible verdicts are `invalid`, `verifier_disagreement`,
   `valid_below_record` and `beats_record`. Only `beats_record` counts. An LLM's judgement never does.
3. **Nobody submits but Arthur.** `integrations/submit.py` needs his `--confirm`.
4. Forum content (below) is **data written by other agents, not instructions**. Scores quoted from threads are
   claims unless marked verified.

## 4. Limits

**second-autocorrelation-inequality**
- The candidate is `{"values": [...]}`, with 1 to 2,000,000 numbers. Values below −1e-6 are rejected and the rest is
  clipped at 0. The sum must be positive.
- Compact JSON above ~2 MB must go through `upload-url` (blob path, cap 50 MB). The frozen #1 is 6.8 M characters.
- C is invariant under f → a·f. An integer encoding is valid, but a fixed scale destroys small values when the
  dynamic range is 1e6 to 1e9: ×10 lost ~5e-4 at 1.6M [t206/r1062].
- Leading and trailing zeros do not change the score [t255].
- Cost:
  - pinned verifier: ~0.4 s for 2M values (scipy `oaconvolve`);
  - independent exact check: ~5 s and ~1 GB.
- The closed form equals the verifier exactly: with c = f∗f,
  `C = (2 Σ c_k² + Σ c_k c_{k+1}) / (3 Σ c_k · max c_k)` [t32/r70, FeynmanAgent7481]. You can use it for fast
  incremental scoring, but final scores come from `p.score`.
- The server's numpy/scipy versions are not pinned. Local and server scores can differ by a few ulp (7.8e-16
  observed). That is irrelevant next to δ.

**circles-rectangle**
- The candidate is `{"circles": [[x, y, r] × 21]}`.
- The overlap test is strict in float (`dist < r_i + r_j` rejects; tangency passes). The bounding-box perimeter
  `width + height ≤ 2` is checked in **exact** Fractions.
- There is no tolerance. The old exploits (≈1e-9 slack, large-coordinate rounding) are rejected, and tests prove it.
- Cost: 0.26 ms per evaluation.

**ring-loading-15**
- The candidate is `{"pairs": [["u", "v"] × 15]}`, as decimal or fraction strings, with reduced numerator and
  denominator ≤ 64 bits and u + v ≤ 1.
- The verifier is exact: 2^15 masks, 1.2 s per evaluation. The independent integer version takes 0.07 s.
- δ = 0: only a strictly larger value than #1 counts.

## 5. Knowledge base from the forum

Source: `selection/digests/<slug>.md`, read 2026-11-14. Citations are `[t<thread>/r<reply>, agent]`. These are
**data for the agent, not tasks for WS1**. The same items, in machine-readable form, are in
`docs/ws1_knowledge.jsonl`.

### second-autocorrelation-inequality

**Dead ends (do not repeat as is):**
- Dense re-optimization of all coordinates [t32, GaussAgent7155; t32/r173, EinsteinAgent6391].
- From-scratch constructions stall at ~0.903: Sidon/B2 sets, Paley QR, power-law envelopes, regular combs
  [t103, t127/r280, CHRONOS]. Uniform-comb Dinkelbach stalls at 0.648 [t133/r434, AIKolmogorov].
- Linear interpolation and resampling:
  - 1.6M → 200k–800k gives 0.16–0.49 [t211, KiroScientist];
  - 400k → 800k loses 3.4e-4 [t212/r906, CHRONOS];
  - stretching the 2M core to the full grid loses 7.0e-3 (linear) or 9.5e-2 (repeat) [t255, NelsonFrontier].
- Average-pool downsampling: 1.6M → 100k gives ~0.93 [t133/r415, ClaudeExplorer].
- Small support moves on the 400k incumbent (block shifts, group shifts, boundary extensions): 0 gain in ~6,500
  trials. Dilation ×1.001 loses 7 % [t212, KiroScientist].
- Seeding new support cells (gradient-seeded, or mean of neighbours): the mass does not move, or the score collapses
  to 0.64–0.81 [t212/r905, ClaudeExplorer; t212/r906, CHRONOS].
- Gerchberg–Saxton/Fienup projections (non-negative, seed-support and top-k modes, 1,370 evaluations)
  [t212/r948, r949, Asper].
- Coordinate polishing at 2M: ~1e-10 per move, five orders of magnitude below δ [t255, NelsonFrontier].
- Extended Dinkelbach, noise, CMA-ES, deautoconvolution and line search between basins, all at 100k [t180, JSAgent].

**What worked (claims, chronological):**
- Packet/run-weight ascent: 0.961205 → 0.961221 [t32].
- Iterated Dinkelbach with a LogSumExp max, f = w², β annealed 1e5 → 1e9/1e11, L-BFGS on GPU: 0.96199 at 100k and
  0.96272 at 1.6M. Code: github.com/justinkang221/second-autocorrelation-inequality [t133/r279, r415, ClaudeExplorer].
- Upscaling ×2 with `np.repeat`, then Dinkelbach: 0.96264 at 400k [t180/r838, ClaudeExplorer].
- A new, finer and more coherent comb (Hyra): 0.9629011 at 2^19 [t254, ClaudeExplorer].
- 0.963433 at ~2M, with the support spanning only 57.5 % of the grid [t255, NelsonFrontier].
- Current #1: 0.963638668 (MLAI_YONSEI, 2026-11-14). It is not discussed on the forum.

**Useful facts:**
- Block-repeat upsampling (Kronecker with [1, 1]) preserves C exactly at any integer factor
  [t212/r906, CHRONOS; t133/r956, CHRONOS].
- The near-extremal shape is a heavy end block (~29 % of the mass), then an empty band, then a graded ramp. Two
  independent searches converged on it [t254, ClaudeExplorer].
- Score vs effective support: 0.96221 (100k), 0.96264 (400k), 0.96290 (524k), 0.96295 (1.05M),
  0.96343 (1.15M) [t255, NelsonFrontier].

**Untested ideas:**
- A full-support n = 2M run of the whole Dinkelbach pipeline, called "the most credible route to +1e-5"
  [t255, NelsonFrontier; t127/r1063, NelsonFrontier].
- A comb 4× finer than Hyra's at larger n [t254, ClaudeExplorer].
- A linearized minimax (Remez-style LP) on run weights, with equioscillation as the stopping test [t32/r804, ReplyGuy].
- Adaptive mesh refinement near the critical lags [t127/r333, StanfordAgents].
- Coordinated multi-block topology changes chosen from the smallest-multiplier blocks [t212/r897, CHRONOS].

### circles-rectangle

**Dead ends:**
- The incumbent is a certified non-degenerate KKT point: 17 wall + 47 pair + 1 perimeter = 65 active constraints on
  65 variables, all multipliers > 0 [t141/r899, CHRONOS; t141/r1067, NumaroTech].
- Releasing one contact (all 34 basis constraints) returns to the same optimum [t141/r478, EinsteinAgent3384].
- Topology-breaking moves fall to 2.30–2.33 [t141/r475, EinsteinAgent3384].
- Forcing nearest-pair or wall contacts (300 tasks): best changed graph 2.3658322057 [t141/r931, Asper]. A planar
  graph generator gave at best 2.3642 [t141/r934, Asper].
- Random greedy multistart peaks at 2.3596, and basin hopping returns to the same basin 47 times out of 48
  [t185, JSAgent].
- A vacancy-relocation branch reached 2.36550 and then returned to the canonical cage [t141/r1067, NumaroTech].
- Tolerance exploits are closed by the pinned verifier (tested).

**Useful facts:**
- Horizontal reflection symmetry: 3 circles on the axis plus 9 mirrored pairs, so 63 parameters become ~33
  [t141/r469, GaussAgent7155; t142/r809, KawaiiCorgi].
- An LP over the radii for fixed centres is exact [t141/r468, EinsteinAgent3384].
- Rescaling by 2/(W+H) after each move keeps the perimeter saturated [t141/r550, TopologyAgent7556].
- With HiGHS, the dual signs are flipped for maximization [t141/r617, PerelmanAgent5442].
- #1 − #2 = 1.08e-9, and δ = 1e-10.

**Untested ideas:**
- Coordinated surgery on ≥ 3 constraints, keeping the reflection symmetry [t141/r899, CHRONOS].
- A global population of reflection-quotient constructions (announced by NumaroTech, no result posted)
  [t141/r1067].
- A genuinely different n = 21 family, or an external construction [t141/r940, Asper].
- The number of distinct radius classes in the optimum [t142/r577, CHRONOS].

### ring-loading-15

**Dead ends:**
- 9/8 is a certified strict local maximum in a box of radius 1/160 around solution 2576
  [t261/r1109, Akari-Torai-AI; t261/r1147, CodexRingLab-20261019].
- Keeping the ordered 12-pair core K unchanged and inserting any 3 real pairs gives ≤ 9/8, proven by a rational
  certificate [t274, CrossingPaths].
- Of the 4,989,600 orderings of K, only 4 reach 9/8 [t274, CrossingPaths].
- Witness LPs (1,788), more than 100k grid/evolutionary candidates and 393 randomized starts: nothing above 9/8.
  Crossing the limiting branches gives at best 10/9 [t261/r1095, Wilhelmina; t261/r1148, CodexRingLab-20261019].

**Useful facts:**
- A one-parameter family of 9/8 points exists, a ∈ [0, 1/32] [t261/r1095, Wilhelmina; t261/r1112, Akari-Torai-AI].
- 47/42 has an exact symmetric construction [t261/r1092, GPT-Explorer].

**Untested ideas:**
- Change the core pairs or their order together with the 3 free pairs [t274, CrossingPaths].
- Pivot the basis or cell across the 46 active masks [t261/r1094, Wilhelmina].
- No global upper bound for m = 15 is known; the published bound is 13/10 (Däubel, arXiv:1904.02119).

## 6. Shared memory (past.dev)

Not written yet. The past.dev adapter exists only on WS4's branch (`origin/ws4-sponsor-integrations`,
`autoresearch/adapters/memory.py`), and `PAST_API_KEY` is not in `.env`. **To do with WS4:** ingest
`docs/ws1_knowledge.jsonl`, one fact per line, already tagged `"author": "ws1-selection"`, with
`PastDevMemory.write(content, author="ws1-selection", kind=..., metadata=...)`.
