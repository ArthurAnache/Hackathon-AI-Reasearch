# Hackathon-AI-Reasearch

The Revolutionaries, SF AI Research Hackathon (14–15 Nov 2026), Track 1. An autonomous research loop that must
beat a known mathematical record. Workstream logs: `README1.md` (WS1, selection and verification),
`README2.md` to `README4.md` (other workstreams).

## Problem & verification

**Problem.** The main problem is EinsteinArena's
[second autocorrelation inequality](https://einsteinarena.com/problems/second-autocorrelation-inequality):

- **Objective:** maximize C = ‖f∗f‖₂² / (‖f∗f‖₁ · ‖f∗f‖∞) over nonnegative step functions f.
- **Format:** up to 2,000,000 values.
- **Backup:** [circles in a rectangle](https://einsteinarena.com/problems/circles-rectangle). Place 21 circles in a
  rectangle of perimeter 4 and maximize the sum of the radii.
- **Record bet:** [ring loading, 15 pairs](https://einsteinarena.com/problems/ring-loading-15), left to the agent.

Why these three: `selection/decision.md`.

**Record and source.** Values from the live EinsteinArena leaderboard, re-verified locally on 2026-11-14:

| Problem | Record | Holder | δ | Success threshold |
|---|---|---|---|---|
| second-autocorrelation | 0.963638667992813 | MLAI_YONSEI, solution 2782, 2026-11-14 | 1e-5 | ≥ live #1 + 1e-5 (today ≥ 0.963648668) |
| circles-rectangle | 2.3658323759098576 | ExoMind-TTS, solution 2544, 2026-08-30 | 1e-10 | ≥ live #1 + 1e-10 |
| ring-loading-15 | 9/8 (exact) | JSAgent-Ring15, solution 2576, 2026-09-12 | 0 | strictly > live #1 |

The record is always the **live** #1. Use `p.best_known_score()` and `p.target()`, never a hard-coded value.

**Frozen interface** (`problems/`, never modified by the agent):

```python
from problems.catalog import load
p = load("second-autocorrelation-inequality")
p.verify(candidate) -> bool        # server schema + pinned EinsteinArena verifier
p.score(candidate) -> float        # raises p.InvalidSolution
p.target(), p.beats_record(score), p.certificate(), p.independent_score(candidate)
```

- Each `problems/<slug>/` holds four frozen pieces:
  - the pinned verifier (`github.com/vinid/einstein-arena` @ `5baf636`, byte-identical to the live one, sha256 in
    `meta.json`);
  - the #1 solution as a certificate, with its sha256;
  - an independent verifier;
  - the limits.
- `tests/test_ws1_frozen_problems.py` fails if any of these changes.

**Independent verification.** A claim counts only after the arbiter runs:

```bash
.venv/bin/python selection/verify_claim.py second-autocorrelation-inequality candidate.json
```

1. The pinned verifier runs in a clean process, as on the server.
2. A second implementation that shares no code with it runs next:
   - second-autocorrelation: exact rational arithmetic (floats → integers, exact autoconvolution, closed-form
     integrals), with no floating-point FFT;
   - circles-rectangle: exact `Fraction` checks of every constraint;
   - ring-loading-15: integer enumeration.
3. The live #1 and δ are re-read.

**Success threshold:** both verifiers accept the candidate and agree within their tolerance, and
gain = score − live #1 satisfies gain ≥ δ and gain > 0 (ties never win). That is verdict `beats_record` in
`selection/claims.jsonl`. Submission is separate (`integrations/submit.py`) and needs Arthur's `--confirm`.

Handoff for the agent and the optimizers: `docs/ws1_handoff.md`.

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env    # keys are optional for reading; never commit .env
.venv/bin/python -m unittest discover -s tests -p "test_ws1_*.py"
```
