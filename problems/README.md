# problems/ — frozen EinsteinArena problems (WS1)

**Freeze rule: agents (WS2) and optimizers (WS3) never modify anything in this directory.** They only change the
search strategy. Any change to a frozen file makes `tests/test_ws1_frozen_problems.py` fail. Only WS1 may re-freeze
a problem, with Arthur's approval (`selection/freeze_problems.py --refreeze`).

| Directory | Role | Live #1 at freeze (2026-11-14) | δ |
|---|---|---|---|
| `second-autocorrelation-inequality/` | **main** | 0.963638667992813 (MLAI_YONSEI) | 1e-5 |
| `circles-rectangle/` | **backup** | 2.3658323759098576 (ExoMind-TTS) | 1e-10 |
| `ring-loading-15/` | record bet | 9/8 (JSAgent-Ring15) | 0 (strict) |

Why these three: `selection/decision.md`.

## Interface

```python
from problems.catalog import load

p = load("second-autocorrelation-inequality")
p.verify(candidate)          # bool: server schema + pinned verifier accept it
p.score(candidate)           # float: pinned verifier score; raises p.InvalidSolution
p.why_invalid(candidate)     # str | None
p.best_known_score()         # {"value", "agent", "live", "source", "read_at"}: LIVE #1 (cached 300 s)
p.target()                   # live #1 + δ (the score a claim must reach)
p.beats_record(s)            # gain >= δ and gain > 0 (ties never win)
p.independent_score(c)       # second implementation: {"score", "fraction", "tolerance_abs", "method"}
p.certificate()              # the frozen #1 solution (sha256-checked)
p.submission_path(c)         # "inline" or "blob" (> ~2 MB of JSON)
p.DIRECTION, p.DELTA, p.LIMITS, p.SOLUTION_FORMAT, p.VERIFIER_SHA256
```

To use the problems from WS2's catalog contract (`verify -> float`), use `problems.catalog.ws2_module(slug)`.

## Content of each problem directory

| File | What it is |
|---|---|
| `verifier_pinned.py` | The EinsteinArena verifier, byte-identical to `github.com/vinid/einstein-arena` @ `5baf636` and to the live verifier on 2026-11-14. It runs in the server's harness (`import json` prefix, `evaluate(solution)`). |
| `meta.json` | id, direction, δ, limits, verifier commit and sha256, record snapshot, certificate sha256 |
| `certificate.json.gz` | The live #1 solution at freeze time. The sha256 of its decompressed bytes is in `meta.json`. |
| `independent.py` | A second implementation with no shared code: exact rational for second-autocorrelation (no float FFT), exact `Fraction` feasibility for circles-rectangle, integer enumeration for ring-loading-15 |
| `problem.py` | The public API above, plus `check_schema` (the server's zod schema) |

`MANIFEST.json` holds the sha256 of every WS1 file here. Claims are arbitrated by `selection/verify_claim.py`, never
by an agent.
