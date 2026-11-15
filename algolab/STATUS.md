# Algorithm Lab integration — status (2026-11-14 evening PT)

Worktree: `C:\Users\thiba\Desktop\Berkeley\AIRESEARCHACK\savant-sci-audit`  
Active campaign `savant-exp1` (pid 39972): **untouched**. No commits / pushes / merges / paid Modal / paid LLM.

## Validation results

| Check | Result |
|---|---|
| `unittest discover -p test_algolab*.py` | **11/11 OK** |
| Gate admits LP / two-box / Dinkelbach–Kelley | **3/3 admitted** |
| Gate rejects hostile plugin | **5 reasons** |
| Runtime audit hook (network/subprocess/write/secret) | **all refused; no file escaped** |
| Kit score vs pinned verifier | **\|Δ\| = 1.1e-16** |
| Kit gradient FD error | **7.1e-9** |
| Modal image audit | **clean** |
| Modal spec `block_network` / `secrets=[]` | **True / []** |
| `live-record` default | **frozen_snapshot only (no network)** |
| Ladder plans T0–T2 | written under `runs/algolab/plans/20261114T205825/` (12 jobs/tier) |

## Safe next-run command (unpaid, this worktree)

```text
cd C:\Users\thiba\Desktop\Berkeley\AIRESEARCHACK\savant-sci-audit
python -m algolab.campaign admit-seeds
python -m algolab.campaign plan-ladder --seeds 101 202 303
python -m algolab.campaign live-record
python -m unittest discover -s tests -p "test_algolab*.py" -v
```

Optional local T0 smoke (CPU only, no Modal):

```text
python -m algolab.bench ladder algolab/plugins/two_box_envelope.py --seeds 101 202 303 --tiers T0
```

**Do not** pass `--confirm-modal`, `--arthur-authorized`, or `--confirm-llm` without Arthur's explicit approval in the current conversation. Do not resume or hot-patch `savant-exp1` with this code; start a new session (e.g. `savant-exp2-algolab`) after review.

Full plan: `algolab/INTEGRATION_PLAN.md`.
