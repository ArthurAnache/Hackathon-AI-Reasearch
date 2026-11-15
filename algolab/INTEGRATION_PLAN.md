# Algorithm Lab → SAVANT: integration and next-campaign plan

Worktree: `savant-sci-audit` (detached `c1e2ef6`). Active campaign `savant-exp1` is **not** modified.
No commits, pushes, merges, paid LLM or Modal jobs in this preparation.

## What landed (implementation status)

| Piece | Path | Status |
|---|---|---|
| Trusted math kit | `algolab/kit.py` | done |
| Static gate + audit-hook child | `algolab/gate.py`, `algolab/child.py` | done |
| Local ladder + promotion floors | `algolab/bench.py`, `algolab/promote.py` | done |
| Author / 3-repair API (offline default) | `algolab/author.py` | done |
| Live-record helper (Arthur-gated) | `algolab/live_record.py` | done |
| Modal image + `block_network` app | `algolab/image.py`, `algolab/modal_app.py` | done (not dispatched) |
| Wave planner | `algolab/campaign.py` | done (plan-only) |
| WS3 adapter optimizer | `optimizers/second-autocorrelation-inequality_algolab.py` | done |
| SAVANT `algolab_plugin` template | `savant/templates.py`, `savant/schemas.py` | done (sci-audit only) |
| CSV/arm/seed fix | `ws3_bench/campaign.py` | done (sci-audit only) |
| CRLF lock | `.gitattributes` | done |
| Queued novel plugins | `lp_packet_ascent.py`, `two_box_envelope.py`, `dinkelbach_kelley.py` | drafted + gate-admitted |

## Isolation model (requirement 6)

1. **Container (real boundary):** `algolab.modal_app.run_algolab_job` sets `block_network=True`, `secrets=[]`, allowlisted image (no `.env`, no `integrations/`, no plugin baked in).
2. **Process:** adapter launches `python -I algolab/child.py` with secrets stripped from the environment.
3. **Static gate + audit hook:** defence in depth; not treated as sufficient alone.
4. **Truth:** parent re-scores with the frozen pinned verifier; claims still go through `selection/verify_claim.py`.

## Staged benchmark (requirement 4)

| Tier | Start | Budget | Seeds | Promote when |
|---|---|---|---|---|
| T0 | random sparse n=4096 | 6 s | 101,202,303 | mean gain ≥ 1e-4, all ≥ 0 |
| T1 | pooled incumbent n=32768 | 20 s | same | mean ≥ 1e-4 and paired ≥ 1e-4 vs control |
| T1b | proxy kink n=4096 | 30 s | same | mean ≥ 1e-8 (escape signal) |
| T2 | frozen 2M incumbent | 60 s | same | mean ≥ 1e-7 and (paired ≥ 1e-8 or ≥ 10× control) |
| T3 | live #1 (after Arthur refresh) | ≥ 900 s Modal | ≥ 3 | mean ≥ 0.1 δ |

Control arm: stock blockshift, same seeds/budget/harness. Heavy tiers (T2/T3) use Modal, not the laptop.

## Next research campaign (safe sequence)

1. **Let `savant-exp1` finish or STOP on its own.** Do not resume it with Algorithm Lab code.
2. Arthur reviews the three draft plugins (provenance "untrusted" until then).
3. Offline validation in this worktree (already runnable):
   ```text
   python -m algolab.bench selftest
   python -m algolab.campaign admit-seeds
   python -m algolab.campaign plan-ladder --seeds 101 202 303
   python -m algolab.campaign live-record
   ```
4. Optional local T0 smoke for one plugin (CPU only, no Modal):
   ```text
   python -m algolab.bench ladder algolab/plugins/two_box_envelope.py --seeds 101 202 303 --tiers T0
   ```
5. **Arthur authorizes** a live leaderboard GET:
   ```text
   python -m algolab.campaign live-record --arthur-authorized
   ```
   Then warm-start T2/T3 from a re-verified copy of the live #1, target = live #1 + δ.
6. **Arthur authorizes Modal spend**, then dispatch T0→T1→T1b→T2 from the written `runs/algolab/plans/...` batch specs through the algolab Modal app (`block_network`). T3 only for arms that promote at T2.
7. Promote survivors with `algolab.promote.pin_as_template`; wire them as `algolab_plugin` hypotheses in a **new** SAVANT session (e.g. `savant-exp2-algolab`), never by hot-patching exp1.
8. Any record-scale candidate: `selection/verify_claim.py` then gated `integrations/submit.py --confirm` (Arthur only).

## Safe next-run command (this worktree, unpaid)

```text
cd C:\Users\thiba\Desktop\Berkeley\AIRESEARCHACK\savant-sci-audit
python -m algolab.campaign admit-seeds
python -m algolab.campaign plan-ladder --seeds 101 202 303
python -m algolab.campaign live-record
```

Do **not** pass `--confirm-modal`, `--arthur-authorized`, or `--confirm-llm` without Arthur's explicit go-ahead in the current conversation.
