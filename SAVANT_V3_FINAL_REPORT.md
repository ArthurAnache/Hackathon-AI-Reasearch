# SAVANT V3 — Final Report

Worktree: `C:\Users\thiba\Desktop\Berkeley\AIRESEARCHACK\savant-v3`  
Branch: `feature/savant-v3-structural`  
Problem: `second-autocorrelation-inequality`  
Campaign window: 2026-11-15

## Verdict

**No record.** Best independently verified score is **above the public live #1** but still **~9.95×10⁻⁶ short** of `live + δ`.

| Metric | Value | Label |
|---|---|---|
| Live #1 (public sol 2783) | 0.9636615620620852 | verified |
| Success target (live + 1e-5) | 0.9636715620620852 | — |
| **V3 best (pinned)** | **0.9636616143678811** | verified |
| **V3 best (independent exact)** | **0.9636616143678813** | verified |
| Gain vs live #1 | **+5.231×10⁻⁸** | verified |
| Gap to beats_record | **9.948×10⁻⁶** | — |
| V2 best (frozen-cert polish) | 0.9636387135161563 | verified (−2.29e-5 vs live) |
| Shared ceiling remaining | **~$293.15 / $300** (iter9 may still hold ~$2.70 reserved) | ledger |
| Arena submission | none | — |

Best certificate (re-verified 2026-11-15):  
- Working copy: `runs/v3/best/54140cd2e955….json.gz` (+ `VERIFY.json`)  
- **Commit pack:** `artifacts/savant-v3/` (`BEST.json`, `VERIFY.json`, certificate `.json.gz`, `README.md`)

### Modal packaging note

`ModuleNotFoundError: No module named 'savant_v3'` came from using `savant_v3.v3_modal_app` as the Modal module path (ephemeral apps import the defining module inside the image). **No Modal Cron/schedule** was found in this workspace. Entrypoint is `ws3_bench.v3_modal_app`; `savant_v3/v3_modal_app.py` is a local shim only; `savant_v3/__init__.py` is on the image allowlist. Local smoke: `SMOKE_OK`.

## Why V2 was the wrong basin

V2 polished the **frozen WS1 certificate** (~0.96363867). The public live #1 is already **+2.29×10⁻⁵** above that point and sits in a **different micro-support basin** (audit F4: 25k cells enter/leave; linear path barrier ~2e-3). Repeating V2 polishers on the frozen cert cannot close the live gap.

## What V3 did differently

1. Isolated worktree + Modal app `savant-v3-bench` / volume `savant-v3-results`.
2. Single shared `$300` ceiling (V2+V3); pilot/stage caps; no second authorization.
3. Engines: comb reconstruct → **comb_surgery**; multires; active-set → **spectral_lowmode**; then **LSE / basin-hop on the public live lineage**.
4. Warm-start from **public** `GET /api/solutions/best` (sol 2783), baked as `live_public_best.json.gz` (no fat job payloads).
5. Chained lineage: each improving wave re-bakes the warm-start.
6. Parent pinned verifier + independent exact rational scorer for promotions.

## Negative results (verified)

| Hypothesis | Result |
|---|---|
| Coarse-first rebuild (8k–64k) then lift | Downsample drops C by ~0.05; cannot recover to beat 2M start |
| Full comb rebuild at 2M | Harness rejects (score must beat start); never sticks |
| Micro tooth-surgery alone | Compounds at ~1e-10 / 10 min — too slow for δ |
| Block-repeat fold of live then improve | Fold destroys C (~0.86); Kronecker preserves C but does not help from a collapsed fold |
| OpenAI hypothesis loop | No `OPENAI_API_KEY` in env — continued with vetted plugins only |

## Positive results (verified)

| Wave | Best score | Gain vs prior warm | Notes |
|---|---|---|---|
| it4 surgery @ live 2783 | 0.963661562201 | +1.39e-10 vs 2783 | First score **above** public live #1 |
| it6 surgery chain | 0.963661562383 | +1.82e-10 | Lineage compounds |
| **it7 LSE @ live lineage** | **0.963661612842** | **+5.05e-8** | Structural (Δteeth −21); breakthrough arm |
| **it8 LSE scale** | **0.963661614368** | **+1.53e-9** | Diminishing returns |

## Iteration / spend table

| Iter | Run | Jobs | Budget | Est $ | Outcome |
|---|---|---|---|---|---|
| offline | — | — | — | 0 | engines gate-admit; topology OK |
| pilot | `…071011Z-v3pilot` | 6 | 90s | 0.054 | frozen start; noise gains |
| 2 | `…071819Z-v3it02` | 9 | 300s | 0.206 | coarse-first dead |
| 3 | aborted | 6 | 600s | ~0.02 | client disconnect / fat payload |
| 4 | `…181500Z-v3it04` | 6 | 600s | 0.257 | +1.39e-10 over live 2783 |
| 5 | `…182908Z-v3it05` | 6 | 900s | 0.376 | Modal timeout==budget; void |
| 6 | `…184550Z-v3it06` | 9 | 600s | 0.385 | +1.82e-10 surgery |
| 7 | `…190149Z-v3it07` | 6 | 900s | 0.376 | **+5.05e-8 LSE** |
| 8 | `…191920Z-v3it08` | 9 | 1200s | 0.743 | +1.53e-9 LSE; client hang recovered |

Approximate V3 Modal spend ≈ **$2.4**; combined V2+V3 ≈ **$4.15** of $300.

## CPU vs GPU

`runs/v3/cpu_gpu_bench.json`: CPU FFT n=262144 ≈ 13.5 ms/conv. No CuPy → GPU not used. Workloads stayed on 1CPU/4GiB WS3 containers.

## Comparison to V2

| | V2 | V3 |
|---|---|---|
| Warm start | frozen cert only | **public live #1 → chained lineage** |
| Best score | 0.9636387135 | **0.9636616144** |
| vs live #1 | −2.29e-5 | **+5.23e-8** |
| vs target (live+δ) | −3.29e-5 | **−9.95e-6** |
| Spend | ~$1.74 | ~$2.4 V3 (shared ledger) |

## Scientific stop (polish arm)

After it8, LSE gains fell from **5e-8 → 1.5e-9** per ~15–20 CPU-min. Closing the remaining **9.95e-6** at that rate is not feasible within the ceiling. Micro-surgery alone is ~100× slower. **δ-scale still needs a new support topology class** (audit F4), not more LSE steps on the current lineage.

Remaining authorization (~$296) should be reserved for a **new construction family** (or GPU Dinkelbach with `f=w²` if/when available), not duplicate local polish.

## Reproducibility

```text
cd savant-v3
set PYTHONPATH=%CD%
set PYTHONUNBUFFERED=1
python -m savant_v3.offline_validate
python -m savant_v3.iterate --confirm --iter N --budget-s 900 --seeds 101 202 --warm-live --stage-cap 20
```

Artifacts: `SAVANT_V3_RESEARCH_PLAN.md`, `SAVANT_V3_EXPERIMENT_LOG.md`, `runs/v3/**`, `runs/v3/best/BEST.json`.

## Explicit non-claims

- Did **not** beat `live + 1e-5`.
- Did **not** submit to EinsteinArena.
- Did **not** use Arthur-private certificates (public best-solutions API only).
- Did **not** spend a second $300 budget.
