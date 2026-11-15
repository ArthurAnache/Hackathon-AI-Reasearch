# SAVANT V3 — Research Plan

Worktree: `savant-v3` (`feature/savant-v3-structural`). Isolated from V1 / V2 / ORBIT / RINGFORGE / C1-LAB.

## Why V2 failed

V2’s best verified gain over the frozen certificate was **+4.55e-8** after 76 Modal jobs (~$1.74). All arms (Dinkelbach, LSE, LP packet, two-box, resolution lift) polished amplitudes inside the **same support basin**. Audit F3–F4: δ-scale gains historically require **rebuilding the comb** (tens of thousands of cells entering/leaving support), which additive polish cannot cross (linear path barrier ~2e-3).

## Success criterion

Refresh **public** EinsteinArena leaderboard:

`LIVE_RECORD + 1e-5`

Do **not** use stale frozen targets. No submission without explicit approval.

## Shared budget

Canonical ledger: `_coordination/savant/SHARED_BUDGET_CEILING.json`  
Authorized **$300** total (V2+V3). V2 spent ≈ **$1.74** → remaining ≈ **$298.26**.  
V3 pilot hard cap: **+$20**. No second $300 authorization.

## Three engines (new; not V2 polishers)

| Engine | Plugin | Hypothesis |
|---|---|---|
| **A — Comb reconstruction** | `algolab/plugins/comb_reconstruct.py` | Birth/death/relocate teeth + mass redistribution → new support topology |
| **B — Multiresolution basin hop** | `algolab/plugins/multires_basin.py` | Low-freq structural perturb + LSE/L-BFGS improve; lift with independent re-score |
| **C — Active-set exchange** | `algolab/plugins/active_set_exchange.py` | Packet epigraph / Kelley-style exchanges on near-max lags |

All run through Algorithm Lab gate + sandboxed child + parent frozen verifier.

## Stages

1. Offline validation (topology + kit) — **done**
2. Modal pilot ≤ **$20** — **done** (~$0.05)
3. Expand under shared ceiling — **done through it8**; live-lineage LSE produced **+5.2e-8 vs public #1**
4. GPU — **deferred** (no CuPy; CPU FFT microbench only)
5. Status: **above live #1, not yet live+δ**; see `SAVANT_V3_FINAL_REPORT.md`

## Anti-patterns (forbidden)

- Another LSE hyperparameter sweep on fixed support
- Counting dry/fake jobs as iterations
- Warm-starting only from V2 best without independent lineage diversity
- Silent mock Modal
- Exceeding shared $300 ceiling
