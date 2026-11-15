# SAVANT V3 — Experiment Log

Shared ceiling remaining at start: see `_coordination/savant/SHARED_BUDGET_CEILING.json` (V2 spent ≈ $1.74 / $300).

## Offline validation (2026-11-15)

Command: `python -m savant_v3.offline_validate`

| Engine | Gate | Synth structural_change | cert_8k structural_change | Notes |
|---|---|---|---|---|
| A comb_reconstruct | admit | True | True | Large topology jumps; score may drop before improve recovers |
| B multires_basin | admit | True | False (amplitude-dominated at 8k) | +kit gain on cert_8k proxy |
| C active_set_exchange | admit | True | True | Packet rebuild + Kelley-style exchanges |

Artifact: `runs/v3/offline/validation.json` — **PASSED**.

## CPU vs GPU benchmark

Pending Stage-3. Pilot uses 1CPU/4GiB WS3 containers (same as V2) until a controlled FFT/L-BFGS microbench shows GPU wall-time advantage ≥2× on representative n≥256k jobs.

## Pilot 20261115T071011Z-v3pilot (Stage 2)

- 6 jobs × 90s, est **$0.0537**, wall 140.5s, app `ap-sLMNQWspsX4u4JKSabzw0L`, volume `savant-v3-results`.
- Shared remaining after: **$298.208**.
- All 6 valid / parent-verified. Best gain **+8.26e-12** (Engine A s101) — noise vs V2’s +4.55e-8.
- Engines B/C returned the max-normalized start (rel_l2=0 after scale fix). Engine A: amplitude-only (jaccard=1, Δteeth=0).
- Root cause: algolab child only accepts score > start; full-res rebuilds tank C and never recover in 90s. Engine B’s dense L-BFGS on 2M bounds overran budget.
- Topology bug fixed (normalize by max before Jaccard). Engines A/B/C rewritten to **coarse-first** (8k–16k rebuild/recover → upsample → fine polish). Offline re-validation PASSED with structural_change=True on synth and cert_8k; Engine B cert_8k kit gain ≈ +0.031.

## Iteration 2 — 20261115T071819Z-v3it02

- 9 jobs × 300s, est **$0.2058**, wall 323s. Shared rem **$298.002**.
- All returned max-normalized frozen start (gain ~1e-15). Coarse-first cannot beat the 2M incumbent after lift (downsample drops C by ~0.05).

## Scientific reset (post-it2)

- Measured: block-mean(cert, 8k) scores ~0.911; improved 8k reaches ~0.940 still ≪ 0.9636.
- New operators: `comb_surgery` (micro tooth edits), `spectral_lowmode`, `independent_comb`.
- Audit P0-a: warm-start from **public** live #1 (sol 2783) via GET `/api/solutions/best`, baked into image as `live_public_best.json.gz`.

## Iteration 3 — aborted

- Client disconnect + fat warm-start payloads → workers KeyboardInterrupt. Est waste **$0.02**. Fixed: image bake-in + spawn/join without streamed output.

## Iteration 4 — 20261115T181500Z-v3it04 ⭐

- 6 jobs × 600s from public live #1, est **$0.2565**, wall 643s, rem **$297.726**.
- **Best:** `engineA-surgery-s101` sha `5fd57d5ce22f…`
  - pinned **0.9636615622011321**
  - independent (exact rational) **0.9636615622011331**
  - gain vs live #1: **+1.390e-10** (verified)
  - gap to live+δ: **9.99986e-6** (not yet beats_record)
- First V3 construction above the public live #1. Lineage promoted into image bake-in.

## Iteration 5 — 20261115T182908Z-v3it05 (failed)

- 6 jobs × 900s, est **$0.3758** charged; all hit Modal `FUNCTION_TIMEOUT_S=900` (budget == timeout). No valid candidates. Timeout raised to **3600s**.

## Iteration 6 — running

- Chained from it4 best bake-in; budget 600s; seeds 101/202/404; aggressive surgery params.

## Iteration 5 — 20261115T182908Z-v3it05 (failed)

- 6 jobs × 900s, est **$0.3758** charged; all hit Modal `FUNCTION_TIMEOUT_S=900` (budget == timeout). No valid candidates. Timeout raised to **3600s**.

## Iteration 6 — running

- Chained from it4 best bake-in; budget 600s; seeds 101/202/404; aggressive surgery params.

## Pilot 20261115T071011Z-v3pilot

- jobs: 6, est $0.0537, wall 140.5s
- app: `ap-sLMNQWspsX4u4JKSabzw0L` volume `savant-v3-results`
- engineA-comb-s101: status=discard valid=True gain=8.26028134781609e-12 score=0.9636386680010726
- engineB-multires-s101: status=void valid=True gain=5.551115123125783e-16 score=0.9636386679928128
- engineC-activeset-s101: status=discard valid=True gain=5.551115123125783e-16 score=0.9636386679928128
- engineA-comb-s202: status=discard valid=True gain=3.971267759084185e-13 score=0.9636386679932094
- engineB-multires-s202: status=void valid=True gain=5.551115123125783e-16 score=0.9636386679928128
- engineC-activeset-s202: status=discard valid=True gain=5.551115123125783e-16 score=0.9636386679928128


## Iteration 2 — 20261115T071819Z-v3it02

- jobs: 9, est $0.2058, wall 323.4s, structural_hits=0
- live target: 0.9636715620620852 (live={'value': 0.9636615620620852, 'agent': 'MLAI_YONSEI', 'live': True, 'source': 'https://einsteinarena.com/api/leaderboard?problem_id=3', 'read_at': '2026-11-15T07:18:18Z'})
- engineA-comb-s101: valid=True gain=5.551115123125783e-16 score=0.9636386679928128 status=discard
- engineB-multires-s101: valid=True gain=5.551115123125783e-16 score=0.9636386679928128 status=discard
- engineC-activeset-s101: valid=True gain=5.551115123125783e-16 score=0.9636386679928128 status=discard
- engineA-comb-s202: valid=True gain=5.551115123125783e-16 score=0.9636386679928128 status=discard
- engineB-multires-s202: valid=True gain=5.551115123125783e-16 score=0.9636386679928128 status=discard
- engineC-activeset-s202: valid=True gain=5.551115123125783e-16 score=0.9636386679928128 status=discard
- engineA-comb-s303: valid=True gain=5.551115123125783e-16 score=0.9636386679928128 status=discard
- engineB-multires-s303: valid=True gain=5.551115123125783e-16 score=0.9636386679928128 status=discard
- engineC-activeset-s303: valid=True gain=5.551115123125783e-16 score=0.9636386679928128 status=discard
  - topo engineA-comb-s101: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineB-multires-s101: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineC-activeset-s101: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineA-comb-s202: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineB-multires-s202: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineC-activeset-s202: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineA-comb-s303: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineB-multires-s303: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineC-activeset-s303: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0


## Iteration 4 — 20261115T181500Z-v3it04

- jobs: 6, est $0.2565, wall 642.6s, structural_hits=0
- live target: 0.9636715620620852 (live={'value': 0.9636615620620852, 'agent': 'MLAI_YONSEI', 'live': True, 'source': 'https://einsteinarena.com/api/leaderboard?problem_id=3', 'read_at': '2026-11-15T18:14:59Z'})
- engineA-surgery-s101: valid=True gain=2.289420831980138e-05 score=0.9636615622011321 status=keep
- engineE-spectral-s101: valid=True gain=2.2894069273138307e-05 score=0.9636615620620854 status=keep
- engineB-multires-s101: valid=True gain=2.2894069273138307e-05 score=0.9636615620620854 status=keep
- engineA-surgery-s202: valid=True gain=2.289409409828025e-05 score=0.9636615620869106 status=keep
- engineE-spectral-s202: valid=True gain=2.2894069273138307e-05 score=0.9636615620620854 status=keep
- engineB-multires-s202: valid=True gain=2.2894069273138307e-05 score=0.9636615620620854 status=keep
  - topo engineA-surgery-s101: structural=False Δteeth=0 jaccard=0.9999958005072987 rel_l2=4.393436646999959e-08
  - topo engineE-spectral-s101: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineB-multires-s101: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineA-surgery-s202: structural=False Δteeth=-1 jaccard=0.9999958005072987 rel_l2=5.253762990857359e-09
  - topo engineE-spectral-s202: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineB-multires-s202: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0


## Iteration 5 — 20261115T182908Z-v3it05

- jobs: 6, est $0.3758, wall 910.8s, structural_hits=0
- live target: 0.9636715620620852 (live={'value': 0.9636615620620852, 'agent': 'MLAI_YONSEI', 'live': True, 'source': 'https://einsteinarena.com/api/leaderboard?problem_id=3', 'read_at': '2026-11-15T18:29:07Z'})
- engineA-surgery-s101: valid=False gain= score= status=crash
- engineE-spectral-s101: valid=False gain= score= status=crash
- engineA2-surgery-s101: valid=False gain= score= status=crash
- engineA-surgery-s303: valid=False gain= score= status=crash
- engineE-spectral-s303: valid=False gain= score= status=crash
- engineA2-surgery-s303: valid=False gain= score= status=crash


## Iteration 6 — 20261115T184550Z-v3it06

- jobs: 9, est $0.3848, wall 665.7s, structural_hits=0
- live target: 0.9636715620620852 (live={'value': 0.9636615620620852, 'agent': 'MLAI_YONSEI', 'live': True, 'source': 'https://einsteinarena.com/api/leaderboard?problem_id=3', 'read_at': '2026-11-15T18:45:49Z'})
- engineA-surgery-s101: valid=True gain=2.289420945245091e-05 score=0.9636615622022647 status=keep
- engineE-spectral-s101: valid=True gain=2.289420831980138e-05 score=0.9636615622011321 status=keep
- engineA2-surgery-s101: valid=True gain=2.2894390482419702e-05 score=0.9636615623832947 status=keep
- engineA-surgery-s202: valid=True gain=2.289424245893734e-05 score=0.9636615622352712 status=keep
- engineE-spectral-s202: valid=True gain=2.289420831980138e-05 score=0.9636615622011321 status=keep
- engineA2-surgery-s202: valid=True gain=2.2894208819623785e-05 score=0.9636615622016319 status=keep
- engineA-surgery-s404: valid=True gain=2.2894294939734827e-05 score=0.963661562287752 status=keep
- engineE-spectral-s404: valid=True gain=2.289420831980138e-05 score=0.9636615622011321 status=keep
- engineA2-surgery-s404: valid=True gain=2.289420831980138e-05 score=0.9636615622011321 status=keep
  - topo engineA-surgery-s101: structural=False Δteeth=-1 jaccard=0.9999979002536493 rel_l2=1.3077141133682702e-08
  - topo engineE-spectral-s101: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineA2-surgery-s101: structural=False Δteeth=-2 jaccard=0.9999937007874016 rel_l2=2.4543279705077987e-08
  - topo engineA-surgery-s202: structural=False Δteeth=-1 jaccard=0.9999958005249344 rel_l2=3.0431814486081717e-09
  - topo engineE-spectral-s202: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineA2-surgery-s202: structural=False Δteeth=0 jaccard=1.0 rel_l2=1.419764438652446e-09
  - topo engineA-surgery-s404: structural=False Δteeth=0 jaccard=0.9999958005249344 rel_l2=1.367807367171537e-08
  - topo engineE-spectral-s404: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineA2-surgery-s404: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0


## Iteration 7 — 20261115T190149Z-v3it07

- jobs: 6, est $0.3758, wall 929.9s, structural_hits=2
- live target: 0.9636715620620852 (live={'value': 0.9636615620620852, 'agent': 'MLAI_YONSEI', 'live': True, 'source': 'https://einsteinarena.com/api/leaderboard?problem_id=3', 'read_at': '2026-11-15T19:01:48Z'})
- engineF-dinkelbach-s101: valid=True gain=2.2894390482419702e-05 score=0.9636615623832947 status=keep
- engineG-lse-s101: valid=True gain=2.2944374213906826e-05 score=0.9636616123670262 status=keep
- engineA-surgery-s101: valid=True gain=2.2894393550743075e-05 score=0.963661562386363 status=keep
- engineF-dinkelbach-s202: valid=True gain=2.2894390482419702e-05 score=0.9636615623832947 status=keep
- engineG-lse-s202: valid=True gain=2.294484888087922e-05 score=0.9636616128416932 status=keep
- engineA-surgery-s202: valid=True gain=2.289444135672447e-05 score=0.963661562434169 status=keep
  - topo engineF-dinkelbach-s101: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineG-lse-s101: structural=True Δteeth=-27 jaccard=0.9995675675675676 rel_l2=1.1493459610415098e-07
  - topo engineA-surgery-s101: structural=False Δteeth=-1 jaccard=0.9999979002580582 rel_l2=1.6067225832172862e-08
  - topo engineF-dinkelbach-s202: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineG-lse-s202: structural=True Δteeth=-21 jaccard=0.9992255529740655 rel_l2=1.67672198094602e-07
  - topo engineA-surgery-s202: structural=False Δteeth=-1 jaccard=0.9999958005337521 rel_l2=3.5928184171628793e-09


## Iteration 8 — 20261115T191920Z-v3it08 (volume-recovered)

- 9 jobs x 1200s, est $0.7427, rem ~$295.85.
- Best engineG-lse-s202 sha 54140cd2e955…: pinned **0.9636616143678811**, indep **0.9636616143678813**.
- Gain vs it7 warm: **+1.53e-9**; vs live 2783: **+5.23e-8**; gap to target: **9.948e-6**.
- Client hung after app stop; results recovered from savant-v3-results volume.

## Iteration 9 — hail-mary (spectral+LSE+surgery @ 1800s)


## Packaging fix + checkpoint freeze (2026-11-15 ~12:00 PT)

- Re-verified it8 best: pinned 0.9636616143678811, indep 0.9636616143678813, sha match (uns/v3/best/VERIFY.json).
- Modal `ModuleNotFoundError: savant_v3` root cause: wrong module entrypoint `savant_v3.v3_modal_app` (not a Cron schedule). Entrypoint remains `ws3_bench.v3_modal_app`; shim + `savant_v3/__init__.py` on image allowlist.
- Modal import smoke: `import_smoke_v3` -> ok / version 3.0.0.
- Submission staging: `runs/v3/submission/` (NOT submitted to Arena).
- iter9 left running if still active; volumes not deleted.


## Iteration 9 — 20261115T195749Z-v3it09

- jobs: 6, est $0.7337, wall 1863.8s, structural_hits=2
- live target: 0.9636715620620852 (live={'value': 0.9636615620620852, 'agent': 'MLAI_YONSEI', 'live': True, 'source': 'https://einsteinarena.com/api/leaderboard?problem_id=3', 'read_at': '2026-11-15T19:57:48Z'})
- engineE-spectral-s101: valid=True gain=2.2946375069277103e-05 score=0.9636616143678816 status=keep
- engineG-lse-s101: valid=True gain=2.2947230289172005e-05 score=0.9636616152231015 status=keep
- engineA-surgery-s101: valid=True gain=2.2946375069277103e-05 score=0.9636616143678816 status=keep
- engineE-spectral-s202: valid=True gain=2.2946375069277103e-05 score=0.9636616143678816 status=keep
- engineG-lse-s202: valid=True gain=2.2946900106513013e-05 score=0.9636616148929188 status=keep
- engineA-surgery-s202: valid=True gain=2.2946375069277103e-05 score=0.9636616143678816 status=keep
  - topo engineE-spectral-s101: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineG-lse-s101: structural=True Δteeth=54 jaccard=0.9989386537683035 rel_l2=2.1703559343176527e-07
  - topo engineA-surgery-s101: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineE-spectral-s202: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0
  - topo engineG-lse-s202: structural=True Δteeth=40 jaccard=0.9992279174962707 rel_l2=1.7758003179984399e-07
  - topo engineA-surgery-s202: structural=False Δteeth=0 jaccard=1.0 rel_l2=0.0


## Iteration 9 — 20261115T195749Z-v3it09 (ITER_OK)

- 6 jobs x 1800s, est $0.7337, rem after **$295.1129** (reservation settled).
- Best engineG-lse-s101 sha 51f3792ebd62…: pinned **0.9636616152231015**, indep **0.9636616152231022**.
- vs Iter8: **+8.55e-10**; vs live 2783: **+5.32e-8**; gap to target: **9.947e-6** (not beats_record).
- Promoted to uns/v3/best/ and rtifacts/savant-v3/.


## Iteration 9 — 20261115T195749Z-v3it09 COMPLETE

- App `ap-cJUBf4b0kteGMYl2OCOKdx`, 6/6 jobs ok, est/committed **$0.7337**, rem **$295.11**.
- Best `engineG-lse-s101` sha `51f3792ebd62…`: pinned **0.9636616152231015**, indep **0.9636616152231022**.
- vs Iter8: **+8.55e-10**; vs live: **+5.32e-8**; gap to live+δ: **9.947e-6** (still not beats_record).
- Promoted to `runs/v3/best/` and `artifacts/savant-v3/`.

