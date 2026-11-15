"""CPU vs GPU microbench for batched FFT / LSE gradient (Modal optional).

Default: local CPU only. GPU path requires cupy; skipped if unavailable.
Does not spend Modal dollars unless --modal --confirm.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def cpu_fft_batch(n: int, batch: int, reps: int) -> dict:
    from scipy.fft import next_fast_len, rfft, irfft
    rng = np.random.default_rng(0)
    xs = [rng.random(n) for _ in range(batch)]
    L = next_fast_len(2 * n - 1, real=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        for x in xs:
            F = rfft(x, L)
            irfft(F * F, L)[: 2 * n - 1]
    wall = time.perf_counter() - t0
    return {"device": "cpu", "n": n, "batch": batch, "reps": reps, "wall_s": wall,
            "per_conv_ms": 1000 * wall / (reps * batch)}


def gpu_fft_batch(n: int, batch: int, reps: int) -> dict | None:
    try:
        import cupy as cp
    except ImportError:
        return None
    rng = np.random.default_rng(0)
    xs = [cp.asarray(rng.random(n)) for _ in range(batch)]
    L = int(2 ** np.ceil(np.log2(2 * n - 1)))
    # warmup
    for x in xs[:1]:
        F = cp.fft.rfft(x, L)
        cp.fft.irfft(F * F, L)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        for x in xs:
            F = cp.fft.rfft(x, L)
            cp.fft.irfft(F * F, L)
    cp.cuda.Stream.null.synchronize()
    wall = time.perf_counter() - t0
    return {"device": "gpu_cupy", "n": n, "batch": batch, "reps": reps, "wall_s": wall,
            "per_conv_ms": 1000 * wall / (reps * batch)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=262144)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args(argv)
    rows = [cpu_fft_batch(args.n, args.batch, args.reps)]
    g = gpu_fft_batch(args.n, args.batch, args.reps)
    if g:
        rows.append(g)
        speedup = rows[0]["per_conv_ms"] / g["per_conv_ms"]
    else:
        speedup = None
    out = {"rows": rows, "gpu_speedup_vs_cpu": speedup,
           "recommend_gpu": bool(speedup and speedup >= 2.0)}
    dest = ROOT / "runs" / "v3" / "cpu_gpu_bench.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
