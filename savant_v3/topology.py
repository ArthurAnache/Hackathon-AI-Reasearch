"""Support topology metrics — prove Engine A/B/C change structure, not just amplitudes."""
from __future__ import annotations

import numpy as np


def teeth(f: np.ndarray, thr_rel: float = 1e-8) -> list[tuple[int, int]]:
    f = np.asarray(f, dtype=np.float64)
    thr = float(f.max()) * thr_rel if f.size else 0.0
    active = f > thr
    out, i, n = [], 0, len(f)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        out.append((i, j))
        i = j
    return out


def topology_report(f0: np.ndarray, f1: np.ndarray) -> dict:
    a = np.maximum(np.asarray(f0, dtype=np.float64).ravel(), 0.0)
    b = np.maximum(np.asarray(f1, dtype=np.float64).ravel(), 0.0)
    # Score is scale-invariant; candidates are often max-normalized while the
    # frozen certificate is not — compare shapes, not absolute scales.
    if a.max() > 0:
        a = a / a.max()
    if b.max() > 0:
        b = b / b.max()
    if a.size != b.size:
        # compare at min length by block-mean downsample of the longer
        if a.size > b.size:
            a = _block_mean(a, b.size)
        else:
            b = _block_mean(b, a.size)
    thr = max(a.max(), b.max()) * 1e-8
    sa, sb = a > thr, b > thr
    left = int(np.sum(sa & ~sb))
    entered = int(np.sum(~sa & sb))
    t0, t1 = teeth(a), teeth(b)
    rel_l2 = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-15))
    return {
        "n": int(a.size),
        "teeth_before": len(t0),
        "teeth_after": len(t1),
        "delta_teeth": len(t1) - len(t0),
        "cells_left_support": left,
        "cells_entered_support": entered,
        "support_jaccard": float(np.sum(sa & sb) / max(np.sum(sa | sb), 1)),
        "rel_l2": rel_l2,
        "structural_change": bool(left + entered > 100 or abs(len(t1) - len(t0)) >= 5),
    }


def _block_mean(x: np.ndarray, n: int) -> np.ndarray:
    # crude downsample
    idx = np.linspace(0, len(x), n + 1).astype(int)
    return np.array([x[idx[i]:idx[i + 1]].mean() if idx[i + 1] > idx[i] else 0.0 for i in range(n)])
