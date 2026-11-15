"""Turn a solution downloaded from the Volume (.npy, float64) into the candidate JSON the arbiter reads.

    modal volume get knockout-v2-gpu <run_id>/<chain>/best.npy runs/anneal_gpu/best.npy
    python -m knockout.v2.gpu.export runs/anneal_gpu/best.npy runs/anneal_gpu/best.json
    python selection/verify_claim.py second-autocorrelation-inequality runs/anneal_gpu/best.json

Floats are written with repr (17 significant digits), so the JSON holds the same float64 values as the .npy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def export(npy: str, out: str) -> dict:
    values = np.load(npy)
    if values.dtype != np.float64 or values.ndim != 1:
        raise ValueError(f"expected a 1-D float64 vector, got {values.dtype} {values.shape}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"values": values.tolist()}, separators=(",", ":")))
    return {"n": int(values.size), "nonzero": int(np.count_nonzero(values)), "min": float(values.min()),
            "bytes": Path(out).stat().st_size}


if __name__ == "__main__":
    print(json.dumps(export(sys.argv[1], sys.argv[2])))
