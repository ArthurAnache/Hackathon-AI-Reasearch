"""Turn a stored solution (.npy, float64) into a candidate file for the arbiter.

    modal volume get knockout-v2 <run_id>/solutions/<episode_id>.npy best.npy
    .venv/bin/python -m knockout.v2.export best.npy best.json
    .venv/bin/python selection/verify_claim.py second-autocorrelation-inequality best.json

The values are written with `repr`, so the candidate holds exactly the float64 numbers that were scored.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def export(npy_path: str, json_path: str) -> int:
    values = np.load(npy_path).astype(np.float64)
    if values.ndim != 1 or values.min() < 0.0 or not np.all(np.isfinite(values)):
        raise SystemExit("not a non-negative finite vector")
    with open(json_path, "w") as fh:
        json.dump({"values": values.tolist()}, fh, separators=(",", ":"))
    return len(values)


if __name__ == "__main__":
    print(f"{export(sys.argv[1], sys.argv[2])} values -> {sys.argv[2]}")
