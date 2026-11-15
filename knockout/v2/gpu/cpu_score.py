"""CPU re-scoring of a stored solution with the frozen verifier (CLAUDE.md, WS3 rule 1: the GPU searches, the CPU
scores). The GPU container runs it in a separate process, so the GPU loop does not wait for it. No torch here.

    python -m knockout.v2.gpu.cpu_score SOLUTION.npy        # prints one JSON line
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

SLUG = "second-autocorrelation-inequality"
_PROBLEM = None


def problem():
    """The frozen problem (its loader re-checks the sha256 of the verifier and of the certificate)."""
    global _PROBLEM
    if _PROBLEM is None:
        from problems.catalog import load

        _PROBLEM = load(SLUG)
    return _PROBLEM


def score_npy(path: str) -> dict:
    """Score the float64 vector stored at `path` (.npy) with the pinned verifier, in float64 on CPU."""
    t0 = time.monotonic()
    values = np.load(path)
    out = {"path": path, "n": int(values.size), "nonzero": int(np.count_nonzero(values)), "dtype": str(values.dtype),
           "score": None, "valid": False, "error": None}
    try:
        out["score"] = float(problem().score({"values": values.tolist()}))
        out["valid"] = True
    except Exception as exc:  # noqa: BLE001 - the verifier's own rejection (InvalidSolution) or a broken file
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    out["cpu_score_s"] = time.monotonic() - t0
    return out


if __name__ == "__main__":
    print(json.dumps(score_npy(sys.argv[1])))
