"""Does a solution carry traces of a block repetition (a coarse-to-fine pipeline with `np.repeat`)? Numpy only.

    .venv/bin/python -m knockout.v2.multires.traces knockout/v2/results/multires/repetition_traces.json

Four checks on the frozen certificate (EA solution 2782) and on the live #1, for refinement factors r:
- repeated cells: share of neighbouring non-zero cells that are equal, exactly or within 1e-6 / 1e-3 relative;
- periodicity modulo r: the mean |log f[i+1] - log f[i]| of the steps that stay inside a group of r cells
  (i mod r != r - 1) against the steps that cross a group boundary, for the best phase. A repetition by r followed
  by a polish would leave the inside steps smaller than the crossing ones (ratio well below 1);
- run boundaries modulo r: how unevenly the first cells of the non-zero runs fall on the residues modulo r
  (largest share times r; 1 = even);
- aggregation: the L1 distance between the two solutions' mass profiles after summing groups of r cells, for
  every r from 1 to 64. A factor at which the two suddenly agree would be the level they were both refined from.
The live #1 is read with one public GET (no key).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from knockout.v2.multires import resample as rs

SLUG = "second-autocorrelation-inequality"
FACTORS = (2, 3, 4, 5, 6, 8, 12, 16, 32)


def repeated_cells(f: np.ndarray) -> dict:
    a, b = f[:-1], f[1:]
    both = (a > 0) & (b > 0)
    rel = np.abs(a[both] - b[both]) / np.maximum(a[both], b[both])
    return {"pairs": int(both.sum()), "equal": float((rel == 0).mean()), "within_1e-6": float((rel < 1e-6).mean()),
            "within_1e-3": float((rel < 1e-3).mean()), "median_relative_step": float(np.median(rel))}


def periodicity(f: np.ndarray, factors=FACTORS) -> list[dict]:
    a, b = f[:-1], f[1:]
    both = (a > 0) & (b > 0)
    idx = np.flatnonzero(both)
    step = np.abs(np.log(b[both]) - np.log(a[both]))
    starts = np.flatnonzero((f > 0) & (np.concatenate(([0.0], f[:-1])) == 0))
    out = []
    for r in factors:
        best = None
        for phase in range(r):
            crossing = (idx + phase) % r == r - 1
            across = float(step[crossing].mean())
            ratio = float(step[~crossing].mean()) / across if across > 0 else float("inf")
            if best is None or ratio < best[1]:
                best = (phase, ratio)
        share = np.bincount(starts % r, minlength=r) / starts.size
        out.append({"r": r, "best_phase": best[0], "inside_over_crossing_step": best[1],
                    "run_start_unevenness": float(share.max() * r)})
    return out


def aggregation(f: np.ndarray, h: np.ndarray, factors=range(1, 65)) -> list[dict]:
    return [{"r": r, "profile_l1": rs.profile_l1(f, h, r)} for r in factors]


def main(out_path: str) -> dict:
    from knockout.problems_api import load_problem
    from knockout.run import load_init
    from problems.catalog import load

    frozen = np.asarray(load(SLUG).certificate()["values"], dtype=np.float64)
    cand, info = load_init(load_problem(SLUG), "live")
    live = np.asarray(cand["values"], dtype=np.float64)
    out = {"live_solution_id": info.get("solution_id")}
    for name, f in (("frozen", frozen), ("live", live)):
        out[name] = {"repeated_cells": repeated_cells(f), "periodicity": periodicity(f)}
    # a positive control: the certificate pooled by 4 and repeated by 4 must light up every check at r = 2 and 4
    control = rs.repeat_upsample(rs.sum_pool(rs.pad_to_multiple(frozen, 4), 4), 4)
    out["control_repeat4"] = {"repeated_cells": repeated_cells(control), "periodicity": periodicity(control, (2, 3, 4))}
    out["aggregation_frozen_vs_live"] = aggregation(frozen, live)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    res = main(sys.argv[1])
    for name in ("frozen", "live", "control_repeat4"):
        print(name, res[name]["repeated_cells"])
        for p in res[name]["periodicity"]:
            print(f"   r {p['r']:<3} inside/crossing step {p['inside_over_crossing_step']:.4f} (phase "
                  f"{p['best_phase']}), run starts unevenness {p['run_start_unevenness']:.3f}")
    agg = res["aggregation_frozen_vs_live"]
    print("aggregation L1 by r:", " ".join(f"{a['r']}:{a['profile_l1']:.4f}" for a in agg))
