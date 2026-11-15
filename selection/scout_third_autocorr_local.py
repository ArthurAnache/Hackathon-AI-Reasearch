"""Run a small plan of step pipelines on local cores, in the same file layout as the Modal rounds.

Usage:  .venv/bin/python selection/scout_third_autocorr_local.py --round-name local1 --budget 840
        .venv/bin/python selection/scout_third_autocorr_local.py --round-name local2 --budget 840 --plan coarseB

Jobs and outputs use the layout of selection/scout_third_autocorr_modal.py (one folder per track and round under
selection/.cache/scout/third-autocorrelation-inequality/modal/), so `harvest --round-name local1` re-scores and
logs them the same way. Track A starts from candidates derived from MILO's published copy; track B from the live
#1 only. A local core was several times faster than a Modal core on this problem (see docs/problem_scout.md).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scout_third_autocorr as core  # noqa: E402
from scout_third_autocorr_modal import CACHE, PROVENANCE, _hops, _path, starts_for  # noqa: E402


def plan(budget: float, name: str = "mixed") -> list[dict]:
    hop = 0.48 * budget
    if name == "coarseB":  # track B only: block average of the live #1 to 3,200 values, repeat x8, one long path
        return [{"track": "B", "label": "localcoarse3200x8", "start_key": "live1_2617_atris", "seed": 33000 + i,
                 "steps": [{"op": "resample", "n": 3200}, {"op": "repeat", "k": 8},
                           _path(t, 0.97 * budget, sigma=sigma, keep_start=False, stages=16)]}
                for i, (t, sigma) in enumerate([(3e-3, 0.0), (6e-3, 0.0), (1e-2, 3e-3), (2e-2, 0.0), (4e-2, 0.0)])]
    if name == "hopsB":  # track B only: continue from the best local1 result (coarse copy of the live #1)
        key = "kept_local1_B002_localcoarse3200x8"
        return [
            {"track": "B", "label": "localhopsB", "start_key": key, "seed": 34001,
             "steps": [_hops((3e-4, 3e-3), (3e-4, 2e-3), hop, stages=14)]},
            {"track": "B", "label": "localhopsB", "start_key": key, "seed": 34002,
             "steps": [_hops((3e-4, 3e-3), (3e-4, 2e-3), hop, stages=14)]},
            {"track": "B", "label": "locallonghopB", "start_key": key, "seed": 34003,
             "steps": [_path(1e-3, 0.97 * budget, sigma=5e-4, keep_start=True, stages=16)]},
            {"track": "B", "label": "localrepeat2B", "start_key": key, "seed": 34004,
             "steps": [{"op": "repeat", "k": 2}, _path(1e-4, 0.97 * budget, sigma=1e-4, keep_start=True, stages=14)]},
        ]
    return [
        {"track": "A", "label": "localhops", "start_key": "cand_c_remelt", "seed": 31001,
         "steps": [_hops((3e-4, 3e-3), (3e-4, 2e-3), hop, stages=14)]},
        {"track": "A", "label": "localhops", "start_key": "cand_c", "seed": 31002,
         "steps": [_hops((3e-4, 3e-3), (3e-4, 2e-3), hop, stages=14)]},
        {"track": "B", "label": "localcoarse3200x8", "start_key": "live1_2617_atris", "seed": 32001,
         "steps": [{"op": "resample", "n": 3200}, {"op": "repeat", "k": 8},
                   _path(1e-2, 0.55 * budget, keep_start=False, stages=16), _hops((3e-4, 3e-3), (3e-4, 2e-3), hop)]},
        {"track": "B", "label": "localcoarse4096x8", "start_key": "live1_2617_atris", "seed": 32002,
         "steps": [{"op": "pool", "k": 16}, {"op": "repeat", "k": 8},
                   _path(1e-2, 0.55 * budget, keep_start=False, stages=16), _hops((3e-4, 3e-3), (3e-4, 2e-3), hop)]},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-name", required=True)
    ap.add_argument("--budget", type=float, default=840.0)
    ap.add_argument("--plan", choices=["mixed", "coarseB", "hopsB"], default="mixed")
    args = ap.parse_args()
    jobs = plan(args.budget, args.plan)
    for i, job in enumerate(jobs):
        track = job["track"]
        starts = CACHE / "modal" / f"track{track}" / "starts_local"
        starts.mkdir(parents=True, exist_ok=True)
        npy = starts / f"{job['start_key']}.npy"
        np.save(npy, np.asarray(json.loads(starts_for(track)[job["start_key"]].read_text())["values"], dtype=np.float64))
        job.update(id=f"{track}{i:03d}_{job['label']}", round=args.round_name, budget_s=args.budget,
                   provenance=PROVENANCE[track], start_path=str(npy), device="local CPU, 1 core",
                   out_dir=str(CACHE / "modal" / f"track{track}" / args.round_name))
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        for res in pool.map(core.run_job_to_dir, jobs, chunksize=1):
            print(res, flush=True)


if __name__ == "__main__":
    main()
