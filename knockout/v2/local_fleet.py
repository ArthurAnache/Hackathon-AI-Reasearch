"""The same fleet on local cores, for pilots and for machines without Modal.

    .venv/bin/python -m knockout.v2.local_fleet --plan smoke --starts DIR --out runs/v2 --workers 6

`DIR` holds the start solutions as <name>.npy. Solutions and records go to <out>/<run_id>/ (runs/ is git-ignored);
the table with values and outcome classes is <out>/<run_id>/episodes_final.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")       # one thread per episode: the parallelism is across episodes

import numpy as np  # noqa: E402

from knockout.v2 import plans  # noqa: E402
from knockout.v2.episode import EpisodeSpec, finalize, run_episode  # noqa: E402


def _run(spec: dict, starts: str, base: str, references: list[str]) -> dict:
    es = EpisodeSpec(**spec)
    record_path = Path(base) / "records" / f"{es.episode_id}.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    refs = {name: np.load(Path(starts) / f"{name}.npy") for name in references
            if (Path(starts) / f"{name}.npy").exists()}
    best_f, record = run_episode(es, np.load(Path(starts) / f"{es.start}.npy"), references=refs)
    solution_path = Path(base) / "solutions" / f"{es.episode_id}.npy"
    solution_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(solution_path, best_f)
    record.update({"run_id": Path(base).name, "device": "cpu-local", "dtype": "float64", "valid": True,
                   "solution_path": str(solution_path), "config": spec})
    record_path.write_text(json.dumps(record))
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="smoke", help="a plan of knockout/v2/plans.py")
    ap.add_argument("--plan-file", default="", help="JSON list of episode specs (wins over --plan)")
    ap.add_argument("--starts", required=True, help="folder with <start name>.npy")
    ap.add_argument("--out", default="runs/v2")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--references", default="sol2782,sol2783")
    args = ap.parse_args()
    specs = json.loads(Path(args.plan_file).read_text()) if args.plan_file else plans.PLANS[args.plan]()
    specs = plans.with_baselines(specs)
    name = Path(args.plan_file).stem if args.plan_file else args.plan
    run_id = args.run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{name}-local"
    base = Path(args.out) / run_id
    base.mkdir(parents=True, exist_ok=True)
    print(f"run_id={run_id} episodes={len(specs)} evaluations={sum(s['budget'] for s in specs)}", flush=True)
    records = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_run, s, args.starts, str(base), args.references.split(",")) for s in specs]
        for fut in as_completed(futures):
            r = fut.result()
            records.append(r)
            print(f"{r['experiment_id']}: score {r['score']:.13f} moved {r['moved_score']:.9f} "
                  f"calls {r['calls']:.0f} wall {r['wall_s']:.0f}s", flush=True)
    rows = finalize(records, lambda r: np.load(r["solution_path"]))
    (base / "episodes_final.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"table: {base / 'episodes_final.jsonl'}")


if __name__ == "__main__":
    main()
