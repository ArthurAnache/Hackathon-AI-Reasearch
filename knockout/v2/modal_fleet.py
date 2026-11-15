"""CPU fleet on Modal: fan a list of episodes out over one-core workers, keep every solution on a Volume.

    # once: the start solutions, as float64 .npy
    modal volume put knockout-v2 sol2783.npy starts/sol2783.npy

    modal run --detach -m knockout.v2.modal_fleet::launch --plan smoke      # prints run_id and a cost estimate
    modal run -m knockout.v2.modal_fleet::finalize --run-id <run_id>        # values, outcome classes, local JSONL
    modal volume get knockout-v2 <run_id>/solutions/<episode_id>.npy .      # one solution, for the arbiter

Rules followed (CLAUDE.md, "WS3 - Modal compute and GPUs"):
- the image copies problems/ unchanged, and each worker scores its result with `load(slug).score` (the frozen
  verifier, sha256 checked on load) in float64 on CPU;
- no secret in the image or in the functions; nothing here submits or calls a write endpoint;
- every function has an explicit timeout and a cap on parallel containers;
- every solution is stored on the Volume `knockout-v2` as `<run_id>/solutions/<episode_id>.npy`, with its record
  in `<run_id>/records/<episode_id>.json`; workers return small metadata only.

The optimizer here is trusted repository code (no LLM-written program), so the workers are plain containers, not
the sandbox of knockout/modal_app.py.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent.parent
IMAGE_ROOT = "/root"
VOL = "/vol"
VOLUME_NAME = "knockout-v2"
WORKER_CPU, WORKER_MEMORY_MB = 1.0, 2048                   # measured peak of one episode at n = 2M: 1.3 GiB
USD_PER_CORE_S, USD_PER_GIB_S = 0.0000131, 0.00000222      # estimates, see knockout/strategy.yaml
SECONDS_PER_EVAL = 0.4                                      # one Modal core; the smoke run measured 0.36 to 0.40
# Cap on parallel workers of one launch. The workspace runs at most 100 containers in total (measured on
# 2026-11-14: a 288-episode launch took 91 and starved the other apps), so several launches must share that.
MAX_WORKERS = int(os.environ.get("KO_V2_MAX_WORKERS", "40"))
_IGNORE = ["**/__pycache__", "**/*.pyc", "**/.DS_Store", "results/**"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Same versions as the local .venv, so the CPU score on Modal matches the local one.
    .pip_install("numpy==2.5.4", "scipy==1.18.1")
    # One thread per worker: a single 1-D FFT does not use more, and the bill follows the cores actually used.
    .env({"PYTHONPATH": IMAGE_ROOT, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
          "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    .add_local_dir(REPO / "knockout", f"{IMAGE_ROOT}/knockout", ignore=_IGNORE)
    .add_local_dir(REPO / "problems", f"{IMAGE_ROOT}/problems", ignore=_IGNORE)
)

app = modal.App("knockout-v2-moves", image=image, include_source=False)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def worker_usd(wall_s: float) -> float:
    return wall_s * (WORKER_CPU * USD_PER_CORE_S + WORKER_MEMORY_MB / 1024 * USD_PER_GIB_S)


@app.function(cpu=WORKER_CPU, memory=WORKER_MEMORY_MB, timeout=3 * 3600, max_containers=MAX_WORKERS, retries=1,
              volumes={VOL: volume})
def run_one(spec: dict, run_id: str, meta: dict) -> dict:
    """One episode on one core. Writes the solution and its record to the Volume, returns the record."""
    import time

    import numpy as np

    from knockout.v2.episode import EpisodeSpec, run_episode

    t0 = time.monotonic()
    es = EpisodeSpec(**spec)
    base = Path(VOL) / run_id
    record_path = base / "records" / f"{es.episode_id}.json"
    if record_path.exists():                       # a retry or a re-launch: the episode is deterministic
        return json.loads(record_path.read_text())
    f_start = np.load(Path(VOL) / "starts" / f"{es.start}.npy")
    references = {name: np.load(Path(VOL) / "starts" / f"{name}.npy") for name in meta.get("references", [])}
    best_f, record = run_episode(es, f_start, references=references)
    solution_path = base / "solutions" / f"{es.episode_id}.npy"
    solution_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(solution_path, best_f)
    record.update({
        "run_id": run_id, "source_commit": meta.get("source_commit"), "device": "cpu", "dtype": "float64",
        "solution_path": f"{run_id}/solutions/{es.episode_id}.npy", "valid": True,
        "best_known_score": meta.get("best_known_score"), "budget_s": None, "evidence_urls": [],
        "hypothesis": f"{es.family} (strength {es.strength:g}) then polish {es.schedule} with {es.budget} "
                      f"evaluations ends above polish alone from {es.start}",
        "config": spec,
    })
    record["worker_wall_s"] = round(time.monotonic() - t0, 2)
    record_path.write_text(json.dumps(record))
    volume.commit()
    return record


@app.function(cpu=0.25, memory=1024, timeout=4 * 3600, max_containers=6, volumes={VOL: volume})
def drive(run_id: str, specs: list[dict], meta: dict) -> dict:
    """Fan the episodes out and append each record to <run_id>/episodes.jsonl as it arrives."""
    base = Path(VOL) / run_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "plan.json").write_text(json.dumps({"meta": meta, "specs": specs}))
    volume.commit()
    done = failed = 0
    with open(base / "episodes.jsonl", "a") as fh:
        for out in run_one.map(specs, kwargs={"run_id": run_id, "meta": meta}, order_outputs=False,
                               return_exceptions=True):
            if isinstance(out, Exception):
                failed += 1
                out = {"run_id": run_id, "valid": False, "error": f"{type(out).__name__}: {out}"[:500]}
            else:
                done += 1
            fh.write(json.dumps(out) + "\n")
            fh.flush()
    volume.commit()
    return {"run_id": run_id, "done": done, "failed": failed}


@app.function(cpu=1.0, memory=4096, timeout=3600, max_containers=4, volumes={VOL: volume})
def finalize_remote(run_id: str) -> list[dict]:
    """Value and outcome class of every finished episode of a run (needs the baseline solutions, so it runs next
    to the Volume). Also writes <run_id>/episodes_final.jsonl."""
    import numpy as np

    from knockout.v2.episode import finalize as finalize_records

    volume.reload()
    base = Path(VOL) / run_id
    records = [json.loads(p.read_text()) for p in sorted((base / "records").glob("*.json"))]
    rows = finalize_records(records, lambda r: np.load(Path(VOL) / r["solution_path"]))
    for r in rows:
        r["usd_estimate"] = round(worker_usd(r["worker_wall_s"]), 5)
    (base / "episodes_final.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    volume.commit()
    return rows


def estimate_usd(specs: list[dict]) -> float:
    return sum(worker_usd(s["budget"] * SECONDS_PER_EVAL + 30.0) for s in specs)


@app.local_entrypoint()
def launch(plan: str = "smoke", plan_file: str = "", source_commit: str = "", best_known_score: float = 0.0,
           max_usd: float = 5.0):
    """Start a run in the background. `plan` names a function of knockout/v2/plans.py; `plan_file` is a JSON list
    of episode specs and wins over `plan`. The launch is refused when the cost estimate exceeds `max_usd`."""
    from knockout.v2 import plans

    specs = json.loads(Path(plan_file).read_text()) if plan_file else plans.PLANS[plan]()
    # Episodes of 6,000 evaluations or more start first (they set the end of the run), then the shortest ones
    # (they tell early whether the plan makes sense).
    specs = sorted(plans.with_baselines(specs),
                   key=lambda s: (0, -s["budget"]) if s["budget"] >= 6000 else (1, s["budget"]))
    name = Path(plan_file).stem if plan_file else plan
    usd = estimate_usd(specs)
    if usd > max_usd:
        raise SystemExit(f"estimated {usd:.2f} USD for {len(specs)} episodes exceeds --max-usd {max_usd}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{name}"
    meta = {"source_commit": source_commit or None, "best_known_score": best_known_score or None,
            "references": ["sol2782", "sol2783"], "plan": name}
    call = drive.spawn(run_id, specs, meta)
    print(f"run_id={run_id} episodes={len(specs)} evaluations={sum(s['budget'] for s in specs)} "
          f"estimate_usd={usd:.2f} call={call.object_id}")


@app.local_entrypoint()
def finalize(run_id: str, out_dir: str = ""):
    rows = finalize_remote.remote(run_id)
    out = Path(out_dir or REPO / "knockout" / "v2" / "results" / "moves_cpu")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_id}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} episodes -> {path}; estimated spend {sum(r['usd_estimate'] for r in rows):.2f} USD")
