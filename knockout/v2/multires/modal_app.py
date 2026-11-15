"""Modal app for the coarse-to-fine runs (GPU, float64) on second-autocorrelation-inequality.

    modal run -m knockout.v2.multires.modal_app::launch --job prepare      # store the warm starts on the Volume
    modal run --detach -m knockout.v2.multires.modal_app::launch --plan e1,e2 --start frozen --evals 8000 \
        --pools 4,64 --max-usd 3 --source-commit $(git rev-parse --short HEAD)
    modal volume get knockout-v2-multires <run_id>/summary.json .          # results; solutions are .npy files

Same image and same rules as knockout/v2/gpu/modal_app.py (CLAUDE.md, "WS3 — Modal compute and GPUs"): float64 on
an H100, pinned packages, problems/ copied unchanged and hash-checked at load, the GPU computes the smoothed
objective only and every stored solution is re-scored on CPU by the frozen verifier, explicit timeout and at most
2 GPU containers, a wall-clock budget derived from the USD cap of the launch, no secret, no write endpoint (the
only network call is the read-only GET of the live #1 in `prepare`). A claim is decided only by
selection/verify_claim.py, run locally on a downloaded solution.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal

from knockout.v2.gpu import schedule as sch
from knockout.v2.gpu.modal_app import GPU, GPU_CPU, GPU_MEMORY_MB, image
from knockout.v2.multires import plan as pl

VOL_PATH = "/vol"
GPU_TIMEOUT_S = 3000          # hard stop of one GPU call (50 min, about 3.8 USD)
MAX_GPU_CONTAINERS = 2
MAX_USD_PER_LAUNCH = 4.0

app = modal.App(pl.APP, image=image, include_source=False)
volume = modal.Volume.from_name(pl.VOLUME, create_if_missing=True)


@app.function(cpu=2.0, memory=8192, timeout=900, max_containers=1, volumes={VOL_PATH: volume})
def prepare() -> dict:
    """CPU only: store the frozen certificate and the live #1 on the Volume (one read-only GET)."""
    from knockout.v2.gpu.worker import prepare_starts

    meta = prepare_starts(Path(VOL_PATH))
    volume.commit()
    return meta


@app.function(gpu=GPU, cpu=GPU_CPU, memory=GPU_MEMORY_MB, timeout=GPU_TIMEOUT_S,
              max_containers=MAX_GPU_CONTAINERS, volumes={VOL_PATH: volume})
def gpu_job(spec: dict) -> dict:
    """The only GPU function, so the container cap covers every job. It returns the run summary; solutions stay
    on the Volume, which is committed after every batch."""
    from knockout.v2.multires import worker

    volume.reload()
    try:
        return worker.run_job(Path(VOL_PATH), spec, commit=volume.commit)
    finally:
        volume.commit()


@app.local_entrypoint()
def launch(job: str = "run", plan: str = "smoke", start: str = "frozen", evals: int = 0, pools: str = "",
           max_usd: float = 0.0, source_commit: str = "", tag: str = "", history: int = 8, wait: bool = False):
    if job == "prepare":
        print(json.dumps(prepare.remote(), indent=1))
        return
    if job != "run":
        raise SystemExit(f"unknown job {job!r}")
    if evals <= 0:
        raise SystemExit("--evals (evaluations per level) is required")
    if not 0 < max_usd <= MAX_USD_PER_LAUNCH:
        raise SystemExit(f"--max-usd must be in (0, {MAX_USD_PER_LAUNCH}]")
    factors = tuple(int(x) for x in pools.split(",") if x) or None
    batches = pl.build(plan, start, evals, factors)
    est = pl.estimate(batches)
    rate = est["usd_per_s"]
    budget_s = min(max_usd / rate, GPU_TIMEOUT_S - 240)
    print(f"estimate: {est['wall_s']:.0f} s, {est['usd']:.2f} USD at {rate * 3600:.2f} USD/h; cap {max_usd:.2f} "
          f"USD = {budget_s:.0f} s of wall clock (the job stops there)")
    if est["usd"] > max_usd:
        raise SystemExit("the estimate is above --max-usd: lower --evals or raise the cap")
    commit = (source_commit or "nogit")[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    start_tag = start.replace("vol:", "").replace("/", "_").replace(".npy", "")[-40:]
    run_id = f"{stamp}-{tag or plan.replace(',', '+')}-{start_tag}-{commit[:7]}"
    spec = {"run_id": run_id, "source_commit": commit, "plan": plan, "start": start, "evals": evals,
            "budget_s": budget_s, "max_usd": max_usd, "usd_per_s": rate, "history": history,
            "batches": [b.to_dict() for b in batches]}
    print(f"run_id={run_id} batches={[(b.name, len(b.rows), pl.level_sizes(b)) for b in batches]}")
    if wait:
        out = gpu_job.remote(spec)
        print(json.dumps({k: v for k, v in out.items() if k != "batches"}, indent=1))
    else:
        call = gpu_job.spawn(spec)
        print(f"spawned {call.object_id}; results: modal volume get {pl.VOLUME} {run_id}/summary.json")
