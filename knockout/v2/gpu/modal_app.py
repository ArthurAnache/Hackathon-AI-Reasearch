"""Modal app for the deep annealing chains (GPU, float64) on second-autocorrelation-inequality.

    modal run -m knockout.v2.gpu.modal_app::launch --job prepare          # store the warm starts on the Volume
    modal run -m knockout.v2.gpu.modal_app::launch --job bench            # E0: agreement and timing
    modal run --detach -m knockout.v2.gpu.modal_app::launch --job anneal --plan deep --start frozen \
        --evals 12000 --s-per-eval 0.12 --max-usd 6 --source-commit $(git rev-parse --short HEAD)
    modal volume get knockout-v2-gpu <run_id>/chains.jsonl .               # results; solutions are <id>.npy

Rules followed (CLAUDE.md, "WS3 — Modal compute and GPUs"):
- the GPU computes the smoothed objective only; every stored solution is re-scored in float64 on CPU inside the
  container with the frozen verifier before it is logged (`gpu_objective` and `score` are two fields);
- float64 on an H100; torch is pinned in requirements-gpu.txt and installed in this image only; numpy and scipy
  are the local versions, so the CPU re-score on Modal matches the local one;
- the image copies problems/ unchanged, and loading the problem re-checks the frozen sha256;
- explicit timeout and container cap on every function (at most 2 GPU containers at a time); each chain batch also
  carries its own wall-clock budget, derived from the USD cap given at launch;
- no secret in the image, no Modal secret, no write endpoint: the only network call is the read-only GET of the
  live #1 in `prepare`;
- solutions live on the Volume `knockout-v2-gpu` as float64 .npy under <run_id>/<chain>/<step>.npy;
- a claim is decided only by selection/verify_claim.py, run locally on a downloaded solution.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal

from knockout.v2.gpu import schedule as sch

REPO = Path(__file__).resolve().parents[3]
IMAGE_ROOT = "/root"
VOL_PATH = "/vol"
GPU = "H100"
GPU_CPU = 6.0
GPU_MEMORY_MB = 32768
GPU_TIMEOUT_S = 5400          # hard stop of one GPU call (90 min, about 6.7 USD)
MAX_GPU_CONTAINERS = 2
MAX_USD_PER_LAUNCH = 12.0
_IGNORE = ["**/__pycache__", "**/*.pyc", "**/.DS_Store", "results/**", "v2/results/**"]

def _gpu_requirements() -> list[str]:
    """The pinned GPU packages of requirements-gpu.txt (the file is also copied into the image, so this module
    reads the same list when Modal imports it inside the container)."""
    lines = (REPO / "requirements-gpu.txt").read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


image = (
    modal.Image.debian_slim(python_version="3.12")
    # Same versions as the local .venv (and as knockout/modal_app.py).
    .pip_install("numpy==2.5.4", "scipy==1.18.1", "cma==4.5.0", "pyyaml==6.0.3", "pydantic==2.14.3",
                 "mpmath==1.3.0", "sympy==1.14.0", "requests==2.34.2", "python-dotenv==1.2.4")
    .pip_install(*_gpu_requirements())
    .env({"PYTHONPATH": IMAGE_ROOT, "PYTHONDONTWRITEBYTECODE": "1"})
    # /root mirrors the repository: knockout/, problems/ and integrations/ next to each other.
    .add_local_dir(REPO / "knockout", f"{IMAGE_ROOT}/knockout", ignore=_IGNORE)
    .add_local_dir(REPO / "problems", f"{IMAGE_ROOT}/problems", ignore=_IGNORE)
    .add_local_dir(REPO / "integrations", f"{IMAGE_ROOT}/integrations", ignore=_IGNORE)
    .add_local_file(REPO / "requirements-gpu.txt", f"{IMAGE_ROOT}/requirements-gpu.txt")
)

app = modal.App("knockout-v2-anneal-gpu", image=image, include_source=False)
volume = modal.Volume.from_name(sch.VOLUME, create_if_missing=True)


@app.function(cpu=2.0, memory=8192, timeout=900, max_containers=1, volumes={VOL_PATH: volume})
def prepare() -> dict:
    """CPU only: store the frozen certificate and the live #1 on the Volume (one read-only GET)."""
    from knockout.v2.gpu.worker import prepare_starts

    meta = prepare_starts(Path(VOL_PATH))
    volume.commit()
    return meta


@app.function(gpu=GPU, cpu=GPU_CPU, memory=GPU_MEMORY_MB, timeout=GPU_TIMEOUT_S,
              max_containers=MAX_GPU_CONTAINERS, volumes={VOL_PATH: volume})
def gpu_job(kind: str, spec: dict) -> dict:
    """The only GPU function, so the container cap covers every GPU job: "bench" (E0) or "anneal" (a batch of
    chains). It returns small metadata; solutions stay on the Volume."""
    import threading

    from knockout.v2.gpu import worker

    volume.reload()
    stop = threading.Event()

    def committer():  # partial results become visible with `modal volume get` while the chains run
        while not stop.wait(120):
            try:
                volume.commit()
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=committer, daemon=True).start()
    try:
        if kind == "bench":
            out = worker.bench(Path(VOL_PATH), spec)
        elif kind == "anneal":
            out = worker.anneal(Path(VOL_PATH), spec)
        else:
            raise ValueError(f"unknown job {kind!r}")
    finally:
        stop.set()
        volume.commit()
    return out


@app.local_entrypoint()
def launch(job: str = "anneal", plan: str = "deep", start: str = "frozen", evals: int = 0,
           s_per_eval: float = 0.0, max_usd: float = 0.0, source_commit: str = "", tag: str = "",
           fft: str = "smooth", history: int = 8, only: str = "", wait: bool = False):
    if job == "prepare":
        print(json.dumps(prepare.remote(), indent=1))
        return
    if job == "bench":
        print(json.dumps(gpu_job.remote("bench", {"history": history}), indent=1))
        return
    if job != "anneal":
        raise SystemExit(f"unknown job {job!r}")
    if evals <= 0 or s_per_eval <= 0:
        raise SystemExit("anneal needs --evals and --s-per-eval (seconds per batched evaluation, from E0)")
    if not 0 < max_usd <= MAX_USD_PER_LAUNCH:
        raise SystemExit(f"--max-usd must be in (0, {MAX_USD_PER_LAUNCH}]")
    chains = sch.build_plan(plan, start, evals)
    if only:
        wanted = set(only.split(","))
        chains = [c for c in chains if c.name in wanted]
        sch.check_batch(chains)
    rate = sch.container_usd_per_s(GPU, GPU_CPU, GPU_MEMORY_MB / 1024)
    est = sch.estimate(evals, s_per_eval, usd_per_s=rate)
    budget_s = min(max_usd / rate, GPU_TIMEOUT_S - 300)
    print(f"estimate: {est['wall_s']:.0f} s, {est['usd']:.2f} USD at {rate * 3600:.2f} USD/h; "
          f"cap {max_usd:.2f} USD = {budget_s:.0f} s of wall clock (the chains stop there)")
    if est["usd"] > max_usd:
        raise SystemExit("the estimate is above --max-usd: lower --evals or raise the cap")
    commit = (source_commit or "nogit")[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    start_tag = start.replace("vol:", "").replace("/", "_").replace(".npy", "")[-40:]
    run_id = f"{stamp}-{tag or plan}-{start_tag}-{commit[:7]}"
    spec = {"run_id": run_id, "source_commit": commit, "plan": plan, "start": start, "evals": evals,
            "budget_s": budget_s, "max_usd": max_usd, "usd_per_s": rate, "fft": fft, "history": history,
            "chains": [c.to_dict() for c in chains]}
    print(f"run_id={run_id} chains={[c.name for c in chains]}")
    if wait:
        print(json.dumps(gpu_job.remote("anneal", spec), indent=1))
    else:
        call = gpu_job.spawn("anneal", spec)
        print(f"spawned {call.object_id}; results: modal volume get {sch.VOLUME} {run_id}/summary.json")
