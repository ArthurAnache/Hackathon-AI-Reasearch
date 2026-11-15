"""Modal app for the autoloop, DEPLOYED (not `modal run`), so jobs never depend on a local client.

    modal deploy autoloop/modal_app.py          # once per code change; the commit is recorded in the run ledger

Why: on 2026-11-14 every call of the `modal run --detach` clients (shared app name revolutionaries-c1d) was cancelled
when the local client starved or lost its link. Here the executor (autoloop/modal_exec.py) only *spawns* calls on the
deployed function, writes each call id to disk, and later polls and pulls. Each job also writes its output and its
checkpoints (every 120 s) to the Volume `revolutionaries-autoloop`, so a result survives a lost client.

One function, run_job(job, start), runs one stage job in a 1-CPU container:
  seed1d   -> autoloop/seed_worker.py      (WS2 wt-compare/optimizers/1d_tuned.py, copied into the image)
  refine1d -> attack/c1d_refine.py         (Brasa's file, copied unchanged)
  seed1a   -> autoloop/tao1a.py seed       (WS2 1a_tuned.py)
  refine1a -> autoloop/tao1a.py refine
Own app name and own Volume: nothing is shared with attack/modal_c1d.py's app `revolutionaries-c1d`.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = __import__("os").environ.get("AUTOLOOP_WORKSPACE", str((lambda q: q[2] if len(q) > 2 else q[-1])(__import__("pathlib").Path(__file__).resolve().parents)))  # folder that holds repo/, wt-compare/, shared/ and .env
WS2 = Path(WORKSPACE + "/wt-compare")
APP_NAME = "revolutionaries-autoloop"
VOLUME = "revolutionaries-autoloop"
CPU, MEM_MB = 1.0, 3072
USD_PER_JOB_S = CPU * 0.0000131 + MEM_MB / 1024 * 0.00000222   # https://modal.com/pricing (read 2026-11-14)

image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.5.4", "scipy==1.18.1")
         .env({"AUTOLOOP_WS2": "/root/ws2", "PYTHONPATH": "/root"})
         .add_local_file(str(REPO / "attack" / "c1d_refine.py"), "/root/attack/c1d_refine.py")
         .add_local_file(str(REPO / "attack" / "c1d_exact.py"), "/root/attack/c1d_exact.py")
         .add_local_file(str(WS2 / "optimizers" / "1d_tuned.py"), "/root/ws2/optimizers/1d_tuned.py")
         .add_local_file(str(WS2 / "optimizers" / "1a_tuned.py"), "/root/ws2/optimizers/1a_tuned.py")
         .add_local_file(str(HERE / "modal_stub_prepare.py"), "/root/ws2/prepare.py")
         .add_local_dir(str(REPO / "problems"), "/root/problems", ignore=["__pycache__", "**/__pycache__"])
         .add_local_dir(str(HERE), "/root/autoloop", ignore=["runs", "runs/**", "__pycache__", "**/__pycache__", "tests"]))
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
app = modal.App(APP_NAME)


def _command(job, start_path, out):
    k, a = job["kind"], job
    if k == "seed1d":
        return ["python", "-m", "autoloop.seed_worker", "--cells", str(a["cells"]), "--budget", str(a["budget"]),
                "--seed", str(a["seed"]), "--out", out]
    if k == "refine1d":
        return ["python", "/root/attack/c1d_refine.py", "--start", start_path, "--out", out, "--split", str(a["split"]),
                "--p0", str(a["p0"]), "--budget", str(a["budget"]), "--threads", "1", "--quiet", "--ckpt-every", "120"]
    if k == "seed1a":
        return ["python", "-m", "autoloop.tao1a", "seed", "--cells", str(a["cells"]), "--budget", str(a["budget"]),
                "--seed", str(a["seed"]), "--out", out]
    if k == "refine1a":
        return ["python", "-m", "autoloop.tao1a", "refine", "--start", start_path, "--out", out,
                "--split", str(a["split"]), "--p0", str(a["p0"]), "--budget", str(a["budget"])]
    if k == "refine3":
        return ["python", "-m", "autoloop.ea_third_autocorr", "refine", "--start", start_path, "--out", out,
                "--split", str(a["split"]), "--p0", str(a["p0"]), "--budget", str(a["budget"])]
    if k.startswith("pack_seed:"):
        return ["python", "-m", "autoloop.ea_default", "seed", "--slug", k.split(":", 1)[1], "--out", out,
                "--seed", str(a["seed"]), "--budget", str(a["budget"])]
    if k.startswith("pack_stage:"):
        return ["python", "-m", "autoloop.ea_default", "stage", "--slug", k.split(":", 1)[1], "--out", out,
                *(["--start", start_path] if start_path else []), "--code", "/tmp/model_stage.py",
                "--seed", str(a["seed"]), "--budget", str(a["budget"])]
    if k.startswith("pack_refine:"):
        return ["python", "-m", "autoloop.ea_default", "refine", "--slug", k.split(":", 1)[1], "--out", out,
                "--start", start_path, "--p0", str(a["p0"]), "--budget", str(a["budget"])]
    raise ValueError(f"unknown job kind {k}")


@app.function(image=image, cpu=CPU, memory=MEM_MB, timeout=3900, retries=0, max_containers=100, volumes={"/vol": vol})
def run_job(job: dict, start: bytes | None = None) -> dict:
    d = Path("/vol") / job["run"]; d.mkdir(parents=True, exist_ok=True)
    out = str(d / job["key"])
    start_path = None
    if start is not None:
        start_path = f"/tmp/{job['key']}_start.npy"; Path(start_path).write_bytes(start)
    if job.get("code"):                                    # model-written stage: code only, no secret in this container
        Path("/tmp/model_stage.py").write_text(job["code"])
    stop = threading.Event()

    def committer():
        while not stop.wait(120):
            try:
                vol.commit()
            except Exception:  # noqa: BLE001
                pass
    threading.Thread(target=committer, daemon=True).start()
    t0 = time.time()
    p = subprocess.run(_command(job, start_path, out), capture_output=True, text=True, cwd="/root",
                       timeout=float(job["budget"]) * 2 + 600)
    stop.set(); vol.commit()
    ok = os.path.exists(out + ".npy") and os.path.exists(out + ".json")
    res = dict(key=job["key"], ok=ok, returncode=p.returncode, seconds=round(time.time() - t0, 1),
               err_tail=(p.stderr or "")[-300:] if not ok else "")
    if ok:
        import json
        res["summary"] = json.load(open(out + ".json"))
        res["npy"] = Path(out + ".npy").read_bytes()
    return res
