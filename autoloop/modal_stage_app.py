"""Isolated Modal app for MODEL-WRITTEN stages only (Lupa's fix, 2026-11-15). Separate from modal_app.py.

    python -m modal deploy autoloop/modal_stage_app.py

Isolation, enforced by Modal and by this image, not by code inspection:
- image: debian_slim + numpy/scipy only, plus autoloop/stage_sandbox.py. No problems/ folder, no certificate, no
  other repo file. The verifier source and the format (key, shape, direction) are passed in the call.
- block_network=True: the container has no network at all.
- no volume mount, no Modal secret; the stage runs in a subprocess with an empty environment (PATH only).
- hard timeout: the function's timeout (TIMEOUT_S) plus a subprocess timeout; retries=0.
- outputs are returned by value (bytes), never written to shared storage.
Each call also returns a self-check made inside the container before the stage runs (network reachable?, problems/
or /vol present?, secret-like env names), and the ledger records it.
"""
from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP_NAME = "revolutionaries-autoloop-stage"
TIMEOUT_S = 1500
ISOLATION = dict(app=APP_NAME, function="run_stage", image="debian_slim py3.12 + numpy 2.5.4 + scipy 1.18.1 + "
                 "stage_sandbox.py only (no problems/, no certificate)", block_network=True, volumes=None, secrets=None,
                 timeout_s=TIMEOUT_S, retries=0, subprocess_env="PATH only", outputs="returned by value")

image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.5.4", "scipy==1.18.1")
         .add_local_file(str(HERE / "stage_sandbox.py"), "/sandbox/stage_sandbox.py"))
app = modal.App(APP_NAME)


def _self_check():
    import os
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=5).close(); net = True
    except OSError:
        net = False
    try:
        socket.getaddrinfo("einsteinarena.com", 443); dns = True
    except OSError:
        dns = False
    return dict(network_reachable=net, dns_resolves=dns,
                problems_dir=any(Path(p).exists() for p in ["/root/problems", "/problems", "/sandbox/problems"]),
                vol_mounted=Path("/vol").exists(),
                files_in_sandbox=sorted(os.listdir("/sandbox")),
                secret_like_env=sorted(k for k in os.environ if any(s in k.upper() for s in ("KEY", "SECRET", "PASSWORD"))
                                       and not k.startswith("MODAL_")))


@app.function(image=image, cpu=1.0, memory=3072, timeout=TIMEOUT_S, retries=0, block_network=True, max_containers=50)
def run_stage(job: dict, verifier_src: str, code: str, start: bytes | None = None) -> dict:
    import json
    import subprocess
    import tempfile
    import time
    chk = _self_check()
    d = Path(tempfile.mkdtemp())
    (d / "job.json").write_text(json.dumps(job)); (d / "verifier.py").write_text(verifier_src)
    (d / "code.py").write_text(code)
    if start is not None:
        (d / "start.npy").write_bytes(start)
    t0 = time.time()
    try:
        p = subprocess.run(["python", "-I", "/sandbox/stage_sandbox.py", str(d)], capture_output=True, text=True,
                           cwd=str(d), env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
                           timeout=min(float(job["budget"]) * 1.5 + 60, TIMEOUT_S - 60))
        rc, err = p.returncode, (p.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        rc, err = -9, "hard timeout"
    ok = (d / "out.npy").exists() and (d / "out.json").exists()
    res = dict(key=job["job"], ok=ok, returncode=rc, seconds=round(time.time() - t0, 1), self_check=chk,
               err_tail="" if ok else err)
    if ok:
        res["summary"] = json.loads((d / "out.json").read_text()); res["npy"] = (d / "out.npy").read_bytes()
    return res
