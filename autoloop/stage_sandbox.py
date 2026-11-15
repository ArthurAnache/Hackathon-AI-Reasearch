"""Runs ONE model-written stage inside the isolated Modal container (autoloop/modal_stage_app.py). Stdlib + numpy only.

    python -I stage_sandbox.py JOB_DIR      # JOB_DIR holds job.json, verifier.py, code.py, [start.npy]

The container holds this file and nothing else from the repo: no problems/ folder, no certificate, no volume, no
secret, no network. The verifier source and the format (key, shape, direction) arrive in the call. The output is
written to JOB_DIR/out.npy and JOB_DIR/out.json and returned by value by the Modal function.
"""
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
job = json.loads((d / "job.json").read_text())
key, shape, maximize = job["key"], tuple(job["shape"]), job["direction"] == "maximize"
n = int(np.prod(shape))
vns = {}
exec(compile((d / "verifier.py").read_text(), "verifier_pinned.py", "exec"), vns)   # noqa: S102


def score(v):
    try:
        s = float(vns["evaluate"]({key: np.asarray(v, dtype=np.float64).reshape(shape).tolist()}))
    except Exception:  # noqa: BLE001
        return float("inf")
    if not np.isfinite(s):
        return float("inf")
    return -s if maximize else s


ns = {}
exec(compile((d / "code.py").read_text(), "model_stage.py", "exec"), ns)            # noqa: S102 (isolated container)
v0 = np.load(d / "start.npy").astype(np.float64).ravel() if (d / "start.npy").exists() else None
s0 = score(v0) if v0 is not None else float("inf")
t0 = time.time()


def _alarm(*_):
    raise TimeoutError("stage passed its budget")


signal.signal(signal.SIGALRM, _alarm); signal.alarm(int(job["budget"] * 1.2) + 30)
try:
    w = np.asarray(ns["refine"](None if v0 is None else v0.copy(), score, float(job["budget"]),
                                np.random.default_rng(int(job["seed"]))), dtype=np.float64).ravel()
    err = None
except BaseException as e:  # noqa: BLE001
    w, err = (v0 if v0 is not None else np.zeros(n)), repr(e)[:300]
signal.alarm(0)
sw = score(w) if w.shape == (n,) else float("inf")
v = w if sw < s0 else (v0 if v0 is not None else w)
np.save(d / "out.npy", v)
(d / "out.json").write_text(json.dumps(dict(start_score=s0, stage_score=sw, kept=bool(sw < s0), error=err,
                                            seconds=time.time() - t0)))
