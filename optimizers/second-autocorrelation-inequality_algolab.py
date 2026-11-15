"""WS3-compatible adapter: run an Algorithm Lab plugin under the existing search(budget_s, seed, config) contract.

CONFIG["PLUGIN_SOURCE"] holds the plugin source (string). CONFIG["PLUGIN_SHA256"] must match. The adapter writes the
plugin into a per-job directory, runs algolab.child in a subprocess (-I), and returns the best array. The parent
re-verifies with the frozen verifier; this file never claims a record.

Trusted control plugins may set TRUSTED=True (skip gate; still no network secrets in the process env we pass).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROBLEM = "second-autocorrelation-inequality"
CONFIG = {
    "PLUGIN_SOURCE": None,
    "PLUGIN_SHA256": None,
    "PLUGIN_PARAMS": {},
    "TRUSTED": False,
    "INIT": None,
}
SEARCH_SPACE = {}  # plugins carry their own PARAMS; overrides go through PLUGIN_PARAMS


def _repo_root() -> Path:
    """Locate the repo root that contains algolab/child.py.

    On Modal, execution.run_job copies this adapter into a temp file, so
    Path(__file__).parent.parent is NOT the repo. Prefer PYTHONPATH / /root/repo.
    """
    candidates: list[Path] = []
    for entry in (os.environ.get("PYTHONPATH") or "").split(os.pathsep):
        if entry:
            candidates.append(Path(entry))
    candidates.append(Path("/root/repo"))
    here = Path(__file__).resolve()
    candidates.extend([here.parent.parent, *list(here.parents)[:6]])
    for cand in candidates:
        if (cand / "algolab" / "child.py").is_file():
            return cand
    raise FileNotFoundError("cannot locate algolab/child.py (PYTHONPATH / /root/repo)")


def _load_start(config) -> np.ndarray:
    if config.get("INIT"):
        with open(config["INIT"]) as fh:
            values = json.load(fh)["solution"]["values"]
    else:
        from problems.catalog import load
        values = load(PROBLEM).certificate()["values"]
    f = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    return f / f.max()


def search(budget_s, seed, config=CONFIG):
    cfg = {**CONFIG, **(config or {})}
    src = cfg.get("PLUGIN_SOURCE")
    sha = cfg.get("PLUGIN_SHA256")
    if not src or not sha:
        raise ValueError("algolab adapter requires PLUGIN_SOURCE and PLUGIN_SHA256")
    if hashlib.sha256(src.encode()).hexdigest() != sha:
        raise ValueError("PLUGIN_SOURCE does not match PLUGIN_SHA256")
    f0 = _load_start(cfg)
    root = _repo_root()
    with tempfile.TemporaryDirectory(prefix="algolab_job_") as tmp:
        job = Path(tmp)
        plugin = job / "plugin.py"
        plugin.write_text(src, encoding="utf-8", newline="\n")
        start = job / "start.npy"
        np.save(start, f0)
        spec = {
            "job_dir": str(job),
            "plugin": str(plugin),
            "plugin_sha256": sha,
            "trusted": bool(cfg.get("TRUSTED")),
            "seed": int(seed),
            "budget_s": float(budget_s),
            "start_npy": str(start),
            "params": dict(cfg.get("PLUGIN_PARAMS") or {}),
        }
        (job / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        env = {k: os.environ[k] for k in ("SYSTEMROOT", "TEMP", "TMP", "PATH", "PYTHONPATH",
                                          "HOME", "USERPROFILE", "USERNAME") if k in os.environ}
        # Strip secrets from the child environment even if the parent somehow has them.
        for k in list(env):
            if any(s in k.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")):
                env.pop(k, None)
        # Keep Modal image PYTHONPATH (/root/repo) first so child can import algolab.kit.
        pp = env.get("PYTHONPATH", "")
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONPATH=str(root) + (os.pathsep + pp if pp else ""))
        child = root / "algolab" / "child.py"
        if not child.is_file():
            raise FileNotFoundError(f"algolab child missing at {child}")
        t0 = time.monotonic()
        proc = subprocess.run([sys.executable, "-I", str(child), str(job / "spec.json")],
                              cwd=job, env=env, capture_output=True, text=True,
                              timeout=float(budget_s) * 1.5 + 90)
        if proc.returncode != 0:
            raise RuntimeError(f"algolab child exit {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}")
        best = np.load(job / "best.npy")
        print(f"algolab wall {time.monotonic() - t0:.1f}s n={best.size}", flush=True)
        return {"values": np.maximum(best, 0.0).tolist()}


if __name__ == "__main__":
    from prepare import run
    run(__file__)
