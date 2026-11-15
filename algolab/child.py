"""Sandboxed child process: run one plugin from one start point for one budget, write best.npy + trace.json.

Started by algolab.bench as `python -I algolab/child.py <spec.json>`. Isolation layers inside this process:
  2. the plugin's sha256 must equal the admitted one, and the static gate is re-run on the exact bytes;
  3. a PEP 578 audit hook (installed before the plugin is executed) refuses network, subprocesses, ctypes,
     file deletion/renaming, writes outside the job directory and imports outside the whitelist.
Layer 4 (truth) is in the parent: the returned array is data, re-scored there by the frozen verifier.
An audit hook is not a security boundary against a determined attacker; on Modal the container itself
(no network, ephemeral filesystem, CPU/memory caps) is the real boundary. This file keeps honest mistakes and
casual escapes out of the experiment record.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import scipy.fft, scipy.interpolate, scipy.linalg, scipy.ndimage, scipy.optimize, scipy.signal  # noqa: E401,E402,F401
import scipy.sparse, scipy.sparse.linalg  # noqa: E401,E402,F401

from algolab import gate, kit  # noqa: E402

MAX_VALUES = 2_000_000
ALLOWED_IMPORT_ROOTS = {"numpy", "scipy", "math", "time", "heapq", "itertools", "functools", "bisect", "collections",
                        "dataclasses", "typing", "algolab"}
BLOCKED_EVENTS = ("socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork",
                  "os.kill", "ctypes.", "os.remove", "os.unlink", "os.rename", "os.rmdir", "os.chdir", "os.chmod",
                  "shutil.", "winreg.", "urllib.", "webbrowser.", "_winapi.", "msvcrt.")


def install_hook(job_dir: Path, trusted: bool) -> None:
    job = str(job_dir.resolve()).lower()
    prefixes = tuple(str(Path(p).resolve()).lower() for p in {sys.prefix, sys.base_prefix, sys.exec_prefix}) + (
        str(ROOT.resolve()).lower(),)

    def hook(event, args):
        if event.startswith(BLOCKED_EVENTS):
            raise PermissionError(f"sandbox: {event} refused")
        if event == "open":
            path, mode = args[0], args[1]
            if not isinstance(path, str):
                return
            p = str(Path(path).resolve()).lower()
            writing = isinstance(mode, str) and any(ch in mode for ch in "wax+")
            if isinstance(mode, int):
                writing = bool(mode & 0b11)  # os.open flags O_WRONLY / O_RDWR
            if writing and not p.startswith(job):
                raise PermissionError(f"sandbox: write to {path} refused")
            if not writing and not (p.startswith(prefixes) or p.startswith(job)):
                raise PermissionError(f"sandbox: read of {path} refused")
        if event == "import" and not trusted:
            root = str(args[0]).split(".")[0]
            if root and not root.startswith("_") and root not in ALLOWED_IMPORT_ROOTS:
                raise PermissionError(f"sandbox: import {args[0]} refused")

    sys.addaudithook(hook)


def clean(x, n_cap: int = MAX_VALUES) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    if not (1 <= a.size <= n_cap) or not np.all(np.isfinite(a)) or a.min() < -1e-6 or a.max() <= 0:
        raise ValueError(f"invalid plugin output: size {a.size}, finite {bool(np.all(np.isfinite(a)))}")
    return np.maximum(a, 0.0)


def main(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    job_dir = Path(spec["job_dir"])
    trace: dict = {"spec": spec, "events": [], "errors": []}
    t0 = time.monotonic()
    budget = float(spec["budget_s"])
    hard_end = t0 + budget
    f0 = np.load(spec["start_npy"])
    src = Path(spec["plugin"]).read_text(encoding="utf-8")
    if not spec.get("trusted"):
        if hashlib.sha256(src.encode()).hexdigest() != spec["plugin_sha256"]:
            raise SystemExit("plugin changed since admission")
        reasons = gate.check_source(src)
        if reasons:
            raise SystemExit(f"gate: {reasons}")
    install_hook(job_dir, bool(spec.get("trusted")))
    ns: dict = {"__name__": "algolab_plugin"}
    if spec.get("trusted"):
        ns["__file__"] = spec["plugin"]
    exec(compile(src, spec["plugin"], "exec"), ns)
    params = {**ns["PARAMS"], **(spec.get("params") or {})}
    rng = np.random.default_rng(int(spec["seed"]))
    alg = ns["Algorithm"](params, rng)

    best = f0.copy()
    best_s = kit.score(best)
    trace["start_score"] = best_s
    has_perturb = callable(getattr(alg, "perturb", None))
    first_slice = budget if not has_perturb else budget * float(params.get("FIRST_IMPROVE_FRAC", 0.5))
    n_calls = accepted = overruns = 0
    sig_hi, sig_lo = float(params.get("SIGMA_MAX", 1e-2)), float(params.get("SIGMA_MIN", 1e-5))
    cycle, k = int(params.get("SIGMA_STEPS", 6)), 0
    while time.monotonic() < hard_end - 0.5:
        slice_end = min(hard_end, time.monotonic() + (first_slice if n_calls == 0 else
                                                      max(1.0, float(params.get("IMPROVE_SLICE_S", budget / 8)))))
        try:
            if n_calls == 0 or not has_perturb:
                x, sigma = alg.improve(best.copy(), slice_end), None
            else:
                sigma = sig_hi * (sig_lo / sig_hi) ** ((k % cycle) / max(1, cycle - 1))
                k += 1
                x = alg.improve(clean(alg.perturb(best.copy(), sigma)), slice_end)
            x = clean(x)
            s = kit.score(x)
        except Exception as e:  # noqa: BLE001  a plugin failure is a result, never a crash of the harness
            trace["errors"].append(f"{type(e).__name__}: {e} | {traceback.format_exc().splitlines()[-2][:200]}")
            s, x, sigma = -math.inf, None, None
        n_calls += 1
        late = time.monotonic() - slice_end
        if late > 2.0:
            overruns += 1
        trace["events"].append({"t": round(time.monotonic() - t0, 3), "sigma": sigma, "score": s,
                                "late_s": round(late, 3)})
        if s > best_s:
            best, best_s = x, s
            accepted += 1
        if not has_perturb or len(trace["errors"]) >= 5:
            break
    np.save(job_dir / "best.npy", best)
    trace.update(best_score_kit=best_s, improve_calls=n_calls, accepted=accepted, overruns=overruns,
                 wall_s=round(time.monotonic() - t0, 3))
    (job_dir / "trace.json").write_text(json.dumps(trace, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
