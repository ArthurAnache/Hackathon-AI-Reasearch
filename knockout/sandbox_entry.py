"""Trusted entry point that runs ONE untrusted candidate program in a child process.

    python -I sandbox_entry.py <job_dir>

<job_dir>/job.json describes the job; the program is <job_dir>/program.py (and optionally adapter.py and init.json).
The result is written to <job_dir>/out.json. The untrusted program shares this process with the guide objective,
so it could fake the guide; that is harmless because the parent re-scores every returned candidate with the
frozen verifier in a separate, trusted process. The guide only steers the tuner.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve(ref: str):
    mod, _, attr = ref.partition(":")
    return getattr(importlib.import_module(mod), attr)


def _jsonable(obj):
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


def main(job_dir: str) -> int:
    t0 = time.monotonic()
    d = Path(job_dir)
    job = json.loads((d / "job.json").read_text())
    sys.path.insert(0, job["repo_root"])
    out: dict = {"ok": False, "error": None, "trials": [], "n_evals": 0, "tuner": None, "tuner_errors": []}
    try:
        from knockout.tuner import ParamSpace, tune

        program = _load(d / "program.py", "candidate_program")
        if not callable(getattr(program, "construct", None)):
            raise TypeError("program must define construct(params, seed, budget_s, init=None)")
        adapter = _load(d / "adapter.py", "candidate_adapter").to_canonical if (d / "adapter.py").exists() else None
        init = json.loads((d / "init.json").read_text()) if (d / "init.json").exists() else None
        guide = _resolve(job["guide_ref"])
        guide_args = job.get("guide_args", [])
        sign = 1.0 if job["direction"] == "maximize" else -1.0
        seed = int(job["seed"])

        def objective(params: dict, call_s: float):
            obj = program.construct(dict(params), seed, float(call_s), init)
            cand = adapter(obj) if adapter else obj
            cand = json.loads(json.dumps(cand, default=_jsonable))  # canonical JSON form, as the parent will see it
            return sign * float(guide(cand, *guide_args)), cand

        if job.get("mode") == "construct":
            params = {**{k: v[0] for k, v in getattr(program, "PARAMS", {}).items()}, **job.get("params", {})}
            value, cand = objective(params, float(job["budget_s"]))
            out["trials"] = [{"params": params, "guide": sign * value, "candidate": cand}]
            out["n_evals"], out["tuner"] = 1, "none"
        else:
            space = ParamSpace(getattr(program, "PARAMS", {}))
            t = job.get("tuner", {})
            res = tune(objective, space, seed=seed, budget_s=float(job["budget_s"]), method=t.get("method", "cma"),
                       sigma0=float(t.get("sigma0", 0.25)), calls=int(t.get("calls", 8)), top_k=int(t.get("top_k", 3)))
            out["trials"] = [{"params": tr.params, "guide": sign * tr.value, "candidate": tr.candidate}
                             for tr in res.trials_top if math.isfinite(tr.value)]
            out["n_evals"], out["tuner"], out["tuner_errors"] = res.n_evals, res.method, res.errors[:5]
            if not out["trials"]:
                raise RuntimeError("no valid trial; first errors: " + " | ".join(res.errors[:3]))
        out["ok"] = True
    except BaseException as exc:  # noqa: BLE001  (report everything, including SystemExit from the program)
        out["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        out["traceback"] = traceback.format_exc()[-4000:]
    out["elapsed_s"] = time.monotonic() - t0
    (d / "out.json").write_text(json.dumps(out, default=_jsonable))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
