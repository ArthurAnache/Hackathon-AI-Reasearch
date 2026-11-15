"""ModalExecutor: the same Job -> JobResult as LocalExecutor, run in Modal containers.

This replaces a separate benchmark script: Modal is just another backend, so the research loop does
not care where work runs. Each job runs `execution.run_job` in a container that holds this repo.

One-time setup (needs `pip install modal` and `modal token new`):
    modal deploy execution/modal_backend.py
Then:
    python -m evolution --problem 1a --executor modal --workers 24

UNTESTED against live Modal in this repo (Modal is not installed here). The tests drive ModalExecutor
through an injected stand-in for the deployed function, which exercises everything except the network.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "ws2-autoresearch"
FUNCTION_NAME = "run_job_remote"
CPU_PER_JOB = 1      # WS3 measured 94-96% CPU per single-threaded NumPy chain: cpu=2 was wasted
MAX_JOB_SECONDS = 3600


def build_app():
    """The Modal app: a numpy image with this repo mounted, and one function that runs a job."""
    import modal
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("numpy", "scipy")
             .add_local_dir(str(ROOT), "/root/repo",
                            ignore=[".git", ".venv", "__pycache__", "certificates", "reports", "solutions"]))
    app = modal.App(APP_NAME)

    @app.function(image=image, cpu=CPU_PER_JOB, timeout=MAX_JOB_SECONDS, serialized=True)
    def run_job_remote(job: dict) -> dict:
        import sys
        sys.path.insert(0, "/root/repo")
        from execution import run_job
        return run_job(job)

    return app, run_job_remote


class ModalExecutor:
    name = "modal"

    def __init__(self, function=None):
        """`function` is the deployed Modal function, or any object with .remote(job) and .map(jobs)."""
        self._fn = function

    def _function(self):
        if self._fn is None:
            import modal
            lookup = getattr(modal.Function, "from_name", None) or modal.Function.lookup
            self._fn = lookup(APP_NAME, FUNCTION_NAME)
        return self._fn

    def submit(self, job):
        return self.map([job])[0]

    def map(self, jobs):
        from . import JobResult
        if not jobs:
            return []
        raw = list(self._function().map([asdict(j) for j in jobs], return_exceptions=True))
        out = []
        for job, r in zip(jobs, raw):
            if isinstance(r, BaseException) or not isinstance(r, dict):
                r = {"job_id": job.job_id, "status": "crash", "candidate": None, "duration_s": 0.0,
                     "worker_id": "modal", "error": f"modal: {type(r).__name__}: {r}", "meta": job.meta}
            out.append(JobResult(**r))
        return out


try:  # `modal deploy execution/modal_backend.py` needs a module-level app
    import modal as _modal  # noqa: F401
    app, run_job_remote = build_app()
except ImportError:
    app = run_job_remote = None
