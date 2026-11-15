"""L2 EXECUTION (WS3): run jobs anywhere behind one interface.

    executor.submit(job) -> JobResult          one job
    executor.map(jobs)   -> [JobResult]        a batch, in parallel (local processes or Modal containers)

A Job is self-contained, so the same job runs identically on a laptop or in a Modal container:
    job = {problem, optimizer, optimizer_hash, optimizer_source, config, seed, budget_s, warm_start, ...}
  - optimizer_source is shipped with the job; the worker re-hashes it and refuses a mismatch;
  - warm_start is a certificate id; the parent loads and re-verifies the certificate and ships its
    solution (warm_start_solution), so a worker can only ever start from verified truth.

Workers return candidates only. Scoring is never done by a worker: the parent re-verifies every
candidate with the frozen problem (see evolution/evaluate.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

ROOT = Path(__file__).resolve().parent.parent
HARD_KILL_FACTOR, HARD_KILL_GRACE_S = 2.0, 30


def source_hash(source: str) -> str:
    """Same as interfaces.FileOptimizer.version(): sha256 of the file bytes, 12 hex chars."""
    return hashlib.sha256(source.encode()).hexdigest()[:12]


@dataclass
class Job:
    problem: str
    optimizer: str                       # path (provenance only; the code travels in optimizer_source)
    optimizer_hash: str
    optimizer_source: str
    config: dict
    seed: int
    budget_s: float
    warm_start: str | None = None        # certificate id
    warm_start_solution: dict | None = None
    job_id: str = ""
    meta: dict = field(default_factory=dict)   # arm, island, genes, ... passed through untouched


@dataclass
class JobResult:
    job_id: str
    status: str                          # ok | crash | timeout
    candidate: dict | None
    duration_s: float
    worker_id: str
    error: str | None = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Executor(Protocol):
    name: str

    def submit(self, job: Job) -> JobResult: ...

    def map(self, jobs: list[Job]) -> list[JobResult]: ...


def make_job(problem, optimizer_path: str | Path, config: dict, seed: int, budget_s: float,
             warm_start: str | None = None, certs=None, job_id: str = "", **meta) -> Job:
    """Build a job in the (trusted) parent. A warm start must be a certificate; it is re-verified here."""
    path = Path(optimizer_path)
    source = path.read_text()
    solution = None
    if warm_start:
        from certificates import CertificateStore
        solution = (certs or CertificateStore()).load(problem, warm_start).solution
    rel = str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path)
    config = {k: v for k, v in (config or {}).items() if k != "INIT"}
    return Job(problem.id, rel, source_hash(source), source, config, int(seed), float(budget_s),
               warm_start, solution, job_id, meta)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def run_job(job: dict) -> dict:
    """Execute one job inside any worker. Returns a JobResult as a dict (picklable / JSON-able)."""
    worker = f"{socket.gethostname()}:{os.getpid()}"
    t0 = time.monotonic()
    out = {"job_id": job.get("job_id", ""), "status": "crash", "candidate": None, "duration_s": 0.0,
           "worker_id": worker, "error": None, "meta": job.get("meta", {})}
    try:
        if source_hash(job["optimizer_source"]) != job["optimizer_hash"]:
            raise RuntimeError("optimizer_source does not match optimizer_hash")
        import importlib.util
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / f"opt_{job['optimizer_hash']}.py"
            src.write_text(job["optimizer_source"])
            config = dict(job["config"])
            if job.get("warm_start_solution") is not None:
                init = Path(tmp) / "warm_start.json"
                init.write_text(json.dumps({"solution": job["warm_start_solution"]}))
                config["INIT"] = str(init)
            spec = importlib.util.spec_from_file_location(f"job_opt_{job['optimizer_hash']}", src)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if module.PROBLEM != job["problem"]:
                raise ValueError(f"optimizer is for {module.PROBLEM}, job is for {job['problem']}")
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(int(job["budget_s"] * HARD_KILL_FACTOR + HARD_KILL_GRACE_S))
            try:
                out["candidate"] = module.search(job["budget_s"], job["seed"], {**module.CONFIG, **config})
                out["status"] = "ok"
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
    except _Timeout:
        out["status"], out["error"] = "timeout", f"killed after {HARD_KILL_FACTOR}x budget + {HARD_KILL_GRACE_S}s"
    except Exception as e:  # noqa: BLE001  a crash is a result
        out["error"] = f"{type(e).__name__}: {e} | " + traceback.format_exc().strip().splitlines()[-1][:200]
    out["duration_s"] = round(time.monotonic() - t0, 3)
    return out


def to_dict(job: Job) -> dict:
    return asdict(job)


from .local import LocalExecutor  # noqa: E402
from .modal_backend import ModalExecutor  # noqa: E402

__all__ = ["Job", "JobResult", "Executor", "LocalExecutor", "ModalExecutor", "make_job", "run_job", "source_hash"]
