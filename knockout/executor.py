"""Executor interface for evaluation jobs (section 5.12): a local process pool, or Modal workers.

Both return concurrent.futures.Future[SandboxResult], so the loop does not care where a job runs.
"""
from __future__ import annotations

import concurrent.futures as cf
from dataclasses import asdict
from typing import Protocol

from knockout.sandbox import SandboxJob, SandboxResult, run_job


class Executor(Protocol):
    name: str
    workers: int

    def submit(self, job: SandboxJob) -> "cf.Future[SandboxResult]": ...

    def cost_usd(self, result: SandboxResult) -> float: ...

    def shutdown(self) -> None: ...


class LocalExecutor:
    """Each job runs `run_job` in its own child process; threads only wait on those processes."""

    name = "local"

    def __init__(self, workers: int, sandbox_cfg: dict):
        self.workers = workers
        self.sandbox_cfg = sandbox_cfg
        self._pool = cf.ThreadPoolExecutor(max_workers=workers)

    def submit(self, job: SandboxJob) -> "cf.Future[SandboxResult]":
        return self._pool.submit(run_job, job, **self.sandbox_cfg)

    def cost_usd(self, result: SandboxResult) -> float:
        return 0.0

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


class ModalExecutor:
    """Jobs go to the `evaluate_job` Modal function (CPU, block_network=True, no secrets).

    Built by the orchestrator function in knockout/modal_app.py, which passes its own hydrated `evaluate_job`.
    The cost is an estimate from the configured per-core and per-GiB prices; Modal's dashboard has the real bill.
    """

    name = "modal"

    def __init__(self, fn, workers: int, sandbox_cfg: dict, cpu: float, memory_mb: int, usd_per_core_s: float,
                 usd_per_gib_s: float, overhead_s: float):
        self.workers = workers
        self.sandbox_cfg = sandbox_cfg
        self._fn = fn
        self._rate = cpu * usd_per_core_s + (memory_mb / 1024) * usd_per_gib_s
        self._overhead_s = overhead_s
        self._pool = cf.ThreadPoolExecutor(max_workers=workers)

    def _call(self, job: SandboxJob) -> SandboxResult:
        out = self._fn.remote(asdict(job), self.sandbox_cfg)
        return SandboxResult(**out)

    def submit(self, job: SandboxJob) -> "cf.Future[SandboxResult]":
        return self._pool.submit(self._call, job)

    def cost_usd(self, result: SandboxResult) -> float:
        return (result.elapsed_s + self._overhead_s) * self._rate

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


def sandbox_cfg(strategy) -> dict:
    return {k: strategy[f"sandbox.{k}"] for k in ("timeout_slack_s", "memory_mb", "deny_network", "max_output_mb")}


def make_executor(strategy, modal_fn=None) -> Executor:
    kind = strategy["executor.kind"]
    if kind == "local":
        return LocalExecutor(int(strategy["executor.workers"]), sandbox_cfg(strategy))
    if kind == "modal":
        if modal_fn is None:
            raise ValueError("the modal executor is built by the orchestrator (knockout/modal_app.py)")
        if float(strategy["run.max_usd"]) <= 0:
            raise ValueError("a Modal run needs run.max_usd > 0")
        return ModalExecutor(modal_fn, int(strategy["modal.max_containers"]), sandbox_cfg(strategy),
                             float(strategy["modal.cpu"]), int(strategy["modal.memory_mb"]),
                             float(strategy["modal.usd_per_core_s"]), float(strategy["modal.usd_per_gib_s"]),
                             float(strategy["modal.overhead_s"]))
    raise ValueError(f"unknown executor kind {kind!r}")
