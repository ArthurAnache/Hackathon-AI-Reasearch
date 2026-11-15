"""LocalExecutor: jobs run in separate local processes, `workers` at a time.

Every job gets its own process (crash isolation, its own watchdog). Time-budgeted jobs that run
side by side share the CPU, so compare arms only within the same batch or the same worker count.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict


class LocalExecutor:
    name = "local"

    def __init__(self, workers: int | None = None):
        self.workers = workers or max(1, (os.cpu_count() or 2) // 2)

    def submit(self, job):
        return self.map([job])[0]

    def map(self, jobs):
        from . import JobResult, run_job
        if not jobs:
            return []
        payloads = [asdict(j) for j in jobs]
        try:
            with ProcessPoolExecutor(max_workers=min(self.workers, len(jobs))) as pool:
                raw = list(pool.map(run_job, payloads))
        except BrokenProcessPool as e:          # a worker died hard (e.g. out of memory)
            raw = [{"job_id": j.job_id, "status": "crash", "candidate": None, "duration_s": 0.0,
                    "worker_id": "local", "error": f"worker process died: {e}", "meta": j.meta} for j in jobs]
        return [JobResult(**r) for r in raw]
