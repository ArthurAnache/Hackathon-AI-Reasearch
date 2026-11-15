"""WS3 Modal app: one function that runs a V2 job with ws3_bench.worker and persists it to a Volume.

Launched only through `python -m ws3_bench run --executor modal --confirm-modal` (ephemeral app, detached so
containers finish and persist even if the local process dies). No `modal deploy` is needed.

Resources per job: cpu=1, memory=4096 MiB, single-threaded NumPy/SciPy, Python 3.11, numpy 2.2.6, scipy 1.16.2.
No secrets are attached; the image holds only ws3_bench.image.relative_files(slug).
"""
from __future__ import annotations

import os

from ws3_bench import image as IMG

APP_NAME = "ws3-v2-bench"
FUNCTION_NAME = "run_job_ws3"
VOLUME_NAME = "ws3-v2-results"
VOLUME_MOUNT = "/vol"
CPU = 1.0
MEMORY_MB = 4096
FUNCTION_TIMEOUT_S = 900          # covers the worker's own kill: 2 x budget + 30 + 90, for budgets up to 380 s
MAX_CONTAINERS = 40          # one wave is up to 40 jobs; they run concurrently, not capped at 6 or 12
SLUG = os.environ.get("WS3_BENCH_PROBLEM", "second-autocorrelation-inequality")


def max_budget_s() -> float:
    from ws3_bench.worker import HARD_KILL_FACTOR, HARD_KILL_GRACE_S, JOIN_GRACE_S
    return (FUNCTION_TIMEOUT_S - HARD_KILL_GRACE_S - JOIN_GRACE_S - 60) / HARD_KILL_FACTOR


def build_image(modal, slug: str = SLUG):
    image = (modal.Image.debian_slim(python_version=IMG.PYTHON_VERSION)
             .pip_install(*IMG.PIP_PINS)
             .env(IMG.IMAGE_ENV))
    for local, remote in IMG.image_files(slug):
        image = image.add_local_file(str(local), remote)
    return image


def spec() -> dict:
    """Everything that defines the execution environment (recorded in every manifest)."""
    return {"app": APP_NAME, "function": FUNCTION_NAME, "volume": VOLUME_NAME, "volume_mount": VOLUME_MOUNT,
            "cpu": CPU, "memory_mb": MEMORY_MB, "timeout_s": FUNCTION_TIMEOUT_S, "max_containers": MAX_CONTAINERS,
            "retries": 0, "secrets": [], "python": IMG.PYTHON_VERSION, "pip": list(IMG.PIP_PINS),
            "env": IMG.IMAGE_ENV, "files": IMG.relative_files(SLUG)}


try:
    import modal
except ImportError:          # local tests without the Modal SDK
    modal = None

if modal is not None:
    app = modal.App(APP_NAME)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    @app.function(image=build_image(modal), cpu=CPU, memory=MEMORY_MB, timeout=FUNCTION_TIMEOUT_S,
                  volumes={VOLUME_MOUNT: volume}, retries=0, max_containers=MAX_CONTAINERS,
                  include_source=False, name=FUNCTION_NAME)
    def run_job_ws3(job: dict) -> dict:
        from ws3_bench.worker import execute
        return execute(job, VOLUME_MOUNT, commit=volume.commit)
else:
    app = volume = run_job_ws3 = None
