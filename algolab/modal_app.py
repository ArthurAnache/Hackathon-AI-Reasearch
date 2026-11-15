"""Modal app for Algorithm Lab plugins: one job per container, no network, no secrets.

Launched only through `python -m algolab.campaign --confirm-modal` (never auto-started from this module).
`block_network=True` is the real isolation boundary for untrusted plugin code; the AST gate and audit hook are
defence in depth, not a substitute.
"""
from __future__ import annotations

import os

from algolab import image as IMG

APP_NAME = "algolab-plugin-bench"
FUNCTION_NAME = "run_algolab_job"
VOLUME_NAME = "algolab-results"
VOLUME_MOUNT = "/vol"
CPU = 1.0
MEMORY_MB = 8192          # LP / packet responses need more than the 4 GiB WS3 default
FUNCTION_TIMEOUT_S = 2400  # allows T3 budgets up to ~15 min with the worker's 2x kill
MAX_CONTAINERS = 40
SLUG = os.environ.get("ALGOLAB_PROBLEM", "second-autocorrelation-inequality")


def build_image(modal, slug: str = SLUG):
    image = (modal.Image.debian_slim(python_version=IMG.PYTHON_VERSION)
             .pip_install(*IMG.PIP_PINS)
             .env(IMG.IMAGE_ENV))
    for local, remote in IMG.image_files(slug):
        image = image.add_local_file(str(local), remote)
    return image


def spec() -> dict:
    return {"app": APP_NAME, "function": FUNCTION_NAME, "volume": VOLUME_NAME, "volume_mount": VOLUME_MOUNT,
            "cpu": CPU, "memory_mb": MEMORY_MB, "timeout_s": FUNCTION_TIMEOUT_S, "max_containers": MAX_CONTAINERS,
            "retries": 0, "secrets": [], "block_network": True,
            "python": IMG.PYTHON_VERSION, "pip": list(IMG.PIP_PINS), "env": IMG.IMAGE_ENV,
            "files": IMG.relative_files(SLUG)}


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App(APP_NAME)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    @app.function(image=build_image(modal), cpu=CPU, memory=MEMORY_MB, timeout=FUNCTION_TIMEOUT_S,
                  volumes={VOLUME_MOUNT: volume}, retries=0, max_containers=MAX_CONTAINERS,
                  include_source=False, name=FUNCTION_NAME,
                  # Real isolation boundary for untrusted LLM plugins:
                  block_network=True)
    def run_algolab_job(job: dict) -> dict:
        # No secrets are attached to this function. Plugin source travels in job["config"]["PLUGIN_SOURCE"].
        from ws3_bench.worker import execute
        return execute(job, VOLUME_MOUNT, commit=volume.commit)
else:
    app = volume = run_algolab_job = None
