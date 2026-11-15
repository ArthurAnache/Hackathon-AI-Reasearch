"""SAVANT V3 Modal app — lives under ws3_bench so it ships in the image allowlist."""
from __future__ import annotations

import os

from ws3_bench import image as IMG

APP_NAME = "savant-v3-bench"
FUNCTION_NAME = "run_job_v3"
VOLUME_NAME = "savant-v3-results"
VOLUME_MOUNT = "/vol"
CPU = 1.0
MEMORY_MB = 4096
FUNCTION_TIMEOUT_S = 3600  # must exceed job budget_s + warm-start/score overhead
MAX_CONTAINERS = 40
SLUG = os.environ.get("WS3_BENCH_PROBLEM", "second-autocorrelation-inequality")


def build_image(modal, slug: str = SLUG):
    image = (modal.Image.debian_slim(python_version=IMG.PYTHON_VERSION)
             .pip_install(*IMG.PIP_PINS)
             .env(IMG.IMAGE_ENV))
    for local, remote in IMG.image_files(slug):
        image = image.add_local_file(str(local), remote)
    return image


def spec() -> dict:
    return {
        "app": APP_NAME, "function": FUNCTION_NAME, "volume": VOLUME_NAME,
        "volume_mount": VOLUME_MOUNT, "cpu": CPU, "memory_mb": MEMORY_MB,
        "timeout_s": FUNCTION_TIMEOUT_S, "max_containers": MAX_CONTAINERS,
        "retries": 0, "secrets": [], "python": IMG.PYTHON_VERSION,
        "pip": list(IMG.PIP_PINS), "env": IMG.IMAGE_ENV, "files": IMG.relative_files(SLUG),
    }


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App(APP_NAME)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    @app.function(image=build_image(modal), cpu=CPU, memory=MEMORY_MB, timeout=FUNCTION_TIMEOUT_S,
                  volumes={VOLUME_MOUNT: volume}, retries=0, max_containers=MAX_CONTAINERS,
                  include_source=False, name=FUNCTION_NAME)
    def run_job_v3(job: dict) -> dict:
        from ws3_bench.worker import execute
        return execute(job, VOLUME_MOUNT, commit=volume.commit)

    @app.function(image=build_image(modal), cpu=0.125, memory=512, timeout=60,
                  include_source=False, name="import_smoke_v3")
    def import_smoke_v3() -> dict:
        """Cheap packaging check: savant_v3 must import inside the image."""
        import savant_v3
        import ws3_bench.v3_modal_app as m
        return {"ok": True, "savant_v3": getattr(savant_v3, "__version__", None), "app": m.APP_NAME}
else:
    app = volume = run_job_v3 = import_smoke_v3 = None
