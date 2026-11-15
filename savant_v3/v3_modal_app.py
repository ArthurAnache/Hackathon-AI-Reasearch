"""DEPRECATED Modal entrypoint shim.

Do NOT use this module as a Modal app target. The live entrypoint is:

    ws3_bench.v3_modal_app

Early V3 pilots failed with ModuleNotFoundError: No module named 'savant_v3'
because Modal tried to import this package inside the container image (allowlist
does not ship the full savant_v3 tree). All paid dispatch uses ws3_bench.
"""
from __future__ import annotations

from ws3_bench.v3_modal_app import (  # noqa: F401
    APP_NAME,
    FUNCTION_NAME,
    VOLUME_NAME,
    VOLUME_MOUNT,
    CPU,
    MEMORY_MB,
    FUNCTION_TIMEOUT_S,
    MAX_CONTAINERS,
    SLUG,
    build_image,
    spec,
    app,
    volume,
    run_job_v3,
)

__all__ = [
    "APP_NAME", "FUNCTION_NAME", "VOLUME_NAME", "VOLUME_MOUNT",
    "CPU", "MEMORY_MB", "FUNCTION_TIMEOUT_S", "MAX_CONTAINERS", "SLUG",
    "build_image", "spec", "app", "volume", "run_job_v3",
]
