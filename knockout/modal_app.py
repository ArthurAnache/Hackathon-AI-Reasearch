"""Modal app: the orchestrator runs the loop; evaluation workers run untrusted programs (CPU, no network, no secrets).

    modal run --detach -m knockout.modal_app::launch                 # parameters from knockout/launch.yaml
    modal run --detach -m knockout.modal_app::launch --source-commit <sha>
    modal volume get knockout-runs <run_id> ./runs/                  # fetch a run's ledger and solutions

LLM key: when launch.yaml asks for the LLM editor, the launcher reads OPENAI_API_KEY from its own environment (a
GitHub Actions secret) or from the repo's .env, and hands it to the orchestrator function only, as an ephemeral
Modal secret. Nothing else from .env is read, and the key is never written to the image or to the Volume.

Rules followed (CLAUDE.md, "WS3 — Modal compute and GPUs"):
- The image copies problems/ unchanged; loading a problem re-checks the frozen sha256 (a modified one refuses to load).
- Workers have block_network=True, no Modal secret and an explicit timeout and container cap. LLM-written programs
  run only there. Nothing here submits.
- Every candidate is re-scored in float64 on CPU with the frozen verifier by the orchestrator before it is logged.
- Solutions are stored on a Modal Volume under <run_id>/solutions/<chain>/<experiment_id>.json.
- Claims are decided only by selection/verify_claim.py, run locally on downloaded solutions.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import modal
import yaml

REPO = Path(__file__).resolve().parent.parent
_CFG = yaml.safe_load((Path(__file__).with_name("strategy.yaml")).read_text())["modal"]
_IGNORE = ["**/__pycache__", "**/*.pyc", "**/.DS_Store"]
IMAGE_ROOT = "/root"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Same versions as the local .venv, so the CPU re-score on Modal matches the local one.
    .pip_install("numpy==2.5.4", "scipy==1.18.1", "cma==4.5.0", "pyyaml==6.0.3", "pydantic==2.14.3",
                 "mpmath==1.3.0", "sympy==1.14.0", "requests==2.34.2", "python-dotenv==1.2.4", "openai==3.51.0")
    # One root for everything, /root, laid out like the repository: the code finds problems/, integrations/ and
    # docs/ next to knockout/ (Path(__file__).parent.parent), in the orchestrator and in the sandboxed child alike.
    # (First Modal run, 2026-11-14: the data sat under /root/repo while Modal mounted the package at
    # /root/knockout, so problems/<slug>/meta.json was not found.)
    .env({"PYTHONPATH": IMAGE_ROOT, "PYTHONDONTWRITEBYTECODE": "1"})
    .add_local_dir(REPO / "knockout", f"{IMAGE_ROOT}/knockout", ignore=_IGNORE)
    .add_local_dir(REPO / "problems", f"{IMAGE_ROOT}/problems", ignore=_IGNORE)
    # integrations/ is needed only for the read-only EinsteinArena reads (live #1, leaderboard). No key is in the
    # image.
    .add_local_dir(REPO / "integrations", f"{IMAGE_ROOT}/integrations", ignore=_IGNORE)
    # Curated forum knowledge, quoted in the LLM prompts as data (knockout/llm_editor.py: problem_brief).
    .add_local_file(REPO / "docs" / "ws1_knowledge.jsonl", f"{IMAGE_ROOT}/docs/ws1_knowledge.jsonl")
)


def _llm_key() -> str:
    """OPENAI_API_KEY of the launcher: its environment first, then the repo's .env (that one key only)."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    env_file = REPO / ".env"
    if not key and env_file.exists():
        from dotenv import dotenv_values

        key = (dotenv_values(env_file).get("OPENAI_API_KEY") or "").strip()
    return key


def _past_key() -> str:
    """PAST_API_KEY of the launcher (environment, then .env), for the shared memory (knockout/memory.py)."""
    key = os.environ.get("PAST_API_KEY", "").strip()
    env_file = REPO / ".env"
    if not key and env_file.exists():
        from dotenv import dotenv_values

        key = (dotenv_values(env_file).get("PAST_API_KEY") or "").strip()
    return key


# Attached to the orchestrator only (never to evaluate_job). Inside a container this resolves to the values Modal
# injected at launch. PAST_API_KEY is empty when memory is not used: the loop then falls back to its local store.
llm_secret = modal.Secret.from_dict({"OPENAI_API_KEY": _llm_key(), "PAST_API_KEY": _past_key()})

# include_source=False: the image above already holds the whole package, at the same place Modal would mount it.
app = modal.App(_CFG["app_name"], image=image, include_source=False)
volume = modal.Volume.from_name(_CFG["volume"], create_if_missing=True)


@app.function(cpu=float(_CFG["cpu"]), memory=int(_CFG["memory_mb"]), timeout=int(_CFG["worker_timeout_s"]),
              block_network=True, max_containers=int(_CFG["max_containers"]))
def evaluate_job(job: dict, sandbox_cfg: dict):
    """One sandboxed tuning job. The program runs in a child process (rlimits, timeout) inside this container.
    The result is yielded as compressed chunks of at most 1 MiB (knockout.executor.pack_result): this container
    has no network, so it cannot use Modal's blob storage for a large return value."""
    from dataclasses import asdict

    from knockout.executor import pack_result
    from knockout.sandbox import SandboxJob, run_job

    yield from pack_result(asdict(run_job(SandboxJob(**job), **sandbox_cfg)))


@app.function(cpu=4.0, memory=int(_CFG["orchestrator_memory_mb"]), timeout=int(_CFG["orchestrator_timeout_s"]),
              volumes={"/runs": volume}, max_containers=4, secrets=[llm_secret])
def orchestrate(problem: str, overrides: dict, run_id: str, source_commit: str | None,
                init_spec: str | None = None) -> dict:
    """The trusted loop: LLM calls, the frozen verifier, the ledger. It never runs an LLM-written program."""
    from knockout.run import run

    stop = threading.Event()

    def committer():  # make partial results visible with `modal volume get` while the run goes on
        while not stop.wait(60):
            try:
                volume.commit()
            except Exception:  # noqa: BLE001
                pass

    t = threading.Thread(target=committer, daemon=True)
    t.start()
    try:
        summary = run(problem, runs_root="/runs", run_id=run_id, source_commit=source_commit,
                      overrides={**overrides, "executor.kind": "modal"}, modal_fn=evaluate_job, init_spec=init_spec)
    finally:
        stop.set()
        volume.commit()
    return summary


@app.local_entrypoint()
def launch(source_commit: str = "", launch_file: str = "", wait: bool = False):
    spec = yaml.safe_load(Path(launch_file or Path(__file__).with_name("launch.yaml")).read_text())
    max_usd = float(spec.get("max_usd") or 0)
    if not 0 < max_usd <= 500:
        raise SystemExit(f"launch.yaml: max_usd must be in (0, 500], got {max_usd}")
    from knockout.run import PROFILES

    profile = spec.get("profile")
    if profile and profile not in PROFILES:
        raise SystemExit(f"launch.yaml: unknown profile {profile!r} (known: {sorted(PROFILES)})")
    overrides = {**(PROFILES[profile] if profile else {}), **dict(spec.get("set") or {})}
    overrides["run.wall_clock_s"] = float(spec["wall_clock_s"])
    overrides["run.max_usd"] = max_usd
    editor = spec.get("editor", "offline")
    overrides["editor.kind"] = editor
    if editor == "openai" and not _llm_key():
        raise SystemExit("launch.yaml asks for the LLM editor, but OPENAI_API_KEY is neither in the environment "
                         "nor in .env")
    init_spec = spec.get("init")
    commit = (source_commit or "nogit")[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = str(spec.get("tag") or "").strip()      # optional: keeps two runs launched together apart
    run_id = f"{stamp}-{spec['problem']}-{commit[:7]}" + (f"-{tag}" if tag else "")
    print(f"run_id={run_id} problem={spec['problem']} wall_clock_s={spec['wall_clock_s']} max_usd={max_usd} "
          f"editor={editor} profile={profile} init={init_spec}")
    if wait:
        print(orchestrate.remote(spec["problem"], overrides, run_id, commit, init_spec))
    else:
        call = orchestrate.spawn(spec["problem"], overrides, run_id, commit, init_spec)
        print(f"spawned orchestrator call {call.object_id}; results: modal volume get {_CFG['volume']} {run_id}")
