"""Run an untrusted candidate program in a child process: timeout, rlimits, temp dir, no secrets, no network.

Isolation actually applied is reported in every result (`isolation`):
- macOS: `sandbox-exec` denies all network access, file writes outside the job directory, and reads of the
  repo's .env and of the user's credential directories. RLIMIT_AS is not enforced by macOS, so the memory limit
  is NOT guaranteed locally.
- Linux (Modal containers): RLIMIT_AS enforces memory; the Modal function itself runs with block_network=True.
  `unshare -rn` is used as well when available.
The child gets an empty environment except PATH and thread caps, so API keys never reach it.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRY = Path(__file__).resolve().with_name("sandbox_entry.py")
# Credential locations that a candidate program may not read (macOS profile). The child already gets an empty
# environment; this closes the file route too, now that programs are written by an LLM.
SECRET_DIRS = (".config", ".ssh", ".aws", ".gnupg", ".claude")
SECRET_FILES = (".modal.toml", ".netrc", ".zsh_history", ".bash_history")


@dataclass
class SandboxJob:
    program_source: str
    seed: int
    budget_s: float
    direction: str
    guide_ref: str
    guide_args: list = field(default_factory=list)
    mode: str = "tune"                     # "tune" | "construct"
    params: dict = field(default_factory=dict)
    init: Any = None                       # warm-start candidate (JSON-able), or None
    adapter_source: str | None = None
    tuner: dict = field(default_factory=dict)


@dataclass
class SandboxResult:
    ok: bool
    error: str | None
    trials: list[dict]                     # [{"params", "guide", "candidate"}], best guide first
    n_evals: int = 0
    tuner: str | None = None
    elapsed_s: float = 0.0                 # wall time of the child process
    timed_out: bool = False
    isolation: dict = field(default_factory=dict)
    stderr_tail: str = ""
    tuner_errors: list[str] = field(default_factory=list)


def _isolation_prefix(job_dir: Path, deny_network: bool) -> tuple[list[str], dict]:
    system = platform.system()
    if not deny_network:
        return [], {"network": "allowed", "fs_write": "unrestricted", "memory": "rlimit" if system == "Linux" else "none"}
    if system == "Darwin" and shutil.which("sandbox-exec"):
        jd = str(job_dir.resolve())
        home = Path.home().resolve()
        secrets = [f'(literal "{REPO_ROOT / ".env"}")'] + [f'(subpath "{home / name}")' for name in SECRET_DIRS] \
            + [f'(literal "{home / name}")' for name in SECRET_FILES]
        profile = ("(version 1)(allow default)(deny network*)(deny file-write*)"
                   f'(allow file-write* (subpath "{jd}") (literal "/dev/null") (literal "/dev/zero"))'
                   f'(deny file-read* {" ".join(secrets)})')
        return ["sandbox-exec", "-p", profile], {"network": "denied (sandbox-exec)", "fs_write": "job dir only",
                                                 "fs_read": "secrets denied (.env, credential dirs)",
                                                 "memory": "not enforced on macOS"}
    if system == "Linux":
        if shutil.which("unshare") and subprocess.run(["unshare", "-rn", "true"], capture_output=True).returncode == 0:
            return ["unshare", "-rn"], {"network": "denied (unshare -rn)", "fs_write": "unrestricted",
                                        "memory": "rlimit"}
        return [], {"network": "denied by the Modal function (block_network)" if os.environ.get("MODAL_TASK_ID")
                    else "NOT isolated", "fs_write": "unrestricted", "memory": "rlimit"}
    return [], {"network": "NOT isolated", "fs_write": "unrestricted", "memory": "none"}


def _limits(cpu_s: int, memory_mb: int, fsize_mb: int):
    def apply():
        import resource

        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb << 20, fsize_mb << 20))
        if platform.system() == "Linux":
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb << 20, memory_mb << 20))
    return apply


def run_job(job: SandboxJob, *, timeout_slack_s: float = 10.0, memory_mb: int = 4096, deny_network: bool = True,
            max_output_mb: int = 256, threads: int = 1) -> SandboxResult:
    """Run one job in a fresh temp dir and return its parsed result. Never raises for program errors."""
    with tempfile.TemporaryDirectory(prefix="ko_job_") as tmp:
        d = Path(tmp)
        (d / "program.py").write_text(job.program_source)
        if job.adapter_source:
            (d / "adapter.py").write_text(job.adapter_source)
        if job.init is not None:
            (d / "init.json").write_text(json.dumps(job.init))
        (d / "job.json").write_text(json.dumps({
            "repo_root": str(REPO_ROOT), "seed": job.seed, "budget_s": job.budget_s, "direction": job.direction,
            "guide_ref": job.guide_ref, "guide_args": job.guide_args, "mode": job.mode, "params": job.params,
            "tuner": job.tuner,
        }))
        prefix, isolation = _isolation_prefix(d, deny_network)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(d), "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
               "MKL_NUM_THREADS": str(threads), "TMPDIR": str(d)}
        hard_timeout = job.budget_s + timeout_slack_s
        cmd = prefix + [sys.executable, "-I", str(ENTRY), str(d)]
        t0 = time.monotonic()
        timed_out = False
        with open(d / "stderr.txt", "wb") as err:
            proc = subprocess.Popen(cmd, cwd=d, env=env, stdout=subprocess.DEVNULL, stderr=err,
                                    preexec_fn=_limits(int(hard_timeout) + 5, memory_mb, max_output_mb))
            try:
                proc.wait(timeout=hard_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        elapsed = time.monotonic() - t0
        stderr_tail = (d / "stderr.txt").read_bytes()[-2000:].decode(errors="replace")
        out_path = d / "out.json"
        if timed_out or not out_path.exists():
            reason = f"timeout after {hard_timeout:.0f}s" if timed_out else f"no output (exit code {proc.returncode})"
            return SandboxResult(False, reason, [], elapsed_s=elapsed, timed_out=timed_out, isolation=isolation,
                                 stderr_tail=stderr_tail)
        try:
            out = json.loads(out_path.read_text())
        except json.JSONDecodeError as exc:
            return SandboxResult(False, f"unreadable output: {exc}", [], elapsed_s=elapsed, isolation=isolation,
                                 stderr_tail=stderr_tail)
        return SandboxResult(bool(out.get("ok")), out.get("error"), list(out.get("trials") or []),
                             int(out.get("n_evals") or 0), out.get("tuner"), elapsed, False, isolation,
                             stderr_tail, list(out.get("tuner_errors") or []))
