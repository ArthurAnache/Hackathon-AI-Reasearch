"""Run one V2 job and persist everything about it, whatever happens.

execute(job, out_root) is called inside a Modal container (out_root = the mounted Volume) or locally (tests).
Layout, one directory per job:

    <out_root>/runs/<run_id>/jobs/<job_id>/
        started.json         written (and committed) before the optimizer starts
        stdout.log           optimizer stdout + stderr, streamed
        candidate.json.gz    the candidate, canonical JSON; sha256 of the decompressed bytes = certificates.candidate_sha
        result.json          status, error, durations, worker score, resources, versions

The job itself runs in a forked child: execution.run_job keeps its SIGALRM watchdog (main thread of the child),
a hard kill or an out-of-memory death of the child is reported instead of losing the container, and the 2M-value
candidate never travels back through Modal: the child writes it to disk and returns its sha256.

Warm start without shipping 2M values: when the parent sets meta.ws3.warm_start = {source: ws1-frozen-certificate,
sha256}, the worker loads WS1's frozen certificate from the image (sha-checked against meta.json by WS1's code) and
refuses to run unless its candidate_sha equals the parent's certificate sha256.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path

HARD_KILL_FACTOR, HARD_KILL_GRACE_S = 2.0, 30      # same watchdog as execution.run_job
JOIN_GRACE_S = 90                                  # extra time for loading, scoring and writing the candidate


def safe_name(job_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in job_id)


def job_dir(out_root, run_id: str, job_id: str) -> Path:
    return Path(out_root) / "runs" / safe_name(run_id) / "jobs" / safe_name(job_id)


def canonical_bytes(candidate) -> bytes:
    """Same serialization as certificates.candidate_sha, so sha256(bytes) is the certificate sha256."""
    return json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    os.replace(tmp, path)


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def environment() -> dict:
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = None
    cpu_model = None
    info = _read("/proc/cpuinfo")
    if info:
        cpu_model = next((line.split(":", 1)[1].strip() for line in info.splitlines()
                          if line.startswith("model name")), None)
    return {
        "host": socket.gethostname(), "platform": platform.platform(), "versions": versions,
        "cpu_model": cpu_model, "os_cpu_count": os.cpu_count(),
        "cgroup_cpu_max": _read("/sys/fs/cgroup/cpu.max"), "cgroup_memory_max": _read("/sys/fs/cgroup/memory.max"),
        "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "modal": {k: os.environ.get(k) for k in ("MODAL_TASK_ID", "MODAL_REGION", "MODAL_CLOUD_PROVIDER")
                  if os.environ.get(k)},
    }


def score_with_frozen_verifier(problem_id: str, candidate) -> tuple[float | None, str | None]:
    """WS1 problems: pinned verifier through problems.catalog (no network). WS2 problems: problems.get(id).verify."""
    from problems import InvalidSolution
    from problems.catalog import SLUGS, load
    try:
        if problem_id in SLUGS:
            return float(load(problem_id).score(candidate)), None
        import problems
        return float(problems.get(problem_id).verify(candidate)), None
    except InvalidSolution as e:
        return None, str(e)[:500]


def resolve_warm_start(job: dict) -> dict:
    ws = ((job.get("meta") or {}).get("ws3") or {}).get("warm_start")
    if job.get("warm_start_solution") is not None or not ws:
        return job
    src = ws.get("source")
    if src == "ws1-frozen-certificate":
        from problems.catalog import load
        solution = load(job["problem"]).certificate()
        sha = hashlib.sha256(canonical_bytes(solution)).hexdigest()
        if sha != ws["sha256"]:
            raise ValueError(f"warm start mismatch: image certificate {sha[:12]} != parent certificate {ws['sha256'][:12]}")
        return {**job, "warm_start_solution": solution}
    if src == "public-live-best":
        # Baked into the Modal image by V3 (optional file). Avoids shipping 2M values per job.
        path = Path("/root/repo") / "problems" / job["problem"] / "live_public_best.json.gz"
        if not path.is_file():
            raise FileNotFoundError(f"public live best missing in image: {path}")
        payload = json.loads(gzip.decompress(path.read_bytes()).decode())
        solution = {"values": payload["values"]}
        if ws.get("solution_id") is not None and payload.get("solution_id") != ws.get("solution_id"):
            raise ValueError("public live best solution_id mismatch vs job meta")
        return {**job, "warm_start_solution": solution}
    raise ValueError(f"unsupported warm start source {src!r}")


def _child(job: dict, jdir: str) -> None:
    jdir = Path(jdir)
    fd = os.open(jdir / "stdout.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    # the host may have replaced sys.stdout (pytest, Modal log streaming): write to the file whatever it was
    sys.stdout = sys.stderr = open(fd, "w", buffering=1, closefd=False)
    out: dict = {}
    try:
        t0 = time.monotonic()
        job = resolve_warm_start(job)
        warm_s = time.monotonic() - t0
        from execution import run_job
        r = run_job(job)
        candidate = r.pop("candidate", None)
        out = {"run_job": r, "warm_start_load_s": round(warm_s, 3)}
        if candidate is not None:
            raw = canonical_bytes(candidate)
            tmp = jdir / "candidate.json.gz.tmp"
            with gzip.open(tmp, "wb", compresslevel=6) as fh:
                fh.write(raw)
            os.replace(tmp, jdir / "candidate.json.gz")
            values = candidate.get("values") if isinstance(candidate, dict) else None
            out.update(candidate_sha256=hashlib.sha256(raw).hexdigest(), candidate_file="candidate.json.gz",
                       candidate_json_bytes=len(raw), candidate_gz_bytes=(jdir / "candidate.json.gz").stat().st_size,
                       candidate_values=len(values) if isinstance(values, list) else None)
            t1 = time.monotonic()
            out["worker_score"], out["worker_invalid"] = score_with_frozen_verifier(job["problem"], candidate)
            out["worker_score_s"] = round(time.monotonic() - t1, 3)
    except BaseException as e:  # noqa: BLE001  every failure is a result
        out.setdefault("run_job", {"status": "crash", "duration_s": 0.0})
        out["run_job"]["status"] = "crash"
        out["run_job"]["error"] = f"{type(e).__name__}: {e} | " + traceback.format_exc().strip().splitlines()[-1][:300]
    finally:
        write_json(jdir / "child.json", out)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _safe_commit(commit) -> str | None:
    if commit is None:
        return None
    try:
        commit()
        return None
    except Exception as e:  # noqa: BLE001  a failed commit must not lose the job; it is recorded instead
        return f"{type(e).__name__}: {e}"


def execute(job: dict, out_root, commit=None) -> dict:
    """Run one job, persist it under out_root, return a JobResult dict with candidate=None (the file holds it)."""
    import resource                                    # Unix only: workers run on Linux (Modal, WSL)
    meta = dict(job.get("meta") or {})
    ws3 = dict(meta.get("ws3") or {})
    run_id, job_id = ws3.get("run_id", "adhoc"), job.get("job_id") or "job"
    jdir = job_dir(out_root, run_id, job_id)
    jdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    env = environment()
    write_json(jdir / "started.json", {
        "job_id": job_id, "run_id": run_id, "problem": job.get("problem"), "optimizer": job.get("optimizer"),
        "optimizer_hash": job.get("optimizer_hash"), "seed": job.get("seed"), "budget_s": job.get("budget_s"),
        "config": job.get("config"), "warm_start": job.get("warm_start"), "warm_start_meta": ws3.get("warm_start"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)), "environment": env})
    commit_errors = [e for e in [_safe_commit(commit)] if e]

    budget = float(job.get("budget_s") or 0.0)
    join_timeout = budget * HARD_KILL_FACTOR + HARD_KILL_GRACE_S + JOIN_GRACE_S
    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(target=_child, args=(job, str(jdir)), daemon=False)
    proc.start()
    proc.join(join_timeout)
    killed = proc.is_alive()
    if killed:
        proc.kill()
        proc.join(10)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)

    child_file = jdir / "child.json"
    child = json.loads(child_file.read_text()) if child_file.exists() else {}
    r = dict(child.get("run_job") or {})
    if killed:
        r.update(status="timeout", error=f"worker killed the job after {join_timeout:.0f}s")
    elif not child:
        r.update(status="crash", error=f"job process died with exit code {proc.exitcode}"
                                       + (" (SIGKILL: out of memory?)" if proc.exitcode == -9 else ""))
    status = r.get("status") or "crash"
    wall = round(time.time() - t_start, 3)
    resources = {"peak_rss_mb": round(usage.ru_maxrss / 1024, 1), "cpu_user_s": round(usage.ru_utime, 2),
                 "cpu_system_s": round(usage.ru_stime, 2), "child_exitcode": proc.exitcode}
    stats = {k: child.get(k) for k in ("candidate_sha256", "candidate_file", "candidate_json_bytes",
                                       "candidate_gz_bytes", "candidate_values", "worker_score", "worker_invalid",
                                       "worker_score_s", "warm_start_load_s")}
    result = {
        "job_id": job_id, "run_id": run_id, "status": status, "error": r.get("error"),
        "run_job_duration_s": r.get("duration_s"), "worker_wall_s": wall, "worker_id": r.get("worker_id")
        or f"{env['host']}:{os.getpid()}", "seed": job.get("seed"), "budget_s": budget,
        "optimizer": job.get("optimizer"), "optimizer_hash": job.get("optimizer_hash"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "resources": resources,
        "environment": env, **stats}
    write_json(jdir / "result.json", result)
    commit_errors += [e for e in [_safe_commit(commit)] if e]
    if commit_errors:
        result["commit_errors"] = commit_errors
        write_json(jdir / "result.json", result)
    meta["ws3"] = {**ws3, "job_dir": str(jdir.relative_to(Path(out_root))).replace(os.sep, "/"),
                   **{k: result.get(k) for k in ("candidate_sha256", "worker_score", "worker_invalid",
                                                 "run_job_duration_s", "worker_wall_s", "resources")}}
    return {"job_id": job_id, "status": status, "candidate": None, "duration_s": r.get("duration_s") or wall,
            "worker_id": result["worker_id"], "error": r.get("error"), "meta": meta}
