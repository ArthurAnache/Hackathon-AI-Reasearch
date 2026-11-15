"""Plan, run and export an optimizer benchmark (no evolution loop) on the WS2 V2 engine.

Fairness: one incumbent certificate for every job, paired seeds, equal budgets, one interleaved batch, and on
Modal one container per job (cpu=1, 4 GiB, single-threaded NumPy). The parent re-verifies every candidate with
the frozen verifier; worker scores are only cross-checks. Local runs share the machine: use them for tests only.

Outputs, under runs/ws3/<run_id>/ (git-ignored):
    plan.json             jobs (without optimizer source), incumbent, image spec, cost estimate; written before launch
    launch.json           executor, Modal app id, timings; slim JobResults as returned by the executor
    volume/...            per-job files (Modal: downloaded from the Volume by `collect`)
    export/               what goes to Marie, see export()
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PROBLEM = "second-autocorrelation-inequality"
DEFAULT_OPTIMIZERS = ("optimizers/second-autocorrelation-inequality_probe.py",
                      "optimizers/second-autocorrelation-inequality_blockshift.py")
DEFAULT_SEEDS = (101, 202, 303)
RUNS_DIR = ROOT / "runs" / "ws3"
CERT_STORE_DIR = RUNS_DIR / "certificates"

# Modal list prices used for the estimate (USD per second). NOT re-checked against modal.com/pricing.
PRICE_CPU_CORE_S = 0.0000394
PRICE_MEM_GIB_S = 0.00000672
STARTUP_OVERHEAD_S = 45          # container start + warm start load + candidate scoring/writing, per job (estimate)


# ---------------------------------------------------------------------------------------------- problem
def load_problem(slug: str, live_record: bool, log_dir: Path):
    """V2 Problem for `slug`. The EinsteinArena log goes to log_dir, never to the tracked selection/arena_log.jsonl.
    Offline (default): the leaderboard is not read; WS1's frozen record snapshot is the reference."""
    import interfaces as I
    try:
        from integrations import einstein_arena as ea
    except ImportError:
        ea = None
    if ea is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        ea.LOG_PATH = log_dir / "arena_log.jsonl"
        if not live_record:
            def _offline(*args, **kwargs):
                raise RuntimeError("ws3_bench offline mode: frozen record snapshot used")
            ea.get_leaderboard = _offline
    return I.get_problem(slug)


def ws1_certificate_sha(slug: str) -> str | None:
    from problems.catalog import SLUGS, load
    from certificates import candidate_sha
    return candidate_sha(load(slug).certificate()) if slug in SLUGS else None


def git_info(path: Path = ROOT) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    info = {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "tracked_changes": [l for l in (git("-c", "core.autocrlf=false", "status", "--porcelain",
                                                "--untracked-files=no") or "").splitlines() if l]}
    if not info["commit"]:
        info.update(_head_from_files(path), tracked_changes=None)
    return info


def _head_from_files(path: Path) -> dict:
    """Read HEAD without git (WSL cannot follow a worktree link written as a Windows path)."""
    try:
        gitdir = (path / ".git").read_text().split("gitdir:", 1)[1].strip()
        if os.name != "nt" and re.match(r"^[A-Za-z]:/", gitdir):
            gitdir = f"/mnt/{gitdir[0].lower()}{gitdir[2:]}"
        head = (Path(gitdir) / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            common = Path(gitdir) / (Path(gitdir) / "commondir").read_text().strip()
            for base in (Path(gitdir), common):
                if (base / ref).exists():
                    return {"commit": (base / ref).read_text().strip(), "branch": ref.removeprefix("refs/heads/")}
            packed = (common / "packed-refs").read_text() if (common / "packed-refs").exists() else ""
            commit = next((l.split()[0] for l in packed.splitlines() if l.endswith(" " + ref)), None)
            return {"commit": commit, "branch": ref.removeprefix("refs/heads/")}
        return {"commit": head, "branch": None}
    except (OSError, IndexError):
        return {"commit": None, "branch": None}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cost_estimate(n_jobs: int, budget_s: float, cpu: float = 1.0, memory_mb: int = 4096) -> dict:
    per_job_s = budget_s + STARTUP_OVERHEAD_S
    cpu_usd = n_jobs * per_job_s * cpu * PRICE_CPU_CORE_S
    mem_usd = n_jobs * per_job_s * (memory_mb / 1024) * PRICE_MEM_GIB_S
    return {"jobs": n_jobs, "billed_s_per_job_estimate": per_job_s, "cpu_usd": round(cpu_usd, 4),
            "memory_usd": round(mem_usd, 4), "total_usd": round(cpu_usd + mem_usd, 4),
            "prices": {"cpu_core_s": PRICE_CPU_CORE_S, "mem_gib_s": PRICE_MEM_GIB_S},
            "note": "list prices not re-checked; billing is on max(request, usage); excludes the one-time image build"}


# ---------------------------------------------------------------------------------------------- plan
def build_plan(slug, optimizers, seeds, budget_s, run_id, live_record=False, run_dir=None, configs=None,
               labels=None):
    import interfaces as I
    from certificates import CertificateStore
    from execution import make_job
    import compare as C
    from ws3_bench import image as IMG
    from ws3_bench import modal_app as MA

    run_dir = Path(run_dir or RUNS_DIR / run_id)
    problem = load_problem(slug, live_record, run_dir)
    certs = CertificateStore(CERT_STORE_DIR)
    inc = C.incumbent(problem, certs)              # WS1 frozen certificate, re-verified on put/load
    ws1_sha = ws1_certificate_sha(slug)
    ship_by_reference = inc is not None and ws1_sha is not None and inc.sha256 == ws1_sha
    configs = list(configs) if configs is not None else [{} for _ in optimizers]
    if len(configs) != len(optimizers):
        raise ValueError("configs must match optimizers 1:1")
    jobs = []
    for seed in seeds:                             # interleaved: seed-major, optimizers alternate
        for k, (opt, cfg) in enumerate(zip(optimizers, configs)):
            stem = Path(opt).stem.split("_")[-1]
            label = ""
            if labels and k < len(labels):
                label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(labels[k]))[:48]
            job_id = f"{label}-s{seed}" if label else f"{stem}-s{seed}"
            job = make_job(problem, ROOT / opt, cfg, seed, budget_s, inc.id if inc else None, certs,
                           job_id=job_id)
            merged = {**I.FileOptimizer(ROOT / opt).default_config(), **cfg}
            ws3 = {"run_id": run_id, "optimizer_name": label or stem, "config": merged}
            if ship_by_reference:
                job.warm_start_solution = None
                ws3["warm_start"] = {"source": "ws1-frozen-certificate", "cert_id": inc.id, "sha256": inc.sha256}
            job.meta = {"ws3": ws3}
            jobs.append(job)
    record = problem._m.BEST_KNOWN, problem._m.BEST_KNOWN_SOURCE
    plan = {
        "run_id": run_id, "kind": "optimizer benchmark (no evolution loop)", "problem": slug,
        "direction": problem.direction, "delta": getattr(problem._m, "DELTA", None),
        "record_reference": {"value": record[0], "source": record[1], "live_read": live_record},
        "incumbent": {"id": inc.id if inc else None, "score": inc.score if inc else None,
                      "sha256": inc.sha256 if inc else None, "ws1_frozen_certificate": ship_by_reference,
                      "transfer": "by reference (loaded from the image)" if ship_by_reference else "shipped in job"},
        "seeds": list(seeds), "budget_s": budget_s,
        "optimizers": [{"path": opt, "hash": hashlib.sha256((ROOT / opt).read_bytes()).hexdigest()[:12],
                        "config": {**I.FileOptimizer(ROOT / opt).default_config(), **cfg}}
                       for opt, cfg in zip(optimizers, configs)],
        "jobs": [{k: v for k, v in asdict(j).items() if k not in ("optimizer_source", "warm_start_solution")}
                 for j in jobs],
        "modal": MA.spec(), "image_audit": IMG.audit(slug),
        "cost_estimate": cost_estimate(len(jobs), budget_s, MA.CPU, MA.MEMORY_MB),
        "git": git_info(), "frozen": __import__("prepare").frozen_status(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return plan, jobs, problem


def payload_sizes(jobs) -> dict:
    import pickle
    sizes = [len(pickle.dumps(asdict(j))) for j in jobs]
    return {"max_job_bytes": max(sizes), "total_job_bytes": sum(sizes)}


# ---------------------------------------------------------------------------------------------- executors
class LocalWorkerPool:
    """Stand-in for the Modal function: same worker, same files, local processes (Linux only: fork + SIGALRM)."""

    def __init__(self, out_root: Path, workers: int):
        self.out_root, self.workers = Path(out_root), workers

    def map(self, payloads, return_exceptions=True):
        from concurrent.futures import ProcessPoolExecutor
        from ws3_bench.worker import execute
        payloads = list(payloads)
        with ProcessPoolExecutor(max_workers=min(self.workers, len(payloads))) as pool:
            futures = [pool.submit(execute, p, str(self.out_root)) for p in payloads]
            out = []
            for f in futures:
                try:
                    out.append(f.result())
                except BaseException as e:  # noqa: BLE001
                    if not return_exceptions:
                        raise
                    out.append(e)
            return out


def execute_jobs(jobs, executor: str, run_dir: Path, workers: int | None = None) -> dict:
    from execution import ModalExecutor
    t0 = time.time()
    launch = {"executor": executor, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))}
    results = []
    try:
        if executor == "local":
            results = ModalExecutor(function=LocalWorkerPool(run_dir / "volume", workers or len(jobs))).map(jobs)
        else:
            import modal
            from ws3_bench import modal_app as MA
            with modal.enable_output():
                with MA.app.run(detach=True):
                    launch["modal_app_id"] = MA.app.app_id
                    (run_dir / "launch.json").write_text(json.dumps(launch, indent=1))
                    results = ModalExecutor(function=MA.run_job_ws3).map(jobs)
    except Exception as exc:  # noqa: BLE001
        launch["error"] = f"{type(exc).__name__}: {exc}"
        launch["wall_s"] = round(time.time() - t0, 1)
        launch["results"] = [asdict(r) for r in results] if results else []
        (run_dir / "launch.json").write_text(json.dumps(launch, indent=1, default=str))
        raise
    launch["wall_s"] = round(time.time() - t0, 1)
    launch["results"] = [asdict(r) for r in results]
    (run_dir / "launch.json").write_text(json.dumps(launch, indent=1, default=str))
    return launch


def download_from_volume(run_id: str, run_dir: Path) -> list[str]:
    """Mirror runs/<run_id>/ of the Modal Volume into run_dir/volume/ (read-only on the Volume)."""
    import modal
    from ws3_bench import modal_app as MA
    vol = modal.Volume.from_name(MA.VOLUME_NAME)
    got = []
    for entry in vol.listdir(f"runs/{run_id}", recursive=True):
        if entry.type.name != "FILE":
            continue
        dest = run_dir / "volume" / entry.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in vol.read_file(entry.path):
                fh.write(chunk)
        got.append(entry.path)
    return got


# ---------------------------------------------------------------------------------------------- export
STAT_PATTERNS = {"evals": r"evals (\d+)", "improvements": r"improvements (\d+)",
                 "proposals": r"proposals (\d+)", "accepted": r"accepted (\d+)"}


def _stdout_stats(log: Path) -> dict:
    text = log.read_text(errors="replace") if log.exists() else ""
    out = {}
    for key, pat in STAT_PATTERNS.items():
        m = re.findall(pat, text)
        if m:
            out[key] = int(m[-1])
    return out


def _gain(direction, new, old):
    return (new - old) if direction == "maximize" else (old - new)


def export(run_dir: Path, exact: bool = False, live_record: bool = False) -> dict:
    import interfaces as I
    import prepare
    from certificates import candidate_sha
    from ws3_bench.worker import job_dir

    plan = json.loads((run_dir / "plan.json").read_text())
    launch = json.loads((run_dir / "launch.json").read_text()) if (run_dir / "launch.json").exists() else {}
    slug, run_id, budget = plan["problem"], plan["run_id"], plan["budget_s"]
    problem = load_problem(slug, live_record, run_dir)
    direction, inc = plan["direction"], plan["incumbent"]
    init_score = inc["score"]
    exp = run_dir / "export"
    if exp.exists():                                   # never overwrite or delete an earlier export
        exp = run_dir / ("export_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    (exp / "candidates").mkdir(parents=True)
    configs = {Path(o["path"]).name: o["config"] for o in plan["optimizers"]}
    (exp / "logs").mkdir()
    store = I.JsonlStore(exp / "experiments.jsonl")
    source_sha = (plan["git"].get("commit") or "")[:7] or None
    rows, details = [], []

    for j in plan["jobs"]:
        jid, seed, stem = j["job_id"], j["seed"], j["meta"]["ws3"]["optimizer_name"]
        jdir = job_dir(run_dir / "volume", run_id, jid)
        res = json.loads((jdir / "result.json").read_text()) if (jdir / "result.json").exists() else None
        started = (jdir / "started.json").exists()
        if (jdir / "stdout.log").exists():
            shutil.copyfile(jdir / "stdout.log", exp / "logs" / f"{jid}.log")
        run_status = res["status"] if res else ("lost" if started else "not_started")
        error = res.get("error") if res else ("no result.json: container lost before finishing" if started
                                             else "job never started")
        score, valid, cand_sha, parent_invalid = None, False, None, None
        gz = jdir / "candidate.json.gz"
        if res and gz.exists():
            raw = gzip.decompress(gz.read_bytes())
            cand_sha = hashlib.sha256(raw).hexdigest()
            if cand_sha != res.get("candidate_sha256"):
                parent_invalid = f"candidate file sha256 {cand_sha[:12]} != worker {str(res.get('candidate_sha256'))[:12]}"
            else:
                candidate = json.loads(raw)
                assert candidate_sha(candidate) == cand_sha
                parent_invalid = problem.why_invalid(candidate)          # frozen verifier, in this process
                if parent_invalid is None:
                    score, valid = problem.score(candidate), True
            shutil.copyfile(gz, exp / "candidates" / f"{cand_sha}.json.gz")
        g = _gain(direction, score, init_score) if valid and init_score is not None else None
        duration = (res or {}).get("run_job_duration_s")
        if run_status != "ok":
            status = "crash"
        elif not valid:
            status, error = "invalid", parent_invalid or error or "no candidate"
        elif duration is not None and duration > budget * prepare.OVER_BUDGET_SLACK + 1:
            status, error = "void", f"over time budget ({duration:.1f}s > {budget}s)"
        elif g is not None and g > prepare.MIN_IMPROVEMENT_REL * max(1.0, abs(init_score)):
            status = "keep"
        else:
            status = "discard"
        cert_id = f"{slug}/{cand_sha[:12]}" if valid and status != "void" else None
        if cert_id:
            cdir = exp / "certificates" / slug
            cdir.mkdir(parents=True, exist_ok=True)
            payload = {"problem": slug, "score": score, "sha256": cand_sha,
                       "provenance": {"run_id": run_id, "job_id": jid, "optimizer": j["optimizer"], "seed": seed,
                                      "parent": inc["id"], "verified_by": "frozen pinned verifier (parent)"},
                       "solution": json.loads(gzip.decompress((exp / "candidates" / f"{cand_sha}.json.gz")
                                                              .read_bytes()))}
            with gzip.open(cdir / f"{cand_sha[:12]}.json.gz", "wt") as fh:
                json.dump(payload, fh)
        worker_score = (res or {}).get("worker_score")
        stats = _stdout_stats(jdir / "stdout.log")
        record = I.new_record(
            experiment_id=f"ws3_{run_id}_{jid}", problem=slug, agent="ws3-bench", arm=f"ws3-bench:{stem}",
            island=None, worker_id=(res or {}).get("worker_id"), optimizer=j["optimizer"],
            optimizer_hash=j["optimizer_hash"], source_sha=source_sha, parent_id=inc["id"], genes=[],
            hypothesis=f"{stem} improves the WS1 frozen certificate within {budget}s (seed {seed})",
            rationale=f"WS3 optimizer benchmark, no evolution loop: same incumbent {inc['id']}, paired seeds "
                      f"{plan['seeds']}, equal budget, executor {launch.get('executor', '?')}",
            config=(j.get("meta") or {}).get("ws3", {}).get("config")
                  or configs[Path(j["optimizer"].replace("\\", "/")).name],
            warm_start=inc["id"], evidence=[], seeds=[seed], budget_s=budget,
            scores=[score] if score is not None else [], score=score, valid=valid, status=status,
            best_before=init_score, beats_record=bool(valid and problem.beats_record(score)),
            duration_s=duration, error=error, certificate_path=cert_id, llm_cost_usd=0.0, llm_usage=None)
        store.append(record)
        res_meta = res or {}
        rows.append({
            "run_id": run_id, "job_id": jid, "optimizer": stem, "seed": seed, "budget_s": budget,
            "status": status, "run_status": run_status, "valid": valid, "initial_score": init_score,
            "final_score": score, "gain": g, "improved_strictly": bool(g is not None and g > 0),
            "worker_score": worker_score,
            "worker_parent_diff": (worker_score - score) if worker_score is not None and score is not None else None,
            "run_job_duration_s": duration, "worker_wall_s": res_meta.get("worker_wall_s"),
            **{k: stats.get(k) for k in STAT_PATTERNS},
            "peak_rss_mb": (res_meta.get("resources") or {}).get("peak_rss_mb"),
            "cpu_user_s": (res_meta.get("resources") or {}).get("cpu_user_s"),
            "cpu_model": (res_meta.get("environment") or {}).get("cpu_model"),
            "worker_id": res_meta.get("worker_id"), "candidate_sha256": cand_sha, "certificate_id": cert_id,
            "error": (error or "")[:300] or None})
        details.append({"job": j, "result": res, "parent_verification": {
            "score": score, "valid": valid, "invalid_reason": parent_invalid, "status": status, "gain": g,
            "candidate_sha256": cand_sha}, "stdout_stats": stats})

    with open(exp / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(exp / "jobs_detail.jsonl", "w") as fh:
        for d in details:
            fh.write(json.dumps(d, default=str) + "\n")

    exact_line = None
    best = max((r for r in rows if r["valid"]), key=lambda r: r["final_score"] if direction == "maximize"
               else -r["final_score"], default=None)
    if exact and best:
        t = time.monotonic()
        cand = json.loads(gzip.decompress((exp / "candidates" / f"{best['candidate_sha256']}.json.gz").read_bytes()))
        fr = problem.certify(cand)
        if fr is not None:
            exact_line = {"candidate": best["candidate_sha256"], "exact": float(fr), "float": best["final_score"],
                          "difference": float(fr) - best["final_score"], "seconds": round(time.monotonic() - t, 1)}

    summary = summarize(plan, launch, rows, best, exact_line, problem)
    (exp / "summary.md").write_text(summary, encoding="utf-8")
    manifest = {
        "run_id": run_id, "kind": plan["kind"], "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan_created_at": plan["created_at"], "git": plan["git"], "frozen_lock": {
            "status": plan["frozen"], "sha256": sha256_file(ROOT / "frozen.lock")},
        "ws1": {"manifest_sha256": sha256_file(ROOT / "problems" / "MANIFEST.json"),
                "meta": json.loads((ROOT / "problems" / slug / "meta.json").read_text()).get("verifier", {}),
                "certificate_sha256": plan["incumbent"]["sha256"]},
        "ws3_bench_files": {p.name: sha256_file(p) for p in sorted((ROOT / "ws3_bench").glob("*.py"))},
        "incumbent": inc, "record_reference": plan["record_reference"], "seeds": plan["seeds"],
        "budget_s": budget, "optimizers": plan["optimizers"], "executor": launch.get("executor"),
        "modal_app_id": launch.get("modal_app_id"), "modal": plan["modal"], "cost_estimate": plan["cost_estimate"],
        "parent_environment": {"python": sys.version.split()[0], "numpy": __import__("numpy").__version__,
                               "scipy": __import__("scipy").__version__},
        "exact_check": exact_line,
        "files": {str(p.relative_to(exp)).replace(os.sep, "/"): sha256_file(p)
                  for p in sorted(exp.rglob("*")) if p.is_file()},
    }
    (exp / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return {"export": str(exp), "rows": rows, "best": best, "exact": exact_line}


def summarize(plan, launch, rows, best, exact_line, problem) -> str:
    seeds, opts = plan["seeds"], [o["path"] for o in plan["optimizers"]]
    names = [Path(o).stem.split("_")[-1] for o in opts]
    by = {(r["optimizer"], r["seed"]): r for r in rows}
    lines = [f"# WS3 optimizer benchmark {plan['run_id']}", "",
             "**Optimizer benchmark, not an agent benchmark.** Fixed optimizer files and configs, paired seeds, "
             "equal budgets, one incumbent. Comparing research agents (WS2 evolution vs Arthur's reflection "
             "architecture) needs both agents run under the same budget, incumbent and executor.", "",
             f"- Problem: `{plan['problem']}` ({plan['direction']}), delta {plan['delta']}",
             f"- Incumbent: `{plan['incumbent']['id']}` = {plan['incumbent']['score']!r} "
             f"(WS1 frozen certificate: {plan['incumbent']['ws1_frozen_certificate']})",
             f"- Record reference: {plan['record_reference']['value']!r} ({plan['record_reference']['source']})",
             f"- Seeds {seeds}, budget {plan['budget_s']} s per job, executor `{launch.get('executor', '?')}`"
             f", wall {launch.get('wall_s', '?')} s",
             f"- Resources per job: cpu {plan['modal']['cpu']}, memory {plan['modal']['memory_mb']} MiB, "
             f"Python {plan['modal']['python']}, {', '.join(plan['modal']['pip'])}", "",
             "## Gain over the incumbent, per seed (parent re-verified, frozen verifier)", "",
             "| seed | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for s in seeds:
        cells = []
        for n in names:
            r = by.get((n, s))
            cells.append(f"{r['gain']:+.4e} ({r['status']})" if r and r["gain"] is not None
                         else f"{r['status'] if r else 'missing'}: {(r or {}).get('error') or ''}"[:60])
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", "## Per optimizer", "",
              "| optimizer | mean gain | best gain | improved seeds | keep/discard/invalid/crash/void | "
              "evals or proposals per s | accepted | peak RSS MiB |", "|---|---|---|---|---|---|---|---|"]
    gains = {}
    for n in names:
        rs = [by[(n, s)] for s in seeds if (n, s) in by]
        g = [r["gain"] for r in rs if r["gain"] is not None]
        gains[n] = [by[(n, s)]["gain"] if (n, s) in by else None for s in seeds]
        counts = "/".join(str(sum(r["status"] == st for r in rs)) for st in ("keep", "discard", "invalid", "crash", "void"))
        rate = [((r.get("evals") or r.get("proposals") or 0) / r["run_job_duration_s"]) for r in rs
                if r.get("run_job_duration_s") and (r.get("evals") or r.get("proposals"))]
        acc = [r.get("improvements") if r.get("improvements") is not None else r.get("accepted") for r in rs]
        rss = [r["peak_rss_mb"] for r in rs if r.get("peak_rss_mb")]
        lines.append(f"| {n} | {statistics.fmean(g):+.4e} | {max(g):+.4e} | {sum(x > 0 for x in g)}/{len(seeds)} | "
                     f"{counts} | {f'{statistics.fmean(rate):.2f}' if rate else '-'} | {acc} | "
                     f"{max(rss) if rss else '?'} |"
                     if g else f"| {n} | - | - | 0/{len(seeds)} | {counts} | - | {acc} | - |")
    lines += ["", "## Reading", ""]
    if len(names) == 2 and all(None not in gains[n] for n in names):
        d = [b - a for a, b in zip(gains[names[0]], gains[names[1]])]
        se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        lines.append(f"- Paired difference ({names[1]} - {names[0]}): mean {statistics.fmean(d):+.4e} "
                     f"(SE {se:.2e}); {names[1]} better on {sum(x > 0 for x in d)}/{len(d)} seeds.")
        winner = names[1] if statistics.fmean(d) > 0 else names[0]
        lines.append(f"- Better on average in this run: **{winner}**. With {len(d)} seeds this is a first signal, "
                     "not a significance test.")
    else:
        lines.append("- Not every job produced a valid result: no paired difference (see failures below).")
    lines.append("- Mechanism evidence: throughput (evals or proposals per second) and accepted moves above; "
                 "probe pays one full verifier evaluation per random try, blockshift scores a grid of steps per "
                 "proposal with exact incremental updates (see the optimizer docstrings).")
    if best:
        to_rec = _gain(plan["direction"], best["final_score"], plan["record_reference"]["value"])
        lines.append(f"- Best candidate: {best['optimizer']} seed {best['seed']} = {best['final_score']!r}, "
                     f"{to_rec:+.4e} vs the record reference (needs >= {plan['delta']}): "
                     f"{'RECORD CANDIDATE: confirm with selection/verify_claim.py' if problem.beats_record(best['final_score']) else 'not a record'}.")
    if exact_line:
        lines.append(f"- Exact check of the best candidate: {exact_line['exact']!r} "
                     f"(difference {exact_line['difference']:+.2e}, {exact_line['seconds']} s).")
    bad = [r for r in rows if r["status"] in ("crash", "invalid", "void")]
    lines += ["", "## Failures, invalid and void results", ""]
    lines += [f"- {r['job_id']}: {r['status']} ({r['run_status']}): {r['error']}" for r in bad] or ["- none"]
    lines += ["", "Files: `experiments.jsonl` (V2 RECORD_FIELDS, one record per job), `comparison.csv`, "
              "`jobs_detail.jsonl`, `candidates/<sha256>.json.gz`, `certificates/<problem>/<sha12>.json.gz` "
              "(gunzip = V2 CertificateStore format), `logs/`, `manifest.json`.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- CLI
def _run_id(budget: float) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-b{int(budget)}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ws3_bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--problem", default=DEFAULT_PROBLEM)
        p.add_argument("--optimizers", nargs="+", default=list(DEFAULT_OPTIMIZERS))
        p.add_argument("--configs", nargs="*", default=None,
                       help="optional JSON object per optimizer (default {})")
        p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
        p.add_argument("--budget", type=float, default=60.0)
        p.add_argument("--live-record", action="store_true", help="read the live #1 (public GET) instead of the snapshot")
        p.add_argument("--run-id", default=None)
    run = sub.choices["run"]
    run.add_argument("--executor", choices=["local", "modal"], required=True)
    run.add_argument("--workers", type=int, default=None)
    run.add_argument("--confirm-modal", action="store_true", help="required to start Modal containers")
    run.add_argument("--exact", action="store_true", help="exact re-check of the best candidate")
    col = sub.add_parser("collect")
    col.add_argument("--run-id", required=True)
    col.add_argument("--from", dest="source", choices=["local", "modal"], default="local")
    col.add_argument("--exact", action="store_true")
    col.add_argument("--live-record", action="store_true")
    sub.add_parser("audit-image").add_argument("--problem", default=DEFAULT_PROBLEM)
    a = ap.parse_args(argv)

    if a.cmd == "audit-image":
        from ws3_bench import image as IMG
        issues = IMG.audit(a.problem)
        print(json.dumps({"files": IMG.relative_files(a.problem), "issues": issues}, indent=1))
        return 1 if issues else 0

    if a.cmd == "collect":
        run_dir = RUNS_DIR / a.run_id
        if a.source == "modal":
            print(f"downloaded {len(download_from_volume(a.run_id, run_dir))} files from the Volume")
        out = export(run_dir, exact=a.exact, live_record=a.live_record)
        print((Path(out["export"]) / "summary.md").read_text(encoding="utf-8"))
        return 0

    if len(a.seeds) < 3:
        ap.error("a fair comparison needs at least 3 paired seeds")
    if a.cmd == "run" and a.executor == "modal":
        from ws3_bench import modal_app as MA
        if not a.confirm_modal:
            ap.error("--executor modal starts paid containers: add --confirm-modal")
        if a.budget > MA.max_budget_s():
            ap.error(f"budget {a.budget}s exceeds what the function timeout covers ({MA.max_budget_s():.0f}s)")
    run_id = a.run_id or _run_id(a.budget)
    run_dir = RUNS_DIR / run_id
    configs = [json.loads(c) for c in a.configs] if a.configs else None
    plan, jobs, _ = build_plan(a.problem, a.optimizers, a.seeds, a.budget, run_id, a.live_record, run_dir,
                               configs=configs)
    plan["payload"] = payload_sizes(jobs)
    if plan["image_audit"]:
        print("image audit failed:", plan["image_audit"])
        return 1
    if a.cmd == "plan":
        print(json.dumps({k: plan[k] for k in ("run_id", "problem", "incumbent", "record_reference", "seeds",
                                                "budget_s", "optimizers", "payload", "cost_estimate", "frozen")},
                         indent=1, default=str))
        print(f"{len(jobs)} jobs: " + ", ".join(j.job_id for j in jobs))
        print(f"image: {len(plan['modal']['files'])} files, audit ok; nothing was launched")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=1, default=str))
    print(f"run {run_id}: {len(jobs)} jobs, {a.executor}, plan written to {run_dir / 'plan.json'}")
    launch = execute_jobs(jobs, a.executor, run_dir, a.workers)
    print(f"executor finished in {launch['wall_s']} s: " + ", ".join(f"{r['job_id']}={r['status']}"
                                                                     for r in launch["results"]))
    if a.executor == "modal":
        print(f"downloaded {len(download_from_volume(run_id, run_dir))} files from the Volume")
    out = export(run_dir, exact=a.exact, live_record=a.live_record)
    print((Path(out["export"]) / "summary.md").read_text(encoding="utf-8"))
    print(f"export: {out['export']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
