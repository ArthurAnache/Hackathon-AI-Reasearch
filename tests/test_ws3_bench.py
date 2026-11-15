"""WS3 benchmark runner (ws3_bench): image safety, worker persistence and failure modes, V2 compatibility of the
exports. No Modal call: the Modal function is replaced by the same worker run in local processes."""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

linux_only = pytest.mark.skipif(sys.platform == "win32", reason="workers need fork + SIGALRM (Linux, Modal, WSL)")
SLUG = "second-autocorrelation-inequality"


@pytest.fixture(autouse=True)
def offline_arena(tmp_path, monkeypatch):
    """No leaderboard read and no write to the tracked selection/arena_log.jsonl; restored after each test."""
    from integrations import einstein_arena as ea

    def offline(*args, **kwargs):
        raise RuntimeError("test: offline")
    monkeypatch.setattr(ea, "LOG_PATH", tmp_path / "arena_log.jsonl")
    monkeypatch.setattr(ea, "get_leaderboard", offline)


# ------------------------------------------------------------------ image
@pytest.mark.process("ws3-bench")
def test_image_allowlist_is_safe(record):
    """The Modal image holds only allowlisted code and frozen WS1 files: no secrets, caches, logs or results."""
    from ws3_bench import image as IMG
    files = IMG.relative_files(SLUG)
    assert IMG.audit(SLUG) == []
    assert all(local.is_file() for local, _ in IMG.image_files(SLUG))
    for bad in (".env", "selection/", "integrations/", "experiments", "certificates/", "runs/", "reports/",
                ".cache", "optimizers/"):
        assert not any(f.startswith(bad) or f == bad for f in files), bad
    assert f"problems/{SLUG}/certificate.json.gz" in files and f"problems/{SLUG}/verifier_pinned.py" in files
    assert IMG.PIP_PINS == ("numpy==2.2.6", "scipy==1.16.2") and IMG.PYTHON_VERSION == "3.11"
    assert IMG.IMAGE_ENV["OMP_NUM_THREADS"] == "1"
    record(image_files=len(files))


@pytest.mark.process("ws3-bench")
def test_image_audit_flags_secrets_and_forbidden_paths(tmp_path, monkeypatch):
    """The audit refuses a forbidden path in the allowlist and a secret-looking string in a shipped file."""
    from ws3_bench import image as IMG
    monkeypatch.setattr(IMG, "CODE_FILES", IMG.CODE_FILES + (".env", "selection/arena_log.jsonl"))
    issues = IMG.audit(SLUG)
    assert any(".env" in i for i in issues) and any("selection/" in i for i in issues)
    monkeypatch.setattr(IMG, "CODE_FILES", ("leak.py",))
    monkeypatch.setattr(IMG, "WS1_PROBLEM_FILES", ())
    (tmp_path / "leak.py").write_text('EXA_API_KEY = "abcdefghijklmnopqrstuvwxyz123456"\n')
    assert any("secret-looking" in i for i in IMG.audit(SLUG, root=tmp_path))


@pytest.mark.process("ws3-bench")
def test_modal_app_spec_without_network(monkeypatch):
    """The Modal app builds lazily (no server contact) with cpu=1, 4096 MiB, no retries, no secrets, pinned image."""
    modal = pytest.importorskip("modal")
    monkeypatch.setenv("MODAL_SERVER_URL", "http://127.0.0.1:9")
    MA = importlib.reload(importlib.import_module("ws3_bench.modal_app"))
    spec = MA.spec()
    assert (spec["cpu"], spec["memory_mb"], spec["retries"], spec["secrets"]) == (1.0, 4096, 0, [])
    assert MA.app is not None and MA.run_job_ws3 is not None
    assert MA.max_budget_s() >= 300
    assert isinstance(MA.build_image(modal), modal.Image)


# ------------------------------------------------------------------ worker
def _job(tmp_path, source: str | None = None, budget=0.5, seed=1, problem="1a", run_id="t", **ws3):
    import interfaces as I
    from execution import make_job
    path = REPO / "optimizers" / "1a_baseline.py"
    if source is not None:
        path = tmp_path / "opt_test.py"
        path.write_text(source)
    job = make_job(I.get_problem(problem), path, {}, seed, budget, job_id=f"job-s{seed}")
    job.meta = {"ws3": {"run_id": run_id, **ws3}}
    from dataclasses import asdict
    return asdict(job)


def _files(out, run_id, job_id):
    from ws3_bench.worker import job_dir
    return job_dir(out, run_id, job_id)


@linux_only
@pytest.mark.process("ws3-bench")
def test_worker_persists_ok_job(tmp_path, record):
    """An ok job leaves started/result/stdout/candidate files; the returned JobResult carries no candidate."""
    from execution import JobResult
    from certificates import candidate_sha
    from ws3_bench.worker import execute
    commits = []
    r = execute(_job(tmp_path), tmp_path / "vol", commit=lambda: commits.append(1))
    jr = JobResult(**r)
    assert jr.status == "ok" and jr.candidate is None and jr.error is None
    d = _files(tmp_path / "vol", "t", "job-s1")
    res = json.loads((d / "result.json").read_text())
    raw = gzip.decompress((d / "candidate.json.gz").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == res["candidate_sha256"] == candidate_sha(json.loads(raw))
    assert res["worker_score"] is not None and res["worker_invalid"] is None
    assert (d / "started.json").exists() and (d / "stdout.log").exists() and len(commits) == 2
    assert res["environment"]["versions"]["numpy"] and res["resources"]["peak_rss_mb"] > 0
    record(worker_score=res["worker_score"], wall=res["worker_wall_s"])


@linux_only
@pytest.mark.process("ws3-bench")
def test_worker_records_crash_and_hash_mismatch(tmp_path):
    """A crashing optimizer and a tampered source are results (status crash + error), never lost jobs."""
    from ws3_bench.worker import execute
    src = 'PROBLEM = "1a"\nCONFIG = {}\ndef search(b, s, c):\n    print("about to fail")\n    raise RuntimeError("boom")\n'
    r = execute(_job(tmp_path, src), tmp_path / "vol")
    assert r["status"] == "crash" and "boom" in r["error"]
    assert "about to fail" in (_files(tmp_path / "vol", "t", "job-s1") / "stdout.log").read_text()
    job = _job(tmp_path, seed=2)
    job["optimizer_source"] += "\n# tampered\n"
    r = execute(job, tmp_path / "vol")
    assert r["status"] == "crash" and "does not match" in r["error"]
    assert json.loads((_files(tmp_path / "vol", "t", "job-s2") / "result.json").read_text())["status"] == "crash"


@linux_only
@pytest.mark.process("ws3-bench")
def test_worker_survives_hard_kill_and_timeout(tmp_path, monkeypatch):
    """A job killed by the OS (as in out-of-memory) and a job past the watchdog are recorded, not lost."""
    import execution
    from ws3_bench.worker import execute
    kill = 'import os, signal\nPROBLEM = "1a"\nCONFIG = {}\ndef search(b, s, c):\n    os.kill(os.getpid(), signal.SIGKILL)\n'
    r = execute(_job(tmp_path, kill), tmp_path / "vol")
    assert r["status"] == "crash" and "-9" in r["error"]
    monkeypatch.setattr(execution, "HARD_KILL_GRACE_S", 1)
    sleep = 'import time\nPROBLEM = "1a"\nCONFIG = {}\ndef search(b, s, c):\n    time.sleep(30)\n'
    r = execute(_job(tmp_path, sleep, budget=0.5, seed=3), tmp_path / "vol")
    assert r["status"] == "timeout"


@linux_only
@pytest.mark.process("ws3-bench")
def test_worker_refuses_mismatched_warm_start(tmp_path):
    """By-reference warm start: the worker refuses to run if the image certificate is not the parent's."""
    import interfaces as I
    from dataclasses import asdict
    from execution import make_job
    from ws3_bench.worker import execute
    j = make_job(I.get_problem(SLUG), REPO / "optimizers" / f"{SLUG}_probe.py", {}, 1, 1.0, job_id="ws-s1")
    j.meta = {"ws3": {"run_id": "t", "warm_start": {"source": "ws1-frozen-certificate", "sha256": "0" * 64}}}
    r = execute(asdict(j), tmp_path / "vol")
    assert r["status"] == "crash" and "warm start mismatch" in r["error"]


# ------------------------------------------------------------------ end to end (local executor, tiny budget)
@linux_only
@pytest.mark.process("ws3-bench")
def test_campaign_end_to_end_local(tmp_path, monkeypatch, record):
    """6 paired jobs (2 optimizers x 3 seeds) from the WS1 certificate: V2 records, CSV, certificates, manifest."""
    import interfaces as I
    from certificates import CertificateStore
    from ws3_bench import campaign as C
    monkeypatch.setattr(C, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(C, "CERT_STORE_DIR", tmp_path / "runs" / "certificates")
    budget = 6.0
    assert C.main(["run", "--executor", "local", "--budget", str(budget), "--run-id", "e2e"]) == 0
    run_dir = tmp_path / "runs" / "e2e"
    plan = json.loads((run_dir / "plan.json").read_text())
    assert plan["incumbent"]["ws1_frozen_certificate"] and plan["image_audit"] == []
    assert plan["payload"]["max_job_bytes"] < 100_000                     # no 2M-value warm start in the job
    assert plan["record_reference"]["live_read"] is False
    exp = run_dir / "export"
    recs = [json.loads(l) for l in (exp / "experiments.jsonl").read_text().splitlines()]
    assert len(recs) == 6 and all(set(I.RECORD_FIELDS) == set(r) for r in recs)
    assert {r["status"] for r in recs} <= set(I.STATUSES)
    assert sorted((r["arm"], r["seeds"][0]) for r in recs) == sorted(
        (f"ws3-bench:{o}", s) for o in ("probe", "blockshift") for s in (101, 202, 303))
    assert len({r["best_before"] for r in recs}) == 1 and len({r["parent_id"] for r in recs}) == 1
    rows = list(csv.DictReader(open(exp / "comparison.csv")))
    assert len(rows) == 6
    assert all(row["evals"] or row["proposals"] for row in rows)            # optimizer stdout was captured
    for row in rows:
        if row["valid"] == "True":
            assert abs(float(row["worker_parent_diff"])) <= 1e-12
            assert (exp / "candidates" / f"{row['candidate_sha256']}.json.gz").exists()
    store = CertificateStore(tmp_path / "v2certs")                          # Marie's store re-verifies on load
    problem = C.load_problem(SLUG, False, tmp_path)
    for r in recs:
        if r["certificate_path"]:
            gz = exp / "certificates" / f"{r['certificate_path']}.json.gz"
            dest = store.root / f"{r['certificate_path']}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(gzip.decompress(gz.read_bytes()))
            assert store.load(problem, r["certificate_path"]).score == r["score"]
    manifest = json.loads((exp / "manifest.json").read_text())
    for rel, sha in manifest["files"].items():
        assert C.sha256_file(exp / rel) == sha
    assert manifest["frozen_lock"]["status"] == "ok" and manifest["git"]["commit"]
    assert "not an agent benchmark" in (exp / "summary.md").read_text()
    record(statuses=[r["status"] for r in recs],
           overhead_s=[round(r["duration_s"] - budget, 2) for r in recs if r["duration_s"] is not None])
