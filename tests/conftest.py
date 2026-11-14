"""Shared fixtures + a small recorder that writes a per-process test report.

Every test is tagged with @pytest.mark.process("<stage>") and may call
record(key=value, ...) to log what it observed (scores, statuses, timings).
At the end of the session the results are written to:

    reports/test_report.md     human overview, grouped by process stage
    reports/test_report.json   the same data, machine-readable

Reference constructions (published records) are used when AR_REFS_DIR points at
a folder containing c1d/construction.json and atris/1{a,b}-*/construction.json.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
REPORT_DIR = REPO / "reports"
_RESULTS: dict[str, dict] = {}


# ------------------------------------------------------------------ recorder
@pytest.fixture
def record(request):
    obs = _RESULTS.setdefault(request.node.nodeid, {"observations": {}})["observations"]

    def _record(**kwargs):
        obs.update(kwargs)

    return _record


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    entry = _RESULTS.setdefault(item.nodeid, {"observations": {}})
    marker = item.get_closest_marker("process")
    entry["process"] = marker.args[0] if marker else "unclassified"
    entry["doc"] = (item.function.__doc__ or "").strip().splitlines()[0] if item.function.__doc__ else item.name
    if rep.when == "call" or (rep.when == "setup" and rep.outcome != "passed"):
        entry["outcome"] = "skipped" if rep.skipped else rep.outcome
        entry["duration_s"] = round(entry.get("duration_s", 0) + rep.duration, 3)
        if rep.failed:
            entry["error"] = str(rep.longrepr).strip().splitlines()[-1][:300]
        if rep.skipped:
            entry["error"] = str(rep.longrepr[-1]) if isinstance(rep.longrepr, tuple) else str(rep.longrepr)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.12g}"
    return json.dumps(v) if isinstance(v, (dict, list)) else str(v)


def pytest_sessionfinish(session, exitstatus):
    if not _RESULTS:
        return
    REPORT_DIR.mkdir(exist_ok=True)
    tests = [{"nodeid": k, **v} for k, v in _RESULTS.items() if "outcome" in v]
    (REPORT_DIR / "test_report.json").write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tests": tests}, indent=2, default=str))
    by_proc: dict[str, list] = {}
    for t in tests:
        by_proc.setdefault(t["process"], []).append(t)
    icon = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}
    lines = ["# Test report", "",
             f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. "
             f"{sum(t['outcome'] == 'passed' for t in tests)} passed, "
             f"{sum(t['outcome'] == 'failed' for t in tests)} failed, "
             f"{sum(t['outcome'] == 'skipped' for t in tests)} skipped.", "",
             "| Process | Passed | Failed | Skipped |", "|---|---|---|---|"]
    for proc in sorted(by_proc):
        ts = by_proc[proc]
        lines.append(f"| {proc} | {sum(t['outcome'] == 'passed' for t in ts)} | "
                     f"{sum(t['outcome'] == 'failed' for t in ts)} | {sum(t['outcome'] == 'skipped' for t in ts)} |")
    for proc in sorted(by_proc):
        lines += ["", f"## {proc}", ""]
        for t in by_proc[proc]:
            name = t["nodeid"].split("::", 1)[1]
            lines.append(f"- **{icon[t['outcome']]}** `{name}` ({t['duration_s']}s): {t['doc']}")
            for k, v in t["observations"].items():
                lines.append(f"  - {k}: {_fmt(v)}")
            if t.get("error"):
                lines.append(f"  - reason: {t['error']}")
    (REPORT_DIR / "test_report.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ workspace
def _git(root, *args, check=True):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=check).stdout


@pytest.fixture
def workspace(tmp_path):
    """A throwaway copy of the repo in its own git repo (main branch, one commit)."""
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("prepare.py", "interfaces.py", "loop.py", "analysis.py", "frozen.lock", "program.md", ".gitignore"):
        shutil.copy(REPO / name, root / name)
    for d in ("problems", "optimizers", "research"):
        shutil.copytree(REPO / d, root / d, ignore=shutil.ignore_patterns("__pycache__"))
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    return root


def run_search(root: Path, budget: float = 1.0, timeout: float = 120, hypothesis: str = "test",
               file: str = "search.py") -> tuple[dict, subprocess.CompletedProcess, float]:
    """`python search.py "<hypothesis>" > run.log 2>&1` in `root`; returns (summary fields, process, wall seconds)."""
    env = {**os.environ, "AR_TIME_BUDGET": str(budget)}
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, file, hypothesis], cwd=root, capture_output=True, text=True, env=env,
                          timeout=timeout)
    wall = time.monotonic() - t0
    (root / "run.log").write_text(proc.stdout + proc.stderr)
    fields = {}
    if "---" in proc.stdout:
        for line in proc.stdout.split("---", 1)[1].splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fields[k.strip()] = v.strip()
    return fields, proc, wall


def refs_dir() -> Path | None:
    p = os.environ.get("AR_REFS_DIR")
    return Path(p) if p and Path(p).exists() else None
