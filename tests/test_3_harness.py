"""Process 3: the experiment protocol (interfaces.run_experiment via `python search.py`):
fixed budget + seed, independent verification, keep/discard rule, solution saved, record stored."""
import json
import subprocess
import sys
import textwrap

import pytest

from conftest import run_search

P = pytest.mark.process("3. Experiment protocol (harness)")


def write_search(root, body, problem="1a", config="{}"):
    (root / "search.py").write_text(textwrap.dedent(f'''
        import time
        from prepare import run
        PROBLEM = "{problem}"
        CONFIG = {config}
    ''') + textwrap.dedent(body) + '\nif __name__ == "__main__":\n    run(__file__)\n')


def records(root):
    p = root / "experiments.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


@P
def test_summary_block_and_record(workspace, record):
    """A valid run prints score/status/decision and appends a complete record to experiments.jsonl."""
    write_search(workspace, '''
        def search(budget_s, seed, config=CONFIG):
            return {"values": [1.0, 1.0]}
    ''')
    f, proc, wall = run_search(workspace, budget=1, hypothesis="flat profile scores 2")
    r = records(workspace)[0]
    record(summary=f, record_fields=sorted(r))
    assert proc.returncode == 0
    assert float(f["score"]) == pytest.approx(2.0) and f["status"] == "keep" and f["decision"] == "keep"
    assert r["hypothesis"] == "flat profile scores 2" and r["valid"] is True and r["seed"] == 42
    assert r["budget_s"] == 1.0 and r["algorithm_version"] and r["status"] == "keep"


@P
def test_keep_then_discard(workspace, record):
    """Second run keeps only if strictly better than the best kept run (same problem, same budget)."""
    import os
    write_search(workspace, '''
        import os
        def search(budget_s, seed, config=CONFIG):
            return {"values": [float(x) for x in os.environ["VALS"].split(",")]}
    ''')
    seen = []
    for vals in ("1,2,1", "1,1", "1,3,3,1", "1,1"):   # 2.25 keep, 2.0 keep, 2.5 discard, 2.0 discard (tie)
        os.environ["VALS"] = vals
        f, _, _ = run_search(workspace)
        seen.append((f["score"], f["status"], f["decision"]))
    del os.environ["VALS"]
    record(runs=seen)
    assert [s for _, s, _ in seen] == ["keep", "keep", "discard", "discard"]
    assert [d for _, _, d in seen] == ["keep", "keep", "reset", "reset"]


@P
def test_budget_seed_and_config_are_passed(workspace, record):
    """search() receives the fixed budget, the fixed seed and the CONFIG dict."""
    write_search(workspace, '''
        import json
        def search(budget_s, seed, config=CONFIG):
            open("args.json", "w").write(json.dumps([budget_s, seed, config]))
            return {"values": [1.0]}
    ''', config='{"K": 3}')
    run_search(workspace, budget=2.5)
    args = json.loads((workspace / "args.json").read_text())
    record(budget_s=args[0], seed=args[1], config=args[2])
    assert args == [2.5, 42, {"K": 3}]


@P
def test_invalid_solution(workspace, record):
    """A candidate that breaks a constraint: status invalid, no score, decision reset, reason recorded."""
    write_search(workspace, '''
        def search(budget_s, seed, config=CONFIG):
            return {"values": [-1.0, 2.0]}
    ''')
    f, _, _ = run_search(workspace)
    r = records(workspace)[0]
    record(status=f.get("status"), error=r["error"])
    assert "score" not in f and f["status"] == "invalid" and f["decision"] == "reset" and r["valid"] is False


@P
def test_crash(workspace, record):
    """A crashing optimizer: exit code 1, status crash, error recorded; the loop can continue."""
    write_search(workspace, '''
        def search(budget_s, seed, config=CONFIG):
            raise ValueError("boom")
    ''')
    f, proc, _ = run_search(workspace)
    record(returncode=proc.returncode, error=f.get("error"))
    assert proc.returncode == 1 and f["status"] == "crash" and "ValueError: boom" in f["error"]


@P
def test_over_budget_is_void(workspace, record):
    """Overrunning the budget by more than 10% + 1s voids the result."""
    write_search(workspace, '''
        def search(budget_s, seed, config=CONFIG):
            time.sleep(budget_s * 1.1 + 1.5)
            return {"values": [1.0]}
    ''')
    f, _, wall = run_search(workspace, budget=1)
    record(status=f.get("status"), error=f.get("error"), wall_s=round(wall, 2))
    assert f["status"] == "void" and "over time budget" in f["error"] and f["decision"] == "reset"


@P
def test_watchdog_kills_runaway_search(workspace, record):
    """An optimizer that never returns is killed at 2x budget + 30s and recorded as a crash."""
    write_search(workspace, '''
        def search(budget_s, seed, config=CONFIG):
            while True:
                time.sleep(1)
    ''')
    f, proc, wall = run_search(workspace, budget=0.5, timeout=90)
    record(status=f.get("status"), wall_s=round(wall, 1), error=f.get("error"))
    assert f["status"] == "crash" and "watchdog" in f["error"] and wall < 60


@P
def test_tampered_verifier_voids_result(workspace, record):
    """If the verifier is edited to cheat, the run is void and never kept."""
    p = workspace / "problems" / "p1a.py"
    p.write_text(p.read_text().replace("return float(2 * len(a)", "return 0.0 * float(2 * len(a)"))
    write_search(workspace, '''
        def search(budget_s, seed, config=CONFIG):
            return {"values": [1.0]}
    ''')
    f, _, _ = run_search(workspace)
    record(status=f.get("status"), error=f.get("error"))
    assert f["status"] == "void" and "MODIFIED" in f["error"] and f["decision"] == "reset"


@P
def test_deterministic_given_seed(workspace, record):
    """An iteration-bounded optimizer gives the identical score on repeated runs (fixed seed)."""
    write_search(workspace, '''
        import numpy as np
        def search(budget_s, seed, config=CONFIG):
            return {"values": np.random.default_rng(seed).uniform(0, 1, 40).tolist()}
    ''')
    a = run_search(workspace)[0]["score"]
    b = run_search(workspace)[0]["score"]
    record(run1=a, run2=b)
    assert a == b


@P
def test_solution_saved_and_certifiable(workspace, record):
    """The saved solution re-verifies with `prepare.py verify --exact` to the logged score."""
    write_search(workspace, '''
        def search(budget_s, seed, config=CONFIG):
            return {"values": [0.3, 1.0, 0.7, 0.2]}
    ''')
    f, _, _ = run_search(workspace)
    out = subprocess.run([sys.executable, "prepare.py", "verify", "1a", f["solution_path"], "--exact"],
                         cwd=workspace, capture_output=True, text=True).stdout
    lines = dict(l.split(":", 1) for l in out.strip().splitlines())
    record(logged=f["score"], reverified=lines["score"].strip(), exact=lines["exact"].strip())
    assert float(lines["exact"].split()[0]) == pytest.approx(float(f["score"]), rel=1e-14)
