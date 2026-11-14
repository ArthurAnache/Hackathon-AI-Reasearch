"""Process 4: every optimizer in optimizers/ runs end to end in Karpathy mode
(`cp optimizers/<x>.py search.py && python search.py "..."`) and yields a verified candidate."""
import shutil

import pytest

import interfaces as I
from conftest import REPO, run_search

P = pytest.mark.process("4. Optimizers")
BUDGET = 2.0
FILES = sorted(p.name for p in (REPO / "optimizers").glob("*.py"))


@P
@pytest.mark.parametrize("name", FILES)
def test_optimizer_runs(workspace, record, name):
    """status keep (first run), within budget, never a (false) record at 2 s."""
    shutil.copy(workspace / "optimizers" / name, workspace / "search.py")
    f, proc, wall = run_search(workspace, budget=BUDGET, hypothesis=f"smoke {name}")
    p = I.get_problem(f.get("problem", "?"))
    record(optimizer=name, status=f.get("status"), score=float(f.get("score", "nan")),
           best_known=p.best_known, duration_s=float(f.get("duration_s", "nan")), budget_s=BUDGET)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert f["status"] == "keep" and float(f["duration_s"]) <= BUDGET * 1.1 + 1
    assert f["beats_record"] == "False"


@P
def test_every_problem_has_baseline_and_tuned(record):
    import problems
    missing = [f"{pid}_{kind}.py" for pid in problems.PROBLEMS for kind in ("baseline", "tuned")
               if f"{pid}_{kind}.py" not in FILES]
    record(optimizers=FILES)
    assert not missing
