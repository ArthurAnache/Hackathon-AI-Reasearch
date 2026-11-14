"""Process 6: analysis.py turns the experiment store into history + best-so-far (demo output)."""
import json
import math
import subprocess
import sys

import pytest

import analysis
import interfaces as I
from conftest import REPO

P = pytest.mark.process("6. Results analysis (demo output)")


def rec(i, problem, score, status, budget=60.0):
    r = {k: None for k in I.RECORD_FIELDS}
    r.update(experiment_id=f"exp_{i:04d}", problem=problem, score=score, status=status, budget_s=budget,
             agent="test", hypothesis=f"h{i}", valid=score is not None, evidence=[], config={})
    return r


@pytest.fixture
def store(tmp_path):
    s = I.JsonlStore(tmp_path / "experiments.jsonl")
    for r in [rec(0, "1a", 1.6, "keep"), rec(1, "1a", 1.55, "keep"), rec(2, "1a", 1.58, "discard"),
              rec(3, "1a", None, "crash"), rec(4, "10c", 1.632993, "keep"), rec(5, "10c", 1.8, "keep"),
              rec(6, "10c", 5 / math.sqrt(8), "discard")]:
        s.append(r)
    return s


@P
def test_summary(store, record):
    """Counts, baseline, best (respecting direction), best-so-far trajectory, record comparison."""
    s = analysis.summarise(store.read())
    record(**{pid: {k: s[pid][k] for k in ("counts", "baseline", "best", "best_so_far", "beats_record")} for pid in s})
    a = s["1a"]
    assert a["counts"]["keep"] == 2 and a["counts"]["discard"] == 1 and a["counts"]["crash"] == 1
    assert a["baseline"] == 1.6 and a["best"] == 1.55 and a["best_experiment"] == "exp_0001"
    assert a["best_so_far"] == [1.6, 1.55, 1.55, 1.55] and a["beats_record"] is False
    assert s["10c"]["best"] == 1.8 and s["10c"]["beats_record"] is True


@P
def test_tie_is_not_a_record(tmp_path, record):
    """A score equal to the record up to float rounding (10c: 5/sqrt 8) is a tie, not a record."""
    s = I.JsonlStore(tmp_path / "e.jsonl")
    s.append(rec(0, "10c", 1.7677669529663687, "keep"))
    out = analysis.summarise(s.read())["10c"]
    record(score=1.7677669529663687, record_value=out["best_known"], beats=out["beats_record"], ties=out["ties_record"])
    assert out["beats_record"] is False and out["ties_record"] is True


@P
def test_cli_json(store, record):
    """`python analysis.py <store> --json` emits valid JSON for a dashboard."""
    out = subprocess.run([sys.executable, "analysis.py", str(store.path), "--json"], capture_output=True,
                         text=True, cwd=REPO)
    data = json.loads(out.stdout)
    record(problems=list(data), fields=sorted(data["1a"]))
    assert set(data) == {"1a", "10c"} and data["1a"]["history"][3]["score"] is None


@P
def test_store_rejects_incomplete_records(tmp_path, record):
    s = I.JsonlStore(tmp_path / "e.jsonl")
    with pytest.raises(ValueError) as e:
        s.append({"experiment_id": "exp_0000", "score": 1.0})
    record(message=str(e.value)[:120])
