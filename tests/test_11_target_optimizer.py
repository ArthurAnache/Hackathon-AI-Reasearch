"""Process 11: a target-specific optimizer (second-autocorrelation-inequality) and matched comparisons.

The optimizer's speed comes from an exact incremental update of f*f for block moves; these tests check that
the incremental score equals the pinned verifier's, that a job never returns anything worse than its warm
start, that the L4 runner refuses a mismatched optimizer instead of crashing every job, that compare.py
runs a paired, equal-budget head-to-head, and that EinsteinArena records require delta.
"""
import importlib.util
import json
import math
import subprocess
import sys
import types

import numpy as np
import pytest

P = pytest.mark.process("11. Target-specific optimizer + matched comparison")
SLUG = "second-autocorrelation-inequality"


@pytest.fixture
def ws(workspace, monkeypatch):
    monkeypatch.chdir(workspace)
    monkeypatch.syspath_prepend(str(workspace))
    monkeypatch.setenv("EINSTEIN_ARENA_BASE_URL", "http://127.0.0.1:9")    # offline: frozen snapshot
    names = ("prepare", "interfaces", "certificates", "execution", "problems", "evolution", "research", "integrations")
    sub = lambda k: any(k.startswith(n + ".") for n in names)
    saved = {k: v for k, v in sys.modules.items() if k in names or sub(k)}
    for k in saved:
        del sys.modules[k]
    yield workspace
    for k in [k for k in sys.modules if k in names or sub(k)]:
        del sys.modules[k]
    sys.modules.update(saved)


def load_blockshift(root):
    spec = importlib.util.spec_from_file_location("blockshift", root / "optimizers" / f"{SLUG}_blockshift.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@P
def test_incremental_score_equals_pinned_verifier(ws, record):
    """Every accepted block move: the incremental log-score change equals the pinned verifier's, to 1e-12."""
    bs = load_blockshift(ws)
    rng = np.random.default_rng(0)
    f = rng.random(2000) * (rng.random(2000) < 0.3)
    st = bs.State(f.copy(), 1e-6)
    worst, accepted = 0.0, 0
    start = bs.pinned_score(f)
    for _ in range(150):
        b = 2 ** int(rng.integers(0, 8))
        a = int(rng.integers(0, len(f) - b + 1))
        D, v, terms = st.propose(a, b, bs.CONFIG, float(f[f > 0].mean()))
        if v > 1e-16:
            before = math.log(bs.pinned_score(st.f))
            st.apply(a, b, D, terms)
            worst = max(worst, abs((math.log(bs.pinned_score(st.f)) - before) - v))
            accepted += 1
    record(accepted=accepted, worst_abs_error=worst, start=start, end=bs.pinned_score(st.f))
    assert accepted > 20 and worst < 1e-12 and bs.pinned_score(st.f) > start
    assert (st.f >= 0).all()


@P
def test_job_improves_and_never_returns_worse(ws, record):
    """Through the real job path (re-verified certificate as warm start, parent re-verification)."""
    from dataclasses import asdict
    import interfaces as I, execution as E
    from certificates import CertificateStore
    p = I.get_problem(SLUG)
    rng = np.random.default_rng(1)
    sol = {"values": (rng.random(4000) * (rng.random(4000) < 0.4)).tolist()}
    certs = CertificateStore(ws / "certs")
    inc = certs.put(p, sol, p.score(sol))
    job = E.make_job(p, ws / "optimizers" / f"{SLUG}_blockshift.py", {"REFRESH_S": 1.0}, 5, 5.0, inc.id, certs)
    r = E.run_job(asdict(job))
    s = p.score(r["candidate"])
    record(status=r["status"], start=inc.score, end=s, gain=s - inc.score)
    assert r["status"] == "ok" and p.why_invalid(r["candidate"]) is None and s > inc.score


@P
def test_l4_runner_refuses_mismatched_optimizer(ws, record):
    """The failure in results/second-autocorrelation-method-comparison.md: every job crashed because the only
    optimizer was for 1a. Now the CLI picks this problem's optimizer, and refuses a mismatched one up front."""
    run = lambda *a: subprocess.run([sys.executable, "-m", "evolution", "--problem", SLUG, *a], cwd=ws,
                                    capture_output=True, text=True, timeout=120,
                                    env={**__import__("os").environ, "EINSTEIN_ARENA_BASE_URL": "http://127.0.0.1:9"})
    bad = run("--optimizer", "optimizers/1a_tuned.py", "--rounds", "1")
    sys.modules.pop("evolution.__main__", None)
    import evolution.__main__ as M
    record(refusal=(bad.stderr or bad.stdout).strip().splitlines()[-1], default=M.default_optimizer(SLUG))
    assert bad.returncode != 0 and "declares PROBLEM = '1a'" in (bad.stderr + bad.stdout)
    assert M.default_optimizer(SLUG) == f"optimizers/{SLUG}_blockshift.py"


@P
def test_compare_is_paired_and_matched(ws, record):
    """compare.py: same incumbent, same seeds, same budget, one record per optimizer, a paired table."""
    store = ws / "cmp.jsonl"
    proc = subprocess.run([sys.executable, "compare.py", "--problem", "1a", "--optimizers", "optimizers/1a_baseline.py",
                           "optimizers/1a_tuned.py", "--seeds", "1", "2", "3", "--budget", "4", "--workers", "6",
                           "--store", str(store), "--certs", str(ws / "certs"), "--report", str(ws / "r.md")],
                          cwd=ws, capture_output=True, text=True, timeout=300)
    recs = [json.loads(l) for l in store.read_text().splitlines()]
    record(stdout_tail=proc.stdout.strip().splitlines()[-6:], arms=[r["arm"] for r in recs])
    assert proc.returncode == 0, proc.stderr[-500:]
    assert len(recs) == 2 and all(r["seeds"] == [1, 2, 3] and r["budget_s"] == 4 for r in recs)
    assert len({r["parent_id"] for r in recs}) == 1 and all(len(r["scores"]) == 3 for r in recs)
    assert "paired difference" in (ws / "r.md").read_text()


@P
def test_einsteinarena_record_needs_delta(ws, record):
    import problems
    m = types.SimpleNamespace(BEST_KNOWN=0.9636615620620852, DIRECTION="maximize", DELTA=1e-5)
    plain = types.SimpleNamespace(BEST_KNOWN=1.5, DIRECTION="minimize")
    out = {"+3e-8": problems.beats_record(m.BEST_KNOWN + 3e-8, m), "+1.1e-5": problems.beats_record(m.BEST_KNOWN + 1.1e-5, m),
           "no delta, -1e-6": problems.beats_record(1.5 - 1e-6, plain)}
    record(**out)
    assert out == {"+3e-8": False, "+1.1e-5": True, "no delta, -1e-6": True}
