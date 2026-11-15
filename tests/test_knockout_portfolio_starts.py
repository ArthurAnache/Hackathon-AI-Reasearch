"""Start portfolio: trusted start generators (knockout/starts.py) and the circles-rectangle seed program.

Fast, no network, no LLM. The frozen verifier is the judge of validity.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from knockout import starts, toy
from knockout.sandbox import SandboxJob, run_job
from problems.catalog import load

ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = ROOT / "knockout" / "seeds" / "circles_rectangle.py"
CIRCLES = {"toy-circles-26": 26, "circles-rectangle": 21}


@pytest.fixture(scope="module")
def cr():
    return load("circles-rectangle")


@pytest.fixture(scope="module")
def seed_program():
    spec = importlib.util.spec_from_file_location("seed_circles_rectangle", SEED_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify(problem_id, cand):
    return toy.verify(cand) if problem_id == "toy-circles-26" else load(problem_id).verify(cand)


@pytest.mark.parametrize("pid", sorted(CIRCLES))
def test_cold_starts_deterministic_and_canonical(pid):
    a, b = starts.cold_starts(pid, 3, seed=7), starts.cold_starts(pid, 3, seed=7)
    assert a == b and len(a) == 3
    assert a != starts.cold_starts(pid, 3, seed=8)
    for cand in a:
        assert json.loads(json.dumps(cand)) == cand          # canonical JSON form
        assert len(cand["circles"]) == CIRCLES[pid]
        assert all(len(row) == 3 and all(type(v) is float for v in row) for row in cand["circles"])
        assert max(row[2] for row in cand["circles"]) <= starts.COLD_RADIUS
        assert _verify(pid, cand)                             # documented: valid except on coinciding centres


@pytest.mark.parametrize("pid", sorted(CIRCLES))
def test_perturb_deterministic_canonical_and_valid(pid, cr):
    base = cr.certificate() if pid == "circles-rectangle" else starts.cold_starts(pid, 1, seed=0)[0]
    for strength in (1e-3, 1e-2, 0.1, 0.5):                  # 0.5: many centres hit the box sides
        a = starts.perturb(pid, base, strength, seed=3)
        assert a == starts.perturb(pid, base, strength, seed=3)
        assert a != starts.perturb(pid, base, strength, seed=4)
        assert len(a["circles"]) == CIRCLES[pid] and json.loads(json.dumps(a)) == a
        assert _verify(pid, a)
    if pid == "circles-rectangle":
        p = np.asarray(base["circles"])
        q = np.asarray(starts.perturb(pid, base, 0.05, seed=1)["circles"])
        lo, hi = (p[:, :2] - p[:, 2:]).min(0), (p[:, :2] + p[:, 2:]).max(0)
        assert ((q[:, :2] - q[:, 2:]) >= lo - 1e-15).all() and ((q[:, :2] + q[:, 2:]) <= hi + 1e-15).all()


def test_descriptor_keys(cr):
    toy_d = starts.descriptors("toy-circles-26", starts.cold_starts("toy-circles-26", 1, 0)[0])
    assert set(toy_d) == {"r_q25_rel", "r_q75_rel", "r_max_rel", "n_boundary"}
    cr_d = starts.descriptors("circles-rectangle", cr.certificate())
    assert set(cr_d) == {"r_q25_rel", "r_q75_rel", "r_max_rel", "aspect", "n_small"}
    assert all(isinstance(v, float) for v in {**toy_d, **cr_d}.values())
    assert 0.9 < cr_d["aspect"] < 1.1


def test_second_autocorrelation_starts():
    pid = "second-autocorrelation-inequality"
    a = starts.cold_starts(pid, 2, seed=1)
    assert a == starts.cold_starts(pid, 2, seed=1)
    assert all(len(c["values"]) == starts.SA_COLD_LEN and min(c["values"]) >= 0 for c in a)
    p = starts.perturb(pid, a[1], 0.5, seed=2)
    assert p == starts.perturb(pid, a[1], 0.5, seed=2) and min(p["values"]) >= 0
    assert set(starts.descriptors(pid, p)) == {"support_fraction", "runs_per_1000", "largest_run_mass_share"}


def test_unknown_problem_raises():
    with pytest.raises(ValueError):
        starts.cold_starts("ring-loading-15", 1, 0)


@pytest.mark.parametrize("start", ["none", "cold", "certificate"])
def test_seed_program_returns_valid_candidate(start, cr, seed_program):
    init = {"none": None, "cold": starts.cold_starts("circles-rectangle", 1, 5)[0], "certificate": cr.certificate()}
    params = {k: v[0] for k, v in seed_program.PARAMS.items()}
    cand = seed_program.construct(params, 0, 2.0, init[start])
    cand = json.loads(json.dumps(cand))
    assert cr.verify(cand), cr.why_invalid(cand)
    score = cr.score(cand)
    if start == "certificate":
        assert score >= cr.score(cr.certificate()) - 1e-12
    else:
        assert score > 2.0                                    # far above the cold start (sum of radii 0.021)


def test_seed_program_in_sandbox_from_cold_start(cr):
    job = SandboxJob(program_source=SEED_PATH.read_text(), seed=1, budget_s=3.0, direction="maximize",
                     guide_ref="knockout.problems_api:catalog_guide", guide_args=["circles-rectangle"],
                     mode="construct", init=starts.cold_starts("circles-rectangle", 1, 9)[0])
    res = run_job(job)
    assert res.ok, res.error
    cand = res.trials[0]["candidate"]
    assert cr.verify(cand) and cr.score(cand) == pytest.approx(res.trials[0]["guide"], abs=0)
