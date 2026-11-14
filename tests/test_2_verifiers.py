"""Process 2: each frozen verifier scores candidates correctly and rejects invalid ones.

Correctness is checked three ways: hand-computable cases, published record
constructions (when AR_REFS_DIR is set), and float-vs-exact-arithmetic agreement.
"""
import json
import math
from fractions import Fraction

import numpy as np
import pytest

import problems
from problems import InvalidSolution, p1a, p1b, p1d, p10c, p11a
from conftest import refs_dir

P = pytest.mark.process("2. Verifiers (scoring)")


# ---------------------------------------------------------------- hand-computable values
@P
@pytest.mark.parametrize("problem,solution,expected,why", [
    (p1a, {"values": [1.0]}, 2.0, "indicator of [-1/4,1/4]: max f*f = 1/2, (∫f)^2 = 1/4"),
    (p1a, {"values": [3.0, 3.0, 3.0]}, 2.0, "scaling and refining a constant changes nothing"),
    (p1d, {"values": [1.0, 1.0]}, 2.0, "constant f: same as 1a"),
    (p1d, {"values": [1.0, -1.0, 2.0]}, 7.5, "v*v = [1,-2,5,-4,4]: 2N*5/(Σv)^2 = 6*5/4"),
    (p1b, {"values": [0.5] * 10}, 0.5, "h = 1/2 everywhere: overlap at shift 0 is ∫ 1/2 * 1/2 = 1/2"),
    (p1b, {"values": [0.25] * 8}, 0.5, "rescaled to mass 1, so identical to the case above"),
    (p10c, {"matrix": [[1.0]]}, 1.0, "n=1: disc = 1"),
    (p10c, {"matrix": [[1, 1], [1, -1]]}, math.sqrt(2), "page: sqrt 2 at n = 2"),
    (p10c, {"matrix": [[0, 0], [0, 0]]}, 0.0, "zero matrix has zero discrepancy"),
    (p11a, {"n": 1, "values": [1.0, -1.0]}, 1.0, "f(x) = x on {-1,1}: E|f| = 1, |grad f| = 1"),
    (p11a, {"n": 2, "values": [1.0, -1.0, 1.0, -1.0]}, 1.0, "f(x) = x_1 on {-1,1}^2"),
])
def test_known_values(record, problem, solution, expected, why):
    """Verifier returns the hand-computed value."""
    got = problem.verify(solution)
    record(problem=problem.ID, expected=expected, got=got, why=why)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


@P
def test_11a_invariances(record):
    """11a ratio is invariant under f -> a f + b and under permuting coordinates."""
    rng = np.random.default_rng(0)
    f = rng.normal(size=16)
    base = p11a.verify({"n": 4, "values": f.tolist()})
    affine = p11a.verify({"n": 4, "values": (3 * f - 7).tolist()})
    idx = np.arange(16)
    swapped = ((idx & 1) << 1) | ((idx >> 1) & 1) | (idx & ~3)  # swap coordinates 0 and 1
    perm = p11a.verify({"n": 4, "values": f[swapped].tolist()})
    record(ratio=base, affine=affine, permuted=perm)
    assert affine == pytest.approx(base) and perm == pytest.approx(base)


@P
def test_10c_invariances_and_brute_force(record):
    """10c matches a naive brute force and is invariant to row/column sign flips and permutations."""
    import itertools
    rng = np.random.default_rng(1)
    a = rng.uniform(-1, 1, size=(6, 6))
    naive = min(np.abs(a @ np.array(x)).max() for x in itertools.product((1, -1), repeat=6)) / math.sqrt(6)
    got = p10c.verify({"matrix": a.tolist()})
    b = (a * np.array([1, -1, 1, 1, -1, 1]))[[2, 0, 1, 5, 3, 4]][:, [1, 0, 2, 3, 5, 4]]
    record(naive=naive, verifier=got, transformed=p10c.verify({"matrix": b.tolist()}))
    assert got == pytest.approx(naive) and p10c.verify({"matrix": b.tolist()}) == pytest.approx(got)


@P
@pytest.mark.parametrize("problem", [p1a, p1d, p1b])
def test_float_matches_exact(record, problem):
    """float64 verify() agrees with the exact rational verify_exact() on random candidates."""
    rng = np.random.default_rng(2)
    worst = 0.0
    for n in (5, 50, 3000):           # 3000 > 2048 exercises the FFT path
        v = rng.uniform(0, 1, size=n) if problem is not p1d else rng.normal(1, 1, size=n)
        if problem is p1b:
            v = v * 0.5 / v.mean() * 0.9
        sol = {"values": v.tolist()}
        f, e = problem.verify(sol), float(problem.verify_exact(sol))
        worst = max(worst, abs(f - e) / abs(e))
    record(problem=problem.ID, worst_relative_error=worst)
    assert worst < 1e-10


# ---------------------------------------------------------------- published records
REFERENCES = [
    ("1d", "c1d/construction.json", 1.460217284022874, "c1d-certificate, 600 cells (teorth: 1.460217284022874)"),
    ("1a", "atris/1a-sidon-autocorrelation/construction.json", 1.5027434867095834, "Atris 65,536 steps (current record)"),
    ("1b", "atris/1b-erdos-minimum-overlap/construction.json", 0.38085855031642856, "Atris 114,688 steps (current record)"),
]


@P
@pytest.mark.parametrize("pid,rel,expected,source", REFERENCES)
def test_published_record_rescored(record, pid, rel, expected, source):
    """Re-scoring a published record construction reproduces its published value."""
    root = refs_dir()
    if root is None or not (root / rel).exists():
        pytest.skip("set AR_REFS_DIR to a folder with the published constructions")
    data = json.loads((root / rel).read_text())
    sol = {"values": [float(Fraction(x)) if isinstance(x, str) else float(x) for x in data["values"]]}
    got = problems.get(pid).verify(sol)
    record(problem=pid, source=source, cells=len(sol["values"]), published=expected, rescored=got,
           relative_error=abs(got - expected) / expected)
    assert got == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------- invalid candidates
INVALID = [
    (p1a, None, "not a dict"),
    (p1a, {"values": []}, "empty"),
    (p1a, {"values": [1.0, -0.1]}, "negative height"),
    (p1a, {"values": [0.0, 0.0]}, "zero integral"),
    (p1a, {"values": [1.0, float("nan")]}, "NaN"),
    (p1a, {"values": "abc"}, "wrong type"),
    (p1d, {"values": [1.0, -1.0]}, "zero integral"),
    (p1d, {"values": [1.0, float("inf")]}, "infinite"),
    (p1b, {"values": [0.5]}, "too short"),
    (p1b, {"values": [1.2, 0.0]}, "height above 1"),
    (p1b, {"values": [1.0, 0.0, 0.0, 0.0]}, "mass too small: rescaling to mass 1 pushes a height to 2"),
    (p10c, {"matrix": [[1, 2], [0, 1]]}, "entry outside [-1,1]"),
    (p10c, {"matrix": [[1, 1, 1], [1, 1, 1]]}, "not square"),
    (p10c, {"matrix": [[1] * 21] * 21}, "n above 20"),
    (p11a, {"n": 2, "values": [1, 2, 3]}, "wrong number of values"),
    (p11a, {"n": 2, "values": [1, 1, 1, 1]}, "constant f"),
    (p11a, {"n": True, "values": [1, 2]}, "n not an integer"),
]


@P
@pytest.mark.parametrize("problem,solution,why", INVALID, ids=[f"{p.ID}-{w}" for p, _, w in INVALID])
def test_invalid_rejected(record, problem, solution, why):
    """Invalid candidates raise InvalidSolution (they never get a score)."""
    with pytest.raises(InvalidSolution) as e:
        problem.verify(solution)
    record(problem=problem.ID, case=why, message=str(e.value))


@P
def test_registry_contract(record):
    """Every problem exposes the full contract and a sensible best-known value."""
    for p in problems.PROBLEMS.values():
        for attr in ("ID", "NAME", "URL", "DIRECTION", "BEST_KNOWN", "BEST_KNOWN_SOURCE", "SOLUTION_FORMAT", "verify"):
            assert hasattr(p, attr), (p.ID, attr)
        assert p.DIRECTION in ("minimize", "maximize") and math.isfinite(p.BEST_KNOWN)
    record(problems={p.ID: [p.DIRECTION, p.BEST_KNOWN] for p in problems.PROBLEMS.values()})
