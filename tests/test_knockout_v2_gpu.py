"""Fast local tests of knockout/v2/gpu (no torch, no Modal): schedules, bookkeeping, and the batched objective,
L-BFGS and annealing loop run on the numpy backend (the GPU runs the same lines on torch).

    .venv/bin/python -m pytest tests/test_knockout_v2_gpu.py -q
"""
import math

import numpy as np
import pytest

from knockout.seeds import second_autocorrelation as seed
from knockout.v2.gpu import schedule as sch
from knockout.v2.gpu.chain import initial_w, reopen_cells, run_chains
from knockout.v2.gpu.lbfgs import BatchedLBFGS
from knockout.v2.gpu.objective import autoconvolution, evaluate, exact_score, fft_length, kappa_for
from knockout.v2.gpu.starts import along_records, coarsen_blocks, kick


def _start(n=60, rows=2, seed_=0, sparse=True):
    rng = np.random.default_rng(seed_)
    f = rng.random((rows, n)) + 0.1
    if sparse:
        f[:, rng.random(n) < 0.4] = 0.0
    return f


# ---------------------------------------------------------------- schedules and bookkeeping

def test_fft_length_is_linear_and_smooth():
    assert fft_length(1_999_999) == 4_000_000
    assert fft_length(1_999_999, "pow2") == 2 ** 22
    for n in (1, 2, 7, 60, 1000, 123_457):
        m = fft_length(n)
        assert m >= 2 * n - 1
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        assert m == 1


def test_geometric_and_anneal():
    assert sch.geometric(1e4, 1e10, 7) == pytest.approx([1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10])
    assert sch.geometric(1e9, 1e9, 1) == [1e9]
    stages = sch.anneal(1e5, 1e9, 3, 1300, polish=(3e9, 1e10), polish_evals=301)
    assert len(stages) == 13 + 2
    assert stages[0].beta == 1e5 and stages[12].beta == 1e9 and stages[-1].beta == 1e10
    assert all(a.beta < b.beta for a, b in zip(stages, stages[1:]))
    assert sch.total_evals(stages) == 1601
    assert [s.evals for s in stages[13:]] == [151, 150]
    assert sch.boundaries(stages)[-1] == 1601
    with pytest.raises(ValueError):
        sch.anneal(1e9, 1e5, 3, 100)
    with pytest.raises(ValueError):
        sch.split_evals(3, 5)


def test_plans_share_one_budget_and_roundtrip():
    for name in sch.PLANS:
        for evals in (400, 12_345):
            chains = sch.build_plan(name, "frozen", evals)
            assert sch.check_batch(chains) == evals
            for c in chains:
                assert sch.ChainSpec.from_dict(c.to_dict()) == c
                assert c.stages[-1].beta >= 1e9          # every chain ends sharp
    kicks = sch.build_plan("kick16", "live", 6000)
    assert len(kicks) == 16 and kicks[0].jitter == 0 and all(c.jitter > 0 for c in kicks[1:])
    assert len({(c.jitter, c.seed, c.stages[0].beta, c.floor) for c in kicks}) == 16
    with pytest.raises(ValueError):
        sch.check_batch([sch.ChainSpec("a", "live", (sch.Stage(1e9, 5),), jitter=-0.1)])
    deep = {c.name: c for c in sch.build_plan("deep", "live", 6000)}
    assert deep["control"].stages[0].beta == 1e9 and deep["control"].floor == 0
    assert deep["b4-slow"].stages[0].beta == 1e4 and deep["b4-open"].floor > 0
    with pytest.raises(ValueError):
        sch.check_batch([sch.ChainSpec("a", "live", (sch.Stage(1e9, 5),)),
                         sch.ChainSpec("b", "live", (sch.Stage(1e9, 6),))])
    with pytest.raises(ValueError):
        sch.check_batch([sch.ChainSpec("a", "live", (sch.Stage(1e9, 5),)),
                         sch.ChainSpec("a", "live", (sch.Stage(1e9, 5),))])


def test_ids_records_and_cost():
    assert sch.experiment_id("r1", "b5-slow", sch.step_name(3)) == "r1/b5-slow/s003"
    assert sch.solution_path("r1/b5-slow/best") == "r1/b5-slow/best.npy"
    with pytest.raises(ValueError):
        sch.experiment_id("r1", "a/b", "s000")
    chain = sch.build_plan("smoke", "live", 20)[0]
    env = {"device": "cuda", "gpu_type": "H100", "dtype": "float64", "torch_version": "x", "cuda_version": "y"}
    common = dict(run_id="r1", chain=chain, step="best", gpu_objective=0.96, gpu_exact=0.9637, env=env,
                  wall_s=1.0, budget_s=10.0, source_commit="abc", start_score=0.9636, best_known_score=0.96366)
    keep = sch.make_record(score=0.9637, valid=True, **common)
    assert keep["decision"] == "keep" and keep["solution_path"] == "r1/polish/best.npy"
    assert keep["gain_over_start"] == pytest.approx(1e-4) and keep["gain_over_live"] == pytest.approx(4e-5)
    assert all(k in keep for k in sch.SHARED_FIELDS + sch.GPU_FIELDS)
    assert "record" not in keep["decision"]                  # a chain never claims a record
    assert sch.make_record(score=0.9635, valid=True, **common)["decision"] == "discard"
    assert sch.make_record(score=None, valid=False, **common)["decision"] == "discard"
    est = sch.estimate(10_000, 0.1, overhead_s=200)
    assert est["wall_s"] == pytest.approx(1200) and est["usd"] == pytest.approx(1200 * sch.container_usd_per_s())
    assert 0.0012 < sch.container_usd_per_s() < 0.0013       # H100 + 6 cores + 32 GiB, prices of 2026-11-14


# ---------------------------------------------------------------- objective (numpy backend)

def test_objective_matches_seed_and_verifier_formula():
    f = _start(n=80, rows=3)
    m = fft_length(80)
    kappa = kappa_for(np, f, m)
    w = np.sqrt(f * (kappa / f.sum(-1)[:, None]))
    for beta in (1e2, 1e5, 1e9):
        phi, grad, aux = evaluate(np, w, np.full((3, 1), beta), kappa, m)
        for i in range(3):
            fi = w[i] ** 2
            assert aux["exact"][i] == pytest.approx(seed.exact_score(fi), rel=1e-13)
            # with max(f*f) = 1 the seed's objective is the same number
            ref, _ = seed._neg_log_c(w[i], beta, m, 80)
            assert phi[i] == pytest.approx(ref, rel=1e-11, abs=1e-13)
            assert aux["smooth"][i] <= aux["exact"][i] * (1 + 1e-14)
        assert exact_score(np, w * w, m) == pytest.approx(aux["exact"], rel=1e-13)
        # scale invariance: the gradient is orthogonal to w
        assert np.abs((grad * w).sum(-1)).max() < 1e-10 * np.abs(grad).max() * np.abs(w).max() * 80


def test_objective_gradient_finite_differences():
    f = _start(n=40, rows=2, seed_=3, sparse=False)
    m = fft_length(40)
    kappa = kappa_for(np, f, m)
    w = np.sqrt(f * (kappa / f.sum(-1)[:, None]))
    beta = np.array([[30.0], [300.0]])
    _, grad, _ = evaluate(np, w, beta, kappa, m)
    rng = np.random.default_rng(1)
    for _ in range(5):
        v = rng.normal(size=w.shape)
        h = 1e-6
        up, _, _ = evaluate(np, w + h * v, beta, kappa, m)
        dn, _, _ = evaluate(np, w - h * v, beta, kappa, m)
        assert (up - dn) / (2 * h) == pytest.approx((grad * v).sum(-1), rel=1e-6, abs=1e-9)


def test_objective_is_scale_invariant_and_linear_convolution():
    f = _start(n=50, rows=2, seed_=5)
    m = fft_length(50)
    kappa = kappa_for(np, f, m)
    beta = np.full((2, 1), 1e4)
    w = np.sqrt(f)
    a, _, _ = evaluate(np, w, beta, kappa, m)
    b, _, _ = evaluate(np, 37.0 * w, beta, kappa, m)
    assert a == pytest.approx(b, rel=1e-12)
    u, _ = autoconvolution(np, f, m)
    assert u[0] == pytest.approx(np.convolve(f[0], f[0]), abs=1e-12)   # full, not circular


# ---------------------------------------------------------------- batched L-BFGS and the annealing loop

def test_lbfgs_rows_are_independent_quadratics():
    rng = np.random.default_rng(0)
    n = 30
    scales = np.stack([np.logspace(0, 1, n), np.logspace(0, 3, n), np.ones(n)])
    target = rng.normal(size=(3, n))

    def fun(x):
        r = x - target
        return 0.5 * (scales * r * r).sum(-1), scales * r

    opt = BatchedLBFGS(np, np.zeros((3, n)), history=8)
    for _ in range(400):
        xt = opt.trial()
        f, g = fun(xt)
        opt.update(xt, f, g)
    assert np.abs(opt.x - target).max() < 1e-6
    assert all(a > 0 for a in opt.accepted)
    # a row never accepts a step that raises its objective
    opt2 = BatchedLBFGS(np, np.zeros((3, n)), history=4)
    last = np.full(3, np.inf)
    for _ in range(60):
        xt = opt2.trial()
        f, g = fun(xt)
        opt2.update(xt, f, g)
        assert (opt2.f <= last + 1e-15).all()
        last = opt2.f.copy()


def test_run_chains_bookkeeping_and_best_point():
    n = 64
    f0 = np.repeat(_start(n=n, rows=1, seed_=7), 3, axis=0)
    m = fft_length(n)
    kappa = kappa_for(np, f0, m)
    chains = [
        sch.ChainSpec("control", "x", (sch.Stage(1e3, 60), sch.Stage(1e4, 60))),
        sch.ChainSpec("reheat", "x", (sch.Stage(3.0, 40), sch.Stage(30.0, 40), sch.Stage(1e4, 40))),
        sch.ChainSpec("open", "x", (sch.Stage(3.0, 40), sch.Stage(30.0, 40), sch.Stage(1e4, 40)), floor=1e-2),
    ]
    assert sch.check_batch(chains) == 120
    floors = np.array([[c.floor] for c in chains])
    w0 = initial_w(np, f0, kappa, floors)
    assert (w0[0] == 0).sum() == (f0[0] == 0).sum() and (w0[2] > 0).all()   # only the floored chain opens cells
    start_exact = exact_score(np, f0, m)
    ends, progress = [], []

    def on_stage_end(row, stage, evals, w_row, info):
        ends.append((row, stage, evals, info["beta"]))
        assert info["exact"] == pytest.approx(float(exact_score(np, (w_row * w_row)[None, :], m)[0]), rel=1e-12)

    out = run_chains(np, w0, kappa, m, [c.stages for c in chains], history=6, on_stage_end=on_stage_end,
                     on_progress=lambda *a: progress.append(a[0]), progress_every=50)
    assert out["evals"] == 120 and not out["stopped_early"]
    assert sorted(ends) == [(0, 0, 60, 1e3), (0, 1, 120, 1e4), (1, 0, 40, 3.0), (1, 1, 80, 30.0),
                            (1, 2, 120, 1e4), (2, 0, 40, 3.0), (2, 1, 80, 30.0), (2, 2, 120, 1e4)]
    assert progress == [50, 100, 120]
    # the best point is never below the chain's own first point, and it is a real point
    assert (out["best_exact"] >= out["first_exact"] - 1e-15).all()
    assert exact_score(np, out["best_w"] ** 2, m) == pytest.approx(out["best_exact"], rel=1e-12)
    assert out["first_exact"][0] == pytest.approx(start_exact[0], rel=1e-12)
    assert out["best_exact"][0] > start_exact[0]              # the polish-only chain improves a random start
    # zero cells of a chain without floor stay exactly zero
    assert ((out["x"][1] == 0) == (f0[1] == 0)).all()
    assert math.isfinite(float(out["smooth"].min()))


def test_reopen_opens_only_closed_cells_with_positive_derivative():
    n = 48
    f0 = _start(n=n, rows=2, seed_=11)
    m = fft_length(n)
    kappa = kappa_for(np, f0, m)
    w = initial_w(np, f0, kappa, np.zeros((2, 1)))
    beta = np.full((2, 1), 50.0)
    _, _, aux = evaluate(np, w, beta, kappa, m)
    eta = np.array([[1e-2], [0.0]])
    w2, counts = reopen_cells(np, w, aux["grad_f"], eta)
    closed = w[0] == 0
    should = closed & (aux["grad_f"][0] > 0)
    assert should.any() and counts[0] == should.sum() and counts[1] == 0
    assert ((w2[0] > 0) == ((w[0] > 0) | should)).all()         # nothing else opens
    assert (w2[0][~should] == w[0][~should]).all() and (w2[1] == w[1]).all()
    mean_open = (w[0] ** 2)[w[0] > 0].mean()
    assert (w2[0][should] ** 2).max() == pytest.approx(1e-2 * mean_open, rel=1e-12)
    # the derivative on closed cells is the true one: opening a violating cell a little raises the objective
    j = int(np.argmax(np.where(should, aux["grad_f"][0], -np.inf)))
    w3 = w.copy()
    w3[0, j] = 1e-4
    before, _, _ = evaluate(np, w, beta, kappa, m)
    after, _, _ = evaluate(np, w3, beta, kappa, m)
    assert after[0] < before[0]
    assert (before[0] - after[0]) == pytest.approx(aux["grad_f"][0, j] * 1e-8, rel=1e-3)


def test_run_chains_with_reopen_keeps_the_budget():
    n = 64
    f0 = np.repeat(_start(n=n, rows=1, seed_=7), 2, axis=0)
    m = fft_length(n)
    kappa = kappa_for(np, f0, m)
    w0 = initial_w(np, f0, kappa, np.zeros((2, 1)))
    stages = [(sch.Stage(1e3, 60), sch.Stage(1e4, 60))] * 2
    ends = []
    out = run_chains(np, w0, kappa, m, stages, reopen=[0.0, 1e-2], reopen_every=30,
                     on_stage_end=lambda row, stage, evals, w_row, info: ends.append((row, evals, info["reopened"])))
    assert out["evals"] == 120 and out["extra_evals"] == 4       # events at 0, 30, 60, 90 (not after 90 % = 108)
    assert out["reopened"][0] == 0 and out["reopened"][1] > 0
    assert ((out["x"][0] == 0) == (f0[0] == 0)).all()            # the row without reopen keeps its support
    assert ((out["x"][1] != 0) & (f0[1] == 0)).any()             # the other row opened cells
    assert sorted(e[:2] for e in ends) == [(0, 60), (0, 120), (1, 60), (1, 120)]
    assert (out["best_exact"] >= out["first_exact"] - 1e-15).all()
    # cells are opened only while the chain is hot (beta <= reopen_beta_max): here during the first stage only
    hot = run_chains(np, w0, kappa, m, stages, reopen=[0.0, 1e-2], reopen_every=30, reopen_beta_max=5e3)
    assert hot["extra_evals"] == 2 and hot["evals"] == 120


def test_start_changes_keep_block_masses():
    rng = np.random.default_rng(0)
    f = rng.random(103) + 0.1
    f[rng.random(103) < 0.4] = 0.0
    f[40:50] = 0.0                                             # one block of 10 with no mass

    def block_mass(x, k):
        pad = (-x.size) % k
        return np.concatenate([x, np.zeros(pad)]).reshape(-1, k).sum(1)

    free = kick(f, 0.5, 3)
    assert free is not f and (free >= 0).all() and ((free > 0) == (f > 0)).all()
    assert kick(f, 0.0, 3) is f and (kick(f, 0.5, 3) == free).all() and not (kick(f, 0.5, 4) == free).all()
    kept = kick(f, 0.5, 3, block=10)
    assert block_mass(kept, 10) == pytest.approx(block_mass(f, 10), rel=1e-12)
    assert ((kept > 0) == (f > 0)).all() and not np.allclose(kept, f)
    assert block_mass(free, 10) != pytest.approx(block_mass(f, 10), rel=1e-3)
    for masked in (True, False):
        c = coarsen_blocks(f, 10, masked)
        assert c.size == f.size and block_mass(c, 10) == pytest.approx(block_mass(f, 10), rel=1e-12)
        assert (c[40:50] == 0).all()
    assert ((coarsen_blocks(f, 10, True) > 0) == (f > 0)).all()               # masked: the support is kept
    dilated = coarsen_blocks(f, 10, False)
    assert (dilated[:40] > 0).all() and len(set(np.round(dilated[:10], 15))) == 1   # one value per block
    assert coarsen_blocks(f, 1, True) is f
    spec = sch.ChainSpec("x", "live", (sch.Stage(1e9, 5),), jitter=1.0, block=100, coarsen=4, masked=True)
    assert sch.ChainSpec.from_dict(spec.to_dict()) == spec
    older, newer = f, kick(f, 0.2, 9, block=10)
    assert along_records(newer, older, newer, 0.0) is newer
    assert along_records(older, older, newer, 1.0) == pytest.approx(newer / newer.sum(), abs=1e-15)
    beyond = along_records(newer, older, newer, 0.5)
    assert (beyond >= 0).all() and beyond == pytest.approx(
        np.maximum(newer / newer.sum() + 0.5 * (newer / newer.sum() - older / older.sum()), 0), abs=1e-15)
    directed = sch.build_plan("directed16", "live", 6000)
    assert len(directed) == 16 and sum(1 for c in directed if c.along != 0) == 10
    assert sch.ChainSpec.from_dict(directed[3].to_dict()) == directed[3]
    plan = sch.build_plan("blockkick16", "live", 6000)
    assert len(plan) == 16 and sum(1 for c in plan if c.jitter > 0 and c.block > 0) == 12
    assert sum(1 for c in plan if c.coarsen > 1 and c.masked) == 3


def test_run_chains_stops_at_deadline():
    f0 = _start(n=32, rows=1, seed_=2)
    m = fft_length(32)
    kappa = kappa_for(np, f0, m)
    w0 = initial_w(np, f0, kappa, np.zeros((1, 1)))
    ticks = iter(range(1000))
    out = run_chains(np, w0, kappa, m, [(sch.Stage(1e3, 500),)], deadline=10, clock=lambda: next(ticks))
    assert out["stopped_early"] and out["evals"] == 10
