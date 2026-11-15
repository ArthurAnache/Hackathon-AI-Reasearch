"""Fast local tests of knockout/v2/multires (no torch, no Modal): pooling, repeat-upsampling, perturbations,
level schedules, and the coarse-to-fine driver and worker run on the numpy backend (the GPU runs the same lines).

    .venv/bin/python -m pytest tests/test_knockout_v2_multires.py -q
"""
import json

import numpy as np
import pytest

from knockout.v2.gpu import schedule as sch
from knockout.v2.multires import driver, perturb
from knockout.v2.multires import plan as pl
from knockout.v2.multires import resample as rs


def _comb(n=240, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.random(n) + 0.1
    f[rng.random(n) < 0.5] = 0.0
    return f


# ---------------------------------------------------------------- changes of resolution

def test_pool_pad_and_repeat_shapes_and_mass():
    f = _comb(241)
    padded = rs.pad_to_multiple(f, 8)
    assert padded.size == 248 and np.all(padded[241:] == 0) and np.array_equal(padded[:241], f)
    pooled = rs.sum_pool(padded, 8)
    assert pooled.shape == (31,) and pooled.sum() == pytest.approx(f.sum())
    assert pooled[3] == pytest.approx(f[24:32].sum())
    up = rs.repeat_upsample(pooled, 8)
    assert up.shape == (248,) and np.array_equal(up[8:16], np.full(8, pooled[1]))
    rows = np.stack([padded, 2 * padded])
    assert rs.sum_pool(rows, 4).shape == (2, 62) and np.allclose(rs.sum_pool(rows, 4)[1], 2 * rs.sum_pool(padded, 4))
    with pytest.raises(ValueError):
        rs.sum_pool(f, 8)
    with pytest.raises(ValueError):
        rs.pad_right(f, 10)
    smooth = rs.smooth_blocks(f, 8)
    assert smooth.shape == f.shape
    assert np.allclose(rs.sum_pool(rs.pad_to_multiple(smooth, 8), 8)[:30], pooled[:30])


def test_repeat_upsampling_keeps_the_frozen_verifiers_score():
    from problems.catalog import load

    p = load(sch.SLUG)
    f = _comb(300, seed=3)
    base = float(p.score({"values": f.tolist()}))
    assert rs.closed_form_score(f) == pytest.approx(base, abs=1e-13)
    for k in (2, 3, 5):
        up = rs.repeat_upsample(f, k)
        assert float(p.score({"values": up.tolist()})) == pytest.approx(base, abs=1e-13)
        assert rs.repeat_invariance_gap(f, k) < 1e-13
    # zero-padding and rescaling are score-neutral too; pooling is not
    assert rs.closed_form_score(rs.pad_right(3.7 * f, 512)) == pytest.approx(base, abs=1e-13)
    assert abs(rs.closed_form_score(rs.sum_pool(f, 2)) - base) > 1e-6


def test_loss_terms_add_up_to_one_minus_the_score():
    for seed, n in ((0, 57), (1, 300), (2, 1001)):
        f = _comb(n, seed=seed)
        terms = rs.loss_terms(f)
        assert terms["plateau"] > 0 and terms["roughness"] > 0
        assert terms["plateau"] + terms["roughness"] == pytest.approx(1.0 - rs.closed_form_score(f), abs=1e-13)
    box = rs.loss_terms(np.ones(64))             # a box: triangle autoconvolution, 1 - C = 1/3 (+ O(1/n^2))
    assert box["plateau"] + box["roughness"] == pytest.approx(1.0 / 3.0, abs=1e-3)


def test_block_profile():
    f = _comb(400, seed=1)
    assert rs.block_masses(f, 50).sum() == pytest.approx(1.0)
    assert rs.profile_l1(f, 5.0 * f, 50) == pytest.approx(0.0, abs=1e-15)
    assert rs.profile_l1(f, rs.smooth_blocks(f, 50), 50) == pytest.approx(0.0, abs=1e-12)
    g = np.zeros(400)
    g[:50] = 1.0
    h = np.zeros(399)
    h[350:] = 1.0
    assert rs.profile_l1(g, h, 50) == pytest.approx(2.0)


# ---------------------------------------------------------------- perturbations

@pytest.mark.parametrize("spec", [
    {"kind": "shuffle", "block": 16, "frac": 1.0},
    {"kind": "shuffle", "block": 16, "frac": 0.25},
    {"kind": "shift", "block": 16, "max_shift": 3, "frac": 1.0},
    {"kind": "resample", "block": 16, "sigma": 1.0, "frac": 1.0},
    {"kind": "redraw", "block": 16, "frac": 1.0},
    {"kind": "blend", "block": 16, "t": 0.3},
])
def test_envelope_preserving_perturbations(spec):
    f = _comb(250, seed=2)                       # 250 is not a multiple of 16: the last block is partial
    g = perturb.apply(f, spec, seed=5)
    assert g.shape == f.shape and np.all(g >= 0)
    assert not np.array_equal(g, f)
    assert np.array_equal(perturb.apply(f, spec, seed=5), g)                    # same seed, same draw
    if spec["kind"] != "blend":
        assert not np.array_equal(perturb.apply(f, spec, seed=6), g)
    assert rs.profile_l1(f, g, 16) == pytest.approx(0.0, abs=1e-12)             # block masses are kept
    if spec["kind"] in ("shuffle", "shift"):
        blocks_f = rs.pad_to_multiple(f, 16).reshape(-1, 16)
        blocks_g = rs.pad_to_multiple(g, 16).reshape(-1, 16)
        assert np.allclose(np.sort(blocks_f, 1), np.sort(blocks_g, 1))          # same cells, moved inside blocks
        changed = (blocks_f != blocks_g).any(1).mean()
        assert changed <= spec["frac"] + 1e-9
    if spec["kind"] == "resample":
        assert np.array_equal(g > 0, f > 0)                                     # support kept


def test_shift_rolls_blocks_and_other_kinds():
    f = np.arange(1.0, 9.0)
    g = perturb.apply(f, {"kind": "shift", "block": 4, "max_shift": 1, "frac": 1.0}, seed=0)
    for a, b in zip(f.reshape(2, 4), g.reshape(2, 4)):
        assert any(np.array_equal(np.roll(a, s), b) for s in (-1, 1))
    f = _comb(100)
    ln = perturb.apply(f, {"kind": "lognormal", "sigma": 0.5}, seed=1)
    assert np.array_equal(ln > 0, f > 0) and not np.allclose(ln, f)
    uni = perturb.apply(f, {"kind": "uniform", "scale": 1e-8}, seed=1)
    assert np.all(uni > 0) and np.abs(uni - f).max() <= 1e-8 * f.max()
    assert np.array_equal(perturb.apply_all(f, (), 3), f)
    sg = perturb.apply(f, {"kind": "sqrt_gauss", "sigma": 0.05}, seed=1)
    assert np.array_equal(sg == 0, f == 0) and not np.allclose(sg, f)          # zero cells stay zero
    assert np.abs(np.sqrt(sg) - np.sqrt(f)).max() < 0.05 * 6 * np.sqrt(f[f > 0].mean())
    other = {"ref": np.array([0.0, 4.0, 0.0])}
    base = np.array([1.0, 1.0, 0.0, 0.0])                                      # the reference is zero-padded
    assert np.allclose(perturb.apply(base, {"kind": "mix", "other": "ref", "t": 0.5}, 0, refs=other),
                       [0.25, 0.75, 0.0, 0.0])
    assert np.allclose(perturb.apply(base, {"kind": "mix", "other": "ref", "t": -0.5}, 0, refs=other),
                       [0.75, 0.25, 0.0, 0.0])
    assert np.allclose(perturb.apply(base, {"kind": "mix", "other": "ref", "t": -2.0}, 0, refs=other),
                       [1.5, 0.0, 0.0, 0.0])                                   # clipped at zero
    with pytest.raises(ValueError):
        perturb.apply(base, {"kind": "mix", "other": "ref", "t": 0.5}, 0)
    relink = pl.build("relink", "live", 2000)[0]
    assert len(relink.rows) == 16 and relink.rows[1].pre == ({"kind": "mix", "other": "frozen", "t": 0.5},)
    g = np.array([0.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(perturb.apply(g, {"kind": "dilate", "radius": 1, "t": 0.5}, 0),
                       [0.0, 0.0, 1.0, 2.0, 4.0, 2.0, 0.0, 0.0, 0.5, 1.0])
    assert np.allclose(perturb.apply(g, {"kind": "dilate", "radius": 2, "t": 0.1}, 0),
                       [0.0, 0.2, 0.4, 2.0, 4.0, 0.4, 0.4, 0.1, 0.1, 1.0])
    two =perturb.apply_all(f, ({"kind": "lognormal", "sigma": 0.1}, {"kind": "blend", "block": 4, "t": 1.0}), 3)
    assert np.allclose(two, rs.smooth_blocks(perturb.apply(f, {"kind": "lognormal", "sigma": 0.1}, 3), 4))
    assert perturb.describe(({"kind": "blend", "block": 4, "t": 0.5},)) == "blend(block=4, t=0.5)"
    with pytest.raises(ValueError):
        perturb.apply(f, {"kind": "nope", "block": 4}, 0)


# ---------------------------------------------------------------- schedules

def test_level_sizes_stages_and_checks():
    rows = (pl.RowSpec("a", beta0=1e5), pl.RowSpec("ctl", beta0=1e9, beta1=1e9))
    b = pl.BatchSpec("v8", "E2", "frozen", 8, (2, 2, 2), (400, 400, 400, 400), rows)
    pl.check_batch(b)
    assert pl.level_sizes(b) == [250_000, 500_000, 1_000_000, 2_000_000]
    mid = pl.stages(b, rows[0], 1)
    assert mid[0].beta == 1e5 and mid[-1].beta == 1e9 and sch.total_evals(mid) == 400
    last = pl.stages(b, rows[0], 3)
    assert last[-1].beta == 1e11 and sch.total_evals(last) == 400
    assert sum(s.evals for s in last if s.beta > 1e9) == 100                    # the polish share
    ctl = pl.stages(b, rows[1], 3)
    assert ctl[0].beta == 1e9 and ctl[-1].beta == 1e11 and sch.total_evals(ctl) == 400
    assert pl.BatchSpec.from_dict(json.loads(json.dumps(b.to_dict()))) == b
    for bad in (pl.BatchSpec("x", "E2", "frozen", 8, (2, 2, 2), (400,) * 3, rows),            # levels / budgets
                pl.BatchSpec("x", "E2", "frozen", 8, (3,), (400,) * 2, rows),                 # 3 does not divide 8
                pl.BatchSpec("x", "E2", "frozen", 7, (), (400,), rows),                       # 7 does not divide 2M
                pl.BatchSpec("x", "E2", "frozen", 1, (2,), (400,) * 2, rows),                 # above the cap
                pl.BatchSpec("x", "E2", "frozen", 2, (2,), (5, 400), rows),                   # fewer evals than stages
                pl.BatchSpec("x", "E2", "frozen", 2, (2,), (400,) * 2, (rows[0], rows[0]))):  # duplicate names
        with pytest.raises(ValueError):
            pl.check_batch(bad)


def test_plans_build_and_estimate():
    for name in pl.PLANS:
        batches = pl.build(name, "frozen", 3000)
        assert batches and all(len(b.rows) <= 16 for b in batches)
        assert all(pl.level_sizes(b)[-1] <= pl.CAP for b in batches)
        est = pl.estimate(batches)
        assert est["usd"] > 0 and est["wall_s"] > est["loop_s"]
    e2 = pl.build("e2", "live", 2000, (8,))
    assert [b.name for b in e2] == ["e2-v8"] and e2[0].ups == (2, 2, 2) and len(e2[0].rows) == 16
    mixed = pl.build("e1:2:4,e2:64:4", "frozen", 2000)
    assert [b.name for b in mixed] == ["e1-p2", "e1-p4", "e2-v64", "e2-v4"] and len(mixed[2].level_evals) == 7
    assert len(pl.build("e3", "frozen", 2000)[0].rows) == 16
    both = pl.build("edge@frozen,edge:1:2@live", "frozen", 3000)
    assert [(b.name, b.start) for b in both] == [("e3b-edge-frozen", "frozen"), ("e3b-edge-live", "live")]
    assert both[1].rows[1].pre == ({"kind": "dilate", "radius": 1, "t": 1e-6},)
    deep = pl.build("vdeep", "live", 3000)[0]
    assert deep.name == "e2b-v2-live" and deep.level_evals == (3000, 3000) and len(deep.rows) == 16
    assert pl.BatchSpec.from_dict(json.loads(json.dumps(deep.to_dict()))) == deep
    row = next(r for r in deep.rows if r.name == "phase-c3-f5")
    assert row.pre == ({"kind": "roll", "by": 1},)
    assert pl.stages(deep, row, 0)[0].beta == 1e3 and pl.stages(deep, row, 1)[0].beta == 1e5
    assert pl.stages(deep, row, 1)[-1].beta == 1e11 and sch.total_evals(pl.stages(deep, row, 1)) == 3000
    kick = next(r for r in deep.rows if r.name == "advised-kick-s2")
    assert kick.coarse == ({"kind": "sqrt_gauss", "sigma": 0.05},) and kick.post[0]["sigma"] == 0.03
    assert [(s.beta, s.evals) for s in pl.stages(deep, kick, 1)] == [(1e7, 300), (1e8, 600), (1e9, 900), (1e10, 1200)]
    assert pl.stages(deep, kick, 0)[0].beta == 1e3 and pl.stages(deep, kick, 0)[-1].beta == 1e9
    padded = rs.pad_right(_comb(95), 96)
    assert rs.closed_form_score(perturb.apply(padded, {"kind": "roll", "by": 1}, 0)) == pytest.approx(
        rs.closed_form_score(padded), abs=1e-13)
    # 16 rows of 2,000,000 cells for 10,000 evaluations: the E0 rate of knockout/v2/gpu (0.0334 s per evaluation)
    one = pl.build("e2-blend", "frozen", 10_000)
    assert pl.estimate(one, overhead_s=0, score_s_per_batch=0)["loop_s"] == pytest.approx(336, rel=0.02)
    with pytest.raises(ValueError):
        pl.build("nope", "frozen", 100)


# ---------------------------------------------------------------- driver and worker on the numpy backend

def _tiny_batch(evals=60):
    rows = (pl.RowSpec("plain", beta0=1e3, beta1=1e6),
            pl.RowSpec("noisy", seed=4, post=({"kind": "lognormal", "sigma": 0.2},), beta0=1e3, beta1=1e6),
            pl.RowSpec("open", beta0=1e3, beta1=1e6, floor=1e-3, up_floor=1e-3))
    return pl.BatchSpec("tiny", "smoke", "frozen", 4, (2, 2), (evals,) * 3, rows, pad_to=96)


def test_ladder_runs_coarse_to_fine_on_numpy():
    batch = _tiny_batch()
    start = _comb(95, seed=7)                    # padded to 96 by the driver
    f0 = driver.level_start(batch, start)
    assert f0.shape == (3, 24) and np.allclose(f0[0], rs.sum_pool(rs.pad_right(start, 96), 4))
    seen = []
    out = driver.run_ladder(np, batch, f0, to_xp=lambda a: np.array(a, dtype=np.float64), to_np=np.asarray,
                            on_level_end=lambda lv, f, info: seen.append((lv, f.copy(), info)))
    assert [s[0] for s in seen] == [0, 1, 2] and [s[1].shape for s in seen] == [(3, 24), (3, 48), (3, 96)]
    assert not out["stopped_early"] and np.all(out["f"] >= 0)
    for lv, f, info in seen:
        assert info["evals"] == 60
        for i in range(3):                       # the reported exact score is the stored solution's score
            assert rs.closed_form_score(f[i]) == pytest.approx(info["exact"][i], abs=1e-12)
    assert seen[1][2]["repeat_gap"] <= driver.REPEAT_TOL and seen[2][2]["repeat_gap"] <= driver.REPEAT_TOL
    # without noise the next level starts exactly where the coarse one ended, and never ends below it
    plain = [s[2]["exact"][0] for s in seen]
    assert seen[1][2]["init_exact"][0] == pytest.approx(plain[0], abs=1e-12)
    assert plain[0] >= seen[0][2]["init_exact"][0] and plain[1] >= plain[0] - 1e-12 and plain[2] >= plain[1] - 1e-12
    assert np.count_nonzero(seen[2][1][0]) < 96 and np.count_nonzero(seen[2][1][2]) == 96    # only "open" opens
    assert seen[1][2]["init_exact"][1] != pytest.approx(seen[0][2]["exact"][1], abs=1e-9)    # the noisy row moved


def test_gradient_guided_growth_cycles_on_numpy():
    from knockout.v2.gpu.objective import evaluate, fft_length, kappa_for

    # the gain of driver.support_gain is the derivative of log C_beta with respect to f, zero cells included
    f = _comb(40, seed=5)[None, :]
    m = fft_length(40)
    beta = np.full((1, 1), 50.0)
    gain = driver.support_gain(np, f, beta, m)[0]

    def log_c(v):
        return -evaluate(np, np.sqrt(v), beta, kappa_for(np, f, m), m)[0][0]

    for i in (int(np.flatnonzero(f[0] == 0)[0]), int(np.flatnonzero(f[0] > 0)[0])):
        up, down = f.copy(), f.copy()
        up[0, i] += 1e-6
        down[0, i] = max(0.0, down[0, i] - 1e-6)
        step = up[0, i] - down[0, i]
        assert (log_c(up) - log_c(down)) / step == pytest.approx(gain[i], rel=1e-4, abs=1e-7)
    # "grow" opens exactly the zero cells next to the support whose gain is positive
    g = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    gains = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 2.0])
    assert np.allclose(perturb.apply(g, {"kind": "grow", "radius": 1, "t": 0.1}, 0, gains),
                       [0.0, 0.3, 3.0, 0.0, 0.0, 0.1, 1.0, 0.1])
    assert np.allclose(perturb.apply(g, {"kind": "grow", "radius": 1, "t": 0.1, "top": 0.34}, 0, gains),
                       [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.1])
    with pytest.raises(ValueError):
        perturb.apply(g, {"kind": "grow", "radius": 1, "t": 0.1}, 0)
    # cycles at the same resolution (upsampling factor 1) through the ladder
    plan = pl.build("grow", "frozen", 40)[0]
    assert plan.ups == (1, 1, 1) and len(plan.rows) == 16 and pl.level_sizes(plan) == [2_000_000] * 4
    rows = (pl.RowSpec("control", beta0=1e4, beta1=1e4),
            pl.RowSpec("grow", post=({"kind": "grow", "radius": 1, "t": 0.01},), beta0=1e4, beta1=1e4))
    batch = pl.BatchSpec("cycles", "E3c", "frozen", 1, (1, 1), (40, 40, 40), rows, pad_to=96)
    f0 = driver.level_start(batch, _comb(95, seed=7))
    seen = []
    driver.run_ladder(np, batch, f0, to_xp=lambda a: np.array(a, dtype=np.float64), to_np=np.asarray,
                      on_level_end=lambda lv, fr, info: seen.append((fr.copy(), info)))
    support = [[int(np.count_nonzero(fr[i])) for fr, _ in seen] for i in range(2)]
    assert support[0][0] == support[0][1] == support[0][2]             # the control never opens a cell
    assert support[1][2] >= support[1][1] >= support[1][0] and support[1][2] > support[1][0]
    assert all(info["n"] == 96 for _, info in seen)


def test_lift_and_project_on_numpy():
    plan = pl.build("lift", "frozen", 400)[0]
    assert plan.lift == 2 and plan.ups == (-2,) and pl.level_sizes(plan) == [4_000_000, 2_000_000]
    assert plan.level_evals == (400, 600) and len(plan.rows) == 8
    assert pl.BatchSpec.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan
    rows = (pl.RowSpec("a", beta0=1e4, beta1=1e5, beta0_up=1e3),)
    batch = pl.BatchSpec("lift", "E6", "frozen", 1, (-2,), (50, 50), rows, pad_to=96, lift=2)
    start = _comb(95, seed=7)
    f0 = driver.level_start(batch, start)
    assert f0.shape == (1, 192) and rs.closed_form_score(f0[0]) == pytest.approx(rs.closed_form_score(start), abs=1e-13)
    seen = []
    driver.run_ladder(np, batch, f0, to_xp=lambda a: np.array(a, dtype=np.float64), to_np=np.asarray,
                      on_level_end=lambda lv, fr, info: seen.append((fr.copy(), info)))
    assert [fr.shape for fr, _ in seen] == [(1, 192), (1, 96)]
    assert seen[0][1]["exact"][0] >= rs.closed_form_score(start) - 1e-12      # the lifted level starts at the start's score
    assert seen[1][1]["init_exact"][0] == pytest.approx(rs.closed_form_score(rs.sum_pool(seen[0][0][0], 2)), abs=1e-12)
    with pytest.raises(ValueError):
        pl.check_batch(pl.BatchSpec("x", "E6", "frozen", 1, (-2,), (50, 50), rows, pad_to=95, lift=1))


def test_ladder_stops_at_the_deadline():
    batch = _tiny_batch()
    f0 = driver.level_start(batch, _comb(95, seed=7))
    ticks = iter(range(10_000))
    out = driver.run_ladder(np, batch, f0, to_xp=lambda a: np.array(a, dtype=np.float64), to_np=np.asarray,
                            deadline=40, clock=lambda: next(ticks))
    assert out["stopped_early"] and len(out["levels"]) == 1 and out["levels"][0]["evals"] < 60


def test_worker_logs_every_level_with_the_frozen_verifier(tmp_path):
    from knockout.v2.multires import worker

    start = _comb(95, seed=7)
    (tmp_path / "starts").mkdir()
    np.save(tmp_path / "starts" / "frozen.npy", start)
    np.save(tmp_path / "starts" / "live.npy", start[::-1].copy())
    (tmp_path / "starts" / "meta.json").write_text(json.dumps({"live": {"score": 0.9}, "frozen": {"score": 0.8}}))
    batch = _tiny_batch(evals=40)
    batch = pl.BatchSpec(batch.name, batch.experiment, batch.start, batch.pool, batch.ups, batch.level_evals,
                         batch.rows[:2], pad_to=batch.pad_to)
    spec = {"run_id": "r1", "source_commit": "abc", "budget_s": 600.0, "usd_per_s": 0.001, "score_workers": 2,
            "batches": [batch.to_dict()]}
    env = {"device": "cpu", "gpu_type": None, "dtype": "float64", "torch_version": None, "cuda_version": None}
    out = worker.run_job(tmp_path, spec, xp=np, to_xp=lambda a: np.array(a, dtype=np.float64), to_np=np.asarray,
                         env=env)
    assert out["finished"] and out["records"] == 6 and not out["skipped"]
    recs = [json.loads(ln) for ln in (tmp_path / "r1" / "levels.jsonl").read_text().splitlines()]
    assert len(recs) == 6
    for rec in recs:
        assert all(k in rec for k in sch.SHARED_FIELDS + sch.GPU_FIELDS)
        sol = np.load(tmp_path / rec["solution_path"])
        assert rec["valid"] and rec["score"] == pytest.approx(rs.closed_form_score(sol), abs=1e-12)
        assert rec["score"] == pytest.approx(rec["gpu_exact"], abs=1e-12) and rec["n"] == sol.size
        assert rec["final"] == (rec["level"] == 2)
        assert ("profile_l1_1000_vs_live" in rec) == (rec["level"] == 2)
    last = [r for r in recs if r["row"] == "plain" and r["level"] == 2][0]
    assert last["parent"] == "r1/tiny.plain/L1" and last["experiment_id"] == "r1/tiny.plain/L2"
    rows = out["batches"][0]["rows"]
    assert rows[0]["last_is_final"] and rows[0]["last_score"] == last["score"]
