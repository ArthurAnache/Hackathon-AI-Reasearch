"""Process 8: the research feedback framework (research/): results in, next experiment out.

Uses synthetic experiment histories so each piece can be checked exactly, plus one
closed-loop test where the agent optimises a known objective with no optimizer runs.
"""
import math

import numpy as np
import pytest

import interfaces as I
from conftest import REPO
from research import (CombineWins, CoordinateStep, EvidenceHints, FeedbackAgent, INIT_KEY, Knowledge, Refine,
                      Space, Surrogate)

P = pytest.mark.process("8. Research feedback framework")
OPT = I.FileOptimizer(REPO / "optimizers" / "1d_tuned.py")
PROB = I.get_problem("1d")          # minimise
SPACE = Space.from_optimizer(OPT)
RNG = lambda: np.random.default_rng(0)


def rec(i, score, status="keep", strategy="manual", parent=None, **cfg):
    return {"experiment_id": f"exp_{i:04d}", "problem": "1d", "optimizer_hash": OPT.version(), "budget_s": 1.0,
            "status": status, "score": score, "config": {**OPT.default_config(), **cfg},
            "rationale": f"strategy={strategy}; parent={parent or 'none'}", "certificate_path": f"1d/cert{i:04d}"}


def kn(*records):
    return Knowledge(PROB, SPACE, list(records))


# ---------------------------------------------------------------- space
@P
def test_space_from_declared_and_default_ranges(record):
    """Declared SEARCH_SPACE ranges are used; undeclared numeric knobs get [v/8, 8v] (log); INIT is not a knob."""
    base = Space.from_optimizer(I.FileOptimizer(REPO / "optimizers" / "1a_baseline.py"))
    record(tuned_knobs={k: (v.low, v.high, v.integer) for k, v in SPACE.knobs.items()},
           baseline_knobs={k: (v.low, v.high, v.integer) for k, v in base.knobs.items()})
    assert SPACE.knobs["N_CELLS"].low == 100 and SPACE.knobs["N_CELLS"].integer and INIT_KEY not in SPACE.knobs
    assert base.knobs["N_CELLS"].low == 6 and base.knobs["N_CELLS"].high == 400


@P
def test_space_roundtrip_and_key(record):
    cfg = OPT.default_config()
    back = SPACE.from_unit(SPACE.to_unit(cfg), cfg)
    record(original={k: cfg[k] for k in SPACE.names}, roundtrip={k: back[k] for k in SPACE.names})
    assert all(math.isclose(back[k], cfg[k], rel_tol=1e-5) for k in SPACE.names)
    assert SPACE.key(cfg) != SPACE.key({**cfg, INIT_KEY: "1d/cert0000"})


# ---------------------------------------------------------------- knowledge
@P
def test_knowledge_attributes_effects_to_single_knob_changes(record):
    """A child that differs from its parent in one knob is a controlled comparison for that knob."""
    k = kn(rec(0, 1.50), rec(1, 1.48, "keep", "coordinate", "exp_0000", LR=0.02),
           rec(2, 1.49, "discard", "coordinate", "exp_0001", LR=0.02, BETA_END=60000.0))
    eff = k.knob_effects()
    record(LR=[(round(e.du, 3), round(e.gain, 3)) for e in eff["LR"]],
           BETA_END=[(round(e.du, 3), round(e.gain, 3)) for e in eff["BETA_END"]], best=k.best.id)
    assert eff["LR"][0].gain == pytest.approx(0.02) and eff["LR"][0].du > 0   # minimise: 1.50 -> 1.48 is +0.02
    assert eff["BETA_END"][0].gain == pytest.approx(-0.01)
    assert k.best.id == "exp_0001"


@P
def test_trust_region_and_failures(record):
    k = kn(rec(0, 1.5), rec(1, 1.4, "keep", "explore", "exp_0000"), rec(2, 1.45, "discard", "explore", "exp_0001"),
           rec(3, None, "crash", "explore", "exp_0001", N_CELLS=3000))
    r_after = [round(kn(*[rec(0, 1.5), rec(1, 1.4, "keep", "x", "exp_0000")][:n]).trust_radius(), 3) for n in (1, 2)]
    record(radius_after_keep=r_after, radius_now=k.trust_radius(),
           near_failure=k.near_failure({**OPT.default_config(), "N_CELLS": 2950}))
    assert r_after[1] > r_after[0]
    assert k.near_failure({**OPT.default_config(), "N_CELLS": 2950})
    assert not k.near_failure(OPT.default_config())
    assert k.strategy_stats()["explore"] == (3, 1)


# ---------------------------------------------------------------- strategies
@P
def test_coordinate_continues_good_and_reverses_bad_moves(record):
    good = kn(rec(0, 1.50), rec(1, 1.48, "keep", "coordinate", "exp_0000", LR=0.02))
    bad = kn(rec(0, 1.50), rec(1, 1.55, "discard", "coordinate", "exp_0000", LR=0.02))
    lr_good = [c for c in CoordinateStep().propose(good, RNG()) if c.config["LR"] != 0.02]
    lr_bad = [c for c in CoordinateStep().propose(bad, RNG()) if c.config["LR"] != 0.01]
    up = next(c for c in lr_good if "LR" in c.why.split(":")[0])
    down = next(c for c in lr_bad if "LR" in c.why.split(":")[0])
    record(after_improvement=up.why, after_regression=down.why)
    assert up.config["LR"] > 0.02 and down.config["LR"] < 0.01


@P
def test_surrogate_moves_towards_optimum(record):
    """On a smooth synthetic landscape the fitted model proposes a point closer to the optimum."""
    rng = np.random.default_rng(1)
    target = {"LR": 0.03, "BETA_END": 100000.0}
    f = lambda c: 1.5 + (math.log(c["LR"] / target["LR"])) ** 2 * 0.01 + (math.log(c["BETA_END"] / target["BETA_END"])) ** 2 * 0.01
    recs = [rec(0, f(OPT.default_config()))]
    for i in range(1, 16):
        cfg = {"LR": float(10 ** rng.uniform(-3, -1)), "BETA_END": float(10 ** rng.uniform(3, 6))}
        recs.append(rec(i, f({**OPT.default_config(), **cfg}), "discard", "explore", "exp_0000", **cfg))
    k = kn(*recs)
    (c,) = Surrogate().propose(k, RNG())
    record(best_before=k.best.score, proposal={x: c.config[x] for x in target}, value_at_proposal=f(c.config),
           predicted_gain=c.gain)
    assert f(c.config) < k.best.score and c.gain > 0


@P
def test_combine_and_evidence_and_refine(record):
    k = kn(rec(0, 1.50), rec(1, 1.48, "discard", "coordinate", "exp_0000", LR=0.02),
           rec(2, 1.49, "discard", "coordinate", "exp_0000", BETA_END=60000.0))
    (comb,) = CombineWins().propose(k, RNG())
    hints = EvidenceHints()
    hints.evidence = [I.Evidence(source="exa", content='paper suggests {"config": {"LR": 0.004}}', citation="arXiv:x")]
    (hint,) = hints.propose(k, RNG())
    (ref,) = Refine().propose(k, RNG())
    stale = kn(rec(0, 1.5), rec(1, 1.5, "discard", "refine", "exp_0000"), rec(2, 1.5, "discard", "refine", "exp_0000"))
    record(combined={x: comb.config[x] for x in ("LR", "BETA_END")}, hint=hint.config["LR"], refine_init=ref.config[INIT_KEY],
           refine_after_two_flat=Refine().propose(stale, RNG()))
    assert comb.config["LR"] == 0.02 and comb.config["BETA_END"] == 60000.0
    assert hint.config["LR"] == 0.004 and "arXiv:x" in hint.why
    assert ref.config[INIT_KEY] == "1d/cert0000"
    assert Refine().propose(stale, RNG()) == []


# ---------------------------------------------------------------- agent
@P
def test_agent_closed_loop_on_known_objective(record):
    """The agent alone (no optimizer runs) improves a known objective and never repeats a config."""
    f = lambda c: 1.46 + 0.01 * math.log(c["LR"] / 0.002) ** 2 + 0.005 * math.log(c["BETA_END"] / 2e5) ** 2 \
        + 0.002 * math.log(c["N_CELLS"] / 1500) ** 2
    agent = FeedbackAgent(PROB, OPT, warm_start=False, patience=100)
    history, keys = [], set()
    for i in range(40):
        exp = agent.propose(history, [])
        cfg = {**OPT.default_config(), **exp.config}
        key = SPACE.key(cfg)
        assert key not in keys, f"repeated config at step {i}"
        keys.add(key)
        score = f(cfg)
        best = min((r["score"] for r in history if r["status"] == "keep"), default=math.inf)
        status = "keep" if score < best else "discard"
        history.append({"experiment_id": f"exp_{i:04d}", "problem": "1d", "optimizer_hash": OPT.version(),
                        "budget_s": 1.0, "status": status, "score": score, "config": cfg,
                        "rationale": exp.rationale, "certificate_path": None})
        agent.review(history[-1])
    traj = [min(r["score"] for r in history[:n] if r["status"] == "keep") for n in (1, 10, 20, 40)]
    used = sorted({r["rationale"].split(";")[0] for r in history})
    record(best_after_1_10_20_40=traj, optimum=1.46, strategies_used=used)
    assert traj[-1] < traj[0] and traj[-1] - 1.46 < 0.5 * (traj[0] - 1.46)
    assert len(used) >= 3


@P
def test_agent_uses_warm_start_once_something_is_kept(record):
    agent = FeedbackAgent(PROB, OPT)            # 1d_tuned supports INIT -> warm start on by default
    first = agent.propose([], [])
    second = agent.propose([rec(0, 1.47)], [])
    record(warm_start=agent.warm_start, first=first.hypothesis, second=second.hypothesis,
           second_init=second.config.get(INIT_KEY))
    assert agent.warm_start and first.config == {}
    assert second.config[INIT_KEY] == "1d/cert0000" and "strategy=refine" in second.rationale
