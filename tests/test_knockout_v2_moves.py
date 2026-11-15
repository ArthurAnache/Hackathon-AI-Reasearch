"""Fast unit tests of knockout/v2 (objective, polisher, moves, episode, plans), on small vectors.

    .venv/bin/python -m pytest tests/test_knockout_v2_moves.py -q
"""
import unittest

import numpy as np

from knockout.v2 import plans
from knockout.v2.episode import EpisodeSpec, classify, rel_l1, run_episode
from knockout.v2.moves import MOVES, apply_move, runs
from knockout.v2.objective import BudgetSpent, Oracle
from knockout.v2.polish import SCHEDULES, polish


def comb(n=600, seed=0):
    """A small comb: runs of non-zero cells separated by gaps, like the real solutions."""
    rng = np.random.default_rng(seed)
    f = np.zeros(n)
    i = 5
    while i < n - 12:
        length = int(rng.integers(1, 8))
        f[i:i + length] = rng.uniform(0.2, 1.0, length)
        i += length + int(rng.integers(1, 9))
    return f


class ObjectiveTest(unittest.TestCase):
    def test_exact_score_matches_frozen_verifier(self):
        from problems.catalog import load

        f = comb()
        frozen = load("second-autocorrelation-inequality").score({"values": f.tolist()})
        self.assertAlmostEqual(Oracle(len(f)).exact(f), frozen, delta=1e-12)

    def test_gradient_matches_finite_differences(self):
        f = comb(200)
        oracle = Oracle(len(f))
        w = np.sqrt(oracle.normalize(f))
        log_lambda = oracle.update_multipliers(w * w, 30.0)
        rng = np.random.default_rng(1)
        for lam in (None, log_lambda):
            _, grad = oracle.value_and_grad(w, 30.0, lam)
            for _ in range(5):
                d = rng.standard_normal(len(w))
                h = 1e-6
                up = oracle.value_and_grad(w + h * d, 30.0, lam)[0]
                down = oracle.value_and_grad(w - h * d, 30.0, lam)[0]
                self.assertAlmostEqual((up - down) / (2 * h), float(grad @ d), delta=1e-6 * max(1.0, abs(grad @ d)))

    def test_counter_and_budget(self):
        f = comb()
        oracle = Oracle(len(f), budget=2.0)
        oracle.exact(f)                            # 0.5
        oracle.value_and_grad(np.sqrt(f), 10.0)    # 1.0
        self.assertEqual(oracle.calls, 1.5)
        with self.assertRaises(BudgetSpent):
            oracle.value_and_grad(np.sqrt(f), 10.0)
        self.assertEqual(oracle.calls, 1.5)

    def test_multipliers_are_normalized(self):
        f = comb()
        oracle = Oracle(len(f))
        log_lambda = oracle.update_multipliers(oracle.normalize(f), 50.0)
        self.assertAlmostEqual(float(np.exp(log_lambda).sum()), 1.0, places=12)


class PolishTest(unittest.TestCase):
    def test_never_below_input_and_within_budget(self):
        f = comb()
        start = Oracle(len(f)).exact(f)
        for schedule in SCHEDULES:
            best, info = polish(f, schedule, budget=80)
            self.assertGreaterEqual(Oracle(len(f)).exact(best), start - 1e-15, schedule)
            self.assertLessEqual(info["calls"], 80, schedule)
            self.assertEqual(best.shape, f.shape)
            self.assertGreaterEqual(best.min(), 0.0)

    def test_improves_a_rough_start_and_is_deterministic(self):
        f = comb()
        schedule = [(30.0, False), (100.0, False), (300.0, False)]
        best1, info1 = polish(f, schedule, budget=150)
        best2, info2 = polish(f, schedule, budget=150)
        self.assertGreater(info1["best_score"], info1["input_score"] + 1e-3)
        np.testing.assert_array_equal(best1, best2)
        self.assertEqual(info1["calls"], info2["calls"])

    def test_keeps_the_support(self):
        f = comb()
        best, _ = polish(f, [(30.0, False)], budget=60)
        self.assertTrue(np.all(best[f == 0] == 0))


class MovesTest(unittest.TestCase):
    def test_contract_and_determinism(self):
        f = comb(2000)
        for family in MOVES:
            a = apply_move(family, f, 7, 0.1)
            b = apply_move(family, f, 7, 0.1)
            np.testing.assert_array_equal(a, b, err_msg=family)
            self.assertEqual(a.shape, f.shape, family)
            self.assertGreaterEqual(a.min(), 0.0, family)
            self.assertGreater(a.sum(), 0.0, family)

    def test_each_family_does_what_it_says(self):
        f = comb(2000)
        n_runs = len(runs(f)[0])
        rng = lambda: np.random.default_rng(3)  # noqa: E731
        np.testing.assert_array_equal(MOVES["reheat"](f, rng(), 0.5), f)
        noisy = MOVES["lognormal"](f, rng(), 0.1)
        self.assertTrue(np.array_equal(noisy > 0, f > 0) and not np.array_equal(noisy, f))
        cut = MOVES["threshold"](f, rng(), 0.05)
        self.assertLess(np.count_nonzero(cut), np.count_nonzero(f))
        self.assertGreaterEqual(cut.sum(), 0.95 * f.sum() - 1e-12)
        opened = MOVES["open_boundary"](f, rng(), 0.1)
        self.assertGreater(np.count_nonzero(opened), np.count_nonzero(f))
        np.testing.assert_array_equal(opened[f > 0], f[f > 0])
        split = MOVES["split_runs"](f, rng(), 1.0)
        self.assertGreater(len(runs(split)[0]), n_runs)
        self.assertAlmostEqual(split.sum(), f.sum(), places=9)
        merged = MOVES["merge_runs"](f, rng(), 1.0)
        self.assertLess(len(runs(merged)[0]), n_runs)
        moved = MOVES["transfer_mass"](f, rng(), 0.3)
        self.assertAlmostEqual(moved.sum(), f.sum(), places=9)
        self.assertFalse(np.array_equal(moved, f))
        shifted = MOVES["shift_runs"](f, rng(), 0.05)
        np.testing.assert_array_equal(np.sort(shifted), np.sort(f))
        self.assertFalse(np.array_equal(shifted, f))
        grown = MOVES["open_gradient"](f, rng(), 0.1, beta=30.0)
        self.assertGreaterEqual(np.count_nonzero(grown), np.count_nonzero(f))
        np.testing.assert_array_equal(grown[f > 0], f[f > 0])

    def test_broken_move_is_rejected(self):
        MOVES["_bad"] = lambda f, rng, s: -f
        try:
            with self.assertRaises(ValueError):
                apply_move("_bad", comb(), 0, 0.1)
        finally:
            del MOVES["_bad"]


class EpisodeTest(unittest.TestCase):
    def test_run_episode_and_ids(self):
        f = comb()
        spec = EpisodeSpec("toy", "lognormal", 0.01, 3, "p0", 40)
        best, record = run_episode(spec, f, score_fn=lambda x: Oracle(len(x)).exact(x), references={"toy": f})
        self.assertEqual(record["experiment_id"], "toy-lognormal-s0.01-r3-p0-b40")
        self.assertEqual(record["baseline_id"], "toy-reheat-s0-r0-p0-b40")
        self.assertFalse(record["is_baseline"])
        self.assertTrue(record["move_changed"])
        self.assertLessEqual(record["calls"], 40)
        self.assertGreaterEqual(record["score"], record["moved_score"] - 1e-15)
        self.assertAlmostEqual(record["dist_to_toy"], rel_l1(best, f))
        self.assertTrue(spec.baseline().is_baseline)

    def test_classify(self):
        self.assertEqual(classify(2e-7, 0.5), "improved")
        self.assertEqual(classify(0.0, 1e-9), "undone")
        self.assertEqual(classify(0.0, 1e-3), "worse_basin")
        self.assertEqual(classify(-1e-5, None), "worse_basin")

    def test_rel_l1(self):
        a = np.array([1.0, 0.0, 1.0])
        self.assertEqual(rel_l1(a, 5 * a), 0.0)
        self.assertAlmostEqual(rel_l1(a, np.array([0.0, 1.0, 0.0])), 2.0)
        self.assertIsNone(rel_l1(a, np.ones(4)))


class PlansTest(unittest.TestCase):
    def test_baselines_and_unique_ids(self):
        for name, plan in plans.PLANS.items():
            specs = plans.with_baselines(plan())
            ids = [EpisodeSpec(**s).episode_id for s in specs]
            self.assertEqual(len(ids), len(set(ids)), name)
            for s in specs:
                self.assertIn(EpisodeSpec(**s).baseline().episode_id, ids, name)
                self.assertIn(s["family"], MOVES, name)
                self.assertIn(s["schedule"], SCHEDULES, name)


if __name__ == "__main__":
    unittest.main()
