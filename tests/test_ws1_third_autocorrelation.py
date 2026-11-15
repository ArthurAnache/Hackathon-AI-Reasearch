"""WS1: tests of third-autocorrelation-inequality, frozen as a PROPOSAL (Arthur has not accepted it yet).

    .venv/bin/python -m unittest discover -s tests -p "test_ws1_*.py"

The freeze rule (hashes, manifest) and the certificate checks (displayed score reproduced exactly, independent
verifier in agreement) are in tests/test_ws1_frozen_problems.py, which covers this problem too. Here:
  - the problem is MINIMIZE: gain, target and beats_record follow that direction; a candidate counts only when it is
    at least delta below the record, and a tie never counts;
  - invalid candidates are rejected by verify() and by the independent verifier;
  - the exact independent verifier equals a direct Fraction computation on small signed inputs, and agrees with the
    pinned verifier on a stored candidate.

No test here says anything about a record: the stored candidates are candidates, arbitrated by
selection/verify_claim.py and scored by nobody else.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
import time
import unittest
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from selection.verify_claim import PROPOSED_SLUGS, load_problem  # noqa: E402

SLUG = "third-autocorrelation-inequality"
DISPLAYED = 1.449109789331911   # live #1 (Atris, solution 2617) at freeze time, 2026-11-15T05:49:44Z
CANDIDATES = REPO_ROOT / "selection" / "scout_candidates" / SLUG


def direct_score(values: list[float]) -> Fraction:
    """2n * |max_k sum_i f_i f_(k-i)| / (sum f)^2 in Fractions: the definition, with no shortcut."""
    f = [Fraction(v) for v in values]
    n = len(f)
    conv = [sum(f[i] * f[k - i] for i in range(max(0, k - n + 1), min(k, n - 1) + 1)) for k in range(2 * n - 1)]
    return 2 * n * abs(max(conv)) / sum(f) ** 2


class Definition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = load_problem(SLUG)

    def test_frozen_definition(self):
        p = self.p
        self.assertIn(SLUG, PROPOSED_SLUGS)
        self.assertEqual((p.SLUG, p.PROBLEM_ID, p.DIRECTION, p.DELTA), (SLUG, 4, "minimize", 1e-5))
        self.assertEqual(p.ROLE, "proposed_main")
        self.assertEqual(p.LIMITS["max_values"], 100_000)
        meta = json.loads((REPO_ROOT / "problems" / SLUG / "meta.json").read_text())
        self.assertIn("Arthur has not yet accepted", meta["proposal"])
        self.assertEqual(meta["record"]["value"], DISPLAYED)

    def test_certificate_is_the_live_top1_at_freeze(self):
        # Its score (exactly the displayed one) is checked in test_ws1_frozen_problems.Certificates.
        c = self.p.certificate()
        self.assertEqual(len(c["values"]), 65_536)
        self.assertEqual(self.p.submission_path(c), "inline")


class Minimize(unittest.TestCase):
    """A lower score is better. The server accepts a new #1 only if it is at least delta below the current one."""

    @classmethod
    def setUpClass(cls):
        cls.p = load_problem(SLUG)

    def test_gain_is_positive_when_the_score_is_lower(self):
        self.assertEqual(self.p.gain(1.25, 1.5), 0.25)
        self.assertEqual(self.p.gain(1.5, 1.25), -0.25)

    def test_beats_record_only_at_least_delta_below(self):
        p, d = self.p, self.p.DELTA
        self.assertTrue(p.beats_record(DISPLAYED - 1.5 * d, best=DISPLAYED))
        self.assertTrue(p.beats_record(DISPLAYED - 28 * d, best=DISPLAYED))
        self.assertFalse(p.beats_record(DISPLAYED - 0.5 * d, best=DISPLAYED))    # lower, but by less than delta
        self.assertFalse(p.beats_record(DISPLAYED + 1.5 * d, best=DISPLAYED))    # higher is worse
        self.assertFalse(p.beats_record(DISPLAYED + 28 * d, best=DISPLAYED))

    def test_a_tie_never_beats_the_record(self):
        self.assertFalse(self.p.beats_record(DISPLAYED, best=DISPLAYED))
        self.assertFalse(self.p.beats_record(math.nextafter(DISPLAYED, 0.0), best=DISPLAYED))

    def test_boundary_is_delta_inclusive(self):
        # 2e-5 - 1e-5 is exactly 1e-5 in binary64, so the gain equals delta: accepted, as on the server (>=).
        # One ulp less of gain is refused.
        p = self.p
        self.assertEqual(2e-5 - 1e-5, p.DELTA)
        self.assertTrue(p.beats_record(1e-5, best=2e-5))
        self.assertFalse(p.beats_record(math.nextafter(1e-5, 1.0), best=2e-5))

    def test_gain_must_exceed_the_numerical_tolerance(self):
        d = self.p.DELTA
        self.assertFalse(self.p.beats_record(DISPLAYED - 1.5 * d, best=DISPLAYED, tolerance_abs=2 * d))

    def test_target_is_the_record_minus_delta(self):
        frozen = sys.modules["ws1_problem_third_autocorrelation_inequality"]._P
        saved = frozen._best_cache
        try:
            frozen._best_cache = {"value": DISPLAYED, "agent": "test", "live": False, "source": "test",
                                  "read_at": "test", "_t": time.time()}
            self.assertEqual(self.p.target(), DISPLAYED - self.p.DELTA)
            self.assertLess(self.p.target(), DISPLAYED)
        finally:
            frozen._best_cache = saved


class InvalidCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = load_problem(SLUG)

    def test_schema(self):
        p = self.p
        for bad in ([1.0, 2.0], {"vals": [1.0]}, {"values": "1,2"}, {"values": []}, {"values": [1.0, "2"]},
                    {"values": [1.0, None]}, {"values": [1.0, True]}, {"values": [1.0, float("nan")]},
                    {"values": [1.0, float("inf")]}, {"values": [1.0, 10 ** 400]}):
            with self.subTest(bad=str(bad)[:40]):
                self.assertFalse(p.verify(bad))
                self.assertRaises(p.InvalidSolution, p.score, bad)
                self.assertRaises(p.InvalidSolution, p.independent_score, bad)

    def test_value_count_limit(self):
        p = self.p
        p.check_schema({"values": [1.0] * 100_000})               # the server's maximum is accepted
        self.assertIn("server schema", p.why_invalid({"values": [1.0] * 100_001}))
        self.assertRaises(p.InvalidSolution, p.independent_score, {"values": [1.0] * 100_001})

    def test_integral_close_to_zero_is_rejected_by_both_verifiers(self):
        p = self.p
        for bad in ({"values": [1.0, -1.0]},                       # zero integral
                    {"values": [0.0, 0.0, 0.0]},
                    {"values": [3.0, -2.0, -1.0, 1e-5]},           # (sum * dx)^2 = 1.6e-12 < 1e-9
                    {"values": [1e-5] * 4}):                       # (sum * dx)^2 = 2.5e-11 < 1e-9
            with self.subTest(bad=bad):
                self.assertFalse(p.verify(bad))
                self.assertIn("pinned verifier rejected", p.why_invalid(bad))
                with self.assertRaises(ValueError):
                    p.independent_score(bad)


class IndependentVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = load_problem(SLUG)

    def test_equals_the_direct_fraction_computation(self):
        p = self.p
        cases = ([1.0], [1.0, 1.0, 1.0], [1.0, -0.5, 2.0], [-3.0, -1.0, -0.25],          # negative values allowed
                 [0.1, 0.2, 0.3, -0.05, 7.5, 0.0, 1e-3], [2.5, -4.0, 3.0, -1.0, 0.5, 6.0, -2.25, 1.0],
                 [1e6, -1e-6, 3.0, 2 ** -40, -7.0, 123.456])
        for values in cases:
            with self.subTest(values=values):
                c = {"values": values}
                ind = p.independent_score(c)
                self.assertEqual(ind["fraction"], direct_score(values))
                self.assertLessEqual(abs(p.score(c) - ind["score"]), ind["tolerance_abs"])

    def test_score_ignores_a_change_of_scale(self):
        values = [1.0, -0.5, 2.0, 0.75]
        a = self.p.independent_score({"values": values})["fraction"]
        b = self.p.independent_score({"values": [8.0 * v for v in values]})["fraction"]
        self.assertEqual(a, b)

    @unittest.skipUnless((CANDIDATES / "MANIFEST.json").exists(), "scout candidates are not on this branch")
    def test_agrees_with_pinned_on_a_stored_candidate(self):
        p = self.p
        manifest = json.loads((CANDIDATES / "MANIFEST.json").read_text())["candidates"]
        entry = manifest["milo_replay_25600_c_remelt"]
        raw = gzip.decompress((CANDIDATES / entry["file"]).read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256_json"])
        c = json.loads(raw)
        pinned, ind = p.score(c), p.independent_score(c)
        self.assertEqual(pinned, entry["score_pinned_verifier"])
        self.assertEqual(ind["score"], entry["score_exact_rational"])
        self.assertLessEqual(abs(pinned - ind["score"]), ind["tolerance_abs"])
        self.assertIsInstance(ind["fraction"], Fraction)


if __name__ == "__main__":
    unittest.main()
