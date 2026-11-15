"""WS1: tests of the frozen problems in problems/<slug>/ (standard library unittest; also runs under pytest).

    .venv/bin/python -m unittest discover -s tests -p "test_ws1_*.py"             (~20 s)
    WS1_LIVE_TESTS=1 ... also checks the live leaderboard read (network)

What they guarantee:
  - freeze rule: the pinned verifiers and the certificates have the sha256 recorded below, and every WS1 file under
    problems/ matches problems/MANIFEST.json. Any change fails here. Agents never modify problems/;
  - each certificate (the live #1 at freeze time) gets the displayed score from the pinned verifier, and the
    independent verifier agrees;
  - circles-rectangle: the pinned verifier rejects the tolerance exploits reported in selection/digests/;
  - the server-schema limits are enforced by verify().

The freeze rule and the certificate tests also cover the problems frozen as a proposal (PROPOSED_SLUGS in
selection/verify_claim.py), which are loaded by path because problems/catalog.py does not list them yet. The tests
specific to third-autocorrelation-inequality are in tests/test_ws1_third_autocorrelation.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from problems.catalog import SLUGS, load  # noqa: E402
from selection.verify_claim import PROPOSED_SLUGS, load_problem  # noqa: E402

# The catalog problems plus the problems frozen as a proposal (not in problems/catalog.py until Arthur accepts).
FROZEN_SLUGS = SLUGS + PROPOSED_SLUGS

# Frozen on 2026-11-14 from github.com/vinid/einstein-arena @ 5baf6361a3e0e2123edb98f676ef44f7ab081ad6.
# third-autocorrelation-inequality: frozen on 2026-11-15 from the same commit, as a proposal.
# Changing these values requires Arthur's approval (selection/freeze_problems.py --refreeze).
VERIFIER_SHA256 = {
    "second-autocorrelation-inequality": "dc5ccffaa20ba6c112cab36ffe602c657372d192d8b486abd66b14bcad1ca768",
    "circles-rectangle": "77cf0c63c000f1abc2d89aae18d4d7407599b8f683f304fe6cf9adeba966482d",
    "ring-loading-15": "a91b09ba472d9506a1d54603f977872f37505bdcc0e258b5dddd1ed99f148a57",
    "third-autocorrelation-inequality": "b8288d5943d72032f1d2bcf5d8a3b3a00cfd428ae0347a26bfc53974ec7ebce9",
}
CERTIFICATE_SHA256 = {
    "second-autocorrelation-inequality": "ae37b4fc12a29c416e219b521f88d09345d8748c3b50926cb18823bc55030cae",
    "circles-rectangle": "6de2c84392d8c1d3ad2601c7836db06cf0917b81fa52ff1fc199509b361e121f",
    "ring-loading-15": "013531d4876692d67aa2819220873f78ec94303e45032be31311eee273840cfe",
    "third-autocorrelation-inequality": "670cf76b9e75b4d727940e36b4936878cd7b6e17017f00f4ae8dcca57257bc96",
}
DISPLAYED = {  # leaderboard #1 at freeze time (bestScore as served by the API)
    "second-autocorrelation-inequality": 0.963638667992813,
    "circles-rectangle": 2.3658323759098576,
    "ring-loading-15": 1.125,
    "third-autocorrelation-inequality": 1.449109789331911,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FreezeRule(unittest.TestCase):
    def test_pinned_verifier_hash_is_unchanged(self):
        for slug in FROZEN_SLUGS:
            with self.subTest(slug=slug):
                path = REPO_ROOT / "problems" / slug / "verifier_pinned.py"
                self.assertEqual(sha256_file(path), VERIFIER_SHA256[slug], f"{slug}: pinned verifier changed")
                meta = json.loads((REPO_ROOT / "problems" / slug / "meta.json").read_text())
                self.assertEqual(meta["verifier"]["sha256"], VERIFIER_SHA256[slug])

    def test_verifier_hash_matches_stage2_registry(self):
        rows = {json.loads(l)["slug"]: json.loads(l)
                for l in (REPO_ROOT / "selection" / "registry_stage2.jsonl").read_text().splitlines()}
        for slug in FROZEN_SLUGS:
            with self.subTest(slug=slug):
                self.assertEqual(rows[slug]["verifier_pinned_sha256"], VERIFIER_SHA256[slug])

    def test_certificate_hash_is_unchanged(self):
        for slug in FROZEN_SLUGS:
            with self.subTest(slug=slug):
                meta = json.loads((REPO_ROOT / "problems" / slug / "meta.json").read_text())
                self.assertEqual(meta["certificate"]["sha256"], CERTIFICATE_SHA256[slug])
                load_problem(slug).certificate()  # raises FrozenFileChanged if the decompressed bytes differ

    def test_every_ws1_file_under_problems_matches_the_manifest(self):
        manifest = json.loads((REPO_ROOT / "problems" / "MANIFEST.json").read_text())["sha256"]
        for rel, digest in manifest.items():
            with self.subTest(file=rel):
                self.assertEqual(sha256_file(REPO_ROOT / rel), digest, f"{rel} changed")
        for slug in FROZEN_SLUGS:
            present = {p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "problems" / slug).rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts}
            with self.subTest(slug=slug):
                self.assertTrue(present, "the frozen problem directory is empty")
                self.assertEqual(present - set(manifest), set(), "unexpected file in a frozen problem directory")

    def test_catalog_lists_only_the_accepted_problems(self):
        # A proposed problem stays out of problems/catalog.py (and so out of load(), ws2_module() and submit.py)
        # until Arthur accepts it.
        self.assertEqual(SLUGS, ("second-autocorrelation-inequality", "circles-rectangle", "ring-loading-15"))
        for slug in PROPOSED_SLUGS:
            with self.subTest(slug=slug):
                self.assertRaises(KeyError, load, slug)
                self.assertEqual(load_problem(slug).ROLE, "proposed_main")


class Certificates(unittest.TestCase):
    def test_certificate_reproduces_displayed_score(self):
        for slug in FROZEN_SLUGS:
            with self.subTest(slug=slug):
                p = load_problem(slug)
                s = p.score(p.certificate())
                if slug == "second-autocorrelation-inequality":
                    # The server runs the same verifier in an E2B sandbox whose numpy/scipy versions are not pinned;
                    # oaconvolve rounding differs by a few ulp (observed 7.8e-16). Bit-exact equality is out of reach
                    # here; the exact rational value is checked in test_independent_agrees_with_pinned.
                    self.assertLessEqual(abs(s - DISPLAYED[slug]), 1e-12 * DISPLAYED[slug])
                else:
                    self.assertEqual(s, DISPLAYED[slug])

    def test_independent_agrees_with_pinned(self):
        for slug in FROZEN_SLUGS:
            with self.subTest(slug=slug):
                p = load_problem(slug)
                c = p.certificate()
                ind = p.independent_score(c)
                self.assertLessEqual(abs(ind["score"] - p.score(c)), ind["tolerance_abs"])
                self.assertIsInstance(ind["fraction"], Fraction)
                self.assertLessEqual(abs(float(ind["fraction"]) - DISPLAYED[slug]), 1e-12)

    def test_ring_loading_record_is_exactly_nine_eighths(self):
        p = load("ring-loading-15")
        self.assertEqual(p.independent_score(p.certificate())["fraction"], Fraction(9, 8))

    def test_certificate_does_not_beat_itself(self):
        for slug in FROZEN_SLUGS:
            with self.subTest(slug=slug):
                p = load_problem(slug)
                step = max(p.DELTA, 1e-12) * 1.5
                better = step if p.DIRECTION == "maximize" else -step    # minimize: a lower score is better
                self.assertFalse(p.beats_record(DISPLAYED[slug], best=DISPLAYED[slug]))
                self.assertTrue(p.beats_record(DISPLAYED[slug] + better, best=DISPLAYED[slug]))
                self.assertFalse(p.beats_record(DISPLAYED[slug] - better, best=DISPLAYED[slug]))


def _grid(r: Fraction, step: Fraction, x0: Fraction = Fraction(0), y0: Fraction = Fraction(0)):
    """7 x 3 grid of 21 circles of radius r, centres `step` apart (exact dyadic values)."""
    return [[float(x0 + r + i * step), float(y0 + r + j * step), float(r)] for j in range(3) for i in range(7)]


class CirclesExploitsRejected(unittest.TestCase):
    """The pinned verifier (commit 5baf636) closes the exploits reported in selection/digests/circles-rectangle.md."""

    @classmethod
    def setUpClass(cls):
        cls.p = load("circles-rectangle")

    def test_overlap_tolerance_exploit(self):
        # [t185, JSAgent]: the old verifier accepted overlaps up to ~1e-9. Touching circles are valid; pushing one
        # circle 2^-31 (~4.7e-10) into its neighbour must be rejected by both verifiers.
        touching = _grid(Fraction(1, 16), Fraction(1, 8))
        self.assertTrue(self.p.verify({"circles": touching}))
        cheat = [list(c) for c in touching]
        cheat[1][0] -= 2.0 ** -31
        self.assertFalse(self.p.verify({"circles": cheat}))
        with self.assertRaises(ValueError):
            self.p.independent_score({"circles": cheat})

    def test_perimeter_tolerance_exploit(self):
        # [t185, JSAgent]: the old verifier accepted width + height up to 2 + ~1e-9. A grid with width + height = 2
        # exactly is valid; widening it by 2^-31 must be rejected (the perimeter is checked in exact Fractions).
        r, step = Fraction(1, 16), Fraction(7, 32)        # width + height = 8 * step + 4 * r = 2 exactly
        exact = _grid(r, step)
        self.assertTrue(self.p.verify({"circles": exact}))
        cheat = [list(c) for c in exact]
        for c in cheat[6::7]:                              # last column, moved right by 2^-31
            c[0] += 2.0 ** -31
        xs = [Fraction(c[0]) for c in cheat]
        width = max(x + r for x in xs) - min(x - r for x in xs)
        self.assertGreater(width + 2 * step + 2 * r, 2)    # really over the perimeter
        self.assertFalse(self.p.verify({"circles": cheat}))
        with self.assertRaises(ValueError):
            self.p.independent_score({"circles": cheat})

    def test_large_coordinate_float_exploit(self):
        # [t141/r1068, CodexLatticeKR]: near 2^50 the float spacing is 0.25, so x + 1/8 rounds back to x and a float
        # width + height computation loses the radii. 21 circles of radius 1/8 would "fit" and score 21/8 = 2.625.
        # Centres at 2^50 + 1 + k/4 (k = 0..6 and 0..2), all in [2^50, 2^51) where the spacing is 1/4. The extreme
        # centres have an even last mantissa bit, so x +/- 1/8 (an exact half-ulp tie) rounds back to x.
        r, step, base = Fraction(1, 8), Fraction(1, 4), Fraction(2 ** 50 + 1)
        cheat = _grid(r, step, base - r, base - r)
        xs, ys = [c[0] for c in cheat], [c[1] for c in cheat]
        naive = (max(x + 0.125 for x in xs) - min(x - 0.125 for x in xs)
                 + max(y + 0.125 for y in ys) - min(y - 0.125 for y in ys))
        self.assertLessEqual(naive, 2.0)                    # the float shortcut would accept it ...
        self.assertEqual(sum(c[2] for c in cheat), 2.625)   # ... with a "record" of 2.625
        self.assertFalse(self.p.verify({"circles": cheat}))  # the pinned verifier does not
        with self.assertRaises(ValueError):
            self.p.independent_score({"circles": cheat})


class SchemaAndValidity(unittest.TestCase):
    def test_second_autocorrelation_value_limit(self):
        p = load("second-autocorrelation-inequality")
        self.assertIn("server schema", p.why_invalid({"values": [1.0] * 2_000_001}))
        self.assertTrue(p.verify({"values": [1.0, 2.0, 1.0]}))

    def test_second_autocorrelation_rejects_negative_values(self):
        p = load("second-autocorrelation-inequality")
        bad = {"values": [1.0, -1e-3, 1.0]}
        self.assertFalse(p.verify(bad))
        with self.assertRaises(ValueError):
            p.independent_score(bad)

    def test_small_second_autocorrelation_agree(self):
        p = load("second-autocorrelation-inequality")
        c = {"values": [0.5, 3.0, 0.0, 1.25, 2.0, 0.0, 0.0, 7.5]}
        self.assertLessEqual(abs(p.independent_score(c)["score"] - p.score(c)), 1e-12)

    def test_ring_loading_rejects_infeasible_pair(self):
        p = load("ring-loading-15")
        c = p.certificate()
        c["pairs"][0] = ["3/4", "1/2"]
        self.assertFalse(p.verify(c))
        with self.assertRaises(ValueError):
            p.independent_score(c)

    def test_large_candidate_uses_blob_path(self):
        p = load("second-autocorrelation-inequality")
        self.assertEqual(p.submission_path(p.certificate()), "blob")
        self.assertEqual(p.submission_path({"values": [1.0] * 1000}), "inline")


@unittest.skipUnless(os.environ.get("WS1_LIVE_TESTS") == "1", "set WS1_LIVE_TESTS=1 to read the live leaderboard")
class LiveRecord(unittest.TestCase):
    def test_best_known_is_read_live(self):
        for slug in FROZEN_SLUGS:
            with self.subTest(slug=slug):
                p = load_problem(slug)
                best = p.best_known_score(max_age_s=0)
                self.assertTrue(best["live"])
                sign = 1 if p.DIRECTION == "maximize" else -1
                self.assertEqual(p.target(), best["value"] + sign * p.DELTA)


if __name__ == "__main__":
    unittest.main()
