"""Test A: dry runs with the fake executor (seconds). Run from the repo root:

    .venv/bin/python -m unittest autoloop.tests.test_autoloop -v
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from autoloop import problem
from autoloop.ledger import Ledger, cert_record, import_memory
from autoloop.loop import cfg_from, run
from autoloop.moves import ContractError, Moves
from autoloop.report import report

P = problem.load("tao-1d")


def cfg(tmp, *extra):
    return cfg_from(["--run", str(tmp), "--executor", "fake", "--quiet", "--env-file", "/nonexistent", *extra])


def ledger(tmp):
    return [json.loads(x) for x in (Path(tmp) / "ledger.jsonl").read_text().splitlines()]


class DryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="autoloop_test_"))

    def test_loop_closes_and_decision_cap(self):
        end = run(cfg(self.tmp, "--max-decisions", "25"))
        self.assertEqual(end["decisions"], 25)
        self.assertIn("25 decisions", end["reason"])
        recs = ledger(self.tmp)
        moves = [r["move"] for r in recs if r["kind"] == "decision"]
        for m in ("check_reference", "seed", "certify"):
            self.assertIn(m, moves)
        self.assertTrue({"refine", "refine_wide"} & set(moves))
        # every decision has a reason, the numbers it looked at, its options, and a result
        res = {r["i"] for r in recs if r["kind"] == "result"}
        for d in (r for r in recs if r["kind"] == "decision"):
            self.assertTrue(d["reason"] and d["options"] and "best" in d["numbers"] and d["i"] in res)
        self.assertTrue(any(r["kind"] == "claim" for r in recs))
        self.assertEqual(recs[-1]["kind"], "run_end")

    def test_wall_cap(self):
        end = run(cfg(self.tmp, "--max-wall-s", "0"))
        self.assertIn("wall clock", end["reason"]); self.assertEqual(end["decisions"], 0)

    def test_usd_cap(self):
        end = run(cfg(self.tmp, "--max-usd", "0.01"))     # fake cost 0.001 USD per CPU-second
        self.assertIn("compute USD", end["reason"])

    def test_menu_exhaustion_stops_by_itself(self):
        end = run(cfg(self.tmp, "--max-wall-s", "200", "--refine-budget-s", "120"))   # too little time to refine
        self.assertIn("Nothing legal left", end["reason"])

    def test_contract_child_worse_than_parent(self):
        L = Ledger(self.tmp / "ledger.jsonl")
        ex = P.FakeExecutor(self.tmp)
        mv = Moves(L, ex, P)
        np.save(self.tmp / "p.npy", np.ones(64))
        L.append(cert_record(problem=P.NAME, cid="a" * 64, cells=800, score_=1.46, parent=None, stage="seed",
                             params={}, cpu_s=0, path=str(self.tmp / "p.npy")))
        ex.refine = lambda *a, **k: ([dict(ok=True, path=str(self.tmp / "p.npy"), reported=1.5, params=dict(split=4, p0=16))], 1.0)
        with self.assertRaises(ContractError):
            mv.refine("a" * 64, 4, 16.0)
        with self.assertRaises(ContractError):
            mv.refine("b" * 64, 4, 16.0)                    # unknown parent
        with self.assertRaises(ContractError):
            mv.refine("a" * 64, 64, 16.0)                   # above the executor's cell limit

    def test_llm_valid_answer_is_used_and_costed(self):
        os.environ["AUTOLOOP_FAKE_LLM"] = '{"choice": 0, "reason": "read the reference first"}'
        run(cfg(self.tmp, "--policy", "llm", "--llm-provider", "fake", "--max-decisions", "3"))
        recs = ledger(self.tmp)
        d = [r for r in recs if r["kind"] == "decision"]
        self.assertTrue(all(x["policy"] == "llm:fake:default" and x["move"] == "check_reference" for x in d))
        self.assertTrue(all(x["llm_usage"]["tokens_in"] > 0 and x["llm_usage"]["usd"] > 0 for x in d))
        self.assertEqual(len(list((self.tmp / "llm").glob("call_*.json"))), 3)

    def test_llm_invalid_answer_falls_back_to_rules(self):
        for bad in ('{"choice": 99, "reason": "x"}', "not json at all", '{"choice": 1}'):
            os.environ["AUTOLOOP_FAKE_LLM"] = bad
            tmp = Path(tempfile.mkdtemp(prefix="autoloop_test_"))
            run(cfg(tmp, "--policy", "llm", "--llm-provider", "fake", "--max-decisions", "2"))
            d = [r for r in ledger(tmp) if r["kind"] == "decision"]
            self.assertTrue(all(x["policy"] == "rules" and "invalid LLM answer" in x["fallback"] for x in d), bad)

    def test_llm_usd_cap(self):
        os.environ["AUTOLOOP_FAKE_LLM"] = '{"choice": 0, "reason": "x"}'
        run(cfg(self.tmp, "--policy", "llm", "--llm-provider", "fake", "--max-decisions", "6", "--llm-usd-cap", "0.005"))
        d = [r for r in ledger(self.tmp) if r["kind"] == "decision"]
        self.assertEqual(sum(1 for x in d if x["policy"].startswith("llm")), 3)    # 3 x 0.002 USD crosses 0.005
        self.assertIn("LLM budget", d[-1]["fallback"])

    def test_memory_import_keeps_origin(self):
        run(cfg(self.tmp, "--max-decisions", "8"))
        tmp2 = Path(tempfile.mkdtemp(prefix="autoloop_test_"))
        run(cfg(tmp2, "--max-decisions", "2", "--memory", str(self.tmp / "ledger.jsonl")))
        recs = ledger(tmp2)
        k = next(i for i, r in enumerate(recs) if r["kind"] == "run_start")
        certs = [r for r in recs[:k] if r["kind"] == "certificate"]
        self.assertTrue(certs and all(c["origin"] == "imported" for c in certs))
        self.assertTrue(all(c["origin"] == "autoloop" for c in recs[k:] if c["kind"] == "certificate"))
        d = [r for r in ledger(tmp2) if r["kind"] == "decision"]
        self.assertNotEqual(d[1]["move"], "seed")      # enough lineages in memory: no new seeds
        # import from a directory of solution files
        L = Ledger(Path(tempfile.mkdtemp()) / "l.jsonl")
        np.save(self.tmp / "certs" / "extra.npy", np.random.default_rng(1).normal(size=200) + 1)
        n = import_memory(L, str(self.tmp / "certs"), P)
        self.assertGreater(n, 0); self.assertTrue(all(c["origin"] == "imported" for c in L.certs()))

    def test_report_reads(self):
        run(cfg(self.tmp, "--max-decisions", "10"))
        txt = report(self.tmp / "ledger.jsonl")
        self.assertIn("RUN policy=rules", txt); self.assertIn("END cap: 10 decisions", txt); self.assertIn("->", txt)

    def test_wide_recipe_advances_all_lineages_then_soft_restarts(self):
        run(cfg(self.tmp, "--max-decisions", "14", "--min-roots", "8", "--workers", "4", "--max-cells", "51200",
                "--executor", "fake", "--seed-base", "5000", "--max-wall-s", "100000"))
        recs = ledger(self.tmp)
        d = [r for r in recs if r["kind"] == "decision"]
        wide = [x for x in d if x["move"] == "refine_wide"]
        self.assertTrue(wide)
        self.assertEqual(len(wide[0]["args"]["certs"]), 8)                 # no early cut: all 8 lineages advance
        self.assertTrue(any(x["args"]["split"] == 1 and x["args"]["p0"][0] == 4.0 for x in wide))   # soft restarts
        seeds = [c["params"]["seed"] for c in recs if c["kind"] == "certificate" and c["stage"] == "seed"]
        self.assertEqual(min(seeds), 5000)
        self.assertIn("commit", next(r for r in recs if r["kind"] == "run_start"))

    def test_guidelines_wide_sizes_and_regate(self):
        g = self.tmp / "g.md"; g.write_text("1. Breadth first.")
        os.environ["AUTOLOOP_FAKE_LLM"] = "garbage"           # LLM invalid -> rules fallback, prompt still saved
        run(cfg(self.tmp, "--policy", "llm", "--llm-provider", "fake", "--guidelines", str(g), "--max-decisions", "6",
                "--seed-sizes", "64,200", "--wide-sizes", "64,30,16", "--min-roots", "100", "--workers", "8",
                "--max-cells", "51200", "--max-wall-s", "100000"))
        recs = ledger(self.tmp)
        self.assertTrue(any(r.get("guidelines_sha256") for r in recs))
        d = [r for r in recs if r["kind"] == "decision"]
        self.assertEqual(d[1]["args"]["n"], 64)               # rules fallback picks the largest seed batch that fits
        prompt = json.loads((self.tmp / "llm" / "call_002.json").read_text())["prompt"]
        self.assertIn("Breadth first", prompt); self.assertIn("seed {\"n\": 200", prompt)

    def test_claim_threshold_is_conservative(self):
        from fractions import Fraction
        self.assertEqual(P.claim_threshold("1.4546733414"), Fraction("1.45467334135"))


if __name__ == "__main__":
    unittest.main()


class TestRefreadCases(unittest.TestCase):
    """Lupa's offline cases E-I (wt-referee checks/refread_offline.py): never ok=True from the page table alone."""
    def test_cases(self):
        import re
        from autoloop.refread import read
        num, page = re.compile(r"1\.4[0-9]{4,}"), "<h2>Known upper bounds</h2> 1.4581 <h2>Known lower bounds</h2> 1.41"
        r = lambda prs: read("1d", "P", num, 1.40, 1.50, gh_fetch=lambda: prs, page_fetch=lambda: page)  # noqa: E731
        self.assertEqual(r([dict(number=1, title="Improve the upper bound for C_1d to 1.4540000001", body="")])["value"], "1.4540000001")
        self.assertEqual(r([dict(number=1, title="1d: upper bound 1.4541", body="")])["value"], "1.4541")
        self.assertEqual(r([dict(number=1, title="1d: better", body="The new bound is 1.4540000001.")])["value"], "1.4540000001")
        self.assertFalse(r([dict(number=1, title="1d: better construction", body="see attached")])["ok"])
        self.assertFalse(r([dict(number=i, title="x", body="") for i in range(300)])["ok"])
        self.assertTrue(r([dict(number=1, title="1d: remove a duplicated row", body="")])["ok"])   # editorial PR skipped
