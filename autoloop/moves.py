"""Typed moves with contract checks: seed(n, budget_s), refine(cert, split, p0), certify(cert), check_reference(),
stop(reason). Problem-specific work is delegated to the problem plug-in and its executor (see autoloop/problem.py).

Contract on every move that makes certificates: same problem, the parent exists in the ledger and is completed, the
child's score is not worse than its parent's (relative tolerance 1e-9, which covers the 2^56 rounding of certify),
the cell count stays within the executor's limit, and with a real executor the score reported by the worker is
recomputed here from the file.
"""
from __future__ import annotations

import numpy as np

from .ledger import Ledger, cert_record

TOL = 1e-9


class ContractError(Exception):
    pass


class Moves:
    def __init__(self, ledger: Ledger, ex, prob, *, refine_budget_s=120.0, ref_fetcher=None, modal_budget_s=3000.0,
                 seed_base=1000):
        self.L, self.ex, self.P = ledger, ex, prob
        self.refine_budget, self.modal_budget, self.seed_base = refine_budget_s, modal_budget_s, seed_base
        self.ref_fetcher = ref_fetcher or prob.fetch_reference
        self.cpu_s = 0.0

    @property
    def usd(self):
        return self.cpu_s * self.ex.usd_per_cpu_s + getattr(self.ex, "spent_usd", 0.0)

    def _child(self, row, parent, stage, cpu_each):
        if not row["ok"]:
            return None
        if self.ex.real:
            v = self.P.load_vec(row["path"]); s = self.P.score(v); cid = self.P.vec_id(v); cells = len(v)
            if abs(s - row["reported"]) > 1e-12 * abs(s):
                raise ContractError(f"worker reported {row['reported']} but the file scores {s}")
        else:
            s = row["reported"]; cid = self.P.vec_id(np.load(row["path"]))
            cells = (parent["cells"] * row["params"]["split"]) if parent else row["params"]["fake_cells"]
        if parent is not None:
            if parent["problem"] != self.P.NAME:
                raise ContractError("parent belongs to another problem")
            if s > parent["score"] + TOL * abs(parent["score"]):
                raise ContractError(f"child {s} worse than parent {parent['score']}")
        return cert_record(problem=self.P.NAME, cid=cid, cells=cells, score_=s,
                           parent=parent["id"] if parent else None,
                           root=parent.get("root", parent["id"]) if parent else None, stage=stage,
                           params=row["params"], cpu_s=cpu_each, path=row["path"])

    def _parent(self, cid):
        c = self.L.cert(cid)
        if c is None:
            raise ContractError(f"unknown certificate {cid}")
        if c["problem"] != self.P.NAME or not c.get("completed", True):
            raise ContractError(f"certificate {cid[:10]} is not a completed {self.P.NAME} certificate")
        return c

    def seed(self, n, budget_s):
        seed0 = self.seed_base + len([c for c in self.L.certs() if c["stage"] == "seed"])
        rows, cpu = self.ex.seed(int(n), float(budget_s), self.P.SEED_CELLS, seed0)
        self.cpu_s += cpu
        out = [self._child(r, None, "seed", cpu / max(1, len(rows))) for r in rows]
        return [c for c in out if c], dict(failed=sum(1 for r in rows if not r["ok"]), cpu_s=round(cpu, 1))

    def refine(self, cert, split, p0):
        par = self._parent(cert)
        p0s = p0 if isinstance(p0, list) else [p0]
        if par["cells"] * int(split) > self.ex.max_cells:
            raise ContractError(f"{par['cells']}x{split} cells above the executor limit {self.ex.max_cells}")
        tag = f"r{len(self.L.decisions()):03d}_{cert[:8]}"
        kw = {} if self.ex.real else dict(parent_score=par["score"])
        budget = self.refine_budget if par["cells"] * int(split) <= self.P.LOCAL_MAX_CELLS else self.modal_budget
        rows, cpu = self.ex.refine(par["path"], int(split), [float(x) for x in p0s], budget, tag, **kw)
        self.cpu_s += cpu
        out = [self._child(r, par, "refine", cpu / max(1, len(rows))) for r in rows]
        return [c for c in out if c], dict(failed=sum(1 for r in rows if not r["ok"]), cpu_s=round(cpu, 1))

    def refine_wide(self, certs, split, p0):
        """Same split and p0 list on several lineages at once (one Modal launch when the executor allows)."""
        pars = [self._parent(c) for c in certs]
        if len({p["cells"] for p in pars}) != 1:
            raise ContractError("refine_wide parents must have the same cell count")
        out_cells = pars[0]["cells"] * int(split)
        if out_cells > self.ex.max_cells:
            raise ContractError(f"{out_cells} cells above the executor limit {self.ex.max_cells}")
        p0s = [float(x) for x in (p0 if isinstance(p0, list) else [p0])]
        budget = self.refine_budget if out_cells <= self.P.LOCAL_MAX_CELLS else self.modal_budget
        tag = f"w{len(self.L.decisions()):03d}"
        kw = {} if self.ex.real else dict(parent_scores=[p["score"] for p in pars])
        rows, cpu = self.ex.refine_many([p["path"] for p in pars], int(split), p0s, budget, tag, **kw)
        self.cpu_s += cpu
        out = [self._child(r, pars[r["pi"]], "refine", cpu / max(1, len(rows))) for r in rows]
        return [c for c in out if c], dict(failed=sum(1 for r in rows if not r["ok"]), cpu_s=round(cpu, 1),
                                           jobs=len(rows), budget_s=budget)

    def certify(self, cert):
        par = self._parent(cert)
        kw = {} if self.ex.real else dict(parent_score=par["score"])
        r, cpu = self.ex.certify(par["path"], f"c{len(self.L.decisions()):03d}_{cert[:8]}", **kw)
        self.cpu_s += cpu
        if not r["ok"]:
            raise ContractError(r["error"])
        ex = r["exact"]
        if float(ex) > par["score"] + TOL * abs(par["score"]):
            raise ContractError(f"exact {float(ex)} worse than parent float {par['score']}")
        cid = self.P.vec_id(np.asarray(r["ints"], dtype=np.float64)) if r["ints"] is not None else "x" + cert[1:]
        c = cert_record(problem=self.P.NAME, cid=cid, cells=par["cells"], score_=float(ex), parent=par["id"],
                        root=par.get("root", par["id"]), stage="certified", params=dict(bits=56, knot=r["knot"]),
                        cpu_s=cpu, path=r["path"], exact=f"{ex.numerator}/{ex.denominator}")
        return [c], dict(cpu_s=round(cpu, 1))

    def check_reference(self):
        return self.ref_fetcher()
