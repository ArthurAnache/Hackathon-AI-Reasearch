"""Main arm of Knockout Evolution (sections 5.1 to 5.8): islands of programs steered by gated insight genes.

One `step()` is one batch on one island:
  island pick (bandit) -> per child: operator, model, parents, genome (Thompson) -> LLM writes the programs ->
  anti-cycling filters -> F0 evaluation of the children and of the parents they are compared with ->
  credit, bandits, knock-out gate, gene birth from diffs -> fidelity ladder -> lineage freezing, migration, re-audit.

Every choice goes through knockout.decisions (pure functions) and is logged with its inputs, outputs and seed, so
knockout.replay can re-derive it. Fitness is always the tuned score from the frozen verifier.

Warm-start ratchet (`main.ratchet`): the arm has its own warm start, separate from the control arm's. The best
verified candidate of a batch becomes `init` of the next batch (a new "epoch"). A child is compared with its parent
only at the same fidelity and the same epoch, so parents are replayed when the warm start has moved.
"""
from __future__ import annotations

import ast
import concurrent.futures as cf
import difflib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from knockout import decisions as D
from knockout import filters as F
from knockout import portfolio as P
from knockout.basins import Archive, cell_name
from knockout.context import Evaluation, RunContext, source_hash
from knockout.genes import Gene, GeneStore, render, similarity
from knockout.llm import LLM
from knockout.llm_editor import (OPERATORS, SCHEMA, SYSTEM, check_program, history_block, memory_section,
                                 problem_brief, state_block, strip_fence)

FIDELITIES = ("F0", "F1", "F2")
MAIN_OPERATORS = {
    "refine": OPERATORS["refine"] + " A child that only changes constants is rejected as a near-duplicate before "
              "evaluation, so change at least a few lines of logic.",
    "restructure": OPERATORS["restructure"],
    "hybridize": "Two parent programs are shown. Write one program that combines what makes each of them work.",
    "reformulate": "Propose another formulation of the problem: `construct` may return any JSON-serializable "
                   "object, and `adapter_source` is a Python module defining `to_canonical(obj)` that maps that "
                   "object to the frozen candidate format. Give the formulation in `formulation` (at most 60 words).",
    "fresh": "No parent program is shown. Write a program from the insights below and the problem statement.",
}
SCHEMA_REFORMULATE = {
    "type": "object",
    "properties": {**SCHEMA["properties"], "formulation": {"type": "string"}, "adapter_source": {"type": "string"}},
    "required": [*SCHEMA["required"], "formulation", "adapter_source"],
    "additionalProperties": False,
}
GENES_SCHEMA = {
    "type": "object",
    "properties": {"genes": {"type": "array", "items": {
        "type": "object",
        "properties": {"type": {"type": "string", "enum": ["structure", "numeric", "formulation"]},
                       "text": {"type": "string"}},
        "required": ["type", "text"], "additionalProperties": False}}},
    "required": ["genes"], "additionalProperties": False,
}
GENE_RULES = ("A gene is one reusable instruction for whoever writes the next program: imperative, self-contained, "
              "at most 60 words, about the method and not about one program's variable names. Types: `structure` "
              "(how the search or the object is organised), `numeric` (a numerical technique or a regime of "
              "values), `formulation` (which unknowns or surrogate to optimise).")
CANONICAL_TEXT = "Canonical formulation: `construct` returns the candidate directly in the verifier's own format."


@dataclass
class Member:
    id: str
    program_id: str
    source: str
    island_id: str
    lineage: str
    genome: list[str]
    parents: list[str]
    operator: str
    model: str | None
    born_at: int
    hypothesis: str = ""
    scores: dict[str, float] = field(default_factory=dict)   # "F0@<epoch>" -> tuned score
    exp_ids: dict[str, str] = field(default_factory=dict)
    fit: float | None = None      # ranking key: F0 score, or F0 gain over the warm start with the ratchet
    alive: bool = True
    fit_by_regime: dict[str, float] = field(default_factory=dict)   # start portfolio: ranking key per regime

    def score(self, fid: str, key) -> float | None:
        """`key` is the epoch in v1, the start set id with the start portfolio."""
        return self.scores.get(f"{fid}@{key}")


@dataclass
class Lineage:
    id: str
    deleted: set[str] = field(default_factory=set)
    since_improvement: int = 0
    frozen: bool = False
    notes: list[str] = field(default_factory=list)     # deletion history shown after a "recycled lines" reject


@dataclass
class Gate:
    gene_id: str
    wins: int = 0
    losses: int = 0
    pairs: int = 0
    audit: bool = False
    evidence: list[str] = field(default_factory=list)


@dataclass
class Island:
    id: str
    formulation: str                      # gene id
    adapter_source: str | None
    canonical: bool
    created_fraction: float
    pool: list[str] = field(default_factory=list)            # member ids that may be parents
    hashes: set[str] = field(default_factory=set)
    tokens: list[frozenset[str]] = field(default_factory=list)
    lineages: dict[str, Lineage] = field(default_factory=dict)
    best_score: float | None = None
    evals: int = 0
    since_improvement: int = 0
    gate: Gate | None = None
    last_migration: int = 0
    last_audit: int = 0


@dataclass
class Spec:
    kind: str                             # "normal" | "with" | "without" (gate pair)
    operator: str
    model: str
    parent: Member                        # the program the child is compared with
    parent_b: Member | None
    genome: list[str]
    negatives: list[str]
    phase: str
    seed: int
    category: str
    trial: str | None = None              # gate: the gene under test (rendered for "with" only)
    pair: str | None = None
    source: str | None = None
    hypothesis: str = ""
    meta: dict = field(default_factory=dict)
    adapter_source: str | None = None
    formulation: str | None = None
    ev: Evaluation | None = None
    exp_id: str | None = None
    member: Member | None = None
    s: int = 0
    ref: float | None = None              # F0 score of the parent, same warm start


class MainArm:
    category = "main"
    chain = "main"
    agent = "knockout/main-gene-evolution"

    def __init__(self, ctx: RunContext, llm: LLM, seed_source: str, rng: np.random.Generator):
        st = ctx.strategy
        self.ctx, self.llm, self.rng = ctx, llm, rng
        self.seed_source = seed_source
        self.brief = problem_brief(ctx.problem)
        self.direction = ctx.problem.direction
        self.eval_seed = int(st["run.seed"])
        self.ratchet = bool(st.get("main.ratchet", st.get("control.ratchet", False)))
        # start portfolio: the `record` regime is v1 with the ratchet, so the portfolio turns the ratchet on
        self.portfolio = bool(st.get("portfolio.enabled", False))
        if self.portfolio:
            self.ratchet = True
        self.show_record = bool(st.get("prompt.show_record", True))
        self.init, self.init_score, self.epoch = ctx.init, ctx.init_score, 0
        self.regimes = list(st.get("portfolio.regimes", P.REGIMES)) if self.portfolio else ["record"]
        self.regime = "record"
        self.cur_set: P.StartSet | None = None
        self.start_sets: dict[str, P.StartSet] = {}
        self.set_worst: dict[str, list] = {}               # start set id -> worst valid score seen per start
        self.regime_arms = {r: [1.0, 1.0] for r in self.regimes}
        self.regime_spend = {r: 0.0 for r in self.regimes}
        self.regime_stats = {r: {"batches": 0, "children": 0, "beat_parent": 0, "best_regime_score": None,
                                 "best_progress": None} for r in self.regimes}
        self.archive: Archive | None = None
        self.promotions: list[dict] = []
        self._n_neighbour = 0
        self._other_basin = False                          # a child improved a cell other than the incumbent's
        self.genes = GeneStore(float(st.get("genes.dedupe_similarity", 0.8)))
        self.islands: dict[str, Island] = {}
        self.members: dict[str, Member] = {}
        self.island_arms: dict[str, list[float]] = {}
        self.operator_arms = {op: [float(a), float(b)] for op, (a, b) in st["operator_priors"].items()
                              if op in MAIN_OPERATORS}
        self.model_arms = {m: [1.0, 1.0] for m in llm.models}
        self.op_arms_r = {r: {k: list(v) for k, v in self.operator_arms.items()} for r in self.regimes}
        self.pair_stats: dict[str, list[float]] = {}      # "ga|gb" -> [n, successes]
        self.gene_obs: dict[str, list[float]] = {}        # gene -> [n, successes]
        self.evals = 0                                    # children evaluated at F0
        self.batches = 0
        self.best_score: float | None = self.init_score if self.ratchet else None
        self.best_experiment: str | None = None
        self.best_solution_path: str | None = None
        self.best_member: str | None = None
        self.curve: list[tuple[float, float]] = []
        self.gains: list[float] = []                      # gain of the best candidate, per batch
        self.progress: list[tuple[float, float]] = []     # (fraction of the run used, best score)
        self.last_improvement_fraction = 0.0
        self._sig_score: float | None = None
        self._n_member = self._n_island = self._n_lineage = self._n_pair = 0

    # ================================================================== helpers
    def _seed(self) -> int:
        return int(self.rng.integers(2**31 - 1))

    def _budget(self, fid: str) -> float:
        return float(self.ctx.strategy[f"fidelity.{fid.lower()}_s"])

    def _new_member(self, **kw) -> Member:
        self._n_member += 1
        m = Member(id=f"m{self._n_member:05d}", **kw)
        self.members[m.id] = m
        return m

    def _new_lineage(self, island: Island) -> Lineage:
        self._n_lineage += 1
        lin = Lineage(f"L{self._n_lineage:03d}")
        island.lineages[lin.id] = lin
        return lin

    def _new_island(self, formulation_text: str, adapter_source: str | None, canonical: bool,
                    evidence: list[str]) -> Island:
        self._n_island += 1
        iid = f"I{self._n_island}"
        gene = self.genes.add(iid, "formulation", formulation_text, "seed" if canonical else "diff", evidence,
                              born_at=self.evals, status="validated") \
            or self.genes.add(iid, "formulation", "Alternative formulation of the problem: " + formulation_text,
                              "diff", evidence, born_at=self.evals, status="validated")
        island = Island(iid, gene.id if gene else "", adapter_source, canonical, self.ctx.governor.fraction_used())
        self.islands[iid] = island
        self.island_arms[iid] = [1.0, 1.0]
        return island

    def _spend(self) -> float:
        g = self.ctx.governor
        return g.spend("main") + g.spend("gate") + g.spend("meta")

    def _phase(self, island: Island) -> tuple[str, dict]:
        st = self.ctx.strategy
        frac = self.ctx.governor.fraction_used()
        rel = (frac - island.created_fraction) / max(1e-9, 1.0 - island.created_fraction)
        inputs = {"budget_fraction": round(rel, 6), "evals_since_improvement": island.since_improvement,
                  "explore_until": float(st["phases.explore_until"]), "window": int(st["phases.window"])}
        return D.phase(**inputs), inputs

    def _ranked(self, island: Island, include_frozen: bool = False) -> list[Member]:
        ms = [self.members[i] for i in island.pool]
        ms = [m for m in ms if m.alive and (include_frozen or not island.lineages[m.lineage].frozen)]
        sgn = D.sign(self.direction)
        if self.portfolio:      # rank with the fit of the batch's regime; members without one come last
            return sorted(ms, key=lambda m: (-(sgn * m.fit_by_regime[self.regime])
                                             if self.regime in m.fit_by_regime else float("inf"), m.id))
        return sorted(ms, key=lambda m: (-(sgn * m.fit) if m.fit is not None else float("inf"), m.id))

    def _add_gene(self, island: Island, type_: str, text: str, origin: str, evidence: list[str],
                  seed: int | None = None, request_hash: str | None = None) -> Gene | None:
        gene = self.genes.add(island.id, type_, text, origin, evidence, born_at=self.evals)
        if gene is not None:
            self.ctx.ledger.event("gene_born", seed=seed, refs=[gene.id, *evidence[:4]], gene=gene.id,
                                  island=island.id, type=gene.type, text=gene.text, origin=origin,
                                  request_hash=request_hash)
        return gene

    # ================================================================== start portfolio (brief sections 4 and 5)
    def _key(self):
        """Key of member scores: the epoch in v1, the id of the batch's start set with the portfolio."""
        return self.cur_set.id if self.portfolio and self.cur_set is not None else self.epoch

    def _submit(self, source: str, budget: float, adapter: str | None) -> list:
        """One sandbox job per start of the batch's start set (v1: one job from the warm start)."""
        starts = self.cur_set.starts if self.portfolio and self.cur_set is not None else (self.init,)
        return [self.ctx.submit(source, budget, self.eval_seed, init=x, adapter_source=adapter) for x in starts]

    def _finish(self, futs: list, category: str) -> Evaluation:
        evs = [self.ctx.finish(f.result(), category) for f in futs]
        if not (self.portfolio and self.cur_set is not None):
            return evs[0]
        sid = self.cur_set.id
        agg = P.aggregate(evs, self.cur_set, self.set_worst[sid], self.direction)
        self.set_worst[sid] = P.update_worst(self.set_worst[sid], evs, self.direction)
        return agg

    def _describe(self, candidate) -> dict:
        from knockout import starts as S

        return S.descriptors(self.ctx.problem.id, candidate)

    def _register_set(self, sset: P.StartSet, seed: int | None, **extra) -> None:
        self.start_sets[sset.regime] = sset
        self.set_worst.setdefault(sset.id, [None] * len(sset))
        self.ctx.ledger.event("start_set", seed=seed, refs=[sset.id], id=sset.id, regime=sset.regime,
                              origins=list(sset.origin), start_scores=list(sset.start_scores), **extra)

    def _init_portfolio(self) -> None:
        """Cold start set (fixed for the run) and the basin archive, whose bin edges are fixed here from the
        descriptors of the cold starts and the incumbent."""
        from knockout import starts as S

        st, pid = self.ctx.strategy, self.ctx.problem.id
        seed = int(st.get("portfolio.cold_seed", 7))
        cands = S.cold_starts(pid, int(st.get("portfolio.k_cold", 2)), seed)
        scores = tuple(P.safe_score(self.ctx.problem, c) for c in cands)
        self.archive = Archive.from_samples([*cands, self.init], self._describe,
                                            int(st.get("portfolio.archive_bins", 4)), self.direction,
                                            float(st["selection.eps_rel"]))
        sset = P.StartSet("cold:0", "cold", tuple(cands), scores, tuple(f"cold:{k}" for k in range(len(cands))))
        self._register_set(sset, seed, archive_keys=self.archive.keys, archive_edges=self.archive.edges)

    def _start_set(self, regime: str) -> P.StartSet:
        st = self.ctx.strategy
        if regime == "record":
            sid = f"record:{self.epoch}"
            cur = self.start_sets.get("record")
            if cur is None or cur.id != sid:
                cur = P.StartSet(sid, "record", (self.init,), (self.init_score,), ("incumbent",))
                self._register_set(cur, None)
            return cur
        if regime == "cold":
            return self.start_sets["cold"]
        # neighbour: redrawn for each batch of the regime, from perturbations of the incumbent and archive entries
        from knockout import starts as S

        seed = self._seed()
        rng = np.random.default_rng(seed)
        base = self.init if self.init is not None else self.start_sets["cold"].starts[0]
        others = self.archive.others(self.init) if self.archive is not None else []
        lo, hi = float(st.get("portfolio.perturb_min", 0.01)), float(st.get("portfolio.perturb_max", 0.3))
        share = float(st.get("portfolio.neighbour_archive_share", 0.5))
        starts, scores, origin = [], [], []
        for _ in range(int(st.get("portfolio.k_neighbour", 2))):
            if others and rng.random() < share:
                e = others[int(rng.integers(len(others)))]
                starts.append(e.candidate)
                scores.append(e.score)
                origin.append(f"archive:{cell_name(e.cell)}")
            else:
                strength = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                cand = S.perturb(self.ctx.problem.id, base, strength, int(rng.integers(2**31 - 1)))
                starts.append(cand)
                scores.append(P.safe_score(self.ctx.problem, cand))
                origin.append(f"perturb:{strength:.4g}")
        self._n_neighbour += 1
        sset = P.StartSet(f"neighbour:{self._n_neighbour}", "neighbour", tuple(starts), tuple(scores), tuple(origin))
        self._register_set(sset, seed, base_score=self.init_score)
        return sset

    def _begin_regime(self) -> None:
        st = self.ctx.strategy
        self.op_arms_r[self.regime] = self.operator_arms      # keeps a strategist shift made between batches
        seed = self._seed()
        floors = {r: float(v) for r, v in (st.get("portfolio.floors") or {}).items() if r in self.regimes}
        inputs = {"arms": {r: list(v) for r, v in self.regime_arms.items()}, "spend": dict(self.regime_spend),
                  "floors": floors}
        out = D.pick_regime(inputs["arms"], inputs["spend"], inputs["floors"], seed)
        self.ctx.ledger.event("regime_pick", seed=seed, refs=[out["choice"]], inputs=inputs, outputs=out)
        self.regime = out["choice"]
        self.operator_arms = self.op_arms_r[self.regime]
        self.cur_set = self._start_set(self.regime)
        self._other_basin = False

    def _end_regime(self, live: list[Spec], spend0: float) -> None:
        r, st = self.regime, self.ctx.strategy
        self.op_arms_r[r] = self.operator_arms
        self.regime_spend[r] += max(0.0, self._spend() - spend0)
        succ = any(s.s for s in live)
        self.regime_arms = D.bandit_update(self.regime_arms, r, (2.0 if self._other_basin else 1.0) if succ else 0.0,
                                           0.0 if succ else 1.0, float(st["bandit_discount"]))
        rs = self.regime_stats[r]
        rs["batches"] += 1
        rs["children"] += len(live)
        rs["beat_parent"] += sum(s.s for s in live)
        sgn = D.sign(self.direction)
        for s in live:
            v = P.selection_value(s.ev) if s.ev is not None and s.ev.valid else None
            if v is not None and (rs["best_regime_score"] is None or sgn * (v - rs["best_regime_score"]) > 0):
                rs["best_regime_score"] = v
                rs["best_progress"] = P.progress(v, P.mean_start_score(self.cur_set), self._reference())

    def _reference(self) -> float | None:
        return self.ctx.best_known()

    def _pcfg(self, ev: Evaluation) -> dict:
        sset = self.cur_set
        d: dict = {"regime": self.regime, "start_set": sset.id if sset else None}
        if isinstance(ev, P.PortfolioEvaluation):
            d.update({"per_start": ev.per_start, "regime_score": ev.regime_score,
                      "progress": P.progress(ev.regime_score, P.mean_start_score(sset), self._reference())})
        if self.archive is not None and ev.valid and ev.candidate is not None:
            d["cell"] = cell_name(self.archive.cell(ev.candidate))
        return d

    def _offer(self, ev: Evaluation, exp_id: str, program_id: str, child: bool) -> list[tuple]:
        """Every valid per-start candidate is offered to the archive (brief 4.5). Returns the cells it entered."""
        if self.archive is None:
            return []
        entered: list[tuple] = []
        evs = ev.per_start_evals if isinstance(ev, P.PortfolioEvaluation) else [ev]
        inc_cell = self.archive.cell(self.init) if self.init is not None else None
        for e in evs:
            if not e.valid or e.candidate is None:
                continue
            out = self.archive.offer(e.candidate, e.score, exp_id, self.regime, program_id)
            if out is None:
                continue
            out["incumbent_cell"] = cell_name(inc_cell) if inc_cell is not None else None
            self.ctx.ledger.event("basin_insert", seed=None, refs=[exp_id], **out)
            entered.append(self.archive.cell(e.candidate))
            if child and inc_cell is not None and self.archive.cell(e.candidate) != inc_cell:
                self._other_basin = True
        return entered

    def _deep_polish(self, island: Island) -> None:
        """Brief 4.6: an archive entry outside the incumbent's cell that comes within `promote_gap` (normalized by
        the distance between the incumbent and the mean cold start) gets one long run of the best record-regime
        program. A result above the incumbent replaces it."""
        ctx, st = self.ctx, self.ctx.strategy
        if self.archive is None or self.init_score is None:
            return
        cold_mean = P.mean_start_score(self.start_sets["cold"])
        scale = abs(self.init_score - cold_mean) if cold_mean is not None else abs(self.init_score)
        gap = float(st.get("portfolio.promote_gap", 0.02))
        sgn = D.sign(self.direction)
        progs = [m for m in self.members.values() if m.alive and "record" in m.fit_by_regime
                 and m.island_id in self.islands]
        if not progs:
            return
        prog = max(progs, key=lambda m: (sgn * m.fit_by_regime["record"], m.id))
        budget = float(st.get("portfolio.deep_factor", 10)) * self._budget("F0")
        done = 0
        for e in self.archive.others(self.init):
            if done >= int(st.get("portfolio.max_deep_per_batch", 1)):
                break
            if e.deep_polished or ctx.problem.gain(self.init_score, e.score) > gap * scale:
                continue
            e.deep_polished = True
            done += 1
            seed = self._seed()
            ev = ctx.finish(ctx.submit(prog.source, budget, seed, init=e.candidate,
                                       adapter_source=self.islands[prog.island_id].adapter_source).result(),
                            self.category)
            exp_id = ctx.ledger.next_experiment_id()
            before = self.init_score
            beats = ev.valid and D.improves(ev.score, before, float(st["selection.eps_rel"]), self.direction)
            rec = ctx.log_experiment(
                experiment_id=exp_id, ev=ev, source=prog.source, program_id=prog.program_id, chain=self.chain,
                agent=self.agent, hypothesis=f"deep polish of basin {cell_name(e.cell)} with {prog.id}",
                rationale=f"archive entry {e.experiment} ({e.score!r}) is within promote_gap={gap:g} of the "
                          f"incumbent ({before!r}), normalized by {scale:.6g}",
                decision="keep" if beats else ("discard" if ev.valid else ev.status),
                status="keep" if beats else ("discard" if ev.valid else ev.status), seed=seed, budget_s=budget,
                best_before=self.best_score, save_solution=True,
                extra_config=self._cfg(island, prog, "deep-polish", "deep", before, ev=ev))
            for c in self._offer(ev, exp_id, prog.program_id, child=False):
                self.archive.cells[c].deep_polished = True     # its own result is not promoted again
            payload = {"entry": {"cell": cell_name(e.cell), "experiment": e.experiment, "score": e.score,
                                 "regime": e.regime}, "program": prog.id, "budget_s": budget,
                       "score_before": e.score, "score_after": ev.score if ev.valid else None,
                       "incumbent_score": before, "new_incumbent_from_other_basin": bool(beats)}
            ctx.ledger.event("basin_promotion", seed=seed, refs=[exp_id, e.experiment], **payload)
            self.promotions.append({**payload, "experiment": exp_id})
            if beats and (self.best_score is None or ctx.problem.gain(ev.score, self.best_score) > 0):
                self._set_best(ev, exp_id, rec, prog)

    def _regime_block(self, spec: Spec) -> str:
        sset = self.cur_set
        lines = [f"Regime: `{self.regime}`."]
        if self.regime == "record":
            lines.append("`construct` is called once, with `init` = the best candidate found so far in this run "
                         "(None if there is none yet).")
        else:
            lines.append(f"`construct` is called once per start ({len(sset)} starts); each start is a canonical "
                         "candidate passed as `init` that may be invalid or poor; return a valid candidate. The "
                         "program's score here is the mean of its verified scores over the starts.")
        for origin, x, sc in zip(sset.origin, sset.starts, sset.start_scores):
            try:
                desc = {k: round(float(v), 4) for k, v in self._describe(x).items()} if x is not None else None
            except Exception:  # noqa: BLE001  (descriptors of an invalid start may not exist)
                desc = None
            lines.append(f"- start {origin}: score {sc!r}, descriptors {desc}")
        val = spec.parent.score("F0", sset.id)
        prog = P.progress(val, P.mean_start_score(sset), self._reference())
        lines.append(f"Parent program on this start set: mean score {val!r}, normalized progress {prog!r}"
                     if val is not None else "The parent program has not been run on this start set yet.")
        lines.append(f"Best normalized progress seen in this regime: {self.regime_stats[self.regime]['best_progress']!r}"
                     " (0 = the mean start, 1 = the reference value).")
        return "\n".join(lines)

    # ================================================================== start
    def start(self) -> Evaluation:
        """Canonical island, baseline evaluation of the seed program, seed genes (one brainstorm call)."""
        ctx = self.ctx
        island = self._new_island(CANONICAL_TEXT, None, True, [])
        lin = self._new_lineage(island)
        if self.portfolio:
            self.cur_set = self._start_set("record")
        ev = ctx.finish(ctx.submit(self.seed_source, self._budget("F0"), self.eval_seed, init=self.init).result(),
                        self.category)
        m = self._new_member(program_id=f"p_{source_hash(self.seed_source)}", source=self.seed_source,
                             island_id=island.id, lineage=lin.id, genome=[], parents=[], operator="seed",
                             model=None, born_at=0, hypothesis="baseline: the seed program as written")
        exp_id = ctx.ledger.next_experiment_id()
        new_best = ev.valid and (self.best_score is None or ctx.problem.gain(ev.score, self.best_score) > 0)
        rec = ctx.log_experiment(
            experiment_id=exp_id, ev=ev, source=m.source, program_id=m.program_id, chain=self.chain,
            agent=self.agent, hypothesis=m.hypothesis, rationale="first evaluation of the main arm's seed program",
            decision="baseline", status="keep" if ev.valid else ev.status, seed=self.eval_seed,
            budget_s=self._budget("F0"), best_before=self.best_score, save_solution=new_best,
            extra_config=self._cfg(island, m, "seed", "F0", self.init_score if self.ratchet else None, ev=ev))
        if ev.valid:
            self._record_score(island, m, "F0", ev, exp_id)
            island.pool.append(m.id)
            island.hashes.add(F.norm_hash(m.source))
            island.tokens.append(F.token_set(m.source))
            island.best_score = ev.score
            if new_best:
                self._set_best(ev, exp_id, rec, m)
        if self.portfolio:
            self._init_portfolio()
            self._offer(ev, exp_id, m.program_id, child=False)
        self._sig_score = self.best_score
        self._end_batch(island, [])
        self._brainstorm(island)
        return ev

    def _brainstorm(self, island: Island) -> None:
        n = int(self.ctx.strategy.get("genes.seed_count", 10))
        seed = self._seed()
        prompt = "\n\n".join([
            "# Problem", self.brief, "# Seed program", "```python\n" + self.seed_source.rstrip() + "\n```",
            "# Your task", f"Propose about {n} candidate genes for this problem. " + GENE_RULES,
            "Each gene will be tested by a controlled experiment (programs written with it versus without it), so "
            "make each one specific enough to change what a programmer would write."])
        data, meta = self.llm.call("seed_genes", "You propose testable search insights for a mathematical "
                                   "optimization problem. Answer in JSON.", prompt, GENES_SCHEMA, seed=seed,
                                   category=self.category, tags={"island": island.id})
        for g in (data or {}).get("genes", [])[: n + 2]:
            self._add_gene(island, g.get("type", "structure"), g.get("text", ""), "seed", [], seed,
                           meta.get("request_hash"))

    # ================================================================== one batch
    def step(self) -> dict:
        ctx = self.ctx
        spend0 = self._spend()
        if self.portfolio:
            self._begin_regime()
        island = self._pick_island()
        phase, phase_inputs = self._phase(island)
        # start portfolio: gene statistics are kept per "<regime>/<phase>" (pooled while a key has few observations)
        gene_key = f"{self.regime}/{phase}" if self.portfolio else phase
        specs = self._plan(island, gene_key, phase_inputs)
        self._write_programs(island, specs)
        self._filter(island, specs)
        live = [s for s in specs if s.source]
        best_before = self.best_score
        batch_evals: list[tuple[Evaluation, str, dict, Member | None]] = []
        ref_scores = self._evaluate_f0(island, live, batch_evals)
        self._account(island, specs, live, ref_scores, batch_evals)
        self._memory_batch(island, live)
        self._gate_results(island, specs)
        self._ladder(island, live, batch_evals)
        improved = self._update_best(island, live, batch_evals, ref_scores)
        self._births(island, live)
        if self.portfolio:
            self._deep_polish(island)
            self._end_regime(live, spend0)
        self._end_batch(island, live, improved)
        valid = [s for s in live if s.ev is not None and s.ev.valid]
        return {"batch": self.batches, "island": island.id, "phase": gene_key, "regime": self.regime, "n": len(specs), "written": len(live),
                "valid": len(valid), "kept": sum(s.s for s in live), "best": self.best_score,
                "new_best": best_before != self.best_score, "n_edits": len(live),
                "genes": {k: sum(g.status == k for g in self.genes.genes.values())
                          for k in ("candidate", "validated", "refuted", "retired")}}

    # ------------------------------------------------------------------ island, operator, parents, genome
    def _pick_island(self) -> Island:
        seed = self._seed()
        weights = self.ctx.strategy.get("islands.budget_shares") or {}
        inputs = {"arms": {k: list(v) for k, v in self.island_arms.items()},
                  "weights": {k: float(v) for k, v in weights.items() if k in self.island_arms}}
        out = D.thompson_pick(inputs["arms"], seed, inputs["weights"])
        self.ctx.ledger.event("island_pick", seed=seed, refs=[out["choice"]], inputs=inputs, outputs=out)
        return self.islands[out["choice"]]

    def _feasible_ops(self, island: Island, gate: bool, used: set[str]) -> list[str]:
        n = len(self._ranked(island))
        ops = ["fresh"]
        if n >= 1:
            ops += ["refine", "restructure"]
        if n >= 2:
            ops.append("hybridize")
        st = self.ctx.strategy
        if not gate and n >= 1 and "reformulate" not in used and (
                len(self.islands) < int(st["islands.max_islands"]) or self._replaceable(island.id) is not None):
            ops.append("reformulate")
        return [op for op in ops if op in self.operator_arms]

    def _replaceable(self, keep: str | None = None) -> str | None:
        """The island a new formulation may replace. `keep` (the island being stepped) is never replaced."""
        items = [{"id": i.id, "alpha": self.island_arms[i.id][0], "beta": self.island_arms[i.id][1],
                  "evals": i.evals, "canonical": i.canonical} for i in self.islands.values() if i.id != keep]
        return D.island_to_replace(items, int(self.ctx.strategy["islands.min_evals_before_replace"]))

    def _draft(self, island: Island, phase: str, phase_inputs: dict, gate: bool, used: set[str],
               exclude_gene: str | None = None) -> Spec:
        ctx, st = self.ctx, self.ctx.strategy
        # operator
        seed = self._seed()
        arms = {op: list(self.operator_arms[op]) for op in self._feasible_ops(island, gate, used)}
        op = D.thompson_pick(arms, seed)
        ctx.ledger.event("operator_pick", seed=seed, refs=[island.id], kind="operator", inputs={"arms": arms},
                         outputs=op)
        operator = op["choice"]
        # model
        model = self.llm.models[0]
        if len(self.model_arms) > 1:
            seed = self._seed()
            arms = {k: list(v) for k, v in self.model_arms.items()}
            pick = D.thompson_pick(arms, seed)
            ctx.ledger.event("operator_pick", seed=seed, refs=[island.id], kind="model", inputs={"arms": arms},
                             outputs=pick)
            model = pick["choice"]
        # parents: A is the program the child is compared with (shown unless the operator is `fresh`)
        ranked = self._ranked(island) or self._ranked(island, include_frozen=True)
        ids = [m.id for m in ranked]
        seed = self._seed()
        inputs = {"ranked_ids": ids, "power": float(st["selection.parent_power"])}
        pa = D.select_parent(ids, inputs["power"], seed)
        ctx.ledger.event("operator_pick", seed=seed, refs=[pa["choice"]], kind="parent", inputs=inputs, outputs=pa)
        parent = self.members[pa["choice"]]
        parent_b = None
        rest = [i for i in ids if i != parent.id]
        if rest:
            seed = self._seed()
            inputs = {"ranked_ids": rest, "power": float(st["selection.parent_power"])}
            pb = D.select_parent(rest, inputs["power"], seed)
            ctx.ledger.event("operator_pick", seed=seed, refs=[pb["choice"]], kind="parent_b", inputs=inputs,
                             outputs=pb)
            parent_b = self.members[pb["choice"]]
        # genome: validated genes of both parents, plus validated genes of the island that have too few
        # observations to have reached a genome yet (otherwise a freshly validated gene could never spread)
        min_obs = int(st["genes.phase_min_obs"])
        ids_pool = set(parent.genome) | set(parent_b.genome if parent_b else [])
        ids_pool |= {g.id for g in self.genes.of_island(island.id, "validated")
                     if sum(v[2] for v in g.stats.values()) < min_obs}
        pool = []
        for gid in sorted(ids_pool):
            g = self.genes.genes.get(gid)
            if g is None or g.status != "validated" or g.type in ("formulation", "negative") or gid == exclude_gene:
                continue
            a, b = g.posterior(phase, min_obs)
            pool.append({"id": gid, "alpha": a, "beta": b, "linked": list(g.linked),
                         "incompatible": list(g.incompatible)})
        seed = self._seed()
        g_max = int(st["genes.g_max"])
        built = D.build_genome(pool, g_max, seed)
        negatives = [g.id for g in self.genes.negatives(island.id, int(st["genes.max_negative"]))]
        ctx.ledger.event("genome_built", seed=seed, refs=[parent.id, *built["genome"]],
                         inputs={"pool": pool, "g_max": g_max, "phase": phase_inputs},
                         outputs={"genome": built["genome"], "draws": built["draws"], "phase": phase.split("/")[-1]},
                         gene_key=phase,
                         negatives=negatives, operator=operator)
        return Spec("normal", operator, model, parent, parent_b if operator == "hybridize" else None,
                    built["genome"], negatives, phase, self._seed(), self.category)

    def _open_gate(self, island: Island) -> Gate | None:
        """The island tests one gene at a time: its current gate, a due re-audit, or its oldest candidate gene."""
        st = self.ctx.strategy
        if island.gate is not None:
            return island.gate
        if island.evals - island.last_audit >= int(st["gate.reaudit_every"]):
            island.last_audit = island.evals
            validated = [g for g in self.genes.of_island(island.id, "validated")
                         if g.type not in ("formulation", "negative") and g.usage > 0]
            if validated:
                g = max(validated, key=lambda g: (g.usage, g.id))
                island.gate = Gate(g.id, audit=True)
                return island.gate
        cands = sorted((g for g in self.genes.of_island(island.id, "candidate") if g.type != "formulation"),
                       key=lambda g: g.id)
        if cands:
            island.gate = Gate(cands[0].id)
        return island.gate

    def _plan(self, island: Island, phase: str, phase_inputs: dict) -> list[Spec]:
        st, gov = self.ctx.strategy, self.ctx.governor
        n = int(st["main.batch_size"])
        specs: list[Spec] = []
        gate = self._open_gate(island)
        if gate is not None and gov.within_cap("gate"):
            for _ in range(int(st["gate.pairs_per_batch"])):
                if len(specs) + 2 > n or gate.pairs + len(specs) // 2 >= int(st["gate.max_pairs"]):
                    break
                base = self._draft(island, phase, phase_inputs, True, set(), exclude_gene=gate.gene_id)
                self._n_pair += 1
                pair = f"pair{self._n_pair:04d}"
                for kind in ("with", "without"):
                    specs.append(Spec(kind, base.operator, base.model, base.parent, base.parent_b, list(base.genome),
                                      list(base.negatives), phase, self._seed(), "gate", gate.gene_id, pair))
        used: set[str] = set()
        while len(specs) < n:
            spec = self._draft(island, phase, phase_inputs, False, used)
            used.add(spec.operator)
            specs.append(spec)
        return specs

    # ------------------------------------------------------------------ prompts and LLM calls
    def _prompt(self, island: Island, spec: Spec) -> str:
        ctx = self.ctx
        positive = [self.genes.genes[g] for g in spec.genome if g in self.genes.genes]
        negative = [self.genes.genes[g] for g in spec.negatives if g in self.genes.genes]
        trial = self.genes.genes.get(spec.trial) if spec.kind == "with" else None
        parts = ["# Problem", self.brief,
                 "# State of the run", state_block(ctx, self.init_score, self.ratchet, self.gains, "batches",
                                                   show_record=self.show_record)]
        if self.portfolio:
            parts += ["# Regime of this batch", self._regime_block(spec)]
        if not island.canonical:
            parts += ["# Formulation of this island", self.genes.genes[island.formulation].text,
                      "`construct` returns the object of this formulation. This adapter (fixed, do not rewrite it) "
                      "maps it to a candidate:", "```python\n" + (island.adapter_source or "").rstrip() + "\n```"]
        parts += ["# Insights", render(positive, negative, trial)]
        if spec.operator != "fresh":
            parts += ["# Parent program", "```python\n" + spec.parent.source.rstrip() + "\n```"]
            notes = island.lineages[spec.parent.lineage].notes[-3:]
            if notes:
                parts += ["# Lines this lineage already deleted (do not bring them back)", "\n".join(notes)]
        if spec.operator == "hybridize" and spec.parent_b is not None:
            parts += ["# Second parent program", "```python\n" + spec.parent_b.source.rstrip() + "\n```"]
        n_hist = int(ctx.strategy.get("llm.history", 24))
        parts += ["# Evidence from this run", history_block(ctx, n_hist, "main", island.id),
                  *memory_section(ctx, "main", self.batches, f"{spec.operator} {getattr(spec.parent, 'hypothesis', '')}"
                                  [:400], n_hist, island.id),
                  "# Your task", f"Operator `{spec.operator}`: {MAIN_OPERATORS[spec.operator]}",
                  "Write one new program that respects the contract and the hard rules."]
        return "\n\n".join(parts)

    def _write_one(self, island: Island, spec: Spec, prompt: str) -> None:
        reform = spec.operator == "reformulate"
        data, meta = self.llm.call("program_reformulate" if reform else "program", SYSTEM, prompt,
                                   SCHEMA_REFORMULATE if reform else SCHEMA, seed=spec.seed, category=spec.category,
                                   model=spec.model, reserved=True,
                                   tags={"operator": spec.operator, "island": island.id, "kind": spec.kind})
        spec.meta = meta
        if data is None:
            return
        spec.source = strip_fence(str(data.get("source", "")))
        spec.hypothesis = str(data.get("hypothesis", "")).strip() or f"{spec.operator} (no hypothesis given)"
        spec.meta["rationale"] = str(data.get("rationale", ""))[:1200]
        if reform:
            spec.formulation = str(data.get("formulation", "")).strip()
            spec.adapter_source = strip_fence(str(data.get("adapter_source", "")))

    def _write_programs(self, island: Island, specs: list[Spec]) -> None:
        """One LLM call per child, in parallel. The worst-case cost of each call is reserved first; a gate pair
        is sent whole or not at all."""
        gov = self.ctx.governor
        prompts = [self._prompt(island, s) for s in specs]
        est = [self.llm.estimate_usd(SYSTEM, p, s.model) for p, s in zip(prompts, specs)]
        send: list[int] = []
        reserved = 0.0
        i = 0
        while i < len(specs):
            group = [i, i + 1] if specs[i].kind == "with" else [i]
            cost = sum(est[j] for j in group)
            if not gov.can_spend_usd("main", reserved + cost):
                self.llm.budget_stopped = True
                break
            reserved += cost
            send += group
            i += len(group)
        if send:
            with cf.ThreadPoolExecutor(max_workers=len(send)) as pool:
                list(pool.map(lambda j: self._write_one(island, specs[j], prompts[j]), send))

    def _filter(self, island: Island, specs: list[Spec]) -> None:
        st, ctx = self.ctx.strategy, self.ctx
        for s in specs:
            if s.source is None:
                continue
            shown = s.parent.source if s.operator != "fresh" else None
            reason = check_program(s.source, shown or "")
            detail: dict = {}
            if reason is None and s.operator == "reformulate":
                reason = _check_adapter(s.adapter_source or "")
            if reason is None:
                lin = island.lineages[s.parent.lineage]
                hit = F.check(s.source, shown, island.hashes, island.tokens, lin.deleted if shown else set(),
                              float(st["filters.jaccard_max"]), float(st["filters.recycled_line_ratio"]))
                if hit:
                    reason, detail = hit["reason"], hit
                    if hit["reason"] == "recycled lines":
                        lin.notes.append("; ".join(hit["examples"])[:600])
            if reason:
                ctx.ledger.event("filter_reject", seed=s.seed, refs=[s.parent.id], reason=reason, detail=detail,
                                 operator=s.operator, island=island.id, kind=s.kind,
                                 request_hash=s.meta.get("request_hash"), hypothesis=s.hypothesis[:400])
                s.source = None

    # ------------------------------------------------------------------ evaluation
    def _cfg(self, island: Island, member: Member | None, operator: str, fid: str, ref: float | None,
             spec: Spec | None = None, ev: Evaluation | None = None) -> dict:
        cfg = {"arm": "main", "island": island.id, "member": member.id if member else None, "operator": operator,
               "fidelity": fid, "epoch": self.epoch, "ref_score": ref, "batch": self.batches}
        if spec is not None:
            cfg.update({"parent_id": spec.parent.id, "parent_b": spec.parent_b.id if spec.parent_b else None,
                        "model": spec.model, "genome": spec.genome, "negatives": spec.negatives,
                        "trial_gene": spec.trial if spec.kind == "with" else None, "gate_pair": spec.pair,
                        "gate_gene": spec.trial, "phase": spec.phase, "llm_request": spec.meta.get("request_hash"),
                        "edit_meta": {k: v for k, v in spec.meta.items() if k not in ("role", "error")}})
        elif member is not None:
            cfg.update({"parent_id": member.parents[0] if member.parents else None, "genome": member.genome})
        if self.portfolio and ev is not None:
            cfg.update(self._pcfg(ev))
        return cfg

    def _record_score(self, island: Island, m: Member, fid: str, ev: Evaluation, exp_id: str) -> None:
        key = self._key()
        value = P.selection_value(ev)
        m.scores[f"{fid}@{key}"] = value
        m.exp_ids[f"{fid}@{key}"] = exp_id
        if fid == "F0":
            relative = self.ratchet and self.init_score is not None and (not self.portfolio or self.regime == "record")
            m.fit = self.ctx.problem.gain(value, self.init_score) if relative else value
            if self.portfolio:
                m.fit_by_regime[self.regime] = m.fit

    def _evaluate_f0(self, island: Island, live: list[Spec], batch_evals: list) -> dict[str, float | None]:
        """Children and the parents they are compared with, all from the same warm start. Returns the F0 score
        of each comparison parent at the current epoch (None if its replay failed)."""
        ctx = self.ctx
        f0 = self._budget("F0")
        key = self._key()
        need = {s.parent.id: s.parent for s in live if s.parent.score("F0", key) is None}
        fut_parents = [(m, self._submit(m.source, f0, self.islands[m.island_id].adapter_source))
                       for m in need.values()]
        fut_children = [(s, self._submit(s.source, f0, s.adapter_source or island.adapter_source)) for s in live]
        for m, fut in fut_parents:
            ev = self._finish(fut, self.category)
            exp_id = ctx.ledger.next_experiment_id()
            save = ev.valid and self._beats_batch(ev, batch_evals)
            rec = ctx.log_experiment(
                experiment_id=exp_id, ev=ev, source=m.source, program_id=m.program_id, chain=self.chain,
                agent=self.agent, hypothesis=f"replay of {m.id} from the current warm start",
                rationale=f"reference for its children in batch {self.batches} (same warm start, same fidelity)",
                decision="baseline" if ev.valid else ev.status, status="keep" if ev.valid else ev.status,
                seed=self.eval_seed, budget_s=f0, best_before=self.best_score, save_solution=save,
                extra_config=self._cfg(self.islands[m.island_id], m, "replay", "F0", self.init_score, ev=ev))
            if self.portfolio:
                self._offer(ev, exp_id, m.program_id, child=False)
            if ev.valid:
                self._record_score(island, m, "F0", ev, exp_id)
                batch_evals.append((ev, exp_id, rec, m))
            elif sum(self.members[i].alive for i in self.islands[m.island_id].pool) > 1:
                m.alive = False       # a program that no longer runs from the current warm start is not a parent
        for s, fut in fut_children:
            s.ev = self._finish(fut, s.category)
        return {s.parent.id: s.parent.score("F0", key) for s in live}

    def _beats_batch(self, ev: Evaluation, batch_evals: list) -> bool:
        """True when `ev` is above the arm's best and above everything already evaluated in this batch."""
        gain = self.ctx.problem.gain
        top = max((e.score for e, *_ in batch_evals), key=lambda v: D.sign(self.direction) * v, default=None)
        return (self.best_score is None or gain(ev.score, self.best_score) > 0) and \
               (top is None or gain(ev.score, top) > 0)

    def _account(self, island: Island, specs: list[Spec], live: list[Spec], ref_scores: dict,
                 batch_evals: list) -> None:
        """Log every child, add the valid ones to the population, then credit genes, operators and models."""
        ctx, st = self.ctx, self.ctx.strategy
        eps = float(st["selection.eps_rel"])
        gamma = float(st["genes.gamma"])
        for s in live:
            ev = s.ev
            ref = s.ref = ref_scores.get(s.parent.id)
            value = P.selection_value(ev)
            s.s = int(ev.valid and D.improves(value, ref, eps, self.direction))
            what = "regime score" if self.portfolio else "tuned score"
            s.exp_id = ctx.ledger.next_experiment_id()
            if not ev.valid:
                decision = status = ev.status
                rationale = f"no valid candidate: {ev.error}"
            elif s.s:
                decision = status = "keep"
                rationale = f"{what} {value:.15g} beats its parent {s.parent.id} ({ref}) by more than " \
                            f"eps_rel={eps:g} at F0, same start set"
            else:
                decision = status = "discard"
                rationale = f"{what} {value:.15g} does not beat its parent {s.parent.id} ({ref}) at F0"
            target = island
            if ev.valid and s.operator == "reformulate":
                target = self._open_island(island, s)
                if target is None:
                    rationale += "; valid, but its formulation already has an island or no island slot was free"
            if ev.valid and target is not None:
                shown = s.operator != "fresh"
                lin = target.lineages[s.parent.lineage] if shown and target is island else self._new_lineage(target)
                s.member = self._new_member(
                    program_id=f"p_{source_hash(s.source)}", source=s.source, island_id=target.id, lineage=lin.id,
                    genome=sorted(set(s.genome) | ({s.trial} if s.kind == "with" and s.trial else set())),
                    parents=[s.parent.id] + ([s.parent_b.id] if s.parent_b else []), operator=s.operator,
                    model=s.model, born_at=self.evals, hypothesis=s.hypothesis)
                if shown:
                    lin.deleted |= F.diff_lines(s.parent.source, s.source)[1]
            save = ev.valid and self._beats_batch(ev, batch_evals)
            rec = ctx.log_experiment(
                experiment_id=s.exp_id, ev=ev, source=s.source, program_id=f"p_{source_hash(s.source)}",
                chain=self.chain, agent=self.agent, hypothesis=s.hypothesis, rationale=rationale, decision=decision,
                status=status, seed=self.eval_seed, budget_s=self._budget("F0"), best_before=self.best_score,
                save_solution=save, extra_config=self._cfg(target or island, s.member, s.operator, "F0", ref, s, ev=ev))
            if self.portfolio:
                self._offer(ev, s.exp_id, f"p_{source_hash(s.source)}", child=True)
            self.evals += 1
            island.evals += 1
            if ev.claim == "numerical_artifact" and ctx.memory is not None:
                ctx.memory.write_fact("refuted", f"Numerical artifact: {s.hypothesis[:200]}",
                                      f"Float score {ev.score!r} above the record was not confirmed by the independent "
                                      "verifier.", arm="main", island=island.id, operator=s.operator, model=s.model,
                                      experiment_ids=[s.exp_id], score=ev.score)
            if ev.claim == "numerical_artifact":
                self._add_gene(island, "negative", "Numerical artifact: a float score above the record that the "
                               "independent verifier did not confirm. " + s.hypothesis[:200], "diff", [s.exp_id])
                self._mark_negative(island, s.exp_id)
            if s.member is not None:
                self._record_score(target, s.member, "F0", ev, s.exp_id)
                target.pool.append(s.member.id)
                target.hashes.add(F.norm_hash(s.source))
                target.tokens.append(F.token_set(s.source))
                batch_evals.append((ev, s.exp_id, rec, s.member))
            self._credit(island, s, ref, eps, gamma)
            lin = island.lineages[s.parent.lineage]
            lin.since_improvement = 0 if s.s else lin.since_improvement + 1
        # operator and model bandits: one update per program asked from the LLM. A child that tops the batch
        # counts double; a call that gave no evaluable program (API error, filter reject) is a failure.
        top = max((s for s in live if s.s), key=lambda s: D.sign(self.direction) * s.ev.score, default=None)
        disc = float(st["bandit_discount"])
        for s in specs:
            if not s.meta:      # never sent: the USD cap was reached
                continue
            succ = (2.0 if s is top else 1.0) if s.s else 0.0
            self.operator_arms = D.bandit_update(self.operator_arms, s.operator, succ, 0.0 if s.s else 1.0, disc)
            if len(self.model_arms) > 1:
                self.model_arms = D.bandit_update(self.model_arms, s.model, succ, 0.0 if s.s else 1.0, disc)
        self._trim(island)

    def _memory_batch(self, island: Island, live: list[Spec]) -> None:
        """Shared memory (knockout/memory.py): one `win` fact per child that beat its parent, one aggregated
        `failed_batch` fact for the others. Not one fact per child: 2 to 6 writes per batch."""
        mem = self.ctx.memory
        if mem is None or not mem.enabled:
            return
        sign, failed = D.sign(self.direction), []
        for s in live:
            ev = s.ev
            gain = sign * (ev.score - s.ref) if ev is not None and ev.valid and s.ref is not None else None
            mem.note_child("main", s.operator, bool(s.s))
            if s.s:
                added, removed = F.diff_lines(s.parent.source, s.source)
                mem.write_fact("win", f"{s.operator}: {s.hypothesis[:300]}",
                               f"Beat its parent by {gain:+.3e} at the same warm start (+{len(added)} / "
                               f"-{len(removed)} lines).", arm="main", island=island.id, operator=s.operator,
                               model=s.model, experiment_ids=[s.exp_id], gain=gain, ref_score=s.ref, score=ev.score)
            else:
                failed.append((s, gain))
        if failed and mem.write_failed:
            lines = [f"- {s.operator}: {s.hypothesis[:140]} ("
                     + ("invalid" if g is None else f"gain {g:+.2e}") + ")" for s, g in failed[:8]]
            mem.write_fact("failed_batch", f"Batch {self.batches} ({island.id}): {len(failed)} edits did not beat "
                           "their parent", "\n".join(lines), arm="main", island=island.id,
                           experiment_ids=[s.exp_id for s, _ in failed if s.exp_id],
                           gain=max((g for _, g in failed if g is not None), default=None))

    def _mark_negative(self, island: Island, exp_id: str) -> None:
        for g in self.genes.of_island(island.id):
            if g.type == "negative" and exp_id in g.evidence:
                g.status = "validated"

    def _credit(self, island: Island, s: Spec, ref: float | None, eps: float, gamma: float) -> None:
        """Section 5.3: the child's genome gets s with weight 1, the genomes of its ancestors at depth d = 1..3
        get the same update with weight 0.5 ** d (clade credit)."""
        st = self.ctx.strategy
        updates = []
        genome = sorted(set(s.genome))
        levels = [(0, genome)]
        anc = s.parent
        for d in range(1, int(st["genes.clade_depth"]) + 1):
            if anc is None:
                break
            levels.append((d, sorted(anc.genome)))
            anc = self.members.get(anc.parents[0]) if anc.parents else None
        for d, ids in levels:
            w = float(st["genes.clade_decay"]) ** d
            for gid in ids:
                g = self.genes.genes.get(gid)
                if g is None:
                    continue
                a, b, n = g.stats.get(s.phase, [0.0, 0.0, 0])
                a2, b2 = D.credit(a, b, float(s.s), gamma, w)
                g.stats[s.phase] = [a2, b2, n + (1 if d == 0 else 0)]
                updates.append({"gene": gid, "depth": d, "weight": w, "before": [a, b], "after": [a2, b2]})
        for gid in genome:
            g = self.genes.genes.get(gid)
            if g is not None:
                g.usage += 1
                o = self.gene_obs.setdefault(gid, [0.0, 0.0])
                o[0] += 1
                o[1] += s.s
        self._linkage(genome, s)
        score = P.selection_value(s.ev) if s.ev.valid else None    # the value `s` was computed from
        self.ctx.ledger.event(
            "credit_update", seed=None, refs=[s.exp_id, s.parent.id],
            inputs={"child_score": score, "parent_score": ref, "eps_rel": eps, "direction": self.direction,
                    "gamma": gamma, "updates": [{k: u[k] for k in ("gene", "depth", "weight", "before")}
                                                for u in updates]},
            outputs={"s": s.s, "after": [u["after"] for u in updates]}, phase=s.phase)

    def _linkage(self, genome: list[str], s: Spec) -> None:
        st = self.ctx.strategy
        for i, ga in enumerate(genome):
            for gb in genome[i + 1:]:
                key = f"{ga}|{gb}"
                p = self.pair_stats.setdefault(key, [0.0, 0.0])
                p[0] += 1
                p[1] += s.s
                oa, ob = self.gene_obs.get(ga, [0, 0]), self.gene_obs.get(gb, [0, 0])
                if not oa[0] or not ob[0]:
                    continue
                inputs = {"n_joint": int(p[0]), "joint_rate": p[1] / p[0], "rate_a": oa[1] / oa[0],
                          "rate_b": ob[1] / ob[0], "min_cooccurrence": int(st["linkage.min_cooccurrence"]),
                          "margin": float(st["linkage.margin"])}
                verdict = D.linkage(**inputs)
                a, b = self.genes.genes[ga], self.genes.genes[gb]
                current = "linked" if gb in a.linked else "incompatible" if gb in a.incompatible else None
                if verdict == current:
                    continue
                for x, y in ((a, b), (b, a)):
                    x.linked = [i for i in x.linked if i != y.id] + ([y.id] if verdict == "linked" else [])
                    x.incompatible = [i for i in x.incompatible if i != y.id] + \
                        ([y.id] if verdict == "incompatible" else [])
                self.ctx.ledger.event("linkage", seed=None, refs=[ga, gb], inputs=inputs, outputs={"verdict": verdict})

    def _trim(self, island: Island) -> None:
        cap = int(self.ctx.strategy.get("main.population_max", 60))
        ranked = self._ranked(island, include_frozen=True)
        for m in ranked[cap:]:
            m.alive = False
        island.pool = [i for i in island.pool if self.members[i].alive]

    def _open_island(self, source_island: Island, s: Spec) -> Island | None:
        """Section 5.7: a valid child of `reformulate` opens an island, replacing the weakest one when full."""
        st, ctx = self.ctx.strategy, self.ctx
        text = s.formulation or s.hypothesis
        if any(i.formulation in self.genes.genes and
               similarity(self.genes.genes[i.formulation].text, text) >= self.genes.dedupe_similarity
               for i in self.islands.values()):
            return None                     # this formulation already has an island
        if len(self.islands) >= int(st["islands.max_islands"]):
            victim = self._replaceable(source_island.id)
            if victim is None:
                return None
            for mid in self.islands[victim].pool:
                self.members[mid].alive = False
            del self.islands[victim], self.island_arms[victim]
            ctx.ledger.event("island_replaced", seed=None, refs=[victim], island=victim,
                             reason="lowest posterior among islands with enough evaluations")
        island = self._new_island(text, s.adapter_source, False, [])
        ctx.ledger.event("island_opened", seed=s.seed, refs=[island.id, source_island.id], island=island.id,
                         formulation=self.genes.genes[island.formulation].text if island.formulation else "",
                         request_hash=s.meta.get("request_hash"))
        return island

    # ------------------------------------------------------------------ knock-out gate (5.2)
    def _gate_results(self, island: Island, specs: list[Spec]) -> None:
        gate, st, ctx = island.gate, self.ctx.strategy, self.ctx
        if gate is None:
            return
        pairs: dict[str, dict[str, Spec]] = {}
        for s in specs:
            if s.pair:
                pairs.setdefault(s.pair, {})[s.kind] = s
        cfg = {"p_validate": float(st["gate.p_validate"]), "p_refute": float(st["gate.p_refute"]),
               "min_pairs": int(st["gate.min_pairs"]), "max_pairs": int(st["gate.max_pairs"])}
        gene = self.genes.genes[gate.gene_id]
        for pid in sorted(pairs):
            w, wo = pairs[pid].get("with"), pairs[pid].get("without")
            if (w is None or w.source is None) and (wo is None or wo.source is None):
                continue                      # neither program was written (budget): the pair did not happen

            def score(s: Spec | None) -> float | None:
                return P.selection_value(s.ev) if s is not None and s.ev is not None and s.ev.valid else None

            inputs = {"score_with": score(w), "score_without": score(wo),
                      "eps_rel": float(st["selection.eps_rel"]), "direction": self.direction,
                      "invalid_is_tie": bool(st.get("gate.invalid_is_tie", True))}
            outcome = D.pair_outcome(**inputs)
            gate.pairs += 1
            gate.wins += outcome == "win"
            gate.losses += outcome == "loss"
            refs = [x.exp_id for x in (w, wo) if x is not None and x.exp_id]
            gate.evidence += refs
            ctx.ledger.event("gate_pair", seed=None, refs=[gate.gene_id, *refs], gene=gate.gene_id, pair=pid,
                             audit=gate.audit, inputs=inputs, outputs={"outcome": outcome})
            if gate.pairs < cfg["min_pairs"]:
                continue
            inputs = {"wins": gate.wins, "losses": gate.losses, "pairs": gate.pairs, **cfg}
            out = D.gate_verdict(**inputs)
            ctx.ledger.event("gate_verdict", seed=None, refs=[gate.gene_id], gene=gate.gene_id, audit=gate.audit,
                             inputs=inputs, outputs=out)
            if out["verdict"] is None:
                continue
            gene.evidence += [e for e in gate.evidence if e not in gene.evidence]
            if gate.audit:      # a re-audit can only demote: inconclusive keeps the gene validated
                gene.status = "refuted" if out["verdict"] == "refuted" else "validated"
            else:
                gene.status = out["verdict"]
            if gene.status == "refuted" and ctx.memory is not None:
                ctx.memory.write_fact("refuted", f"Insight refuted at the knock-out gate: {gene.text[:300]}",
                                      f"{gate.wins} wins / {gate.losses} losses in {gate.pairs} pairs.", arm="main",
                                      island=island.id, gene=gene.id, experiment_ids=gate.evidence[-8:],
                                      evidence={"wins": gate.wins, "losses": gate.losses, "pairs": gate.pairs})
            island.gate = None
            break

    # ------------------------------------------------------------------ fidelity ladder (5.4)
    def _ladder(self, island: Island, live: list[Spec], batch_evals: list) -> None:
        st, ctx = self.ctx.strategy, self.ctx
        if self.portfolio and self.regime != "record":
            return      # the ladder re-runs a program from the incumbent: it belongs to the record regime only
        levels = int(st.get("fidelity.levels", 3))
        current = {s.exp_id: s for s in live if s.member is not None}
        scores = {k: s.ev.score for k, s in current.items()}
        for fid in FIDELITIES[1:levels]:
            if not scores:
                return
            inputs = {"scores": scores, "ratio": float(st["fidelity.promote_ratio"]), "direction": self.direction}
            promoted = D.promote(**inputs)
            ctx.ledger.event("fidelity_promotion", seed=None, refs=promoted, to=fid, inputs=inputs,
                             outputs={"promoted": promoted})
            budget = self._budget(fid)
            futs = [(k, ctx.submit(current[k].source, budget, self.eval_seed, init=self.init,
                                   adapter_source=self.islands[current[k].member.island_id].adapter_source))
                    for k in promoted]
            nxt, nxt_scores = {}, {}
            for k, fut in futs:
                s = current[k]
                ev = ctx.finish(fut.result(), s.category)
                exp_id = ctx.ledger.next_experiment_id()
                save = ev.valid and self._beats_batch(ev, batch_evals)
                rec = ctx.log_experiment(
                    experiment_id=exp_id, ev=ev, source=s.source, program_id=s.member.program_id, chain=self.chain,
                    agent=self.agent, hypothesis=s.hypothesis,
                    rationale=f"promoted to {fid}: in the top {inputs['ratio']:.2f} of its batch at the previous "
                              f"fidelity (experiment {k})",
                    decision="keep" if ev.valid else ev.status, status="keep" if ev.valid else ev.status,
                    seed=self.eval_seed, budget_s=budget, best_before=self.best_score, save_solution=save,
                    extra_config=self._cfg(self.islands[s.member.island_id], s.member, s.operator, fid, None, s,
                                           ev=ev))
                if self.portfolio:
                    self._offer(ev, exp_id, s.member.program_id, child=False)
                if ev.valid:
                    self._record_score(island, s.member, fid, ev, exp_id)
                    batch_evals.append((ev, exp_id, rec, s.member))
                    nxt[exp_id], nxt_scores[exp_id] = s, ev.score
            current, scores = nxt, nxt_scores

    # ------------------------------------------------------------------ best candidate, warm start
    def _set_best(self, ev: Evaluation, exp_id: str, rec: dict, member: Member | None) -> None:
        before = self.best_score
        self.best_score, self.best_experiment = ev.score, exp_id
        self.best_solution_path = rec.get("solution_path")
        self.best_member = member.id if member else None
        if self.ratchet:
            self.init, self.init_score = ev.candidate, ev.score
            self.epoch += 1
            self.ctx.ledger.event("warm_start_update", seed=None, refs=[exp_id], arm="main", epoch=self.epoch,
                                  score_before=before, score_after=ev.score, solution_path=self.best_solution_path,
                                  regime=self.regime if self.portfolio else None)

    def _update_best(self, island: Island, live: list[Spec], batch_evals: list, ref_scores: dict) -> bool:
        """Returns True when the island improved: a child beat the island's reference (its best score, or, with
        the ratchet, the best replay of the batch)."""
        ctx = self.ctx
        gain = ctx.problem.gain
        kids = [s.ev.score for s in live if s.member is not None]
        refs = [v for v in ref_scores.values() if v is not None]
        if self.ratchet:
            reference = max(refs, key=lambda v: D.sign(self.direction) * v) if refs else self.init_score
        else:
            reference = island.best_score
        best_kid = max(kids, key=lambda v: D.sign(self.direction) * v) if kids else None
        improved = best_kid is not None and D.improves(best_kid, reference, float(ctx.strategy["selection.eps_rel"]),
                                                       self.direction)
        if self.portfolio:      # a regime's batch improves when a child beat its parent on the same start set
            improved = any(s.s for s in live)
        if best_kid is not None and (island.best_score is None or gain(best_kid, island.best_score) > 0):
            island.best_score = best_kid
        before = self.best_score
        top = None
        for item in batch_evals:
            if top is None or gain(item[0].score, top[0].score) > 0:
                top = item
        if top is not None and (self.best_score is None or gain(top[0].score, self.best_score) > 0):
            self._set_best(top[0], top[1], top[2], top[3])     # stored at logging time (see _beats_batch)
        self.gains.append(gain(self.best_score, before) if before is not None and self.best_score is not None else 0.0)
        return improved

    # ------------------------------------------------------------------ gene birth from diffs (5.1)
    def _births(self, island: Island, live: list[Spec]) -> None:
        st = self.ctx.strategy
        room = int(st.get("genes.max_candidates", 12)) - len(self.genes.of_island(island.id, "candidate"))
        winners = sorted((s for s in live if s.s and s.member is not None and s.operator != "fresh"),
                         key=lambda s: -D.sign(self.direction) * s.ev.score)[: int(st.get("genes.births_per_batch", 2))]
        if room <= 0 or not winners:
            return

        def review(s: Spec):
            diff = "\n".join(difflib.unified_diff(s.parent.source.splitlines(), s.source.splitlines(), "parent",
                                                  "child", lineterm="", n=2))[:12000]
            prompt = "\n\n".join([
                "# Problem", self.brief[:3000], "# Parent program", "```python\n" + s.parent.source.rstrip() + "\n```",
                "# Diff from parent to child", "```diff\n" + diff + "\n```",
                f"# Scores (frozen verifier, same warm start, same budget)\nparent: "
                f"{s.ref!r}\nchild: {P.selection_value(s.ev)!r}",
                "# Your task", "The child beat its parent. State the reusable idea behind the gain as 0 to 2 genes. "
                + GENE_RULES + " Return no gene if the gain looks like tuning noise."])
            seed = self._seed()
            data, meta = self.llm.call("review_diff", "You review a code change in a mathematical optimization "
                                       "program and extract reusable insights. Answer in JSON.", prompt,
                                       GENES_SCHEMA, seed=seed, category=self.category, tags={"island": island.id})
            return s, seed, data, meta

        with cf.ThreadPoolExecutor(max_workers=len(winners)) as pool:
            results = list(pool.map(review, winners))
        for s, seed, data, meta in results:
            for g in (data or {}).get("genes", [])[:2]:
                self._add_gene(island, g.get("type", "structure"), g.get("text", ""), "diff", [s.exp_id], seed,
                               meta.get("request_hash"))

    # ------------------------------------------------------------------ end of batch
    def _end_batch(self, island: Island, live: list[Spec], improved: bool = False) -> None:
        ctx, st = self.ctx, self.ctx.strategy
        if live:
            # island bandit: success when a child beat the island's reference; a new best of the arm counts double
            new_arm_best = bool(self.gains and self.gains[-1] > 0) and improved
            self.island_arms = D.bandit_update(self.island_arms, island.id, (2.0 if new_arm_best else 1.0)
                                               if improved else 0.0, 0.0 if improved else 1.0,
                                               float(st["bandit_discount"]))
            island.since_improvement = 0 if improved else island.since_improvement + len(live)
            self._freeze(island)
            self._migrate(island)
            self.batches += 1
        frac = ctx.governor.fraction_used()
        if self.best_score is not None:
            self.curve.append((self._spend(), self.best_score))
            self.progress.append((frac, self.best_score))
            min_gain = float(st.get("strategist.min_progress", 0.0))
            if self._sig_score is None or ctx.problem.gain(self.best_score, self._sig_score) > min_gain:
                self._sig_score, self.last_improvement_fraction = self.best_score, frac
        self.save_state()

    def _freeze(self, island: Island) -> None:
        """Section 5.8: a lineage without improvement for W evaluations gets one F2 tuning of its best program,
        then no more LLM calls."""
        ctx, st = self.ctx, self.ctx.strategy
        window = int(st["phases.window"])
        for lin in list(island.lineages.values()):
            inputs = {"evals_since_improvement": lin.since_improvement, "window": window}
            if lin.frozen or not D.should_freeze(**inputs):
                continue
            members = [m for m in self._ranked(island, include_frozen=True) if m.lineage == lin.id]
            lin.frozen = True
            refs = []
            if members and int(st.get("fidelity.levels", 3)) >= 3:
                m = members[0]
                ev = ctx.finish(ctx.submit(m.source, self._budget("F2"), self.eval_seed, init=self.init,
                                           adapter_source=island.adapter_source).result(), self.category)
                exp_id = ctx.ledger.next_experiment_id()
                new_best = ev.valid and (self.best_score is None or ctx.problem.gain(ev.score, self.best_score) > 0)
                rec = ctx.log_experiment(
                    experiment_id=exp_id, ev=ev, source=m.source, program_id=m.program_id, chain=self.chain,
                    agent=self.agent, hypothesis=f"final F2 tuning of {m.id}, best program of frozen lineage {lin.id}",
                    rationale=f"lineage {lin.id} had no improvement for {lin.since_improvement} evaluations",
                    decision="keep" if ev.valid else ev.status, status="keep" if ev.valid else ev.status,
                    seed=self.eval_seed, budget_s=self._budget("F2"), best_before=self.best_score,
                    save_solution=new_best, extra_config=self._cfg(island, m, "freeze-tune", "F2", None, ev=ev))
                refs = [exp_id]
                if ev.valid:
                    self._record_score(island, m, "F2", ev, exp_id)
                    if new_best:
                        self._set_best(ev, exp_id, rec, m)
            ctx.ledger.event("lineage_frozen", seed=None, refs=[lin.id, *refs], lineage=lin.id, island=island.id,
                             inputs=inputs, outputs={"freeze": True})

    def _migrate(self, island: Island) -> None:
        """Section 5.7: validated non-formulation genes with a high posterior are copied to the other islands as
        candidates, where they must pass the gate again. Programs never migrate."""
        st = self.ctx.strategy
        if len(self.islands) < 2 or island.evals - island.last_migration < int(st["islands.migrate_every"]):
            return
        island.last_migration = island.evals
        items = []
        for g in self.genes.of_island(island.id):
            a = 1.0 + sum(v[0] for v in g.stats.values())
            b = 1.0 + sum(v[1] for v in g.stats.values())
            items.append({"id": g.id, "type": g.type, "status": g.status, "alpha": a, "beta": b})
        inputs = {"genes": items, "min_posterior": float(st["islands.migrate_min_posterior"])}
        out = D.migrants(**inputs)
        copies = []
        for gid in out:
            g = self.genes.genes[gid]
            for other in self.islands.values():
                if other.id != island.id:
                    new = self._add_gene(other, g.type, g.text, "migration", [gid])
                    if new is not None:
                        copies.append(new.id)
        self.ctx.ledger.event("migration", seed=None, refs=[island.id, *out], inputs=inputs,
                              outputs={"migrants": out}, copies=copies)

    # ------------------------------------------------------------------ state for the report
    def save_state(self) -> None:
        state = {
            "genes": self.genes.dump(),
            "members": [{k: v for k, v in asdict(m).items() if k != "source"} for m in self.members.values()],
            "islands": [{"id": i.id, "formulation": i.formulation, "canonical": i.canonical, "evals": i.evals,
                         "best_score": i.best_score, "pool": len(i.pool), "arm": self.island_arms.get(i.id),
                         "gate": asdict(i.gate) if i.gate else None,
                         "lineages": {k: {"frozen": v.frozen, "since_improvement": v.since_improvement}
                                      for k, v in i.lineages.items()}} for i in self.islands.values()],
            "operator_arms": self.operator_arms, "model_arms": self.model_arms, "evals": self.evals,
            "batches": self.batches, "epoch": self.epoch, "best_score": self.best_score,
            "best_experiment": self.best_experiment, "best_member": self.best_member,
            "best_solution_path": self.best_solution_path, "curve": self.curve,
        }
        if self.portfolio:
            state["portfolio"] = {
                "regime_arms": self.regime_arms, "regime_spend": self.regime_spend, "regime_stats": self.regime_stats,
                "operator_arms_by_regime": self.op_arms_r,
                "start_sets": {r: {"id": v.id, "origin": list(v.origin), "start_scores": list(v.start_scores)}
                               for r, v in self.start_sets.items()},
                "archive": self.archive.table() if self.archive else [],
                "archive_edges": self.archive.edges if self.archive else {},
                "promotions": self.promotions}
        (self.ctx.ledger.dir / "state.json").write_text(json.dumps(state, indent=1))


def _check_adapter(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"adapter syntax error: {exc}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "to_canonical" for n in tree.body):
        return "adapter has no top-level to_canonical() function"
    return None
