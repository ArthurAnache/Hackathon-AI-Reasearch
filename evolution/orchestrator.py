"""L4 ORCHESTRATION: meta-bandit over arms, islands, migration, one shared store.

Each round:
  1. the meta-bandit splits every island's candidate slots between arms (Thompson sampling on each
     arm's keep rate; the control arm always gets at least one slot),
  2. arms plan specs, plus pending knock-out pairs (with / without the gene, same parent),
  3. ALL specs of ALL islands go to the executor in one batch: N workers x >= 3 seeds,
  4. every candidate is re-verified; method decisions use real_margin; better solutions become
     certificates and advance the island's lineage,
  5. arms learn, the bandit is credited, every evaluation is appended to the shared store,
  6. islands migrate validated genes and best certificates through the stores.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import interfaces as I
from certificates import CertificateStore
from execution import source_hash
from research import Space

from .arms import FreshSeeds, Formulation, GeneEvolution, HillSampling
from .evaluate import Spec, evaluate_batch, real_margin, utility
from .genes import express
from .islands import Island, migrate
from .tools import ResearchTools, cite

MIN_SEEDS = 3


NEW_BEST_RTOL = 1e-7     # a new certificate counts as an arm success only above this relative gain


class MetaBandit:
    """Thompson sampling over arms. Reserved arms (the control, and exploration) always get one slot."""

    def __init__(self, arms: list[str], control: str = "hill_sampling", seed: int = 0,
                 reserved: tuple = ("hill_sampling", "fresh_seeds")):
        self.ab = {a: [1.0, 1.0] for a in arms}
        self.control, self.rng = control, np.random.default_rng(seed)
        self.reserved = [a for a in dict.fromkeys((control, *reserved)) if a in self.ab]
        self.spent = {a: 0 for a in arms}

    def allocate(self, slots: int) -> dict[str, int]:
        alloc = {a: 0 for a in self.ab}
        if slots <= 0:
            return alloc
        for a in self.reserved[:slots]:
            alloc[a] = 1
        for _ in range(slots - min(slots, len(self.reserved))):
            draws = {a: self.rng.beta(*self.ab[a]) for a in self.ab}
            alloc[max(draws, key=draws.get)] += 1
        return alloc

    def credit(self, arm: str, success: bool) -> None:
        if arm in self.ab:
            self.ab[arm][0 if success else 1] += 1
            self.spent[arm] += 1

    def summary(self) -> dict:
        return {a: {"p_keep": round(x / (x + y), 2), "candidates": self.spent[a]} for a, (x, y) in self.ab.items()}


@dataclass
class RoundReport:
    round: int
    allocation: dict
    results: list = field(default_factory=list)      # (island, arm, status, mean, best, significant)
    migrated: dict = field(default_factory=dict)
    best: dict = field(default_factory=dict)          # island -> (cert, score)
    tools: dict = field(default_factory=dict)         # what Exa / past.dev added this round


class Orchestrator:
    def __init__(self, problem_id: str, optimizer: str, executor, store=None, certs=None, islands: int = 2,
                 seeds: int = 3, budget_s: float = 10.0, candidates: int = 4, seed: int = 0, editor=None,
                 arms=None, tools: ResearchTools | None = None):
        if seeds < MIN_SEEDS:
            raise ValueError(f"fair comparisons need at least {MIN_SEEDS} seeds per candidate")
        self.problem = I.get_problem(problem_id)
        self.optimizer = optimizer
        opt = I.FileOptimizer(I.ROOT / optimizer)
        base = {k: v for k, v in opt.default_config().items() if k != "INIT"}
        space = Space.from_optimizer(opt)
        self.executor, self.store = executor, store or I.JsonlStore()
        self.certs = certs or CertificateStore()
        self.seeds, self.budget_s, self.candidates = list(range(seed, seed + seeds)), float(budget_s), candidates
        self.arms = arms or [HillSampling(), GeneEvolution(editor=editor), Formulation(), FreshSeeds()]
        self.by_name = {a.name: a for a in self.arms}
        self.bandit = MetaBandit([a.name for a in self.arms], seed=seed)
        self.islands = [Island(f"island{i}", self.problem, optimizer, base, space, I.ROOT, seed + 100 * i)
                        for i in range(islands)]
        for i, isl in enumerate(self.islands):   # same seeds within an island (fair), different across (diverse)
            isl.seeds = [s + 1000 * i for s in self.seeds]
        best = self.certs.best(self.problem)          # resume from the best verified certificate, if any
        if best is not None:
            for isl in self.islands:
                isl.adopt(best)
        self.tools = tools or ResearchTools()
        self.round = 0

    # ------------------------------------------------------------------ one round
    def plan(self) -> tuple[list[Spec], dict]:
        specs, alloc_all = [], {}
        for isl in self.islands:
            if isl.best_cert is None:                     # no parent yet: one cold-start baseline
                specs.append(Spec(self.optimizer, dict(isl.base_config), None, "hill_sampling", isl.id, [],
                                  "Cold-start baseline (no certificate yet).", "arm=hill_sampling; parent=none"))
                alloc_all[isl.id] = {"hill_sampling": 1}
                continue
            alloc = self.bandit.allocate(self.candidates)
            alloc_all[isl.id] = alloc
            for arm in self.arms:
                if alloc.get(arm.name):
                    specs += arm.plan(isl, alloc[arm.name])
            while isl.pending_knockouts:
                gene, with_g, without_g = isl.pending_knockouts.pop(0)
                for tag, genome in (("with", with_g), ("without", without_g)):
                    opt, cfg = express(genome, isl.base_optimizer, isl.base_config)
                    specs.append(Spec(opt, cfg, isl.best_cert, "knockout", isl.id, [g.to_dict() for g in genome],
                                      f"Knock-out {tag} {gene.id}: does it earn its place?",
                                      f"knockout={gene.id}; side={tag}; parent={isl.best_cert}", isl.best_score))
        for s in specs:
            if s.seeds is None:                    # fresh_seeds brings its own
                s.seeds = next(i.seeds for i in self.islands if i.id == s.island)
        return specs, alloc_all

    def run_round(self) -> RoundReport:
        self.round += 1
        before = {i.id: i.best_score for i in self.islands}
        editor = getattr(self.by_name.get("gene_evolution"), "editor", None)
        if editor is not None:
            editor.new_round(self.round)
        tools_in = self.tools.before_round(self)
        specs, alloc = self.plan()
        report = RoundReport(self.round, alloc, tools=tools_in)
        evals = evaluate_batch(self.problem, self.executor, specs, self.seeds, self.budget_s, self.certs)
        islands = {i.id: i for i in self.islands}
        per_arm: dict = {}
        knock: dict = {}
        records = []
        start = len(self.store.read())
        for n, ev in enumerate(evals):
            isl, spec = islands[ev.spec.island], ev.spec
            significant, gain = False, -math.inf
            judge = getattr(self.by_name.get(spec.arm), "judge", "mean")
            if ev.valid and spec.parent_score is not None:
                if judge == "best":              # best-of-N: did any seed beat the parent?
                    gain = utility(self.problem, ev.best_score) - utility(self.problem, spec.parent_score)
                    significant = gain > NEW_BEST_RTOL * max(1.0, abs(spec.parent_score))
                else:
                    significant, gain, _ = real_margin(self.problem, ev.scores, [spec.parent_score])
            cert_id = None
            parent_best = isl.best_score
            if ev.valid and (isl.best_score is None or self.problem.better(ev.best_score, isl.best_score)):
                cert = self.certs.put(self.problem, ev.best_candidate, ev.best_score,
                                      {"arm": spec.arm, "island": isl.id, "round": self.round})
                cert_id = cert.id
                isl.adopt(cert)
            status = ("keep" if (significant or spec.parent_score is None) else "discard") if ev.valid else ev.status
            if spec.arm == "knockout":
                knock.setdefault((isl.id, spec.rationale.split(";")[0]), {})[spec.rationale.split("side=")[1].split(";")[0]] = (ev, len(records))
            else:
                per_arm.setdefault((isl.id, spec.arm), []).append((ev, significant, gain))
                new_best = cert_id is not None and parent_best is not None and (
                    utility(self.problem, ev.best_score) - utility(self.problem, parent_best)
                    > NEW_BEST_RTOL * max(1.0, abs(parent_best)))
                self.bandit.credit(spec.arm, significant or new_best)   # progress is best-of-N
            records.append(I.new_record(
                experiment_id=f"evo_{start + n:05d}", problem=self.problem.id, agent=self.tools.tag, arm=spec.arm,
                island=isl.id, worker_id=",".join(sorted(set(ev.worker_ids))), optimizer=spec.optimizer,
                optimizer_hash=source_hash((I.ROOT / spec.optimizer).read_text()), source_sha=I.source_sha(),
                parent_id=spec.warm_start, genes=spec.genes, hypothesis=spec.hypothesis, rationale=spec.rationale,
                evidence=cite(spec.genes),
                config=spec.config, warm_start=spec.warm_start, seeds=ev.seeds, budget_s=self.budget_s,
                scores=ev.scores, score=ev.mean, valid=ev.valid, status=status, best_before=spec.parent_score,
                beats_record=bool(ev.valid and self.problem.beats_record(ev.best_score)),
                duration_s=ev.duration_s, error=ev.error, certificate_path=cert_id,
                llm_cost_usd=(spec.llm or {}).get("cost_usd", 0.0), llm_usage=spec.llm))
        for (iid, arm), res in per_arm.items():
            self.by_name[arm].learn(islands[iid], res)
        for (iid, tag), sides in knock.items():                 # the knock-out gate
            if "with" not in sides or "without" not in sides:
                continue
            gene_id = tag.split("=", 1)[1]
            (w, wi), (wo, _) = sides["with"], sides["without"]
            pool = islands[iid].pool
            if w.valid and wo.valid:
                passed, gain, need = real_margin(self.problem, w.scores, wo.scores)
            else:
                passed, gain, need = False, -math.inf, math.inf
            if gene_id not in pool.genes:
                continue
            pool.credit(gene_id, passed, 2.0)
            if passed:
                pool.validate(gene_id, f"knock-out passed: +{gain:.3g} > {need:.3g}")
            else:
                pool.bury(gene_id, f"knock-out failed: gain {gain:.3g} <= margin {need:.3g}")
            records[wi]["status"] = "keep" if passed else "discard"   # 'with' side record = the gate verdict
        for r, ev in zip(records, evals):                        # report AFTER the knock-out verdicts
            self.store.append(r)
            report.results.append((r["island"], r["arm"], r["status"], r["score"], ev.best_score,
                                   r["status"] == "keep" and r["best_before"] is not None))
        report.migrated = migrate(self.islands, self.store.read(), self.certs, self.problem)
        report.best = {i.id: (i.best_cert, i.best_score) for i in self.islands}
        improved = any(i.best_score != before[i.id] for i in self.islands)
        for i in self.islands:
            i.stalled_rounds = 0 if i.best_score != before[i.id] else i.stalled_rounds + 1
        if editor is not None:
            report.tools["llm"] = {"calls": editor.calls_this_round, "spent_usd": round(editor.ledger.spent, 4),
                                   "budget_usd": editor.ledger.budget_usd, "skipped": dict(editor.skipped)}
        report.tools["shared"] = self.tools.after_round(self, report, improved)
        return report

    def run(self, rounds: int, on_round=None) -> list[RoundReport]:
        reports = []
        for _ in range(rounds):
            reports.append(self.run_round())
            if on_round:
                on_round(self, reports[-1])
        return reports
