"""Islands and migration (L4).

An island is an independent lineage: its own gene pool, population, best certificate and RNG. Islands
never share their contexts. After every round they sync ONLY through the shared stores:
  - validated genes: read from knock-out records (arm "knockout", status "keep") written by other
    islands, and imported as validated genes with origin "migration:<island>";
  - best certificates: the best certificate in the shared certificate store (re-verified on load)
    replaces an island's parent when it is better.
Because migration only reads the stores, islands can run in separate processes or Modal apps.
"""
from __future__ import annotations

import re

import numpy as np

from .genes import Gene, GenePool, seed_genes

KO_TAG = re.compile(r"knockout=([^;]+)")


class Island:
    def __init__(self, island_id: str, problem, base_optimizer: str, base_config: dict, space, root, seed: int):
        self.id, self.problem = island_id, problem
        self.base_optimizer, self.base_config, self.space = base_optimizer, base_config, space
        self.rng = np.random.default_rng(seed)
        self.pool = GenePool()
        for g in seed_genes(space, base_optimizer, problem.id, root):
            self.pool.add(g)
        self.population: list = []          # [(genome, fitness)], kept sorted, top POPULATION_SIZE
        self.tested: set = set()
        self.tried: set = set()
        self.pending_knockouts: list = []   # [(gene, with_genome, without_genome)]
        self.evidence: list = []            # literature (Exa) shown to the LLM editor; data, never instructions
        self.best_cert: str | None = None
        self.best_score: float | None = None
        self.hill_radius = 0.15
        self.stalled_rounds = 0             # rounds since this island's best certificate last improved
        self.next_seed = 1_000_000 + 10_000 * seed   # fresh seeds for cold restarts (never the island's seeds)

    def fresh_seeds(self, n: int) -> list[int]:
        out = list(range(self.next_seed, self.next_seed + n))
        self.next_seed += n
        return out

    def adopt(self, cert) -> bool:
        if self.best_score is None or self.problem.better(cert.score, self.best_score):
            self.best_cert, self.best_score = cert.id, cert.score
            return True
        return False

    def schedule_knockout(self, genome) -> None:
        for g in genome:
            if self.pool.state.get(g.id) == "candidate" and all(k[0].id != g.id for k in self.pending_knockouts):
                self.pending_knockouts.append((g, list(genome), [x for x in genome if x.id != g.id]))
                return


def migrate(islands: list[Island], records: list[dict], certs, problem) -> dict:
    """Sync islands through the stores. Returns what moved, for the round report."""
    moved = {"genes": [], "certificate": None}
    for r in records:
        if r.get("arm") != "knockout" or r.get("status") != "keep" or r.get("problem") != problem.id:
            continue
        m = KO_TAG.search(r.get("rationale") or "")
        if not m or "side=with;" not in (r.get("rationale") or ""):   # only the gate verdict counts
            continue
        gid = m.group(1)
        spec = next((d for d in r.get("genes", []) if d.get("id") == gid), None)
        if spec is None:
            continue
        for isl in islands:
            if isl.id == r.get("island"):
                continue
            gene = Gene(spec["kind"], spec["name"], spec["params"], f"migration:{r.get('island')}")
            if gene.id not in isl.pool.genes:
                isl.pool.add(gene, state="validated")
                isl.pool.notes[gene.id] = f"validated on island {r.get('island')}"
                moved["genes"].append((r.get("island"), isl.id, gene.id))
            elif isl.pool.state[gene.id] == "candidate":        # known but untested here: upgrade
                isl.pool.validate(gene.id, f"validated on island {r.get('island')}")
                moved["genes"].append((r.get("island"), isl.id, gene.id))
    best = certs.best(problem)
    if best is not None:
        for isl in islands:
            if isl.adopt(best):
                moved["certificate"] = best.id
    return moved
