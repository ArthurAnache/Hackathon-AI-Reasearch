"""Arms: competing research strategies. The meta-bandit splits the budget between them (L4).

Every arm turns an island's state into Specs (plan), and learns from their evaluations (learn).
All arms start from the island's best certificate, use the same seeds and the same budget per
candidate, and are judged by the same rule (real_margin vs the parent), so they can be compared.

  hill_sampling   CONTROL. Refine the best certificate with the base optimizer: one pure refinement,
                  plus configs sampled in a trust region around the base config (an inner tuner,
                  no LLM). The radius grows after a keep and shrinks after a miss.
  gene_evolution  Survival of the fittest over genomes. First every gene is tested on its own,
                  then children are bred by Thompson crossover of tournament-selected parents, plus
                  mutation (nudge / add / drop / LLM structural edit). Only the top genomes survive.
  formulation     Representation changes (e.g. resolution upscaling with exact refinement), each
                  tested on its own. This targets gaps that tuning cannot close.
  fresh_seeds     EXPLORATION. Cold restarts from new seeds (no warm start), carrying the island's
                  validated genes. WS3's 88-chain Modal campaign on 1a showed that the seed matters
                  most (same config: 1.508 to 1.521), that the median chain does not improve, and
                  that new bests come from fresh seeds rather than from extending the best chain:
                  progress is best-of-N. This arm is judged on its BEST seed (a new verified best
                  is a real improvement), not on its mean.
"""
from __future__ import annotations

import json
import math

import numpy as np

from .evaluate import Spec
from .genes import compatible, express
from .operators import crossover, mutate, tournament

POPULATION_SIZE = 6


class Arm:
    name = "arm"
    judge = "mean"          # mean: a method is better when its mean beats the parent by a real margin
                            # best: a lottery arm is better when its best seed beats the parent

    def plan(self, island, n: int) -> list[Spec]:
        raise NotImplementedError

    def learn(self, island, results: list[tuple]) -> None:
        """results: [(evaluation, significant, gain)]"""


def _spec(island, arm: str, genome, hypothesis: str) -> Spec:
    optimizer, config = express(genome, island.base_optimizer, island.base_config)
    return Spec(optimizer, config, island.best_cert, arm, island.id, [g.to_dict() for g in genome],
                hypothesis, f"arm={arm}; parent={island.best_cert}", island.best_score)


class HillSampling(Arm):
    name = "hill_sampling"

    def plan(self, island, n):
        specs = [Spec(island.base_optimizer, dict(island.base_config), island.best_cert, self.name, island.id, [],
                      "Refine the best certificate with the base optimizer (control).",
                      f"arm={self.name}; parent={island.best_cert}", island.best_score)]
        x0 = island.space.to_unit(island.base_config)
        for _ in range(n - 1):
            x = np.clip(x0 + island.rng.uniform(-island.hill_radius, island.hill_radius, island.space.dim), 0, 1)
            cfg = island.space.from_unit(x, island.base_config)
            changed = {k: cfg[k] for k in island.space.names if cfg[k] != island.base_config.get(k)}
            specs.append(Spec(island.base_optimizer, cfg, island.best_cert, self.name, island.id, [],
                              f"Refine with a nearby config {changed} (trust radius {island.hill_radius:.2f}).",
                              f"arm={self.name}; parent={island.best_cert}", island.best_score))
        return specs[:n]

    def learn(self, island, results):
        for _, significant, _ in results:
            island.hill_radius = min(0.5, island.hill_radius * 1.5) if significant else max(0.02, island.hill_radius * 0.7)


class GeneEvolution(Arm):
    name = "gene_evolution"
    kinds = ("operator", "config", "formulation")

    def __init__(self, p_mutate: float = 0.5, editor=None):
        self.p_mutate, self.editor = p_mutate, editor

    def _untested(self, island):
        todo = [g for g in island.pool.alive(self.kinds) if g.id not in island.tested and island.pool.state[g.id] == "candidate"]
        return sorted(todo, key=lambda g: -island.pool.mean(g.id) - (g.origin.startswith(("exa:", "past.dev:"))))
        # evidence first: genes other agents validated (favourable prior) or suggested, then the seed genes

    def plan(self, island, n):
        specs, seen = [], set(island.tried)
        for g in self._untested(island)[:n]:                      # every method on its own first
            specs.append(_spec(island, self.name, [g], f"Test {g.id} on its own."))
            seen.add((g.id,))
        tries = 0
        while len(specs) < n and island.population and tries < 20 * n:
            tries += 1
            a, b = tournament(island.population, island.rng), tournament(island.population, island.rng)
            child = crossover(a, b, island.pool, island.rng)
            child, how = mutate(child, island.pool, island.space, island.rng, self.p_mutate, self.editor,
                                {"problem": island.problem, "base_optimizer": island.base_optimizer,
                                 "best_score": island.best_score, "evidence": island.evidence,
                                 "stalled_rounds": island.stalled_rounds})
            key = tuple(sorted(g.id for g in child))
            if not child or key in seen:
                continue
            seen.add(key)
            spec = _spec(island, self.name, child, f"Child of {[g.id for g in a]} x {[g.id for g in b]} ({how}).")
            if how.startswith("llm structural edit") and self.editor is not None:
                spec.llm = self.editor.last_call
            specs.append(spec)
        return specs

    def learn(self, island, results):
        for ev, significant, gain in results:
            genes = [island.pool.genes[d["id"]] for d in ev.spec.genes if d["id"] in island.pool.genes]
            island.tried.add(tuple(sorted(g.id for g in genes)))
            for g in genes:
                island.tested.add(g.id)
                island.pool.credit(g.id, significant, 1.0 / max(1, len(genes)))
            fitness = gain if ev.valid else -math.inf
            if genes:
                island.population.append((genes, fitness))
            if significant:
                island.schedule_knockout(genes)
        island.population.sort(key=lambda p: -p[1])
        del island.population[POPULATION_SIZE:]                     # survival of the fittest


class Formulation(GeneEvolution):
    name = "formulation"
    kinds = ("formulation",)

    def plan(self, island, n):
        out = []
        for g in island.pool.alive(self.kinds):
            if len(out) >= n:
                break
            key = (g.id,)
            if key in island.tried and island.pool.state[g.id] != "validated":
                continue
            out.append(_spec(island, self.name, [g], f"Change the representation: {g.id} {json.loads(g.params)}."))
        return out

    def learn(self, island, results):
        for ev, significant, gain in results:
            for d in ev.spec.genes:
                if d["id"] in island.pool.genes:
                    island.tried.add((d["id"],))
                    island.pool.credit(d["id"], significant)
                    if significant:
                        island.schedule_knockout([island.pool.genes[d["id"]]])


class FreshSeeds(Arm):
    name = "fresh_seeds"
    judge = "best"

    def plan(self, island, n):
        genome = []
        for g in island.pool.validated():                       # what has been proven, kept compatible
            if compatible(genome + [g]):
                genome.append(g)
        optimizer, config = express(genome, island.base_optimizer, island.base_config)
        label = [g.id for g in genome] or "the base config"
        return [Spec(optimizer, config, None, self.name, island.id, [g.to_dict() for g in genome],
                     f"Cold restart from fresh seeds with {label}: a new basin may beat the refined lineage.",
                     f"arm={self.name}; parent=none", island.best_score, island.fresh_seeds(len(island.seeds)))
                for _ in range(n)]
