"""Variation operators: crossover, mutation, knock-out (L3).

crossover   Take the union of two parents' genes and keep each gene with probability equal to a
            Thompson sample from its Beta belief. Genes that have helped tend to survive, and
            uncertain ones still get tried. Conflicts (two values for one knob) go to the higher sample.
mutate      With probability p: nudge a config value (log-normal), add a random living gene,
            drop a gene, or ask the LLM for a structural edit (a new operator gene).
knockout    "Does this gene earn its place?" Evaluate the genome with and without the gene,
            from the same parent certificate, on the same seeds and budget. Pass gives validated;
            fail gives graveyard.
"""
from __future__ import annotations

import math

from .genes import Gene, compatible


def crossover(a: list[Gene], b: list[Gene], pool, rng) -> list[Gene]:
    union = {g.id: g for g in [*a, *b] if pool.state.get(g.id) != "graveyard"}
    draws = {gid: pool.thompson(gid, rng) if gid in pool.ab else 0.5 for gid in union}
    child = []
    for gid in sorted(union, key=lambda i: -draws[i]):            # strongest belief first wins conflicts
        g = union[gid]
        if rng.random() < draws[gid] and compatible([*child, g]):
            child.append(g)
    return child


def mutate(genome: list[Gene], pool, space, rng, p_mutate: float = 0.5, editor=None, context=None) -> tuple[list[Gene], str]:
    """Returns (new genome, description). With probability 1 - p_mutate the genome is unchanged.

    When the editor says it wants to act (the island has stalled long enough, the round cap and the
    budget allow it), the mutation is an LLM structural edit; otherwise a free random operator."""
    wants = getattr(editor, "wants", None)
    if editor is not None and wants is not None and wants((context or {}).get("stalled_rounds")):
        op = "llm"
    elif rng.random() >= p_mutate:
        return list(genome), "no mutation"
    else:
        ops = ["nudge", "add", "drop"] + (["llm"] if editor is not None and wants is None else [])
        op = ops[int(rng.integers(len(ops)))]
    genome = list(genome)
    if op == "nudge":
        config_genes = [g for g in genome if g.kind == "config"]
        if config_genes:
            g = config_genes[int(rng.integers(len(config_genes)))]
            k, v = next(iter(g.p.items()))
            knob = space.knobs.get(k)
            if knob is not None:
                u = min(max(knob.to_unit(v) + float(rng.normal(0, 0.08)), 0.0), 1.0)
                nv = knob.from_unit(u)
                if nv != v:
                    new = Gene.make("config", f"{k}={nv}", origin="mutation", **{k: nv})
                    pool.add(new)
                    return [x for x in genome if x.id != g.id] + [new], f"nudge {k} {v} -> {nv}"
        op = "add"
    if op == "add":
        options = [g for g in pool.alive() if g.id not in {x.id for x in genome} and compatible([*genome, g])]
        if options:
            g = options[int(rng.integers(len(options)))]
            return genome + [g], f"add {g.id}"
        op = "drop"
    if op == "drop" and genome:
        g = genome[int(rng.integers(len(genome)))]
        return [x for x in genome if x.id != g.id], f"drop {g.id}"
    if op == "llm" and editor is not None:
        gene = editor.propose(genome=genome, pool=pool, **(context or {}))
        if gene is not None:
            pool.add(gene)
            return [x for x in genome if x.kind != "operator"] + [gene], f"llm structural edit {gene.id}"
        return mutate(genome, pool, space, rng, p_mutate, None, context)    # no edit: a free mutation instead
    return genome, "no applicable mutation"


def knockout_plan(genome: list[Gene], pool, limit: int = 1) -> list[tuple[Gene, list[Gene]]]:
    """(gene under test, genome without it) for unvalidated genes of a winning genome."""
    out = []
    for g in genome:
        if pool.state.get(g.id) == "candidate":
            out.append((g, [x for x in genome if x.id != g.id]))
        if len(out) >= limit:
            break
    return out


def tournament(population: list[tuple[list[Gene], float]], rng, k: int = 2):
    picks = [population[int(rng.integers(len(population)))] for _ in range(min(k, len(population)))]
    return max(picks, key=lambda p: p[1] if not math.isnan(p[1]) else -math.inf)[0]
