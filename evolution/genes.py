"""Genes and the gene pool (L3).

A gene is one heritable, testable change to how a run is made:
  config       set knob values                      e.g. config:LR=0.003
  formulation  change the problem's representation  e.g. formulation:upscale_x2 (N_CELLS x2, exact refinement)
  operator     use another optimizer file           e.g. operator:1a_tuned (or an LLM-written variant)

A genome is a set of genes (at most one operator gene, one value per knob). Expressing a genome on a
base optimizer gives the (optimizer, config) a job runs.

The pool keeps, per gene, a Beta(alpha, beta) belief that the gene helps (used for Thompson-sampled
crossover) and a state: candidate -> validated (passed the knock-out gate) or graveyard (failed it,
or kept losing). Graveyard genes are never proposed again and are shown to the LLM as "do not retry".
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

KINDS = ("operator", "config", "formulation")


@dataclass(frozen=True)
class Gene:
    kind: str
    name: str
    params: str            # canonical JSON, keeps the gene hashable
    origin: str = "seed"   # seed | mutation | crossover | llm | migration:<island>

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.name}"

    @property
    def p(self) -> dict:
        return json.loads(self.params)

    @classmethod
    def make(cls, kind: str, name: str, origin: str = "seed", **params) -> "Gene":
        assert kind in KINDS, kind
        return cls(kind, name, json.dumps(params, sort_keys=True), origin)

    def to_dict(self) -> dict:
        return asdict(self) | {"id": self.id}

    @classmethod
    def from_dict(cls, d: dict) -> "Gene":
        return cls(d["kind"], d["name"], d["params"], d.get("origin", "seed"))

    def keys(self) -> set:
        """What this gene controls (two genes that control the same thing cannot coexist)."""
        if self.kind == "operator":
            return {"__operator__"}
        if self.kind == "formulation":
            return {f"__formulation__{self.p['knob']}"}
        return set(self.p)


def compatible(genes) -> bool:
    seen = set()
    for g in genes:
        if seen & g.keys():
            return False
        seen |= g.keys()
    return True


def express(genes, base_optimizer: str, base_config: dict) -> tuple[str, dict]:
    """(optimizer path, config) for a genome. Order: operator, then config, then formulation."""
    optimizer, config = base_optimizer, dict(base_config)
    for g in sorted(genes, key=lambda g: KINDS.index(g.kind)):
        if g.kind == "operator":
            optimizer = g.p["optimizer"]
        elif g.kind == "config":
            config.update(g.p)
        else:
            knob, factor = g.p["knob"], g.p["factor"]
            v = config.get(knob, base_config.get(knob))
            config[knob] = int(round(v * factor)) if isinstance(v, int) else v * factor
    return optimizer, config


class GenePool:
    def __init__(self):
        self.genes: dict[str, Gene] = {}
        self.ab: dict[str, list] = {}           # Beta(alpha, beta) per gene
        self.state: dict[str, str] = {}
        self.notes: dict[str, str] = {}

    def add(self, gene: Gene, state: str = "candidate") -> bool:
        if gene.id in self.genes:
            return False
        self.genes[gene.id], self.ab[gene.id], self.state[gene.id] = gene, [1.0, 1.0], state
        return True

    def alive(self, kinds=KINDS) -> list[Gene]:
        return [g for i, g in self.genes.items() if self.state[i] != "graveyard" and g.kind in kinds]

    def validated(self) -> list[Gene]:
        return [g for i, g in self.genes.items() if self.state[i] == "validated"]

    def graveyard(self) -> list[tuple[Gene, str]]:
        return [(g, self.notes.get(i, "")) for i, g in self.genes.items() if self.state[i] == "graveyard"]

    def thompson(self, gene_id: str, rng) -> float:
        a, b = self.ab[gene_id]
        return float(rng.beta(a, b))

    def mean(self, gene_id: str) -> float:
        a, b = self.ab[gene_id]
        return a / (a + b)

    def credit(self, gene_id: str, success: bool, weight: float = 1.0) -> None:
        self.ab[gene_id][0 if success else 1] += weight
        a, b = self.ab[gene_id]
        if self.state[gene_id] == "candidate" and a + b >= 6 and a / (a + b) < 0.2:
            self.bury(gene_id, f"kept losing (Beta mean {a / (a + b):.2f} after {a + b - 2:.0f} trials)")

    def validate(self, gene_id: str, note: str) -> None:
        self.state[gene_id], self.notes[gene_id] = "validated", note

    def bury(self, gene_id: str, note: str) -> None:
        self.state[gene_id], self.notes[gene_id] = "graveyard", note

    def summary(self) -> dict:
        return {i: {"state": self.state[i], "p_helps": round(self.mean(i), 2), "note": self.notes.get(i, "")}
                for i in self.genes}


def seed_genes(space, optimizer_path: str, problem_id: str, root: Path) -> list[Gene]:
    """The initial 'bunch of methods' tested separately: knob moves, formulation changes, other optimizers."""
    genes = []
    for k, knob in space.knobs.items():
        if k in ("N_CELLS",):
            continue                       # resolution is handled by formulation genes
        base = space.defaults[k]
        for name, f in (("x2", 2.0), ("/2", 0.5)):
            v = base * f
            v = max(int(knob.low), int(round(v))) if knob.integer else float(f"{v:.6g}")
            if knob.low <= v <= knob.high and v != base:
                genes.append(Gene.make("config", f"{k}={v}", **{k: v}))
    if "N_CELLS" in space.knobs:
        for f in (2, 3):
            genes.append(Gene.make("formulation", f"upscale_x{f}", knob="N_CELLS", factor=f))
    for other in sorted((root / "optimizers").glob(f"{problem_id}_*.py")):
        rel = str(other.relative_to(root))
        if rel != optimizer_path:
            genes.append(Gene.make("operator", other.stem, optimizer=rel))
    return genes
