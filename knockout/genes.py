"""Insight genes (section 4): small typed statements whose status is local to an island and earned at the gate.

Statistics are kept per search phase as discounted pseudo-counts [a, b, n]; the posterior is Beta(1 + a, 1 + b)
(knockout.decisions.gene_posterior). A candidate gene is used only inside gate experiments.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from knockout import decisions as D

GENE_TYPES = ("formulation", "structure", "numeric", "negative")
MAX_WORDS = 60


class Gene(BaseModel):
    id: str
    type: Literal["formulation", "structure", "numeric", "negative"]
    text: str
    status: Literal["candidate", "validated", "refuted", "retired"] = "candidate"
    origin: Literal["seed", "diff", "literature", "strategist", "migration"]
    island_id: str
    evidence: list[str] = Field(default_factory=list)            # experiment ids and URLs
    stats: dict[str, list[float]] = Field(default_factory=dict)  # phase -> [a, b, n]
    linked: list[str] = Field(default_factory=list)
    incompatible: list[str] = Field(default_factory=list)
    usage: int = 0                                               # genomes this gene was part of
    born_at: int = 0                                             # main-arm evaluation count at birth

    def posterior(self, phase: str, phase_min_obs: int) -> list[float]:
        return D.gene_posterior(self.stats, phase, phase_min_obs)

    def mean(self) -> float:
        a = 1.0 + sum(v[0] for v in self.stats.values())
        b = 1.0 + sum(v[1] for v in self.stats.values())
        return a / (a + b)


def clean_text(text: str) -> str:
    words = " ".join(str(text).split()).split(" ")
    return " ".join(words[:MAX_WORDS])


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / max(1, len(ta | tb))


class GeneStore:
    """All genes of a run, keyed by id. A gene belongs to one island; a migrated gene is a new gene there."""

    def __init__(self, dedupe_similarity: float = 0.8):
        self.genes: dict[str, Gene] = {}
        self.dedupe_similarity = dedupe_similarity
        self._n = 0

    def of_island(self, island_id: str, status: str | None = None) -> list[Gene]:
        return [g for g in self.genes.values() if g.island_id == island_id and (status is None or g.status == status)]

    def duplicate_of(self, island_id: str, text: str) -> Gene | None:
        for g in self.of_island(island_id):
            if similarity(g.text, text) >= self.dedupe_similarity:
                return g
        return None

    def add(self, island_id: str, type_: str, text: str, origin: str, evidence: list[str] | None = None,
            born_at: int = 0, status: str = "candidate") -> Gene | None:
        """Add a gene unless the island already has a near-duplicate. Returns the new gene, or None."""
        text = clean_text(text)
        if type_ not in GENE_TYPES or len(text.split()) < 3 or self.duplicate_of(island_id, text):
            return None
        self._n += 1
        gene = Gene(id=f"g{self._n:04d}", type=type_, text=text, origin=origin, island_id=island_id,
                    evidence=list(evidence or []), born_at=born_at, status=status)
        self.genes[gene.id] = gene
        return gene

    def negatives(self, island_id: str, k: int) -> list[Gene]:
        """The island's most recent refuted genes and artifact warnings."""
        neg = [g for g in self.of_island(island_id) if g.status == "refuted"
               or (g.type == "negative" and g.status == "validated")]
        return sorted(neg, key=lambda g: g.id)[-k:]

    def dump(self) -> list[dict]:
        return [g.model_dump() for g in sorted(self.genes.values(), key=lambda g: g.id)]


def render(positive: list[Gene], negative: list[Gene], trial: Gene | None = None) -> str:
    """The part of a prompt that carries a genome."""
    parts = []
    pos = list(positive) + ([trial] if trial is not None else [])
    if pos:
        parts.append("Apply these tested insights:\n" + "\n".join(f"- ({g.type}) {g.text}" for g in pos))
    if negative:
        parts.append("Already refuted, do not retry:\n" + "\n".join(
            f"- {g.text} [evidence: {', '.join(g.evidence[-4:]) or 'gate'}]" for g in negative))
    return "\n\n".join(parts) or "No tested insight yet."
