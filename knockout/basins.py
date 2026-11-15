"""Basin archive of the start portfolio (docs/knockout_portfolio_brief.md, section 4.5). Trusted code.

A basin cell is a discretization of trusted descriptors of a candidate (`knockout.starts.descriptors`), with
`bins` bins per descriptor whose edges are fixed at run start from the descriptors of the cold starts and the
incumbent. Each cell keeps its best valid candidate. Candidates of the problems this runs on (circles) are small,
so the archive keeps the object in memory, plus the experiment id that produced it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from knockout import decisions as D


@dataclass
class Entry:
    cell: tuple
    score: float
    experiment: str
    regime: str
    program: str
    candidate: Any
    deep_polished: bool = False


def cell_name(cell: tuple) -> str:
    return "c" + "-".join(str(int(c)) for c in cell)


class Archive:
    def __init__(self, keys: list[str], edges: dict[str, list[float]], describe: Callable[[Any], dict],
                 direction: str, eps_rel: float):
        self.keys, self.edges, self.describe = keys, edges, describe
        self.direction, self.eps_rel = direction, eps_rel
        self.cells: dict[tuple, Entry] = {}

    @classmethod
    def from_samples(cls, samples: list[Any], describe: Callable[[Any], dict], bins: int, direction: str,
                     eps_rel: float) -> "Archive":
        """Inner bin edges per descriptor: `bins` equal-width bins between the min and the max over `samples`."""
        descs = [describe(c) for c in samples if c is not None]
        keys = sorted(descs[0]) if descs else []
        edges = {}
        for k in keys:
            vals = [float(d[k]) for d in descs]
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-12:
                lo, hi = lo - 0.5 * max(1e-9, abs(lo)), hi + 0.5 * max(1e-9, abs(hi))
            edges[k] = [float(x) for x in np.linspace(lo, hi, bins + 1)[1:-1]]
        return cls(keys, edges, describe, direction, eps_rel)

    def cell(self, candidate: Any) -> tuple:
        d = self.describe(candidate)
        return tuple(int(np.searchsorted(self.edges[k], float(d[k]), side="right")) for k in self.keys)

    def offer(self, candidate: Any, score: float, experiment: str, regime: str, program: str) -> dict | None:
        """Insert when the cell is empty or the candidate strictly improves on the occupant. Returns the event
        payload of the insertion, or None."""
        cell = self.cell(candidate)
        old = self.cells.get(cell)
        if old is not None and not D.improves(score, old.score, self.eps_rel, self.direction):
            return None
        self.cells[cell] = Entry(cell, score, experiment, regime, program, candidate)
        return {"cell": cell_name(cell), "experiment": experiment, "score": score, "regime": regime,
                "program": program, "previous": None if old is None else
                {"score": old.score, "experiment": old.experiment}, "eps_rel": self.eps_rel,
                "direction": self.direction}

    def others(self, incumbent: Any) -> list[Entry]:
        """Occupied cells other than the incumbent's, in a fixed order."""
        inc = self.cell(incumbent) if incumbent is not None else None
        return [self.cells[c] for c in sorted(self.cells) if c != inc]

    def table(self) -> list[dict]:
        return [{"cell": cell_name(c), "score": e.score, "experiment": e.experiment, "regime": e.regime,
                 "program": e.program, "deep_polished": e.deep_polished} for c, e in sorted(self.cells.items())]
