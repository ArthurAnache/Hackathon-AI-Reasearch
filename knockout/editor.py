"""Editors produce child programs from a parent program.

M0 ships only `OfflineParamEditor`, a deterministic stand-in used until the LLM editor exists (M1). It rewrites the
PARAMS literal of the parent (move a default, widen or narrow a range). It is a test fixture for the plumbing, not a
research operator: real edits come from the LLM editor, which plugs in through the same `Editor` protocol.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class EditProposal:
    source: str
    description: str          # becomes the experiment's hypothesis
    operator: str
    meta: dict = field(default_factory=dict)


class Editor(Protocol):
    name: str

    def propose(self, parent_source: str, n: int, rng: np.random.Generator) -> list[EditProposal]: ...


def _params_node(tree: ast.Module) -> ast.Assign | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets):
            return node
    return None


def read_params(source: str) -> dict[str, tuple]:
    node = _params_node(ast.parse(source))
    return {} if node is None else dict(ast.literal_eval(node.value))


def write_params(source: str, params: dict[str, tuple]) -> str:
    tree = ast.parse(source)
    node = _params_node(tree)
    if node is None:
        raise ValueError("program has no PARAMS literal")
    lines = source.splitlines(keepends=True)
    body = "".join(f"    {k!r}: {tuple(v)!r},\n" for k, v in params.items())
    new = "PARAMS = {\n" + body + "}\n"
    return "".join(lines[: node.lineno - 1]) + new + "".join(lines[node.end_lineno:])


def _cast(v: float, integer: bool):
    return int(round(v)) if integer else float(f"{v:.6g}")


class OfflineParamEditor:
    name = "offline-params"

    def _mutate(self, params: dict[str, tuple], rng: np.random.Generator) -> tuple[dict, str]:
        name = sorted(params)[int(rng.integers(len(params)))]
        spec = list(params[name])
        default, low, high = spec[:3]
        log = len(spec) == 4 and spec[3] == "log"
        integer = all(isinstance(v, int) for v in (default, low, high))
        kind = ("move", "widen", "narrow")[int(rng.integers(3))]
        to = (lambda v: math.log(v)) if log else (lambda v: v)
        fro = (lambda v: math.exp(v)) if log else (lambda v: v)
        lo, hi, d = to(low), to(high), to(default)
        span = hi - lo or 1.0
        if kind == "move":
            d = min(hi, max(lo, d + rng.normal(0, 0.3) * span))
        elif kind == "widen":
            lo, hi = d - 1.5 * (d - lo) - 0.05 * span, d + 1.5 * (hi - d) + 0.05 * span
        else:
            lo, hi = d - 0.5 * (d - lo), d + 0.5 * (hi - d)
        new_low, new_high, new_def = fro(lo), fro(hi), fro(d)
        if low > 0 and new_low <= 0:  # keep positive ranges positive
            new_low = low / 2
        new_low, new_high = _cast(new_low, integer), _cast(new_high, integer)
        new_def = min(new_high, max(new_low, _cast(new_def, integer)))
        new_spec = (new_def, new_low, new_high) + (("log",) if log else ())
        out = dict(params)
        out[name] = new_spec
        return out, f"{kind} PARAMS[{name!r}]: {tuple(spec)} -> {new_spec}"

    def propose(self, parent_source: str, n: int, rng: np.random.Generator) -> list[EditProposal]:
        params = read_params(parent_source)
        if not params:
            return []
        out, seen = [], {parent_source}
        for _ in range(4 * n):
            if len(out) == n:
                break
            new_params, desc = self._mutate(params, rng)
            src = write_params(parent_source, new_params)
            if src in seen:
                continue
            seen.add(src)
            out.append(EditProposal(src, desc, "offline-params", {"params_after": new_params}))
        return out
