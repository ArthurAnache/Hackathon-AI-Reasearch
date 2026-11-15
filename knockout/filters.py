"""Anti-cycling and novelty filters, applied before any evaluation (section 5.5).

- exact duplicates: hash of the normalized source (comments, docstrings and layout removed);
- near-duplicates: token-set Jaccard above `filters.jaccard_max` against programs of the same island;
- cycling: a child is rejected when at least `filters.recycled_line_ratio` of its added lines were deleted
  earlier in its own lineage.
"""
from __future__ import annotations

import ast
import hashlib
import io
import tokenize


def normalize(source: str) -> str:
    """Source without comments, docstrings and formatting differences. Falls back to stripped lines."""
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                    and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
        return ast.unparse(tree)
    except SyntaxError:
        return "\n".join(line.strip() for line in source.splitlines() if line.strip())


def norm_hash(source: str) -> str:
    return hashlib.sha256(normalize(source).encode()).hexdigest()[:16]


def token_set(source: str) -> frozenset[str]:
    try:
        toks = tokenize.generate_tokens(io.StringIO(normalize(source)).readline)
        return frozenset(t.string for t in toks if t.type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING,
                                                               tokenize.OP))
    except (tokenize.TokenError, IndentationError):
        return frozenset(source.split())


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def code_lines(source: str) -> set[str]:
    """Normalized, non-trivial lines of a program."""
    return {line.strip() for line in normalize(source).splitlines() if len(line.strip()) > 3}


def diff_lines(parent: str, child: str) -> tuple[set[str], set[str]]:
    """(added, deleted) normalized lines from parent to child."""
    p, c = code_lines(parent), code_lines(child)
    return c - p, p - c


def recycled_ratio(added: set[str], lineage_deleted: set[str]) -> float:
    return len(added & lineage_deleted) / len(added) if added else 0.0


def check(child: str, parent: str | None, island_hashes: set[str], island_tokens: list[frozenset[str]],
          lineage_deleted: set[str], jaccard_max: float, recycled_max: float) -> dict | None:
    """The reason a child is rejected before evaluation, or None. `parent` is None for a `fresh` child."""
    h = norm_hash(child)
    if h in island_hashes:
        return {"reason": "exact duplicate", "hash": h}
    toks = token_set(child)
    best = max((jaccard(toks, t) for t in island_tokens), default=0.0)
    if best > jaccard_max:
        return {"reason": "near duplicate", "jaccard": round(best, 4)}
    if parent is not None:
        added, _ = diff_lines(parent, child)
        ratio = recycled_ratio(added, lineage_deleted)
        if added and ratio >= recycled_max:
            return {"reason": "recycled lines", "ratio": round(ratio, 3),
                    "examples": sorted(added & lineage_deleted)[:5]}
    return None
