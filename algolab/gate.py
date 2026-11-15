"""Static admission gate for LLM-written optimizer plugins (layer 1 of 4; the runtime layers are in child.py).

A plugin is one Python file that defines PARAMS (dict) and a class Algorithm(params, rng) with
    improve(f, deadline) -> np.ndarray      local optimization until time.monotonic() >= deadline
    perturb(f, intensity) -> np.ndarray     optional; exploration move of the given intensity
The harness owns the loop, the clock budget, the incumbent, scoring and acceptance.

The gate is a whitelist, not a sandbox: it rejects code that could touch files, processes, the network, the
verifier or the interpreter, so that rejected ideas fail fast with a readable reason the LLM can fix.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

MAX_BYTES = 60_000
ALLOWED_IMPORTS = {
    "math", "time", "heapq", "itertools", "functools", "bisect", "collections", "dataclasses", "typing",
    "numpy", "scipy", "scipy.fft", "scipy.optimize", "scipy.signal", "scipy.sparse", "scipy.sparse.linalg",
    "scipy.linalg", "scipy.ndimage", "scipy.interpolate", "algolab.kit",
}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "help", "memoryview", "exit", "quit", "__builtins__", "__loader__",
    "__spec__",
}
FORBIDDEN_ATTRS = {
    "load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile",
    "tofile", "memmap", "ctypeslib", "ctypes", "dump", "dumps", "system", "popen", "lib", "testing", "f2py",
    "distutils", "show_config", "io", "misc", "datasets",
}


def _disallowed_imports(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names if a.name not in ALLOWED_IMPORTS]
    if isinstance(node, ast.ImportFrom):
        if node.level:
            return ["." * node.level + (node.module or "")]
        if node.module in ALLOWED_IMPORTS:
            return []
        return [f"{node.module}.{a.name}" for a in node.names if f"{node.module}.{a.name}" not in ALLOWED_IMPORTS]
    return []


def check_source(src: str) -> list[str]:
    """Reasons for rejection; an empty list means admitted."""
    if len(src.encode()) > MAX_BYTES:
        return [f"file larger than {MAX_BYTES} bytes"]
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    problems: list[str] = []
    for node in ast.walk(tree):
        for mod in _disallowed_imports(node):
            problems.append(f"line {node.lineno}: import {mod!r} not allowed")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"line {node.lineno}: name {node.id!r} not allowed")
        if isinstance(node, ast.Attribute):
            # Block dunders (__x__) and known exfiltration attrs; allow ordinary _helpers.
            if node.attr.startswith("__") and node.attr.endswith("__"):
                problems.append(f"line {node.lineno}: dunder attribute {node.attr!r} not allowed")
            elif node.attr in FORBIDDEN_ATTRS:
                problems.append(f"line {node.lineno}: attribute {node.attr!r} not allowed")
        if isinstance(node, ast.Global):
            problems.append(f"line {node.lineno}: global statement not allowed")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            problems.append(f"line {node.lineno}: dunder string literal not allowed")
    names = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    assigned = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    if "Algorithm" not in names:
        problems.append("missing class Algorithm")
    if "PARAMS" not in assigned:
        problems.append("missing PARAMS dict")
    for n in tree.body:
        if not isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign,
                              ast.Expr)):
            problems.append(f"line {n.lineno}: top-level {type(n).__name__} not allowed (no side effects at import)")
        if isinstance(n, ast.Expr) and not (isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)):
            problems.append(f"line {n.lineno}: top-level expression not allowed (docstrings only)")
    return problems


def admit(path: Path) -> dict:
    src = Path(path).read_text(encoding="utf-8")
    reasons = check_source(src)
    return {"plugin": str(path), "sha256": hashlib.sha256(src.encode()).hexdigest(), "admitted": not reasons,
            "reasons": reasons}
