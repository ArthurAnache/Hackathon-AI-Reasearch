"""Entry point to the frozen EinsteinArena problems. FROZEN: owned by WS1; agents never modify problems/.

    from problems.catalog import load
    p = load("second-autocorrelation-inequality")
    p.verify(candidate)             # -> bool
    p.score(candidate)              # -> float (raises InvalidSolution)
    p.best_known_score()            # -> {"value", "agent", "live", "source", "read_at"}
    p.target()                      # live #1 + delta

Problem directories use the EinsteinArena slug (with dashes), so they are loaded by path, not imported as packages.

ws2_module(slug) adapts a problem to WS2's catalog contract (INTERFACES.md on feature/ws2-autoresearch-loop):
ID, NAME, URL, DIRECTION, BEST_KNOWN, BEST_KNOWN_SOURCE, SOLUTION_FORMAT, verify(solution) -> float (raises
InvalidSolution), verify_exact(solution) -> Fraction. WS2 registers it in its own problems/__init__.py.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

PROBLEMS_DIR = Path(__file__).resolve().parent

MAIN = "second-autocorrelation-inequality"
BACKUP = "circles-rectangle"
BET = "ring-loading-15"
SLUGS = (MAIN, BACKUP, BET)


def _module_name(slug: str) -> str:
    return "ws1_problem_" + slug.replace("-", "_")


def load(slug: str):
    """The problems/<slug>/problem.py module (cached)."""
    if slug not in SLUGS:
        raise KeyError(f"unknown frozen problem {slug!r}; known: {list(SLUGS)}")
    name = _module_name(slug)
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PROBLEMS_DIR / slug / "problem.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def ws2_module(slug: str, max_age_s: float = 300.0) -> types.SimpleNamespace:
    """A WS2-style problem module. BEST_KNOWN is the live #1 at call time (snapshot if offline)."""
    p = load(slug)
    best = p.best_known_score(max_age_s)

    def verify_exact(solution):
        return p.independent_score(solution)["fraction"]

    return types.SimpleNamespace(
        ID=p.SLUG, NAME=p.NAME, URL=p.URL, DIRECTION=p.DIRECTION,
        BEST_KNOWN=best["value"],
        BEST_KNOWN_SOURCE=f"EinsteinArena #1 {best['agent']} ({'live' if best['live'] else 'snapshot'}, "
                          f"{best['read_at']}); a new record needs gain >= delta = {p.DELTA}",
        SOLUTION_FORMAT=p.SOLUTION_FORMAT,
        verify=p.score, verify_exact=verify_exact, DELTA=p.DELTA, beats_record=p.beats_record,
    )
