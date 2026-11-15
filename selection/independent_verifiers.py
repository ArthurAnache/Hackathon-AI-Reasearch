"""WS1: independent verifiers, used by integrations/submit.py (gate b) and selection/verify_claim.py.

Since the freeze (2026-11-14), the implementations live with each frozen problem in problems/<slug>/independent.py;
this module only relays to them, so submission and arbitration use exactly the same code:

  ring-loading-15                    integer enumeration over a common denominator (pinned: Fraction loop)
  circles-rectangle                  exact Fraction feasibility (perimeter, squared distances), exact sum of radii
  second-autocorrelation-inequality  exact rational: dyadic -> integers, Kronecker substitution squared with decimal,
                                     closed-form piecewise-linear integrals (pinned: scipy oaconvolve, float64)

score(slug, solution) returns {"score": float, "exact": "num/den" | None, "tolerance_abs": float, "method": str}.
A ValueError means the independent check judges the solution invalid. A problem without an implementation is never
submitted by submit.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from problems.catalog import SLUGS, load  # noqa: E402


def _relay(slug: str):
    def run(solution: dict) -> dict:
        p = load(slug)
        try:
            res = p.independent_score(solution)
        except p.InvalidSolution as exc:
            raise ValueError(str(exc)) from exc
        frac = res.get("fraction")
        return {"score": res["score"], "exact": f"{frac.numerator}/{frac.denominator}" if frac is not None else None,
                "tolerance_abs": res["tolerance_abs"], "method": res["method"]}
    return run


IMPLEMENTATIONS = {slug: _relay(slug) for slug in SLUGS}


def score(slug: str, solution: dict) -> dict:
    if slug not in IMPLEMENTATIONS:
        raise KeyError(f"no independent verifier for {slug}")
    return IMPLEMENTATIONS[slug](solution)
