"""Certificate store (L1 truth). Hash-locked with prepare.py and interfaces.py.

A certificate is a verified solution: {problem, score, sha256, solution, provenance}.
  - It is stored once, under certificates/<problem>/<sha12>.json (content-addressed).
  - It is RE-VERIFIED every time it is loaded: the frozen verifier recomputes the score, and a
    mismatch (edited file, changed verifier) makes the load fail.
  - Certificates are the ONLY allowed warm starts. A job's warm start is a certificate id,
    never a free-form path, so every refinement chain starts from verified truth. The parent
    certificate also gives the knock-out test its "same parent".
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "certificates"
SCORE_RTOL = 1e-12


class CertificateError(ValueError):
    """The certificate is missing, malformed, or no longer verifies to its recorded score."""


def candidate_sha(candidate) -> str:
    return hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class Certificate:
    id: str                 # "<problem>/<sha12>"
    problem: str
    score: float
    sha256: str
    solution: dict
    provenance: dict
    path: Path


class CertificateStore:
    def __init__(self, root: str | Path = DEFAULT_DIR):
        self.root = Path(root)

    # ---------------------------------------------------------------- write
    def put(self, problem, candidate, score: float, provenance: dict | None = None) -> Certificate:
        """Verify, then store. Raises CertificateError if the candidate does not verify to `score`."""
        self._check(problem, candidate, score)
        sha = candidate_sha(candidate)
        cid = f"{problem.id}/{sha[:12]}"
        path = self.root / f"{cid}.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"problem": problem.id, "score": score, "sha256": sha,
                                        "provenance": provenance or {}, "solution": candidate}))
        return self.load(problem, cid)

    # ---------------------------------------------------------------- read (always re-verified)
    def load(self, problem, cert_id: str) -> Certificate:
        path = self.root / f"{cert_id}.json"
        if not path.exists():
            raise CertificateError(f"no certificate {cert_id}")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise CertificateError(f"{cert_id}: unreadable ({e})") from e
        if data.get("problem") != problem.id:
            raise CertificateError(f"{cert_id} is for problem {data.get('problem')}, not {problem.id}")
        if candidate_sha(data["solution"]) != data.get("sha256"):
            raise CertificateError(f"{cert_id}: solution does not match its sha256 (edited?)")
        self._check(problem, data["solution"], data["score"])
        return Certificate(cert_id, problem.id, data["score"], data["sha256"], data["solution"],
                           data.get("provenance", {}), path)

    def ids(self, problem_id: str) -> list[str]:
        d = self.root / problem_id
        return sorted(f"{problem_id}/{p.stem}" for p in d.glob("*.json")) if d.exists() else []

    def best(self, problem) -> Certificate | None:
        """Best certificate that still verifies (bad ones are skipped, never trusted)."""
        best = None
        for cid in self.ids(problem.id):
            try:
                c = self.load(problem, cid)
            except CertificateError:
                continue
            if best is None or problem.better(c.score, best.score):
                best = c
        return best

    @staticmethod
    def _check(problem, candidate, score: float) -> None:
        reason = problem.why_invalid(candidate)
        if reason is not None:
            raise CertificateError(f"candidate is invalid: {reason}")
        actual = problem.score(candidate)
        if not math.isclose(actual, score, rel_tol=SCORE_RTOL, abs_tol=SCORE_RTOL):
            raise CertificateError(f"recorded score {score!r} but the verifier gives {actual!r}")
