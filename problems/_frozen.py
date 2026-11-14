"""Shared machinery for the frozen EinsteinArena problems in problems/<slug>/. FROZEN: owned by WS1.

The agent (WS2) and the optimizers (WS3) never modify anything under problems/. tests/test_ws1_frozen_problems.py
fails if a pinned verifier or a certificate changes.

A FrozenProblem wraps one directory problems/<slug>/ that holds:
  meta.json            definition: id, direction, delta, limits, pinned commit + sha256, record snapshot
  verifier_pinned.py   the EinsteinArena verifier, byte-identical to the pinned commit and to the live one
  certificate.json.gz  the #1 solution at freeze time (its sha256 is in meta.json)
  independent.py       a second implementation of the score, with different code (used to confirm claims)
  problem.py           the public API of that problem (built on this class)

Public API (also exposed as module-level functions by each problems/<slug>/problem.py):
  verify(candidate) -> bool                 accepted by the server schema AND the pinned verifier
  score(candidate) -> float                 pinned-verifier score; raises InvalidSolution
  why_invalid(candidate) -> str | None
  independent_score(candidate) -> dict      {"score", "fraction", "tolerance_abs", "method"}; raises InvalidSolution
  best_known_score(max_age_s=300) -> dict   live #1 from the EinsteinArena leaderboard (cached), snapshot fallback
  target(max_age_s=300) -> float            live #1 +/- delta
  beats_record(score, best=None, tolerance_abs=0.0) -> bool   gain >= delta, gain > 0, gain > tolerance
  certificate() -> dict                     the frozen #1 solution (sha256-checked)
  submission_path(candidate) -> "inline" | "blob"
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

PROBLEMS_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBLEMS_DIR.parent
INLINE_LIMIT_CHARS = 2_000_000   # server: inline JSON.stringify(solution) must stay under 2,000,000 chars
INLINE_SAFE_CHARS = 1_900_000    # we switch to the blob upload path a bit earlier

try:  # share WS2's exception class when both catalogs live in the same tree
    from problems.common import InvalidSolution  # type: ignore
except Exception:  # noqa: BLE001  (ImportError, or WS2's package not present on this branch)
    class InvalidSolution(ValueError):
        """The candidate breaks a constraint of the problem. It gets no score."""


class FrozenFileChanged(RuntimeError):
    """A frozen file (verifier or certificate) no longer matches the sha256 recorded in meta.json."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class FrozenProblem:
    def __init__(self, directory: Path, check_schema, independent):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        m = self.meta
        self.slug: str = m["slug"]
        self.problem_id: int = m["problem_id"]
        self.name: str = m["title"]
        self.url: str = m["url"]
        self.role: str = m["role"]
        self.direction: str = m["direction"]
        self.delta: float = float(m["delta"])
        self.solution_format: str = m["solution_format"]
        self.limits: dict = m["limits"]
        self._check_schema = check_schema
        self._independent = independent
        self._best_cache: dict | None = None

        src = (self.dir / m["verifier"]["file"]).read_bytes()
        if sha256_bytes(src) != m["verifier"]["sha256"]:
            raise FrozenFileChanged(f"{self.slug}: verifier_pinned.py does not match the frozen sha256")
        self.verifier_sha256 = m["verifier"]["sha256"]
        # Same harness as the server (web/src/app/api/evaluate/route.ts): prefix "import json", call evaluate(data).
        ns: dict = {"__name__": f"einstein_verifier_{self.slug}"}
        exec(compile("import json\n" + src.decode(), str(self.dir / m["verifier"]["file"]), "exec"), ns)
        self._evaluate = ns["evaluate"]

    # ------------------------------------------------------------------ validity and score
    def score(self, candidate) -> float:
        self._check_schema(candidate)
        try:
            value = self._evaluate(candidate)
        except Exception as exc:  # noqa: BLE001  (the verifier signals invalid input by raising)
            raise InvalidSolution(f"pinned verifier rejected the candidate: {type(exc).__name__}: {exc}") from exc
        value = float(value)
        if not math.isfinite(value):
            raise InvalidSolution(f"pinned verifier returned {value} (a constraint is violated)")
        return value

    def why_invalid(self, candidate) -> str | None:
        try:
            self.score(candidate)
        except InvalidSolution as exc:
            return str(exc)
        return None

    def verify(self, candidate) -> bool:
        return self.why_invalid(candidate) is None

    def independent_score(self, candidate) -> dict:
        self._check_schema(candidate)
        try:
            return self._independent(candidate)
        except ValueError as exc:
            raise InvalidSolution(f"independent verifier rejected the candidate: {exc}") from exc

    # ------------------------------------------------------------------ record and target
    def best_known_score(self, max_age_s: float = 300.0) -> dict:
        """Live #1 of the EinsteinArena leaderboard, cached for max_age_s seconds (0 forces a fresh read).
        Falls back to the frozen snapshot (live=False) when the API cannot be reached."""
        now = time.time()
        if self._best_cache and now - self._best_cache["_t"] <= max_age_s:
            return {k: v for k, v in self._best_cache.items() if k != "_t"}
        try:
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            from integrations import einstein_arena as ea
            top = ea.get_leaderboard(self.problem_id, limit=1)[0]
            out = {"value": float(top["bestScore"]), "agent": top["agentName"], "live": True,
                   "source": f"{ea.base_url()}/api/leaderboard?problem_id={self.problem_id}",
                   "read_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))}
        except Exception as exc:  # noqa: BLE001  (offline, rate limited, API change: use the snapshot)
            rec = self.meta["record"]
            out = {"value": float(rec["value"]), "agent": rec["agent"], "live": False,
                   "source": f"frozen snapshot ({rec['read_at']}); live read failed: {type(exc).__name__}",
                   "read_at": rec["read_at"]}
        self._best_cache = {**out, "_t": now}
        return out

    def gain(self, score: float, best: float) -> float:
        """How much `score` beats `best` (positive = better), as the server computes it."""
        return best - score if self.direction == "minimize" else score - best

    def target(self, max_age_s: float = 300.0) -> float:
        best = self.best_known_score(max_age_s)["value"]
        return best - self.delta if self.direction == "minimize" else best + self.delta

    def beats_record(self, score: float, best: float | None = None, tolerance_abs: float = 0.0) -> bool:
        """Server rule for a new #1: gain >= delta, and a tie never wins (gain > 0). We also require the gain to
        exceed the numerical tolerance of the verifiers (CLAUDE.md rule 2)."""
        if best is None:
            best = self.best_known_score()["value"]
        g = self.gain(score, best)
        return g >= self.delta and g > 0 and g > tolerance_abs

    # ------------------------------------------------------------------ certificate and submission
    def certificate(self) -> dict:
        c = self.meta["certificate"]
        raw = gzip.decompress((self.dir / c["file"]).read_bytes())
        if sha256_bytes(raw) != c["sha256"]:
            raise FrozenFileChanged(f"{self.slug}: certificate does not match the frozen sha256")
        return json.loads(raw)

    def submission_path(self, candidate) -> str:
        size = len(json.dumps(candidate, separators=(",", ":")))
        return "blob" if size > INLINE_SAFE_CHARS else "inline"


def number(x) -> bool:
    """A JSON number as zod's z.number() sees it (bool is not a number)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
