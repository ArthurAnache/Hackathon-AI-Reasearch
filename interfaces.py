"""The team's shared technical contract. Hash-locked: change only with all workstreams' agreement.

    problem.verify(candidate) -> bool                                  WS1  (Problem)
    problem.score(candidate) -> float                                  WS1  (Problem)
    optimizer.run(problem, budget_s, seed, config) -> candidate        WS3  (Optimizer)
    research_agent.propose(history, evidence) -> experiment_config     WS2  (ResearchAgent)
    research_agent.review(experiment_result) -> next_action            WS2  (ResearchAgent)
    experiment_store.append(result) -> None                            WS2/WS4 (ExperimentStore)

plus EvidenceSource.gather(problem, history) -> [Evidence] for WS4 (Exa, past.dev).

`run_experiment()` is the one fixed experiment protocol everybody shares. It runs an
optimizer under the fixed budget and seed, verifies independently, applies the
keep/discard rule, saves the solution and appends one record to the store.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import problems
import prepare
from problems import InvalidSolution

ROOT = Path(__file__).resolve().parent


# ===================================================================== WS1: problem
class Problem:
    """A frozen problem. Wraps problems/<id>.py; nothing here is editable by an agent."""

    def __init__(self, problem_id: str):
        self._m = problems.get(problem_id)
        self.id: str = self._m.ID
        self.name: str = self._m.NAME
        self.url: str = self._m.URL
        self.direction: str = self._m.DIRECTION
        self.best_known: float = self._m.BEST_KNOWN
        self.best_known_source: str = self._m.BEST_KNOWN_SOURCE
        self.solution_format: str = self._m.SOLUTION_FORMAT

    def verify(self, candidate) -> bool:
        """True if the candidate satisfies every constraint."""
        return self.why_invalid(candidate) is None

    def score(self, candidate) -> float:
        """The objective value. Raises InvalidSolution if the candidate is not valid."""
        return self._m.verify(candidate)

    def why_invalid(self, candidate) -> str | None:
        try:
            self._m.verify(candidate)
        except InvalidSolution as e:
            return str(e)
        return None

    def certify(self, candidate) -> Fraction | None:
        """Exact rational score, if this problem has an exact verifier (1a, 1b, 1d)."""
        return self._m.verify_exact(candidate) if hasattr(self._m, "verify_exact") else None

    def better(self, a: float, b: float) -> bool:
        return problems.better(a, b, self.direction)

    def beats_record(self, score: float) -> bool:
        return problems.beats_record(score, self._m)


def get_problem(problem_id: str) -> Problem:
    return Problem(problem_id)


# ===================================================================== WS3: optimizer
@runtime_checkable
class Optimizer(Protocol):
    name: str

    def run(self, problem: Problem, budget_s: float, seed: int, config: dict) -> Any: ...

    def version(self) -> str: ...


class FileOptimizer:
    """Adapter for an optimizer file: PROBLEM, CONFIG and search(budget_s, seed, config).

    `search.py` (Karpathy mode) and every file in optimizers/ are such files. The file is
    re-read on every run, so edits made between experiments are picked up.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.name = self.path.stem

    def _load(self):
        spec = importlib.util.spec_from_file_location(f"optimizer_{self.name}", self.path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr in ("PROBLEM", "CONFIG", "search"):
            if not hasattr(module, attr):
                raise TypeError(f"{self.path.name} must define {attr}")
        return module

    @property
    def problem_id(self) -> str:
        return self._load().PROBLEM

    def default_config(self) -> dict:
        return dict(self._load().CONFIG)

    def version(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()[:12]

    def run(self, problem: Problem, budget_s: float, seed: int, config: dict) -> Any:
        module = self._load()
        if module.PROBLEM != problem.id:
            raise ValueError(f"{self.path.name} is written for problem {module.PROBLEM}, not {problem.id}")
        return module.search(budget_s, seed, {**module.CONFIG, **(config or {})})


# ===================================================================== WS4: evidence
@dataclass
class Evidence:
    source: str                  # "exa", "past.dev", "paper", ...
    content: str                 # what was learned, in one or two sentences
    citation: str = ""           # URL / arXiv id / memory id
    query: str = ""
    author: str = ""             # which agent produced it (for shared memory)
    retrieved_at: str = ""
    latency_s: float = 0.0


@runtime_checkable
class EvidenceSource(Protocol):
    def gather(self, problem: Problem, history: list[dict]) -> list[Evidence]: ...


class NoEvidence:
    """Default: the loop works with no external credentials."""

    def gather(self, problem: Problem, history: list[dict]) -> list[Evidence]:
        return []


# ===================================================================== WS2: research agent
@dataclass
class ExperimentConfig:
    hypothesis: str                       # one falsifiable sentence
    config: dict = field(default_factory=dict)   # overrides of the optimizer's CONFIG
    optimizer: str | None = None          # path of another optimizer file; None = the current one
    rationale: str = ""                   # which history / evidence led here


NEXT_ACTIONS = ("continue", "stop")


@runtime_checkable
class ResearchAgent(Protocol):
    name: str

    def propose(self, history: list[dict], evidence: list[Evidence]) -> ExperimentConfig: ...

    def review(self, experiment_result: dict) -> str: ...   # one of NEXT_ACTIONS


# ===================================================================== WS2/WS4: experiment store
RECORD_FIELDS = (
    "experiment_id", "timestamp", "problem", "hypothesis", "rationale", "agent", "optimizer",
    "algorithm_version", "git_commit", "config", "seed", "budget_s", "valid", "score", "status",
    "best_before", "beats_record", "duration_s", "error", "evidence", "solution_path",
)
STATUSES = ("keep", "discard", "invalid", "crash", "void")


@runtime_checkable
class ExperimentStore(Protocol):
    def append(self, result: dict) -> None: ...

    def read(self) -> list[dict]: ...


class JsonlStore:
    """Default store: one JSON record per line (experiments.jsonl, never committed)."""

    def __init__(self, path: str | Path = ROOT / "experiments.jsonl"):
        self.path = Path(path)

    def append(self, result: dict) -> None:
        missing = [k for k in RECORD_FIELDS if k not in result]
        if missing:
            raise ValueError(f"record missing fields: {missing}")
        with self.path.open("a") as f:
            f.write(json.dumps(result, default=str) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


# ===================================================================== the fixed protocol
class _Watchdog(Exception):
    pass


def _alarm(signum, frame):
    raise _Watchdog()


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=path.parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def best_record(history: list[dict], problem: Problem, budget_s: float) -> dict | None:
    """Best kept record for this problem under the same budget (only like-for-like comparisons)."""
    kept = [r for r in history if r["problem"] == problem.id and r["status"] == "keep" and r["budget_s"] == budget_s]
    if not kept:
        return None
    pick = min if problem.direction == "minimize" else max
    return pick(kept, key=lambda r: r["score"])


def save_solution(problem: Problem, experiment_id: str, candidate, score: float) -> Path:
    d = prepare.SOLUTIONS_DIR / problem.id
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{experiment_id}.json"
    path.write_text(json.dumps({"problem": problem.id, "experiment_id": experiment_id, "score": score,
                                "solution": candidate}))
    return path


def run_experiment(problem: Problem, optimizer: Optimizer, experiment: ExperimentConfig,
                   store: ExperimentStore, *, budget_s: float | None = None, seed: int = prepare.SEED,
                   agent: str = "human", evidence: list[Evidence] | None = None) -> dict:
    """Run one experiment under the fixed protocol and append its record to the store."""
    budget_s = prepare.time_budget() if budget_s is None else float(budget_s)
    history = store.read()
    experiment_id = f"exp_{len(history):04d}"
    best = best_record(history, problem, budget_s)
    frozen = prepare.frozen_status()
    path = getattr(optimizer, "path", None)
    effective = experiment.config
    if hasattr(optimizer, "default_config"):
        try:
            effective = {**optimizer.default_config(), **experiment.config}
        except Exception:  # broken file: reported as a crash by run() below
            pass

    candidate, error, status = None, None, None
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(budget_s * prepare.HARD_KILL_FACTOR + 30))
    t0 = time.monotonic()
    try:
        candidate = optimizer.run(problem, budget_s, seed, experiment.config)
    except _Watchdog:
        status, error = "crash", f"killed by watchdog after {prepare.HARD_KILL_FACTOR}x budget + 30s"
    except Exception as e:  # a crash is a result, not a reason to stop the loop
        status, error = "crash", f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    duration = time.monotonic() - t0

    score, valid = None, False
    if status is None:
        reason = problem.why_invalid(candidate)
        valid = reason is None
        if not valid:
            status, error = "invalid", reason
        else:
            score = problem.score(candidate)
            if frozen != "ok":
                status, error = "void", f"frozen files {frozen}"
            elif duration > budget_s * prepare.OVER_BUDGET_SLACK + 1:
                status, error = "void", f"over time budget ({duration:.1f}s > {budget_s}s)"
            elif best is None or problem.better(score, best["score"]) and \
                    abs(score - best["score"]) > prepare.MIN_IMPROVEMENT_REL * max(1.0, abs(best["score"])):
                status = "keep"
            else:
                status = "discard"

    record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "problem": problem.id,
        "hypothesis": experiment.hypothesis,
        "rationale": experiment.rationale,
        "agent": agent,
        "optimizer": str(path.relative_to(ROOT)) if path and path.is_relative_to(ROOT) else getattr(optimizer, "name", "?"),
        "algorithm_version": optimizer.version() if hasattr(optimizer, "version") else None,
        "git_commit": _git_commit(path) if path else None,
        "config": effective,
        "seed": seed,
        "budget_s": budget_s,
        "valid": valid,
        "score": score,
        "status": status,
        "best_before": best["score"] if best else None,
        "beats_record": bool(valid and status in ("keep", "discard") and problem.beats_record(score)),
        "duration_s": round(duration, 2),
        "error": error,
        "evidence": [asdict(e) for e in (evidence or [])],
        "solution_path": None,
    }
    if valid and status != "void":
        record["solution_path"] = str(save_solution(problem, experiment_id, candidate, score).relative_to(ROOT))
    store.append(record)
    return record
