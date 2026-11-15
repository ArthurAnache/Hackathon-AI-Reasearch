"""Run ledger (section 6): experiments.jsonl (shared format) + events.jsonl (one line per decision).

The experiment record is the union of the three formats the repo asks for, so every consumer can read it:
- CLAUDE.md shared fields: experiment_id, hypothesis, config, seed, budget_s, valid, score, best_known_score,
  decision, rationale, evidence_urls;
- WS2's interfaces.py RECORD_FIELDS (JsonlStore refuses lines without them);
- CLAUDE.md WS3 rule 9 fields for Modal runs (run_id, source_commit, device, ..., gpu_objective, wall_s).
Anything else goes inside `config` or into events.jsonl.

    python -m knockout.ledger validate runs/<run_id>
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

STATUS = Literal["keep", "discard", "invalid", "crash", "void"]
DECISION = Literal["baseline", "keep", "discard", "invalid", "crash", "rejected", "record_claim", "numerical_artifact"]
EVENT_TYPES = (
    # section 6
    "gene_born", "gate_pair", "gate_verdict", "genome_built", "operator_pick", "island_pick", "filter_reject",
    "fidelity_promotion", "credit_update", "migration", "lineage_frozen", "strategy_patch", "strategy_revert",
    "promotion",
    # run lifecycle and the control arm (not in the prompt's list; needed to replay the control arm)
    "run_start", "run_end", "control_round",
    # LLM editor and warm-start ratchet
    "llm_call", "warm_start_update",
    # main arm: islands and linkage (decisions with no type of their own in the prompt's list)
    "island_opened", "island_replaced", "linkage",
)


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # CLAUDE.md shared format
    experiment_id: str
    hypothesis: str
    config: dict[str, Any]
    seed: int
    budget_s: float
    valid: bool
    score: float | None
    best_known_score: float | None
    decision: DECISION
    rationale: str
    evidence_urls: list[str]
    # WS2 interfaces.py RECORD_FIELDS (the overlap with the block above is not repeated)
    timestamp: str
    problem: str
    agent: str
    optimizer: str
    algorithm_version: str
    git_commit: str | None
    status: STATUS
    best_before: float | None
    beats_record: bool
    duration_s: float
    error: str | None
    evidence: list[dict]
    solution_path: str | None
    # CLAUDE.md WS3 rule 9 (Modal runs)
    run_id: str
    source_commit: str | None
    device: str
    gpu_type: str | None
    dtype: str | None
    torch_version: str | None
    cuda_version: str | None
    gpu_objective: float | None          # the sandbox-side guide objective (GPU or CPU), never used as the score
    wall_s: float


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    run_id: str
    timestamp: str
    type: Literal[EVENT_TYPES]  # type: ignore[valid-type]
    seed: int | None
    refs: list[str]
    payload: dict[str, Any]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Ledger:
    """Append-only, validated JSONL files under one run directory. Thread-safe."""

    def __init__(self, run_dir: str | Path, run_id: str):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "programs").mkdir(exist_ok=True)
        (self.dir / "solutions").mkdir(exist_ok=True)
        self.run_id = run_id
        self.experiments_path = self.dir / "experiments.jsonl"
        self.events_path = self.dir / "events.jsonl"
        self._lock = threading.Lock()
        self._n_exp = sum(1 for _ in self.experiments_path.open()) if self.experiments_path.exists() else 0
        self._n_evt = sum(1 for _ in self.events_path.open()) if self.events_path.exists() else 0

    # experiment_store.append(record) from the shared contract
    def append(self, record: dict) -> None:
        rec = ExperimentRecord(**record)
        with self._lock, self.experiments_path.open("a") as f:
            f.write(rec.model_dump_json() + "\n")

    def read(self) -> list[dict]:
        if not self.experiments_path.exists():
            return []
        return [json.loads(line) for line in self.experiments_path.read_text().splitlines() if line.strip()]

    def next_experiment_id(self) -> str:
        with self._lock:
            self._n_exp += 1
            return f"exp_{self._n_exp:06d}"

    def event(self, type_: str, *, seed: int | None, refs: list[str] | None = None, **payload) -> str:
        with self._lock:
            self._n_evt += 1
            event_id = f"evt_{self._n_evt:06d}"
            rec = EventRecord(event_id=event_id, run_id=self.run_id, timestamp=now_iso(), type=type_, seed=seed,
                              refs=list(refs or []), payload=payload)
            with self.events_path.open("a") as f:
                f.write(rec.model_dump_json() + "\n")
        return event_id

    def events(self) -> list[dict]:
        if not self.events_path.exists():
            return []
        return [json.loads(line) for line in self.events_path.read_text().splitlines() if line.strip()]

    def save_program(self, program_id: str, source: str) -> str:
        path = self.dir / "programs" / f"{program_id}.py"
        if not path.exists():
            path.write_text(source)
        return str(path.relative_to(self.dir))

    def save_solution(self, chain: str, experiment_id: str, candidate: Any) -> str:
        """Global id <run_id>/<chain>/<experiment_id> (CLAUDE.md WS3 rule 9); path relative to the run dir."""
        d = self.dir / "solutions" / chain
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{experiment_id}.json"
        path.write_text(json.dumps(candidate))
        return str(path.relative_to(self.dir))


def validate_run(run_dir: str | Path) -> dict:
    """Validate both JSONL files line by line. Raises ValueError on the first bad line."""
    run_dir = Path(run_dir)
    counts = {}
    for name, model in (("experiments.jsonl", ExperimentRecord), ("events.jsonl", EventRecord)):
        path = run_dir / name
        n = 0
        ids = set()
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = model(**json.loads(line))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{path}:{i}: {exc}") from exc
            key = rec.experiment_id if name == "experiments.jsonl" else rec.event_id
            if key in ids:
                raise ValueError(f"{path}:{i}: duplicate id {key}")
            ids.add(key)
            n += 1
        counts[name] = n
    return counts


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "validate":
        print(__doc__)
        sys.exit(2)
    print(json.dumps(validate_run(sys.argv[2])))
