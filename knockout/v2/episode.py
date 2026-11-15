"""One episode: start solution -> move -> polish -> exact score from the frozen verifier.

    value = final score - score of "polish alone" (schedule p0, same start, same evaluation budget)

Outcome classes (docs/knockout_v2_brief.md, section 6):
- `improved`: value > MARGIN;
- `undone`: not improved, and the result is the polished start again (relative L1 distance below UNDONE_L1);
- `worse_basin`: everything else.

The distance to the polished start needs the solution of the baseline episode, so the class is set after the run
(`classify`), once both solutions exist.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from knockout.v2.moves import MOVE_EVALS, apply_move, runs
from knockout.v2.polish import polish

SLUG = "second-autocorrelation-inequality"
BASELINE_FAMILY, BASELINE_SCHEDULE = "reheat", "p0"
MARGIN = 1e-7          # an episode is `improved` above this value (five runs of the polisher agree within 2e-9)
UNDONE_L1 = 1e-6


@dataclass(frozen=True)
class EpisodeSpec:
    start: str            # name of the start solution (starts/<name>.npy on the Volume)
    family: str
    strength: float
    seed: int
    schedule: str
    budget: int           # objective-and-gradient evaluations, move included

    @property
    def episode_id(self) -> str:
        return f"{self.start}-{self.family}-s{self.strength:g}-r{self.seed}-{self.schedule}-b{self.budget}"

    @property
    def is_baseline(self) -> bool:
        return self.family == BASELINE_FAMILY and self.schedule == BASELINE_SCHEDULE

    def baseline(self) -> "EpisodeSpec":
        """Polish alone, from the same start, with the same budget."""
        return EpisodeSpec(self.start, BASELINE_FAMILY, 0.0, 0, BASELINE_SCHEDULE, self.budget)


def rel_l1(a: np.ndarray, b: np.ndarray) -> float | None:
    """Mass-normalized L1 distance, in [0, 2]. None when the two grids differ."""
    if a.shape != b.shape:
        return None
    return float(np.abs(a / a.sum() - b / b.sum()).sum())


def frozen_score(f: np.ndarray) -> float:
    """The only score that counts: the pinned verifier of the frozen catalog (it re-checks its sha256 on load)."""
    from problems.catalog import load

    return float(load(SLUG).score({"values": np.asarray(f, dtype=np.float64).tolist()}))


def run_episode(spec: EpisodeSpec, f_start: np.ndarray, score_fn=frozen_score, references: dict | None = None):
    """Run one episode. Returns (final solution, record). `references` maps a name to a solution; the record gets
    the distance of the final solution to each of them."""
    t0 = time.monotonic()
    f_start = np.asarray(f_start, dtype=np.float64)
    moved = apply_move(spec.family, f_start, spec.seed, spec.strength, references)
    best_f, info = polish(moved, spec.schedule, spec.budget - MOVE_EVALS.get(spec.family, 0.0))
    starts, _ = runs(best_f)
    record = {
        "experiment_id": spec.episode_id,
        **asdict(spec),
        "is_baseline": spec.is_baseline,
        "baseline_id": spec.baseline().episode_id,
        "move_changed": bool(not np.array_equal(moved, f_start)),
        "moved_score": info.get("input_score"),          # closed form, right after the move
        "score": score_fn(best_f),                       # frozen verifier
        "score_closed_form": info["best_score"],
        "calls": info["calls"] + MOVE_EVALS.get(spec.family, 0.0),
        "stages": info["stages"],
        "nonzero": int(np.count_nonzero(best_f)),
        "n_runs": int(len(starts)),
        "dist_to_start": rel_l1(best_f, f_start),
    }
    for name, ref in (references or {}).items():
        record[f"dist_to_{name}"] = rel_l1(best_f, ref)
    record["wall_s"] = round(time.monotonic() - t0, 2)
    return best_f, record


def classify(value: float, dist_to_polished_start: float | None, margin: float = MARGIN) -> str:
    if value > margin:
        return "improved"
    if dist_to_polished_start is not None and dist_to_polished_start < UNDONE_L1:
        return "undone"
    return "worse_basin"


def finalize(records: list[dict], load_solution) -> list[dict]:
    """Add `baseline_score`, `value`, `dist_to_polished_start`, `decision` and `rationale` to finished episodes.
    `load_solution(record)` returns the stored solution of an episode. Episodes whose baseline has not finished
    get the decision `no_baseline`."""
    by_id = {r["experiment_id"]: r for r in records}
    rows = []
    for baseline_id in sorted({r["baseline_id"] for r in records}):
        baseline = by_id.get(baseline_id)
        polished = load_solution(baseline) if baseline else None
        for r in (r for r in records if r["baseline_id"] == baseline_id):
            r = dict(r)
            if baseline is None:
                r.update({"value": None, "dist_to_polished_start": None, "decision": "no_baseline",
                          "rationale": "the polish-alone episode of this start and budget has not finished"})
            else:
                r["baseline_score"] = baseline["score"]
                r["value"] = r["score"] - baseline["score"]
                r["dist_to_polished_start"] = rel_l1(load_solution(r), polished)
                r["decision"] = "baseline" if r["is_baseline"] else classify(r["value"], r["dist_to_polished_start"])
                r["rationale"] = (f"value {r['value']:+.3e} against polish alone; relative L1 distance to the "
                                  f"polished start {r['dist_to_polished_start']:.3g}")
            rows.append(r)
    return rows
