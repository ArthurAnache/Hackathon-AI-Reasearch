"""What runs inside the Modal GPU container: the batches of one coarse-to-fine job, one after the other.

Layout on the Volume `knockout-v2-multires` (mounted at `vol`):
    starts/frozen.npy, starts/live.npy, starts/meta.json          warm starts (written by knockout/v2/gpu's prepare)
    <run_id>/<batch>.<row>/L<level>.npy                           the row's solution at the end of each level
    <run_id>/levels.jsonl                                         one record per stored solution (shared format)
    <run_id>/progress.jsonl, <run_id>/spec.json, <run_id>/summary.json

Rules followed (CLAUDE.md, "WS3 — Modal compute and GPUs"): the GPU computes the smoothed objective only; every
stored solution, at every level, is re-scored in float64 on CPU by the frozen verifier in a separate process
before it is logged (`gpu_objective` and `score` are two fields); nothing here submits or holds a secret; warm
starts are files of the Volume.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from knockout.v2.gpu import schedule as sch
from knockout.v2.gpu.worker import PROBLEM_URL, _env, _start_path, score_in_subprocess
from knockout.v2.multires import driver, perturb
from knockout.v2.multires import plan as pl
from knockout.v2.multires.resample import profile_l1

EVIDENCE = [PROBLEM_URL, "https://github.com/justinkang221/second-autocorrelation-inequality",
            "https://arxiv.org/abs/2508.02803", "https://arxiv.org/abs/2506.16750"]
PROFILE_BLOCK = 1000


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class _Chain:
    """What knockout/v2/gpu/schedule.make_record needs to know about a row of a batch."""

    def __init__(self, batch: pl.BatchSpec, row: pl.RowSpec):
        self.name = f"{batch.name}.{row.name}"
        self.start = batch.start
        self.seed = row.seed
        self.hypothesis = row.hypothesis
        self._config = {"experiment": batch.experiment, "batch": batch.name, "start": batch.start,
                        "pad_to": batch.pad_to, "pool": batch.pool, "ups": list(batch.ups),
                        "level_evals": list(batch.level_evals), "row": row.to_dict()}

    def to_dict(self) -> dict:
        return self._config


def run_job(vol: Path, spec: dict, commit=lambda: None, xp=None, to_xp=None, to_np=None, env=None) -> dict:
    """Run spec["batches"] in order. Stops cleanly at spec["budget_s"] of wall clock: the batch in flight stores
    where it stands and the remaining batches are skipped. `xp`, `to_xp`, `to_np` and `env` default to torch on
    the GPU; the tests pass numpy."""
    t0 = time.monotonic()
    if xp is None:
        import torch

        torch.set_grad_enabled(False)
        dev = torch.device("cuda")
        xp = torch
        env = _env(torch)

        def to_xp(a):
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64)).to(dev)

        def to_np(x):
            return x.cpu().numpy()

    run_id = spec["run_id"]
    budget_s = float(spec["budget_s"])
    deadline = t0 + budget_s - float(spec.get("reserve_s", 150.0))
    batches = [pl.BatchSpec.from_dict(d) for d in spec["batches"]]
    run_dir = vol / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((vol / "starts" / "meta.json").read_text())
    best_known = float(meta["live"]["score"])
    (run_dir / "spec.json").write_text(json.dumps({**spec, "env": env, "started_at": _now(),
                                                   "best_known_score": best_known,
                                                   "best_known_source": meta["live"]}, indent=1))
    refs = {name: np.load(vol / "starts" / f"{name}.npy") for name in ("frozen", "live")}
    pool = ThreadPoolExecutor(max_workers=int(spec.get("score_workers", 4)))    # each task is a CPU subprocess
    start_scores: dict = {}
    levels_file, progress_file = run_dir / "levels.jsonl", run_dir / "progress.jsonl"
    records, summary_batches, skipped = [], [], []

    def write_summary(done: bool):
        out = {"run_id": run_id, "finished": done, "written_at": _now(), "env": env,
               "wall_s": time.monotonic() - t0, "usd_per_s": spec.get("usd_per_s"),
               "usd_estimate": (time.monotonic() - t0) * float(spec.get("usd_per_s", 0.0)),
               "best_known_score": best_known, "records": len(records), "batches": summary_batches,
               "skipped": skipped}
        (run_dir / "summary.json").write_text(json.dumps(out, indent=1))
        commit()
        return out

    for batch in batches:
        if time.monotonic() >= deadline - 60.0:
            skipped.append(batch.name)
            continue
        pl.check_batch(batch)
        start_file = _start_path(vol, batch.start)
        if batch.start not in start_scores:           # stored solutions only, re-scored by the frozen verifier
            res = score_in_subprocess(str(start_file))
            if not res["valid"]:
                raise RuntimeError(f"start {batch.start!r} is rejected by the frozen verifier: {res['error']}")
            start_scores[batch.start] = res["score"]
        start_score = start_scores[batch.start]
        f0 = driver.level_start(batch, np.load(start_file))
        chains = [_Chain(batch, r) for r in batch.rows]
        n_levels = len(batch.level_evals)
        pending, batch_records = [], []

        def on_level_end(level, f_rows, info, batch=batch, chains=chains, pending=pending):
            for i, chain in enumerate(chains):
                exp_id = sch.experiment_id(run_id, chain.name, f"L{level}")
                path = vol / sch.solution_path(exp_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, f_rows[i])
                item = {"i": i, "level": level, "info": info, "wall_s": time.monotonic() - t0, "path": path}
                pending.append((item, pool.submit(score_in_subprocess, str(path))))

        def flush(wait=False, batch=batch, chains=chains, pending=pending, batch_records=batch_records,
                  start_score=start_score, n_levels=n_levels):
            keep = []
            for item, fut in pending:
                if not (wait or fut.done()):
                    keep.append((item, fut))
                    continue
                res, info, i, level = fut.result(), item["info"], item["i"], item["level"]
                final = level == n_levels - 1 and not info["stopped_early"]
                extra = {"experiment": batch.experiment, "batch": batch.name, "row": batch.rows[i].name,
                         "level": level, "levels": n_levels, "n": res["n"], "nonzero": res["nonzero"],
                         "pool": batch.pool, "ups": list(batch.ups), "evals_done": info["evals"],
                         "planned_evals": info["planned_evals"], "init_exact": info["init_exact"][i],
                         "final": final, "stopped_early": info["stopped_early"], "s_per_eval": info["s_per_eval"],
                         "repeat_gap": info["repeat_gap"], "verifier_error": res["error"],
                         "pre": perturb.describe(batch.rows[i].pre), "post": perturb.describe(batch.rows[i].post),
                         "parent": batch.start if level == 0 else
                         sch.experiment_id(run_id, chains[i].name, f"L{level - 1}")}
                for key in ("accepted", "rejected", "restarts"):
                    extra[key] = info[key][i] if info.get(key) is not None else None
                if res["n"] == batch.pad_to and res["valid"]:      # the envelope, against both reference solutions
                    sol = np.load(item["path"])
                    for name, ref in refs.items():
                        extra[f"profile_l1_{PROFILE_BLOCK}_vs_{name}"] = profile_l1(sol, ref, PROFILE_BLOCK)
                rec = sch.make_record(
                    run_id=run_id, chain=chains[i], step=f"L{level}", score=res["score"], valid=res["valid"],
                    gpu_objective=info["smooth"][i], gpu_exact=info["exact"][i], start_score=start_score,
                    best_known_score=best_known, env=env, wall_s=item["wall_s"], budget_s=budget_s,
                    source_commit=spec.get("source_commit", ""), evidence_urls=EVIDENCE, extra=extra)
                records.append(rec)
                batch_records.append(rec)
                with levels_file.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            pending[:] = keep

        def on_progress(level, evals, total, exact, best, batch=batch, flush=flush):
            line = {"batch": batch.name, "level": level, "evals": evals, "total": total,
                    "wall_s": time.monotonic() - t0, "exact": exact, "best_exact": best}
            with progress_file.open("a") as fh:
                fh.write(json.dumps(line) + "\n")
            print(f"[{run_id}] {batch.name} L{level} evals {evals}/{total} wall {line['wall_s']:.0f}s "
                  f"best exact - start: min {min(best) - start_score:+.2e} max {max(best) - start_score:+.2e}",
                  flush=True)
            flush()

        out = driver.run_ladder(xp, batch, f0, to_xp=to_xp, to_np=to_np, history=int(spec.get("history", 8)),
                                on_level_end=on_level_end, on_progress=on_progress,
                                progress_every=int(spec.get("progress_every", 500)), deadline=deadline)
        flush(wait=True)
        rows = []
        for i, row in enumerate(batch.rows):
            mine = sorted((r for r in batch_records if r["row"] == row.name), key=lambda r: r["level"])
            last = mine[-1] if mine else None
            rows.append({
                "row": row.name, "seed": row.seed, "pre": perturb.describe(row.pre),
                "post": perturb.describe(row.post), "beta0": row.beta0, "floor": row.floor,
                "up_floor": row.up_floor, "start_score": start_score,
                "levels": [{"level": r["level"], "n": r["n"], "init_exact": r["init_exact"], "score": r["score"],
                            "evals": r["evals_done"]} for r in mine],
                "last_id": last and last["experiment_id"], "last_score": last and last["score"],
                "last_is_final": bool(last and last["final"]),
                "gain_over_start": last and last["gain_over_start"],
                "gain_over_live": last and last["gain_over_live"],
                "profile_vs_frozen": last and last.get(f"profile_l1_{PROFILE_BLOCK}_vs_frozen"),
                "profile_vs_live": last and last.get(f"profile_l1_{PROFILE_BLOCK}_vs_live")})
        summary_batches.append({
            "batch": batch.name, "experiment": batch.experiment, "start": batch.start, "pool": batch.pool,
            "ups": list(batch.ups), "level_sizes": pl.level_sizes(batch), "level_evals": list(batch.level_evals),
            "evals_done": [lv["evals"] for lv in out["levels"]],
            "s_per_eval": [lv["s_per_eval"] for lv in out["levels"]],
            "repeat_gap_max": max((lv["repeat_gap"] for lv in out["levels"] if lv["repeat_gap"] is not None),
                                  default=None),
            "stopped_early": out["stopped_early"], "rows": rows})
        write_summary(False)
        if out["stopped_early"]:
            skipped += [b.name for b in batches[batches.index(batch) + 1:]]
            break
    pool.shutdown()
    return write_summary(True)
