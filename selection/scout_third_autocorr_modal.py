"""Modal CPU runs for the third-autocorrelation scout: many cooling paths, re-melt hops and resolution steps.

Two tracks, kept apart in files, logs, Modal calls and on the Volume:

- Track A, provenance "milo_published_copy": starts are MILO's published 3,200-value copy (block-repeated x8) and
  the candidates re-optimized from it.
- Track B, provenance "public_leaderboard_only": starts are the live #1 (solution 2617, Atris) and #2 (solution
  2609, Ghazt) of EinsteinArena. No MILO data is uploaded under trackB/ and no Track B job reads trackA/.

Usage (from the repo root), one track per call:

    .venv/bin/modal run --detach selection/scout_third_autocorr_modal.py::launch --round-name r1 --track A --budget 1200
    .venv/bin/modal run --detach selection/scout_third_autocorr_modal.py::launch --round-name r1 --track B --budget 1200
    .venv/bin/modal volume get scout-third-autocorr /trackA/r1 selection/.cache/scout/third-autocorrelation-inequality/modal/trackA/
    .venv/bin/python selection/scout_third_autocorr_modal.py harvest --round-name r1 --keep 3

`launch` uploads the track's start vectors to the Volume `scout-third-autocorr` and spawns one container that runs
all the jobs of that track, one process and one core per job; each job checkpoints its best point to the Volume.
`harvest` re-scores the best points of each downloaded track with the pinned verifier in a clean process and with
the exact-rational scorer, and appends one line per job to selection/scout_third_autocorr_track{A,B}_runs.jsonl.

The container runs trusted repository code (selection/scout_third_autocorr.py) with no secret. It never submits
and never calls EinsteinArena. Cost: about 0.047 USD per core-hour plus 0.008 USD per GiB-hour.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
IMAGE_ROOT = "/root"
VOLUME_NAME = "scout-third-autocorr"
SLUG = "third-autocorrelation-inequality"
CACHE = HERE / ".cache" / "scout" / SLUG
PROVENANCE = {"A": "milo_published_copy", "B": "public_leaderboard_only"}
USD_PER_CORE_H, USD_PER_GIB_H = 0.047, 0.008
BATCH_CPU, BATCH_MEMORY_MB = 48, 24 * 1024  # one container per track and round: 48 cores, at most 48 jobs

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.4", "scipy==1.18.1")  # same versions as the local .venv
    .env({"PYTHONPATH": IMAGE_ROOT, "PYTHONDONTWRITEBYTECODE": "1"})
    # Laid out like the repository: /root/selection/scout_third_autocorr.py
    .add_local_file(HERE / "scout_third_autocorr.py", f"{IMAGE_ROOT}/selection/scout_third_autocorr.py")
)
app = modal.App("scout-third-autocorr", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(cpu=BATCH_CPU, memory=BATCH_MEMORY_MB, timeout=3000, max_containers=4, volumes={"/vol": volume})
def run_batch(jobs: list[dict]) -> dict:
    """All the jobs of one track and one round, one process per job (one core each), in a single container.

    Why one big container: the workspace's container limit was used up by the team's other campaigns, so one-core
    containers stayed queued. A detached `modal run` also keeps only the last triggered function alive, which
    this single call is. Children write plain files under /vol; this process commits the Volume."""
    import multiprocessing as mp
    import os
    import threading

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # Modal exports cpu=48 here
        os.environ[name] = "1"
    sys.path.insert(0, f"{IMAGE_ROOT}/selection")
    import scout_third_autocorr as core

    assert len({j["track"] for j in jobs}) == 1, "one track per batch"
    assert len(jobs) <= BATCH_CPU, "one core per job"
    for j in jobs:  # paths inside this track's folder only
        j["start_path"] = f"/vol/track{j['track']}/starts/{j['start_key']}.npy"
        j["out_dir"] = f"/vol/track{j['track']}/{j['round']}"
        j["device"] = f"Modal CPU, 1 of {BATCH_CPU} cores"
    stop = threading.Event()

    def committer():  # make partial results visible to `modal volume get` while the batch runs
        while not stop.wait(90):
            try:
                volume.commit()
            except Exception:  # noqa: BLE001
                pass

    marker = Path(jobs[0]["out_dir"])
    marker.mkdir(parents=True, exist_ok=True)
    (marker / "_started.txt").write_text(f"{len(jobs)} jobs\n")
    volume.commit()
    print(f"batch started: {len(jobs)} jobs of track {jobs[0]['track']}, round {jobs[0]['round']}", flush=True)
    threading.Thread(target=committer, daemon=True).start()
    try:
        with mp.get_context("fork").Pool(len(jobs)) as pool:
            results = pool.map(core.run_job_to_dir, jobs, chunksize=1)
    finally:
        stop.set()
        volume.commit()
    return {"done": len(results), "best": min(r["score_fft"] for r in results)}


# ----------------------------------------------------------------------------------------------------------------
# Plans

def _hops(t_hot, sigma, hop_seconds, stages=12, t_cold=1e-8):
    return {"op": "hops", "t_hot": list(t_hot), "sigma": list(sigma), "t_cold": t_cold, "stages": stages,
            "hop_seconds": hop_seconds}


def _path(t_hot, seconds, sigma=0.0, keep_start=True, stages=14, t_cold=1e-8):
    return {"op": "path", "t_hot": t_hot, "t_cold": t_cold, "stages": stages, "seconds": seconds, "sigma": sigma,
            "keep_start": keep_start}


def plan_r1(track: str, budget: float) -> list[dict]:
    jobs = []

    def add(label, start_key, steps, seed):
        jobs.append({"track": track, "label": label, "start_key": start_key, "steps": steps, "seed": seed})

    if track == "A":  # from MILO's published copy and the candidates derived from it
        hot = [3e-3, 5e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2, 7e-2, 1e-1, 2e-1]
        for i, t in enumerate(hot):  # direct paths from the copy, then hops from the landing
            for j, sigma in enumerate((0.0, 3e-3)):
                add("direct", "milo_record_x8_25600",
                    [_path(t, 0.4 * budget, sigma=sigma, keep_start=False, stages=16),
                     _hops((3e-4, 3e-3), (3e-4, 3e-3), 0.18 * budget)], 1000 + 10 * i + j)
        cands = ["cand_c_remelt", "cand_c", "cand_e", "cand_d"]
        k = 0
        for t in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):  # hop chains from the best candidates, one setting per job
            for sigma in (1e-4, 1e-3, 1e-2):
                add("hops", cands[k % len(cands)], [_hops((t, t), (sigma, sigma), 0.18 * budget)], 2000 + k)
                k += 1
        for t in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-4):  # longer hops, mixed noise, from the best candidate
            add("longhops", "cand_c_remelt", [_hops((t, t), (3e-4, 3e-3), 0.3 * budget, stages=14)], 2100 + k)
            k += 1
        for i, (rep, t) in enumerate([(2, 1e-4), (2, 1e-3), (2, 3e-3), (3, 1e-4), (3, 1e-3), (3, 3e-3)]):
            add(f"repeat{rep}", "cand_c_remelt",  # resolution steps: 51,200 and 76,800 values (limit 100,000)
                [{"op": "repeat", "k": rep}, _path(t, 0.5 * budget, sigma=1e-4),
                 _hops((1e-4, 1e-3), (1e-4, 1e-3), 0.25 * budget)], 3000 + i)
        return jobs

    # Track B: from the live #1 and #2 only
    for s, key in enumerate(("live1_2617_atris", "live2_2609_ghazt")):
        for i, t in enumerate((1e-3, 3e-3, 1e-2, 3e-2, 1e-1)):  # hot re-melt at the native 65,536 values
            add("hot", key, [_path(t, 0.6 * budget, keep_start=False, stages=16),
                             _hops((3e-4, 3e-3), (3e-4, 3e-3), 0.35 * budget)], 5000 + 100 * s + 10 * i)
        for i, kpool in enumerate((2, 4, 8)):  # coarse to fine: sum-pool, re-optimize, repeat x2, re-optimize, ...
            levels = {2: 1, 4: 2, 8: 3}[kpool]
            share = 0.9 / (levels + 1)
            for j, t in enumerate((1e-3, 1e-2)):
                for r in range(2):
                    steps = [{"op": "pool", "k": kpool},
                             _path(t, share * budget, sigma=1e-3 * r, keep_start=False, stages=14)]
                    for _ in range(levels):
                        steps += [{"op": "repeat", "k": 2}, _path(1e-4, share * budget, sigma=1e-4, stages=12)]
                    add(f"c2f_pool{kpool}", key, steps, 6000 + 1000 * s + 100 * i + 10 * j + r)
        hop_set = [(1e-4, 1e-3), (1e-3, 1e-3), (1e-2, 1e-3), (1e-4, 1e-2), (1e-3, 1e-2)][: 5 if s == 0 else 3]
        for i, (t, sigma) in enumerate(hop_set):
            add("hops", key, [_hops((t, t), (sigma, sigma), 0.3 * budget)], 8000 + 100 * s + i)
    for i, (m, t) in enumerate([(100_000, 1e-4), (100_000, 1e-3), (25_600, 1e-2), (25_600, 3e-2),
                                (51_200, 1e-2), (16_384, 1e-2)]):  # other grids, by exact block averaging
        add(f"resample{m}", "live1_2617_atris",
            [{"op": "resample", "n": m}, _path(t, 0.55 * budget, keep_start=False, stages=16),
             _hops((3e-4, 3e-3), (3e-4, 3e-3), 0.35 * budget)], 9000 + i)
    return jobs


def starts_for(track: str) -> dict[str, Path]:
    """Start vectors of one track, as local solution files. Track B lists leaderboard solutions only."""
    runs = CACHE / "runs"
    if track == "A":
        out = {"milo_record_x8_25600": CACHE / "milo" / "milo_record_x8_25600.json"}
        for key, tag in (("cand_c_remelt", "milo_replay_25600_c_remelt"), ("cand_c", "milo_replay_25600_c"),
                         ("cand_e", "milo_replay_25600_e"), ("cand_d", "milo_replay_25600_d")):
            out[key] = runs / tag / "best.json"
    else:
        out = {"live1_2617_atris": CACHE / "top" / "2617.json", "live2_2609_ghazt": CACHE / "top" / "2609.json"}
    for p in sorted((CACHE / "modal" / f"track{track}").glob("*/kept/*.json")):  # kept results of earlier rounds
        out[f"kept_{p.parent.parent.name}_{p.stem}"] = p
    return out


@app.local_entrypoint()
def launch(round_name: str, track: str, budget: float = 1200.0, plan: str = "r1"):
    import io

    import numpy as np

    jobs = plan_r1(track, budget) if plan == "r1" else json.loads(Path(plan).read_text())
    if any(j["track"] != track for j in jobs) or len(jobs) > BATCH_CPU:
        raise SystemExit(f"the plan must hold at most {BATCH_CPU} jobs, all of track {track}")
    starts = starts_for(track)
    with volume.batch_upload(force=True) as batch:
        for key in sorted({j["start_key"] for j in jobs}):
            values = np.asarray(json.loads(starts[key].read_text())["values"], dtype=np.float64)
            buf = io.BytesIO()
            np.save(buf, values)
            buf.seek(0)
            batch.put_file(buf, f"/track{track}/starts/{key}.npy")
    for i, job in enumerate(jobs):
        job.update(id=f"{track}{i:03d}_{job['label']}", round=round_name, budget_s=budget,
                   provenance=PROVENANCE[track])
    hours = (budget + 90) / 3600
    usd = hours * (BATCH_CPU * USD_PER_CORE_H + BATCH_MEMORY_MB / 1024 * USD_PER_GIB_H)
    print(f"round {round_name} track {track}: {len(jobs)} jobs x {budget:.0f}s on one {BATCH_CPU}-core container, "
          f"about {usd:.2f} USD")
    (CACHE / "modal").mkdir(parents=True, exist_ok=True)
    (CACHE / "modal" / f"{round_name}_track{track}_jobs.json").write_text(json.dumps(jobs))
    call = run_batch.spawn(jobs)
    print(f"spawned {call.object_id}; results: modal volume get {VOLUME_NAME} /track{track}/{round_name}")


# ----------------------------------------------------------------------------------------------------------------
# Harvest (plain Python; the round is downloaded first with `modal volume get`)

def harvest(round_name: str, keep: int, dry: bool = False) -> None:
    import datetime as dt

    import numpy as np

    sys.path.insert(0, str(HERE))
    import scout_third_autocorr as core
    from scout_third_autocorr_exact import exact_score

    live = next(json.loads(l) for l in (HERE / "scout_live.jsonl").read_text().splitlines()
                if json.loads(l)["slug"] == SLUG)
    live1 = live["top1"]["score"]
    milo = json.loads((HERE / "scout_milo_replay.json").read_text())["claimed_record"]
    for track in ("A", "B"):
        folder = CACHE / "modal" / f"track{track}" / round_name
        metas = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json")) if ".tmp" not in p.name]
        if not metas:
            continue
        metas.sort(key=lambda m: m["score_fft"])
        kept_dir = folder / "kept"
        kept_dir.mkdir(exist_ok=True)
        log_path = HERE / f"scout_third_autocorr_track{track}_runs.jsonl"
        rows = []
        for rank, m in enumerate(metas):
            row = {
                "experiment_id": f"scout-third-autocorr/track{track}/{round_name}/{m['id']}", "slug": SLUG,
                "track": track, "provenance": m["provenance"], "label_job": m["label"], "start_key": m["start_key"],
                "start_sha256_float64": m["start_sha256_float64"],
                "checked_on": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "hypothesis": "paths, hops and resolution steps lower the score further"
                if track == "A" else "a score at least delta below the live #1 is reachable without MILO's data",
                "config": {"steps": m["steps"]}, "seed": m["seed"], "budget_s": m["budget_s"], "wall_s": m["wall_s"],
                "final": m["final"], "device": m["device"], "n": m["n"], "score_fft": m["score_fft"],
                "best_known_score": live1, "delta": core.DELTA,
                "below_live_top1_over_delta_fft": (live1 - m["score_fft"]) / core.DELTA,
                "hops": sum(1 for t in m["trace"] if t["op"] == "hop"),
                "hops_accepted": sum(1 for t in m["trace"] if t["op"] == "hop" and t["accepted"]),
                "trace_tail": m["trace"][-6:], "valid": None, "score": None,
                "evidence_urls": ["https://einsteinarena.com/problems/" + SLUG]
                + (["https://jprithwish.github.io/MILO/static/einstein-run.js"] if track == "A" else []),
            }
            if rank < keep:  # re-score the best ones: pinned verifier (clean process) and exact rationals
                values = np.load(folder / f"{m['id']}.npy").tolist()
                sol = kept_dir / f"{m['id']}.json"
                sol.write_text(json.dumps({"values": values}, separators=(",", ":")))
                v = core.verify_clean(sol)
                exact, lag = exact_score(values)
                row.update(valid=bool(v.get("ok")), score=v.get("score"), verifier_sha256=v.get("verifier_sha256"),
                           score_exact_rational=float(exact), pinned_minus_exact=v.get("score") - float(exact),
                           argmax_lag=lag, solution_path=str(sol.relative_to(REPO)),
                           solution_sha256=hashlib.sha256(sol.read_bytes()).hexdigest(),
                           below_live_top1_over_delta=(live1 - v["score"]) / core.DELTA,
                           below_milo_claim_over_delta=(milo - v["score"]) / core.DELTA,
                           label="verified: pinned verifier in a clean process, and exact-rational scorer")
            else:
                row["label"] = "score from the FFT scorer on Modal; not re-scored by the pinned verifier"
            margin = live1 - (row["score"] if row["score"] is not None else row["score_fft"])
            row["decision"] = "candidate" if margin >= core.DELTA else "no_gain"
            row["rationale"] = (f"{margin / core.DELTA:.2f} delta below the live #1; a candidate for WS1 "
                                "arbitration, not a record claim" if row["decision"] == "candidate"
                                else "not delta below the live #1")
            rows.append(row)
        if not dry:  # one line per job; a later harvest of the same round replaces its lines
            merged = ({json.loads(l)["experiment_id"]: json.loads(l) for l in log_path.read_text().splitlines()}
                      if log_path.exists() else {})
            merged.update({row["experiment_id"]: row for row in rows})
            log_path.write_text("".join(json.dumps(r) + "\n" for r in merged.values()))
        print(f"--- track {track} ({PROVENANCE[track]}), round {round_name}: {len(rows)} jobs, "
              f"{sum(m['final'] for m in metas)} finished")
        for row in rows[: max(keep, 14)]:
            pinned = f"pinned {row['score']!r} exact {row.get('score_exact_rational')!r}" if row["score"] else ""
            print(f"  {row['experiment_id'].split('/')[-1]:26s} {row['start_key'][:22]:22s} n={row['n']:6d} "
                  f"fft {row['score_fft']:.12f} {row['below_live_top1_over_delta_fft']:+7.2f} d  hops "
                  f"{row['hops_accepted']}/{row['hops']} {pinned}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["harvest"])
    ap.add_argument("--round-name", required=True)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--dry", action="store_true", help="print the table only: no re-scoring, no log line")
    a = ap.parse_args()
    harvest(a.round_name, 0 if a.dry else a.keep, a.dry)
