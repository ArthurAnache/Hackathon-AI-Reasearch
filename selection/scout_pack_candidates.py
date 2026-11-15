"""Problem scout: copy third-autocorrelation candidates out of the git-ignored cache, with a manifest.

Usage (from the repo root):

    .venv/bin/python selection/scout_pack_candidates.py TAG [TAG ...]                     # local runs (track A)
    .venv/bin/python selection/scout_pack_candidates.py --track B --round r1 ID [ID ...]  # Modal runs

TAG is the last part of an experiment_id in selection/scout_third_autocorr_runs.jsonl (local runs, all of track A).
With --track and --round, ID is a job id of selection/scout_third_autocorr_track{A,B}_runs.jsonl.

For each one it writes selection/scout_candidates/third-autocorrelation-inequality/<name>.json.gz (the solution,
`{"values": [...]}`) and an entry in MANIFEST.json there: track and provenance, sha256 of the uncompressed JSON,
the pinned-verifier score recorded by the run, the exact rational score recomputed here, and their difference.

Track A ("milo_published_copy") candidates derive from the copy of MILO's record published on its project page;
that copy is stored too, so the experiment can be replayed. Track B ("public_leaderboard_only") candidates start
from EinsteinArena leaderboard solutions only. All of them are candidates and evidence, not claimed records, and
they are not part of the frozen catalog `problems/`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scout_third_autocorr_exact import exact_score  # noqa: E402

SLUG = "third-autocorrelation-inequality"
OUT_DIR = ROOT / "scout_candidates" / SLUG
CACHE = ROOT / ".cache" / "scout" / SLUG
DELTA = 1e-5
PROVENANCE = {
    "A": ("milo_published_copy: re-optimized by selection/scout_third_autocorr.py from the plot-resolution copy of "
          "MILO's record published at https://jprithwish.github.io/MILO/static/einstein-run.js (Jana et al., "
          "arXiv:2609.38349). Any use must credit MILO for the starting construction."),
    "B": ("public_leaderboard_only: re-optimized from EinsteinArena leaderboard solutions (2617, Atris; 2609, "
          "Ghazt) with no MILO data. Any use must credit the authors of the starting solution."),
}


def gz_write(path: Path, raw: bytes) -> None:
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as fh:  # mtime=0: reproducible bytes
        fh.write(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["A", "B"])
    ap.add_argument("--round")
    ap.add_argument("tags", nargs="+")
    args = ap.parse_args()
    modal_run = bool(args.track and args.round)
    track = args.track or "A"
    log = ROOT / (f"scout_third_autocorr_track{track}_runs.jsonl" if modal_run else "scout_third_autocorr_runs.jsonl")
    runs = {json.loads(l)["experiment_id"].split("/")[-1]: json.loads(l) for l in log.read_text().splitlines()
            if not modal_run or f"/{args.round}/" in json.loads(l)["experiment_id"]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    milo = json.loads((ROOT / "scout_milo_replay.json").read_text())
    live = next(json.loads(l) for l in (ROOT / "scout_live.jsonl").read_text().splitlines()
                if json.loads(l)["slug"] == SLUG)
    live1 = live["top1"]["score"]
    manifest_path = OUT_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"candidates": {}}
    manifest.pop("provenance", None)
    manifest.update({
        "slug": SLUG, "delta": DELTA,
        "status": "candidates only: not submitted, not arbitrated by selection/verify_claim.py, not a record claim",
        "provenance_by_track": PROVENANCE,
        "live_top1_when_packed": {"score": live1, "agent": live["top1"]["agent"], "read_on": live["checked_on"]},
        "milo_claimed_record": milo["claimed_record"],
    })
    for c in manifest["candidates"].values():  # entries written before the tracks existed are all track A
        c.setdefault("track", "A")
        c.setdefault("provenance", "milo_published_copy")
    seed = CACHE / "milo" / "milo_record_3200.json"
    if seed.exists():
        raw = seed.read_bytes()
        gz_write(OUT_DIR / "milo_published_copy_3200.json.gz", raw)
        manifest["milo_published_copy"] = {
            "file": "milo_published_copy_3200.json.gz", "sha256_json": hashlib.sha256(raw).hexdigest(),
            "what": "3,200 of the 25,600 values of MILO's record, 6 decimals, as published in the demo data file",
            "source_url": milo["source_url"], "accessed_on": milo["accessed_on"],
            "source_file_sha256": milo["raw_sha256"],
            "score_as_published": milo["published_copy"]["record"]["score_as_published"]}
    for tag in args.tags:
        run = runs[tag]
        if not run.get("valid") or run.get("score") is None:
            raise SystemExit(f"{tag}: no pinned-verifier score recorded for this run")
        raw = (ROOT.parent / run["solution_path"]).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != run["solution_sha256"]:
            raise SystemExit(f"{tag}: the cached solution changed since the run (sha256 mismatch)")
        values = json.loads(raw)["values"]
        score, lag = exact_score(values)
        name = f"track{track}_{args.round}_{tag}" if modal_run else tag
        gz_write(OUT_DIR / f"{name}.json.gz", raw)
        manifest["candidates"][name] = {
            "file": f"{name}.json.gz", "track": track,
            "provenance": "milo_published_copy" if track == "A" else "public_leaderboard_only",
            "n": len(values), "sha256_json": sha,
            "score_pinned_verifier": run["score"], "verifier_sha256": run["verifier_sha256"],
            "score_exact_rational": float(score), "pinned_minus_exact": run["score"] - float(score),
            "argmax_lag": lag,
            "below_live_top1_by": live1 - run["score"], "below_live_top1_by_over_delta": (live1 - run["score"]) / DELTA,
            "below_milo_claim_by": milo["claimed_record"] - run["score"],
            "below_milo_claim_by_over_delta": (milo["claimed_record"] - run["score"]) / DELTA,
            "run": {k: run.get(k) for k in ("experiment_id", "checked_on", "start", "start_key",
                                            "start_sha256_float64", "upsample", "up_noise", "config", "seed",
                                            "budget_s", "wall_s", "device") if run.get(k) is not None},
        }
        c = manifest["candidates"][name]
        print(f"{name}: track {track} n={c['n']} pinned {c['score_pinned_verifier']!r} exact "
              f"{c['score_exact_rational']!r} below live #1 by {c['below_live_top1_by_over_delta']:.2f} delta, "
              f"below MILO's claim by {c['below_milo_claim_by_over_delta']:.2f} delta")
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
