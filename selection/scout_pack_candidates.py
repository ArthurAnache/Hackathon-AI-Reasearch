"""Problem scout: copy the third-autocorrelation candidates out of the git-ignored cache, with a manifest.

Usage (from the repo root):  .venv/bin/python selection/scout_pack_candidates.py TAG [TAG ...]

For each run tag of selection/scout_third_autocorr_runs.jsonl it writes
selection/scout_candidates/third-autocorrelation-inequality/<tag>.json.gz (the solution, `{"values": [...]}`) and
adds an entry to MANIFEST.json there: sha256 of the uncompressed JSON, the pinned-verifier score recorded by the
run, and the exact rational score recomputed here with selection/scout_third_autocorr_exact.py.

It also stores the 3,200 published values the candidates were re-optimized from (MILO's project page), so the
experiment can be replayed if that page changes. These files are candidates and evidence, not claimed records, and
they are not part of the frozen catalog `problems/`.
"""

from __future__ import annotations

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


def gz_write(path: Path, raw: bytes) -> None:
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as fh:  # mtime=0: reproducible bytes
        fh.write(raw)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {json.loads(l)["experiment_id"].split("/")[-1]: json.loads(l)
            for l in (ROOT / "scout_third_autocorr_runs.jsonl").read_text().splitlines()}
    milo = json.loads((ROOT / "scout_milo_replay.json").read_text())
    live = next(json.loads(l) for l in (ROOT / "scout_live.jsonl").read_text().splitlines()
                if json.loads(l)["slug"] == SLUG)
    live1 = live["top1"]["score"]
    manifest_path = OUT_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"candidates": {}}
    manifest.update({
        "slug": SLUG, "delta": DELTA,
        "status": "candidates only: not submitted, not arbitrated by selection/verify_claim.py, not a record claim",
        "provenance": ("Re-optimized by selection/scout_third_autocorr.py from the plot-resolution copy of MILO's "
                       "record published at " + milo["source_url"] + " (Jana et al., arXiv:2609.38349). Any use "
                       "must credit MILO for the starting construction."),
        "live_top1_when_packed": {"score": live1, "agent": live["top1"]["agent"], "read_on": live["checked_on"]},
        "milo_claimed_record": milo["claimed_record"],
    })
    seed = CACHE / "milo" / "milo_record_3200.json"
    if seed.exists():
        raw = seed.read_bytes()
        gz_write(OUT_DIR / "milo_published_copy_3200.json.gz", raw)
        manifest["milo_published_copy"] = {
            "file": "milo_published_copy_3200.json.gz", "sha256_json": hashlib.sha256(raw).hexdigest(),
            "what": "3,200 of the 25,600 values of MILO's record, 6 decimals, as published in the demo data file",
            "source_url": milo["source_url"], "accessed_on": milo["accessed_on"],
            "source_file_sha256": milo["raw_sha256"], "score_as_published": milo["published_copy"]["record"]["score_as_published"]}
    for tag in sys.argv[1:]:
        run = runs[tag]
        raw = (ROOT.parent / run["solution_path"]).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != run["solution_sha256"]:
            raise SystemExit(f"{tag}: the cached solution changed since the run (sha256 mismatch)")
        values = json.loads(raw)["values"]
        score, lag = exact_score(values)
        gz_write(OUT_DIR / f"{tag}.json.gz", raw)
        manifest["candidates"][tag] = {
            "file": f"{tag}.json.gz", "n": len(values), "sha256_json": sha,
            "score_pinned_verifier": run["score"], "verifier_sha256": run["verifier_sha256"],
            "score_exact_rational": float(score), "pinned_minus_exact": run["score"] - float(score),
            "argmax_lag": lag,
            "below_live_top1_by": live1 - run["score"], "below_live_top1_by_over_delta": (live1 - run["score"]) / DELTA,
            "below_milo_claim_by": milo["claimed_record"] - run["score"],
            "below_milo_claim_by_over_delta": (milo["claimed_record"] - run["score"]) / DELTA,
            "run": {k: run.get(k) for k in ("experiment_id", "checked_on", "start", "upsample", "up_noise", "config",
                                            "budget_s", "wall_s", "device")},
        }
        c = manifest["candidates"][tag]
        print(f"{tag}: n={c['n']} pinned {c['score_pinned_verifier']!r} exact {c['score_exact_rational']!r} "
              f"below live #1 by {c['below_live_top1_by_over_delta']:.2f} delta, "
              f"below MILO's claim by {c['below_milo_claim_by_over_delta']:.2f} delta")
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
