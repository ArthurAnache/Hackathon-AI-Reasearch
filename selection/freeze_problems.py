"""WS1: freeze the chosen problems into problems/<slug>/ (verifier, certificate, meta.json).

For each problem it:
  1. takes the pinned verifier extracted by stage 2 (selection/.cache/stage2/<slug>/verifier_pinned.py), checks its
     sha256 against registry_stage2.jsonl, and checks it is byte-identical to the verifier served live;
  2. re-reads the live leaderboard and refuses to freeze if the #1 is no longer the record cached by stage 2
     (re-run selection/stage2_profile.py first);
  3. writes verifier_pinned.py, certificate.json.gz (the #1 solution bytes, gzip with mtime 0) and meta.json.

Existing frozen files are never overwritten unless --refreeze is passed. A refreeze changes the frozen hashes, so the
test in tests/test_ws1_frozen_problems.py must be updated in the same commit, with Arthur's approval.

Usage: .venv/bin/python selection/freeze_problems.py [--refreeze] [slug ...]
       .venv/bin/python selection/freeze_problems.py --manifest     # rewrite problems/MANIFEST.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from integrations import einstein_arena as ea  # noqa: E402

CACHE = ROOT / ".cache" / "stage2"
REGISTRY = ROOT / "registry_stage2.jsonl"
PROBLEMS = REPO_ROOT / "problems"
EA_REPO = "https://github.com/vinid/einstein-arena"
EA_COMMIT = "5baf6361a3e0e2123edb98f676ef44f7ab081ad6"

ROLES = {
    "second-autocorrelation-inequality": "main",
    "circles-rectangle": "backup",
    "ring-loading-15": "record_bet",
}
LIMITS = {
    "second-autocorrelation-inequality": {
        "max_values": 2_000_000, "inline_json_limit_chars": 2_000_000, "blob_max_bytes": 50 * 1024 * 1024,
        "notes": "Above ~2 MB of compact JSON, submit through POST /api/solutions/upload-url (blob). "
                 "The score is invariant under f -> a*f.",
    },
    "circles-rectangle": {
        "n_circles": 21, "inline_json_limit_chars": 2_000_000,
        "notes": "No overlap tolerance (float dist < r_i + r_j rejects); perimeter checked in exact Fractions.",
    },
    "ring-loading-15": {
        "n_pairs": 15, "max_string_chars": 80, "max_reduced_bits": 64, "inline_json_limit_chars": 2_000_000,
        "notes": "Exact verifier; delta = 0, so only a strictly larger value than #1 counts.",
    },
}
FORMATS = {
    "second-autocorrelation-inequality": '{"values": [f_0, ..., f_{n-1}]}, 1 <= n <= 2,000,000 numbers >= -1e-6 '
                                         '(clipped at 0), positive sum',
    "circles-rectangle": '{"circles": [[x, y, r] x 21]}, r > 0, bounding box width + height <= 2, no overlap',
    "ring-loading-15": '{"pairs": [["u", "v"] x 15]}, nonnegative decimal or fraction strings, u + v <= 1',
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def freeze(slug: str, refreeze: bool) -> None:
    out = PROBLEMS / slug
    targets = [out / "verifier_pinned.py", out / "certificate.json.gz", out / "meta.json"]
    if any(p.exists() for p in targets) and not refreeze:
        sys.exit(f"{slug}: already frozen; pass --refreeze (with Arthur's approval) to replace it")
    row = next(json.loads(line) for line in REGISTRY.read_text().splitlines() if json.loads(line)["slug"] == slug)

    verifier = (CACHE / slug / "verifier_pinned.py").read_bytes()
    if sha(verifier) != row["verifier_pinned_sha256"]:
        sys.exit(f"{slug}: cached pinned verifier does not match registry_stage2.jsonl")
    live = ea.get_problem(slug)
    if sha(live["verifier"].encode()) != sha(verifier):
        sys.exit(f"{slug}: live verifier differs from the pinned one; do not freeze")
    checked_at = now()

    lb = ea.get_leaderboard(live["id"], limit=2)
    rec = row["record"]
    if lb[0]["agentName"] != rec["agent"] or float(lb[0]["bestScore"]) != float(rec["value"]):
        sys.exit(f"{slug}: live #1 is now {lb[0]['agentName']} {lb[0]['bestScore']!r}, not the stage 2 record; "
                 f"re-run selection/stage2_profile.py {slug} first")
    solution = (CACHE / slug / "record_solution.json").read_bytes()
    if sha(solution) != rec["solution_sha256"]:
        sys.exit(f"{slug}: cached record solution does not match its registry sha256")

    meta = {
        "slug": slug,
        "problem_id": live["id"],
        "title": live["title"],
        "url": f"https://einsteinarena.com/problems/{slug}",
        "role": ROLES[slug],
        "direction": live["scoring"],
        "delta": live["minImprovement"],
        "solution_format": FORMATS[slug],
        "limits": LIMITS[slug],
        "verifier": {
            "file": "verifier_pinned.py",
            "sha256": sha(verifier),
            "source_repo": EA_REPO,
            "commit": EA_COMMIT,
            "source_path": f"web/src/lib/problems/{slug}.ts",
            "source_url": f"{EA_REPO}/blob/{EA_COMMIT}/web/src/lib/problems/{slug}.ts",
            "harness": "server prefixes 'import json' and calls evaluate(solution) (web/src/app/api/evaluate/route.ts)",
            "live_identical": True,
            "live_checked_at": checked_at,
        },
        "record": {
            "value": float(lb[0]["bestScore"]),
            "agent": lb[0]["agentName"],
            "solution_id": rec["solution_id"],
            "submitted_at": rec["solution_created_at"],
            "runner_up": {"agent": lb[1]["agentName"], "value": float(lb[1]["bestScore"])} if len(lb) > 1 else None,
            "source": f"{ea.base_url()}/api/leaderboard?problem_id={live['id']}",
            "read_at": checked_at,
            "note": "Snapshot only. The record to beat is always the LIVE #1 (problem.best_known_score()).",
        },
        "certificate": {
            "file": "certificate.json.gz",
            "sha256": sha(solution),
            "sha256_of": "decompressed bytes, identical to GET /api/solutions/best for this solution",
            "agent": rec["solution_agent"],
            "solution_id": rec["solution_id"],
            "displayed_score": rec["solution_score"],
            "bytes": len(solution),
        },
        "frozen_on": checked_at,
        "frozen_by": "ws1-selection",
        "decision": "selection/decision.md",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "verifier_pinned.py").write_bytes(verifier)
    (out / "certificate.json.gz").write_bytes(gzip.compress(solution, compresslevel=9, mtime=0))
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{slug}: frozen ({ROLES[slug]}), #1 {meta['record']['agent']} = {meta['record']['value']!r}, "
          f"verifier {meta['verifier']['sha256'][:12]}, certificate {meta['certificate']['sha256'][:12]}")


MANIFEST = PROBLEMS / "MANIFEST.json"
SHARED_FILES = ["_frozen.py", "catalog.py", "README.md"]


def manifest_files() -> list[Path]:
    """Every WS1 file under problems/: the shared files and everything in the frozen problem directories."""
    files = [PROBLEMS / f for f in SHARED_FILES]
    for slug in ROLES:
        files += sorted(p for p in (PROBLEMS / slug).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return files


def write_manifest() -> None:
    entries = {p.relative_to(REPO_ROOT).as_posix(): sha(p.read_bytes()) for p in manifest_files()}
    MANIFEST.write_text(json.dumps({"generated_at": now(), "generated_by": "ws1-selection", "sha256": entries},
                                   indent=2) + "\n")
    print(f"problems/MANIFEST.json: {len(entries)} files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("slugs", nargs="*", default=list(ROLES))
    ap.add_argument("--refreeze", action="store_true")
    ap.add_argument("--manifest", action="store_true",
                    help="only rewrite problems/MANIFEST.json (sha256 of every WS1 file under problems/)")
    args = ap.parse_args()
    if args.manifest:
        write_manifest()
        return
    for slug in args.slugs:
        freeze(slug, args.refreeze)


if __name__ == "__main__":
    main()
