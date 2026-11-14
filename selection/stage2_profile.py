"""WS1 stage 2: automatic profile of each candidate kept by stage 1.

For each problem it:
  1. fetches the live problem, leaderboard and #1 solution from the public EinsteinArena API (GETs need no token);
  2. extracts the verifier from the pinned EinsteinArena commit (the version we would freeze) and compares it with
     the verifier served live;
  3. runs the verifier locally on the #1 solution and checks that it reproduces the leaderboard score;
  4. times one evaluation;
  5. records delta (minImprovement), the #1/#2 gap and the date of the #1 solution.

Writes selection/registry_stage2.jsonl (one line per problem). Raw API responses and extracted verifiers go to
selection/.cache/stage2/ (git-ignored) and are referenced by sha256.

Usage: .venv/bin/python selection/stage2_profile.py [slug ...]
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CACHE = ROOT / ".cache" / "stage2"
PINNED_DIR = ROOT / ".cache" / "einstein-arena"
STAGE1_REGISTRY = ROOT / "registry_stage1.jsonl"
OUT = ROOT / "registry_stage2.jsonl"

BASE = "https://einsteinarena.com"
EA_REPO = "https://github.com/vinid/einstein-arena.git"
EA_COMMIT = "5baf6361a3e0e2123edb98f676ef44f7ab081ad6"
EA_BLOB = f"https://github.com/vinid/einstein-arena/blob/{EA_COMMIT}/web/src/lib/problems"
HEADERS = {"User-Agent": "ws1-stage2-profile/1.0"}
SERVER_TIMEOUT_S = 120  # VERIFIER_TIMEOUT_MS in web/src/app/api/evaluate/route.ts
SLOW_EVAL_S = 60.0

# Fallback when selection/registry_stage1.jsonl is not available: the 13 problems stage 1 marked keep_stage2.
STAGE1_KEEP = [
    "first-autocorrelation-inequality",
    "second-autocorrelation-inequality",
    "third-autocorrelation-inequality",
    "erdos-min-overlap",
    "min-distance-ratio-2d",
    "circles-rectangle",
    "heilbronn-triangles",
    "heilbronn-convex",
    "hexagon-packing",
    "difference-bases",
    "ring-loading-15",
    "sum-difference-2",
    "uncertainty-principle",
]

# Added on 2026-11-14 by Arthur: live on the site although the source marks them hidden=true.
# edges-vs-triangles stays eliminated and hadamard-det-51 stays moonshot_only.
STAGE2_ADDED = [
    "circle-packing",
    "tammes-problem",
    "thomson-problem",
    "flat-polynomials",
    "kakeya-needle-128",
]

# Pinned-repo baselines usable when a problem is not live (file, key).
PINNED_BASELINES = {"uncertainty-principle": ("alphaevolve.json", "uncertainty-principle")}


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stage1_slugs() -> tuple[list[str], str]:
    if STAGE1_REGISTRY.exists():
        rows = [json.loads(line) for line in STAGE1_REGISTRY.read_text().splitlines() if line.strip()]
        # Stage 1 stores "problem_id": "einsteinarena/<slug>".
        slugs = [r["problem_id"].split("/", 1)[1] for r in rows
                 if r.get("decision") == "keep_stage2" and r.get("problem_id", "").startswith("einsteinarena/")]
        return slugs + [s for s in STAGE2_ADDED if s not in slugs], str(STAGE1_REGISTRY.relative_to(REPO_ROOT))
    return STAGE1_KEEP + STAGE2_ADDED, "hard-coded STAGE1_KEEP (registry_stage1.jsonl missing)"


def ensure_pinned_repo() -> Path:
    if not (PINNED_DIR / ".git").exists():
        PINNED_DIR.mkdir(parents=True, exist_ok=True)
        for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", EA_REPO],
                    ["git", "fetch", "-q", "--depth", "1", "origin", EA_COMMIT],
                    ["git", "checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(cmd, cwd=PINNED_DIR, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PINNED_DIR, capture_output=True, text=True).stdout.strip()
    if head != EA_COMMIT:
        raise SystemExit(f"pinned repo is at {head}, expected {EA_COMMIT}")
    return PINNED_DIR / "web" / "src" / "lib" / "problems"


def cook_template_literal(ts: str, start: int) -> str:
    """Return the cooked value of the JS template literal whose opening backtick is at ts[start]."""
    escapes = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "b": "\b", "f": "\f", "v": "\v", "\n": ""}
    out, i = [], start + 1
    while True:
        c = ts[i]
        if c == "\\":
            nxt = ts[i + 1]
            if nxt == "u":
                out.append(chr(int(ts[i + 2:i + 6], 16)))
                i += 6
                continue
            out.append(escapes.get(nxt, nxt))
            i += 2
        elif c == "`":
            return "".join(out)
        elif c == "$" and ts[i + 1] == "{":
            raise ValueError("template interpolation in verifier; cannot extract statically")
        else:
            out.append(c)
            i += 1


def pinned_problem(problems_dir: Path, slug: str) -> dict:
    ts = (problems_dir / f"{slug}.ts").read_text()
    marker = "verifier: `"
    verifier = cook_template_literal(ts, ts.index(marker) + len(marker) - 1)

    def field(name: str) -> str | None:
        key = f"\n  {name}: "
        if key not in ts:
            return None
        value = ts.split(key, 1)[1].split("\n", 1)[0].rstrip(",").strip()
        return value.strip('"')

    return {"verifier": verifier, "reference": field("reference"), "hidden": field("hidden"),
            "minImprovement": field("minImprovement"), "scoring": field("scoring")}


def get_json(path: str, params: dict | None = None) -> tuple[int, object]:
    resp = requests.get(f"{BASE}{path}", params=params, headers=HEADERS, timeout=120)
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, resp.text


def run_verifier(verifier_path: Path, solution_path: Path) -> dict:
    cmd = [sys.executable, "-I", str(ROOT / "eval_runner.py"), str(verifier_path), str(solution_path), "5", "30"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10 * SERVER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"local timeout after {10 * SERVER_TIMEOUT_S}s"}
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if not lines:
        return {"ok": False, "error": f"no output (rc={proc.returncode}): {proc.stderr[-800:]}"}
    return json.loads(lines[-1])


def compare(local: float, displayed: float) -> dict:
    diff = abs(local - displayed)
    rel = diff / max(abs(displayed), 1e-300)
    if local == displayed:
        status = "exact"
    elif rel <= 1e-12:
        status = "match_within_1e-12_rel"
    else:
        status = "mismatch"
    return {"status": status, "abs_diff": diff, "rel_diff": rel}


def clearance(new: float, old: float, scoring: str) -> float:
    """How much `new` beats `old` (positive = better), as in web/src/lib/evaluate.ts."""
    return old - new if scoring == "minimize" else new - old


def profile(slug: str, problems_dir: Path) -> dict:
    checked_on = now_utc()
    cache = CACHE / slug
    cache.mkdir(parents=True, exist_ok=True)
    pinned = pinned_problem(problems_dir, slug)
    verifier_path = cache / "verifier_pinned.py"
    verifier_path.write_text(pinned["verifier"])

    row: dict = {
        "slug": slug,
        "stage": 2,
        "checked_on": checked_on,
        "pinned_commit": EA_COMMIT,
        "pinned_hidden_flag": pinned["hidden"],
        "reference": pinned["reference"],
        "verifier_pinned_sha256": sha256(pinned["verifier"]),
        "flags": [],
        "evidence_urls": [f"{EA_BLOB}/{slug}.ts"],
    }

    status, prob = get_json(f"/api/problems/{slug}")
    row["live_status"] = status
    if status != 200:
        row["live"] = False
        row["flags"].append(f"not_live: GET /api/problems/{slug} returned {status}")
        row["direction"] = pinned["scoring"]
        row["delta"] = float(pinned["minImprovement"]) if pinned["minImprovement"] else None
        row["delta_source"] = "pinned source (problem not live)"
        if slug in PINNED_BASELINES:
            fname, key = PINNED_BASELINES[slug]
            base = json.loads((problems_dir.parents[2] / "data" / "baselines" / fname).read_text())[key]
            sol_path = cache / "baseline_solution.json"
            sol_path.write_text(json.dumps(base["solution"]))
            res = run_verifier(verifier_path, sol_path)
            row["baseline"] = {"source": f"pinned repo web/data/baselines/{fname}", "local": res,
                               "eval_median_s": statistics.median(res["times_s"]) if res.get("ok") else None}
            row["evidence_urls"].append(
                f"https://github.com/vinid/einstein-arena/blob/{EA_COMMIT}/web/data/baselines/{fname}")
        row["decision"] = "reserve"
        row["rationale"] = ("Not served by the live EinsteinArena API (404), so there is no live leaderboard to beat "
                            "or submit to; the source code marks it hidden=false, so the stage 1 'hidden' flag is "
                            "wrong for this problem.")
        return row

    (cache / "problem.json").write_text(json.dumps(prob))
    pid, scoring, delta = prob["id"], prob["scoring"], prob["minImprovement"]
    row.update({"live": True, "problem_id": pid, "title": prob["title"], "direction": scoring,
                "delta": delta, "delta_source": "live API minImprovement"})
    row["evidence_urls"] += [f"{BASE}/problems/{slug}", f"{BASE}/api/problems/{slug}",
                             f"{BASE}/api/leaderboard?problem_id={pid}&limit=10",
                             f"{BASE}/api/solutions/best?problem_id={pid}&limit=1"]
    if str(delta) != str(pinned["minImprovement"]) and float(delta) != float(pinned["minImprovement"] or "nan"):
        row["flags"].append(f"delta_differs: live={delta} pinned={pinned['minImprovement']}")
    if scoring != pinned["scoring"]:
        row["flags"].append(f"direction_differs: live={scoring} pinned={pinned['scoring']}")

    live_verifier = prob.get("verifier") or ""
    (cache / "verifier_live.py").write_text(live_verifier)
    row["verifier_live_sha256"] = sha256(live_verifier)
    row["verifier_live_equals_pinned"] = live_verifier.strip() == pinned["verifier"].strip()
    if not row["verifier_live_equals_pinned"]:
        row["flags"].append("verifier_live_differs_from_pinned")

    _, lb = get_json("/api/leaderboard", {"problem_id": pid, "limit": 10})
    (cache / "leaderboard.json").write_text(json.dumps(lb))
    _, best = get_json("/api/solutions/best", {"problem_id": pid, "limit": 1})
    if not lb or not best:
        row["flags"].append("empty_leaderboard")
        row["decision"], row["rationale"] = "flag", "Live problem with no leaderboard entry."
        return row

    top = best[0]
    sol_path = cache / "record_solution.json"
    sol_text = json.dumps(top["data"])
    sol_path.write_text(sol_text)
    s1 = lb[0]["bestScore"]
    s2 = lb[1]["bestScore"] if len(lb) > 1 else None
    row["record"] = {
        "value": s1,
        "agent": lb[0]["agentName"],
        "solution_id": top["id"],
        "solution_agent": top["agentName"],
        "solution_score": top["score"],
        "solution_created_at": top["createdAt"],
        "solution_sha256": sha256(sol_text),
        "solution_bytes": len(sol_text),
        "source": "EinsteinArena live leaderboard",
        "accessed_on": checked_on,
        "label": "unverified",
    }
    if top["agentName"] != lb[0]["agentName"] or top["score"] != s1:
        row["flags"].append("best_solution_not_leaderboard_top")
    row["runner_up"] = {"value": s2, "agent": lb[1]["agentName"] if s2 is not None else None}
    if s2 is not None:
        gap = clearance(s1, s2, scoring)
        row["gap_1_2"] = gap
        row["gap_1_2_over_delta"] = (gap / delta) if delta else None
    row["entries_within_delta_of_1"] = sum(1 for e in lb[1:] if clearance(s1, e["bestScore"], scoring) <= delta)
    row["last_record_date"] = top["createdAt"]
    row["last_record_date_note"] = ("createdAt of the current #1 solution; the API does not expose the date the "
                                    "#1 score was first reached by any agent.")

    runs = {"pinned": run_verifier(verifier_path, sol_path)}
    if not row["verifier_live_equals_pinned"]:
        runs["live"] = run_verifier(cache / "verifier_live.py", sol_path)
    row["local_check"] = {}
    for name, res in runs.items():
        entry = {"ok": res.get("ok", False)}
        if res.get("ok"):
            entry.update(score=res["score"], score_repr=res["score_repr"], deterministic=res["deterministic"],
                         eval_median_s=statistics.median(res["times_s"]), eval_runs=len(res["times_s"]),
                         import_s=res["import_s"], **compare(res["score"], s1))
        else:
            entry["error"] = res.get("error")
        row["local_check"][name] = entry

    main_check = row["local_check"]["pinned"]
    row["eval_s"] = main_check.get("eval_median_s")
    if not main_check["ok"]:
        row["flags"].append("verifier_rejected_record_solution")
    elif main_check["status"] == "mismatch":
        row["flags"].append("score_not_reproduced")
    if main_check.get("ok") and not main_check["deterministic"]:
        row["flags"].append("non_deterministic_verifier")
    if row["eval_s"] is not None and row["eval_s"] > SLOW_EVAL_S:
        row["flags"].append(f"slow_eval: {row['eval_s']:.1f}s > {SLOW_EVAL_S}s")

    reproduced = main_check.get("ok") and main_check["status"] != "mismatch"
    row["record"]["label"] = (f"verified: pinned verifier reproduces the score locally ({main_check['status']}, "
                              f"abs_diff={main_check['abs_diff']:.3g})" if reproduced
                              else "unverified: local verifier did not reproduce the score")

    blocking = [f for f in row["flags"] if not f.startswith("best_solution_not")]
    if blocking:
        row["decision"] = "flag"
        row["rationale"] = "Needs a look before stage 3: " + "; ".join(blocking) + "."
    else:
        row["decision"] = "keep_stage3"
        row["rationale"] = (f"Pinned verifier reproduces the #1 score ({main_check['status']}) in "
                            f"{row['eval_s']:.3g}s; #1-#2 gap = {row.get('gap_1_2')} vs delta = {delta}.")
    return row


def main() -> None:
    slugs, slug_source = stage1_slugs()
    if len(sys.argv) > 1:
        slugs = sys.argv[1:]
    problems_dir = ensure_pinned_repo()
    rows = []
    for slug in slugs:
        print(f"[stage2] {slug} ...", flush=True)
        row = profile(slug, problems_dir)
        row["stage1_source"] = slug_source
        rows.append(row)
        print(f"[stage2] {slug}: {row['decision']} {row['flags']}", flush=True)
    existing = {}
    if OUT.exists() and len(sys.argv) > 1:
        existing = {json.loads(l)["slug"]: l for l in OUT.read_text().splitlines() if l.strip()}
    for row in rows:
        existing[row["slug"]] = json.dumps(row, sort_keys=True)
    OUT.write_text("\n".join(existing.values()) + "\n")
    print(f"[stage2] wrote {OUT.relative_to(REPO_ROOT)} ({len(existing)} rows)")


if __name__ == "__main__":
    main()
