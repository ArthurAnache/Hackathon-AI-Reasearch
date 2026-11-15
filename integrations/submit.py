"""Gated, single-shot submission of one solution to EinsteinArena.

    .venv/bin/python integrations/submit.py SLUG SOLUTION.json --expected-score X            # checks only
    .venv/bin/python integrations/submit.py SLUG SOLUTION.json --expected-score X --confirm  # checks, then sends
    .venv/bin/python integrations/submit.py --status SOLUTION_ID                             # one status GET

SOLUTION.json holds the `solution` object exactly as it would be submitted (e.g. {"pairs": [...]}).

A solution is sent only if ALL gates pass:
  (a) the frozen pinned verifier (problems/<slug>/verifier_pinned.py, sha256-checked against
      registry_stage2.jsonl and byte-identical to the live verifier) gives the expected score (relative 1e-12);
  (b) the independent implementation (problems/<slug>/independent.py, via selection/independent_verifiers.py)
      confirms the pinned score within its a-priori tolerance;
  (c) the leaderboard, re-read just before sending, shows a gain over the current #1 that is >= delta, > 0 (the
      server rejects ties) and > the numerical tolerance of (b); for exact problems the gain is computed in Fractions;
  (d) size: <= 2,000,000 values; a compact JSON body above 1.9 MB goes through the upload-url blob path; above the
      server blob cap (50 MB) it is refused;
  (e) --confirm was passed by hand.
Plus two safety rules: at most 10 sends in the last 30 minutes (from selection/submissions.jsonl), and the same
solution (sha256) is never sent twice for the same problem.

One submission per call, never retried, never looped. The verdict is read later with --status (one GET, at most one
every 10 minutes per solution; the server queue runs every 15-20 minutes).

Every attempt (sent, blocked, error) and every status check is appended to selection/submissions.jsonl with its
reasons. The API key is read from .env by integrations.einstein_arena and never printed or logged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "selection"))
from integrations import einstein_arena as ea  # noqa: E402
import independent_verifiers  # noqa: E402

SELECTION = REPO_ROOT / "selection"
LOG = SELECTION / "submissions.jsonl"
REGISTRY = SELECTION / "registry_stage2.jsonl"
FROZEN_DIR = REPO_ROOT / "problems"
EVAL_RUNNER = SELECTION / "eval_runner.py"

MAX_VALUES = 2_000_000
INLINE_LIMIT_CHARS = 1_900_000   # server rejects inline bodies whose JSON.stringify(solution) >= 2,000,000 chars
BLOB_LIMIT_BYTES = 50 * 1024 * 1024
SEND_LIMIT = (10, 1800)          # submissions per window (s), mirrors the server limit
STATUS_MIN_INTERVAL_S = 600
EXPECTED_REL_TOL = 1e-12


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def log(entry: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_log() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def registry_row(slug: str) -> dict | None:
    if not REGISTRY.exists():
        return None
    for line in REGISTRY.read_text().splitlines():
        row = json.loads(line)
        if row.get("slug") == slug:
            return row
    return None


def count_values(obj) -> int:
    if isinstance(obj, list):
        return sum(count_values(x) for x in obj)
    if isinstance(obj, dict):
        return sum(count_values(x) for x in obj.values())
    return 1


def clearance(new: float, old: float, scoring: str) -> float:
    """How much `new` beats `old` (positive = better), as in web/src/lib/evaluate.ts."""
    return old - new if scoring == "minimize" else new - old


def run_pinned(verifier: Path, solution_path: Path) -> dict:
    cmd = [sys.executable, "-I", str(EVAL_RUNNER), str(verifier), str(solution_path), "1", "600"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        return {"ok": False, "error": f"no output (rc={proc.returncode}): {proc.stderr[-500:]}"}
    return json.loads(lines[-1])


# ------------------------------------------------------------------------------------------------ gates

def gate_a(slug: str, solution_path: Path, expected: float, live: dict | None, row: dict | None) -> dict:
    verifier = FROZEN_DIR / slug / "verifier_pinned.py"   # the frozen copy (sha256-locked by the WS1 tests)
    if row is None:
        return {"ok": False, "reason": f"{slug} not in registry_stage2.jsonl"}
    if not verifier.exists():
        return {"ok": False, "reason": f"{slug} is not frozen in problems/ (no {verifier.relative_to(REPO_ROOT)})"}
    pinned_src = verifier.read_text()
    pinned_sha = sha256_bytes(pinned_src.encode())
    out = {"pinned_sha256": pinned_sha}
    if pinned_sha != row.get("verifier_pinned_sha256"):
        return {**out, "ok": False, "reason": "pinned verifier file does not match the frozen sha256 in the registry"}
    if live is None:
        return {**out, "ok": False, "reason": "live problem could not be read"}
    live_sha = sha256_bytes(live.get("verifier", "").encode())
    out["live_sha256"] = live_sha
    if live_sha != pinned_sha:
        return {**out, "ok": False, "reason": "live verifier differs from the frozen pinned verifier"}
    res = run_pinned(verifier, solution_path)
    if not res.get("ok"):
        return {**out, "ok": False, "reason": f"pinned verifier rejected the solution: {res.get('error')}"}
    score = res["score"]
    out.update(score=score, score_repr=res.get("score_repr"), eval_s=res.get("times_s", [None])[0])
    if not (score == score and abs(score) != float("inf")):
        return {**out, "ok": False, "reason": f"pinned verifier returned {score} (invalid solution)"}
    diff = abs(score - expected)
    out["abs_diff_expected"] = diff
    if diff > EXPECTED_REL_TOL * max(abs(expected), 1e-300) and diff != 0:
        return {**out, "ok": False, "reason": f"pinned score {score!r} != expected {expected!r} (diff {diff:.3e})"}
    return {**out, "ok": True}


def gate_b(slug: str, solution: dict, pinned_score: float | None) -> dict:
    if slug not in independent_verifiers.IMPLEMENTATIONS:
        return {"ok": False, "reason": f"no independent verifier for {slug} (add one to selection/independent_verifiers.py)"}
    try:
        t0 = time.perf_counter()
        res = independent_verifiers.score(slug, solution)
        res["eval_s"] = round(time.perf_counter() - t0, 3)
    except ValueError as exc:
        return {"ok": False, "reason": f"independent verifier judges the solution invalid: {exc}"}
    if pinned_score is None:
        return {**res, "ok": False, "reason": "no pinned score to compare with"}
    diff = abs(res["score"] - pinned_score)
    res["abs_diff_pinned"] = diff
    if diff > res["tolerance_abs"]:
        return {**res, "ok": False,
                "reason": f"independent score {res['score']!r} disagrees with pinned {pinned_score!r} "
                          f"(diff {diff:.3e} > tol {res['tolerance_abs']:.1e})"}
    return {**res, "ok": True}


def gate_c(live: dict | None, row: dict | None, a: dict, b: dict) -> dict:
    if live is None or a.get("score") is None:
        return {"ok": False, "reason": "no live problem or no pinned score"}
    scoring = live["scoring"]
    delta_live = float(live.get("minImprovement", 0))
    delta_reg = float(row.get("delta", 0)) if row else 0.0
    delta = max(delta_live, delta_reg)
    out = {"scoring": scoring, "delta_live": delta_live, "delta_registry": delta_reg, "delta_used": delta}
    try:
        lb = ea.get_leaderboard(live["id"], limit=3)
    except ea.ArenaError as exc:
        return {**out, "ok": False, "reason": f"leaderboard read failed: {exc}"}
    out["read_at"] = now().isoformat(timespec="seconds")
    out["top3"] = [{"rank": r["rank"], "agent": r["agentName"], "score": r["bestScore"]} for r in lb]
    if not lb:
        return {**out, "ok": False, "reason": "empty leaderboard: refusing (nothing to compare with)"}
    top = float(lb[0]["bestScore"])
    tol = float(b.get("tolerance_abs", 0.0)) if b else 0.0
    gain = clearance(a["score"], top, scoring)
    out.update(top_score=top, top_agent=lb[0]["agentName"], gain=gain, numerical_tol=tol)
    if b.get("exact"):
        exact_gain = clearance(Fraction(b["exact"]), Fraction(top), scoring)
        out["gain_exact"] = f"{exact_gain.numerator}/{exact_gain.denominator}"
        if exact_gain <= 0:
            return {**out, "ok": False, "reason": f"exact gain over #1 is {exact_gain} (not strictly positive)"}
    if b.get("score") is not None:
        out["gain_independent"] = clearance(b["score"], top, scoring)
    reasons = []
    for name, g in (("pinned", gain), ("independent", out.get("gain_independent", gain))):
        if not g > 0:
            reasons.append(f"{name} gain {g:.3e} is not > 0 (ties are rejected)")
        elif g < delta:
            reasons.append(f"{name} gain {g:.3e} < delta {delta:.1e}")
        elif not g > tol:
            reasons.append(f"{name} gain {g:.3e} does not exceed the numerical tolerance {tol:.1e}")
    if reasons:
        return {**out, "ok": False, "reason": "; ".join(reasons)}
    return {**out, "ok": True}


def gate_d(solution: dict) -> dict:
    n_values = count_values(solution)
    body = json.dumps(solution, separators=(",", ":"), allow_nan=False)
    size = len(body)
    out = {"n_values": n_values, "json_chars": size}
    if n_values > MAX_VALUES:
        return {**out, "ok": False, "reason": f"{n_values} values > {MAX_VALUES}"}
    if len(body.encode()) > BLOB_LIMIT_BYTES:
        return {**out, "ok": False, "reason": f"{size} bytes > blob cap {BLOB_LIMIT_BYTES}"}
    out["mode"] = "blob" if size > INLINE_LIMIT_CHARS else "inline"
    return {**out, "ok": True}


def rate_and_duplicate(slug: str, sol_sha: str) -> dict:
    entries = read_log()
    cutoff = now().timestamp() - SEND_LIMIT[1]
    recent = [e for e in entries if e.get("kind") == "submission" and e.get("decision") in ("sent", "error_after_send")
              and dt.datetime.fromisoformat(e["timestamp"]).timestamp() >= cutoff]
    dup = [e for e in entries if e.get("kind") == "submission" and e.get("decision") == "sent"
           and e.get("slug") == slug and e.get("solution_sha256") == sol_sha]
    out = {"sent_last_30min": len(recent)}
    if len(recent) >= SEND_LIMIT[0]:
        return {**out, "ok": False, "reason": f"{len(recent)} submissions in the last 30 min (limit {SEND_LIMIT[0]})"}
    if dup:
        return {**out, "ok": False, "reason": f"this exact solution was already sent (solution id "
                                              f"{dup[-1].get('response', {}).get('id')})"}
    return {**out, "ok": True}


# ------------------------------------------------------------------------------------------------ commands

def submit(slug: str, solution_path: Path, expected: float, confirm: bool) -> int:
    raw = solution_path.read_bytes()
    entry = {"timestamp": now().isoformat(timespec="seconds"), "kind": "submission", "slug": slug,
             "solution_file": str(solution_path), "solution_sha256": sha256_bytes(raw), "expected_score": expected,
             "confirm_flag": confirm, "api_key_present": ea.has_api_key()}
    try:
        solution = json.loads(raw)
        if not isinstance(solution, dict):
            raise ValueError("the file must hold the solution object")
    except ValueError as exc:
        entry.update(decision="blocked", reasons=[f"unreadable solution file: {exc}"])
        log(entry)
        print(f"BLOCKED: {entry['reasons'][0]}")
        return 2

    row = registry_row(slug)
    try:
        live = ea.get_problem(slug)
    except ea.ArenaError:
        live = None
    entry["problem_id"] = live.get("id") if live else (row or {}).get("problem_id")

    checks: dict = {}
    checks["a_pinned"] = gate_a(slug, solution_path, expected, live, row)
    checks["b_independent"] = gate_b(slug, solution, checks["a_pinned"].get("score"))
    checks["d_size"] = gate_d(solution)
    checks["rate_and_duplicate"] = rate_and_duplicate(slug, entry["solution_sha256"])
    checks["e_confirm"] = {"ok": confirm, **({} if confirm else {"reason": "--confirm not passed"})}
    # (c) last, so the leaderboard is read as close as possible to the send.
    checks["c_leaderboard"] = gate_c(live, row, checks["a_pinned"], checks["b_independent"])
    entry["checks"] = checks
    reasons = [f"{k}: {v['reason']}" for k, v in checks.items() if not v.get("ok")]
    if not reasons and not ea.has_api_key():
        reasons.append("EINSTEIN_ARENA_API_KEY missing in .env")

    print(f"{slug}  pinned={checks['a_pinned'].get('score')!r}  independent={checks['b_independent'].get('score')!r}")
    c = checks["c_leaderboard"]
    if "top_score" in c:
        print(f"#1 {c['top_agent']} = {c['top_score']!r}   gain = {c['gain']:.3e}   delta = {c['delta_used']:.1e}")
    for k, v in checks.items():
        print(f"  [{'ok' if v.get('ok') else 'NO'}] {k}" + ("" if v.get("ok") else f": {v['reason']}"))

    if reasons:
        entry.update(decision="blocked", reasons=reasons)
        log(entry)
        if not any(not v.get("ok") for v in checks.values()):
            print(f"  [NO] {reasons[0]}")
        print("BLOCKED: nothing was sent.")
        return 1

    # All gates passed: exactly one send, no retry.
    try:
        if checks["d_size"]["mode"] == "blob":
            upload = ea.get_upload_url()
            ea.upload_blob(upload, json.dumps(solution, separators=(",", ":")).encode())
            resp = ea.submit_solution(live["id"], blob_key=upload["blobKey"])
        else:
            resp = ea.submit_solution(live["id"], solution=solution)
    except ea.ArenaError as exc:
        entry.update(decision="error_after_send" if exc.status not in (None, 401) else "error",
                     reasons=[f"server: HTTP {exc.status}: {exc}"],
                     retry_after_s=getattr(exc, "retry_after_s", None))
        log(entry)
        print(f"ERROR: {entry['reasons'][0]}")
        return 3
    entry.update(decision="sent", reasons=[], response={"id": resp.get("id"), "status": resp.get("status")})
    log(entry)
    print(f"SENT: solution id {resp.get('id')}, status {resp.get('status')}. Check in >= 20 min with:\n"
          f"  .venv/bin/python integrations/submit.py --status {resp.get('id')}")
    return 0


def status(solution_id: int) -> int:
    entries = [e for e in read_log() if e.get("kind") == "status_check" and e.get("solution_id") == solution_id]
    if entries:
        last = dt.datetime.fromisoformat(entries[-1]["timestamp"]).timestamp()
        wait = STATUS_MIN_INTERVAL_S - (now().timestamp() - last)
        if wait > 0:
            print(f"Last check of {solution_id} was less than {STATUS_MIN_INTERVAL_S // 60} min ago; "
                  f"try again in {wait / 60:.0f} min (no polling).")
            return 1
    entry = {"timestamp": now().isoformat(timespec="seconds"), "kind": "status_check", "solution_id": solution_id}
    try:
        res = ea.get_solution_status(solution_id)
        entry.update(result={k: res.get(k) for k in ("status", "score", "error", "evaluatedAt")})
    except ea.ArenaError as exc:
        # 404 = discarded (worse than personal best) or rejected (below minImprovement) and deleted by the server.
        entry.update(result={"http_status": exc.status, "error": str(exc)},
                     note="404 means discarded or rejected_min_improvement" if exc.status == 404 else None)
    log(entry)
    print(json.dumps(entry, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Gated single-shot EinsteinArena submission.")
    ap.add_argument("slug", nargs="?")
    ap.add_argument("solution", nargs="?", type=Path)
    ap.add_argument("--expected-score", type=float)
    ap.add_argument("--confirm", action="store_true", help="really send if every gate passes")
    ap.add_argument("--status", type=int, metavar="SOLUTION_ID", help="read the verdict of a sent solution (one GET)")
    args = ap.parse_args()
    if args.status is not None:
        return status(args.status)
    if not (args.slug and args.solution and args.expected_score is not None):
        ap.error("SLUG, SOLUTION.json and --expected-score are required")
    return submit(args.slug, args.solution, args.expected_score, args.confirm)


if __name__ == "__main__":
    sys.exit(main())
