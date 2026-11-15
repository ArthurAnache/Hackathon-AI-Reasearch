"""Problem scout, step 1: refresh the live facts of every problem the EinsteinArena API serves.

Usage (from the repo root):  .venv/bin/python selection/scout_refresh.py [--top 8]

For each live problem it reads the problem (delta = minImprovement, direction), the leaderboard (top 10) and the
`--top` best solutions (score, agent, creation date; the solution data is dropped unless it is the #1, which is
cached for the experiments). It writes:

- selection/.cache/scout/<slug>/{problem.json, leaderboard.json, best_meta.json, record_solution.json}  (git-ignored)
- selection/scout_live.jsonl  (one line per problem: #1, #2, delta, gap, dates of the top solutions, activity)

Read-only: public GET endpoints through integrations/einstein_arena.py (every call is logged to
selection/arena_log.jsonl). Nothing is submitted. Scores read here are *displayed* scores: a score is "verified"
only once the pinned verifier has reproduced it locally (see selection/scout_third_autocorr.py for the one we did).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from integrations import einstein_arena as ea  # noqa: E402

CACHE = ROOT / ".cache" / "scout"
OUT = ROOT / "scout_live.jsonl"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def better(a: float, b: float, scoring: str) -> float:
    """How much `a` beats `b` (positive = better)."""
    return b - a if scoring == "minimize" else a - b


def refresh(slug: str, top: int) -> dict:
    checked = now_utc()
    cache = CACHE / slug
    cache.mkdir(parents=True, exist_ok=True)
    problem = ea.get_problem(slug)
    (cache / "problem.json").write_text(json.dumps(problem, indent=1))
    pid, scoring, delta = problem["id"], problem["scoring"], problem.get("minImprovement")
    board = ea.get_leaderboard(pid, limit=10)
    (cache / "leaderboard.json").write_text(json.dumps(board, indent=1))
    best = ea.get_best_solutions(pid, limit=top)
    meta = []
    for i, sol in enumerate(best):
        data = sol.get("data")
        raw = json.dumps(data, separators=(",", ":")) if data is not None else ""
        meta.append({"id": sol.get("id"), "agent": sol.get("agentName"), "score": sol.get("score"),
                     "created_at": sol.get("createdAt"), "data_bytes": len(raw),
                     "data_sha256": hashlib.sha256(raw.encode()).hexdigest() if raw else None})
        if i == 0 and data is not None:
            (cache / "record_solution.json").write_text(raw)
    (cache / "best_meta.json").write_text(json.dumps(meta, indent=1))

    row: dict = {
        "slug": slug, "problem_id": f"einsteinarena/{slug}", "arena_id": pid, "title": problem.get("title"),
        "direction": scoring, "delta": delta, "evaluation_mode": problem.get("evaluationMode"),
        "checked_on": checked.isoformat(timespec="seconds"),
        "source": f"https://einsteinarena.com/problems/{slug}",
        "n_leaderboard": len(board),
        "label": "displayed (live API); not re-verified by this script",
    }
    if not board:
        row["note"] = "empty leaderboard"
        return row
    s1 = board[0]["bestScore"]
    row["top1"] = {"agent": board[0]["agentName"], "score": s1}
    if len(board) > 1:
        s2 = board[1]["bestScore"]
        gap = better(s1, s2, scoring)
        row["top2"] = {"agent": board[1]["agentName"], "score": s2}
        row["gap_1_2"] = gap
        row["gap_1_2_over_delta"] = (gap / delta) if delta else None
    tol = delta if delta else 0.0
    row["within_delta_of_1"] = sum(1 for e in board[1:] if better(s1, e["bestScore"], scoring) <= tol)
    row["top_solutions"] = [{"agent": m["agent"], "score": m["score"], "created_at": m["created_at"]} for m in meta]
    dated = [m for m in meta if m["created_at"]]
    if dated:
        row["top1_created_at"] = dated[0]["created_at"]
        row["days_since_top1"] = round((checked - parse(dated[0]["created_at"])).total_seconds() / 86400, 1)
        # Activity: how many of the top solutions were created in the last 30 / 60 days, and the improvement the
        # leaders made over the best solution that is older than 60 days (a proxy for "the record still moves").
        for days in (30, 60):
            row[f"top_created_last_{days}d"] = sum(
                1 for m in dated if (checked - parse(m["created_at"])).days < days)
        old = [m for m in dated if (checked - parse(m["created_at"])).days >= 60]
        if old:
            best_old = max(old, key=lambda m: better(m["score"], s1, scoring))
            row["gain_over_best_older_than_60d"] = better(s1, best_old["score"], scoring)
            row["gain_over_best_older_than_60d_over_delta"] = (
                row["gain_over_best_older_than_60d"] / delta if delta else None)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("slugs", nargs="*")
    args = ap.parse_args()
    live = ea.list_problems()
    slugs = args.slugs or sorted(p["slug"] for p in live)
    rows = []
    for slug in slugs:
        try:
            row = refresh(slug, args.top)
        except ea.ArenaError as exc:
            row = {"slug": slug, "problem_id": f"einsteinarena/{slug}", "error": str(exc)[:200],
                   "checked_on": now_utc().isoformat(timespec="seconds")}
        rows.append(row)
        t1, t2 = row.get("top1", {}), row.get("top2", {})
        print(f"{slug:36s} {row.get('direction', '?'):8s} d={row.get('delta')!s:7s} "
              f"#1={t1.get('score')!s:22s} {t1.get('agent', '')[:18]:18s} "
              f"gap/d={row.get('gap_1_2_over_delta')!s:10.10s} tie={row.get('within_delta_of_1')} "
              f"age={row.get('days_since_top1')}d new30={row.get('top_created_last_30d')} "
              f"new60={row.get('top_created_last_60d')} "
              f"gain60/d={row.get('gain_over_best_older_than_60d_over_delta')!s:.10s}", flush=True)
    with OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
