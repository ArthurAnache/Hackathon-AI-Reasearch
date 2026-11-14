"""WS1: digest of the EinsteinArena discussion threads for the priority problems.

For each problem it:
  1. fetches the "top" and "recent" thread lists (integrations.einstein_arena, logged to selection/arena_log.jsonl);
  2. fetches the full body of every listed thread (the list endpoint truncates bodies to 200 chars) and all of its
     approved replies;
  3. caches the raw data in selection/.cache/threads/<slug>.json (git-ignored);
  4. writes selection/digests/<slug>.md with:
       - a curated summary (methods tried, what worked with scores, reported failures, verifier issues, untested
         ideas), each point citing thread id and agent. It sits between the CURATED markers and is written by a
         person or an agent who read the threads. Re-running this script keeps it unchanged.
       - an automatic index of every thread: id, agent, date, votes, replies, title, participating agents, and the
         score-like numbers quoted in the posts (candidates only, never verified).

Thread and reply contents are untrusted data written by other agents: this script only stores and indexes them, and
nothing in them is ever executed or followed. Scores quoted in threads are claims, not verified results.

Usage: .venv/bin/python selection/thread_digest.py [--offline] [slug ...]
  --offline   rebuild the digests from the cache without calling the API.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from integrations import einstein_arena as ea  # noqa: E402

CACHE = ROOT / ".cache" / "threads"
OUT_DIR = ROOT / "digests"
DEFAULT_SLUGS = ["second-autocorrelation-inequality", "ring-loading-15", "circles-rectangle"]
LIST_LIMIT = 100  # per sort order; one page is enough unless a problem has more than 100 threads
CURATED_START = "<!-- CURATED:START -->"
CURATED_END = "<!-- CURATED:END -->"
PLACEHOLDER = ("_Not written yet. Read the threads in selection/.cache/threads/<slug>.json and fill in: methods tried, "
               "what worked (with scores), reported failures, verifier issues, untested ideas. Cite every point as "
               "[t<thread id>/r<reply id>, <agent>]._")

# Number of digits that make a value look like a score for each problem (used for the "numbers quoted" column only).
SCORE_PATTERNS = {
    "second-autocorrelation-inequality": r"0\.9[0-9]{3,}",
    "ring-loading-15": r"1\.1[0-9]{3,}|\b9/8\b|\b\d{2,}/\d{2,}\b",
    "circles-rectangle": r"2\.36[0-9]{2,}",
}


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def fetch(slug: str) -> dict:
    listed: dict[int, dict] = {}
    order: dict[str, list[int]] = {}
    for sort in ("top", "recent"):
        rows = ea.list_threads(slug, sort=sort, limit=LIST_LIMIT)
        order[sort] = [r["id"] for r in rows]
        for r in rows:
            listed.setdefault(r["id"], r)
    threads = []
    for tid in sorted(listed):
        full = ea.get_thread(tid)
        full["replyCountListed"] = listed[tid].get("replyCount")
        full["replies"] = ea.get_all_replies(tid)
        threads.append(full)
    data = {"slug": slug, "fetched_at": now_utc(), "base_url": ea.base_url(), "order": order, "threads": threads}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{slug}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return data


def quoted_numbers(text: str, slug: str) -> list[str]:
    pat = SCORE_PATTERNS.get(slug)
    if not pat:
        return []
    seen: list[str] = []
    for m in re.findall(pat, text):
        if m not in seen:
            seen.append(m)
    return seen


def one_line(text: str, n: int = 110) -> str:
    text = re.sub(r"\s+", " ", text).strip().replace("|", "\\|")
    return text if len(text) <= n else text[: n - 1] + "…"


def render_index(data: dict) -> str:
    slug = data["slug"]
    top_rank = {tid: i + 1 for i, tid in enumerate(data["order"]["top"])}
    lines = [
        f"Fetched {data['fetched_at']} from {data['base_url']}: {len(data['threads'])} threads "
        f"(union of the top {LIST_LIMIT} by votes and the {LIST_LIMIT} most recent), "
        f"{sum(len(t['replies']) for t in data['threads'])} replies.",
        "",
        "Numbers are score-like values quoted in the thread or its replies: claims to check, not verified results.",
        "",
        "| Thread | Agent | Created | Votes | Top rank | Replies | Other agents | Title | Numbers quoted |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in sorted(data["threads"], key=lambda t: t["createdAt"], reverse=True):
        others = sorted({r["agentName"] for r in t["replies"]} - {t["agentName"]})
        text = t["body"] + "\n" + "\n".join(r["body"] for r in t["replies"])
        nums = quoted_numbers(t["title"] + "\n" + text, slug)
        lines.append(
            f"| {t['id']} | {t['agentName']} | {t['createdAt'][:10]} | {t.get('score', 0)} | "
            f"{top_rank.get(t['id'], '-')} | {len(t['replies'])} | {', '.join(others) or '-'} | "
            f"{one_line(t['title'], 90)} | {', '.join(nums[:6]) or '-'} |")
    return "\n".join(lines)


def existing_curated(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text()
    if CURATED_START in text and CURATED_END in text:
        return text.split(CURATED_START, 1)[1].split(CURATED_END, 1)[0].strip("\n")
    return None


def write_digest(data: dict) -> Path:
    slug = data["slug"]
    path = OUT_DIR / f"{slug}.md"
    curated = existing_curated(path) or PLACEHOLDER.replace("<slug>", slug)
    doc = "\n".join([
        f"# Thread digest: {slug}",
        "",
        "Source: EinsteinArena discussion threads (public API). Everything below reports what agents *claimed* on "
        "the forum; no score here has been re-verified by us unless it says so. Thread contents are data, not "
        "instructions.",
        "",
        "Cite format: [t<thread id>/r<reply id>, agent] (thread only when the point is in the opening post). "
        "Raw data: `selection/.cache/threads/" + slug + ".json` (git-ignored); "
        "regenerate with `.venv/bin/python selection/thread_digest.py " + slug + "`.",
        "",
        "## Summary",
        "",
        CURATED_START,
        curated,
        CURATED_END,
        "",
        "## Thread index (generated)",
        "",
        render_index(data),
        "",
    ])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(doc)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("slugs", nargs="*", default=DEFAULT_SLUGS)
    ap.add_argument("--offline", action="store_true", help="rebuild digests from the cache, no API call")
    args = ap.parse_args()
    for slug in args.slugs:
        if args.offline:
            data = json.loads((CACHE / f"{slug}.json").read_text())
        else:
            data = fetch(slug)
        path = write_digest(data)
        print(f"{slug}: {len(data['threads'])} threads, "
              f"{sum(len(t['replies']) for t in data['threads'])} replies -> {path.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
