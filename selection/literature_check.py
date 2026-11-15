"""WS1: literature cross-check of each live #1 record, using Exa.

For each problem with decision keep_stage3 in registry_stage2.jsonl:
  1. run 2-3 Exa queries (integrations.exa_client.search, logged to selection/exa_log.jsonl);
  2. extract from the highlights every number in the same range as the EinsteinArena record (candidates to review);
  3. compare the record with the curated literature values in LITERATURE below.

Rule: a literature value is only used if it is listed in LITERATURE with the URL where it was read (by a human, or by
an agent that quotes the source), the date, and whether its definition matches the EinsteinArena verifier. Numbers
that Exa returns are only *candidates*: they never set the status on their own.

Status:
  - "EA is best known": a matching-definition literature value exists, the EA record is at least as good, and no
    unreviewed Exa candidate beats the EA record.
  - "literature better": a matching-definition literature value beats the EA record.
  - "unclear": anything else (no curated value, definition mismatch only, or unreviewed better candidates).

Writes selection/literature_check.jsonl (one line per problem) and adds the cited URLs to evidence_urls in
registry_stage2.jsonl.

Usage: .venv/bin/python selection/literature_check.py [slug ...]
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from integrations.exa_client import ExaUnavailable, search  # noqa: E402

STAGE2 = ROOT / "registry_stage2.jsonl"
OUT = ROOT / "literature_check.jsonl"

# Human-readable names and Exa queries (2-3 per problem).
QUERIES = {
    "ring-loading-15": ("ring loading problem discrepancy constant", [
        ("ring loading problem Schrijver Seymour Winkler improved lower bound", None),
        ("ring loading discrepancy constant new lower bound 2026", "2025-06-01"),
        ("ring loading problem 9/8 lower bound", None)]),
    "third-autocorrelation-inequality": ("third autocorrelation inequality", [
        ("third autocorrelation inequality improved upper bound max f*f", None),
        ("autocorrelation inequality |max f*f| constant new record 2026", "2025-06-01"),
        ("third autocorrelation constant 1.4557 AlphaEvolve improved", "2025-11-01")]),
    "second-autocorrelation-inequality": ("second autocorrelation inequality", [
        ("second autocorrelation inequality improved lower bound L2 norm", None),
        ("second autocorrelation inequality new record 2026", "2025-06-01")]),
    "first-autocorrelation-inequality": ("first autocorrelation inequality", [
        ("first autocorrelation inequality improved upper bound Matolcsi Vinuesa", None),
        ("first autocorrelation inequality new record 2026", "2025-06-01")]),
    "erdos-min-overlap": ("Erdos minimum overlap problem", [
        ("Erdos minimum overlap problem improved upper bound", None),
        ("Erdos minimum overlap new record 2026", "2025-06-01")]),
    "circles-rectangle": ("21 circles in a rectangle of perimeter 4 maximize sum of radii", [
        ("circle packing rectangle perimeter 4 sum of radii 21 circles improved", None),
        ("AlphaEvolve circles in rectangle sum of radii new record 2026", "2025-06-01")]),
    "flat-polynomials": ("flat Littlewood polynomials degree 69", [
        ("flat Littlewood polynomial sup norm minimal degree 70 coefficients record", None),
        ("Littlewood polynomials flatness new record 2026", "2025-06-01")]),
    "kakeya-needle-128": ("Kakeya needle problem 128 triangles", [
        ("Kakeya needle set minimal area triangles construction improved", None),
        ("Kakeya needle 128 AlphaEvolve new record 2026", "2025-06-01")]),
    "circle-packing": ("26 circles in unit square maximize sum of radii", [
        ("26 circles unit square maximum sum of radii improved 2.635", None),
        ("circle packing sum of radii new record 2026", "2025-06-01")]),
    "tammes-problem": ("Tammes problem 50 points", [
        ("Tammes problem 50 points best known minimum distance", None),
        ("Tammes problem N=50 improved packing 2026", "2025-06-01")]),
    "thomson-problem": ("Thomson problem 282 electrons", [
        ("Thomson problem 282 electrons minimum energy", None),
        ("Thomson problem N=282 lower energy configuration 2026", "2025-06-01")]),
    "min-distance-ratio-2d": ("16 points minimum ratio of max to min distance", [
        ("16 points plane minimize ratio maximum minimum distance", None),
        ("min distance ratio 16 points AlphaEvolve improved 2026", "2025-06-01")]),
    "heilbronn-triangles": ("Heilbronn problem 11 points in a triangle", [
        ("Heilbronn triangle problem 11 points unit triangle best known", None),
        ("Heilbronn 11 points triangle new record 2026", "2025-06-01")]),
    "difference-bases": ("difference bases", [
        ("difference bases ratio k^2/n improved upper bound", None),
        ("difference basis construction new record 2026", "2025-06-01")]),
}

# Curated literature values: only values read in the cited source. Fields:
#   value (float or "p/q"), kind ("lower_bound"/"upper_bound"/"construction"), url, date (publication or access),
#   definition_matches_ea (bool), comparable (bool: same direction as the EA record, i.e. a construction/achieved
#   bound and not the opposite-side bound), verified_by_us (bool), read_in (where the quoted text was read),
#   reviewed_urls (Exa hits checked by hand for this problem), note.
ACCESSED = "2026-11-14"
LITERATURE: dict[str, list[dict]] = {
    "ring-loading-15": [
        {"value": "47/42", "kind": "lower_bound (construction, m=15)", "date": "2025-11-03",
         "url": "https://aitonomy.org/projects/open-problems/fs-0001", "accessed": ACCESSED,
         "definition_matches_ea": True, "comparable": True, "verified_by_us": False,
         "read_in": "Exa highlight: 'current record 47/42 = 1.1190476190476190 (best known lower bound); the "
                    "constant is pinned to the interval [47/42, 13/10]'; source cited there: Georgiev, Gomez-Serrano, "
                    "Tao, Wagner, arXiv:2511.02864",
         "note": "Published AlphaEvolve construction. EinsteinArena recomputes it exactly as 1.1190475684692773. "
                 "Definition: Skutella's beta (arXiv:1405.0789 p.3, read in the PDF); k-range differences are "
                 "absorbed by padding with a (0,0) pair (checked locally: EA #1 gives 9/8 both ways)."},
        {"value": "9/8", "kind": "construction (m=15)", "date": "2026-09-12",
         "url": "https://discoveria.ai/entry/ea-ring-loading-15", "accessed": ACCESSED,
         "definition_matches_ea": True, "comparable": False, "verified_by_us": True,
         "read_in": "Exa highlight: 'an agent submitted a score of 1.125 on September 12, 2026, against 1.1190475685 "
                    "for AlphaEvolve ... DiscoverIA replayed it on Oct 12, 2026: reproduced ... no third-party "
                    "evaluation found'",
         "note": "Replay of the EA #1 itself, not independent literature (comparable=False). Re-verified here in exact rationals."},
        {"value": "13/10", "kind": "upper_bound (all m)", "date": "2019-04-03",
         "url": "https://arxiv.org/abs/1904.02119", "accessed": ACCESSED,
         "definition_matches_ea": True, "comparable": False, "verified_by_us": False,
         "read_in": "Exa highlights (aitonomy fs-0001 and github.com/karldaeubel/RingLoading): 'we showed that "
                    "L <= L* + 13/10 D'; '13/10 is still the best published upper bound'",
         "note": "Daeubel. Caps any construction at 1.3. Nothing found on optimality for exactly 15 pairs."},
    ],
    "third-autocorrelation-inequality": [
        {"value": 1.44889, "kind": "upper_bound (construction, claimed)", "date": "2026-09-29",
         "url": "https://arxiv.org/abs/2609.38349", "accessed": ACCESSED,
         "definition_matches_ea": True, "comparable": True, "verified_by_us": False,
         "read_in": "arXiv abstract (MILO, Jana et al.): 'On EinsteinArena open problems, MILO improves best-known "
                    "upper bounds ... first and third autocorrelation inequalities (1.50274365 -> 1.50274360; "
                    "1.45081 -> 1.44889)'. Also cited as 'about 1.4489 for C'3' in "
                    "https://hexagonmath.org/pdf/2611.00395v1 (ref. [11], read in the PDF).",
         "note": "Claimed with the EA verifier (|max f*f|, i.e. C'). NOT on the live EA leaderboard (checked "
                 "2026-11-14, 27 entries, best 1.449109789); construction not available to us, so unverified."},
        {"value": 1.4557, "kind": "upper_bound (construction)", "date": "2025-11-03",
         "url": "https://arxiv.org/abs/2511.02864", "accessed": ACCESSED,
         "definition_matches_ea": True, "comparable": True, "verified_by_us": False,
         "read_in": "Exa highlight: 'AlphaEvolve improved these to C <= 1.4688 and C' <= 1.4557'",
         "note": "AlphaEvolve, C' = |max f*f| / (int f)^2, the EA definition."},
        {"value": 1.4630715826075594, "kind": "upper_bound (Tao table, verified certificate)", "date": ACCESSED,
         "url": "https://teorth.github.io/optimizationproblems/constants/1d.html", "accessed": ACCESSED,
         "definition_matches_ea": False, "comparable": False, "verified_by_us": False,
         "read_in": "Exa highlight: 'This is the maximum of |f*f| over t, for signed f. It is not the quantity "
                    "|max_t f*f|, whose records are smaller and which is a different constant'",
         "note": "Tao 1d = C (max |f*f|). Unverified entries on the page go down to 1.457854884921989*. "
                 "EA #1 recomputed under this definition gives 4.6127, so EA's 1.4491 is not a 1d record."},
    ],
}
REVIEWED = {
    "ring-loading-15": ["https://aitonomy.org/projects/open-problems/fs-0001",
                        "https://github.com/karldaeubel/RingLoading", "https://discoveria.ai/entry/ea-ring-loading-15"],
    "third-autocorrelation-inequality": ["https://github.com/teorth/optimizationproblems",
                                         "https://hexagonmath.org/pdf/2611.00395v1", "https://aiwiki.ai/wiki/alphaevolve",
                                         "https://arxiv.org/html/2611.08483v1",
                                         "https://github.com/togethercomputer/EinsteinArena-new-SOTA",
                                         "https://teorth.github.io/optimizationproblems/constants/1d.html",
                                         "https://docs.hillclimb.sh/examples/autocorr-3",
                                         # C1 (first autocorrelation) lower bounds, not this constant:
                                         "https://www.mefyi.com/papers/first-autocorrelation-constant/"],
}

NUM = re.compile(r"(?<![\w.])(\d+\.\d{3,}|\d+/\d+)(?![\w.])")


def parse_num(s: str) -> float | None:
    try:
        return float(Fraction(s)) if "/" in s else float(s)
    except (ValueError, ZeroDivisionError):
        return None


def better(a: float, b: float, direction: str) -> bool:
    return a < b if direction == "minimize" else a > b


def candidates(hits: list[dict], record: float) -> list[dict]:
    out = []
    for h in hits:
        for hl in h["highlights"]:
            for m in NUM.finditer(hl):
                v = parse_num(m.group(1))
                if v is not None and record and 0.8 <= v / record <= 1.25:
                    a, b = max(0, m.start() - 120), min(len(hl), m.end() + 120)
                    out.append({"value": v, "text": m.group(1), "url": h["url"],
                                "published_date": h["published_date"], "context": hl[a:b].replace("\n", " ")})
    return out


def check(slug: str, row: dict) -> dict:
    name, queries = QUERIES[slug]
    record, direction = row["record"]["value"], row["direction"]
    hits, errors = {}, []
    for q, since in queries:
        try:
            for h in search(q, since=since, n=8, decision=f"stage2 literature check: {slug}"):
                hits.setdefault(h["url"], h)
        except ExaUnavailable as exc:
            errors.append(str(exc))
    cands = candidates(list(hits.values()), record)
    reviewed = {l["url"] for l in LITERATURE.get(slug, [])} | set(REVIEWED.get(slug, []))
    better_cands = [c for c in cands if better(c["value"], record, direction) and c["url"] not in reviewed]

    lits = LITERATURE.get(slug, [])
    matching = [l for l in lits if l["definition_matches_ea"] and l["comparable"]]
    best_lit = None
    if matching:
        best_lit = min(matching, key=lambda l: parse_num(str(l["value"])) * (1 if direction == "minimize" else -1))
    if best_lit and better(parse_num(str(best_lit["value"])), record, direction):
        status = "literature better"
    elif best_lit and not better_cands:
        status = "EA is best known"
    else:
        status = "unclear"

    reasons = []
    if not lits:
        reasons.append("no curated literature value yet")
    elif not matching:
        reasons.append("curated values use a different definition")
    if better_cands:
        reasons.append(f"{len(better_cands)} unreviewed Exa number(s) beat the EA record")
    if errors:
        reasons.append("Exa errors: " + "; ".join(sorted(set(errors))))
    return {
        "slug": slug,
        "problem_name": name,
        "checked_on": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "direction": direction,
        "ea_record": {"value": record, "agent": row["record"]["agent"],
                      "date": row["record"]["solution_created_at"],
                      "url": f"https://einsteinarena.com/problems/{slug}",
                      "label": row["record"]["label"]},
        "best_literature": best_lit,
        "other_literature": [l for l in lits if l is not best_lit],
        "status": status,
        "status_reasons": reasons,
        "exa_queries": [q for q, _ in queries],
        "exa_urls": list(hits),
        "exa_hits": list(hits.values()),
        "exa_candidates_better_than_ea": better_cands,
        "exa_candidates_count": len(cands),
        "evidence_urls": sorted({l["url"] for l in lits}),
    }


def main() -> None:
    rows = {json.loads(l)["slug"]: json.loads(l) for l in STAGE2.read_text().splitlines() if l.strip()}
    slugs = sys.argv[1:] or [s for s, r in rows.items() if r["decision"] == "keep_stage3"]
    existing = {}
    if OUT.exists():
        existing = {json.loads(l)["slug"]: l for l in OUT.read_text().splitlines() if l.strip()}
    for slug in slugs:
        res = check(slug, rows[slug])
        existing[slug] = json.dumps(res, ensure_ascii=False)
        # Add cited literature URLs to stage 2 evidence.
        ev = rows[slug].setdefault("evidence_urls", [])
        ev.extend(u for u in res["evidence_urls"] if u not in ev)
        print(f"[lit] {slug}: {res['status']} ({'; '.join(res['status_reasons']) or 'ok'})", flush=True)
    OUT.write_text("\n".join(existing.values()) + "\n")
    STAGE2.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows.values()) + "\n")


if __name__ == "__main__":
    main()
