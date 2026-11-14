"""Sponsor metrics computed from the logs (never from estimates).

    python -m integrations.sponsor_metrics [--store experiments.jsonl] [--json]

Exa       calls, cached calls, blocked calls, USD spent (total and per key label),
          experiments whose evidence cited Exa, and how many of those were kept.
past.dev  writes and reads per agent, experiments informed by a fact written by
          ANOTHER agent (the sponsor requirement), and how many of those were kept.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import exa_client, pastdev_client
from ._common import REPO_ROOT


def _jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def compute(store_path: Path) -> dict:
    exa_log = _jsonl(exa_client.LOG_PATH)
    past_log = _jsonl(pastdev_client.LOG_PATH)
    exps = _jsonl(store_path)

    paid = [e for e in exa_log if e.get("ok") and not e.get("cached")]
    by_key = Counter()
    for e in paid:
        by_key[e.get("key_label") or "unknown"] += e.get("cost_dollars") or 0.0

    def cites(r, source):
        return [e for e in (r.get("evidence") or []) if e.get("source") == source]

    exa_exps = [r for r in exps if cites(r, "exa")]
    cross = []
    for r in exps:
        mine = (r.get("agent") or "").split("@")[-1]      # "feedback@laptop-a" -> "laptop-a"
        others = [e for e in cites(r, "past.dev") if e.get("author") and e.get("author") != mine
                  and e.get("author") != "exa-scout"]
        if others:
            cross.append((r, others))

    return {
        "exa": {
            "calls_paid": len(paid),
            "calls_cached": sum(1 for e in exa_log if e.get("cached")),
            "calls_blocked_by_stop_loss": sum(1 for e in exa_log if e.get("blocked")),
            "calls_failed": sum(1 for e in exa_log if e.get("ok") is False and not e.get("blocked")),
            "usd_spent": round(sum(by_key.values()), 4),
            "usd_by_key": {k: round(v, 4) for k, v in by_key.items()},
            "experiments_citing_exa": len(exa_exps),
            "of_which_kept": sum(r.get("status") == "keep" for r in exa_exps),
        },
        "pastdev": {
            "writes_by_agent": dict(Counter(e.get("author") for e in past_log
                                            if e.get("tool") in ("past.ingest", "local.write"))),
            "reads_by_agent": dict(Counter(e.get("reader") for e in past_log
                                           if e.get("tool") in ("past.recall", "local.read"))),
            "experiments_using_another_agents_fact": len(cross),
            "of_which_kept": sum(r.get("status") == "keep" for r, _ in cross),
            "examples": [{"experiment_id": r["experiment_id"], "agent": r.get("agent"),
                          "used_fact": o[0].get("citation"), "fact_author": o[0].get("author"),
                          "status": r.get("status")} for r, o in cross[:5]],
        },
        "experiments_total": len(exps),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default=str(REPO_ROOT / "experiments.jsonl"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    m = compute(Path(a.store))
    if a.json:
        print(json.dumps(m, indent=2))
        return 0
    x, p = m["exa"], m["pastdev"]
    print("Exa:      %d paid calls (%d cached, %d blocked, %d failed), $%.4f %s"
          % (x["calls_paid"], x["calls_cached"], x["calls_blocked_by_stop_loss"], x["calls_failed"],
             x["usd_spent"], x["usd_by_key"]))
    print("          %d experiments cited Exa, %d kept" % (x["experiments_citing_exa"], x["of_which_kept"]))
    print("past.dev: writes %s, reads %s" % (p["writes_by_agent"], p["reads_by_agent"]))
    print("          %d experiments used a fact written by another agent, %d kept"
          % (p["experiments_using_another_agents_fact"], p["of_which_kept"]))
    for ex in p["examples"]:
        print("          - %(experiment_id)s by %(agent)s used %(used_fact)s from %(fact_author)s -> %(status)s" % ex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
