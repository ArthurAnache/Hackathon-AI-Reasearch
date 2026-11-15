"""Compare memory-on and memory-off Knockout runs using offline ledgers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from knockout.filters import jaccard, token_set
from knockout.memory_backfill import read_jsonl


EXCLUDED_OPERATORS = {"replay", "seed", "freeze-tune"}
REFUTED_KINDS = {"refuted", "failed_batch"}


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _fact_rows(path: Path | None) -> list[dict]:
    facts = read_jsonl(path) if path else []
    return facts


def _run_facts(run_dir: Path, known: dict[str, dict]) -> list[dict]:
    """Recover locally written facts from pastdev logs/events when their content is logged."""
    out: list[dict] = []
    log = run_dir / "pastdev_log.jsonl"
    for row in read_jsonl(log):
        candidates = row.get("facts") or row.get("results") or row.get("sources") or []
        if isinstance(candidates, dict):
            candidates = [candidates]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            fid = item.get("fact_id") or item.get("id")
            if fid in known:
                out.append(known[fid])
            elif item.get("content"):
                out.append(item)
    for event in read_jsonl(run_dir / "events.jsonl"):
        if event.get("type") != "memory_write":
            continue
        payload = event.get("payload") or {}
        fid = payload.get("fact_id")
        if fid in known:
            out.append(known[fid])
        elif payload.get("content"):
            out.append({**payload, "timestamp": event.get("timestamp")})
    # Deduplicate writes that were available from both ledgers.
    unique: dict[str, dict] = {}
    for fact in out:
        unique[str(fact.get("fact_id") or (fact.get("content"), fact.get("timestamp")))] = fact
    return list(unique.values())


def _identity_fields(fact: dict) -> tuple[str | None, str | None]:
    meta = fact.get("metadata") or {}
    identity = fact.get("identity") or fact.get("author")
    run = meta.get("run_id") or fact.get("run_id")
    agent = meta.get("agent") or fact.get("agent")
    return (str(run) if run is not None else None,
            str(agent or identity) if (agent or identity) is not None else None)


def analyze_run(run_dir: str | Path, facts: list[dict] | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    experiments = read_jsonl(run_dir / "experiments.jsonl")
    events = read_jsonl(run_dir / "events.jsonl")
    strategy_path = run_dir / "strategy.yaml"
    strategy = yaml.safe_load(strategy_path.read_text(encoding="utf-8")) if strategy_path.exists() else {}
    eps = _float(((strategy or {}).get("selection") or {}).get("eps_rel")) or 0.0
    run_id = next((x.get("run_id") for x in experiments if x.get("run_id")), run_dir.name)
    problem = next((x.get("problem") for x in experiments if x.get("problem")), None)
    if problem is None:
        problem = next(((e.get("payload") or {}).get("problem") for e in events
                        if (e.get("payload") or {}).get("problem")), None)

    fact_map = {str(f.get("fact_id")): f for f in (facts or []) if f.get("fact_id")}
    def same_problem(fact: dict) -> bool:
        fproblem = (fact.get("metadata") or {}).get("problem") or fact.get("scope")
        return problem is None or fproblem == problem

    relevant = [f for f in (facts or []) if f.get("kind") in REFUTED_KINDS and same_problem(f)]
    relevant += [f for f in _run_facts(run_dir, fact_map)
                 if f.get("kind") in REFUTED_KINDS and same_problem(f)]

    edit_proposals = [x for x in experiments
                      if (x.get("config") or {}).get("operator") not in (None, *EXCLUDED_OPERATORS)
                      and isinstance(x.get("hypothesis"), str) and x["hypothesis"].strip()]
    repeats = 0
    for exp in edit_proposals:
        ts = exp.get("timestamp") or ""
        proposal = token_set(exp["hypothesis"])
        repeated = False
        for fact in relevant:
            fts = fact.get("timestamp") or ""
            if not fts or not ts or fts >= ts:
                continue
            content = fact.get("content") or ""
            if jaccard(proposal, token_set(content)) >= 0.6:
                repeated = True
                break
        repeats += int(repeated)

    scored = []
    for exp in experiments:
        cfg = exp.get("config") or {}
        ref, score = _float(cfg.get("ref_score")), _float(exp.get("score"))
        if ref is not None and score is not None:
            scored.append((exp, cfg, score - ref, ref, score))
    wins = [x for x in scored if x[2] > eps * max(1.0, abs(x[3]))]
    cost = sum((_float(((e.get("payload") or {}).get("usd"))) or 0.0)
               for e in events if e.get("type") == "llm_call")
    cost += sum((_float(((x.get("config") or {}).get("cost_usd"))) or 0.0) for x in experiments)

    recall_events = [e for e in events if e.get("type") == "memory_recall"]
    error_events = [e for e in events if e.get("type") == "memory_error"]
    recalled_ids: list[str] = []
    external = 0
    for event in recall_events:
        payload = event.get("payload") or {}
        ids = payload.get("fact_ids") or []
        if isinstance(ids, list):
            recalled_ids.extend(str(i) for i in ids)
        authors = payload.get("authors") or event.get("authors") or []
        if isinstance(authors, list):
            for author in authors:
                if isinstance(author, dict):
                    author_run = author.get("run_id")
                    author_name = author.get("agent") or author.get("identity")
                else:
                    author_run, author_name = None, str(author)
                if (author_run and author_run != run_id) or (author_name and author_name not in {run_id, "backfill", ""}):
                    external += 1
    known_ids = set(fact_map)
    # Recall logs may include facts directly, so collect their authors as a fallback.
    recalled_facts = [fact_map[i] for i in recalled_ids if i in fact_map]
    for fact in recalled_facts:
        author_run, author_agent = _identity_fields(fact)
        if (author_run and author_run != run_id) or (author_agent and author_agent not in {run_id, "backfill", ""}):
            # Prefer authors attached to the recall event when supplied; otherwise use fact metadata.
            if not any(((e.get("payload") or {}).get("authors") or e.get("authors")) for e in recall_events):
                external += 1
    max_gain = max((x[2] for x in scored), default=None)
    return {
        "run_id": run_id,
        "repeat_rate": (repeats / len(edit_proposals)) if edit_proposals else None,
        "repeat_count": repeats,
        "edit_count": len(edit_proposals),
        "win_fraction": (len(wins) / len(scored)) if scored else None,
        "wins": len(wins),
        "scored": len(scored),
        "usd_per_win": (cost / len(wins)) if wins else None,
        "total_cost": cost,
        "memory_recalls": len(recall_events),
        "facts_recalled": len(recalled_ids),
        "known_facts_recalled": len([i for i in recalled_ids if i in known_ids]),
        "external_facts_recalled": external,
        "memory_errors": len(error_events),
        "has_memory_events": bool(recall_events or error_events or any(e.get("type") == "memory_write" for e in events)),
        "max_gain": max_gain,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def render_report(on: dict, off: dict) -> str:
    rows = [
        ("Repeat rate", _pct(on["repeat_rate"]), _pct(off["repeat_rate"])),
        ("Children beating ref", f"{_pct(on['win_fraction'])} ({on['wins']}/{on['scored']})",
         f"{_pct(off['win_fraction'])} ({off['wins']}/{off['scored']})"),
        ("USD per winning child", _money(on["usd_per_win"]), _money(off["usd_per_win"])),
        ("Memory events", _memory_summary(on), _memory_summary(off)),
        ("Best gain over warm start (diagnostic only; not a record claim)",
         _signed(on["max_gain"]), _signed(off["max_gain"])),
    ]
    return ("| Metric | Memory on | Memory off |\n|---|---:|---:|\n" +
            "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows) + "\n")


def _memory_summary(result: dict) -> str:
    if not result["has_memory_events"]:
        return "no memory events"
    return (f"{result['memory_recalls']} recalls; {result['facts_recalled']} facts "
            f"({result['external_facts_recalled']} from another run/agent); "
            f"{result['memory_errors']} errors")


def _signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.8g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on", required=True, type=Path, help="memory-enabled run directory")
    parser.add_argument("--off", required=True, type=Path, help="memory-disabled run directory")
    parser.add_argument("--facts", type=Path, help="backfilled / known facts JSONL")
    parser.add_argument("--out", type=Path, default=Path("runs/memory_metrics.md"))
    args = parser.parse_args(argv)
    facts = _fact_rows(args.facts)
    on, off = analyze_run(args.on, facts), analyze_run(args.off, facts)
    report = f"# Memory run comparison\n\nOn: `{on['run_id']}`; off: `{off['run_id']}`.\n\n" + render_report(on, off)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
