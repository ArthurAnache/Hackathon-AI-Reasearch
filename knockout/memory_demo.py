"""Build a static, evidence-linked report from Knockout memory run artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from knockout.memory import RunMemory
from knockout.memory_backfill import read_jsonl


MEMORY_HEADER = "# Memory from earlier runs and other agents"


def _payload(event: dict) -> dict:
    """Ledger events have flat fields; accept payload-wrapped fixture/older events too."""
    return {**event, **(event.get("payload") or {})}


def _cache_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _cache_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _cache_strings(child)


def _prompt_section(cache_dir: Path) -> str | None:
    """Extract the exact saved memory heading section when the cache retained request text."""
    for path in sorted(cache_dir.rglob("*.json")) if cache_dir.exists() else []:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for value in _cache_strings(obj):
            start = value.find(MEMORY_HEADER)
            if start < 0:
                continue
            end = re.search(r"\n# (?!Memory from earlier runs and other agents)", value[start + 1:])
            if end:
                stop = start + 1 + end.start()
                return value[start:stop].rstrip()
            return value[start:].rstrip()
    return None


def _experiment_index(run_dir: Path) -> dict[str, dict]:
    return {str(row.get("experiment_id")): row for row in read_jsonl(run_dir / "experiments.jsonl")
            if row.get("experiment_id")}


def _write_headline(kind: str, event: dict, refs: list, experiments: dict, summary: dict, run_id: str) -> str:
    if event.get("headline"):
        return str(event["headline"])
    linked = [experiments.get(str(r)) for r in refs]
    linked = [e for e in linked if e]
    if kind == "win" and linked:
        exp = linked[0]
        cfg = exp.get("config") or {}
        return f"{cfg.get('operator') or 'edit'}: {exp.get('hypothesis') or exp.get('experiment_id')}"
    if kind == "failed_batch" and linked:
        cfg = linked[0].get("config") or {}
        if event.get("arm") == "control":
            return f"Control round {cfg.get('round', '?')}: {len(refs)} edits did not beat the incumbent"
        return f"Batch {cfg.get('batch', '?')} ({cfg.get('island', '?')}): {len(refs)} edits did not beat their parent"
    if kind == "lesson":
        slug = summary.get("problem") or "unknown problem"
        return f"Run {run_id} on {slug}: best {summary.get('best')!r}, {len(experiments)} experiments"
    if kind == "refuted" and linked:
        return f"Insight refuted: {linked[0].get('hypothesis') or linked[0].get('experiment_id')}"
    return "Headline not retained in the run ledgers"


def _fact_from_write(event: dict, run_id: str, summary: dict, experiments: dict, log: dict) -> dict:
    data = _payload(event)
    refs = event.get("refs") or data.get("refs") or []
    kind = str(data.get("kind") or log.get("kind") or "unknown")
    headline = str(log.get("headline") or (log.get("metadata") or {}).get("headline") or
                   _write_headline(kind, data, refs, experiments, summary, run_id))
    linked = [experiments.get(str(ref)) for ref in refs]
    linked = [row for row in linked if row]
    gain = None
    if kind == "win" and linked:
        cfg = linked[0].get("config") or {}
        try:
            gain = float(linked[0]["score"]) - float(cfg["ref_score"])
        except (KeyError, TypeError, ValueError):
            pass
    elif kind == "failed_batch" and linked:
        gains = []
        for row in linked:
            try:
                gains.append(float(row["score"]) - float((row.get("config") or {})["ref_score"]))
            except (KeyError, TypeError, ValueError):
                continue
        gain = max(gains, default=None)
    arm = data.get("arm") or "?"
    timestamp = event.get("timestamp") or data.get("timestamp") or log.get("timestamp")
    metadata = {**(log.get("metadata") or {}), "kind": kind, "run_id": run_id, "arm": arm,
                "author": f"{arm}@{run_id}", "headline": headline, "timestamp": timestamp}
    if gain is not None:
        metadata["gain"] = gain
    return {"fact_id": data.get("fact_id"), "kind": kind, "author": metadata["author"],
            "content": f"[scope {summary.get('problem', 'unknown')}] [{kind}] {headline}",
            "metadata": metadata}


def _render_hash_verified(event: dict, facts_by_id: dict[str, dict], max_chars: int) -> str | None:
    data = _payload(event)
    ids = data.get("fact_ids") or event.get("refs") or []
    facts = [facts_by_id[str(fid)] for fid in ids if str(fid) in facts_by_id]
    renderer = object.__new__(RunMemory)
    renderer.max_chars = max_chars
    block = renderer._render(facts)
    expected = data.get("block_sha16") or event.get("block_sha16")
    if facts and block and expected and hashlib.sha256(block.encode()).hexdigest()[:16] == expected:
        return block
    return None


def build_report(run_dirs: list[str | Path]) -> str:
    runs: list[dict] = []
    writes: dict[str, dict] = {}
    facts_by_id: dict[str, dict] = {}
    store_counts: Counter = Counter()
    store_errors: Counter = Counter()
    memory_errors: list[tuple[str, dict]] = []

    for raw_dir in run_dirs:
        run_dir = Path(raw_dir)
        run_id = run_dir.name
        summary_path = run_dir / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
        experiments = _experiment_index(run_dir)
        try:
            strategy = yaml.safe_load((run_dir / "strategy.yaml").read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            strategy = {}
        memory_cfg = strategy.get("memory") or {}
        max_chars = memory_cfg.get("max_chars", strategy.get("memory.max_chars", 2500))
        events = read_jsonl(run_dir / "events.jsonl")
        log_rows = read_jsonl(run_dir / "pastdev_log.jsonl")
        log_by_id = {str(row.get("fact_id")): row for row in log_rows if row.get("fact_id")}
        run = {"id": run_id, "dir": run_dir, "events": [], "logs": log_rows,
               "summary": summary, "experiments": experiments}
        for row in log_rows:
            store = row.get("store") or "unknown"
            if row.get("op") in {"ingest", "recall"}:
                store_counts[store] += 1
            if row.get("status") == "error":
                store_errors[store] += 1
        for event in events:
            data = _payload(event)
            typ = event.get("type")
            if typ == "memory_error":
                memory_errors.append((run_id, data))
            if typ == "memory_write":
                fact_id = str(data.get("fact_id") or "")
                refs = event.get("refs") or data.get("refs") or []
                kind = str(data.get("kind") or "unknown")
                headline = _write_headline(kind, data, refs, experiments, summary, run_id)
                log = log_by_id.get(fact_id, {})
                fact = _fact_from_write(event, run_id, summary, experiments, log)
                if fact_id:
                    facts_by_id[fact_id] = fact
                writes[fact_id] = {
                    "run": run_id, "arm": data.get("arm") or "?", "kind": kind,
                    "headline": headline, "time": event.get("timestamp") or data.get("timestamp") or "?",
                    "store": data.get("store") or log.get("store") or "unknown",
                    "fact_id": fact_id, "fact": fact, "max_chars": max_chars,
                }
            elif typ == "memory_recall":
                run["events"].append((event, data))
            elif typ == "memory_error":
                store_errors[str(data.get("store") or "unknown")] += 1
        # Some older ledgers only recorded store names in the memory events.
        for event in events:
            if event.get("type") not in {"memory_write", "memory_recall"}:
                continue
            data = _payload(event)
            if not data.get("store"):
                continue
            operation = "ingest" if event.get("type") == "memory_write" else "recall"
            if not any(row.get("op") == operation and row.get("fact_id") == data.get("fact_id")
                       for row in log_rows):
                store_counts[str(data["store"])] += 1
        runs.append(run)

    recalls_by_fact: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        for event, data in run["events"]:
            ids = data.get("fact_ids") or event.get("refs") or []
            queries = data.get("queries") or ([data.get("query")] if data.get("query") else [])
            prompt = " ".join(" ".join(str(q).split()) for q in queries)
            for fact_id in ids:
                fact_id = str(fact_id)
                if fact_id not in writes:
                    continue
                written = writes[fact_id]
                if run["id"] == written["run"]:
                    continue
                recalls_by_fact[fact_id].append({
                    "run": run["id"], "arm": data.get("arm") or "?", "prompt": prompt,
                    "time": event.get("timestamp") or data.get("timestamp") or "?",
                })

    lines = ["# Knockout memory demo", "",
             "Evidence from the supplied run ledgers and local call logs. Fact headlines are reconstructed from "
             "memory-write references and experiment records where the store log omitted content.", "",
             "## Fact timeline", "",
             "| Written | Run | Arm | Kind | Headline | Store | Later recalled by |",
             "|---|---|---|---|---|---|---|"]
    if not writes:
        lines.append("| — | — | — | — | No memory_write events found | — | — |")
    for fact in sorted(writes.values(), key=lambda f: (f["time"], f["fact_id"])):
        later = recalls_by_fact.get(fact["fact_id"], [])
        if later:
            recalled = "<br>".join(f"{r['run']} / {r['arm']} ({r['time']}) via: "
                                   f"{r['prompt'][:180].replace('|', '&#124;')}" for r in later)
        else:
            recalled = "not recalled by another supplied run"
        lines.append(f"| {fact['time']} | `{fact['run']}` | {fact['arm']} | {fact['kind']} | "
                     f"{fact['headline'].replace('|', '&#124;')} | {fact['store']} | {recalled} |")
    lines.extend(["", "## Recall prompts", "",
                  "Queries below are taken verbatim from each `memory_recall` event; they are not the full editor prompt.", ""])
    any_recalls = False
    for run in runs:
        for event, data in run["events"]:
            any_recalls = True
            queries = data.get("queries") or ([data.get("query")] if data.get("query") else [])
            ids = data.get("fact_ids") or event.get("refs") or []
            lines.append(f"- `{run['id']}` / arm `{data.get('arm', '?')}` at "
                         f"{event.get('timestamp', '?')}: recalled {len(ids)} fact(s).")
            for query in queries:
                lines.append(f"  - Query: `{str(query).replace('`', '\\`')[:500]}`")
    if not any_recalls:
        lines.append("No `memory_recall` events were found.")

    lines.extend(["", "## Example rendered memory section", ""])
    section = None
    for run in runs:
        section = _prompt_section(run["dir"] / "llm_cache")
        if section:
            lines.extend([f"Exact section found in `{run['id']}/llm_cache`: ", "", "```text", section, "```"])
            break
    if not section:
        for run in runs:
            for event, _data in run["events"]:
                section = _render_hash_verified(event, facts_by_id, run.get("max_chars", 2500))
                if section is not None:
                    lines.extend([f"Hash-verified reconstruction for `{run['id']}` "
                                  "(matches the event's block_sha16):", "", "```text", section, "```"])
                    break
            if section is not None:
                break
    if not section:
        lines.extend([
            "Unavailable: neither cached prompt text nor a reconstruction from logged fact data matched a "
            "`block_sha16`. No prompt is fabricated here.",
        ])

    lines.extend(["", "## Store counts and errors", "", "| Store | Ingests / writes | Recalls | Errors |",
                  "|---|---:|---:|---:|"])
    stores = sorted(set(store_counts) | set(store_errors))
    for store in stores:
        # Store counts above include both API call-log operations and event-only records.
        ingest = sum(1 for run in runs for row in run["logs"]
                     if row.get("store") == store and row.get("op") == "ingest")
        recall = sum(1 for run in runs for row in run["logs"]
                     if row.get("store") == store and row.get("op") == "recall")
        if ingest + recall == 0:
            ingest = store_counts[store]
        errors = store_errors[store]
        lines.append(f"| {store} | {ingest} | {recall} | {errors} |")
    if not stores:
        lines.append("| no store activity | 0 | 0 | 0 |")
    if memory_errors:
        lines.extend(["", "Memory errors:"])
        lines.extend(f"- `{run}`: {e.get('error') or e.get('message') or e.get('op') or 'unspecified'}"
                     for run, e in memory_errors)
    else:
        lines.extend(["", "No `memory_error` events or failed store operations were logged."])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/memory_demo.md"))
    args = parser.parse_args(argv)
    report = build_report(args.run_dirs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
