"""Turn committed Knockout v1 ledgers into deterministic past.dev memory facts.

The default mode is offline: it reads result folders and writes a JSONL facts file.
No network client is imported unless ``--push`` is explicitly supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "knockout" / "results"
DEFAULT_OUT = ROOT / "runs" / "memory_backfill" / "facts.jsonl"
FACT_KEYS = ("fact_id", "kind", "content", "timestamp", "identity", "scope", "metadata")
METADATA_KEYS = ("problem", "kind", "run_id", "arm", "island", "operator", "model",
                 "experiment_id", "gain", "ref_score", "score", "source_commit", "agent")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _gain(exp: dict) -> tuple[float, float, float] | None:
    cfg = exp.get("config") or {}
    score, ref = _number(exp.get("score")), _number(cfg.get("ref_score"))
    if score is None or ref is None:
        return None
    return score - ref, ref, score


def _trunc(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= limit:
        return s
    return s[:max(0, limit - 1)].rstrip() + "…"


def _timestamp(*rows: dict | None) -> str | None:
    for row in rows:
        if not row:
            continue
        value = row.get("timestamp") or row.get("created_at")
        if isinstance(value, str) and value:
            return value
    return None


def _experiment_meta(exp: dict, problem: str, run_id: str, eps_rel: float) -> dict:
    cfg = exp.get("config") or {}
    vals = _gain(exp)
    gain, ref, score = vals if vals is not None else (None, None, None)
    op_wins = gain is not None and gain > eps_rel * max(1.0, abs(ref))
    return {
        "experiment": exp,
        "problem": exp.get("problem") or problem,
        "run_id": exp.get("run_id") or run_id,
        "arm": cfg.get("arm") or cfg.get("chain"),
        "island": cfg.get("island"),
        "operator": cfg.get("operator"),
        "model": cfg.get("model"),
        "experiment_id": exp.get("experiment_id"),
        "gain": gain,
        "ref_score": ref,
        "score": score,
        "win": op_wins,
    }


def _fact(run_id: str, problem: str, kind: str, source_id: str, headline: str, details: str,
          timestamp: str | None, meta: dict[str, Any]) -> dict:
    key = f"{run_id}|{kind}|{source_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    content = _trunc(f"[scope {problem}] {_trunc(headline, 180)}\n{details}", 600)
    metadata = {k: meta.get(k) for k in METADATA_KEYS}
    metadata.update(problem=problem, kind=kind, run_id=run_id, agent="backfill")
    return {
        "fact_id": f"{kind}-{digest}",
        "kind": kind,
        "content": content,
        "timestamp": timestamp,
        "identity": f"backfill@{run_id}",
        "scope": problem,
        "metadata": metadata,
    }


def _load_run(run_dir: Path) -> tuple[str, str, float, list[dict], list[dict], str | None]:
    experiments = read_jsonl(run_dir / "experiments.jsonl")
    events = read_jsonl(run_dir / "events.jsonl")
    run_id = next((e.get("run_id") for e in experiments if e.get("run_id")), None)
    run_id = run_id or next((e.get("run_id") for e in events if e.get("run_id")), None) or run_dir.name
    problem = next((e.get("problem") for e in experiments if e.get("problem")), None)
    if not problem:
        for event in events:
            payload = event.get("payload") or {}
            problem = event.get("problem") or payload.get("problem")
            if problem:
                break
    if not problem:
        raise ValueError(f"{run_dir}: no problem slug in experiments.jsonl or events.jsonl")
    strategy_path = run_dir / "strategy.yaml"
    strategy = yaml.safe_load(strategy_path.read_text(encoding="utf-8")) if strategy_path.exists() else {}
    eps_rel = _number(((strategy or {}).get("selection") or {}).get("eps_rel"))
    eps_rel = 0.0 if eps_rel is None else eps_rel
    source_commit = next((e.get("source_commit") for e in experiments if e.get("source_commit")), None)
    return run_id, problem, eps_rel, experiments, events, source_commit


def facts_for_run(run_dir: str | Path) -> list[dict]:
    run_dir = Path(run_dir)
    run_id, problem, eps_rel, experiments, events, source_commit = _load_run(run_dir)
    facts: list[dict] = []
    entries = [_experiment_meta(e, problem, run_id, eps_rel) for e in experiments]
    comparable = [x for x in entries if x["gain"] is not None]
    wins = [x for x in comparable if x["win"]]
    losses = [x for x in comparable if not x["win"]]

    base_meta = {"problem": problem, "run_id": run_id, "source_commit": source_commit, "agent": "backfill"}

    # Individual wins carry the original experiment timestamp and the measured delta.
    for item in wins:
        exp, gain, ref, score = item["experiment"], item["gain"], item["ref_score"], item["score"]
        operator = item["operator"]
        headline = f"[win] {operator or 'operator unspecified'}: {_trunc(exp.get('hypothesis'), 180)}"
        detail = f"gain {gain:+.6g} over ref_score {ref:.12g}; score {score:.12g}; experiment {item['experiment_id']}"
        meta = {**base_meta, **{k: item[k] for k in METADATA_KEYS if k in item}}
        meta["source_commit"] = exp.get("source_commit") or source_commit
        facts.append(_fact(run_id, problem, "win", str(item["experiment_id"]), headline, detail,
                           _timestamp(exp), meta))

    # Map gene birth records to the later gate verdicts.
    born: dict[str, dict] = {}
    for event in events:
        if event.get("type") == "gene_born":
            payload = event.get("payload") or {}
            gid = payload.get("gene")
            if gid:
                born[str(gid)] = event
    refuted_facts: list[dict] = []
    seen_refuted: set[str] = set()

    def add_refuted(source_id: str, headline: str, details: str, timestamp: str | None,
                    meta: dict[str, Any]) -> None:
        if source_id in seen_refuted:
            return
        seen_refuted.add(source_id)
        f = _fact(run_id, problem, "refuted", source_id, headline, details, timestamp,
                  {**base_meta, **meta})
        facts.append(f)
        refuted_facts.append(f)

    for event in events:
        if event.get("type") != "gate_verdict":
            continue
        payload = event.get("payload") or {}
        outputs = payload.get("outputs") or {}
        if outputs.get("verdict") != "refuted":
            continue
        gid = str(payload.get("gene") or next(iter(event.get("refs") or []), "unknown-gene"))
        birth = born.get(gid)
        bp = (birth or {}).get("payload") or {}
        inputs = payload.get("inputs") or {}
        text = bp.get("text") or payload.get("text") or ""
        wins_n, losses_n = inputs.get("wins"), inputs.get("losses")
        headline = f"[refuted] {gid}: {_trunc(text, 150)}"
        details = f"gate evidence {wins_n} wins / {losses_n} losses ({inputs.get('pairs')} pairs)"
        add_refuted(f"gene:{gid}", headline, details, _timestamp(event),
                    {"arm": bp.get("arm"), "island": payload.get("island") or bp.get("island"),
                     "experiment_id": gid, "ref_score": None, "score": None, "gain": None})

    # Numerical artifacts are permanently unsafe candidates, regardless of the optimizer's self-report.
    for item in entries:
        exp = item["experiment"]
        if exp.get("decision") != "numerical_artifact" and exp.get("status") != "numerical_artifact":
            continue
        eid = str(item["experiment_id"] or "unknown-experiment")
        add_refuted(f"artifact:{eid}", f"[refuted] numerical artifact from {eid}",
                    f"hypothesis: {_trunc(exp.get('hypothesis'), 180)}; score {item['score']}; ref_score {item['ref_score']}",
                    _timestamp(exp), {**{k: item[k] for k in METADATA_KEYS if k in item},
                                    "source_commit": exp.get("source_commit") or source_commit})

    # Aggregate per-operator evidence. A cold operator with at least ten children and no wins is refuted.
    by_operator: dict[str, list[dict]] = defaultdict(list)
    for item in comparable:
        if item["operator"]:
            by_operator[str(item["operator"])].append(item)
    op_counts: dict[str, tuple[int, int]] = {}
    for operator, group in sorted(by_operator.items()):
        n_win = sum(1 for x in group if x["win"])
        op_counts[operator] = (n_win, len(group) - n_win)
        if len(group) >= 10 and n_win == 0:
            add_refuted(f"operator:{operator}", f"[refuted] operator {operator}: 0 wins / {len(group)} children",
                        f"No child exceeded the configured relative improvement threshold eps_rel={eps_rel:g} in run {run_id}.",
                        _timestamp(group[-1]["experiment"]),
                        {"operator": operator, "experiment_id": f"operator:{operator}",
                         "gain": 0.0, "ref_score": None, "score": None})

    # One compact failed-batch fact per operator with losing/flat children.
    for operator, group in sorted(by_operator.items()):
        failed = [x for x in group if not x["win"]]
        if not failed:
            continue
        representative = failed[:8]
        lines = []
        for x in representative:
            exp = x["experiment"]
            lines.append(f"- {_trunc(exp.get('hypothesis') or x['experiment_id'], 95)} (gain {x['gain']:+.4g})")
        detail = "\n".join(lines)
        if len(failed) > len(representative):
            detail += f"\n- … and {len(failed) - len(representative)} more non-improving children"
        facts.append(_fact(run_id, problem, "failed_batch", f"operator:{operator}",
                           f"[failed batch] {operator}: {len(failed)} of {len(group)} children did not gain",
                           detail, _timestamp(failed[-1]["experiment"]),
                           {**base_meta, "operator": operator,
                            "experiment_id": f"failed_batch:{operator}",
                            "gain": min((x["gain"] for x in failed), default=None)}))

    # One templated lesson per run. No model calls or invented recommendations.
    best = max(comparable, key=lambda x: x["gain"], default=None)
    op_summary = ", ".join(f"{op} {w}/{w+l}" for op, (w, l) in sorted(op_counts.items())) or "no comparable operators"
    top_wins = sorted(wins, key=lambda x: x["gain"], reverse=True)[:3]
    win_lines = [f"{x['operator'] or 'unspecified'}: {_trunc(x['experiment'].get('hypothesis'), 70)} ({x['gain']:+.3g})"
                 for x in top_wins]
    top_refuted = [f["content"].split("\n", 1)[0].replace(f"[scope {problem}] ", "") for f in refuted_facts[:3]]
    best_text = (f"best gain {best['gain']:+.6g} (score {best['score']:.12g}, ref {best['ref_score']:.12g})"
                 if best else "no comparable score/ref_score pairs")
    details = f"{best_text}. Operator wins/losses: {op_summary}. Top wins: {'; '.join(win_lines) or 'none'}. " \
              f"Top refuted: {'; '.join(top_refuted) or 'none'}."
    last_ts = max((_timestamp(e) or "" for e in experiments + events), default="") or None
    best_meta = {k: best[k] for k in METADATA_KEYS if k in best} if best else {}
    facts.append(_fact(run_id, problem, "lesson", "run", f"[lesson] v1 outcomes for run {run_id}", details,
                       last_ts, {**base_meta, **best_meta}))
    return facts


def facts_for_runs(run_dirs: list[str | Path]) -> list[dict]:
    out: list[dict] = []
    for run_dir in run_dirs:
        out.extend(facts_for_run(run_dir))
    return out


def summarize(facts: list[dict]) -> dict[str, dict[str, int]]:
    summary: dict[str, Counter] = defaultdict(Counter)
    for fact in facts:
        summary[str(fact.get("metadata", {}).get("run_id"))][fact["kind"]] += 1
    return {run_id: dict(counts) for run_id, counts in sorted(summary.items())}


def write_facts(facts: list[dict], out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for fact in facts:
            if tuple(fact.keys()) != FACT_KEYS:
                raise ValueError(f"invalid fact keys: {tuple(fact.keys())}")
            f.write(json.dumps(fact, ensure_ascii=False, separators=(",", ":")) + "\n")


def _print_summary(facts: list[dict], out: Path) -> None:
    print(f"wrote {len(facts)} facts to {out}")
    for run_id, kinds in summarize(facts).items():
        print(f"{run_id}: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))


def push_facts(path: str | Path) -> int:
    try:
        pastdev = importlib.import_module("knockout.pastdev")
        get_store = pastdev.get_store
    except (ImportError, AttributeError):
        print("past.dev client is not installed; run --push after merging feature/ws2-pastdev-memory", file=sys.stderr)
        return 2
    store = get_store()
    for fact in read_jsonl(Path(path)):
        metadata = dict(fact["metadata"])
        metadata.setdefault("scope", fact.get("scope"))
        metadata.setdefault("author", fact.get("identity"))
        headline = (fact.get("content") or "").split("\n", 1)[0]
        prefix = f"[scope {fact.get('scope')}] "
        if headline.startswith(prefix):
            headline = headline[len(prefix):]
        metadata.setdefault("headline", headline)
        store.ingest(fact["fact_id"], fact["content"], fact["kind"], fact["timestamp"],
                     fact["identity"], metadata)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", type=Path,
                        help="run folders (default: every knockout/results/<run>/ folder)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSONL facts output path")
    parser.add_argument("--push", type=Path, metavar="FACTS_JSONL",
                        help="push an existing facts file through knockout.pastdev.get_store")
    args = parser.parse_args(argv)
    if args.push:
        return push_facts(args.push)
    run_dirs = args.run_dirs or sorted(p for p in DEFAULT_RUNS.glob("*") if p.is_dir())
    facts = facts_for_runs(run_dirs)
    write_facts(facts, args.out)
    _print_summary(facts, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
