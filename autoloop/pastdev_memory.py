"""past.dev as shared memory for the autoloop: a mirror of the local ledgers (the ledger stays the source of truth).

    python -m autoloop.pastdev_memory sync LEDGER [LEDGER ...] [--scope autoloop-1d] [--dry-run]
    python -m autoloop.pastdev_memory recall [--scope autoloop-1d] [--query "..."] [--limit 20]

Reuses the WS4 client tested live (<workspace>/repo/integrations/pastdev_client.py:
PastDevMemory with key failover, LocalMemory fallback when no key). Keys are loaded by that client at run time from
the repo's key file; this module never opens or prints it.

What is sent: facts about experiments only (problem, scores, cells, parameters, moves, reasons, claim verdicts).
Local paths are replaced by "<local>" and file names are dropped. Fact ids are stable
(autoloop/<scope>/<run>/<kind>/<n>), so a second sync replaces instead of duplicating. Imported memory certificates
are mirrored once per scope (top 10 by score + the memory baseline), not once per run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

WORKSPACE = __import__("os").environ.get("AUTOLOOP_WORKSPACE", str((lambda q: q[2] if len(q) > 2 else q[-1])(__import__("pathlib").Path(__file__).resolve().parents)))  # folder that holds repo/, wt-compare/, shared/ and .env
WS4_REPO = Path(WORKSPACE + "/repo")
HERE = Path(__file__).resolve().parent
PATH_RE = re.compile(r"(/Users/|/private/|/tmp/|/var/|[A-Za-z]:\\)[^\s'\",]*")


def _client():
    """Import the WS4 integrations package under another name (wt-attack has its own `integrations`)."""
    os.environ.setdefault("PASTDEV_LOG_PATH", str(HERE / "runs" / "pastdev_log.jsonl"))
    os.environ.setdefault("INTEGRATIONS_STATE_DIR", str(HERE / "runs" / ".integrations_state"))
    if "ws4_integrations" not in sys.modules:
        spec = importlib.util.spec_from_file_location("ws4_integrations", WS4_REPO / "integrations" / "__init__.py",
                                                      submodule_search_locations=[str(WS4_REPO / "integrations")])
        mod = importlib.util.module_from_spec(spec); sys.modules["ws4_integrations"] = mod; spec.loader.exec_module(mod)
    from ws4_integrations import pastdev_client          # noqa: E402
    return pastdev_client


def scrub(x):
    if isinstance(x, str):
        return PATH_RE.sub("<local>", x)
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items() if k not in ("path", "file", "llm_log", "trace", "config")}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    return x


def facts_from_ledger(recs, run, scope):
    """Yield (fact_id, kind, content, metadata) for one ledger. Only experiment facts."""
    base = f"autoloop:{scope}:{run}"
    imported = sorted((r for r in recs if r["kind"] == "certificate" and r.get("origin") == "imported"),
                      key=lambda r: r["score"])[:10]
    for r in imported:
        yield (f"autoloop:{scope}:memory:cert:{r['id'][:16]}", "certificate",
               f"[{r['problem']}] memory certificate {r['id'][:10]}: score {r['score']:.16g} at {r['cells']} cells, "
               f"stage {r['stage']}" + (", exact check passed" if r.get("exact") else ""),
               scrub(dict(problem=r["problem"], cert=r["id"], score=r["score"], cells=r["cells"], stage=r["stage"],
                          exact=r.get("exact"), origin="imported")))
    for r in recs:
        k = r["kind"]
        if k == "memory_baseline":
            yield (f"autoloop:{scope}:memory:baseline", "memory_baseline",
                   f"Best exact score in the frozen memory: {r['best_exact_float']:.16g} ({r['cells']} cells, "
                   f"{r['n_certificates']} certificates)", dict(best_exact=r["best_exact"], cells=r["cells"]))
        elif k == "certificate" and r.get("origin") == "autoloop":
            yield (f"{base}:cert:{r['id'][:16]}", "certificate",
                   f"[{r['problem']}] run {run}: {r['stage']} certificate {r['id'][:10]} from parent "
                   f"{(r.get('parent') or '-')[:10]}: score {r['score']:.16g} at {r['cells']} cells, params "
                   f"{json.dumps(scrub(r.get('params', {})), sort_keys=True)}",
                   scrub(dict(problem=r["problem"], cert=r["id"], parent=r.get("parent"), root=r.get("root"),
                              score=r["score"], cells=r["cells"], stage=r["stage"], params=r.get("params"),
                              exact=r.get("exact"), cpu_s=r.get("cpu_s"))))
        elif k == "decision":
            yield (f"{base}:decision:{r['i']:04d}", "decision",
                   f"run {run} decision #{r['i']} by {r['policy']}: {r['move']} {json.dumps(scrub(r['args']))}. "
                   f"Reason: {scrub(r['reason'])}" + (f" (fallback: {scrub(r['fallback'])})" if r.get("fallback") else ""),
                   scrub(dict(i=r["i"], move=r["move"], args=r["args"], policy=r["policy"], numbers=r.get("numbers"),
                              llm_usage=r.get("llm_usage"))))
        elif k == "result":
            ok = "error" not in r
            yield (f"{base}:result:{r['i']:04d}", "result" if ok else "failure",
                   f"run {run} result of decision #{r['i']} ({r['move']}): {scrub(r.get('summary', ''))}",
                   scrub(dict(i=r["i"], move=r["move"], ok=ok, best_new=r.get("best_new"), rel_gain=r.get("rel_gain"),
                              seconds=r.get("seconds"), error=r.get("error"))))
        elif k == "claim":
            yield (f"{base}:claim:{r['cert'][:16]}", "claim",
                   f"run {run} claim gate: {r['text']}; exact {r['exact_float']:.16g} vs reference {r['reference']}",
                   scrub(dict(cert=r["cert"], exact=r.get("exact"), below_target=r.get("below_target"),
                              reference=r["reference"], reference_source=r["reference_source"])))
        elif k == "run_end":
            yield (f"{base}:run_end", "run_summary",
                   f"run {run} ended ({scrub(r['reason'])}): {r['decisions']} decisions, best {r['best']}, best exact "
                   f"{r['best_exact']}, compute USD {r['usd']}, LLM USD {r['llm_usd']}",
                   scrub({k2: r[k2] for k2 in ("decisions", "best", "best_exact", "usd", "llm_usd", "claims_below")}))


def sync(ledgers, scope, dry_run=False, author="autoloop"):
    mem = None if dry_run else _client().get_memory()
    seen, n = set(), 0
    for lp in ledgers:
        lp = Path(lp)
        run = lp.parent.name
        recs = [json.loads(x) for x in lp.read_text().splitlines() if x.strip()]
        for fid, kind, content, meta in facts_from_ledger(recs, run, scope):
            if fid in seen:
                continue
            seen.add(fid); n += 1
            meta = dict(meta, run=run, source="autoloop")
            if dry_run:
                print(f"{fid}\t{content[:160]}")
            else:
                mem.write(content, author=f"{author}:{run}", kind=kind, label=kind, metadata=meta, wait=False,
                          scope=scope, fact_id=fid)
    backend = "dry-run" if dry_run else mem.name
    print(f"{n} facts -> {backend} (scope {scope})")
    return n


def recall(scope, query="what was tried, what failed, what worked: best certificates, refine settings, claim gate",
           limit=20, reader="autoloop"):
    """Facts for a new run: returned as dicts (fact_id, kind, content, metadata). Data, not instructions."""
    mem = _client().get_memory()
    return mem.read(query, reader=reader, limit=limit, scope=scope)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync"); s.add_argument("ledgers", nargs="+"); s.add_argument("--scope", default="autoloop-1d")
    s.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("recall"); r.add_argument("--scope", default="autoloop-1d"); r.add_argument("--query", default=None)
    r.add_argument("--limit", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "sync":
        sync(a.ledgers, a.scope, a.dry_run)
    else:
        for f in recall(a.scope, **({"query": a.query} if a.query else {}), limit=a.limit):
            print(f"{f['fact_id']}\t{f.get('kind')}\t{(f.get('content') or '')[:200]}")
