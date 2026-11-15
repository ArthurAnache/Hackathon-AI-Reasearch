"""Tournament: the same budget and the same memory under several policies, then one comparison table.

    python -m autoloop.tournament --out autoloop/runs/T1 \
        --entrant rules --entrant llm:openai:MODEL --entrant llm:claude-cli: \
        [--parallel] -- --executor local --max-wall-s 1200 --memory SRC ...

Everything after `--` is passed unchanged to every entrant's `python -m autoloop.loop`. An entrant is `rules` or
`llm:<provider>:<model>` (empty model = the provider's default). Entrants run one after another by default, so they
do not share the CPU; `--parallel` runs them at once (only fair when workers x entrants <= cores).
Results: OUT/<entrant>/ledger.jsonl, OUT/comparison.md, OUT/comparison.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def entrant_args(e):
    if e == "rules":
        return ["--policy", "rules"]
    _, prov, model = (e.split(":", 2) + [""])[:3]
    return ["--policy", "llm", "--llm-provider", prov] + (["--llm-model", model] if model else [])


def slug(e):
    return e.replace(":", "_").replace("/", "-").strip("_") or "x"


def summarize(name, path):
    recs = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    end = next((r for r in reversed(recs) if r["kind"] == "run_end"), {})
    start = next((r for r in recs if r["kind"] == "run_start"), {"t": recs[0]["t"] if recs else 0})
    dec = [r for r in recs if r["kind"] == "decision"]
    res = [r for r in recs if r["kind"] == "result"]
    final = end.get("best")
    t_best = next((r["t"] - start["t"] for r in res if final is not None and r.get("best_after") == final), None)
    tok = sum(d.get("llm_usage", {}).get("tokens_in", 0) + d.get("llm_usage", {}).get("tokens_out", 0) for d in dec)
    return dict(entrant=name, decisions=end.get("decisions"), wall_s=end.get("wall_s"), cpu_s=end.get("cpu_s"),
                compute_usd=end.get("usd"), llm_usd=end.get("llm_usd"), llm_tokens=tok,
                fallbacks=sum(1 for d in dec if d.get("fallback")), errors=sum(1 for r in res if r.get("error")),
                best=final, best_exact=end.get("best_exact"), claims_below=end.get("claims_below"),
                time_to_best_s=round(t_best) if t_best is not None else None, stop=end.get("reason"))


def table(rows):
    cols = ["entrant", "best", "best_exact", "claims_below", "time_to_best_s", "decisions", "wall_s", "cpu_s",
            "compute_usd", "llm_usd", "llm_tokens", "fallbacks", "errors", "stop"]
    f = lambda v: f"{v:.13f}" if isinstance(v, float) and 1 < v < 2 else str(v)  # noqa: E731
    rows = sorted(rows, key=lambda r: (r["best_exact"] is None, r["best_exact"] or 9, r["best"] or 9))
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(f(r[c]) for c in cols) + " |" for r in rows])


def main():
    argv = sys.argv[1:]
    shared = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--entrant", action="append", required=True)
    ap.add_argument("--parallel", action="store_true")
    a = ap.parse_args(own)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    procs = []
    for e in a.entrant:
        d = out / slug(e)
        cmd = [sys.executable, "-m", "autoloop.loop", "--run", str(d), *shared, *entrant_args(e), "--no-distill"]
        log = open(out / f"{slug(e)}.log", "w")
        print(f"[tournament] {e}: {' '.join(cmd[2:])}", flush=True)
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        if a.parallel:
            procs.append(p)
        else:
            p.wait()
    for p in procs:
        p.wait()
    rows = [summarize(e, out / slug(e) / "ledger.jsonl") for e in a.entrant if (out / slug(e) / "ledger.jsonl").exists()]
    (out / "comparison.json").write_text(json.dumps(rows, indent=1))
    if "fake" not in shared:                    # one distillation over all arms (no race on lessons.json)
        from .distill import distill
        ledgers = [out / slug(e) / "ledger.jsonl" for e in a.entrant if (out / slug(e) / "ledger.jsonl").exists()]
        print("[tournament] lessons:", json.dumps(distill(ledgers, outdir=out, pastdev=True))[:300])
    md = table(rows)
    (out / "comparison.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
