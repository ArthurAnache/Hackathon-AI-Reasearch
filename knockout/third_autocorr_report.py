"""Tables for the exploratory runs on third-autocorrelation-inequality, from their ledgers only.

    .venv/bin/python -m knockout.third_autocorr_report knockout/results/third_autocorr/<run_id> [...]
    .venv/bin/python -m knockout.third_autocorr_report --rescore <full run dir with solutions/> [...]

For each run: the row of the README table, the kept edits, and the distribution of the differences between a
child program and the replay of its parent from the same warm start, in units of the record margin delta (1e-5).
With `--rescore`, the best candidate of each run is scored again here by the pinned verifier and by the exact
rational scorer, and the result is written to <run dir>/best_scores.json (about 3 s at 25,600 values and 20 s at
65,536). Nothing here is a record claim: the problem is not frozen by WS1.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from knockout.problems_api import ScoutProblem

SLUG = "third-autocorrelation-inequality"
SKIP = (None, "replay", "seed", "freeze-tune")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _quantile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def analyse(run_dir: Path, problem: ScoutProblem) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    events = _jsonl(run_dir / "events.jsonl")
    rows = _jsonl(run_dir / "experiments.jsonl")
    delta = problem.delta
    ref = (summary.get("reference") or {}).get("components") or {}
    live, claim = ref.get("live_top1"), ref.get("literature_claim")
    start, best = summary["init_score"], summary["best"]
    calls = [e["payload"] for e in events if e["type"] == "llm_call"]
    by_model: dict[str, dict] = {}
    for c in calls:
        m = by_model.setdefault(c.get("model"), {"calls": 0, "failed": 0, "usd": 0.0, "in": 0, "out": 0})
        m["calls"] += 1
        m["failed"] += not c["ok"]
        m["usd"] += c.get("usd") or 0.0
        m["in"] += c.get("input_tokens") or 0
        m["out"] += c.get("output_tokens") or 0
    # One entry per program written by an editor: its gain over the replay of its parent, and over the warm start.
    children = []
    for r in rows:
        cfg = r["config"]
        if cfg.get("operator") in SKIP or cfg.get("fidelity", "F0") != "F0":
            continue
        model = cfg.get("model") or str(cfg.get("editor", "")).removeprefix("openai:")
        vs_parent = None if r["score"] is None or cfg.get("ref_score") is None else (cfg["ref_score"] - r["score"]) / delta
        warm = r.get("best_before")
        children.append({"experiment_id": r["experiment_id"], "arm": cfg.get("arm"), "operator": cfg.get("operator"),
                         "model": model, "valid": r["valid"], "status": r["status"], "decision": r["decision"],
                         "vs_parent_delta": vs_parent,
                         "vs_warm_start_delta": None if r["score"] is None or warm is None else (warm - r["score"]) / delta,
                         "hypothesis": r["hypothesis"][:300], "error": (r.get("error") or "")[:200]})
    replays = [(r["best_before"] - r["score"]) / delta for r in rows
               if r["config"].get("operator") == "replay" and r["score"] is not None and r.get("best_before") is not None]

    def dist(items: list[dict]) -> dict:
        d = [c["vs_parent_delta"] for c in items if c["vs_parent_delta"] is not None]
        w = [c["vs_warm_start_delta"] for c in items if c["vs_warm_start_delta"] is not None]
        if not d:
            return {"programs": len(items), "valid": 0}
        return {"programs": len(items), "valid": len(d), "kept": sum(c["decision"] == "keep" for c in items),
                "better_than_parent": sum(x > 0 for x in d), "equal_to_parent": sum(x == 0 for x in d),
                "worse_than_parent": sum(x < 0 for x in d),
                "abs_diff_above_0.001_delta": sum(abs(x) > 1e-3 for x in d),
                "abs_diff_above_0.01_delta": sum(abs(x) > 1e-2 for x in d),
                "abs_diff_above_0.1_delta": sum(abs(x) > 1e-1 for x in d),
                "min": min(d), "median": statistics.median(d), "p90": _quantile(d, 0.9), "max": max(d),
                "median_abs": statistics.median(abs(x) for x in d),
                "gained_over_warm_start": sum(x > 0 for x in w), "max_gain_over_warm_start": max(w) if w else None}

    groups = {"all": children}
    for c in children:
        groups.setdefault(f"model={c['model']}", []).append(c)
        groups.setdefault(f"arm={c['arm']}", []).append(c)
        groups.setdefault(f"operator={c['operator']}", []).append(c)
    budget = summary["budget"]
    fired = [e for e in events if e["type"] == "promotion" and e["payload"]["outputs"]["decision"] == "record_claim"]
    return {
        "run_id": summary["run_id"], "provenance": summary.get("provenance"), "exploratory": summary.get("exploratory"),
        "editor": summary["editor"], "arms": {k: {"best": v["best"], "steps": v["steps"]} for k, v in summary["arms"].items()},
        "models": sorted(m for m in by_model if m), "stop_reason": summary["stop_reason"], "wall_s": summary["wall_s"],
        "experiments": len(rows), "llm_calls": len(calls), "llm_failed": sum(not c["ok"] for c in calls),
        "llm_by_model": {k: {**v, "usd": round(v["usd"], 4)} for k, v in by_model.items()},
        "start": start, "best": best, "best_arm": summary.get("best_arm"), "best_experiment": summary.get("best_experiment"),
        "gain_over_start_delta": (start - best) / delta,
        "below_live_top1_delta": None if live is None else (live - best) / delta,
        "below_literature_claim_delta": None if claim is None else (claim - best) / delta,
        "live_top1": live, "literature_claim": claim, "reference_live_read": (summary.get("reference") or {}).get("live"),
        "usd_llm": round(sum(budget["llm_usd"].values()), 4), "usd_total_estimate": budget["total_usd"],
        "usd_modal_estimate": round(budget["total_usd"] - sum(budget["llm_usd"].values()), 4),
        "compute_s": round(sum(budget["compute_s"].values()), 1),
        "exploratory_gate_fired": len(fired),
        "replay_gain_over_warm_start_delta": replays,
        "child_vs_parent": {k: dist(v) for k, v in groups.items()},
        "kept_edits": [c for c in children if c["decision"] == "keep"],
        "top_children": sorted((c for c in children if c["vs_parent_delta"] is not None),
                               key=lambda c: -c["vs_parent_delta"])[:5],
        "invalid_or_crashed": sum(not c["valid"] for c in children),
    }


def rescore(run_dir: Path, problem: ScoutProblem) -> dict | None:
    """Pinned verifier and exact rational scorer on the best candidate of a run (needs its solutions/)."""
    summary = json.loads((run_dir / "summary.json").read_text())
    rel = summary.get("best_solution_path")
    if not rel or not (run_dir / rel).exists():
        return None
    cand = json.loads((run_dir / rel).read_text())
    pinned = problem.score(cand)
    exact = problem.independent(cand)
    out = {"run_id": summary["run_id"], "experiment_id": summary["best_experiment"], "solution_path": rel,
           "n_values": len(cand["values"]), "score_pinned_verifier": pinned, "score_exact_rational": exact["score"],
           "pinned_minus_exact": pinned - exact["score"], "argmax_lag": exact["argmax_lag"],
           "ledger_score": summary["best"], "verifier_sha256": problem.meta["verifier"]["sha256"],
           "label": "verified here with both scorers; a candidate, not a record claim (problem not frozen)"}
    (run_dir / "best_scores.json").write_text(json.dumps(out, indent=1))
    return out


def _f(x, nd=4) -> str:
    return "n/a" if x is None else f"{x:+.{nd}f}"


def main(argv: list[str]) -> int:
    do_rescore = "--rescore" in argv
    dirs = [Path(a) for a in argv if not a.startswith("--")]
    if not dirs:
        print(__doc__)
        return 2
    problem = ScoutProblem(SLUG)
    print("| Run | Provenance | Arms | Models | Wall clock | Experiments | LLM calls | Start | Best | Gain over its "
          "start | Below live #1 | Below MILO's claim | LLM cost | Modal (est.) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    out = []
    for d in dirs:
        a = analyse(d, problem)
        out.append(a)
        (d / "analysis.json").write_text(json.dumps(a, indent=1))
        print(f"| `{a['run_id']}` | `{a['provenance']}` | {' + '.join(a['arms'])} | {', '.join(a['models'])} | "
              f"{a['wall_s'] / 60:.1f} min | {a['experiments']} | {a['llm_calls']} ({a['llm_failed']} failed) | "
              f"{a['start']!r} | {a['best']!r} | {a['gain_over_start_delta']:.4f} δ | "
              f"{_f(a['below_live_top1_delta'], 3)} δ | {_f(a['below_literature_claim_delta'], 3)} δ | "
              f"{a['usd_llm']:.2f} USD | {a['usd_modal_estimate']:.2f} USD |")
    for a in out:
        print(f"\n### {a['run_id']} ({a['provenance']}; stop: {a['stop_reason']})")
        print(f"replays of the incumbent, gain over the warm start (δ): "
              + ", ".join(f"{x:.2e}" for x in a["replay_gain_over_warm_start_delta"]))
        for k, v in a["child_vs_parent"].items():
            print(f"- {k}: {json.dumps({kk: (float(f"{vv:.3g}") if isinstance(vv, float) else vv) for kk, vv in v.items()})}")
        print(f"- exploratory gate fired: {a['exploratory_gate_fired']} time(s); LLM by model: {a['llm_by_model']}")
        for c in a["kept_edits"]:
            print(f"  kept {c['experiment_id']} [{c['arm']}/{c['operator']}/{c['model']}] "
                  f"{_f(c['vs_parent_delta'])} δ vs parent, {_f(c['vs_warm_start_delta'])} δ vs warm start: "
                  f"{c['hypothesis'][:220]}")
    if do_rescore:
        print()
        for d in dirs:
            print(json.dumps(rescore(d, problem)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
