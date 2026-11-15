"""Session reporting for SAVANT."""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write_consolidated_export(session_dir: Path) -> Path | None:
    """Merge every wave export (Marie V2 format) into one directory; per-wave exports stay untouched."""
    session_dir = Path(session_dir)
    waves = sorted(p for p in session_dir.glob("iteration_[0-9][0-9]/modal/export") if (p / "experiments.jsonl").exists())
    if not waves:
        return None
    out = session_dir / "final_export"
    if out.exists():
        shutil.rmtree(out)
    (out / "certificates").mkdir(parents=True)
    header = None
    rows: list[str] = []
    manifest = {"waves": [], "format": "Marie V2: experiments.jsonl (RECORD_FIELDS), comparison.csv, "
                                       "certificates/<slug>/<sha12>.json.gz"}
    with (out / "experiments.jsonl").open("w", encoding="utf-8") as exp:
        for w in waves:
            exp.write((w / "experiments.jsonl").read_text(encoding="utf-8"))
            lines = (w / "comparison.csv").read_text(encoding="utf-8").splitlines()
            if lines:
                header = header or lines[0]
                rows += lines[1:]
            certs = w / "certificates"
            if certs.exists():
                for f in certs.rglob("*.json.gz"):
                    dest = out / "certificates" / f.relative_to(certs)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f, dest)
            manifest["waves"].append(str(w.relative_to(session_dir)))
    (out / "comparison.csv").write_text("\n".join([header or ""] + rows) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def write_session_report(session_dir: Path) -> Path:
    from savant.config import SavantConfig
    from savant.loop import CONTROL_ID, build_evidence, collect_waves, _valid_decisions
    from savant.memory import ResearchMemory

    session_dir = Path(session_dir)
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8")) if (session_dir / "session.json").exists() else {}
    memory = ResearchMemory(session_dir / "research_memory.jsonl")
    rows_all = memory.read_all()
    rows = memory.read()
    report = json.loads((session_dir / "problem_report.json").read_text(encoding="utf-8")) \
        if (session_dir / "problem_report.json").exists() else {}
    cfg = SavantConfig()
    evidence = build_evidence({"dir": session_dir, "memory": memory, "report": report}, cfg)
    waves = collect_waves(session_dir)
    decisions = _valid_decisions(session_dir)
    spend = json.loads((session_dir / "spend_ledger.json").read_text(encoding="utf-8")) \
        if (session_dir / "spend_ledger.json").exists() else {}
    stats_md = (session_dir / "statistical_report.md").read_text(encoding="utf-8") \
        if (session_dir / "statistical_report.md").exists() else ""
    export = write_consolidated_export(session_dir)

    best = evidence.get("best_verified")
    lines = [
        f"# SAVANT report: {session.get('session_id', session_dir.name)}",
        "",
        "Multi-agent council on Marie V2 + WS3 Modal. Optimizer templates only; no LLM-generated code is run. "
        "Scores are re-verified in the parent by the frozen WS1 verifier.",
        "",
        "## Result",
        "",
        f"- Baseline (WS1 frozen certificate): `{evidence.get('baseline_score')!r}`",
        f"- Record target (displayed #1 + delta): `{evidence.get('record_target_score')!r}`",
    ]
    if best:
        lines += [f"- Best verified score: `{best['score']!r}` ({best['label']}, seed {best['seed']}, "
                  f"iteration {best['iteration']})",
                  f"- Gain over baseline: `{best['score'] - (evidence.get('baseline_score') or 0):.6e}`; "
                  f"distance to record target: `{evidence.get('best_distance_to_target'):.6e}`"]
    else:
        lines.append("- No verified wave yet.")
    lines += ["", "## Progress per iteration", "",
              "| it | budget s | arms | jobs | best mean gain (non-control) | best score | decision |",
              "|---|---|---|---|---|---|---|"]
    for w in waves:
        arms = [a for a in w["arms"] if a != CONTROL_ID]
        means = [s["mean_gain"] for s in evidence["statistics"] if s["iteration"] == w["iteration"]
                 and s["mean_gain"] is not None]
        scores = [sc for a in w["arms"].values() for sc in a["scores"].values() if sc is not None]
        d = decisions.get(w["iteration"], {})
        lines.append(f"| {w['iteration']} | {w['budget_s']:.0f} | {len(arms)}"
                     f"{' + control' if CONTROL_ID in w['arms'] else ''} | {w['plan'].get('n_jobs')} | "
                     f"{max(means):.4e} | {max(scores)!r} | {d.get('decision', '-')}"
                     f"{' (re-held)' if d.get('reheld') else ''} |" if means and scores else
                     f"| {w['iteration']} | {w['budget_s']:.0f} | {len(arms)} | {w['plan'].get('n_jobs')} | - | - | "
                     f"{d.get('decision', '-')} |")
    lines += ["", "## Paired statistics (seeds are the unit; one-sided exact sign-flip)", "",
              "| it | strategy | mean gain | vs control: n, mean diff, p, claim | vs Marie smoke blockshift: n, mean diff, p |",
              "|---|---|---|---|---|"]
    for s in evidence["statistics"]:
        vc, vm = s.get("vs_control") or {}, s.get("vs_marie_smoke") or {}
        fmt = lambda c: (f"{c.get('n')}, {c.get('mean_diff'):+.3e}, {c.get('p_sign_flip_one_sided'):.4f}"
                         if c else "-")
        lines.append(f"| {s['iteration']} | {s['label']} | "
                     f"{(s['mean_gain'] if s['mean_gain'] is not None else float('nan')):.4e} | "
                     f"{fmt(vc)}{', ' + str(vc.get('claim_allowed')) if vc else ''} | {fmt(vm)} |")
    lines += ["", "Marie's smoke benchmark comparisons use only seeds 101/202/303 at 60 s, so n<=3 and no "
                  "significance claim is possible (minimum p = 0.125).", ""]

    lines += ["## Hypotheses selected", ""]
    for e in rows:
        if e.get("kind") == "manager_decision":
            mr = e.get("manager_review") or {}
            lines.append(f"- {e.get('ts')}: {', '.join(e.get('selected_ids') or [])}"
                         + (f" — manager: {mr.get('rationale', '')[:300]}" if mr else ""))
    lines += ["", "## Debate", ""]
    crit = [e for e in rows if e.get("kind") == "critique"]
    lines.append(f"- Critiques: {len(crit)}; votes: {sum(1 for e in rows if e.get('kind') == 'vote')}; "
                 f"revisions: {sum(1 for e in rows if e.get('kind_detail') == 'revision')}")
    for c in crit[-10:]:
        obj = (c.get("objections") or [""])[0]
        lines.append(f"  - {c.get('reviewer_role')} on {c.get('target_id')}: score {c.get('score')}; {obj[:200]}")
    lines += ["", "## Rejected or queued ideas", ""]
    for e in rows:
        if e.get("kind") == "rejected_idea":
            lines.append(f"- {e.get('idea')}: {str(e.get('reason'))[:200]}{' (queued)' if e.get('queued') else ''}")
        if e.get("kind") == "note" and e.get("kind_detail") == "llm_call_failed":
            lines.append(f"- failed request {e.get('stage')}: {str(e.get('error'))[:200]}")
    inval = len(rows_all) - len(rows)
    if inval:
        lines.append(f"- {inval} records invalidated (mock-LLM outputs kept on disk for audit, hidden from decisions)")

    jobs = sum(int(w["plan"].get("n_jobs") or 0) for w in waves)
    search_s = sum(int(w["plan"].get("n_jobs") or 0) * w["budget_s"] for w in waves)
    cpu_s = sum(a["cpu_user_s"] for w in waves for a in w["arms"].values())
    modal_usd = sum(float(w["plan"].get("estimated_modal_usd") or 0) for w in waves)
    lines += ["", "## Compute and spending", "",
              f"- Modal jobs: {jobs}; search budget {search_s:.0f} CPU-s; measured worker CPU user time "
              f"{cpu_s:.0f} s; committed Modal estimate ${modal_usd:.4f} (list prices, startup included, "
              "not an invoice)",
              f"- OpenAI spent (ledger, conservative): ${spend.get('spent_usd', 0):.4f} of ${spend.get('max_usd', 0):.2f}"]
    for model, s in (spend.get("by_model") or {}).items():
        lines.append(f"  - {model} (effort {s.get('reasoning_effort')}): {s.get('calls')} requests, "
                     f"input {s.get('input_tokens')} (cached {s.get('cached_input_tokens')}), output "
                     f"{s.get('output_tokens')} (reasoning {s.get('reasoning_tokens')}), ${s.get('usd', 0):.4f}")
    lines += ["", "## Exports", "",
              f"- Consolidated (Marie V2 format): `{export}`" if export else "- No wave export yet.",
              "- Per-wave exports: " + ", ".join(f"`{Path(w['dir']).name}/modal/export`" for w in waves),
              "", "## Ingested smoke statistics", "", stats_md]
    path = session_dir / "savant_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
