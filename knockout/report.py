"""Markdown report of a run, written from its ledger only (section 6).

    .venv/bin/python -m knockout.report runs/<run_id>          # writes runs/<run_id>/report.md

Sections: summary, control arm versus main arm at equal cumulative spend, genealogy of the best candidate (genes
active at each step, with their posteriors and knock-out evidence), gene table, strategy patches and their outcomes.
The report never states a record unless the promotion gate logged `record_claim`, and it says that such a claim
still has to pass selection/verify_claim.py.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _at(curve: list, spend: float):
    best = None
    for x, s in curve:
        if x <= spend + 1e-12:
            best = s
    return best


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.12g}" if isinstance(v, float) else str(v)


def build(run_dir: str | Path) -> str:
    d = Path(run_dir)
    summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    state = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
    events = _read_jsonl(d / "events.jsonl")
    exps = _read_jsonl(d / "experiments.jsonl")
    by_exp = {r["experiment_id"]: r for r in exps}
    genes = {g["id"]: g for g in state.get("genes", [])}
    members = {m["id"]: m for m in state.get("members", [])}
    out: list[str] = [f"# Run report — {d.name}", ""]

    # ------------------------------------------------------------------ summary
    claims = [e for e in events if e["type"] == "promotion"]
    record_claims = [e for e in claims if e["payload"]["outputs"]["decision"] == "record_claim"]
    arms = summary.get("arms", {})
    out += ["## Summary", "",
            f"- Problem: `{summary.get('problem')}`; editor: `{summary.get('editor')}`; stop reason: "
            f"{summary.get('stop_reason')}.",
            f"- Record to beat (read at run time): {_fmt(summary.get('best_known'))}. Warm start: "
            f"{_fmt(summary.get('init_score'))}.",
            f"- Best verified score of the run: **{_fmt(summary.get('best'))}** (arm `{summary.get('best_arm')}`, "
            f"experiment `{summary.get('best_experiment')}`, solution `{summary.get('best_solution_path')}`)."]
    for name, a in arms.items():
        out.append(f"- Arm `{name}`: best {_fmt(a.get('best'))}, {a.get('steps')} steps, spend {_fmt(a.get('spend'))}.")
    b = summary.get("budget", {})
    out.append(f"- Budget: {b.get('elapsed_s')} s of {b.get('wall_clock_s')} s; {b.get('total_usd')} USD of "
               f"{b.get('max_usd')} USD (estimates: Modal compute + LLM).")
    if record_claims:
        out.append(f"- **Promotion gate: {len(record_claims)} `record_claim`** ("
                   + ", ".join(e["refs"][0] for e in record_claims) + "). A claim is only a candidate until "
                   "`selection/verify_claim.py` confirms it against the live leaderboard. Nothing was submitted.")
    else:
        out.append("- Promotion gate: no `record_claim`. **This run does not beat the record.**")
    artifacts = [e for e in claims if e["payload"]["outputs"]["decision"] == "numerical_artifact"]
    if artifacts:
        out.append(f"- Numerical artifacts rejected by the independent verifier: {len(artifacts)}.")
    st = Counter((r["config"].get("arm"), r["status"]) for r in exps)
    out.append("- Experiments by arm and status: " + ", ".join(f"{a}/{s}: {n}" for (a, s), n in sorted(st.items(), key=str)))
    calls = [e["payload"] for e in events if e["type"] == "llm_call"]
    if calls:
        roles = Counter(c.get("role") for c in calls)
        out.append(f"- LLM calls: {len(calls)} ({sum(not c['ok'] for c in calls)} failed), "
                   f"{sum(c.get('usd', 0) for c in calls):.3f} USD estimated; by role: "
                   + ", ".join(f"{k}: {v}" for k, v in sorted(roles.items(), key=str)))
    rejects = Counter(e["payload"].get("reason", "").split(":")[0] for e in events if e["type"] == "filter_reject")
    if rejects:
        out.append("- Programs rejected before evaluation: " + ", ".join(f"{k}: {v}" for k, v in rejects.items()))
    out.append("")

    # ------------------------------------------------------------------ arms at equal spend
    cc, mc = summary.get("control_curve") or [], summary.get("main_curve") or []
    out += ["## Control arm versus main arm at equal cumulative spend", ""]
    if cc and mc:
        top = min(cc[-1][0], mc[-1][0])
        unit = "USD" if b.get("max_usd") and b.get("total_usd") else "compute seconds"
        out += [f"Spend unit: {unit}. Each cell is the best verified score the arm had reached at that spend.", "",
                "| Spend | Control (Hill Sampling) | Main (genes) | Main − control |", "|---|---|---|---|"]
        for k in range(1, 9):
            x = top * k / 8
            c, m = _at(cc, x), _at(mc, x)
            diff = "—" if c is None or m is None else f"{m - c:+.3e}"
            out.append(f"| {x:.4g} | {_fmt(c)} | {_fmt(m)} | {diff} |")
        out += ["", f"Control spent {cc[-1][0]:.4g} in total and the main arm {mc[-1][0]:.4g}; the table stops at "
                    f"the smaller of the two."]
    else:
        out.append("Only one arm ran, so there is no comparison.")
    out.append("")

    # ------------------------------------------------------------------ genealogy
    verdicts: dict[str, list[dict]] = {}
    for e in events:
        if e["type"] == "gate_verdict" and e["payload"]["outputs"]["verdict"]:
            verdicts.setdefault(e["payload"]["gene"], []).append(e["payload"])

    def gene_line(gid: str) -> str:
        g = genes.get(gid)
        if g is None:
            return f"  - `{gid}` (unknown)"
        a = 1 + sum(v[0] for v in g["stats"].values())
        bb = 1 + sum(v[1] for v in g["stats"].values())
        ev = "; ".join(f"{'re-audit' if v.get('audit') else 'gate'}: {v['inputs']['wins']} wins, "
                       f"{v['inputs']['losses']} losses in {v['inputs']['pairs']} pairs, p = {v['outputs']['p']:.3f} "
                       f"→ {v['outputs']['verdict']}" for v in verdicts.get(gid, [])) or "no verdict logged"
        return f"  - `{gid}` ({g['type']}, {g['status']}, posterior mean {a / (a + bb):.2f}): {g['text']} — {ev}"

    out += ["## Genealogy of the best candidate of the main arm", ""]
    mid = state.get("best_member")
    if mid is None or mid not in members:
        out.append("The main arm did not run, or its best candidate is the warm start itself.")
    else:
        depth = 0
        while mid in members and depth < 30:
            m = members[mid]
            exp = next(iter(sorted(m["exp_ids"].values())), None)
            score = by_exp.get(exp, {}).get("score")
            out.append(f"{depth}. `{m['id']}` (island {m['island_id']}, lineage {m['lineage']}, operator "
                       f"`{m['operator']}`, model `{m['model']}`), first score {_fmt(score)} in `{exp}`: "
                       f"{m['hypothesis'][:300]}")
            out += [gene_line(g) for g in m["genome"]] or ["  - no gene active at this step"]
            mid = m["parents"][0] if m["parents"] else None
            depth += 1
    out.append("")

    # ------------------------------------------------------------------ gene table
    out += ["## Gene table", ""]
    if genes:
        out += ["| Gene | Island | Type | Status | Origin | Used | Posterior mean | Text |", "|---|---|---|---|---|---|---|---|"]
        for g in genes.values():
            a = 1 + sum(v[0] for v in g["stats"].values())
            bb = 1 + sum(v[1] for v in g["stats"].values())
            out.append(f"| {g['id']} | {g['island_id']} | {g['type']} | {g['status']} | {g['origin']} | {g['usage']} "
                       f"| {a / (a + bb):.2f} | {g['text'].replace('|', '/')} |")
        gs = Counter(g["status"] for g in genes.values())
        out += ["", "By status: " + ", ".join(f"{k}: {v}" for k, v in sorted(gs.items())) + ". A gene is `validated` "
                    "or `refuted` only by its knock-out pairs (see the genealogy for the evidence format)."]
    else:
        out.append("No gene (the main arm did not run).")
    out.append("")

    # ------------------------------------------------------------------ strategy patches
    out += ["## Strategy patches and their outcomes", ""]
    patches = [e for e in events if e["type"] == "strategy_patch"]
    reverts = {e["refs"][0]: e["payload"] for e in events if e["type"] == "strategy_revert" and e["refs"]}
    if not patches:
        out.append("The strategist was not triggered.")
    for e in patches:
        p = e["payload"]
        r = reverts.get(e["event_id"])
        outcome = "still under observation at the end of the run" if r is None else (
            f"{'reverted' if r['outputs']['revert'] else 'kept'}: progress {r['inputs']['progress_after']:.3e} after "
            f"vs {r['inputs']['progress_before']:.3e} before")
        out += [f"- `{e['event_id']}` at {p['inputs']['fraction_now']:.0%} of the budget: `{json.dumps(p['patch'])}`",
                f"  - Rationale (LLM): {p['rationale'][:600]}",
                f"  - Rejected entries: {p['rejected'] or 'none'}", f"  - Outcome: {outcome}"]
    out.append("")
    out += _portfolio_section(state, events, exps)
    out += memory_section(d, events, exps)
    return "\n".join(out)


def _portfolio_section(state: dict, events: list[dict], exps: list[dict]) -> list[str]:
    """Start portfolio tables (docs/knockout_portfolio_brief.md, section 7). Empty when the portfolio was off."""
    pf = state.get("portfolio")
    if not pf:
        return []
    out = ["## Start portfolio", "", "### Regimes", "",
           "| Regime | Batches | Spend (USD) | Children | Beat their parent | Share | Best regime score | Best progress |",
           "|---|---|---|---|---|---|---|---|"]
    for r, rs in pf["regime_stats"].items():
        share = rs["beat_parent"] / rs["children"] if rs["children"] else None
        out.append(f"| {r} | {rs['batches']} | {_fmt(pf['regime_spend'].get(r))} | {rs['children']} | "
                   f"{rs['beat_parent']} | {_fmt(share)} | {_fmt(rs['best_regime_score'])} | "
                   f"{_fmt(rs['best_progress'])} |")
    picks = Counter((e["payload"]["outputs"]["choice"], e["payload"]["outputs"]["reason"])
                    for e in events if e["type"] == "regime_pick")
    out += ["", "Regime picks (regime, reason): " + ", ".join(f"{k[0]}/{k[1]}: {v}" for k, v in sorted(picks.items())),
            "", "Selection inside `neighbour` and `cold` uses the mean verified score over the start set "
            "(`config.regime_score`); the experiment's `score` is its best per-start candidate, the one stored.", ""]
    ops: dict[tuple[str, str], list[int]] = {}
    for r in exps:
        c = r["config"]
        if c.get("arm") != "main" or c.get("operator") in (None, "seed", "replay", "freeze-tune", "deep-polish") \
                or c.get("fidelity") != "F0" or "regime" not in c:
            continue
        v = ops.setdefault((c["regime"], c["operator"]), [0, 0])
        v[0] += 1
        v[1] += r["decision"] == "keep"
    out += ["### Operator success per regime", "", "| Regime | Operator | Children | Beat their parent |", "|---|---|---|---|"]
    out += [f"| {k[0]} | {k[1]} | {v[0]} | {v[1]} |" for k, v in sorted(ops.items())] or ["| — | — | 0 | 0 |"]
    out += ["", "### Genes per regime", "",
            "Statistics are kept under `<regime>/<phase>`; the verdict of the knock-out gate is still global "
            "(limitation: per-regime verdicts were not built).", "",
            "| Gene | Regime | Posterior mean | Observations | Status (global) |", "|---|---|---|---|---|"]
    for g in state.get("genes", []):
        per: dict[str, list[float]] = {}
        for key, (a, b, n) in g.get("stats", {}).items():
            reg = key.split("/")[0] if "/" in key else "—"
            v = per.setdefault(reg, [0.0, 0.0, 0])
            v[0] += a
            v[1] += b
            v[2] += n
        for reg, (a, b, n) in sorted(per.items()):
            out.append(f"| {g['id']} | {reg} | {(1 + a) / (2 + a + b):.2f} | {n} | {g['status']} |")
    out += ["", "### Basin archive", "", f"Descriptor bin edges: `{json.dumps(pf.get('archive_edges'))}`", "",
            "| Cell | Best score | Experiment | Regime that found it | Deep-polished |", "|---|---|---|---|---|"]
    out += [f"| {e['cell']} | {_fmt(e['score'])} | {e['experiment']} | {e['regime']} | {e['deep_polished']} |"
            for e in pf.get("archive", [])]
    out += ["", "### Deep polish (basin promotions)", ""]
    if not pf.get("promotions"):
        out.append("No archive entry came within `promote_gap` of the incumbent outside its cell.")
    for pr in pf.get("promotions", []):
        line = (f"- `{pr['experiment']}`: cell {pr['entry']['cell']} ({pr['entry']['regime']}), program {pr['program']}, "
                f"{_fmt(pr['score_before'])} → {_fmt(pr['score_after'])} (incumbent {_fmt(pr['incumbent_score'])})")
        out.append(f"**{line} — new incumbent from another basin**" if pr["new_incumbent_from_other_basin"] else line)
    out.append("")
    return out


def memory_section(d: Path, events: list[dict], exps: list[dict]) -> list[str]:
    """Shared memory (knockout/memory.py): what was recalled, from whom, what was written, and the efficiency
    metrics of the memory brief: repeat rate of refuted ideas, share of children that beat their parent, USD per
    winning child. No score claim is made from these numbers."""
    from knockout.memory import jaccard

    out = ["## Memory (past.dev)", ""]
    recalls = [e["payload"] for e in events if e["type"] == "memory_recall"]
    writes = [e["payload"] for e in events if e["type"] == "memory_write"]
    errors = [e["payload"] for e in events if e["type"] == "memory_error"]
    if not (recalls or writes or errors):
        return out + ["Memory was off for this run.", ""]
    run_id = d.name
    store = (recalls or writes or [{}])[0].get("store")
    foreign = [r for r in recalls if any(x != run_id for x in r.get("runs", []))]
    authors = sorted({a for r in recalls for a in r.get("authors", []) if not a.endswith("@" + run_id)})
    kinds: dict = {}
    for w in writes:
        kinds[w["kind"]] = kinds.get(w["kind"], 0) + 1
    out += [f"- Store: {store}. Recalls: {len(recalls)} (one per main batch / control round), "
            f"{sum(len(r.get('fact_ids', [])) for r in recalls)} facts shown, errors: {len(errors)}.",
            f"- Prompts that contained a fact from another run or agent: {len(foreign)} of {len(recalls)}; "
            f"authors recalled from elsewhere: {', '.join(authors) or 'none'}.",
            "- Facts written: " + (", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) or "none") + "."]
    refuted = [t for r in recalls for t in r.get("refuted_texts", [])]
    props = [x for x in exps if x.get("config", {}).get("operator") not in (None, "replay", "freeze-tune", "seed")]
    if refuted and props:
        rep = sum(any(jaccard(x["hypothesis"], t) >= 0.6 for t in refuted) for x in props)
        out.append(f"- Repeat rate: {rep} of {len(props)} proposals ({rep / len(props):.1%}) are near-duplicates "
                   "(word Jaccard >= 0.6) of a recalled refuted idea or failed edit.")
    keep = sum(x.get("decision") == "keep" for x in props)
    usd = sum(e["payload"].get("usd", 0.0) for e in events if e["type"] == "llm_call")
    out.append(f"- Children that beat their parent: {keep} of {len(props)}"
               + (f"; LLM USD per winning child: {usd / keep:.3f}." if keep else "; no winning child."))
    out.append("")
    return out


def write(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "report.md"
    path.write_text(build(run_dir))
    return path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    print(write(sys.argv[1]))
