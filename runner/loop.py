"""One iteration of the autoresearch loop.

    hash-check frozen files
    -> read history
    -> agent proposes hypothesis + edits (strategy/ only)
    -> apply edits, re-check scope
    -> run with fixed seeds + budget, verify independently
    -> keep (git commit) or reject (git revert)
    -> append a row to experiments.jsonl and commit the log
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from problem import DIRECTION
from . import frozen, gitops, history
from .agent import Context, Proposal, make_agent
from .evaluate import evaluate


class ScopeViolation(RuntimeError):
    pass


def load_config(root: Path) -> dict:
    return yaml.safe_load((root / "runner" / "config.yaml").read_text())


def _strategy_files(root: Path, cfg: dict) -> dict[str, str]:
    return {p: (root / p).read_text() for p in cfg["editable_paths"] if (root / p).exists()}


def _improves(new: float | None, best: float | None, cfg: dict) -> bool:
    if new is None:
        return False
    if best is None:
        return True
    delta = new - best if DIRECTION == "maximize" else best - new
    return delta > float(cfg["min_improvement"])


def apply_proposal(root: Path, proposal: Proposal, allowed: list[str]) -> list[str]:
    bad = [p for p in proposal.files if p not in allowed]
    if bad:
        raise ScopeViolation(f"agent tried to edit non-editable paths: {bad}")
    already_dirty = set(gitops.dirty_paths(root))  # your own uncommitted work is not the agent's
    for path, content in proposal.files.items():
        (root / path).write_text(content)
    touched = gitops.dirty_paths(root)
    outside = [p for p in touched
               if not p.startswith("strategy/") and p != history.LOG_NAME and p not in already_dirty]
    if outside:
        raise ScopeViolation(f"files outside strategy/ changed: {outside}")
    return sorted(proposal.files)


def _row(exp_id, cfg, agent_name, proposal, edited, diff, res, best_before, decision, failure, fhash, commit):
    return {
        "id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent": agent_name,
        "hypothesis": proposal.hypothesis,
        "rationale_from_history": proposal.rationale_from_history,
        "idea_key": proposal.idea_key,
        "edited_files": edited,
        "diff": diff,
        "seeds": cfg["seeds"],
        "budget_s": cfg["budget_s"],
        "scores": {str(k): v for k, v in (res.scores if res else {}).items()},
        "score": res.score if res else None,
        "best_before": best_before,
        "decision": decision,
        "failure_reason": failure,
        "duration_s": res.duration_s if res else 0.0,
        "frozen_hash": fhash,
        "commit": commit,
    }


def run_baseline(root: Path) -> dict:
    cfg = load_config(root)
    fhash = frozen.check(root)
    rows = history.read(root)
    res = evaluate(root, cfg["seeds"], cfg["budget_s"], cfg["aggregate"])
    p = Proposal("Baseline: score the current strategy unchanged.", "First run; no history yet.", agent="none")
    row = _row(history.next_id(rows), cfg, "none", p, [], "", res, None,
               "baseline" if res.failure_reason is None else "reject", res.failure_reason, fhash, gitops.head(root))
    history.append(root, row)
    gitops.commit(root, [history.LOG_NAME], f"{row['id']} baseline score={row['score']}")
    return row


def run_experiment(root: Path, agent=None) -> dict:
    cfg = load_config(root)
    fhash = frozen.check(root)                       # 1. frozen files untouched?
    if any(p.startswith("strategy/") for p in gitops.dirty_paths(root)):
        raise RuntimeError("strategy/ has uncommitted changes; commit or revert them before running the loop")
    rows = history.read(root)                         # 2. read history
    if not any(r["decision"] == "baseline" for r in rows):
        return run_baseline(root)
    best_row = history.best(rows, DIRECTION)
    best_before = best_row["score"] if best_row else None
    agent = agent or make_agent(cfg["agent"], root, cfg["model"], cfg["effort"])
    ctx = Context(root, rows[-cfg["history_window"]:], best_row, _strategy_files(root, cfg),
                  (root / "program.md").read_text() if (root / "program.md").exists() else "")
    exp_id = history.next_id(rows)

    proposal = agent.propose(ctx)                     # 3. hypothesis + edits
    res, edited, diff, failure = None, [], "", None
    try:
        edited = apply_proposal(root, proposal, cfg["editable_paths"])   # 4. edit strategy only
        diff = gitops.diff(root, ["strategy"])
        frozen.check(root)
        res = evaluate(root, cfg["seeds"], cfg["budget_s"], cfg["aggregate"])  # 5. run + verify
        failure = res.failure_reason
    except Exception as e:
        failure = f"{type(e).__name__}: {e}"

    keep = failure is None and _improves(res.score if res else None, best_before, cfg)  # 6. decide
    if failure is None and not keep:
        failure = f"no improvement: {res.score:.6f} vs best {best_before:.6f}"
    decision = "keep" if keep else "reject"
    if keep:
        commit = gitops.commit(root, edited, f"{exp_id} keep: {proposal.hypothesis[:60]}")
    else:
        gitops.revert(root, "strategy")
        commit = gitops.head(root)

    row = _row(exp_id, cfg, getattr(agent, "name", "?") if not proposal.agent else proposal.agent,
               proposal, edited, diff, res, best_before, decision, failure, fhash, commit)
    history.append(root, row)                         # 7. log
    gitops.commit(root, [history.LOG_NAME], f"{exp_id} log: {decision} score={row['score']}")
    return row


def status(root: Path) -> str:
    rows = history.read(root)
    if not rows:
        return "No experiments yet. Run `python -m runner.cli run`."
    from problem import KNOWN_BEST
    b = history.best(rows, DIRECTION)
    lines = [f"{'id':9} {'decision':8} {'score':>10} {'best_before':>11}  hypothesis / failure"]
    for r in rows:
        sc = "-" if r["score"] is None else f"{r['score']:.6f}"
        bb = "-" if r["best_before"] is None else f"{r['best_before']:.6f}"
        note = r["hypothesis"][:70] + ("" if not r["failure_reason"] else f"  [{r['failure_reason'][:50]}]")
        lines.append(f"{r['id']:9} {r['decision']:8} {sc:>10} {bb:>11}  {note}")
    if b:
        lines.append(f"\nBest: {b['score']:.6f} ({b['id']}), literature reference {KNOWN_BEST:.6f}, "
                     f"gap {KNOWN_BEST - b['score']:.6f}")
    return "\n".join(lines)
