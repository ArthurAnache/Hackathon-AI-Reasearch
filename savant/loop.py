"""Adaptive research loop: council → plan → (optional) Modal → stats → stop meeting.

Paid LLM/Modal require explicit confirmation flags. A session resumes from disk at the first iteration
without a valid decision; completed Modal waves and stored LLM outputs are never redone.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from savant.config import SavantConfig
from savant.debate import run_council, stop_meeting
from savant.ingest import ingest_export, validate_export
from savant.llm import BudgetExceeded, LLMClient, LLMUnavailable
from savant.memory import ResearchMemory
from savant.planner import JobPlan, ModalAdapter, ResourceLedger, allocate_wave, strategy_signature, write_plan
from savant.problem_report import write_problem_report
from savant.stats import paired_comparison, report_markdown
from savant.templates import default_hypotheses_for_mock, materialize

CONTROL_ID = "control-blockshift-stock"
CONTROL_HYPOTHESIS = {
    "hypothesis_id": CONTROL_ID, "role": "statistical", "template": "blockshift",
    "title": "Paired control: stock blockshift (Marie V2 benchmark optimizer)",
    "claim": "Reference arm on the same seeds and budget, so every strategy has a paired comparison.",
    "mechanism": "Unmodified blockshift.", "algorithmic_change": "none",
    "config_overrides": {}, "expected_gain": 0.0, "expected_gain_lo": 0.0, "expected_gain_hi": 0.0,
    "risk": 0.0, "assumptions": [], "falsification": "n/a (control)", "compute_jobs": 10, "novelty": 0.0,
    "literature_refs": [], "notes": "Added by the allocator, not by a vote.",
}
MOCK_HYPOTHESIS_IDS = frozenset(h["hypothesis_id"] for h in default_hypotheses_for_mock())
LEGACY_USD_PER_CALL = 0.01  # gpt-4.1-mini, <=12k input + 2.5k output at list price, rounded up


def init_session(cfg: SavantConfig, session_id: str | None = None) -> dict:
    session_id = session_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-savant"
    sdir = cfg.session_dir(session_id)
    sdir.mkdir(parents=True, exist_ok=True)
    memory = ResearchMemory(sdir / "research_memory.jsonl")
    ingest = ingest_export(cfg.seed_export, memory)
    report_path = write_problem_report(sdir, cfg.seed_export)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # baseline statistical report from ingested smoke test
    paired = ingest["paired_blockshift_vs_probe"]
    (sdir / "statistical_report.json").write_text(json.dumps(paired, indent=2) + "\n", encoding="utf-8")
    (sdir / "statistical_report.md").write_text(report_markdown(paired), encoding="utf-8")
    memory.append("statistical_report", {"path": "statistical_report.json", "summary": paired["interpretation"]},
                  source="savant")
    (sdir / "session.json").write_text(json.dumps({
        "session_id": session_id, "config": cfg.to_dict(), "problem_report_sha": report["content_sha256"],
        "ingest_run_id": ingest["run_id"],
    }, indent=2, default=str), encoding="utf-8")
    return {"session_id": session_id, "dir": sdir, "memory": memory, "report": report, "ingest": ingest}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
    except Exception:
        return False
    return False


def campaign_is_running(session_dir: Path) -> int | None:
    lock = Path(session_dir) / "campaign.lock"
    if not lock.exists():
        return None
    try:
        pid = int(lock.read_text().strip())
    except ValueError:
        return None
    return pid if _pid_alive(pid) else None


def open_session(cfg: SavantConfig, session_id: str) -> dict:
    """Resume an existing session without re-ingesting or re-running stored experiments."""
    sdir = cfg.session_dir(session_id)
    mem_path = sdir / "research_memory.jsonl"
    report_path = sdir / "problem_report.json"
    if mem_path.exists() and report_path.exists() and mem_path.stat().st_size > 0:
        running = campaign_is_running(sdir)
        if running:
            raise SystemExit(f"session {session_id} is already running (pid {running}); not launching a duplicate")
        old = {}
        if (sdir / "session.json").exists():
            old = json.loads((sdir / "session.json").read_text(encoding="utf-8"))
        checkpoints = sdir / "checkpoints.jsonl"
        with checkpoints.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "checkpointed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "previous_config": old.get("config"), "new_config": cfg.to_dict(),
                "note": "Resume in place. Stored hypotheses, critiques, votes and experiments are kept; "
                        "completed Modal waves are not rerun.",
            }, default=str) + "\n")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        (sdir / "session.json").write_text(json.dumps({
            "session_id": session_id, "config": cfg.to_dict(),
            "problem_report_sha": report.get("content_sha256"),
            "ingest_run_id": old.get("ingest_run_id"), "resumed": True,
            "llm": {"model": cfg.llm.model, "reasoning_effort": cfg.llm.reasoning_effort,
                    "effort_by_purpose": cfg.llm.effort_by_purpose, "provider": cfg.llm.provider},
        }, indent=2, default=str), encoding="utf-8")
        return {"session_id": session_id, "dir": sdir, "memory": ResearchMemory(mem_path),
                "report": report, "ingest": {"run_id": old.get("ingest_run_id")}, "resumed": True}
    return init_session(cfg, session_id)


# ---------------------------------------------------------------------------------------------- audit
def _is_mock_record(e: dict) -> bool:
    kind = e.get("kind")
    if e.get("hypothesis_id") in MOCK_HYPOTHESIS_IDS or e.get("idea") in MOCK_HYPOTHESIS_IDS \
            or e.get("topic") in MOCK_HYPOTHESIS_IDS or e.get("target_id") in MOCK_HYPOTHESIS_IDS:
        return True
    if kind == "critique" and e.get("objections") == ["Mock critique: verify against matched-seed evidence."]:
        return True
    if kind == "vote" and e.get("rationale") == "Mock vote prefers executable, diversified templates.":
        return True
    if kind == "stop_vote" and e.get("rationale") == "Mock stop vote.":
        return True
    if kind == "note" and e.get("kind_detail") == "revision" \
            and e.get("response") == "Acknowledged; keeping template within vetted bounds.":
        return True
    if kind == "manager_decision" and e.get("selected_ids") \
            and set(e["selected_ids"]) <= MOCK_HYPOTHESIS_IDS:
        return True
    if kind == "iteration_decision" and e.get("votes") \
            and all(v.get("rationale") == "Mock stop vote." for v in e["votes"]):
        return True
    return False


def audit_mock_outputs(session: dict) -> dict:
    """Hide mock-LLM records from a paid session (kept on disk) and move an unexecuted mock iteration aside."""
    memory: ResearchMemory = session["memory"]
    sdir = Path(session["dir"])
    bad = [e for e in memory.read() if _is_mock_record(e)]
    n = memory.invalidate(bad, "mock LLM output in a paid session (SAVANT_LLM_PROVIDER=mock was inherited); "
                               "kept for audit, excluded from research decisions")
    moved = []
    for it_dir in sorted(sdir.glob("iteration_[0-9][0-9]")):
        council = it_dir / "council.json"
        if not council.exists() or (it_dir / "modal" / "launch.json").exists():
            continue
        sel = json.loads(council.read_text(encoding="utf-8")).get("selected_ids") or []
        if sel and set(sel) <= MOCK_HYPOTHESIS_IDS:
            dest = sdir / f"void_mock_{it_dir.name}"
            shutil.move(str(it_dir), str(dest))
            moved.append(dest.name)
    return {"invalidated_records": n, "moved_iterations": moved}


def _valid_decisions(sdir: Path) -> dict[int, dict]:
    path = Path(sdir) / "iteration_decisions.jsonl"
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("votes") and all(v.get("rationale") == "Mock stop vote." for v in d["votes"]):
            continue
        out[int(d["iteration"])] = d
    return out


def iteration_state(sdir: Path, it: int) -> str:
    it_dir = Path(sdir) / f"iteration_{it:02d}"
    if it in _valid_decisions(sdir):
        return "decided"
    if (it_dir / "modal" / "export" / "comparison.csv").exists():
        return "executed"
    if (it_dir / "modal" / "launch.json").exists():
        return "dispatched_without_export"
    return "open"


# ---------------------------------------------------------------------------------------------- ledgers
def rebuild_modal_ledger(sdir: Path) -> ResourceLedger:
    """Modal commitments read from disk: every wave whose launch reached Modal is counted at its estimate."""
    led = ResourceLedger()
    for it_dir in sorted(Path(sdir).glob("iteration_[0-9][0-9]")):
        plan_path = it_dir / "job_plan.json"
        if not plan_path.exists() or not (it_dir / "modal" / "launch.json").exists():
            continue
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        led.jobs_used += int(plan.get("n_jobs") or 0)
        led.search_seconds_used += int(plan.get("n_jobs") or 0) * float(plan.get("budget_s") or 0)
        led.modal_usd_est += float(plan.get("estimated_modal_usd") or 0.0)
    led.iterations_done = len(_valid_decisions(sdir))
    return led


def book_legacy_llm(client: LLMClient, memory: ResearchMemory) -> dict | None:
    """Count paid calls made before the spend ledger existed (records without llm metadata)."""
    if any(e.get("kind") == "legacy_estimate" for e in client.prior["entries"]):
        return None
    calls = 0
    for e in memory.read_all():
        if "llm" in e or _is_mock_record(e):
            continue
        kind = e.get("kind")
        if kind == "hypothesis" and e.get("stage") == "independent" and str(e.get("source", "")).startswith("savant:"):
            calls += 1
        elif kind in ("critique", "vote", "stop_vote"):
            calls += 1
        elif kind == "note" and e.get("kind_detail") == "revision":
            calls += 1
    if calls == 0:
        return None
    calls += 1  # one request was interrupted (KeyboardInterrupt) before it was stored
    entry = {"model": "gpt-4.1-mini", "reasoning_effort": None, "calls": calls,
             "usd": round(calls * LEGACY_USD_PER_CALL, 4), "input_tokens": 0, "output_tokens": 0,
             "note": f"{calls} earlier gpt-4.1-mini requests booked at ${LEGACY_USD_PER_CALL}/request (upper bound)"}
    client.record_legacy(entry)
    return entry


# ---------------------------------------------------------------------------------------------- evidence
def _sanitize(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(label))[:48]


def wave_rows(wave_dir: Path) -> list[dict]:
    path = Path(wave_dir) / "export" / "comparison.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x: Any) -> float | None:
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def smoke_gains(memory: ResearchMemory) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for e in memory.by_kind("experiment"):
        arm = e.get("arm") or ""
        if arm.startswith("ws3-bench:") and e.get("seed") is not None and e.get("gain") is not None:
            out.setdefault(arm.split(":", 1)[1], {})[int(e["seed"])] = float(e["gain"])
    return out


def collect_waves(sdir: Path) -> list[dict]:
    waves = []
    for it_dir in sorted(Path(sdir).glob("iteration_[0-9][0-9]")):
        rows = wave_rows(it_dir / "modal")
        plan_path = it_dir / "job_plan.json"
        if not rows or not plan_path.exists():
            continue
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        strategies = {}
        for s in plan.get("strategies", []):
            strategies[_sanitize(s.get("hypothesis_id"))] = s
        arms: dict[str, dict] = {}
        for r in rows:
            label = r["optimizer"]
            arm = arms.setdefault(label, {"gains": {}, "scores": {}, "status": {}, "cpu_user_s": 0.0})
            seed = int(r["seed"])
            arm["status"][seed] = r.get("status")
            if r.get("valid") in ("True", "true", "1") and _f(r.get("gain")) is not None:
                arm["gains"][seed] = _f(r["gain"])
                arm["scores"][seed] = _f(r.get("final_score"))
            arm["cpu_user_s"] += _f(r.get("cpu_user_s")) or 0.0
        waves.append({"iteration": int(it_dir.name.split("_")[1]), "budget_s": float(plan.get("budget_s") or 60),
                      "plan": plan, "strategies": strategies, "arms": arms, "dir": str(it_dir)})
    return waves


def wave_statistics(waves: list[dict], smoke: dict[str, dict[int, float]], *, alpha_spend: float) -> list[dict]:
    """Paired tests per strategy: vs the stock-blockshift control at the same budget (matched seeds), and
    vs Marie's smoke benchmark (blockshift, seeds 101/202/303, 60 s)."""
    controls: dict[float, dict[int, float]] = {}
    for w in waves:
        ctl = w["arms"].get(CONTROL_ID)
        if ctl:
            controls.setdefault(w["budget_s"], {}).update(ctl["gains"])
    out = []
    look = 0
    for w in waves:
        for label, arm in w["arms"].items():
            if label == CONTROL_ID:
                continue
            row = {"iteration": w["iteration"], "label": label, "budget_s": w["budget_s"],
                   "n_valid": len(arm["gains"]),
                   "mean_gain": (sum(arm["gains"].values()) / len(arm["gains"])) if arm["gains"] else None,
                   "best_score": max([s for s in arm["scores"].values() if s is not None], default=None)}
            ctl = controls.get(w["budget_s"], {})
            seeds = sorted(set(arm["gains"]) & set(ctl))
            if len(seeds) >= 2:
                look += 1
                row["vs_control"] = paired_comparison(
                    [arm["gains"][s] for s in seeds], [ctl[s] for s in seeds], label_a=label,
                    label_b=CONTROL_ID, seeds=seeds, decision_index=look, alpha_spend=alpha_spend)
            base = smoke.get("blockshift", {})
            seeds_b = sorted(set(arm["gains"]) & set(base))
            if seeds_b and w["budget_s"] == 60.0:
                row["vs_marie_smoke_blockshift"] = paired_comparison(
                    [arm["gains"][s] for s in seeds_b], [base[s] for s in seeds_b], label_a=label,
                    label_b="marie_smoke_blockshift_60s", seeds=seeds_b, decision_index=0, alpha_spend=alpha_spend)
            out.append(row)
    return out


def _brief(cmp: dict | None) -> dict | None:
    if not cmp:
        return None
    return {k: cmp.get(k) for k in ("n", "mean_diff", "se", "sign_positive", "p_sign_flip_one_sided",
                                    "alpha_look", "claim_allowed", "practical_vs_record_delta")}


def build_evidence(session: dict, cfg: SavantConfig) -> dict:
    sdir = Path(session["dir"])
    memory: ResearchMemory = session["memory"]
    mf = session["report"].get("measured_facts", {})
    dist = mf.get("distance_to_record_threshold") or {}
    waves = collect_waves(sdir)
    smoke = smoke_gains(memory)
    stats = wave_statistics(waves, smoke, alpha_spend=cfg.budgets.alpha_spend_per_decision)
    best = None
    for w in waves:
        for label, arm in w["arms"].items():
            for seed, score in arm["scores"].items():
                if score is not None and (best is None or score > best["score"]):
                    best = {"score": score, "label": label, "seed": seed, "iteration": w["iteration"]}
    target = dist.get("target_score_at_freeze")
    completed = []
    for w in waves:
        for label, s in w["strategies"].items():
            st = next((x for x in stats if x["iteration"] == w["iteration"] and x["label"] == label), {})
            completed.append({"hypothesis_id": s.get("hypothesis_id"), "template": s.get("template"),
                              "optimizer": s.get("optimizer"), "config": s.get("config"),
                              "budget_s": w["budget_s"], "iteration": w["iteration"],
                              "mean_gain": st.get("mean_gain"), "n_valid": st.get("n_valid"),
                              "vs_control": _brief(st.get("vs_control")),
                              "vs_marie_smoke": _brief(st.get("vs_marie_smoke_blockshift"))})
    queued = [e.get("idea") for e in memory.by_kind("rejected_idea") if e.get("queued")]
    decisions = _valid_decisions(sdir)
    last = None
    if decisions:
        d = decisions[max(decisions)]
        last = {"iteration": max(decisions), "decision": d.get("decision"),
                "rationales": {v.get("role"): v.get("rationale") for v in d.get("votes") or []}}
    return {
        "last_council_decision": last,
        "baseline_score": mf.get("verified_baseline_score"),
        "record_target_score": target,
        "delta": dist.get("delta"),
        "best_verified": best,
        "best_distance_to_target": (target - best["score"]) if (best and target is not None) else None,
        "marie_smoke_gains_60s": {k: {str(s): g for s, g in v.items()} for k, v in smoke.items()},
        "completed_strategies": completed,
        "queued_novel_ideas": sorted({q for q in queued if q})[:20],
        "statistics": [{"iteration": s["iteration"], "label": s["label"], "budget_s": s["budget_s"],
                        "mean_gain": s["mean_gain"], "vs_control": _brief(s.get("vs_control")),
                        "vs_marie_smoke": _brief(s.get("vs_marie_smoke_blockshift"))} for s in stats],
        "note": "Gains are improvements over the WS1 frozen certificate, re-verified by the frozen verifier. "
                "Within-job evaluations are not independent samples; seeds are the unit.",
    }


# ---------------------------------------------------------------------------------------------- iteration
def plan_iteration(cfg: SavantConfig, session: dict, *, iteration: int, client: LLMClient,
                   ledger: ResourceLedger, evidence: dict | None = None) -> dict:
    sdir = Path(session["dir"])
    it_dir = sdir / f"iteration_{iteration:02d}"
    it_dir.mkdir(parents=True, exist_ok=True)
    caps = cfg.budgets
    manager_context = None
    if client.cfg.provider != "mock":
        manager_context = {
            "iteration": iteration, "max_strategies": caps.max_strategies_per_iteration,
            "evidence": evidence or {},
            "remaining": ledger.remaining(caps),
            "llm_usd_remaining": round(client.remaining_usd(), 2),
            "control_arm_added_automatically": CONTROL_ID,
        }
    council = run_council(session["report"], session["memory"], client,
                          voting_seed=cfg.voting_seed + iteration,
                          max_select=caps.max_strategies_per_iteration,
                          evidence=evidence, batch_critiques=client.cfg.is_reasoning,
                          manager_context=manager_context)
    (it_dir / "council.json").write_text(json.dumps({
        "hypotheses": council["hypotheses"], "ranking": [
            {"id": r["hypothesis_id"], "score": r["score"]} for r in council["ranking"]],
        "selected_ids": [h["hypothesis_id"] for h in council["selected"]],
        "materialized": council["materialized"],
        "manager_review": council.get("manager_review"),
        "llm": {"model": client.cfg.model, "reasoning_effort": client.cfg.reasoning_effort,
                "provider": client.cfg.provider},
        "llm_usage": client.usage.__dict__,
    }, indent=2, default=str), encoding="utf-8")
    review = council.get("manager_review") or {}
    waves = collect_waves(sdir)
    control_budgets = {w["budget_s"] for w in waves if CONTROL_ID in w["arms"]}
    plan = allocate_wave(council["selected"], ledger, caps, cfg.campaign_seeds,
                         iteration=iteration, completed=_completed_signatures(session["memory"], evidence),
                         prior_best_gains=_prior_gains(session["dir"]),
                         n_seeds=review.get("seeds_per_strategy"), budget_request=review.get("budget_s_request"),
                         control=CONTROL_HYPOTHESIS, control_done_budgets=control_budgets)
    write_plan(plan, it_dir)
    return {"iteration": iteration, "dir": it_dir, "council": council, "plan": plan}


def _record_wave(memory: ResearchMemory, wave_dir: Path, plan: JobPlan) -> None:
    """Persist verified per-job gains so the next wave can skip unproductive repeats."""
    rows = wave_rows(wave_dir)
    if not rows:
        return
    by_label: dict[str, list] = {}
    for row in rows:
        by_label.setdefault(row["optimizer"], []).append(row)
    best = None
    for strat in plan.strategies:
        label = _sanitize(strat.get("hypothesis_id") or "")
        group = by_label.get(label) or []
        gains = [float(r["gain"]) for r in group if r.get("gain")]
        mean = sum(gains) / len(gains) if gains else None
        memory.append("experiment", {
            "optimizer": strat.get("optimizer"),
            "config": strat.get("config"),
            "hypothesis_id": label,
            "gain": mean,
            "n_seeds": len(gains),
            "budget_s": plan.budget_s,
            "iteration": plan.iteration,
        }, source="savant-modal")
        if label != CONTROL_ID and mean is not None and (best is None or mean > best):
            best = mean
    if best is not None:
        with (Path(wave_dir).parent.parent / "iteration_gains.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"iteration": plan.iteration, "best_gain": best}) + "\n")


def _completed_signatures(memory: ResearchMemory, evidence: dict | None = None) -> dict:
    """Signatures already run by SAVANT. Gains are taken in excess of the paired control when one exists."""
    ctl = {}
    for s in (evidence or {}).get("statistics", []):
        if s.get("vs_control"):
            ctl[(s["iteration"], s["label"])] = s["vs_control"]["mean_diff"]
    out: dict = {}
    for row in memory.read():
        if row.get("kind") != "experiment" or row.get("source") != "savant-modal":
            continue
        if row.get("hypothesis_id") == CONTROL_ID:
            continue
        sig = (row.get("optimizer"), tuple(sorted((row.get("config") or {}).items())))
        slot = out.setdefault(sig, {"runs": 0, "best_gain": None, "first_gain": None})
        gain = ctl.get((row.get("iteration"), row.get("hypothesis_id")), row.get("gain"))
        slot["runs"] += 1
        if slot["first_gain"] is None:
            slot["first_gain"] = gain
        if gain is not None and (slot["best_gain"] is None or gain > slot["best_gain"]):
            slot["best_gain"] = gain
    return out


def _prior_gains(session_dir: Path) -> list[float]:
    path = Path(session_dir) / "iteration_gains.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["best_gain"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze_batch_vs_baseline(export_dir: Path, baseline_gains_by_seed: dict[int, float],
                              *, label_a: str, decision_index: int, alpha_spend: float) -> dict:
    rep = validate_export(export_dir) if (Path(export_dir) / "comparison.csv").exists() else None
    if rep is None:
        raise FileNotFoundError(export_dir)
    rows = rep["rows"]
    opts = sorted({r["optimizer"] for r in rows})
    cand = "blockshift" if "blockshift" in opts else opts[-1]
    gains_a, seeds, gains_b = [], [], []
    for r in rows:
        if r["optimizer"] != cand:
            continue
        s = int(r["seed"])
        if s not in baseline_gains_by_seed:
            continue
        seeds.append(s)
        gains_a.append(float(r["gain"]))
        gains_b.append(baseline_gains_by_seed[s])
    return paired_comparison(gains_a, gains_b, label_a=label_a, label_b="baseline_probe_smoke",
                             seeds=seeds, decision_index=decision_index, alpha_spend=alpha_spend)


def hold_stop_meeting(cfg: SavantConfig, session: dict, client: LLMClient, ledger: ResourceLedger, *,
                      iteration: int, max_iterations: int, evidence: dict, plan: dict | None,
                      reheld: bool = False) -> dict:
    it_stats = [s for s in evidence.get("statistics", []) if s["iteration"] == iteration]
    uncertainty_high = not any((s.get("vs_control") or {}).get("claim_allowed") for s in it_stats)
    context = {
        "iteration": iteration,
        "max_iterations": max_iterations,
        "plan": plan,
        "remaining": ledger.remaining(cfg.budgets),
        "llm_usd_remaining": round(client.remaining_usd(), 2),
        "modal_usd_remaining": round(cfg.budgets.max_modal_usd - ledger.modal_usd_est, 2),
        "candidate_ids": [s["label"] for s in it_stats],
        "uncertainty_high": uncertainty_high,
        "evidence": evidence,
        "this_iteration_statistics": it_stats,
        "preregistered_stop_rules": [
            "max iterations/budget", "no credible new executable hypotheses",
            "repeated stagnation vs paired control", "invalid candidates",
            "verified research goal (gain>=delta over the record)",
        ],
    }
    decision = stop_meeting(context, session["memory"], client)
    decision["reheld"] = reheld
    it_dir = Path(session["dir"]) / f"iteration_{iteration:02d}"
    it_dir.mkdir(parents=True, exist_ok=True)
    (it_dir / "stop_decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    with (Path(session["dir"]) / "iteration_decisions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"iteration": iteration, **decision}, default=str) + "\n")
    return decision


def run_loop(cfg: SavantConfig, *, max_iterations: int = 10, confirm_llm: bool = False,
             confirm_modal: bool = False, mock: bool = True, session_id: str | None = None) -> dict:
    """Adaptive loop with resume. Default: mock LLM, no Modal. Stops early via the stop meeting or budget."""
    paid_llm = confirm_llm and not mock
    if paid_llm and cfg.llm.provider != "openai":
        raise SystemExit(f"--confirm-llm was given but the LLM provider resolves to {cfg.llm.provider!r} "
                         "(check SAVANT_LLM_PROVIDER); refusing to run a paid campaign on a mock model")
    if not paid_llm:
        cfg.llm.provider = "mock"

    session = open_session(cfg, session_id or "savant-exp1")
    sdir = Path(session["dir"])
    lock = sdir / "campaign.lock"
    lock.write_text(str(os.getpid()), encoding="utf-8")
    client = LLMClient(cfg.llm, max_usd=cfg.budgets.max_llm_usd, max_tokens=cfg.budgets.max_llm_tokens,
                       allow_mock=not paid_llm, ledger_path=sdir / "spend_ledger.json")
    if paid_llm and client.cfg.provider != "openai":
        raise SystemExit("paid LLM requested but the client fell back to mock; refusing")
    history: list[dict] = []
    stop_reason = None
    try:
        if paid_llm:
            audit = audit_mock_outputs(session)
            legacy = book_legacy_llm(client, session["memory"])
            session["memory"].append("note", {"kind_detail": "resume", "audit": audit, "legacy_llm": legacy,
                                              "llm": {"model": client.cfg.model,
                                                      "reasoning_effort": client.cfg.reasoning_effort,
                                                      "effort_by_purpose": client.cfg.effort_by_purpose,
                                                      "max_llm_usd": cfg.budgets.max_llm_usd,
                                                      "max_modal_usd": cfg.budgets.max_modal_usd}},
                                     source="savant:resume")
        ledger = rebuild_modal_ledger(sdir)
        max_iterations = min(max_iterations, cfg.budgets.max_iterations)
        it = 1
        while it <= max_iterations and iteration_state(sdir, it) == "decided":
            it += 1
        decisions = _valid_decisions(sdir)
        if decisions and decisions[max(decisions)].get("decision") == "STOP":
            stop_reason = f"iteration {max(decisions)} already decided STOP"
        pauses = 0
        while stop_reason is None and it <= max_iterations:
            try:
                it, stop_reason = _run_one(cfg, session, client, ledger, history, it, max_iterations,
                                           confirm_modal)
            except LLMUnavailable as exc:
                pauses += 1
                session["memory"].append("note", {"kind_detail": "pause", "iteration": it, "error": str(exc)[:300],
                                                  "pause": pauses}, source="savant:llm")
                if pauses > 3 or not client.wait_for_api(900):
                    stop_reason = f"paused: OpenAI unreachable ({exc}); resume later with the same command"
                    break
                # API back: re-enter the same iteration; stored council outputs are reused, not re-requested.
        if stop_reason is None and it > max_iterations:
            stop_reason = "max iterations reached"
    except BudgetExceeded as exc:
        stop_reason = f"budget: {exc}"
        session["memory"].append("iteration_decision", {"decision": "STOP", "additional_jobs": 0,
                                                        "reason": stop_reason}, source="savant:budget")
    finally:
        ledger_final = rebuild_modal_ledger(sdir)
        (sdir / "loop_summary.json").write_text(json.dumps({
            "session_id": session["session_id"], "history": history, "stop_reason": stop_reason,
            "modal_ledger": ledger_final.__dict__, "llm_spent_usd": client.spent_usd(),
            "llm_usage_this_run": client.usage.__dict__,
            "llm": {"model": client.cfg.model, "reasoning_effort": client.cfg.reasoning_effort},
        }, indent=2, default=str), encoding="utf-8")
        lock.unlink(missing_ok=True)
    return {"session": session, "history": history, "ledger": ledger_final, "client": client,
            "stop_reason": stop_reason}


def _run_one(cfg: SavantConfig, session: dict, client: LLMClient, ledger: ResourceLedger, history: list,
             it: int, max_iterations: int, confirm_modal: bool) -> tuple[int, str | None]:
    """One iteration from its current on-disk state. Returns (next iteration, stop reason or None)."""
    sdir = Path(session["dir"])
    stop_reason = None
    while True:  # single pass; `break` exits with stop_reason set
            state = iteration_state(sdir, it)
            if state == "dispatched_without_export":
                stop_reason = (f"iteration {it} reached Modal but has no export; recover it with ws3_bench collect/"
                               "export before resuming (never re-dispatched automatically)")
                break
            plan_dict = None
            if state == "open":
                evidence = build_evidence(session, cfg)
                step = plan_iteration(cfg, session, iteration=it, client=client, ledger=ledger, evidence=evidence)
                plan: JobPlan = step["plan"]
                plan_dict = plan.to_dict()
                if not plan.afford and not step["council"]["hypotheses"]:
                    # Infrastructure, not science: no proposal came back. Leave the iteration open.
                    stop_reason = "paused: the council produced no hypotheses (see llm_call_failed notes)"
                    history.append({"iteration": it, "plan": plan_dict, "execution": {"status": "paused"}})
                    break
                if not plan.afford:
                    session["memory"].append("iteration_decision", {
                        "decision": "STOP", "additional_jobs": 0, "reason": plan.afford_reason,
                        "estimated_modal_usd": plan.estimated_modal_usd}, source="savant:budget")
                    history.append({"iteration": it, "plan": plan_dict,
                                    "execution": {"status": "refused", "reason": plan.afford_reason}})
                    stop_reason = plan.afford_reason
                    break
                # Commit the estimate before dispatch so a crash cannot start another wave past the cap.
                ledger.modal_usd_est += plan.estimated_modal_usd
                ledger.jobs_used += plan.n_jobs
                ledger.search_seconds_used += plan.n_jobs * plan.budget_s
                if ledger.modal_usd_est > cfg.budgets.max_modal_usd + 1e-6:
                    stop_reason = "modal usd hard stop"
                    break
                wave_dir = sdir / f"iteration_{it:02d}" / "modal"
                exec_result = ModalAdapter(dry_run=not confirm_modal).execute(plan, wave_dir,
                                                                             confirm_modal=confirm_modal)
                (step["dir"] / "execution.json").write_text(json.dumps(exec_result, indent=2, default=str),
                                                            encoding="utf-8")
                history.append({"iteration": it, "plan": plan_dict, "execution": {
                    k: exec_result.get(k) for k in ("status", "n_jobs", "modal_app_id", "wall_s", "run_dir")}})
                if exec_result.get("status") == "executed":
                    _record_wave(session["memory"], wave_dir, plan)
                elif exec_result.get("status") != "skipped_completed":
                    if confirm_modal:
                        stop_reason = f"wave not executed: {exec_result.get('status')} {exec_result.get('error', '')}"
                        break
            else:  # executed but not validly decided: hold (or re-hold) the meeting on the stored results
                jp = sdir / f"iteration_{it:02d}" / "job_plan.json"
                plan_dict = json.loads(jp.read_text(encoding="utf-8")) if jp.exists() else None
            ledger.iterations_done = it
            evidence = build_evidence(session, cfg)
            decision = hold_stop_meeting(cfg, session, client, ledger, iteration=it,
                                         max_iterations=max_iterations, evidence=evidence, plan=plan_dict,
                                         reheld=(state == "executed"))
            history.append({"iteration": it, "decision": decision.get("decision"), "reheld": state == "executed"})
            if decision["decision"] == "STOP":
                stop_reason = decision.get("reason") or "council voted STOP"
                break
            return it + 1, None
    return it, stop_reason


def offline_smoke_council(cfg: SavantConfig) -> dict:
    """Deterministic offline path: ingest + problem report + mock council + plan. No paid calls."""
    cfg.llm.provider = "mock"
    client = LLMClient(cfg.llm, allow_mock=True)
    session = init_session(cfg, session_id="offline-smoke")
    ledger = ResourceLedger()
    step = plan_iteration(cfg, session, iteration=1, client=client, ledger=ledger)
    return {"session": session, "step": step, "usage": client.usage.__dict__}
