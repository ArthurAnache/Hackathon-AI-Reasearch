"""Modal job planning with hard safety caps. Reuses ws3_bench; no auto-retry."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from savant.config import CAMPAIGN_SEEDS, BudgetCaps, PROBLEM, estimate_modal_usd
from savant.templates import materialize


@dataclass
class ResourceLedger:
    jobs_used: int = 0
    search_seconds_used: float = 0.0
    llm_usd_used: float = 0.0
    llm_tokens_used: int = 0
    iterations_done: int = 0
    modal_usd_est: float = 0.0

    def remaining(self, caps: BudgetCaps) -> dict:
        return {
            "jobs": caps.max_jobs_total - self.jobs_used,
            "search_seconds": caps.max_search_seconds_total - self.search_seconds_used,
            "iterations": caps.max_iterations - self.iterations_done,
            "llm_usd": caps.max_llm_usd - self.llm_usd_used,
            "llm_tokens": caps.max_llm_tokens - self.llm_tokens_used,
            "modal_usd_cap": caps.max_modal_usd - self.modal_usd_est,
        }

    def can_afford(self, n_jobs: int, budget_s: float, caps: BudgetCaps) -> tuple[bool, str]:
        rem = self.remaining(caps)
        if n_jobs <= 0:
            return False, "n_jobs=0"
        if n_jobs > caps.max_jobs_per_iteration:
            return False, f">{n_jobs} > max_jobs_per_iteration={caps.max_jobs_per_iteration}"
        if rem["jobs"] < n_jobs:
            return False, f"global jobs remaining {rem['jobs']} < {n_jobs}"
        if rem["search_seconds"] < n_jobs * budget_s:
            return False, "search-second cap exceeded"
        if rem["iterations"] <= 0:
            return False, "max iterations reached"
        est = estimate_modal_usd(n_jobs, budget_s, caps)
        if self.modal_usd_est + est > caps.max_modal_usd + 1e-9:
            return False, f"modal usd cap: committed {self.modal_usd_est:.2f} + {est:.2f} > {caps.max_modal_usd:.2f}"
        if self.llm_usd_used > caps.max_llm_usd + 1e-9:
            return False, f"llm usd cap: {self.llm_usd_used:.2f} > {caps.max_llm_usd:.2f}"
        return True, "ok"


@dataclass
class JobPlan:
    plan_id: str
    problem: str
    iteration: int
    strategies: list[dict]
    seeds: list[int]
    budget_s: float
    n_jobs: int
    ledger_before: dict
    ledger_after_if_run: dict
    afford: bool
    afford_reason: str
    estimated_modal_usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def strategy_signature(mat: dict) -> tuple:
    cfg = tuple(sorted((k, v) for k, v in (mat.get("config") or {}).items()))
    return (mat.get("optimizer"), cfg)


def fit_wave(n_strategies: int, seeds: list[int], budget_s: float, ledger: ResourceLedger,
             caps: BudgetCaps) -> tuple[int, list[int], float, str]:
    """Shrink a wave until it fits jobs, CPU-seconds and the $50 Modal cap. Never exceeds them."""
    seeds = list(seeds)
    full = list(seeds)
    note = "ok"
    while n_strategies > 0 and seeds:
        n_jobs = n_strategies * len(seeds)
        ok, reason = ledger.can_afford(n_jobs, budget_s, caps)
        if ok:
            return n_strategies, seeds, budget_s, note
        note = reason
        if budget_s > caps.budget_s_per_job:
            budget_s = caps.budget_s_per_job
            continue
        if len(seeds) > 3:
            seeds = seeds[:-1]
            continue
        if n_strategies > 1:
            n_strategies -= 1
            seeds = list(full)
            continue
        return 0, [], budget_s, reason
    return 0, [], budget_s, note or "empty wave"


def build_job_plan(selected_hypotheses: list[dict], ledger: ResourceLedger, caps: BudgetCaps,
                   *, iteration: int, seeds: tuple[int, ...] | None = None,
                   budget_s: float | None = None,
                   force_n_strategies: int | None = None,
                   notes: list[str] | None = None) -> JobPlan:
    seed_pool = list(seeds or CAMPAIGN_SEEDS[: caps.seeds_per_strategy])
    budget_s = caps.budget_s_per_job if budget_s is None else budget_s
    mats = []
    for h in selected_hypotheses:
        m = materialize(h)
        if m.get("executable"):
            mats.append(m)
    if force_n_strategies is not None:
        mats = mats[:force_n_strategies]
    mats = mats[: caps.max_strategies_per_iteration]
    n_keep, seed_pool, budget_s, fit_note = fit_wave(len(mats), seed_pool, budget_s, ledger, caps)
    notes = list(notes or [])
    if n_keep < len(mats):
        notes.append(f"trimmed strategies {len(mats)} -> {n_keep} ({fit_note})")
        mats = mats[:n_keep]
    n_jobs = len(mats) * len(seed_pool)
    est = estimate_modal_usd(n_jobs, budget_s, caps) if n_jobs else 0.0
    ok, reason = ledger.can_afford(n_jobs, budget_s, caps) if n_jobs else (False, "no executable strategies")
    after = ResourceLedger(**{**ledger.__dict__})
    after.jobs_used += n_jobs
    after.search_seconds_used += n_jobs * budget_s
    after.iterations_done += 1
    after.modal_usd_est += est
    return JobPlan(
        plan_id=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-it{iteration}",
        problem=PROBLEM,
        iteration=iteration,
        strategies=mats,
        seeds=seed_pool,
        budget_s=budget_s,
        n_jobs=n_jobs,
        ledger_before=ledger.remaining(caps),
        ledger_after_if_run=after.remaining(caps),
        afford=ok and n_jobs > 0,
        afford_reason=reason if n_jobs else "no executable strategies",
        estimated_modal_usd=round(est, 4),
        notes=notes,
    )


def allocate_wave(selected: list[dict], ledger: ResourceLedger, caps: BudgetCaps, seed_pool: tuple[int, ...],
                  *, iteration: int, completed: dict | None = None,
                  prior_best_gains: list[float] | None = None,
                  n_seeds: int | None = None, budget_request: float | None = None,
                  control: dict | None = None, control_done_budgets: set | None = None) -> JobPlan:
    """Pick a wave from the council shortlist. Skip strategies that already failed to improve.

    A paired control arm is added once per budget level so every strategy has matched-seed evidence.
    The manager may ask for fewer seeds (floor 3) or 300 s; 300 s is granted only by the evidence rule.
    """
    completed = completed or {}
    prior_best_gains = list(prior_best_gains or [])
    notes = []
    fresh = []
    if n_seeds is not None:
        k = max(3, min(int(n_seeds), caps.seeds_per_strategy, len(seed_pool)))
        if k < len(seed_pool):
            notes.append(f"manager asked for {k} paired seeds per strategy")
        seed_pool = tuple(seed_pool[:k])
    for h in selected:
        mat = materialize(h)
        if not mat.get("executable"):
            notes.append(f"skip {h.get('hypothesis_id')}: {mat.get('reason')}")
            continue
        prev = completed.get(strategy_signature(mat))
        if prev and prev.get("runs", 0) >= 1 and prev.get("best_gain", 0) <= 0:
            notes.append(f"skip unproductive repeat {h.get('hypothesis_id')} (best gain {prev.get('best_gain')})")
            continue
        if prev and prev.get("runs", 0) >= 2 and prev.get("best_gain", 0) <= prev.get("first_gain", 0):
            notes.append(f"skip diminishing repeat {h.get('hypothesis_id')}")
            continue
        fresh.append(h)
    budget = caps.budget_s_per_job
    if (len(prior_best_gains) >= 2 and prior_best_gains[-1] > prior_best_gains[-2] > 0):
        if budget_request is None or float(budget_request) >= caps.budget_s_expanded:
            budget = caps.budget_s_expanded
            notes.append("budget expanded to 300s: verified gain increased over the previous iteration")
        else:
            notes.append("300s allowed by the evidence rule but the manager kept 60s")
    elif budget_request is not None and float(budget_request) > caps.budget_s_per_job:
        notes.append("manager asked for 300s; refused (no verified increase over two iterations)")
    if not fresh:
        notes.append("no new productive strategy; wave empty (stop or pivot, do not repeat)")
    elif control is not None and budget not in (control_done_budgets or set()):
        fresh = [control] + fresh[: caps.max_strategies_per_iteration - 1]
        notes.append(f"paired control {control.get('hypothesis_id')} added for budget {budget:.0f}s")
    return build_job_plan(fresh, ledger, caps, iteration=iteration, seeds=seed_pool,
                          budget_s=budget, notes=notes)


def write_plan(plan: JobPlan, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "job_plan.json"
    path.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


class ModalAdapter:
    """Thin adapter over ws3_bench.campaign. Real runs require --confirm-modal."""

    def __init__(self, *, dry_run: bool = True):
        self.dry_run = dry_run

    def execute(self, plan: JobPlan, run_dir: Path, *, confirm_modal: bool = False) -> dict:
        if self.dry_run or not confirm_modal:
            return {
                "status": "planned_only",
                "message": "Modal not launched (dry_run or missing --confirm-modal)",
                "plan": plan.to_dict(),
                "run_dir": str(run_dir),
            }
        # One interleaved Modal map for the whole wave. Independent jobs run concurrently
        # (max_containers on the function). No per-strategy loop and no automatic retry.
        import sys
        from pathlib import Path as P
        root = P(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from ws3_bench import campaign as C
        from ws3_bench.modal_app import MAX_CONTAINERS

        if plan.n_jobs > MAX_CONTAINERS:
            return {"status": "refused", "error": f"plan has {plan.n_jobs} jobs but MAX_CONTAINERS={MAX_CONTAINERS}",
                    "plan": plan.to_dict()}
        run_dir.mkdir(parents=True, exist_ok=True)
        export_dir = run_dir / "export" / "experiments.jsonl"
        if export_dir.exists():
            return {"status": "skipped_completed", "run_dir": str(run_dir),
                    "message": "export already exists; not rerunning these jobs"}
        spec = batch_spec(plan)
        (run_dir / "batch_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        run_id = plan.plan_id
        built, jobs, _problem = C.build_plan(
            plan.problem, spec["optimizers"], plan.seeds, plan.budget_s, run_id,
            live_record=False, run_dir=run_dir, configs=spec["configs"], labels=spec["labels"])
        (run_dir / "plan.json").write_text(json.dumps(built, indent=1, default=str), encoding="utf-8")
        if len(jobs) != plan.n_jobs:
            return {"status": "refused", "error": f"built {len(jobs)} jobs, plan asked {plan.n_jobs}",
                    "plan": plan.to_dict()}
        if len({j.job_id for j in jobs}) != len(jobs):
            return {"status": "refused", "error": "job ids are not unique; refusing to dispatch",
                    "plan": plan.to_dict()}
        launch = C.execute_jobs(jobs, "modal", run_dir)
        downloaded = C.download_from_volume(run_id, run_dir)
        out = C.export(run_dir, exact=True, live_record=False)
        return {"status": "executed", "n_jobs": len(jobs), "concurrent": True,
                "max_containers": MAX_CONTAINERS, "run_dir": str(run_dir), "run_id": run_id,
                "downloaded": len(downloaded), "wall_s": launch.get("wall_s"),
                "export": out.get("export"), "modal_app_id": launch.get("modal_app_id"),
                "results": launch.get("results")}


def batch_spec(plan: JobPlan) -> dict:
    """One Modal batch: every strategy and every seed, dispatched together."""
    return {
        "n_calls": 1,
        "n_jobs": plan.n_jobs,
        "concurrent": True,
        "optimizers": [s["optimizer"] for s in plan.strategies],
        "configs": [s.get("config") or {} for s in plan.strategies],
        "labels": [s.get("hypothesis_id") or f"s{i}" for i, s in enumerate(plan.strategies)],
        "seeds": list(plan.seeds),
        "budget_s": plan.budget_s,
    }
