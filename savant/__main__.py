"""CLI: python -m savant analyze|plan|run|report"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from savant.ab_eval import write_ab_protocol
from savant.config import SavantConfig, env_llm_config
from savant.loop import init_session, offline_smoke_council, plan_iteration, run_loop
from savant.llm import LLMClient
from savant.planner import ResourceLedger
from savant.report import write_session_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m savant")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze", help="Ingest Modal export + deterministic problem_report (no LLM)")
    p.add_argument("--session-id", default=None)
    p.add_argument("--export", default=None, help="WS3 export dir (default: validated smoke run)")

    p = sub.add_parser("plan", help="analyze + mock/real council + job plan (LLM mock by default)")
    p.add_argument("--session-id", default="plan")
    p.add_argument("--confirm-llm", action="store_true", help="allow paid LLM calls")
    p.add_argument("--reuse-council", action="store_true",
                   help="rank the council already stored in this session; no LLM calls")
    p.add_argument("--export", default=None)

    p = sub.add_parser("run", help="adaptive loop (requires confirmations for paid work)")
    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument("--session-id", default=None)
    p.add_argument("--confirm-llm", action="store_true")
    p.add_argument("--confirm-modal", action="store_true")
    p.add_argument("--mock", action="store_true", default=False,
                   help="force mock LLM (default unless --confirm-llm)")
    p.add_argument("--astra", action="store_true",
                   help="gpt-6-astra for every council request (no fallback): effort max, xhigh for "
                        "critiques, revisions and selection votes")
    p.add_argument("--uniform-max-effort", action="store_true",
                   help="with --astra: effort max for every stage")
    p.add_argument("--max-llm-usd", type=float, default=None,
                   help="LLM USD hard cap for this run (default BudgetCaps.max_llm_usd); never raised implicitly")
    p.add_argument("--export", default=None)

    p = sub.add_parser("report", help="write session markdown report")
    p.add_argument("--session-id", required=True)

    p = sub.add_parser("ab-protocol", help="write A/B evaluation protocol (not executed)")

    a = ap.parse_args(argv)
    cfg = SavantConfig()
    cfg.llm = env_llm_config(cfg.llm)
    if a.cmd in ("analyze", "plan", "run") and getattr(a, "export", None):
        cfg.seed_export = Path(a.export)

    if a.cmd == "analyze":
        session = init_session(cfg, a.session_id or "analyze")
        print(json.dumps({
            "session_id": session["session_id"],
            "dir": str(session["dir"]),
            "problem_report_sha": session["report"]["content_sha256"],
            "ingest_run_id": session["ingest"]["run_id"],
            "baseline": session["ingest"]["baseline_score"],
            "best": session["ingest"]["best"],
            "probe_mean": session["ingest"]["probe"]["mean"],
            "blockshift_mean": session["ingest"]["blockshift"]["mean"],
            "paired_mean_diff": session["ingest"]["paired_blockshift_vs_probe"]["mean_diff"],
            "claim_allowed": session["ingest"]["paired_blockshift_vs_probe"]["claim_allowed"],
            "caveat": session["ingest"]["caveat"],
        }, indent=2))
        write_ab_protocol(session["dir"])
        write_session_report(session["dir"])
        return 0

    if a.cmd == "plan" and a.reuse_council:
        from savant.debate import council_from_memory, finish_council
        from savant.memory import ResearchMemory
        from savant.planner import build_job_plan, write_plan
        sdir = cfg.session_dir(a.session_id)
        mem_path = sdir / "research_memory.jsonl"
        if not mem_path.exists():
            raise SystemExit(f"no research memory at {mem_path}")
        memory = ResearchMemory(mem_path)
        cached = council_from_memory(memory)
        if cached is None:
            raise SystemExit("session memory has no complete council (hypotheses, critiques, votes)")
        council = finish_council(
            cached["hypotheses"], cached["critiques"], cached["votes"], memory,
            voting_seed=cfg.voting_seed + 1, max_select=cfg.budgets.max_strategies_per_iteration)
        it_dir = sdir / "iteration_01"
        it_dir.mkdir(parents=True, exist_ok=True)
        plan = build_job_plan(council["selected"], ResourceLedger(), cfg.budgets,
                              iteration=1, seeds=cfg.matched_seeds)
        write_plan(plan, it_dir)
        (it_dir / "council.json").write_text(json.dumps({
            "reused_from_memory": True,
            "hypotheses": council["hypotheses"],
            "ranking": [{"id": r["hypothesis_id"], "score": r["score"]} for r in council["ranking"]],
            "selected_ids": [h["hypothesis_id"] for h in council["selected"]],
            "materialized": council["materialized"],
        }, indent=2, default=str), encoding="utf-8")
        print(json.dumps({
            "session_id": a.session_id, "reused_from_memory": True, "llm_calls": 0,
            "selected": [h["hypothesis_id"] for h in council["selected"]],
            "materialized": council["materialized"],
            "job_plan": plan.to_dict(),
        }, indent=2, default=str))
        return 0

    if a.cmd == "plan":
        if a.confirm_llm:
            client = LLMClient(cfg.llm, max_usd=cfg.budgets.max_llm_usd, allow_mock=False)
        else:
            cfg.llm.provider = "mock"
            client = LLMClient(cfg.llm, allow_mock=True)
        session = init_session(cfg, a.session_id)
        step = plan_iteration(cfg, session, iteration=1, client=client, ledger=ResourceLedger())
        print(json.dumps({
            "session_id": session["session_id"],
            "selected": [h["hypothesis_id"] for h in step["council"]["selected"]],
            "materialized": step["council"]["materialized"],
            "job_plan": step["plan"].to_dict(),
            "llm_provider": client.cfg.provider,
            "llm_usd": client.usage.usd,
        }, indent=2, default=str))
        return 0

    if a.cmd == "run":
        if a.confirm_modal and not a.confirm_llm and not a.mock:
            # Modal without LLM still ok with mock council
            a.mock = True
        if a.max_llm_usd is not None:
            cfg.budgets.max_llm_usd = a.max_llm_usd
        if a.confirm_llm or a.confirm_modal:
            print("Paid execution flags set. Caps:", cfg.budgets)
        if not a.confirm_llm:
            a.mock = True
        if a.astra:
            from savant.config import astra_config
            cfg.llm = astra_config(cfg.llm)
            if a.uniform_max_effort:
                cfg.llm.effort_by_purpose = {}
            if a.confirm_llm and not a.mock:
                print(f"LLM: {cfg.llm.model} reasoning_effort={cfg.llm.reasoning_effort} "
                      f"by_purpose={cfg.llm.effort_by_purpose} "
                      f"max_output_tokens={cfg.llm.max_output_tokens} (no fallback)")
        out = run_loop(cfg, max_iterations=a.max_iterations, confirm_llm=a.confirm_llm,
                       confirm_modal=a.confirm_modal, mock=a.mock or not a.confirm_llm,
                       session_id=a.session_id)
        write_session_report(out["session"]["dir"])
        print(json.dumps({
            "session_id": out["session"]["session_id"],
            "dir": str(out["session"]["dir"]),
            "n_steps": len(out["history"]),
            "stop_reason": out.get("stop_reason"),
            "modal_ledger": out["ledger"].__dict__,
            "llm_spent_usd_total": out["client"].spent_usd(),
            "llm_model": out["client"].cfg.model,
            "reasoning_effort": out["client"].cfg.reasoning_effort,
        }, indent=2, default=str))
        return 0

    if a.cmd == "report":
        path = write_session_report(cfg.session_dir(a.session_id))
        print(path.read_text(encoding="utf-8"))
        return 0

    if a.cmd == "ab-protocol":
        path = write_ab_protocol(cfg.runs_dir / "ab")
        print(f"wrote {path}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
