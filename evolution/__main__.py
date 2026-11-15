"""python -m evolution --problem 1a --optimizer optimizers/1a_tuned.py [--islands 2] [--rounds 4] [--seeds 3]
                      [--budget 10] [--candidates 4] [--workers 8] [--executor local|modal]
                      [--llm auto|none|openai|claude] [--llm-model M] [--llm-budget-usd 10] [--llm-max-edits 2] [--llm-stall-rounds 2]
                      [--exa] [--pastdev] [--agent-id NAME] [--memory-scope NAME] [--memory-since ISO]

Keys come from the git-ignored .env (OPENAI_API_KEY, ANTHROPIC_API_KEY, EXA_API_KEYS, PAST_API_KEYS); every tool is optional.
"""
from __future__ import annotations

import argparse
import sys

import interfaces as I
from certificates import CertificateStore
from execution import LocalExecutor, ModalExecutor

from .llm import EditorPolicy, editor as make_editor
from .orchestrator import Orchestrator
from .tools import ResearchTools


def show(orch, rep) -> None:
    fmt = lambda v: "-" if v is None else f"{v:.10g}"
    print(f"\n== round {rep.round}  allocation {rep.allocation}")
    for island, arm, status, mean, best, sig in rep.results:
        print(f"   {island:8} {arm:15} {status:8} mean {fmt(mean):>14}  best seed {fmt(best):>14}{'  *' if sig else ''}")
    if rep.migrated.get("genes") or rep.migrated.get("certificate"):
        print(f"   migration: genes {rep.migrated['genes']}  certificate {rep.migrated['certificate']}")
    for isl in orch.islands:
        alive = [g for g, s in isl.pool.state.items() if s == "validated"]
        dead = [g for g, s in isl.pool.state.items() if s == "graveyard"]
        print(f"   {isl.id}: best {fmt(isl.best_score)} ({isl.best_cert}) validated {alive} graveyard {dead}")
    t = rep.tools
    if t.get("literature") or t.get("genes") or t.get("buried") or t.get("shared"):
        print(f"   tools: exa +{t.get('literature', 0)} sources | past.dev genes in {t.get('genes', [])} "
              f"buried {t.get('buried', [])} | facts shared {t.get('shared', 0)}")
    if t.get("llm"):
        l = t["llm"]
        print(f"   llm: {l['calls']} call(s) this round, spent ${l['spent_usd']:.4f} of ${l['budget_usd']:.2f}, "
              f"skipped {l['skipped']}")
    print(f"   bandit {orch.bandit.summary()}  record {orch.problem.best_known}")


def default_optimizer(problem_id: str) -> str:
    """optimizers/<id>_tuned.py, else the only (or first non-baseline) optimizers/<id>_*.py."""
    tuned = I.ROOT / "optimizers" / f"{problem_id}_tuned.py"
    if tuned.exists():
        return str(tuned.relative_to(I.ROOT))
    found = sorted((I.ROOT / "optimizers").glob(f"{problem_id}_*.py"))
    found = [f for f in found if not f.stem.endswith(("_baseline", "_probe"))] or found
    if not found:
        raise SystemExit(f"no optimizer for {problem_id}: add optimizers/{problem_id}_<name>.py "
                         f'with PROBLEM = "{problem_id}"')
    return str(found[0].relative_to(I.ROOT))


def check_optimizer(problem_id: str, optimizer: str) -> None:
    """Refuse to start (instead of letting every job crash) when the optimizer is for another problem."""
    declared = I.FileOptimizer(I.ROOT / optimizer).problem_id
    if declared != problem_id:
        raise SystemExit(f"{optimizer} declares PROBLEM = {declared!r}, not {problem_id!r}; "
                         f"use an optimizer written for this problem (optimizers/{problem_id}_*.py)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--optimizer", help="base optimizer (default optimizers/<problem>_tuned.py, else <problem>_*.py)")
    ap.add_argument("--islands", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=float, default=10.0, help="seconds per job")
    ap.add_argument("--candidates", type=int, default=4, help="candidates per island per round")
    ap.add_argument("--workers", type=int, default=None, help="local processes")
    ap.add_argument("--executor", choices=["local", "modal"], default="local")
    ap.add_argument("--llm", choices=["auto", "none", "openai", "claude"], default="auto",
                    help="structural editor; auto = openai if OPENAI_API_KEY, else claude if ANTHROPIC_API_KEY, else none")
    ap.add_argument("--llm-model", default=None, help="default gpt-6.1-terra (openai) / claude-opus-5-5 (claude)")
    ap.add_argument("--llm-budget-usd", type=float, default=10.0,
                    help="hard cap on total LLM spend, tracked across runs in llm_spend.json")
    ap.add_argument("--llm-max-edits", type=int, default=2, help="max LLM edits per round, all islands together")
    ap.add_argument("--llm-stall-rounds", type=int, default=2,
                    help="call the LLM only for an island that has not improved for this many rounds")
    ap.add_argument("--llm-max-tokens", type=int, default=12000, help="output cap per LLM call")
    ap.add_argument("--store", default=str(I.ROOT / "experiments.jsonl"))
    ap.add_argument("--certs", default=str(I.ROOT / "certificates"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exa", action="store_true", help="literature search (record audit, then ideas on plateau)")
    ap.add_argument("--pastdev", action="store_true", help="share gene verdicts and best scores with other agents")
    ap.add_argument("--agent-id", default="evolution", help="identity in shared memory")
    ap.add_argument("--memory-scope", default=None, help="isolate shared memory for a fair comparison")
    ap.add_argument("--memory-since", default=None, help="ISO time; read only facts written after it")
    a = ap.parse_args(argv)
    from integrations._common import load_env
    load_env()                                         # keys from .env; existing variables win
    since = a.memory_since
    if a.memory_scope and not since:
        from integrations.run_loop import scope_start
        since = scope_start(a.memory_scope)
    tools = ResearchTools.from_flags(a.exa, a.pastdev, a.agent_id, a.memory_scope, since)
    ed = make_editor(a.llm, a.llm_model, a.llm_budget_usd,
                     EditorPolicy(a.llm_stall_rounds, a.llm_max_edits, a.llm_max_tokens))
    executor = ModalExecutor() if a.executor == "modal" else LocalExecutor(a.workers)
    optimizer = a.optimizer or default_optimizer(a.problem)
    check_optimizer(a.problem, optimizer)
    orch = Orchestrator(a.problem, optimizer, executor,
                        I.JsonlStore(a.store), CertificateStore(a.certs), a.islands, a.seeds, a.budget,
                        a.candidates, a.seed, ed, tools=tools)
    print(f"problem {a.problem} ({orch.problem.direction}), record {orch.problem.best_known}; "
          f"{a.islands} islands x {a.candidates} candidates x {a.seeds} seeds, {a.budget}s/job, executor {executor.name}, "
          f"tools {tools.tag}, llm {ed.name + f' (spent ${ed.ledger.spent:.2f} of ${ed.ledger.budget_usd:.2f})' if ed else 'off'}")
    orch.run(a.rounds, on_round=show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
