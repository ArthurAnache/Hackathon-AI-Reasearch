"""Run the WS2 research loop with the WS4 integrations plugged in.

    python -m integrations.run_loop --problem 1a -n 5 --budget 20 --agent-id laptop-a --exa --pastdev
    python -m integrations.run_loop --problem 1a -n 5 --budget 20 --agent-id laptop-b --pastdev

--exa           literature search (record audit, then ideas on plateau)
--pastdev       read other agents' facts as evidence and mirror every experiment
                to shared memory, attributed to --agent-id
--memory-scope  isolate this comparison from earlier runs: facts are tagged with the
                scope and only facts with the same tag are read. The first use of a
                scope on this machine records its start time, and only facts written
                after it are requested from past.dev (--memory-since overrides it,
                e.g. for Modal workers that share a scope started elsewhere).
                Scopes named import-* (integrations.import_results) use the tag only.
                Without a scope, agents read the accumulated memory of every run.
IMPORTANT for comparisons: give every arm its own --store file. The agent reads
the experiment history from the store, so a shared store leaks results between
arms even with every sponsor switched off.
Every experiment record gets an "integrations" field with the switches in force.
Without flags it behaves exactly like `python loop.py` (plus that field).
"""
from __future__ import annotations

import argparse
import sys

import interfaces as I
import prepare
from loop import _print, research_loop
from research import FeedbackAgent

from .evidence import PastDevMirrorStore, TaggedStore, from_flags
from .pastdev_client import get_memory, scope_start


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--optimizer")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--store", default=str(prepare.ROOT / "experiments.jsonl"))
    ap.add_argument("--agent-id", default="agent", help="identity used in shared memory")
    ap.add_argument("--exa", action="store_true")
    ap.add_argument("--pastdev", action="store_true")
    ap.add_argument("--memory-scope", default=None, help="isolate shared memory for a fair comparison")
    ap.add_argument("--memory-since", default=None, help="ISO time; read only facts written after it")
    ap.add_argument("--no-warm-start", action="store_true")
    a = ap.parse_args(argv)

    scope = a.memory_scope
    # import-* scopes hold facts written before this run (integrations.import_results): tag filter only
    since = a.memory_since or (scope_start(scope) if scope and not scope.startswith("import-") else None)
    problem = I.get_problem(a.problem)
    opt = I.FileOptimizer(a.optimizer or prepare.ROOT / "optimizers" / f"{a.problem}_baseline.py")
    agent = FeedbackAgent(problem, opt, warm_start=False if a.no_warm_start else None)
    agent.name = f"{agent.name}@{a.agent_id}"     # experiment records show which machine/agent ran it

    memory = get_memory() if a.pastdev else None
    base = PastDevMirrorStore(a.store, memory, author=a.agent_id, scope=scope) if a.pastdev \
        else I.JsonlStore(a.store)
    store = TaggedStore(base, {"exa": a.exa, "pastdev": a.pastdev, "memory_scope": scope,
                               "memory_since": since, "agent_id": a.agent_id,
                               "memory_backend": memory.name if memory else None})
    evidence = from_flags(a.agent_id, a.exa, a.pastdev, memory=memory, scope=scope, since=since)
    print(f"agent: {agent.name} | exa: {a.exa} | past.dev: {a.pastdev} | scope: {scope} since {since}")

    def show(r):
        _print(r)
        for e in r.get("evidence") or []:
            e = e if isinstance(e, dict) else e.__dict__
            print(f"           evidence [{e.get('source')}] by {e.get('author')}: {e.get('citation')}")

    research_loop(problem, opt, agent, store, evidence_source=evidence, n=a.n, budget_s=a.budget, on_result=show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
