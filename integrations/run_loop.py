"""Run the WS2 research loop with the WS4 integrations plugged in.

    python -m integrations.run_loop --problem 1a -n 5 --budget 20 --agent-id laptop-a --exa --pastdev
    python -m integrations.run_loop --problem 1a -n 5 --budget 20 --agent-id laptop-b --pastdev

--exa       literature search (record audit, then ideas on plateau)
--pastdev   read other agents' facts as evidence and mirror every experiment
            to shared memory, attributed to --agent-id
Without flags it behaves exactly like `python loop.py`.
"""
from __future__ import annotations

import argparse
import sys

import interfaces as I
import prepare
from loop import _print, research_loop
from research import FeedbackAgent

from .evidence import PastDevMirrorStore, from_flags
from .pastdev_client import get_memory


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
    ap.add_argument("--no-warm-start", action="store_true")
    a = ap.parse_args(argv)

    problem = I.get_problem(a.problem)
    opt = I.FileOptimizer(a.optimizer or prepare.ROOT / "optimizers" / f"{a.problem}_baseline.py")
    agent = FeedbackAgent(problem, opt, warm_start=False if a.no_warm_start else None)
    agent.name = f"{agent.name}@{a.agent_id}"     # experiment records show which machine/agent ran it
    store = PastDevMirrorStore(a.store, get_memory(), author=a.agent_id) if a.pastdev else I.JsonlStore(a.store)
    evidence = from_flags(a.agent_id, a.exa, a.pastdev)
    print(f"agent: {agent.name} as {a.agent_id} | exa: {a.exa} | past.dev: {a.pastdev}")

    def show(r):
        _print(r)
        for e in r.get("evidence") or []:
            e = e if isinstance(e, dict) else e.__dict__
            print(f"           evidence [{e.get('source')}] by {e.get('author')}: {e.get('citation')}")

    research_loop(problem, opt, agent, store, evidence_source=evidence, n=a.n, budget_s=a.budget, on_result=show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
