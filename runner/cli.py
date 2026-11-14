"""CLI.

    python -m runner.cli run [-n N] [--agent auto|claude|heuristic]
    python -m runner.cli status
    python -m runner.cli freeze        # re-lock frozen files after a deliberate human edit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import frozen
from .agent import make_agent
from .loop import load_config, run_experiment, status

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run N experiments (a baseline first if history is empty)")
    r.add_argument("-n", type=int, default=1)
    r.add_argument("--agent", choices=["auto", "claude", "heuristic"], default=None)
    sub.add_parser("status", help="print the experiment log and best score")
    sub.add_parser("freeze", help="write frozen.lock from the current problem/ and runner/")
    a = ap.parse_args(argv)

    if a.cmd == "freeze":
        frozen.write_lock(ROOT)
        print(f"wrote {frozen.LOCK_NAME} (hash {frozen.check(ROOT)})")
    elif a.cmd == "status":
        print(status(ROOT))
    elif a.cmd == "run":
        cfg = load_config(ROOT)
        agent = make_agent(a.agent or cfg["agent"], ROOT, cfg["model"], cfg["effort"])
        print(f"agent: {agent.name}")
        for _ in range(a.n):
            row = run_experiment(ROOT, agent)
            if row["decision"] == "baseline":
                print(f"{row['id']} baseline score={row['score']:.6f}")
                row = run_experiment(ROOT, agent)
            sc = "-" if row["score"] is None else f"{row['score']:.6f}"
            print(f"{row['id']} {row['decision']:6} score={sc}  {row['hypothesis'][:90]}")
            if row["failure_reason"]:
                print(f"         reason: {row['failure_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
