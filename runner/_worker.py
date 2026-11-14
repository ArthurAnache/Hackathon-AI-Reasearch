"""Subprocess entry point: run strategy.search once and print the solution as JSON.

Usage: python -m runner._worker <seed> <budget_s>
Runs in a fresh interpreter so a crashing / hanging / cheating strategy cannot
touch the runner's state. The parent process does the verification.
"""
import json
import sys
from pathlib import Path

import yaml


def main() -> None:
    seed, budget_s = int(sys.argv[1]), float(sys.argv[2])
    root = Path.cwd()
    sys.path.insert(0, str(root))
    from strategy.search import search  # noqa: E402

    config = yaml.safe_load((root / "strategy" / "config.yaml").read_text()) or {}
    solution = search(config, seed, budget_s)
    sys.stdout.write("\n__SOLUTION__" + json.dumps(solution) + "\n")


if __name__ == "__main__":
    main()
