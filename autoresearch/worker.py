"""Sandboxed child process: runs one candidate strategy for one seed.

    python -m autoresearch.worker <strategy_dir> <seed> <budget_s> <out_json>

It only produces a solution. Scoring happens in the parent with problem.verify,
so nothing the strategy computes or prints is trusted.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import yaml


def load_strategy(strategy_dir: Path):
    spec = importlib.util.spec_from_file_location("candidate_search", strategy_dir / "search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((strategy_dir / "config.yaml").read_text()) or {}
    if not isinstance(config, dict):
        raise TypeError("config.yaml must be a mapping")
    return module, config


def main(argv: list[str]) -> int:
    strategy_dir, seed, budget_s, out = Path(argv[0]), int(argv[1]), float(argv[2]), Path(argv[3])
    module, config = load_strategy(strategy_dir)
    t0 = time.monotonic()
    solution = module.search(config, seed, budget_s)
    elapsed = time.monotonic() - t0
    out.write_text(json.dumps({"solution": solution, "elapsed_s": elapsed}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
