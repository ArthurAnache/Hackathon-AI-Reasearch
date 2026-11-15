"""A/B evaluation plan: Marie V2 alone vs Marie V2 + SAVANT (and optional no-debate ablation).

Does not claim superiority without controlled evidence. Provides a protocol only.
"""
from __future__ import annotations

import json
from pathlib import Path

from savant.config import HOLDOUT_SEEDS, MATCHED_SEEDS, PROBLEM


def write_ab_protocol(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "problem": PROBLEM,
        "arms": {
            "A": "Marie V2 without SAVANT (compare.py / evolution with fixed protocol)",
            "B": "Marie V2 + SAVANT council (this package)",
            "C_optional": "SAVANT without debate/voting (independent proposals → manager picks by rubric only)",
        },
        "controls": {
            "frozen_problem": True,
            "same_incumbent_certificate": True,
            "matched_seeds_exploratory": list(MATCHED_SEEDS),
            "holdout_seeds_confirmatory": list(HOLDOUT_SEEDS),
            "budget_s_per_job": 60,
            "cpu": 1, "memory_mb": 4096,
            "max_jobs_total_per_arm": 60,
            "verifier": "WS1 pinned + independent exact on best",
        },
        "metrics": [
            "best_verified_score",
            "improvement_per_cpu_second",
            "improvement_per_dollar_including_llm",
            "n_distinct_meaningful_hypotheses",
            "success_failure_rates",
            "wall_clock_s",
            "statistical_uncertainty",
            "reproducibility_on_holdout_seeds",
        ],
        "analysis": {
            "primary": "paired gains on holdout seeds vs shared incumbent",
            "predeclared_alpha_spend": True,
            "no_claim_without_controlled_evidence": True,
        },
        "status": "protocol_only_not_executed",
    }
    path = out_dir / "ab_protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (out_dir / "ab_protocol.md").write_text(
        "# A/B protocol\n\nSee `ab_protocol.json`. Do not claim SAVANT is better until this protocol is run.\n",
        encoding="utf-8")
    return path
