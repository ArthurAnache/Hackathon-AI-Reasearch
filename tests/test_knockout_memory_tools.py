from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from knockout.memory_backfill import FACT_KEYS, facts_for_run, write_facts
from knockout.memory_metrics import analyze_run


ROOT = Path(__file__).resolve().parents[1]
RUN_45 = ROOT / "knockout/results/20261115T023802Z-second-autocorrelation-inequality-45e7be9"


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")


def test_backfill_is_deterministic_and_refutes_gene_and_operator(tmp_path: Path) -> None:
    first, second = facts_for_run(RUN_45), facts_for_run(RUN_45)
    out1, out2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_facts(first, out1)
    write_facts(second, out2)
    assert out1.read_bytes() == out2.read_bytes()
    lines = [json.loads(line) for line in out1.read_text().splitlines()]
    assert all(tuple(row) == FACT_KEYS for row in lines)
    refuted = [f for f in lines if f["kind"] == "refuted"]
    assert any(f["metadata"]["experiment_id"] == "g0003" and "0 wins / 3 losses" in f["content"]
               for f in refuted)
    assert any(f["metadata"]["operator"] == "fresh" for f in refuted)


def test_push_without_memory_client_exits_two_with_guidance(tmp_path: Path) -> None:
    facts_path = tmp_path / "facts.jsonl"
    facts_path.write_text("", encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "knockout.memory_backfill", "--push", str(facts_path)],
                          cwd=ROOT, text=True, capture_output=True, check=False)
    assert proc.returncode == 2
    assert "run --push after merging feature/ws2-pastdev-memory" in proc.stderr


def test_metrics_repeat_rate_win_fraction_and_usd_per_win(tmp_path: Path) -> None:
    on = tmp_path / "on"
    on.mkdir()
    (on / "strategy.yaml").write_text("selection:\n  eps_rel: 0.000001\n", encoding="utf-8")
    experiments = [
        {"experiment_id": "e1", "run_id": "on", "timestamp": "2026-01-02T00:00:00Z",
         "hypothesis": "alpha beta gamma delta epsilon zeta eta theta", "score": 1.2,
         "config": {"operator": "refine", "ref_score": 1.0, "cost_usd": 0.2}},
        {"experiment_id": "e2", "run_id": "on", "timestamp": "2026-01-03T00:00:00Z",
         "hypothesis": "novel different proposal words", "score": 0.9,
         "config": {"operator": "fresh", "ref_score": 1.0, "cost_usd": 0.1}},
    ]
    _jsonl(on / "experiments.jsonl", experiments)
    _jsonl(on / "events.jsonl", [
        {"type": "llm_call", "payload": {"usd": 0.3}},
        {"type": "memory_recall", "payload": {"fact_ids": ["f1"]}},
    ])
    facts = [{"fact_id": "f1", "kind": "refuted",
              "content": "[scope toy] alpha beta gamma delta epsilon zeta eta theta",
              "timestamp": "2026-01-01T00:00:00Z", "identity": "agent-a@prior",
              "metadata": {"problem": "toy", "run_id": "prior", "agent": "agent-a"}}]
    result = analyze_run(on, facts)
    assert result["repeat_rate"] == 0.5
    assert result["win_fraction"] == 0.5
    assert abs(result["usd_per_win"] - 0.6) < 1e-12
    assert result["facts_recalled"] == 1
    assert result["external_facts_recalled"] == 1
