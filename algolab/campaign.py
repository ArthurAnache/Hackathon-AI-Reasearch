"""Plan (and optionally launch) Algorithm Lab Modal waves for the three queued novel plugins.

DEFAULT: plan only. Real Modal dispatch requires --confirm-modal (and Arthur's approval for spend).
Never touches the active savant-exp1 session. Reuses ws3_bench job construction + the algolab Modal app
(block_network=True, no secrets).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algolab import author, gate, image as IMG, live_record, promote  # noqa: E402
from algolab.modal_app import spec as modal_spec  # noqa: E402

ADAPTER = "optimizers/second-autocorrelation-inequality_algolab.py"
CONTROL = "optimizers/second-autocorrelation-inequality_blockshift.py"
TIERS = {
    "T0": {"n": 4096, "budget_s": 6.0, "note": "contract: random sparse"},
    "T1": {"n": 32768, "budget_s": 20.0, "note": "climb: pooled incumbent"},
    "T1b": {"n": 4096, "budget_s": 30.0, "note": "proxy kink: polish then freeze, then escape"},
    "T2": {"n": None, "budget_s": 60.0, "note": "full 2M frozen incumbent"},
    "T3": {"n": None, "budget_s": 900.0, "note": "long-horizon Modal; GPU optional later"},
}


def plugin_config(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    adm = gate.admit(path)
    if not adm["admitted"]:
        raise SystemExit(f"plugin not admitted: {path}: {adm['reasons']}")
    return {"PLUGIN_SOURCE": src, "PLUGIN_SHA256": adm["sha256"], "PLUGIN_PARAMS": {}, "TRUSTED": False}


def build_wave_plan(tier: str, plugins: list[Path], seeds: list[int], *, include_control: bool = True) -> dict:
    t = TIERS[tier]
    issues = IMG.audit("second-autocorrelation-inequality")
    if issues:
        raise SystemExit(f"image audit failed: {issues}")
    strategies = []
    if include_control:
        strategies.append({"hypothesis_id": f"control-blockshift-{tier}", "optimizer": CONTROL,
                           "config": {}, "template": "blockshift"})
    for p in plugins:
        cfg = plugin_config(p)
        strategies.append({"hypothesis_id": f"algolab-{p.stem}-{tier}", "optimizer": ADAPTER,
                           "config": cfg, "template": "algolab_plugin", "plugin": str(p),
                           "plugin_sha256": cfg["PLUGIN_SHA256"]})
    n_jobs = len(strategies) * len(seeds)
    plan = {
        "plan_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{tier}",
        "kind": "algolab staged benchmark",
        "tier": tier, "tier_meta": t, "seeds": list(seeds), "budget_s": t["budget_s"],
        "n_jobs": n_jobs, "strategies": [
            {k: v for k, v in s.items() if k != "config"} | {
                "config_sha12": hashlib.sha256(json.dumps(s["config"], sort_keys=True, default=str).encode()).hexdigest()[:12],
                "has_plugin_source": "PLUGIN_SOURCE" in s["config"],
            } for s in strategies
        ],
        # Full configs (with PLUGIN_SOURCE) stay in batch_spec.json next to the plan, not in this summary.
        "modal": modal_spec(),
        "image_audit": "ok",
        "record": live_record.refresh(arthur_authorized=False),
        "promotion_thresholds": promote.THRESHOLDS.get(tier),
        "notes": [
            "Plan only unless --confirm-modal is passed.",
            "Isolation: Modal block_network=True, secrets=[], allowlisted image, gate + audit hook, parent verifier.",
            "Do not hot-patch savant-exp1; run this as a separate campaign.",
        ],
        "_strategies_full": strategies,  # used locally to build the batch; stripped before writing summary
    }
    return plan


def write_plan(plan: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    full = plan.pop("_strategies_full")
    (out_dir / "job_plan.json").write_text(json.dumps(plan, indent=2, default=str) + "\n", encoding="utf-8")
    batch = {
        "optimizers": [s["optimizer"] for s in full],
        "configs": [s["config"] for s in full],
        "labels": [s["hypothesis_id"] for s in full],
        "seeds": plan["seeds"],
        "budget_s": plan["budget_s"],
        "n_jobs": plan["n_jobs"],
        "tier": plan["tier"],
        "modal_app": plan["modal"]["app"],
        "block_network": True,
    }
    (out_dir / "batch_spec.json").write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
    plan["_strategies_full"] = full  # restore for caller
    return out_dir / "job_plan.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_seeds = sub.add_parser("admit-seeds", help="Gate-admit the three queued novel plugins (no LLM, no Modal)")
    p_seeds.add_argument("--out", type=Path, default=None)

    p_plan = sub.add_parser("plan", help="Write a Modal wave plan (no dispatch)")
    p_plan.add_argument("--tier", choices=list(TIERS), default="T0")
    p_plan.add_argument("--plugins", nargs="+", default=[
        "algolab/plugins/lp_packet_ascent.py",
        "algolab/plugins/two_box_envelope.py",
        "algolab/plugins/dinkelbach_kelley.py",
    ])
    p_plan.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    p_plan.add_argument("--out", type=Path, default=None)

    p_ladder = sub.add_parser("plan-ladder", help="Write plans for T0→T1→T1b→T2 (T3 separate)")
    p_ladder.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    p_ladder.add_argument("--out", type=Path, default=None)

    p_live = sub.add_parser("live-record", help="Show frozen (default) or live record target")
    p_live.add_argument("--arthur-authorized", action="store_true",
                        help="Arthur authorizes a live leaderboard GET in THIS call")

    p_dispatch = sub.add_parser("dispatch", help="Dispatch a previously written plan via Modal (PAID)")
    p_dispatch.add_argument("plan_dir", type=Path)
    p_dispatch.add_argument("--confirm-modal", action="store_true")

    a = ap.parse_args(argv)

    if a.cmd == "admit-seeds":
        rows = author.materialize_seed_plugins(a.out)
        print(json.dumps({"admitted": sum(r["status"] == "admitted" for r in rows), "rows": rows}, indent=2))
        return 0 if all(r["status"] == "admitted" for r in rows) else 1

    if a.cmd == "live-record":
        print(json.dumps(live_record.refresh(arthur_authorized=a.arthur_authorized), indent=2, default=str))
        return 0

    if a.cmd == "plan":
        out = a.out or (ROOT / "runs" / "algolab" / "plans" / time.strftime("%Y%m%dT%H%M%S") / a.tier)
        plan = build_wave_plan(a.tier, [ROOT / p for p in a.plugins], a.seeds)
        path = write_plan(plan, out)
        print(json.dumps({"plan": str(path), "n_jobs": plan["n_jobs"], "tier": a.tier,
                          "target": plan["record"]["target"], "mode": plan["record"]["mode"]}, indent=2))
        return 0

    if a.cmd == "plan-ladder":
        base = a.out or (ROOT / "runs" / "algolab" / "plans" / time.strftime("%Y%m%dT%H%M%S"))
        plugins = [ROOT / p for p in (
            "algolab/plugins/lp_packet_ascent.py",
            "algolab/plugins/two_box_envelope.py",
            "algolab/plugins/dinkelbach_kelley.py",
        )]
        summary = []
        for tier in ("T0", "T1", "T1b", "T2"):
            plan = build_wave_plan(tier, plugins, a.seeds)
            path = write_plan(plan, base / tier)
            summary.append({"tier": tier, "plan": str(path), "n_jobs": plan["n_jobs"],
                            "budget_s": plan["budget_s"]})
        (base / "ladder.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ladder": str(base / "ladder.json"), "tiers": summary}, indent=2))
        return 0

    if a.cmd == "dispatch":
        if not a.confirm_modal:
            print("Refusing: pass --confirm-modal after Arthur approves Modal spend.")
            return 2
        # Paid path: wire through ws3_bench.execute_jobs with the algolab Modal app.
        # Not exercised in this offline integration.
        raise SystemExit(
            "Dispatch is wired for Arthur's approval window. "
            "Use: python -m ws3_bench run --executor modal --confirm-modal with the batch_spec.json "
            "optimizers/configs/labels, after pointing WS3_BENCH at algolab.modal_app (see INTEGRATION_PLAN.md)."
        )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
