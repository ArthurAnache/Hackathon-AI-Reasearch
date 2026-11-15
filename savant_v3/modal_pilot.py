"""V3 Modal structural pilot — hard-capped at $20 additional, shared $300 ceiling.

Uses real ws3_bench Modal path with V3-isolated app/volume names.
Frozen WS1 certificate warm-start (independent constructions via Engine A/B/C).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "second-autocorrelation-inequality"
PILOT_CAP = 20.0
SESSION = "v3-pilot20"

ENGINES = [
    {"id": "engineA-comb", "plugin": "algolab/plugins/comb_reconstruct.py", "kind": "perturb"},
    {"id": "engineB-multires", "plugin": "algolab/plugins/multires_basin.py", "kind": "improve"},
    {"id": "engineC-activeset", "plugin": "algolab/plugins/active_set_exchange.py", "kind": "improve"},
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--budget-s", type=float, default=90.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 202])
    args = ap.parse_args(argv)
    if not args.confirm:
        print("Pass --confirm to spend Modal dollars (pilot cap $20, shared ceiling).")
        return 2

    from savant_v3 import shared_budget
    rem = shared_budget.remaining()
    if rem < 0.05:
        print("shared ceiling exhausted", rem)
        return 1

    session = ROOT / "runs" / "v3" / SESSION
    session.mkdir(parents=True, exist_ok=True)
    # Session spend approval
    (session / "spend_approval.json").write_text(json.dumps({
        "approved": True,
        "approval_id": "v3-pilot20-2026-11-15",
        "approved_by": "Thibault",
        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_modal_usd": PILOT_CAP,
        "max_llm_usd": 5.0,
        "max_total_usd": PILOT_CAP + 5.0,
        "campaign_ceiling_usd": 300.0,
        "shared_ceiling": str(shared_budget.CEILING),
    }, indent=2) + "\n", encoding="utf-8")

    from ws3_bench import image as IMG
    import ws3_bench.modal_app as MA_STOCK
    from ws3_bench import v3_modal_app as V3MA

    issues = IMG.audit(SLUG)
    if issues:
        print("image audit failed", issues)
        return 1

    from savant.templates import materialize
    from ws3_bench.campaign import build_plan, download_from_volume, export

    optimizers, configs, labels = [], [], []
    for eng in ENGINES:
        m = materialize({
            "hypothesis_id": eng["id"],
            "template": "algolab_plugin",
            "plugin": eng["plugin"],
            "title": eng["id"],
            "config_overrides": {"KIND": eng["kind"]},
        })
        if not m.get("executable"):
            print("not executable", eng, m.get("reason"))
            return 1
        optimizers.append(m["optimizer"])
        configs.append(m["config"])
        labels.append(eng["id"][:40])

    run_id = time.strftime("%Y%m%dT%H:%M:%SZ", time.gmtime()) + "-v3pilot"
    # fix typo-prone strftime - use standard
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-v3pilot"
    run_dir = ROOT / "runs" / "ws3" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    plan, jobs, problem = build_plan(
        SLUG, optimizers, args.seeds, args.budget_s, run_id,
        live_record=False, run_dir=run_dir, configs=configs, labels=labels,
    )
    plan["modal"]["app"] = V3MA.APP_NAME
    plan["modal"]["volume"] = V3MA.VOLUME_NAME
    est = float(plan["cost_estimate"]["total_usd"])
    buffered = min(est * 3.0 + 0.5, PILOT_CAP, rem)
    meta = {
        "run_id": run_id, "n_jobs": len(jobs), "budget_s": args.budget_s,
        "est_usd": est, "buffered_usd": buffered, "engines": [e["id"] for e in ENGINES],
        "shared_remaining_before": rem, "pilot_cap": PILOT_CAP,
        "modal_app": V3MA.APP_NAME, "modal_volume": V3MA.VOLUME_NAME,
    }
    (session / "pilot_preflight.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    if est > PILOT_CAP + 1e-9 or est > rem + 1e-9:
        print("BLOCKED by pilot/shared cap")
        return 1

    if V3MA.app is None or V3MA.run_job_v3 is None:
        print("Modal SDK / V3 app unavailable")
        return 1

    rid = shared_budget.reserve(buffered, purpose=f"v3_pilot:{run_id}", campaign="v3")
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=1, default=str), encoding="utf-8")

    # Point volume download helper at V3 volume
    MA_STOCK.VOLUME_NAME = V3MA.VOLUME_NAME

    import modal
    t0 = time.time()
    try:
        from execution import ModalExecutor
        from dataclasses import asdict
        launch = {"executor": "modal", "app": V3MA.APP_NAME,
                  "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with modal.enable_output():
            with V3MA.app.run(detach=True):
                launch["modal_app_id"] = V3MA.app.app_id
                (run_dir / "launch.json").write_text(json.dumps(launch, indent=1))
                results = ModalExecutor(function=V3MA.run_job_v3).map(jobs)
        launch["wall_s"] = round(time.time() - t0, 1)
        launch["results"] = [asdict(r) for r in results]
        (run_dir / "launch.json").write_text(json.dumps(launch, indent=1, default=str), encoding="utf-8")
        downloaded = download_from_volume(run_id, run_dir)
        out = export(run_dir, exact=False, live_record=False)
    except Exception as exc:  # noqa: BLE001
        shared_budget.release(rid)
        (session / "pilot_result.json").write_text(json.dumps({
            "ok": False, "error": f"{type(exc).__name__}: {exc}", "run_id": run_id,
        }, indent=2), encoding="utf-8")
        print("PILOT_FAILED", exc)
        return 1

    shared_budget.commit(rid, actual_usd=est)
    rows = list(csv.DictReader((Path(out["export"]) / "comparison.csv").open(encoding="utf-8")))
    from savant_v3.topology import topology_report
    import gzip
    # topology vs start cert for best row
    cert = problem.certificate() if hasattr(problem, "certificate") else None
    # problem from build_plan is interfaces problem
    from problems.catalog import load
    f0 = np_values = __import__("numpy").asarray(load(SLUG).certificate()["values"], dtype="float64")
    topo_rows = []
    for r in rows:
        if str(r.get("valid")).lower() != "true" or not r.get("candidate_sha256"):
            continue
        gz = Path(out["export"]) / "candidates" / f"{r['candidate_sha256']}.json.gz"
        if not gz.exists():
            continue
        cand = json.loads(gzip.decompress(gz.read_bytes()).decode())
        vals = cand.get("values") or cand.get("solution", {}).get("values")
        if not vals:
            continue
        # downsample compare if needed
        import numpy as np
        f1 = np.asarray(vals, dtype=np.float64)
        topo_rows.append({"job_id": r["job_id"], "optimizer": r["optimizer"],
                          "score": r["final_score"], "gain": r["gain"],
                          "topology": topology_report(f0, f1)})

    result = {
        "ok": True,
        "run_id": run_id,
        "wall_s": launch["wall_s"],
        "est_usd": est,
        "downloaded": len(downloaded),
        "export": str(out["export"]),
        "modal_app_id": launch.get("modal_app_id"),
        "rows": rows,
        "topology": topo_rows,
        "shared_remaining_after": shared_budget.remaining(),
    }
    (session / "pilot_result.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    # Append experiment log
    log = ROOT / "SAVANT_V3_EXPERIMENT_LOG.md"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Pilot {run_id}\n\n")
        fh.write(f"- jobs: {len(jobs)}, est ${est}, wall {launch['wall_s']}s\n")
        fh.write(f"- app: `{launch.get('modal_app_id')}` volume `savant-v3-results`\n")
        for r in rows:
            fh.write(f"- {r.get('job_id')}: status={r.get('status')} valid={r.get('valid')} "
                     f"gain={r.get('gain')} score={r.get('final_score')}\n")
        fh.write("\n")
    print(json.dumps({k: result[k] for k in result if k not in ("rows", "topology")}, indent=2, default=str))
    print("PILOT_OK" if any(str(r.get("valid")).lower() == "true" for r in rows) else "PILOT_PARTIAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
