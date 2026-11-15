"""SAVANT V3 scientific iterations — structural engines only, shared $300 ceiling.

Usage:
  python -m savant_v3.iterate --confirm --iter 2 --budget-s 300 --seeds 101 202 303
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SLUG = "second-autocorrelation-inequality"
PILOT_CAP_REMAINING_HINT = 20.0  # soft stage-2; stage-3 expands under shared ceiling
STAGE3_CUMULATIVE = 100.0

# Iteration ≥3: micro surgery + spectral low-mode + soft multires.
# Warm-start preferably from PUBLIC live #1 (see --warm-live).
ENGINES = [
    # it9 hail-mary: larger spectral kick + LSE recover + surgery (lineage it08).
    {"id": "engineE-spectral", "plugin": "algolab/plugins/spectral_lowmode.py", "kind": "improve",
     "params": {"MODE_FRAC": 0.08, "SIGMA0": 5e-3, "ROUNDS": 16, "LBFGS_MAXITER": 60}},
    {"id": "engineG-lse", "plugin": "algolab/plugins/lse_lbfgs_spectral.py", "kind": "improve",
     "params": {"KIND": "improve", "FIRST_IMPROVE_FRAC": 0.95, "LBFGS_MAXITER": 100}},
    {"id": "engineA-surgery", "plugin": "algolab/plugins/comb_surgery.py", "kind": "perturb",
     "params": {"TRIES": 220, "MAX_EDITS": 4, "SHIFT_FRAC": 0.002, "POLISH_ITERS": 20}},
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def refresh_target(session: Path) -> dict:
    from integrations import einstein_arena as ea
    from problems.catalog import load
    log = session / "arena_logs"
    log.mkdir(parents=True, exist_ok=True)
    ea.LOG_PATH = log / "arena_log.jsonl"
    p = load(SLUG)
    best = p.best_known_score(max_age_s=0)
    out = {
        "live": best,
        "delta": float(p.DELTA),
        "target": float(best["value"]) + float(p.DELTA),
        "read_at": _now(),
    }
    (session / "target.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--budget-s", type=float, default=300.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    ap.add_argument("--stage-cap", type=float, default=20.0,
                    help="additional USD cap for this wave (still bounded by shared ceiling)")
    ap.add_argument("--warm-live", action="store_true",
                    help="warm-start from PUBLIC live #1 (GET /api/solutions/best), parent-verified")
    args = ap.parse_args(argv)
    if not args.confirm:
        print("Pass --confirm to spend Modal dollars.")
        return 2

    from savant_v3 import shared_budget
    from savant_v3.topology import topology_report

    rem = shared_budget.remaining()
    if rem < 0.05:
        print("shared ceiling exhausted", rem)
        return 1

    session = ROOT / "runs" / "v3" / f"iter{args.iter:02d}"
    session.mkdir(parents=True, exist_ok=True)
    target = refresh_target(ROOT / "runs" / "v3")
    (session / "spend_approval.json").write_text(json.dumps({
        "approved": True,
        "approval_id": f"v3-iter{args.iter}-shared300",
        "approved_by": "Thibault",
        "approved_at": _now(),
        "max_modal_usd": min(args.stage_cap, rem),
        "campaign_ceiling_usd": 300.0,
        "shared_ceiling": str(shared_budget.CEILING),
        "target": target,
    }, indent=2) + "\n", encoding="utf-8")

    from ws3_bench import image as IMG
    import ws3_bench.modal_app as MA_STOCK
    from ws3_bench import v3_modal_app as V3MA
    from savant.templates import materialize
    from ws3_bench.campaign import build_plan, download_from_volume, export
    from problems.catalog import load
    from execution import ModalExecutor
    from dataclasses import asdict
    import modal

    issues = IMG.audit(SLUG)
    if issues:
        print("image audit failed", issues)
        return 1

    optimizers, configs, labels = [], [], []
    for eng in ENGINES:
        ov = {"KIND": eng["kind"], **(eng.get("params") or {})}
        m = materialize({
            "hypothesis_id": eng["id"],
            "template": "algolab_plugin",
            "plugin": eng["plugin"],
            "title": eng["id"],
            "config_overrides": ov,
        })
        if not m.get("executable"):
            print("not executable", eng, m.get("reason"))
            return 1
        optimizers.append(m["optimizer"])
        configs.append(m["config"])
        labels.append(eng["id"][:40])

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-v3it{args.iter:02d}"
    run_dir = ROOT / "runs" / "ws3" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    plan, jobs, _problem = build_plan(
        SLUG, optimizers, args.seeds, args.budget_s, run_id,
        live_record=False, run_dir=run_dir, configs=configs, labels=labels,
    )
    warm_meta = None
    if args.warm_live:
        import gzip as _gzip
        bake = ROOT / "problems" / SLUG / "live_public_best.json.gz"
        if bake.is_file():
            payload = json.loads(_gzip.decompress(bake.read_bytes()).decode())
            warm_meta = {
                "solution_id": payload.get("solution_id"),
                "verified_score": payload.get("verified_score"),
                "api_score": payload.get("verified_score"),
                "values": payload["values"],
                "agentName": payload.get("agentName"),
                "source": "image-bake-in",
            }
        else:
            from savant_v3.live_incumbent import fetch_and_verify
            warm_meta = fetch_and_verify(force=False)
        # Reference bake-in on the image — do NOT ship 2M values in the job payload.
        for j in jobs:
            j.warm_start_solution = None
            ws3 = dict((j.meta or {}).get("ws3") or {})
            ws3["warm_start"] = {
                "source": "public-live-best",
                "solution_id": warm_meta.get("solution_id"),
                "api_score": warm_meta.get("api_score"),
                "verified_score": warm_meta.get("verified_score"),
            }
            j.meta = {**(j.meta or {}), "ws3": ws3}
        plan["incumbent"]["warm_live"] = {
            "solution_id": warm_meta.get("solution_id"),
            "verified_score": warm_meta.get("verified_score"),
            "agentName": warm_meta.get("agentName"),
            "transfer": "by reference (live_public_best.json.gz in image)",
        }
    plan["modal"]["app"] = V3MA.APP_NAME
    plan["modal"]["volume"] = V3MA.VOLUME_NAME
    est = float(plan["cost_estimate"]["total_usd"])
    buffered = min(est * 3.0 + 0.5, args.stage_cap, rem)
    meta = {
        "run_id": run_id, "iter": args.iter, "n_jobs": len(jobs),
        "budget_s": args.budget_s, "est_usd": est, "buffered_usd": buffered,
        "engines": [e["id"] for e in ENGINES], "shared_remaining_before": rem,
        "target": target, "modal_app": V3MA.APP_NAME,
        "warm_live": bool(args.warm_live),
        "warm_meta": None if not warm_meta else {
            "solution_id": warm_meta.get("solution_id"),
            "verified_score": warm_meta.get("verified_score"),
            "api_score": warm_meta.get("api_score"),
        },
    }
    (session / "preflight.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, default=str))
    if est > args.stage_cap + 1e-9 or est > rem + 1e-9:
        print("BLOCKED by stage/shared cap")
        return 1
    if V3MA.app is None or V3MA.run_job_v3 is None:
        print("Modal SDK / V3 app unavailable")
        return 1

    rid = shared_budget.reserve(buffered, purpose=f"v3_iter{args.iter}:{run_id}", campaign="v3")
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=1, default=str), encoding="utf-8")
    MA_STOCK.VOLUME_NAME = V3MA.VOLUME_NAME

    t0 = time.time()
    results = []
    launch = {"executor": "modal", "app": V3MA.APP_NAME, "started_at": _now()}
    try:
        # No enable_output() stream — long waves were killed by client disconnect.
        with V3MA.app.run(detach=True):
            launch["modal_app_id"] = V3MA.app.app_id
            (run_dir / "launch.json").write_text(json.dumps(launch, indent=1))
            # Spawn then join so a dropped stream does not KeyboardInterrupt workers mid-flight.
            from dataclasses import asdict as _asdict
            handles = [V3MA.run_job_v3.spawn(_asdict(j)) for j in jobs]
            launch["spawned"] = len(handles)
            (run_dir / "launch.json").write_text(json.dumps(launch, indent=1))
            raw = []
            for h in handles:
                try:
                    raw.append(h.get())
                except Exception as e:  # noqa: BLE001
                    raw.append(e)
            from execution import JobResult
            out_results = []
            for job, r in zip(jobs, raw):
                if isinstance(r, BaseException) or not isinstance(r, dict):
                    r = {"job_id": job.job_id, "status": "crash", "candidate": None, "duration_s": 0.0,
                         "worker_id": "modal", "error": f"modal: {type(r).__name__}: {r}", "meta": job.meta}
                out_results.append(JobResult(**{k: r[k] for k in JobResult.__dataclass_fields__ if k in r}))
            results = out_results
    except Exception as exc:  # noqa: BLE001
        launch["map_error"] = f"{type(exc).__name__}: {exc}"
        launch["note"] = "client lost connection; attempting volume recovery"
        print("MAP_WARN", exc)
    try:
        launch["wall_s"] = round(time.time() - t0, 1)
        launch["results"] = [asdict(r) for r in results]
        (run_dir / "launch.json").write_text(json.dumps(launch, indent=1, default=str), encoding="utf-8")
        downloaded = download_from_volume(run_id, run_dir)
        if not downloaded and not results:
            raise RuntimeError("no Modal results and volume empty — jobs may still be running")
        out = export(run_dir, exact=False, live_record=False)
    except Exception as exc:  # noqa: BLE001
        shared_budget.release(rid)
        (session / "result.json").write_text(json.dumps({
            "ok": False, "error": f"{type(exc).__name__}: {exc}", "run_id": run_id,
            "modal_app_id": launch.get("modal_app_id"),
        }, indent=2), encoding="utf-8")
        print("ITER_FAILED", exc)
        return 1

    shared_budget.commit(rid, actual_usd=est)
    rows = list(csv.DictReader((Path(out["export"]) / "comparison.csv").open(encoding="utf-8")))
    # Topology baseline = warm-start used for this wave (live public or frozen cert)
    if args.warm_live and warm_meta:
        f0 = np.asarray(warm_meta["values"], dtype=np.float64)
        start_score = float(warm_meta["verified_score"])
    else:
        f0 = np.asarray(load(SLUG).certificate()["values"], dtype=np.float64)
        start_score = float(load(SLUG).score(load(SLUG).certificate()))
    live_target = float(target["target"])
    topo_rows = []
    best_row = None
    best_true_gain = -1e9
    for r in rows:
        if str(r.get("valid")).lower() != "true" or not r.get("candidate_sha256"):
            continue
        gz = Path(out["export"]) / "candidates" / f"{r['candidate_sha256']}.json.gz"
        if not gz.exists():
            continue
        cand = json.loads(gzip.decompress(gz.read_bytes()).decode())
        vals = cand.get("values")
        if not vals:
            continue
        f1 = np.asarray(vals, dtype=np.float64)
        tr = topology_report(f0, f1)
        final = float(r["final_score"])
        true_gain = final - start_score
        gap = live_target - final
        topo_rows.append({
            "job_id": r["job_id"], "score": r["final_score"],
            "gain_vs_frozen_export": r["gain"], "gain_vs_warm_start": true_gain,
            "gap_to_live_plus_delta": gap, "topology": tr,
        })
        if true_gain > best_true_gain:
            best_true_gain = true_gain
            best_row = {**r, "gain_vs_warm_start": true_gain, "gap_to_live_plus_delta": gap}

    result = {
        "ok": True, "iter": args.iter, "run_id": run_id, "wall_s": launch["wall_s"],
        "est_usd": est, "downloaded": len(downloaded), "export": str(out["export"]),
        "modal_app_id": launch.get("modal_app_id"), "rows": rows, "topology": topo_rows,
        "best": best_row, "target": target, "start_score": start_score,
        "best_gain_vs_warm_start": best_true_gain if best_row else None,
        "beats_live_plus_delta": bool(best_row and float(best_row["final_score"]) >= live_target),
        "shared_remaining_after": shared_budget.remaining(),
        "structural_hits": sum(1 for t in topo_rows if t["topology"].get("structural_change")),
    }
    (session / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")

    log = ROOT / "SAVANT_V3_EXPERIMENT_LOG.md"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Iteration {args.iter} — {run_id}\n\n")
        fh.write(f"- jobs: {len(jobs)}, est ${est}, wall {launch['wall_s']}s, "
                 f"structural_hits={result['structural_hits']}\n")
        fh.write(f"- live target: {target.get('target')} (live={target.get('live')})\n")
        for r in rows:
            fh.write(f"- {r.get('job_id')}: valid={r.get('valid')} gain={r.get('gain')} "
                     f"score={r.get('final_score')} status={r.get('status')}\n")
        for t in topo_rows:
            top = t["topology"]
            fh.write(f"  - topo {t['job_id']}: structural={top.get('structural_change')} "
                     f"Δteeth={top.get('delta_teeth')} jaccard={top.get('support_jaccard')} "
                     f"rel_l2={top.get('rel_l2')}\n")
        fh.write("\n")

    print(json.dumps({k: result[k] for k in result if k not in ("rows", "topology")}, indent=2, default=str))
    print("ITER_OK" if result["structural_hits"] or any(float(r.get("gain") or 0) > 1e-10 for r in rows) else "ITER_WEAK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
