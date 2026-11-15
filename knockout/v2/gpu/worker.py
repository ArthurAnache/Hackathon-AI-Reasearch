"""What runs inside the Modal containers: storing the warm starts, the E0 benchmark, and a batch of annealing
chains. torch is imported inside the GPU functions only (it is not installed locally).

Layout on the Volume (mounted at `vol`):
    starts/frozen.npy, starts/live.npy, starts/meta.json      warm starts (float64) and where they come from
    bench/<stamp>.json                                        E0 measurements
    <run_id>/<chain>/<step>.npy                               every solution a chain returned (float64)
    <run_id>/chains.jsonl                                     one record per stored solution (shared JSONL format)
    <run_id>/progress.jsonl, <run_id>/summary.json

Rules followed (CLAUDE.md, "WS3 — Modal compute and GPUs"): the GPU computes the smoothed objective only; every
stored solution is re-scored in float64 on CPU with the frozen verifier before it is logged; nothing here submits
or holds a secret.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from knockout.v2.gpu import schedule as sch
from knockout.v2.gpu.chain import initial_w, run_chains
from knockout.v2.gpu.cpu_score import problem
from knockout.v2.gpu.objective import evaluate, exact_score, fft_length, kappa_for

PROBLEM_URL = "https://einsteinarena.com/problems/second-autocorrelation-inequality"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def prepare_starts(vol: Path) -> dict:
    """Store the two warm starts on the Volume: the frozen certificate (EinsteinArena solution 2782) and the live
    #1 (read-only GET). Later containers start from these files, not from the network."""
    from knockout.problems_api import load_problem
    from knockout.run import load_init

    p = problem()
    out_dir = vol / "starts"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"read_at": _now(), "delta": float(p.DELTA)}
    cert = p.certificate()
    arr = np.asarray(cert["values"], dtype=np.float64)
    np.save(out_dir / "frozen.npy", arr)
    meta["frozen"] = {"path": "starts/frozen.npy", "n": int(arr.size), "nonzero": int(np.count_nonzero(arr)),
                      "score": float(p.score(cert)), "source": "problems/<slug>/certificate.json.gz (frozen)"}
    cand, info = load_init(load_problem(sch.SLUG), "live")
    arr = np.asarray(cand["values"], dtype=np.float64)
    np.save(out_dir / "live.npy", arr)
    meta["live"] = {"path": "starts/live.npy", "n": int(arr.size), "nonzero": int(np.count_nonzero(arr)),
                    "score": float(p.score(cand)), "source": "EinsteinArena live #1 (GET best solutions)",
                    "solution_id": info.get("solution_id"), "agent": info.get("agent"),
                    "displayed_score": info.get("displayed_score"), "sha256": info.get("sha256")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def score_in_subprocess(path: str) -> dict:
    """The frozen verifier's score of a stored .npy solution, computed in a fresh CPU process."""
    proc = subprocess.run([sys.executable, "-m", "knockout.v2.gpu.cpu_score", path], capture_output=True,
                          text=True, timeout=900)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        return {"path": path, "n": None, "nonzero": None, "score": None, "valid": False,
                "error": f"scorer failed (rc={proc.returncode}): {proc.stderr[-300:]}"}
    return json.loads(lines[-1])


def _start_path(vol: Path, start: str) -> Path:
    if start in ("frozen", "live"):
        return vol / "starts" / f"{start}.npy"
    if start.startswith("vol:"):
        rel = start[4:]
        if rel.startswith("/") or ".." in rel.split("/"):
            raise ValueError(f"bad start {start!r}")
        return vol / rel
    raise ValueError(f"unknown start {start!r}")


def _env(torch) -> dict:
    import scipy

    # plain str: torch.__version__ is a TorchVersion object, which the local client (no torch) cannot unpickle
    return {"device": "cuda", "gpu_type": str(torch.cuda.get_device_name(0)), "dtype": "float64",
            "torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda),
            "numpy_version": str(np.__version__), "scipy_version": str(scipy.__version__)}


# ---------------------------------------------------------------------------------------------------------------
# E0: numerical agreement and timing

def bench(vol: Path, spec: dict) -> dict:
    import scipy.fft as sf
    import torch

    from knockout.seeds import second_autocorrelation as seed

    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    meta = json.loads((vol / "starts" / "meta.json").read_text())
    f_live = np.load(vol / "starts" / "live.npy")
    n = int(f_live.size)
    out = {"measured_at": _now(), "env": _env(torch), "n": n, "start": meta["live"], "agreement": [],
           "gpu_timing": [], "cpu": {}}
    verifier = float(problem().score({"values": f_live.tolist()}))
    out["verifier_score"] = verifier

    m = fft_length(n)
    f1 = f_live[None, :]
    zero = np.zeros((1, 1))
    kap_np = kappa_for(np, f1, m)
    w_np = initial_w(np, f1, kap_np, zero)
    f_t = torch.from_numpy(f1).to(dev)
    kap_t = kappa_for(torch, f_t, m)
    w_t = initial_w(torch, f_t, kap_t, torch.zeros((1, 1), dtype=torch.float64, device=dev))
    for beta in (1e5, 1e7, 1e9, 1e11):
        phi_n, g_n, aux_n = evaluate(np, w_np, np.full((1, 1), beta), kap_np, m)
        phi_t, g_t, aux_t = evaluate(torch, w_t, torch.full((1, 1), beta, dtype=torch.float64, device=dev),
                                     kap_t, m)
        g_t = g_t.cpu().numpy()
        out["agreement"].append({
            "beta": beta, "phi_numpy": float(phi_n[0]), "phi_torch": float(phi_t[0]),
            "phi_abs_diff": abs(float(phi_n[0]) - float(phi_t[0])),
            "grad_max_abs_diff_over_max_abs": float(np.abs(g_t - g_n).max() / np.abs(g_n).max()),
            "exact_numpy": float(aux_n["exact"][0]), "exact_torch": float(aux_t["exact"][0]),
            "exact_torch_minus_verifier": float(aux_t["exact"][0]) - verifier,
            "exact_numpy_minus_verifier": float(aux_n["exact"][0]) - verifier})

    # CPU reference: the seed program's value-and-gradient (scipy.fft), and this module's numpy backend
    mm = sf.next_fast_len(2 * n - 1, real=True)
    seed._neg_log_c(w_np[0], 1e9, mm, n)
    t0 = time.perf_counter()
    for _ in range(4):
        seed._neg_log_c(w_np[0], 1e9, mm, n)
    out["cpu"]["seed_s_per_eval"] = (time.perf_counter() - t0) / 4
    t0 = time.perf_counter()
    for _ in range(2):
        evaluate(np, w_np, np.full((1, 1), 1e9), kap_np, m)
    out["cpu"]["numpy_backend_s_per_eval"] = (time.perf_counter() - t0) / 2

    # GPU: one batched value-and-gradient evaluation, for several batch sizes and both FFT sizes
    for kind in ("smooth", "pow2"):
        mk = fft_length(n, kind)
        for b in spec.get("batches", (1, 2, 4, 8, 16)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            w = w_t.repeat(b, 1)
            kap = kap_t.repeat(b, 1)
            beta = torch.full((b, 1), 1e7, dtype=torch.float64, device=dev)
            for _ in range(3):
                evaluate(torch, w, beta, kap, mk)
            torch.cuda.synchronize()
            reps = 12
            t0 = time.perf_counter()
            for _ in range(reps):
                evaluate(torch, w, beta, kap, mk)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / reps
            out["gpu_timing"].append({"fft": kind, "m": mk, "batch": b, "s_per_batched_eval": dt,
                                      "chain_evals_per_s": b / dt,
                                      "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30})
            del w, kap, beta

    # GPU: the whole optimizer step (trial point, evaluation, L-BFGS update), which is what a chain pays
    from knockout.v2.gpu.lbfgs import BatchedLBFGS

    out["gpu_loop"] = []
    for b in spec.get("loop_batches", (1, 8, 16)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        opt = BatchedLBFGS(torch, w_t.repeat(b, 1), history=int(spec.get("history", 8)))
        kap = kap_t.repeat(b, 1)
        beta = torch.full((b, 1), 1e7, dtype=torch.float64, device=dev)
        for k in range(42):
            if k == 12:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            xt = opt.trial()
            phi, grad, _ = evaluate(torch, xt, beta, kap, m)
            opt.update(xt, phi, grad)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 30
        out["gpu_loop"].append({"batch": b, "s_per_step": dt, "chain_evals_per_s": b / dt,
                                "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                                "accepted": list(opt.accepted)})
        del opt
    (vol / "bench").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (vol / "bench" / f"{stamp}.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------------------------------------------
# A batch of annealing chains

def anneal(vol: Path, spec: dict, commit=lambda: None) -> dict:
    """Run spec["chains"] in one batched tensor. Every stored solution gets one record in <run_id>/chains.jsonl,
    scored on CPU by the frozen verifier. The loop stops at spec["budget_s"] of wall clock whatever happens."""
    import torch

    t0 = time.monotonic()
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    chains = [sch.ChainSpec.from_dict(d) for d in spec["chains"]]
    total = sch.check_batch(chains)
    run_id = spec["run_id"]
    budget_s = float(spec["budget_s"])
    deadline = t0 + budget_s - float(spec.get("reserve_s", 150.0))
    run_dir = vol / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env = _env(torch)
    meta = json.loads((vol / "starts" / "meta.json").read_text())
    best_known = float(meta["live"]["score"])
    (run_dir / "spec.json").write_text(json.dumps({**spec, "env": env, "started_at": _now(),
                                                   "best_known_score": best_known,
                                                   "best_known_source": meta["live"]}, indent=1))
    pool = ThreadPoolExecutor(max_workers=int(spec.get("score_workers", 4)))   # each task is a CPU subprocess

    # warm starts: stored solutions only (WS3 rule 7), each re-scored here with the frozen verifier
    start_paths = {c.start: _start_path(vol, c.start) for c in chains}
    start_futures = {name: pool.submit(score_in_subprocess,str(path)) for name, path in start_paths.items()}
    arrays = {name: np.load(path) for name, path in start_paths.items()}
    sizes = {a.size for a in arrays.values()}
    if len(sizes) != 1:
        raise ValueError(f"chains of one batch need starts of the same length, got {sorted(sizes)}")
    n = sizes.pop()
    m = fft_length(n, spec.get("fft", "smooth"))
    rows = [np.maximum(arrays[c.start], 0.0) for c in chains]
    start_exact_gpu = exact_score(torch, torch.from_numpy(np.stack(rows)).to(dev), m).tolist()
    for i, c in enumerate(chains):
        if c.jitter > 0:      # the kick is drawn on CPU from the chain's seed, so it is the same on any device
            rows[i] = rows[i] * np.exp(c.jitter * np.random.default_rng(c.seed).standard_normal(n))
    f0 = torch.from_numpy(np.stack(rows)).to(dev)
    del rows
    kappa = kappa_for(torch, f0, m)
    floors = torch.tensor([[c.floor] for c in chains], dtype=torch.float64, device=dev)
    w0 = initial_w(torch, f0, kappa, floors)
    del f0
    start_scores = {}
    for name, fut in start_futures.items():
        res = fut.result()
        if not res["valid"]:
            raise RuntimeError(f"start {name!r} is rejected by the frozen verifier: {res['error']}")
        start_scores[name] = res["score"]

    pending, records = [], []
    chains_file = run_dir / "chains.jsonl"
    progress_file = run_dir / "progress.jsonl"

    def store(i, step, w_row, info, evals_done, extra=None):
        exp_id = sch.experiment_id(run_id, chains[i].name, step)
        path = vol / sch.solution_path(exp_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, (w_row * w_row).cpu().numpy())
        item = {"i": i, "step": step, "info": dict(info), "evals_done": evals_done,
                "wall_s": time.monotonic() - t0, "extra": dict(extra or {})}
        pending.append((item, pool.submit(score_in_subprocess,str(path))))

    def flush(wait=False):
        keep = []
        for item, fut in pending:
            if not (wait or fut.done()):
                keep.append((item, fut))
                continue
            res = fut.result()
            c = chains[item["i"]]
            info = item["info"]
            rec = sch.make_record(
                run_id=run_id, chain=c, step=item["step"], score=res["score"], valid=res["valid"],
                gpu_objective=info.get("smooth"), gpu_exact=info.get("exact"), start_score=start_scores[c.start],
                best_known_score=best_known, env=env, wall_s=item["wall_s"], budget_s=budget_s,
                source_commit=spec.get("source_commit", ""), evidence_urls=[PROBLEM_URL],
                extra={"beta": info.get("beta"), "evals_done": item["evals_done"], "n": res["n"],
                       "nonzero": res["nonzero"], "accepted": info.get("accepted"),
                       "rejected": info.get("rejected"), "restarts": info.get("restarts"),
                       "verifier_error": res["error"], "fft_len": m, "history": int(spec.get("history", 8)),
                       **item["extra"]})
            records.append(rec)
            with chains_file.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        pending[:] = keep
        if wait:
            commit()

    def on_stage_end(i, stage, evals_done, w_row, info):
        last = stage == len(chains[i].stages) - 1
        store(i, sch.step_name(stage), w_row, info, evals_done, {"stage": stage, "final": last})

    def on_progress(evals_done, total_evals, smooth, exact, best, stage):
        line = {"evals": evals_done, "total": total_evals, "wall_s": time.monotonic() - t0, "smooth": smooth,
                "exact": exact, "best_exact": best, "stage": stage}
        with progress_file.open("a") as fh:
            fh.write(json.dumps(line) + "\n")
        gains = " ".join(f"{c.name}:{e - s:+.2e}" for c, e, s in zip(chains, exact, start_exact_gpu))
        print(f"[{run_id}] evals {evals_done}/{total_evals} wall {line['wall_s']:.0f}s exact-start {gains}",
              flush=True)
        flush()

    out = run_chains(torch, w0, kappa, m, [c.stages for c in chains], history=int(spec.get("history", 8)),
                     on_stage_end=on_stage_end, on_progress=on_progress,
                     progress_every=int(spec.get("progress_every", 200)), deadline=deadline)
    loop_s = time.monotonic() - t0

    # the chain ends: the point where a stopped chain stands, and the best exact point each chain saw
    best_exact = out["best_exact"].tolist()
    for i, c in enumerate(chains):
        if out["stopped_early"]:
            info = {"beta": c.stages[min(out["stage"][i], len(c.stages) - 1)].beta,
                    "smooth": float(out["smooth"][i]), "exact": float(out["exact"][i]),
                    "accepted": out["accepted"][i], "rejected": out["rejected"][i], "restarts": out["restarts"][i]}
            store(i, "partial", out["x"][i], info, out["evals"], {"final": True, "stopped_early": True})
        if best_exact[i] > start_exact_gpu[i]:
            store(i, "best", out["best_w"][i], {"exact": best_exact[i]}, out["evals"],
                  {"best": True, "start_exact_gpu": start_exact_gpu[i]})
    flush(wait=True)
    pool.shutdown()

    table = []
    for i, c in enumerate(chains):
        mine = [r for r in records if r["chain"] == c.name]
        final = next((r for r in mine if r.get("final")), None)
        best = next((r for r in mine if r.get("best")), None)
        table.append({
            "chain": c.name, "start": c.start, "start_score": start_scores[c.start], "floor": c.floor,
            "jitter": c.jitter, "seed": c.seed,
            "beta_start": c.stages[0].beta, "beta_end": c.stages[-1].beta, "stages": len(c.stages),
            "evals": out["evals"], "final_id": final and final["experiment_id"],
            "final_score": final and final["score"], "final_gain_over_start": final and final["gain_over_start"],
            "final_gain_over_live": final and final["gain_over_live"],
            "best_id": best["experiment_id"] if best else c.start,
            "best_score": best["score"] if best else start_scores[c.start],
            "best_gain_over_start": best["gain_over_start"] if best else 0.0,
            "best_gain_over_live": (best["score"] if best else start_scores[c.start]) - best_known,
            "min_exact_seen_minus_start": min((r["score"] for r in mine if r["score"] is not None),
                                              default=start_scores[c.start]) - start_scores[c.start],
            "accepted": out["accepted"][i], "rejected": out["rejected"][i], "restarts": out["restarts"][i]})
    summary = {"run_id": run_id, "finished_at": _now(), "env": env, "n": n, "fft_len": m,
               "evals_planned": total, "evals_done": out["evals"], "stopped_early": out["stopped_early"],
               "loop_s": loop_s, "wall_s": time.monotonic() - t0,
               "s_per_batched_eval": loop_s / max(1, out["evals"]), "batch": len(chains),
               "peak_gpu_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
               "best_known_score": best_known, "records": len(records), "table": table}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    commit()
    return summary
