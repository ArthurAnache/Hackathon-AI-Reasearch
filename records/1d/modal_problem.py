"""Modal CPU: many independent runs of the old loop (tuned optimizer) on any problem of the loop, one seed per container.

Same as modal_seeds.py arm A, with --problem (1a, 1b, 1d, ...) and --maximize for maximize problems. No GPU, no secrets, no network use inside the container.
Safety: retries=0, hard timeout per container, at most MAX_CONTAINERS containers, seeds capped at 200 per launch.

    python -m modal run modal_problem.py --problem 1b --seed-start 100 --n-seeds 100 --nexp 10 --budget 45 --out modal_results_1b
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SRC = HERE / "modal_src"          # clean copy of the checkout (no .git, .env, caches, old runs) + ideas_clean.jsonl
MAX_CONTAINERS = 100

app = modal.App("revolutionaries-loop-problems")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy==2.5.4")
         .add_local_dir(str(SRC), "/src"))


@app.function(image=image, cpu=2.0, memory=2048, timeout=3600, retries=0, max_containers=MAX_CONTAINERS)
def run(seed: int, nexp: int, budget: int, problem: str = "1d", maximize: bool = False) -> str:
    import glob
    import os
    import shutil
    import subprocess
    import time

    import numpy as np

    work = f"/tmp/work_{seed}"
    shutil.copytree("/src", work)
    env = dict(os.environ, RESEARCH_MAP_IDEAS=f"{work}/ideas_clean.jsonl")
    t0 = time.time()
    mode = ["--optimizer", f"optimizers/{problem}_tuned.py"]
    p = subprocess.run(
        ["python", "-m", "integrations.run_loop", "--problem", problem, *mode, "--seed", str(seed),
         "-n", str(nexp), "--budget", str(budget), "--store", "exp.jsonl"],
        cwd=work, env=env, capture_output=True, text=True, timeout=nexp * budget + 900)
    best = None
    scores = []
    for f in sorted(glob.glob(f"{work}/solutions/{problem}/*.json")):
        d = json.load(open(f))
        s = float(d["score"])
        scores.append([os.path.basename(f), s])
        if best is None or (s > best[0] if maximize else s < best[0]):
            best = (s, d["solution"])
    exp = open(f"{work}/exp.jsonl").read() if os.path.exists(f"{work}/exp.jsonl") else ""
    return json.dumps({
        "seed": seed, "arm": "A", "problem": problem, "exit": p.returncode, "seconds": round(time.time() - t0, 1),
        "best_score": best[0] if best else None, "best_values": best[1] if best else None,
        "scores": scores, "exp_jsonl": exp, "log_tail": (p.stdout + p.stderr)[-3000:]})


@app.local_entrypoint()
def main(seed_start: int = 100, n_seeds: int = 2, nexp: int = 2, budget: int = 10, out: str = "modal_results", problem: str = "1b", maximize: bool = False):
    if n_seeds > 200 or nexp * budget > 2400:
        raise SystemExit("refused: more than 200 seeds or more than 40 minutes per seed")
    outdir = HERE / out
    outdir.mkdir(exist_ok=True)
    seeds = list(range(seed_start, seed_start + n_seeds))
    done = []
    for raw in run.map(seeds, kwargs={"nexp": nexp, "budget": budget, "problem": problem, "maximize": maximize}, order_outputs=False, return_exceptions=True):
        if isinstance(raw, Exception):
            print("FAILED", repr(raw)[:300], flush=True)
            continue
        r = json.loads(raw)
        (outdir / f"seed_{r['seed']}.json").write_text(raw)
        done.append(((-r["best_score"] if maximize else r["best_score"]) if r["best_score"] is not None else 9.0, r["seed"], r["exit"], r["seconds"]))
        print(f"seed {r['seed']} exit {r['exit']} {r['seconds']}s best {r['best_score']}", flush=True)
    done.sort()
    print(f"\n{len(done)}/{len(seeds)} seeds returned. Top 10:")
    for s, seed, ex, sec in done[:10]:
        print(f"  seed {seed}: {(-s if maximize else s)!r}")
