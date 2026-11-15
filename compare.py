"""Matched head-to-head of optimizers: same incumbent, paired seeds, equal budgets, one batch.

    python compare.py --problem second-autocorrelation-inequality \\
        --optimizers optimizers/second-autocorrelation-inequality_probe.py \\
                     optimizers/second-autocorrelation-inequality_blockshift.py \\
        --seeds 101 202 303 --budget 300 --workers 6 --incumbent-file live1.json --exact --report out.md

Fairness, by construction:
  - every optimizer starts from the SAME incumbent certificate (re-verified when loaded; --incumbent-file
    imports an outside solution, e.g. the live #1, only after the frozen verifier re-scores it);
  - every optimizer runs on the SAME seeds with the SAME budget;
  - all jobs go to the executor in ONE interleaved batch, so every method sees the same machine load;
  - every candidate is re-verified here (frozen verifier), never trusted from the worker.

Output: one record per optimizer in the experiment store (seeds[], scores[] per seed), a certificate for every
candidate better than the incumbent, a paired table (gain per seed, and the per-seed difference between the
methods), and, with --exact, the best candidate re-scored by the problem's independent exact verifier.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import interfaces as I
from certificates import CertificateStore
from execution import LocalExecutor, ModalExecutor, make_job, source_hash


def incumbent(problem, certs, cert_id=None, file=None):
    if file:
        sol = json.loads(Path(file).read_text())
        sol = sol.get("solution", sol)
        return certs.put(problem, sol, problem.score(sol), {"source": f"imported from {file}", "re-verified": True})
    if cert_id:
        return certs.load(problem, cert_id)
    best = certs.best(problem)
    if best is None:
        from problems.catalog import SLUGS, load       # WS1's frozen #1, re-verified by put()
        if problem.id not in SLUGS:
            return None                                # nothing verified yet: matched cold starts
        sol = load(problem.id).certificate()
        best = certs.put(problem, sol, problem.score(sol), {"source": "WS1 frozen certificate"})
    return best


def gain(problem, new, old):
    return (new - old) if problem.direction == "maximize" else (old - new)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--optimizers", nargs="+", required=True)
    ap.add_argument("--configs", nargs="*", default=None, help="one JSON config per optimizer (default {})")
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--executor", choices=["local", "modal"], default="local")
    ap.add_argument("--incumbent", default=None, help="certificate id to start from (default: best in --certs)")
    ap.add_argument("--incumbent-file", default=None, help="outside solution JSON; re-verified, then stored")
    ap.add_argument("--store", default=str(I.ROOT / "experiments.jsonl"))
    ap.add_argument("--certs", default=str(I.ROOT / "certificates"))
    ap.add_argument("--exact", action="store_true", help="re-score the best candidate with the exact verifier")
    ap.add_argument("--report", default=None, help="write the paired table as Markdown")
    a = ap.parse_args(argv)
    if len(a.seeds) < 3:
        ap.error("a fair comparison needs at least 3 paired seeds")

    problem, store, certs = I.get_problem(a.problem), I.JsonlStore(a.store), CertificateStore(a.certs)
    inc = incumbent(problem, certs, a.incumbent, a.incumbent_file)
    configs = [json.loads(c) for c in a.configs] if a.configs else [{} for _ in a.optimizers]
    delta = getattr(problem._m, "DELTA", 0.0) or 0.0
    inc_id = inc.id if inc else None
    ref = inc.score if inc else problem.best_known          # gains vs the incumbent, else vs the record
    ref_name = f"incumbent {inc.id} = {inc.score!r}" if inc else "no incumbent (cold starts; gains are vs the record)"
    print(f"{problem.id}: {ref_name}; record {problem.best_known!r} "
          f"({problem.best_known_source}); a record needs >= {delta or 'tolerance'} over it")
    print(f"{len(a.optimizers)} optimizers x seeds {a.seeds} x {a.budget}s, one interleaved batch")

    jobs = [make_job(problem, opt, cfg, seed, a.budget, inc_id, certs, job_id=f"{k}:{seed}")
            for seed in a.seeds for k, (opt, cfg) in enumerate(zip(a.optimizers, configs))]
    executor = ModalExecutor() if a.executor == "modal" else LocalExecutor(a.workers or len(jobs))
    t0 = time.monotonic()
    results = executor.map(jobs)
    wall = time.monotonic() - t0

    per = {k: {} for k in range(len(a.optimizers))}            # optimizer -> seed -> (score | None, error)
    best = (None, None, None)
    for job, r in zip(jobs, results):
        k, seed = int(job.job_id.split(":")[0]), int(job.job_id.split(":")[1])
        if r.status != "ok":
            per[k][seed] = (None, f"{r.status}: {(r.error or '')[:200]}")
            continue
        why = problem.why_invalid(r.candidate)                    # trusted re-verification
        if why:
            per[k][seed] = (None, f"invalid: {why}")
            continue
        s = problem.score(r.candidate)
        per[k][seed] = (s, None)
        if best[0] is None or problem.better(s, best[0]):
            best = (s, r.candidate, k)

    cert_best = None
    if best[0] is not None and (inc is None or problem.better(best[0], inc.score)):
        cert_best = certs.put(problem, best[1], best[0], {"compare": Path(a.optimizers[best[2]]).stem,
                                                           "parent": inc_id, "budget_s": a.budget})
    start = len(store.read())
    rows = {}
    for k, opt in enumerate(a.optimizers):
        scores = [per[k][s][0] for s in a.seeds]
        ok = [x for x in scores if x is not None]
        errors = [f"seed {s}: {per[k][s][1]}" for s in a.seeds if per[k][s][1]]
        gains = [gain(problem, x, ref) if x is not None else None for x in scores]
        rows[k] = gains
        top = None
        if ok:
            top = max(ok) if problem.direction == "maximize" else min(ok)
        store.append(I.new_record(
            experiment_id=f"cmp_{start + k:05d}", problem=problem.id, agent="compare", arm=f"compare:{Path(opt).stem}",
            optimizer=opt, optimizer_hash=source_hash(Path(opt).read_text()), source_sha=I.source_sha(),
            parent_id=inc_id, hypothesis=f"{Path(opt).stem} improves the incumbent within {a.budget}s",
            rationale=f"matched comparison: same incumbent {inc_id}, paired seeds {a.seeds}, equal budget",
            config=configs[k], warm_start=inc_id, seeds=list(a.seeds), budget_s=a.budget, scores=scores,
            score=statistics.fmean(ok) if len(ok) == len(scores) else None, valid=len(ok) == len(scores),
            status=("keep" if ok and all(g is not None and g > 0 for g in gains) else "discard") if not errors else "crash",
            best_before=inc.score if inc else None, beats_record=bool(top is not None and problem.beats_record(top)),
            duration_s=round(wall, 1), error="; ".join(errors) or None,
            certificate_path=cert_best.id if cert_best and best[2] == k else None))

    lines = [f"| seed | " + " | ".join(Path(o).stem.split("_")[-1] for o in a.optimizers) + " |",
             "|---|" + "---|" * len(a.optimizers)]
    for i, s in enumerate(a.seeds):
        cells = []
        for k in range(len(a.optimizers)):
            g = rows[k][i]
            cells.append(f"{g:+.4e}" if g is not None else f"FAILED ({per[k][s][1][:40]})")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    summary = []
    for k, o in enumerate(a.optimizers):
        g = [x for x in rows[k] if x is not None]
        if g:
            summary.append(f"{Path(o).stem}: mean gain {statistics.fmean(g):+.4e}, best {max(g):+.4e}, "
                           f"improved on {sum(x > 0 for x in g)}/{len(a.seeds)} seeds, failures {len(a.seeds) - len(g)}")
        else:
            summary.append(f"{Path(o).stem}: no valid result")
    if len(a.optimizers) == 2 and all(None not in rows[k] for k in (0, 1)):
        d = [y - x for x, y in zip(rows[0], rows[1])]
        se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        summary.append(f"paired difference ({Path(a.optimizers[1]).stem} - {Path(a.optimizers[0]).stem}): "
                       f"mean {statistics.fmean(d):+.4e} (SE {se:.2e}), second better on {sum(x > 0 for x in d)}/{len(d)} seeds")
    exact_line = None
    if a.exact and cert_best is not None:
        t = time.monotonic()
        fr = problem.certify(cert_best.solution)
        if fr is not None:
            exact_line = (f"exact check of {cert_best.id}: {float(fr)!r} (float verifier {cert_best.score!r}, "
                          f"difference {float(fr) - cert_best.score:+.2e}, {time.monotonic() - t:.0f}s)")
    if cert_best is not None:
        to_record = gain(problem, cert_best.score, problem.best_known)
        summary.append(f"best candidate {cert_best.id} = {cert_best.score!r}: {gain(problem, cert_best.score, ref):+.4e} "
                       f"over the {'incumbent' if inc else 'record'}; {to_record:+.4e} vs the live record (needs >= {delta or 'tolerance'}): "
                       f"{'RECORD (verify with WS1 arbiter)' if problem.beats_record(cert_best.score) else 'not a record'}")
    if exact_line:
        summary.append(exact_line)
    text = "\n".join(lines + [""] + [f"- {x}" for x in summary])
    print("\n" + text + f"\n\nwall {wall:.0f}s")
    if a.report:
        Path(a.report).write_text(f"{ref_name}; seeds {a.seeds}; budget {a.budget}s each; "
                                  f"wall {wall:.0f}s\n\n{text}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
