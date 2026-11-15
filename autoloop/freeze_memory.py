"""Freeze a memory once: copy the solution files, import them into a memory ledger (duplicates dropped), and record
the best EXACT score it contains (c1d_round + c1d_exact on the best few by float score).

    python -m autoloop.freeze_memory --out autoloop/runs/C_memory --src DIR [--src DIR ...] [--exact-top 6]

After the freeze nothing reads the source directories again: every arm starts from OUT/ledger.jsonl.
"""
import argparse
import numpy as np
import shutil
import time
from pathlib import Path

from . import problem
from .ledger import Ledger, cert_record, import_memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--exact-top", type=int, default=6); ap.add_argument("--problem", default="tao-1d")
    a = ap.parse_args()
    P = problem.load(a.problem)
    out = Path(a.out); files = out / "files"; files.mkdir(parents=True, exist_ok=False)
    for i, s in enumerate(a.src):
        shutil.copytree(s, files / f"src{i}_{Path(s).name}",
                        ignore=shutil.ignore_patterns("*.log", "*.ckpt.json", "*.tmp.npy", "__pycache__"))
    L = Ledger(out / "ledger.jsonl")
    L.append(dict(kind="note", text=f"memory frozen at {time.strftime('%Y-%m-%d %H:%M:%S %Z')} from {a.src}"))
    tmp = Ledger(out / "all_files.jsonl")
    import_memory(tmp, str(files), P)
    # drop duplicates (same vector as .npy and as rounded _int.json): same cells and same score to 1e-12
    kept = []
    for c in sorted(tmp.certs(), key=lambda c: c["score"]):
        if any(k["cells"] == c["cells"] and abs(k["score"] - c["score"]) <= 1e-12 * c["score"] for k in kept):
            continue
        kept.append(c)
    for c in kept:
        L.append(c)
    ex = P.LocalExecutor(out)
    best_exact = None
    for c in kept[: a.exact_top]:
        r, cpu = ex.certify(c["path"], f"mem_{c['id'][:8]}")
        if not r["ok"]:
            L.append(dict(kind="note", text=f"exact check failed for {c['id'][:10]}: {r['error'][:200]}")); continue
        f = r["exact"]
        cid = P.vec_id(np.asarray(r["ints"], dtype=np.float64))
        L.append(cert_record(problem=P.NAME, cid=cid, cells=c["cells"], score_=float(f), parent=c["id"], root=c["id"],
                             stage="certified", params=dict(bits=56, knot=r["knot"], memory_baseline=True), cpu_s=cpu,
                             path=r["path"], exact=f"{f.numerator}/{f.denominator}", origin="imported"))
        if best_exact is None or f < best_exact[0]:
            best_exact = (f, c)
        print(f"exact {float(f):.16f}  float {c['score']:.16f}  {c['cells']} cells  {Path(c['path']).name}", flush=True)
    f, c = best_exact
    L.append(dict(kind="memory_baseline", best_exact=f"{f.numerator}/{f.denominator}", best_exact_float=float(f),
                  cert=c["id"], cells=c["cells"], file=c["path"], n_certificates=len(kept),
                  n_files=len(tmp.certs()), sources=a.src))
    (out / "all_files.jsonl").unlink()
    print(f"memory: {len(kept)} certificates; best exact {float(f):.16f} ({c['cells']} cells)")


if __name__ == "__main__":
    main()
