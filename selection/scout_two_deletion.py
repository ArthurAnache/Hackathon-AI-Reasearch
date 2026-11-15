"""Problem scout probe on two-deletion-code-16: is the live #1 code locally improvable by simple moves?

Usage (from the repo root, after `selection/scout_refresh.py`):

    .venv/bin/python selection/scout_two_deletion.py --budget 300 --seed 0 --tag NAME

A code is a set of 16-bit words whose two-deletion balls (the 14-bit words left after deleting two positions) are
pairwise disjoint; the score is the number of words (maximize, delta = 1). This is a maximum independent set in a
graph on 65,536 words. The probe starts from the live #1 code and runs the classic independent-set local search:

1. add any word that conflicts with no codeword (is the code maximal?);
2. (1,2)-swap: remove one codeword and add two words that conflicted only with it;
3. otherwise a batch of random (1,1)-swaps (plateau moves), then back to 1.

It reports the best size reached. The best code is re-scored in a clean process with the verifier served by the
live API (`selection/eval_runner.py`). Nothing is submitted. One JSON line per run goes to
selection/scout_two_deletion_runs.jsonl.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SLUG = "two-deletion-code-16"
CACHE = ROOT / ".cache" / "scout" / SLUG
OUT = ROOT / "scout_two_deletion_runs.jsonl"
N = 16


def balls() -> np.ndarray:
    """D[w] = the 120 two-deletion descendants of word w (as 14-bit integers, with repeats)."""
    words = np.arange(1 << N, dtype=np.uint32)
    bits = ((words[:, None] >> (N - 1 - np.arange(N))) & 1).astype(np.int64)
    weights = 1 << (N - 3 - np.arange(N - 2))
    cols = []
    for i in range(N - 1):
        for j in range(i + 1, N):
            keep = [k for k in range(N) if k not in (i, j)]
            cols.append(bits[:, keep] @ weights)
    return np.stack(cols, axis=1).astype(np.int32)


def verify_clean(path: Path) -> dict:
    source = json.loads((CACHE / "problem.json").read_text())["verifier"]
    verifier = CACHE / "verifier_live.py"
    verifier.write_text(source)
    proc = subprocess.run([sys.executable, "-I", str(ROOT / "eval_runner.py"), str(verifier), str(path), "1", "60"],
                          capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    out = json.loads(lines[-1]) if lines else {"ok": False, "error": proc.stderr[-300:]}
    out["verifier_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    stage1 = ROOT / "verifiers" / f"{SLUG}.py"
    out["equals_stage1_extraction"] = stage1.exists() and stage1.read_text().strip() == source.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--plateau-batch", type=int, default=6)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    start_path = CACHE / "record_solution.json"
    v0 = verify_clean(start_path)
    code = [int(w, 2) for w in json.loads(start_path.read_text())["words"]]
    D = balls()
    masks: dict[int, int] = {}

    def mask(w: int) -> int:
        if w not in masks:
            m = 0
            for d in set(D[w].tolist()):
                m |= 1 << d
            masks[w] = m
        return masks[w]

    owner = np.full(1 << (N - 2), -1, dtype=np.int32)
    in_code = np.zeros(1 << N, dtype=bool)
    for w in code:
        assert (owner[D[w]] == -1).all() or set(owner[D[w]].tolist()) <= {-1, w}, "start code is not valid"
        owner[D[w]] = w
        in_code[w] = True
    start_size = best_size = int(in_code.sum())
    best_code = sorted(np.flatnonzero(in_code).tolist())
    t0 = time.monotonic()
    rounds = added = swaps12 = plateau_moves = 0
    first = None
    while time.monotonic() - t0 < args.budget:
        rounds += 1
        own = owner[D]
        own.sort(axis=1)
        tight = ((own[:, 1:] != own[:, :-1]) & (own[:, 1:] >= 0)).sum(axis=1) + (own[:, 0] >= 0)
        tight[in_code] = 99
        only = own[:, -1]
        free = np.flatnonzero(tight == 0)
        one = np.flatnonzero(tight == 1)
        if first is None:
            first = {"free_words": int(len(free)), "one_tight_words": int(len(one))}
        improved = False
        for w in rng.permutation(free).tolist():  # 1. free words
            if (owner[D[w]] == -1).all():
                owner[D[w]] = w
                in_code[w] = True
                added += 1
                improved = True
        if not improved:  # 2. (1,2)-swaps
            groups: dict[int, list[int]] = {}
            for w, u in zip(one.tolist(), only[one].tolist()):
                groups.setdefault(u, []).append(w)
            if first is not None and "pairs_checked" not in first:
                first["codewords_with_one_tight_neighbours"] = len(groups)
                first["pairs_checked"] = int(sum(len(g) * (len(g) - 1) // 2 for g in groups.values()))
            for u in rng.permutation(list(groups)).tolist():
                cand = groups[u]
                hit = next(((a, b) for i, a in enumerate(cand) for b in cand[i + 1:] if not mask(a) & mask(b)), None)
                if hit:
                    owner[D[u]] = -1
                    in_code[u] = False
                    for w in hit:
                        owner[D[w]] = w
                        in_code[w] = True
                    swaps12 += 1
                    improved = True
                    break
            if not improved:  # 3. plateau: a batch of independent (1,1)-swaps
                used = 0
                for u in rng.permutation(list(groups)).tolist()[: args.plateau_batch]:
                    w = int(rng.choice(groups[u]))
                    if mask(w) & used:
                        continue
                    used |= mask(w)
                    owner[D[u]] = -1
                    in_code[u] = False
                    if (owner[D[w]] == -1).all():
                        owner[D[w]] = w
                        in_code[w] = True
                        plateau_moves += 1
                    else:  # w conflicts with a word added earlier in this batch: undo
                        owner[D[u]] = u
                        in_code[u] = True
        size = int(in_code.sum())
        if size > best_size:
            best_size, best_code = size, sorted(np.flatnonzero(in_code).tolist())
            print(f"t={time.monotonic() - t0:.0f}s size {size}", flush=True)

    run_dir = CACHE / "runs" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.json"
    best_path.write_text(json.dumps({"words": [format(w, "016b") for w in best_code]}))
    v1 = verify_clean(best_path)
    row = {"experiment_id": f"scout-two-deletion/{args.tag}", "slug": SLUG,
           "checked_on": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "hypothesis": "the live #1 code is improvable by add / (1,2)-swap / plateau local search in minutes",
           "seed": args.seed, "budget_s": args.budget, "wall_s": round(time.monotonic() - t0, 1),
           "device": "local CPU, 1 process", "start_score_pinned": v0.get("score"), "score": v1.get("score"),
           "valid": bool(v1.get("ok")), "gain": (v1.get("score") or 0) - (v0.get("score") or 0), "delta": 1,
           "rounds": rounds, "free_words_added": added, "swaps_1_2": swaps12, "plateau_moves": plateau_moves,
           "at_start": first, "verifier_sha256": v1.get("verifier_sha256"),
           "verifier_equals_stage1_extraction": v1.get("equals_stage1_extraction"),
           "solution_path": str(best_path.relative_to(ROOT.parent)),
           "label": "verified: sizes recomputed in a clean process with the live verifier"}
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in ("start_score_pinned", "score", "gain", "rounds", "free_words_added",
                                          "swaps_1_2", "plateau_moves", "at_start", "wall_s",
                                          "verifier_equals_stage1_extraction")}))


if __name__ == "__main__":
    main()
