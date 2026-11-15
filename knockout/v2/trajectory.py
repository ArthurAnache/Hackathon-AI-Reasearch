"""How does the leader move? Compare the best public solutions of second-autocorrelation-inequality.

    .venv/bin/python -m knockout.v2.trajectory --fetch DIR      # read-only GET, saves DIR/sol<id>.npy
    .venv/bin/python -m knockout.v2.trajectory DIR              # analysis of the files already in DIR

`--fetch` calls `integrations.einstein_arena.get_best_solutions(3, limit=5)` (public read, logged to
selection/arena_log.jsonl). On 2026-11-14 it returned solutions 2783, 2782, 2781, 2780 (MLAI_YONSEI, all of that
day) and 2416 (ClaudeExplorer, 2026-07-22), all with n = 1,999,999.

For each pair of successive solutions the script prints: the mass-normalized L1 distance, the cosine between
successive displacements, cell by cell and for block masses (blocks of B cells), and the mass share of a few
regions of the grid. Scores are closed-form values (steering only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from knockout.v2.objective import Oracle

BLOCKS = (1, 16, 256, 4096, 16384, 65536)
REGIONS = (0.0, 0.05, 0.20, 0.35, 0.50, 0.65, 0.90, 0.95, 1.0)


def fetch(out_dir: Path) -> None:
    from integrations import einstein_arena as ea

    out_dir.mkdir(parents=True, exist_ok=True)
    for s in ea.get_best_solutions(3, limit=5):
        np.save(out_dir / f"sol{s['id']}.npy", np.asarray(s["data"]["values"], dtype=np.float64))
        print(f"solution {s['id']} by {s.get('agentName')} at {s.get('createdAt')}: displayed score {s.get('score')}")


def block_mass(f: np.ndarray, block: int) -> np.ndarray:
    k = len(f) // block
    return f[: k * block].reshape(k, block).sum(1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def analyse(folder: Path) -> None:
    ids = sorted(int(p.stem[3:]) for p in folder.glob("sol*.npy"))
    sol = {}
    for i in ids:
        v = np.load(folder / f"sol{i}.npy")
        sol[i] = v / v.sum()
    if len({len(v) for v in sol.values()}) != 1:
        raise SystemExit("the solutions do not share one grid")
    n = len(sol[ids[0]])
    oracle = Oracle(n)
    scores = {i: oracle.exact(sol[i]) for i in ids}
    steps = list(zip(ids[:-1], ids[1:]))
    print("| step | score gain | L1 distance | cells lost | cells new |\n|---|---|---|---|---|")
    for a, b in steps:
        lost = int(((sol[a] > 0) & (sol[b] == 0)).sum())
        new = int(((sol[a] == 0) & (sol[b] > 0)).sum())
        print(f"| {a} -> {b} | {scores[b] - scores[a]:+.3e} | {np.abs(sol[b] - sol[a]).sum():.4f} | {lost} | {new} |")
    print("\n| block size | L1 distance of block masses per step | cosine between successive steps |\n|---|---|---|")
    for block in BLOCKS:
        d = [block_mass(sol[b], block) - block_mass(sol[a], block) for a, b in steps]
        dist = ", ".join(f"{np.abs(x).sum():.5f}" for x in d)
        cos = ", ".join(f"{cosine(x, y):+.3f}" for x, y in zip(d[:-1], d[1:]))
        print(f"| {block} | {dist} | {cos} |")
    print("\n| region of the grid | " + " | ".join(f"mass share in {i}" for i in ids) + " |\n|---|" + "---|" * len(ids))
    for lo, hi in zip(REGIONS[:-1], REGIONS[1:]):
        shares = " | ".join(f"{sol[i][int(lo * n):int(hi * n)].sum():.5f}" for i in ids)
        print(f"| {lo:.2f} to {hi:.2f} | {shares} |")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--fetch"]
    if "--fetch" in sys.argv:
        fetch(Path(args[0]))
    analyse(Path(args[0]))
