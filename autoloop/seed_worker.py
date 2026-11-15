"""One seed run: the WS2 optimizer wt-compare/optimizers/1d_tuned.py, search(budget_s, seed, config), from noise.

    python -m autoloop.seed_worker --cells 800 --budget 60 --seed 3 --out PREFIX   ->  PREFIX.npy, PREFIX.json
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WORKSPACE = __import__("os").environ.get("AUTOLOOP_WORKSPACE", str((lambda q: q[2] if len(q) > 2 else q[-1])(__import__("pathlib").Path(__file__).resolve().parents)))  # folder that holds repo/, wt-compare/, shared/ and .env
WS2 = Path(os.environ.get("AUTOLOOP_WS2", WORKSPACE + "/wt-compare"))   # Modal: /root/ws2


def load_optimizer():
    sys.path[:0] = [str(WS2), str(WS2 / "optimizers")]   # 1d_tuned imports `prepare` from wt-compare
    spec = importlib.util.spec_from_file_location("ws2_1d_tuned", WS2 / "optimizers" / "1d_tuned.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=800); ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    opt = load_optimizer()
    cfg = dict(opt.CONFIG, N_CELLS=a.cells, INIT=None)
    t0 = time.time()
    v = np.asarray(opt.search(a.budget, a.seed, cfg)["values"], dtype=np.float64)
    v = v / np.abs(v).max()
    np.save(a.out + ".npy", v)
    out = dict(n=len(v), best=float(opt.score(v)), seed=a.seed, budget=a.budget, seconds=time.time() - t0, config=cfg)
    json.dump(out, open(a.out + ".json", "w"))
    print(json.dumps(out))
