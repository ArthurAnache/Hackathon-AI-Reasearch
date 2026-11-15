"""TRUSTED control: SAVANT's stock blockshift template (optimizers/second-autocorrelation-inequality_blockshift.py),
wrapped in the plugin interface so that it runs under exactly the same harness, start point and clock as a candidate.
It imports a repository file, so it cannot pass the gate; bench.py runs it with trusted=True.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("blockshift_ctl", _ROOT / "optimizers" /
                                               "second-autocorrelation-inequality_blockshift.py")
_bs = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(_ROOT))
_spec.loader.exec_module(_bs)

PARAMS = dict(_bs.CONFIG)


class Algorithm:
    def __init__(self, params, rng):
        self.cfg = {**_bs.CONFIG, **params}
        self.rng = rng

    def improve(self, f, deadline):
        cfg, rng = self.cfg, self.rng
        f = np.maximum(np.asarray(f, dtype=np.float64), 0.0)
        f = f / f.max()
        best_f, best = f.copy(), _bs.pinned_score(f)
        st = _bs.State(f, float(cfg["GAP_REL"]))
        scale = float(f[f > 0].mean())
        n = st.n
        last = time.monotonic()
        while time.monotonic() < deadline - 1.0:
            b = min(2 ** int(rng.integers(0, int(cfg["BLOCK_LOG2_MAX"]) + 1)), n)
            if rng.random() < cfg["GUIDED"] and len(st.top):
                k = int(rng.choice(st.top))
                a = int(np.clip(rng.integers(max(0, k - n + 1), min(n, k + 1)) - b // 2, 0, n - b))
            else:
                a = int(rng.integers(0, n - b + 1))
            D, v, terms = st.propose(a, b, cfg, scale)
            if v > 1e-16:
                st.apply(a, b, D, terms)
            if time.monotonic() - last > cfg["REFRESH_S"]:
                last = time.monotonic()
                exact = _bs.pinned_score(st.f)
                if exact > best:
                    best_f, best = st.f.copy(), exact
                else:
                    st.f[:] = best_f
                st.rebuild()
        exact = _bs.pinned_score(st.f)
        return st.f.copy() if exact > best else best_f
