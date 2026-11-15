"""ENGINE D — Independent comb families (not incumbent perturbations).

Build a fresh multi-scale comb from scratch at working resolution N_WORK,
polish, optionally upsample toward the start length. Competitive only if
kit.score exceeds the warm-start; otherwise harness keeps the start.
Explores basins the incumbent topology cannot reach via local edits.
"""
import time

import numpy as np
from scipy.interpolate import interp1d

from algolab import kit

PARAMS = {
    "N_WORK": 131072,
    "N_TEETH": 256,
    "TOOTH_LEN": 24,
    "HIER_LEVELS": 3,
    "POLISH_SWAPS": 600,
    "FIRST_IMPROVE_FRAC": 0.4,
    "SIGMA_MAX": 2e-2,
    "SIGMA_MIN": 1e-4,
    "SIGMA_STEPS": 6,
}


def _upsample(f, n2):
    f = np.asarray(f, dtype=np.float64).ravel()
    if n2 == f.size:
        return f.copy()
    x = np.linspace(0.0, 1.0, f.size, endpoint=False)
    xi = np.linspace(0.0, 1.0, n2, endpoint=False)
    return np.maximum(interp1d(x, f, kind="linear", bounds_error=False,
                               fill_value=(f[0], f[-1]))(xi), 0.0)


def _make_comb(n, n_teeth, tooth_len, rng, hier=3):
    out = np.zeros(n, dtype=np.float64)
    for level in range(hier):
        nt = max(4, n_teeth // (2 ** level))
        spacing = n / nt
        amp = 1.0 / (1.5 ** level)
        phase = rng.uniform(0, spacing)
        for k in range(nt):
            c = int(phase + k * spacing) % n
            a = max(0, c - tooth_len // 2)
            b = min(n, a + tooth_len)
            if b <= a:
                continue
            w = np.maximum(np.minimum(np.linspace(0, 1, b - a, endpoint=False), 0.5), 1e-12)
            out[a:b] += amp * w / w.sum()
        tooth_len = max(3, tooth_len // 2)
    # nonuniform jitter of 10% of teeth
    thr = out.max() * 1e-8
    idx = np.flatnonzero(out > thr)
    if idx.size:
        for _ in range(max(1, n_teeth // 10)):
            i = int(rng.choice(idx))
            j = int(rng.integers(0, n))
            out[j] += out[i] * 0.5
    s = out.sum()
    if s > 0:
        out /= out.max()
    return out


class Algorithm:
    def __init__(self, params, rng):
        self.p = dict(PARAMS)
        self.p.update(params or {})
        self.rng = rng

    def perturb(self, f, intensity):
        n_target = len(f)
        n_work = min(n_target, int(self.p["N_WORK"]))
        g = _make_comb(n_work,
                       int(self.p["N_TEETH"] * self.rng.uniform(0.5, 1.5)),
                       int(self.p["TOOTH_LEN"] * self.rng.uniform(0.5, 1.5)),
                       self.rng,
                       hier=int(self.p["HIER_LEVELS"]))
        # mild blend with downsampled incumbent so score is not hopeless
        from math import gcd  # noqa: kept simple
        idx = np.linspace(0, n_target, n_work, endpoint=False).astype(int)
        inc = np.asarray(f, dtype=np.float64)[idx]
        if inc.max() > 0:
            inc = inc / inc.max()
        blend = 0.35 * (1.0 - 0.5 * float(intensity))
        g = (1 - blend) * g + blend * inc
        return _upsample(g / max(g.max(), 1e-15), n_target)

    def improve(self, f, deadline):
        f0 = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
        if f0.max() > 0:
            f0 = f0 / f0.max()
        n = f0.size
        n_work = min(n, int(self.p["N_WORK"]))
        best = f0.copy()
        best_s = kit.score(best)

        # try several independent families at working res
        for fam in range(4):
            if time.monotonic() >= deadline - 0.5:
                break
            gw = _make_comb(n_work,
                            int(self.p["N_TEETH"] * self.rng.uniform(0.4, 1.8)),
                            int(max(4, self.p["TOOTH_LEN"] * self.rng.uniform(0.4, 2.0))),
                            self.rng,
                            hier=int(self.p["HIER_LEVELS"]))
            idx = np.linspace(0, n, n_work, endpoint=False).astype(int)
            inc = f0[idx]
            gw = 0.55 * gw + 0.45 * (inc / max(inc.max(), 1e-15))
            # polish at working res
            thr = gw.max() * 1e-8
            supp = np.flatnonzero(gw > thr)
            gs = kit.score(gw)
            swaps = int(self.p["POLISH_SWAPS"])
            t_end = min(deadline, time.monotonic() + max(2.0, (deadline - time.monotonic()) / 4))
            for _ in range(swaps):
                if time.monotonic() >= t_end - 0.1 or supp.size < 2:
                    break
                i, j = self.rng.choice(supp, size=2, replace=False)
                step = 0.03 * gw[i]
                if step <= 0:
                    continue
                trial = gw.copy()
                trial[i] -= step
                trial[j] += step
                trial = np.maximum(trial, 0.0)
                s = kit.score(trial)
                if s > gs:
                    gw, gs = trial, s
                    supp = np.flatnonzero(gw > thr)
            cand = _upsample(gw, n)
            cs = kit.score(cand)
            # short fine polish
            thr = cand.max() * 1e-8
            supp = np.flatnonzero(cand > thr)
            t_end = min(deadline, time.monotonic() + 1.5)
            while time.monotonic() < t_end - 0.1 and supp.size >= 2:
                i, j = self.rng.choice(supp, size=2, replace=False)
                step = 0.02 * cand[i]
                if step <= 0:
                    continue
                trial = cand.copy()
                trial[i] -= step
                trial[j] += step
                trial = np.maximum(trial, 0.0)
                s = kit.score(trial)
                if s > cs:
                    cand, cs = trial, s
                    supp = np.flatnonzero(cand > thr)
            if cs > best_s:
                best, best_s = cand, cs
        return best
