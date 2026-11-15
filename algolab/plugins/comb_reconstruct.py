"""ENGINE A — Comb reconstruction at coarse grid, then verified lift.

Full-resolution rebuilds are rejected by the harness (score must beat start).
Strategy: downsample → birth/death/relocate teeth → recover with many cheap
evals → upsample → short fine polish. Only returns arrays that can beat start
after recovery; otherwise returns the best recovered attempt (harness gates).
"""
import math
import time

import numpy as np
from scipy.interpolate import interp1d

from algolab import kit

PARAMS = {
    "COARSE_N": 16384,
    "N_TEETH_TARGET": 0,
    "BIRTH_RATE": 0.20,
    "DEATH_RATE": 0.15,
    "RELOCATE_RATE": 0.35,
    "CLUSTER_RATE": 0.20,
    "SPACING_JITTER": 0.12,
    "MASS_REDIST_FRAC": 0.5,
    "MIN_TOOTH_LEN": 2,
    "MAX_TOOTH_LEN": 48,
    "REBUILD_TRIES": 6,
    "COORD_SWAPS": 400,
    "FIRST_IMPROVE_FRAC": 0.25,
    "SIGMA_MAX": 2e-2,
    "SIGMA_MIN": 1e-4,
    "SIGMA_STEPS": 8,
}


def _teeth(f, thr_rel=1e-8):
    f = np.asarray(f, dtype=np.float64)
    thr = float(f.max()) * thr_rel
    active = f > thr
    teeth, i, n = [], 0, len(f)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        teeth.append((i, j))
        i = j
    return teeth


def _place_tooth(out, start, length, mass, profile="tri"):
    n = len(out)
    a = max(0, start)
    b = min(n, start + length)
    if b <= a or mass <= 0:
        return
    L = b - a
    if profile == "box":
        w = np.ones(L)
    else:
        x = np.linspace(0, 1, L, endpoint=False)
        w = np.maximum(np.minimum(x, 1 - x), 1e-12)
    out[a:b] += w / w.sum() * mass


def _block_mean(f, n2):
    f = np.asarray(f, dtype=np.float64).ravel()
    if n2 >= f.size:
        return f.copy()
    idx = np.linspace(0, f.size, n2 + 1).astype(int)
    return np.array([f[idx[i]:idx[i + 1]].mean() if idx[i + 1] > idx[i] else 0.0 for i in range(n2)])


def _upsample(f, n2):
    f = np.asarray(f, dtype=np.float64).ravel()
    if n2 == f.size:
        return f.copy()
    x = np.linspace(0.0, 1.0, f.size, endpoint=False)
    xi = np.linspace(0.0, 1.0, n2, endpoint=False)
    g = interp1d(x, f, kind="linear", bounds_error=False, fill_value=(f[0], f[-1]))(xi)
    return np.maximum(g, 0.0)


class Algorithm:
    def __init__(self, params, rng):
        self.p = dict(PARAMS)
        self.p.update(params or {})
        self.rng = rng

    def _rebuild(self, f):
        f = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
        n = f.size
        teeth = _teeth(f)
        masses = [float(f[a:b].sum()) for a, b in teeth]
        total = float(f.sum()) or 1.0
        target_n = int(self.p.get("N_TEETH_TARGET") or 0)
        if target_n <= 0:
            jitter = self.rng.uniform(0.7, 1.3)
            target_n = max(8, int(len(teeth) * jitter)) if teeth else max(8, n // 200)

        keep = []
        for (a, b), m in zip(teeth, masses):
            if self.rng.random() < float(self.p["DEATH_RATE"]) and len(teeth) > 4:
                continue
            keep.append((a, b, m))
        if not keep and teeth:
            a, b = teeth[0]
            keep = [(a, b, masses[0])]

        n_birth = int(math.ceil(target_n * float(self.p["BIRTH_RATE"])))
        for _ in range(n_birth):
            length = int(self.rng.integers(int(self.p["MIN_TOOTH_LEN"]), int(self.p["MAX_TOOTH_LEN"]) + 1))
            start = int(self.rng.integers(0, max(1, n - length)))
            mass = total / max(target_n, 1) * self.rng.uniform(0.3, 1.7)
            keep.append((start, start + length, mass))

        rebuilt = []
        for a, b, m in keep:
            length = max(1, b - a)
            if self.rng.random() < float(self.p["RELOCATE_RATE"]):
                start = int(self.rng.integers(0, max(1, n - length)))
            else:
                shift = int(self.rng.normal(0, float(self.p["SPACING_JITTER"]) * n / max(len(keep), 1)))
                start = int(np.clip(a + shift, 0, max(0, n - length)))
            if self.rng.random() < float(self.p["CLUSTER_RATE"]) and rebuilt:
                ref = rebuilt[int(self.rng.integers(0, len(rebuilt)))]
                start = int(np.clip(ref[0] + int(self.rng.integers(-3 * length, 3 * length + 1)),
                                    0, max(0, n - length)))
            rebuilt.append((start, start + length, m))

        masses2 = np.array([m for _, _, m in rebuilt], dtype=np.float64)
        if masses2.size and self.rng.random() < float(self.p["MASS_REDIST_FRAC"]):
            k = max(2, masses2.size // 3)
            idx = self.rng.choice(masses2.size, size=min(k, masses2.size), replace=False)
            pool = float(masses2[idx].sum())
            masses2[idx] = self.rng.dirichlet(np.ones(len(idx))) * pool
        for i, (a, b, _) in enumerate(rebuilt):
            rebuilt[i] = (a, b, float(masses2[i]))

        out = np.zeros(n, dtype=np.float64)
        for a, b, m in rebuilt:
            _place_tooth(out, a, b - a, m, profile="tri" if self.rng.random() < 0.7 else "box")
        if out.sum() <= 0:
            return f.copy()
        out *= total / out.sum()
        return out

    def _coord_polish(self, f, deadline, swaps):
        best = np.maximum(f, 0.0)
        best_s = kit.score(best)
        thr = best.max() * 1e-8
        support = np.flatnonzero(best > thr)
        for _ in range(swaps):
            if time.monotonic() >= deadline - 0.2 or support.size < 2:
                break
            i, j = self.rng.choice(support, size=2, replace=False)
            step = 0.03 * best[i]
            if step <= 0:
                continue
            trial = best.copy()
            trial[i] -= step
            trial[j] += step
            trial = np.maximum(trial, 0.0)
            s = kit.score(trial)
            if s > best_s:
                best, best_s = trial, s
                support = np.flatnonzero(best > thr)
        return best, best_s

    def perturb(self, f, intensity):
        f = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
        n = f.size
        n_c = min(n, int(self.p["COARSE_N"]))
        g = _block_mean(f, n_c)
        reps = 1 + int(4 * float(intensity))
        for _ in range(reps):
            g = self._rebuild(g)
        return _upsample(g, n)

    def improve(self, f, deadline):
        f0 = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
        if f0.max() > 0:
            f0 = f0 / f0.max()
        n = f0.size
        n_c = min(n, int(self.p["COARSE_N"]))
        start_s = kit.score(f0)
        best = f0.copy()
        best_s = start_s
        fc0 = _block_mean(f0, n_c)
        tries = int(self.p["REBUILD_TRIES"])
        swaps = int(self.p["COORD_SWAPS"])

        for t in range(tries):
            if time.monotonic() >= deadline - 0.5:
                break
            gc = self._rebuild(fc0 if t == 0 else _block_mean(best, n_c))
            # always spend recovery time on the NEW support (even if score dropped)
            gc, _ = self._coord_polish(gc, min(deadline, time.monotonic() + max(2.0, (deadline - time.monotonic()) / max(tries - t, 1))),
                                      swaps)
            cand = _upsample(gc, n)
            # brief fine polish
            cand, cs = self._coord_polish(cand, min(deadline, time.monotonic() + 1.5), max(40, swaps // 10))
            if cs > best_s:
                best, best_s = cand, cs
            elif t == 0 and cs > best_s - 1e-4:
                # keep first structural lineage for further refine even if slightly worse;
                # harness still requires final > start, so continue recovering
                refined, rs = self._coord_polish(cand, deadline, swaps * 2)
                if rs > best_s:
                    best, best_s = refined, rs
        return best
