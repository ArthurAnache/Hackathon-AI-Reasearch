"""ENGINE A2 — Micro tooth-surgery with cheap accept/reject at full resolution.

Each kit.score on ~2M costs ~0.5–1s. Strategy: many tiny edits, ONE score each;
only if an edit helps (or is within EPS), run a short neighborhood polish.
"""
import time

import numpy as np
from scipy.optimize import minimize

from algolab import kit

PARAMS = {
    "MAX_EDITS": 2,
    "SHIFT_FRAC": 0.0008,
    "BIRTH_LEN": 10,
    "TRIES": 120,
    "EPS_ACCEPT": 0.0,
    "POLISH_ITERS": 8,
    "FIRST_IMPROVE_FRAC": 0.35,
    "SIGMA_MAX": 8e-3,
    "SIGMA_MIN": 1e-5,
    "SIGMA_STEPS": 10,
}


def _teeth(f, thr_rel=1e-8):
    f = np.asarray(f, dtype=np.float64)
    thr = float(f.max()) * thr_rel
    active = f > thr
    out, i, n = [], 0, len(f)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        out.append((i, j))
        i = j
    return out


class Algorithm:
    def __init__(self, params, rng):
        self.p = dict(PARAMS)
        self.p.update(params or {})
        self.rng = rng

    def _edit(self, f):
        g = f.copy()
        teeth = _teeth(g)
        if not teeth:
            return g, []
        centers = []
        n_edits = int(self.rng.integers(1, int(self.p["MAX_EDITS"]) + 1))
        for _ in range(n_edits):
            op = self.rng.choice(["shift", "kill", "birth", "split", "mass", "widen", "narrow"])
            if op == "shift" and teeth:
                k = int(self.rng.integers(0, len(teeth)))
                a, b = teeth[k]
                L = b - a
                shift = int(self.rng.normal(0, max(1.0, float(self.p["SHIFT_FRAC"]) * g.size)))
                if shift == 0:
                    shift = int(self.rng.choice([-2, -1, 1, 2]))
                na = int(np.clip(a + shift, 0, max(0, g.size - L)))
                chunk = g[a:b].copy()
                g[a:b] = 0.0
                g[na:na + L] += chunk
                centers.append(na + L // 2)
            elif op == "kill" and len(teeth) > 16:
                k = int(self.rng.integers(0, len(teeth)))
                a, b = teeth[k]
                mass = float(g[a:b].sum())
                g[a:b] = 0.0
                j = int(self.rng.integers(0, len(teeth)))
                if j == k:
                    j = (j + 3) % len(teeth)
                ca, cb = teeth[j]
                g[ca:cb] *= 1.0 + mass / max(float(g[ca:cb].sum()), 1e-15)
                centers.extend([(a + b) // 2, (ca + cb) // 2])
            elif op == "birth":
                L = int(self.p["BIRTH_LEN"])
                s = int(self.rng.integers(0, max(1, g.size - L)))
                masses = [float(g[a:b].sum()) for a, b in teeth]
                donor = int(np.argmax(masses))
                da, db = teeth[donor]
                g[da:db] *= 0.985
                steal = 0.015 * masses[donor]
                w = np.maximum(np.minimum(np.linspace(0, 1, L, endpoint=False), 0.5), 1e-12)
                g[s:s + L] += w / w.sum() * steal
                centers.extend([s + L // 2, (da + db) // 2])
            elif op == "split" and teeth:
                k = int(self.rng.integers(0, len(teeth)))
                a, b = teeth[k]
                if b - a < 8:
                    continue
                mid = (a + b) // 2
                gap = max(1, (b - a) // 12)
                g[mid:mid + gap] = 0.0
                centers.append(mid)
            elif op == "mass" and len(teeth) >= 2:
                i, j = self.rng.choice(len(teeth), size=2, replace=False)
                a1, b1 = teeth[i]
                a2, b2 = teeth[j]
                g[a1:b1] *= 0.97
                transfer = 0.03 * float(f[a1:b1].sum())
                g[a2:b2] *= 1.0 + transfer / max(float(g[a2:b2].sum()), 1e-15)
                centers.extend([(a1 + b1) // 2, (a2 + b2) // 2])
            elif op == "widen" and teeth:
                k = int(self.rng.integers(0, len(teeth)))
                a, b = teeth[k]
                if a > 0:
                    g[a - 1] = 0.5 * g[a]
                if b < g.size:
                    g[b] = 0.5 * g[b - 1]
                centers.append((a + b) // 2)
            elif op == "narrow" and teeth:
                k = int(self.rng.integers(0, len(teeth)))
                a, b = teeth[k]
                if b - a > 4:
                    g[a] = 0.0
                    g[b - 1] = 0.0
                centers.append((a + b) // 2)
            teeth = _teeth(g)
        return np.maximum(g, 0.0), centers

    def _neighborhood_lbfgs(self, f, centers, deadline):
        """Optimize amplitudes only on union of neighborhoods around edit centers."""
        best = f.copy()
        if not centers:
            return best
        n = best.size
        rad = 48
        mask = np.zeros(n, dtype=bool)
        for c in centers:
            mask[max(0, c - rad):min(n, c + rad)] = True
        idx = np.flatnonzero(mask)
        if idx.size < 4:
            return best
        x0 = best[idx].copy()

        def pack(z):
            trial = best.copy()
            trial[idx] = np.maximum(z, 0.0)
            return trial

        def fun(z):
            v, g = kit.smoothed_log_score(pack(z), beta_abs=50.0 / max(float(kit.autoconv(best).max()), 1e-12))
            return -v, -g[idx]

        rem = deadline - time.monotonic()
        if rem < 0.5:
            return best
        try:
            res = minimize(lambda z: fun(z)[0], x0, jac=lambda z: fun(z)[1],
                           method="L-BFGS-B", bounds=[(0.0, None)] * idx.size,
                           options={"maxiter": int(self.p["POLISH_ITERS"]), "ftol": 1e-14})
            cand = pack(res.x)
            if kit.score(cand) > kit.score(best):
                return cand
        except Exception:
            return best
        return best

    def perturb(self, f, intensity):
        g = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
        reps = 1 + int(3 * float(intensity))
        for _ in range(reps):
            g, _ = self._edit(g)
        return g

    def improve(self, f, deadline):
        best = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
        if best.max() > 0:
            best = best / best.max()
        best_s = kit.score(best)
        eps = float(self.p["EPS_ACCEPT"])
        tries = int(self.p["TRIES"])
        for _ in range(tries):
            if time.monotonic() >= deadline - 0.5:
                break
            trial, centers = self._edit(best)
            s = kit.score(trial)
            if s > best_s + eps:
                # refine neighborhood
                trial = self._neighborhood_lbfgs(trial, centers, deadline)
                s = kit.score(trial)
                if s > best_s:
                    best, best_s = trial, s
            elif s > best_s - 5e-7:
                # near-miss: try neighborhood polish anyway
                trial = self._neighborhood_lbfgs(trial, centers, min(deadline, time.monotonic() + 2.0))
                s = kit.score(trial)
                if s > best_s:
                    best, best_s = trial, s
        return best
