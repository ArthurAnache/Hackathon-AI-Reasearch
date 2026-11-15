"""ENGINE B — Multiresolution basin hopping (coarse → lift → fine).

Never runs dense L-BFGS on 2M bounds. Structural perturbations and LSE/L-BFGS
happen at COARSE_N; promotion uses linear upsample + short fine polish.
Interpolation does not preserve C — fine kit.score gates every lift.
"""
import time

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from algolab import kit

PARAMS = {
    "COARSE_N": 8192,
    "SIGMA0": 5e-3,
    "SIGMA_MAX": 3e-2,
    "SIGMA_MIN": 1e-4,
    "SIGMA_STEPS": 10,
    "UPSAMPLE": 2,
    "MAX_N": 2_000_000,
    "FIRST_IMPROVE_FRAC": 0.2,
    "LBFGS_MAXITER": 60,
    "BETA0": 15.0,
    "BETA_GROW": 1.6,
    "STRUCTURAL_PERTURB_P": 0.55,
    "BASIN_ROUNDS": 5,
}


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

    def lift(self, f):
        return _upsample(f, min(int(f.size) * int(self.p.get("UPSAMPLE", 2)),
                                int(self.p.get("MAX_N", 2_000_000))))

    def _structural(self, f, intensity):
        f = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
        n = f.size
        noise = self.rng.standard_normal(n)
        k = max(3, int(n * 0.02))
        low = np.convolve(noise, np.ones(k) / k, mode="same")
        g = f * np.exp(float(intensity) * float(self.p["SIGMA0"]) * 12 * low)
        if self.rng.random() < float(self.p.get("STRUCTURAL_PERTURB_P", 0.55)):
            # micro band wipe (large wipes are unrecoverable vs the 2M incumbent)
            width = int(self.rng.uniform(0.001, 0.008) * n)
            a = int(self.rng.integers(0, max(1, n - width)))
            dumped = float(g[a:a + width].sum())
            g[a:a + width] *= self.rng.uniform(0.0, 0.25)
            for _ in range(int(self.rng.integers(1, 3))):
                b = int(self.rng.integers(0, n))
                w = int(self.rng.integers(3, max(4, width)))
                g[b:min(n, b + w)] += dumped / 2.0
        return np.maximum(g, 0.0)

    def perturb(self, f, intensity):
        f = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
        n = f.size
        n_c = min(n, int(self.p["COARSE_N"]))
        g = self._structural(_block_mean(f, n_c), intensity)
        return _upsample(g, n)

    def _lse_lbfgs(self, x, deadline):
        best = np.maximum(x, 0.0)
        best_s = kit.score(best)
        beta = float(self.p.get("BETA0", 15.0))
        grow = float(self.p.get("BETA_GROW", 1.6))
        maxiter = int(self.p.get("LBFGS_MAXITER", 60))
        while time.monotonic() < deadline - 0.4:
            c0 = kit.autoconv(best)
            beta_abs = beta / max(float(c0.max()), 1e-12)

            def fun(z):
                v, g = kit.smoothed_log_score(np.maximum(z, 0.0), beta_abs)
                return -v, -g

            rem = deadline - time.monotonic()
            it = max(5, min(maxiter, int(rem * 15)))
            try:
                res = minimize(lambda z: fun(z)[0], best, jac=lambda z: fun(z)[1],
                               method="L-BFGS-B", bounds=[(0.0, None)] * best.size,
                               options={"maxiter": it, "ftol": 1e-14})
                cand = np.maximum(res.x, 0.0)
            except Exception:
                break
            s = kit.score(cand)
            if s > best_s:
                best, best_s = cand, s
            beta *= grow
            if beta > 4000:
                break
        return best, best_s

    def improve(self, f, deadline):
        f0 = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
        if f0.max() > 0:
            f0 = f0 / f0.max()
        n = f0.size
        n_c = min(n, int(self.p["COARSE_N"]))
        best = f0.copy()
        best_s = kit.score(best)
        rounds = int(self.p.get("BASIN_ROUNDS", 5))
        for r in range(rounds):
            if time.monotonic() >= deadline - 0.6:
                break
            slice_end = min(deadline, time.monotonic() + max(3.0, (deadline - time.monotonic()) / max(rounds - r, 1)))
            xc = _block_mean(best, n_c)
            if r > 0:
                xc = self._structural(xc, intensity=0.4 + 0.15 * r)
            xc, _ = self._lse_lbfgs(xc, slice_end - 0.8)
            cand = _upsample(xc, n)
            # short fine LSE only if n is not huge OR remaining time allows tiny polish
            rem = deadline - time.monotonic()
            if rem > 2.0 and n <= 65000:
                cand, cs = self._lse_lbfgs(cand, deadline)
            else:
                # coordinate micro-polish at full res
                cs = kit.score(cand)
                thr = cand.max() * 1e-8
                supp = np.flatnonzero(cand > thr)
                t_end = min(deadline, time.monotonic() + min(2.0, rem))
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
