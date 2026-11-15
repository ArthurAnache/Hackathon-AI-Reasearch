"""ENGINE E — Spectral low-mode perturbation + LSE improve (forum class, public claim).

Perturb only the lowest ~1–5% Fourier modes of log-amplitudes on the support,
then recover with neighborhood-aware L-BFGS on a smoothed log-score. Designed
to micro-rebuild like the public 2782→2783 jump without full comb destruction.
"""
import time

import numpy as np
from scipy.optimize import minimize

from algolab import kit

PARAMS = {
    "MODE_FRAC": 0.03,
    "SIGMA0": 2e-3,
    "SIGMA_MAX": 1e-2,
    "SIGMA_MIN": 1e-5,
    "SIGMA_STEPS": 8,
    "FIRST_IMPROVE_FRAC": 0.25,
    "LBFGS_MAXITER": 25,
    "BETA0": 30.0,
    "BETA_GROW": 1.8,
    "ROUNDS": 6,
}


class Algorithm:
    def __init__(self, params, rng):
        self.p = dict(PARAMS)
        self.p.update(params or {})
        self.rng = rng

    def perturb(self, f, intensity):
        f = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
        thr = f.max() * 1e-8
        support = f > thr
        g = f.copy()
        x = np.log(np.maximum(g[support], 1e-30))
        X = np.fft.rfft(x)
        kmax = max(2, int(len(X) * float(self.p["MODE_FRAC"])))
        noise = (self.rng.standard_normal(kmax) + 1j * self.rng.standard_normal(kmax))
        noise[0] = noise[0].real  # keep DC real-ish
        scale = float(intensity) * float(self.p["SIGMA0"]) * 8
        X[:kmax] *= np.exp(scale * noise)
        x2 = np.fft.irfft(X, n=x.size).real
        g[support] = np.exp(x2)
        # tiny mass birth in a few empty cells near support edges
        idx = np.flatnonzero(support)
        if idx.size:
            for _ in range(3):
                i = int(self.rng.choice(idx))
                j = int(np.clip(i + int(self.rng.integers(-20, 21)), 0, g.size - 1))
                if g[j] <= thr:
                    g[j] = 0.05 * g[i]
        return np.maximum(g, 0.0)

    def improve(self, f, deadline):
        best = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
        if best.max() > 0:
            best = best / best.max()
        best_s = kit.score(best)
        beta = float(self.p["BETA0"])
        grow = float(self.p["BETA_GROW"])
        maxiter = int(self.p["LBFGS_MAXITER"])
        rounds = int(self.p["ROUNDS"])

        for r in range(rounds):
            if time.monotonic() >= deadline - 0.6:
                break
            # optional mid-round micro spectral kick
            x = best.copy()
            if r > 0:
                x = self.perturb(x, intensity=0.3 + 0.1 * r)
            # optimize only on support (+ thin halo) to cut degrees of freedom
            thr = x.max() * 1e-8
            mask = x > thr
            # dilate halo by 2 cells
            halo = mask.copy()
            halo[1:] |= mask[:-1]
            halo[:-1] |= mask[1:]
            idx = np.flatnonzero(halo)
            if idx.size < 8:
                break
            x0 = x[idx].copy()
            c0 = kit.autoconv(x)
            beta_abs = beta / max(float(c0.max()), 1e-12)

            def pack(z):
                trial = x.copy()
                trial[idx] = np.maximum(z, 0.0)
                return trial

            def fun(z):
                v, g = kit.smoothed_log_score(pack(z), beta_abs)
                return -v, -g[idx]

            rem = deadline - time.monotonic()
            it = max(4, min(maxiter, int(rem * 6)))
            try:
                res = minimize(lambda z: fun(z)[0], x0, jac=lambda z: fun(z)[1],
                               method="L-BFGS-B", bounds=[(0.0, None)] * idx.size,
                               options={"maxiter": it, "ftol": 1e-14})
                cand = pack(res.x)
            except Exception:
                break
            s = kit.score(cand)
            if s > best_s:
                best, best_s = cand, s
            beta *= grow
            if beta > 8000:
                break
        return best
