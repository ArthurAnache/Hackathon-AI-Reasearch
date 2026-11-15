"""Example of an LLM-scientist plugin that no SAVANT template can express (it passes the gate; see gate.py).

improve: maximize the smoothed log-score log C_beta (max c replaced by a log-sum-exp at absolute temperature
         beta / max c) with bound-constrained L-BFGS-B on f >= 0 directly (not f = w^2, so cells at zero can re-enter
         the support), cascading beta upward. Every function evaluation also yields the EXACT score for free, and the
         best exactly-scored point seen anywhere (line-search trials included) is returned. This is the Dinkelbach /
         LSE-cascade family that produced the forum's 0.9620 -> 0.9627 gains, here in plain numpy/scipy.
perturb: multiplicative low-frequency noise f * exp(sigma * lowpass(noise)), which keeps the support and changes the
         macro-shape (reported on the forum as a ~70 % hit rate against ~0 % for pixel noise).
"""
import math
import time

import numpy as np
from scipy.fft import irfft
from scipy.optimize import minimize

from algolab import kit

PARAMS = {
    "BETAS": [1e6, 1e7, 1e8, 1e9],   # relative temperatures (times 1 / max c at stage start)
    "MAXCOR": 8,
    "LOWFREQ_FRAC": 0.01,           # perturb: fraction of the spectrum that receives noise
    "FIRST_IMPROVE_FRAC": 0.5,
    "IMPROVE_SLICE_S": 20.0,
    "SIGMA_MAX": 1e-3,
    "SIGMA_MIN": 1e-6,
    "SIGMA_STEPS": 4,
}


class _Stop(Exception):
    pass


class Algorithm:
    def __init__(self, params, rng):
        self.p = params
        self.rng = rng

    def improve(self, f, deadline):
        f = np.maximum(np.asarray(f, dtype=np.float64), 0.0)
        f = f / f.max()
        best = {"x": f.copy(), "s": kit.score(f)}
        betas = list(self.p["BETAS"])
        for i, beta in enumerate(betas):
            now = time.monotonic()
            if now >= deadline - 0.5:
                break
            stage_end = now + (deadline - now) / (len(betas) - i)
            c_max = float(kit.autoconv(best["x"]).max())
            beta_abs = beta / c_max

            def fun(x):
                if time.monotonic() > stage_end:
                    raise _Stop()
                val, grad = kit.smoothed_log_score(x, beta_abs)
                s = kit.score(x)
                if s > best["s"]:
                    best["x"], best["s"] = x.copy(), s
                return -val, -grad

            try:
                minimize(fun, best["x"], jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * len(f),
                         options={"maxcor": int(self.p["MAXCOR"]), "ftol": 0.0, "gtol": 0.0, "maxiter": 10 ** 7,
                                  "maxfun": 10 ** 7})
            except _Stop:
                pass
        return best["x"]

    def perturb(self, f, intensity):
        n = len(f)
        k = max(2, int(self.p["LOWFREQ_FRAC"] * (n // 2 + 1)))
        spec = np.zeros(n // 2 + 1, dtype=np.complex128)
        spec[1:k] = self.rng.normal(size=k - 1) + 1j * self.rng.normal(size=k - 1)
        noise = irfft(spec, n)
        noise /= max(float(noise.std()), 1e-300)
        return f * np.exp(intensity * noise)
