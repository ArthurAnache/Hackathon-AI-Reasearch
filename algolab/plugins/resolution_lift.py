"""Resolution lift: upsample by 2x (linear), then local improve via packet mass transfer.

Operator KIND = lift. Caps at MAX_VALUES (2e6) so EinsteinArena submissions stay valid.
Uses only algolab.kit + numpy (gate allowlist).
"""
import time

import numpy as np

from algolab import kit

PARAMS = {
    "UPSAMPLE": 2,
    "MAX_N": 2_000_000,
    "N_PACKETS": 16,
    "STEP_FRAC": 1e-3,
    "SIGMA_MAX": 1e-2,
    "SIGMA_MIN": 1e-5,
    "SIGMA_STEPS": 4,
    "FIRST_IMPROVE_FRAC": 0.4,
}


class Algorithm:
    def __init__(self, params, rng):
        self.p = dict(PARAMS)
        self.p.update(params or {})
        self.rng = rng

    def lift(self, f):
        f = np.asarray(f, dtype=np.float64).ravel()
        factor = max(1, int(self.p.get("UPSAMPLE", 2)))
        n2 = min(int(f.size) * factor, int(self.p.get("MAX_N", 2_000_000)))
        if n2 <= f.size:
            return np.maximum(f, 0.0)
        x_old = np.linspace(0.0, 1.0, f.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, n2, endpoint=False)
        g = np.interp(x_new, x_old, f)
        return np.maximum(g, 0.0)

    def perturb(self, f, intensity):
        g = self.lift(f)
        noise = self.rng.standard_normal(g.size)
        return np.maximum(g * np.exp(float(intensity) * 0.01 * noise), 0.0)

    def improve(self, f, deadline):
        g = self.lift(f)
        best = g.copy()
        best_s = kit.score(best)
        n_packets = max(2, int(self.p.get("N_PACKETS", 16)))
        step = float(self.p.get("STEP_FRAC", 1e-3))
        while time.monotonic() < deadline - 0.2:
            idx = self.rng.choice(best.size, size=min(n_packets, best.size), replace=False)
            trial = best.copy()
            donor, recv = int(idx[0]), int(idx[1 % len(idx)])
            mass = step * trial[donor]
            if mass <= 0:
                continue
            trial[donor] -= mass
            trial[recv] += mass
            trial = np.maximum(trial, 0.0)
            s = kit.score(trial)
            if s > best_s:
                best, best_s = trial, s
        return best
