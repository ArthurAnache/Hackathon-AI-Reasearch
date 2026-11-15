"""ENGINE C — Coarse packet / active-set exchange with lift.

Epigraph idea: C = Q/(3 S^2 M) with M = max c. Exchange mass among sparse
packets to reduce active near-max lags. All heavy search runs at COARSE_N;
full-res is only for final upsample + short polish.
"""
import time

import numpy as np
from scipy.interpolate import interp1d

from algolab import kit

PARAMS = {
    "COARSE_N": 16384,
    "N_PACKETS": 96,
    "N_CUTS": 64,
    "TRUST": 0.08,
    "REL_GAP": 1e-5,
    "ROUNDS": 40,
    "FIRST_IMPROVE_FRAC": 0.25,
    "SIGMA_MAX": 2e-2,
    "SIGMA_MIN": 1e-4,
    "SIGMA_STEPS": 8,
    "BLEND_START": 0.15,  # keep little incumbent mass (was 0.5 — killed jumps)
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


def _packets_from_support(f, n_packets, rng):
    f = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
    n = f.size
    thr = f.max() * 1e-8
    idx = np.flatnonzero(f > thr)
    if idx.size == 0:
        idx = np.arange(n)
    centers = rng.choice(idx, size=min(n_packets, idx.size), replace=False)
    centers = np.sort(centers)
    packets = []
    mass_each = float(f.sum()) / max(len(centers), 1)
    half = max(2, n // (4 * max(len(centers), 1)))
    for c in centers:
        a = int(max(0, c - half))
        b = int(min(n, c + half + 1))
        packets.append([a, b, mass_each])
    return packets


def _assemble(n, packets):
    out = np.zeros(n, dtype=np.float64)
    for a, b, m in packets:
        L = b - a
        if L <= 0 or m <= 0:
            continue
        w = np.linspace(0, 1, L, endpoint=False)
        w = np.maximum(np.minimum(w, 1 - w), 1e-12)
        out[a:b] += w / w.sum() * m
    return out


class Algorithm:
    def __init__(self, params, rng):
        self.p = dict(PARAMS)
        self.p.update(params or {})
        self.rng = rng

    def perturb(self, f, intensity):
        f = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
        n = f.size
        n_c = min(n, int(self.p["COARSE_N"]))
        fc = _block_mean(f, n_c)
        pk = _packets_from_support(fc, int(self.p["N_PACKETS"]), self.rng)
        masses = np.array([m for _, _, m in pk], dtype=np.float64)
        masses = self.rng.dirichlet(np.ones(len(masses)) * (0.3 / max(float(intensity), 1e-3)))
        masses *= float(fc.sum())
        pk2 = [[a, b, float(m)] for (a, b, _), m in zip(pk, masses)]
        # relocate a fraction of packets
        for i in range(len(pk2)):
            if self.rng.random() < 0.25 * float(intensity):
                L = pk2[i][1] - pk2[i][0]
                s = int(self.rng.integers(0, max(1, n_c - L)))
                pk2[i][0], pk2[i][1] = s, s + L
        return _upsample(_assemble(n_c, pk2), n)

    def improve(self, f, deadline):
        f0 = np.maximum(np.asarray(f, dtype=np.float64).ravel(), 0.0)
        if f0.max() > 0:
            f0 = f0 / f0.max()
        n = f0.size
        n_c = min(n, int(self.p["COARSE_N"]))
        fc0 = _block_mean(f0, n_c)
        packets = _packets_from_support(fc0, int(self.p["N_PACKETS"]), self.rng)
        blend = float(self.p.get("BLEND_START", 0.15))
        best_c = (1.0 - blend) * _assemble(n_c, packets) + blend * fc0
        best_c *= (fc0.sum() / max(best_c.sum(), 1e-15))
        best_c_s = kit.score(best_c)
        trust = float(self.p["TRUST"])
        rel = float(self.p["REL_GAP"])
        rounds = int(self.p["ROUNDS"])

        for _ in range(rounds):
            if time.monotonic() >= deadline - 0.8:
                break
            c = kit.autoconv(best_c)
            m = float(c.max())
            active = np.flatnonzero(c >= m * (1.0 - rel))
            if active.size == 0:
                active = np.array([int(c.argmax())])
            cuts = active if active.size <= int(self.p["N_CUTS"]) else self.rng.choice(
                active, size=int(self.p["N_CUTS"]), replace=False)

            heats = np.array([float(best_c[a:b].sum()) for a, b, _ in packets])
            # bias donors that contribute to high autocorrelation energy near cuts
            if cuts.size and heats.sum() > 0:
                donor = int(np.argmax(heats))
                recv = int(np.argmin(heats))
            else:
                donor = int(self.rng.integers(0, len(packets)))
                recv = int(self.rng.integers(0, len(packets)))
            if donor == recv:
                continue
            da, db, dm = packets[donor]
            ra, rb, rm = packets[recv]
            step = trust * dm
            if step <= 0:
                continue
            packets[donor] = [da, db, dm - step]
            packets[recv] = [ra, rb, rm + step]
            # occasional geometric jump of a packet
            if self.rng.random() < 0.12:
                L = rb - ra
                s = int(self.rng.integers(0, max(1, n_c - L)))
                packets[recv] = [s, s + L, packets[recv][2]]
            trial = _assemble(n_c, packets)
            trial = 0.85 * trial + 0.15 * best_c
            trial *= (best_c.sum() / max(trial.sum(), 1e-15))
            s = kit.score(trial)
            if s > best_c_s:
                best_c, best_c_s = trial, s
                trust = min(0.25, trust * 1.15)
            else:
                packets[donor] = [da, db, dm]
                packets[recv] = [ra, rb, rm]
                trust *= 0.75
                if trust < 1e-4:
                    packets = _packets_from_support(best_c, int(self.p["N_PACKETS"]), self.rng)
                    trust = float(self.p["TRUST"])

        best = _upsample(best_c, n)
        # fine coordinate polish
        best_s = kit.score(best)
        thr = best.max() * 1e-8
        supp = np.flatnonzero(best > thr)
        while time.monotonic() < deadline - 0.2 and supp.size >= 2:
            i, j = self.rng.choice(supp, size=2, replace=False)
            step = 0.02 * best[i]
            if step <= 0:
                continue
            trial = best.copy()
            trial[i] -= step
            trial[j] += step
            trial = np.maximum(trial, 0.0)
            s = kit.score(trial)
            if s > best_s:
                best, best_s = trial, s
                supp = np.flatnonzero(best > thr)
        return best
