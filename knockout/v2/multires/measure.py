"""Local CPU measurements behind the coarse-to-fine experiments (numpy only, about two minutes):

    .venv/bin/python -m knockout.v2.multires.measure knockout/v2/results/multires/local_measurements.json

- the frozen verifier against the closed form, and the repeat-upsampling invariance, on real data;
- E1, part 1: the score of the frozen certificate (EA solution 2782) and of the live #1 after sum-pooling by
  2 ... 128, for two pooling phases;
- the score lost at once by every perturbation used in E2 / E3 (before any optimization);
- how the live #1 differs from the certificate: block-mass profiles, cells born and dead, how far the born cells
  sit from the old support, and how much the two autoconvolutions differ.
The live #1 is read with one public GET (no key). Nothing here optimizes or submits.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from knockout.v2.multires import perturb as pt
from knockout.v2.multires import resample as rs

SLUG = "second-autocorrelation-inequality"


def pooled_scores(f: np.ndarray, factors=(2, 3, 4, 8, 16, 32, 64, 128)) -> list[dict]:
    base, out = rs.closed_form_score(f), []
    for k in factors:
        for phase in sorted({0, k // 2}):
            g = rs.pad_to_multiple(np.concatenate([np.zeros(phase), f]), k)     # phase: zeros in front
            pooled = rs.sum_pool(g, k)
            s = rs.closed_form_score(pooled)
            out.append({"pool": k, "phase": phase, "n": int(pooled.size), "nonzero": int(np.count_nonzero(pooled)),
                        "score": s, "minus_start": s - base})
    return out


def perturbation_dips(f: np.ndarray, seed: int = 1) -> list[dict]:
    specs = [{"kind": "blend", "block": k, "t": t} for k in (4, 16, 64, 256) for t in (0.01, 0.1, 0.3, 1.0)]
    for k in (32, 128, 512):
        for frac in (1.0, 0.1, 0.01):
            specs += [{"kind": "shuffle", "block": k, "frac": frac},
                      {"kind": "shift", "block": k, "max_shift": 2, "frac": frac},
                      {"kind": "redraw", "block": k, "frac": frac}]
        specs += [{"kind": "resample", "block": k, "sigma": s, "frac": 1.0} for s in (0.1, 0.3, 1.0)]
    specs += [{"kind": "lognormal", "sigma": s} for s in (0.03, 0.1, 0.3, 1.0)]
    specs += [{"kind": "uniform", "scale": a} for a in (1e-8, 1e-6, 1e-4)]
    specs += [{"kind": "dilate", "radius": r, "t": t} for r in (1, 2, 4, 8) for t in (1e-6, 1e-5, 1e-4, 1e-3)]
    base, out = rs.closed_form_score(f), []
    for spec in specs:
        g = pt.apply(f, spec, seed)
        s = rs.closed_form_score(g)
        out.append({"perturbation": pt.describe((spec,)), "score": s, "minus_start": s - base,
                    "nonzero": int(np.count_nonzero(g)), "profile_l1_1000": rs.profile_l1(g, f, 1000)})
    return out


def difference(old: np.ndarray, new: np.ndarray) -> dict:
    """How `new` (the live #1) differs from `old` (the certificate); both have the same length."""
    f, h = old / old.sum(), new / new.sum()
    n = f.size
    sf, sh = f > 0, h > 0
    born, dead = sh & ~sf, sf & ~sh
    idx, b = np.flatnonzero(sf), np.flatnonzero(born)
    pos = np.searchsorted(idx, b)
    dist = np.minimum(np.abs(b - idx[np.clip(pos - 1, 0, idx.size - 1)]),
                      np.abs(idx[np.clip(pos, 0, idx.size - 1)] - b))
    m = 2 * n
    spec_f, spec_h = np.fft.rfft(f, m), np.fft.rfft(h, m)
    gf, gh = np.fft.irfft(spec_f * spec_f, m)[: 2 * n - 1], np.fft.irfft(spec_h * spec_h, m)[: 2 * n - 1]
    top_f, top_h = gf >= gf.max() * (1 - 1e-7), gh >= gh.max() * (1 - 1e-7)
    return {
        "relative_l1": float(np.abs(h - f).sum()),
        "profile_l1": {str(k): rs.profile_l1(f, h, k) for k in (20, 100, 1000, 10000, 100000)},
        "support_old": int(sf.sum()), "support_new": int(sh.sum()), "support_common": int((sf & sh).sum()),
        "born_cells": int(born.sum()), "born_mass_share": float(h[born].sum()),
        "dead_cells": int(dead.sum()), "dead_mass_share": float(f[dead].sum()),
        "born_within_cells_of_old_support": {str(k): float((dist <= k).mean()) for k in (1, 2, 4, 8, 16, 64)},
        "born_mass_within_cells_of_old_support": {str(k): float(h[b][dist <= k].sum() / h[b].sum())
                                                 for k in (1, 2, 4, 8, 16, 64)},
        "mass_by_5pct_old": [round(float(x), 5) for x in np.add.reduceat(f, np.arange(0, n, n // 20 + 1))],
        "mass_by_5pct_new": [round(float(x), 5) for x in np.add.reduceat(h, np.arange(0, n, n // 20 + 1))],
        "autoconvolution_relative_l1": float(np.abs(gh - gf).sum() / gf.sum()),
        "lags_within_1e-7_of_max": {"old": int(top_f.sum()), "new": int(top_h.sum()),
                                    "common": int((top_f & top_h).sum())},
    }


def main(out_path: str) -> dict:
    from knockout.problems_api import load_problem
    from knockout.run import load_init
    from problems.catalog import load

    p = load(SLUG)
    frozen = np.asarray(p.certificate()["values"], dtype=np.float64)
    cand, info = load_init(load_problem(SLUG), "live")
    live = np.asarray(cand["values"], dtype=np.float64)
    out = {"frozen": {"n": int(frozen.size), "verifier": float(p.score({"values": frozen.tolist()})),
                      "closed_form": rs.closed_form_score(frozen)},
           "live": {"n": int(live.size), "verifier": float(p.score({"values": live.tolist()})),
                    "closed_form": rs.closed_form_score(live), "solution_id": info.get("solution_id"),
                    "agent": info.get("agent"), "displayed_score": info.get("displayed_score")}}
    pooled = rs.sum_pool(rs.pad_to_multiple(frozen, 64), 64)
    v0 = float(p.score({"values": pooled.tolist()}))
    out["repeat_check_frozen_verifier"] = {
        "vector": "certificate pooled by 64", "n": int(pooled.size), "score": v0,
        "x2_minus": float(p.score({"values": rs.repeat_upsample(pooled, 2).tolist()})) - v0,
        "x3_minus": float(p.score({"values": rs.repeat_upsample(pooled, 3).tolist()})) - v0,
        "closed_form_minus_verifier": rs.closed_form_score(pooled) - v0}
    out["pooled"] = {"frozen": pooled_scores(frozen), "live": pooled_scores(live)}
    out["perturbation_dips_frozen"] = perturbation_dips(frozen)
    out["live_vs_frozen"] = difference(frozen, live)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def loss_table(out_path: str, named_paths: list[str]) -> list[dict]:
    """The two exact loss terms (resample.loss_terms) of the frozen certificate, of the certificate pooled by
    2 ... 64, and of the stored solutions given as name=path.npy (downloaded from the Volume)."""
    from problems.catalog import load

    frozen = np.asarray(load(SLUG).certificate()["values"], dtype=np.float64)
    items = [("certificate (EA solution 2782)", frozen)]
    items += [(f"certificate pooled by {k}", rs.sum_pool(rs.pad_to_multiple(frozen, k), k))
              for k in (2, 4, 8, 16, 32, 64)]
    items += [(name, np.load(path)) for name, path in (x.split("=", 1) for x in named_paths)]
    out = []
    for name, f in items:
        terms = rs.loss_terms(f)
        out.append({"solution": name, "n": int(f.size), "score": rs.closed_form_score(f), **terms,
                    "roughness_share": terms["roughness"] / (terms["plateau"] + terms["roughness"])})
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--loss":
        for row in loss_table(sys.argv[2], sys.argv[3:]):
            print(f"{row['solution']:<42} n {row['n']:<8} score {row['score']:.9f} plateau {row['plateau']:.6f} "
                  f"roughness {row['roughness']:.3e} ({row['roughness_share']:.2%} of the loss)")
        raise SystemExit(0)
    res = main(sys.argv[1])
    print(json.dumps({k: res[k] for k in ("frozen", "live", "repeat_check_frozen_verifier")}, indent=1))
    for name, rows in res["pooled"].items():
        for r in rows:
            print(f"{name:<7} pool {r['pool']:<4} phase {r['phase']:<3} n {r['n']:<8} score {r['score']:.6f} "
                  f"({r['minus_start']:+.3e})")
