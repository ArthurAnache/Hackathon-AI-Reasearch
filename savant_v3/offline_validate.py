"""Offline validation for V3 engines (no Modal spend)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _run_plugin(plugin: Path, f0: np.ndarray, budget_s: float, seed: int, params: dict | None = None):
    from algolab import gate, kit
    src = plugin.read_text(encoding="utf-8")
    reasons = gate.check_source(src)
    if reasons:
        return {"ok": False, "gate": reasons}
    ns: dict = {"__name__": "algolab_plugin"}
    exec(compile(src, str(plugin), "exec"), ns)
    rng = np.random.default_rng(seed)
    alg = ns["Algorithm"]({**(ns.get("PARAMS") or {}), **(params or {})}, rng)
    t0 = time.monotonic()
    deadline = t0 + budget_s
    x = f0.copy()
    if hasattr(alg, "perturb"):
        x = alg.perturb(x, intensity=1.0)
    x = alg.improve(x, deadline)
    wall = time.monotonic() - t0
    return {
        "ok": True,
        "wall_s": wall,
        "score_kit": kit.score(x),
        "score_start": kit.score(f0),
        "values": x,
        "sha256": hashlib.sha256(src.encode()).hexdigest(),
    }


def main():
    from savant_v3.topology import topology_report, teeth
    from problems.catalog import load
    from algolab import kit, gate

    out_dir = ROOT / "runs" / "v3" / "offline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic small landscape + downsampled cert proxy
    rng = np.random.default_rng(0)
    synth = np.zeros(4096)
    for c in rng.integers(0, 4096, size=40):
        w = int(rng.integers(3, 20))
        a, b = max(0, c - w), min(4096, c + w)
        synth[a:b] += rng.uniform(0.1, 1.0)
    synth = np.maximum(synth, 0.0)

    p = load("second-autocorrelation-inequality")
    cert = np.asarray(p.certificate()["values"], dtype=np.float64)
    # coarse proxy of cert for speed (block mean to 8k)
    cert_8k = np.array([cert[i:i + len(cert) // 8192].mean()
                        for i in range(0, len(cert), max(1, len(cert) // 8192))])[:8192]
    cert_8k = np.maximum(cert_8k, 0.0)

    engines = [
        ("A_comb", ROOT / "algolab/plugins/comb_reconstruct.py", {}),
        ("B_multires", ROOT / "algolab/plugins/multires_basin.py", {}),
        ("C_active_set", ROOT / "algolab/plugins/active_set_exchange.py", {}),
    ]
    rows = []
    for name, path, params in engines:
        adm = gate.admit(path)
        row = {"engine": name, "plugin": str(path.relative_to(ROOT)), "admitted": adm["admitted"],
               "gate_reasons": adm.get("reasons")}
        if not adm["admitted"]:
            rows.append(row)
            continue
        # synthetic
        r = _run_plugin(path, synth, budget_s=4.0, seed=101, params=params)
        topo = topology_report(synth, r["values"]) if r.get("ok") else None
        row["synthetic"] = {
            "ok": r.get("ok"), "gain_kit": (r["score_kit"] - r["score_start"]) if r.get("ok") else None,
            "topology": topo, "wall_s": r.get("wall_s"),
        }
        # coarse cert proxy
        r2 = _run_plugin(path, cert_8k, budget_s=6.0, seed=202, params=params)
        topo2 = topology_report(cert_8k, r2["values"]) if r2.get("ok") else None
        row["cert_8k"] = {
            "ok": r2.get("ok"), "gain_kit": (r2["score_kit"] - r2["score_start"]) if r2.get("ok") else None,
            "topology": topo2, "wall_s": r2.get("wall_s"),
            "structural_change": (topo2 or {}).get("structural_change"),
        }
        # kit vs frozen on a tiny vector sanity
        tiny = np.linspace(1.0, 0.01, 256)
        sk = kit.score(tiny)
        try:
            sf = float(p.score({"values": tiny.tolist()}))
            row["kit_vs_frozen_tiny"] = {"kit": sk, "frozen": sf, "abs_diff": abs(sk - sf)}
        except Exception as exc:  # noqa: BLE001
            row["kit_vs_frozen_tiny"] = {"error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(name, "synth_struct", (topo or {}).get("structural_change"),
              "cert8k_struct", (topo2 or {}).get("structural_change"),
              "gains", row["synthetic"]["gain_kit"], row["cert_8k"]["gain_kit"])

    report = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cert_teeth": len(teeth(cert)),
        "engines": rows,
        "pass_criterion": "each engine admits + produces structural_change on synth OR cert_8k",
    }
    report["passed"] = all(
        r.get("admitted") and (
            (r.get("synthetic") or {}).get("topology", {}).get("structural_change")
            or (r.get("cert_8k") or {}).get("structural_change")
        )
        for r in rows
    )
    (out_dir / "validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print("PASSED" if report["passed"] else "FAILED", "->", out_dir / "validation.json")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
