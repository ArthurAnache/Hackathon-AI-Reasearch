"""Map hypotheses to vetted optimizer templates, or to Algorithm Lab plugins.

Knob templates (probe / blockshift / …) never execute arbitrary LLM Python.
`algolab_plugin` materializes a gate-admitted plugin through the WS3 adapter
`optimizers/second-autocorrelation-inequality_algolab.py`. Plugin source travels
in the job config; Modal runs it with block_network (see algolab/modal_app.py) or
via the WS3 bench image. Structural templates route to the same adapter once a
plugin path is present.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from savant.config import ROOT

# Allowed CONFIG keys and bounds (from optimizer SEARCH_SPACE / safe ranges)
BLOCKSHIFT_BOUNDS = {
    "BLOCK_LOG2_MAX": (4, 21, "int"),
    "D_LOG10_LO": (-14.0, -8.0, "linear"),
    "D_LOG10_HI": (-6.0, -1.0, "linear"),
    "D_GRID": (6, 24, "int"),
    "GOLDEN_ITERS": (0, 16, "int"),
    "GAP_REL": (1e-8, 1e-4, "log"),
    "GUIDED": (0.0, 1.0, "linear"),
    "REFRESH_S": (5.0, 120.0, "log"),
}
PROBE_BOUNDS = {
    "POLISH_FRAC": (0.3, 0.95, "linear"),
    "BLOCK_LOG2_MAX": (4, 16, "int"),
}

TEMPLATE_FILES = {
    "probe": "optimizers/second-autocorrelation-inequality_probe.py",
    "probe_mutate": "optimizers/second-autocorrelation-inequality_probe.py",
    "blockshift": "optimizers/second-autocorrelation-inequality_blockshift.py",
    "blockshift_mutate": "optimizers/second-autocorrelation-inequality_blockshift.py",
    "hybrid_refresh": "optimizers/second-autocorrelation-inequality_blockshift.py",
    "algolab_plugin": "optimizers/second-autocorrelation-inequality_algolab.py",
    # Structural move classes → Algorithm Lab plugins (executed via algolab adapter).
    "perturb_mutate": "algolab/plugins/lse_lbfgs_spectral.py",
    "improve_mutate": "algolab/plugins/lse_lbfgs_spectral.py",
    "lift_mutate": "algolab/plugins/resolution_lift.py",
    "basin_hop": "algolab/plugins/lse_lbfgs_spectral.py",
    "dinkelbach_lse": "algolab/plugins/dinkelbach_kelley.py",
}
ALGOLAB_ADAPTER = TEMPLATE_FILES["algolab_plugin"]

# Move classes that count toward the ≥50% structural compute reserve (planner).
STRUCTURAL_TEMPLATES = frozenset({
    "algolab_plugin", "perturb_mutate", "improve_mutate", "lift_mutate",
    "basin_hop", "dinkelbach_lse", "queue_novel",
})
POLISH_TEMPLATES = frozenset({
    "probe", "probe_mutate", "blockshift", "blockshift_mutate", "hybrid_refresh",
})


def _clip(key: str, value: Any, bounds: dict) -> Any:
    if key not in bounds:
        raise ValueError(f"config key {key!r} not allowed")
    lo, hi, kind = bounds[key]
    if kind == "int":
        return int(max(lo, min(hi, int(round(float(value))))))
    v = float(value)
    return max(lo, min(hi, v))


def bounds_for(template: str) -> dict | None:
    if template in ("probe", "probe_mutate"):
        return PROBE_BOUNDS
    if template in ("blockshift", "blockshift_mutate", "hybrid_refresh"):
        return BLOCKSHIFT_BOUNDS
    if template in ("algolab_plugin", "perturb_mutate", "improve_mutate", "lift_mutate",
                    "basin_hop", "dinkelbach_lse"):
        return {}
    return None


def sanitize_overrides(template: str, overrides: dict | None) -> tuple[dict, list[dict]]:
    """Keep only parameters this template accepts. Drop None and foreign keys; do not raise.

    Returns (clipped_config, rejected) where each rejection is {key, value, reason}.
    """
    overrides = dict(overrides or {})
    rejected: list[dict] = []
    if "INIT" in overrides:
        rejected.append({"key": "INIT", "value": overrides.pop("INIT"),
                         "reason": "INIT is set by the job runner, not by a hypothesis"})
    if template == "queue_novel":
        for k, v in overrides.items():
            rejected.append({"key": k, "value": v, "reason": "queue_novel is not executed; route through author.py"})
        return {}, rejected
    if template in ("algolab_plugin", "perturb_mutate", "improve_mutate", "lift_mutate",
                    "basin_hop", "dinkelbach_lse"):
        reserved = {"PLUGIN_SOURCE", "PLUGIN_SHA256", "TRUSTED", "INIT"}
        params, rejected_local = {}, []
        for k, v in overrides.items():
            if k in reserved:
                rejected_local.append({"key": k, "value": v, "reason": "reserved for the adapter"})
            elif v is None:
                rejected_local.append({"key": k, "value": None, "reason": "null"})
            else:
                params[k] = v
        return {"PLUGIN_PARAMS": params}, rejected_local
    bounds = bounds_for(template)
    if bounds is None:
        raise ValueError(f"unknown template {template!r}")
    kept: dict = {}
    for k, v in overrides.items():
        if v is None:
            rejected.append({"key": k, "value": None, "reason": "null"})
            continue
        if k not in bounds:
            rejected.append({"key": k, "value": v,
                             "reason": f"not a parameter of template {template}"})
            continue
        kept[k] = _clip(k, v, bounds)
    if template == "hybrid_refresh":
        kept.setdefault("REFRESH_S", 15.0)
        kept.setdefault("GUIDED", 0.7)
    return kept, rejected


def is_structural(template: str | None) -> bool:
    return (template or "") in STRUCTURAL_TEMPLATES


def _materialize_plugin(hypothesis: dict, *, default_plugin: str | None = None) -> dict:
    """Gate-admit a plugin file (or inline source) and build an adapter job config."""
    from algolab import gate
    hid = hypothesis.get("hypothesis_id")
    plugin_path = hypothesis.get("plugin") or hypothesis.get("plugin_path") or default_plugin
    src = hypothesis.get("plugin_source")
    if plugin_path and not src:
        path = ROOT / plugin_path
        if not path.is_file():
            return {"executable": False, "queued": True, "hypothesis_id": hid,
                    "reason": f"plugin file missing: {plugin_path}", "rejected_overrides": []}
        src = path.read_text(encoding="utf-8")
    if not src:
        return {"executable": False, "queued": True, "hypothesis_id": hid,
                "reason": "algolab_plugin needs plugin= path or plugin_source=", "rejected_overrides": []}
    reasons = gate.check_source(src)
    if reasons:
        return {"executable": False, "queued": True, "hypothesis_id": hid,
                "reason": "gate: " + "; ".join(reasons[:5]), "gate_reasons": reasons,
                "rejected_overrides": []}
    cfg, rejected = sanitize_overrides("algolab_plugin", hypothesis.get("config_overrides"))
    sha = hashlib.sha256(src.encode()).hexdigest()
    cfg.update(PLUGIN_SOURCE=src, PLUGIN_SHA256=sha, TRUSTED=False)
    return {
        "executable": True, "queued": False, "structural": True, "hypothesis_id": hid,
        "role": hypothesis.get("role"), "title": hypothesis.get("title"),
        "template": hypothesis.get("template") or "algolab_plugin",
        "optimizer": ALGOLAB_ADAPTER, "config": cfg, "rejected_overrides": rejected,
        "plugin": plugin_path, "plugin_sha256": sha,
        "claim": hypothesis.get("claim"), "expected_gain": hypothesis.get("expected_gain"),
        "falsification": hypothesis.get("falsification"),
        "isolation": "modal:block_network + gate + audit_hook + parent_verifier",
    }


def materialize(hypothesis: dict) -> dict:
    """Return an executable job spec or a queued novel idea. Never raises on a foreign CONFIG key."""
    template = hypothesis.get("template")
    hid = hypothesis.get("hypothesis_id")
    if template == "queue_novel":
        if hypothesis.get("plugin") or hypothesis.get("plugin_source") or hypothesis.get("plugin_path"):
            return _materialize_plugin({**hypothesis, "template": "algolab_plugin"})
        plugin = (hypothesis.get("config_overrides") or {}).get("PLUGIN_PATH")
        if plugin:
            return _materialize_plugin({**hypothesis, "template": "algolab_plugin", "plugin": plugin})
        _, rejected = sanitize_overrides(template, hypothesis.get("config_overrides"))
        return {
            "executable": False, "queued": True, "structural": True,
            "reason": "Novel algorithm parked until author.py admits a plugin (up to 3 repairs)",
            "hypothesis_id": hid, "title": hypothesis.get("title"),
            "algorithmic_change": hypothesis.get("algorithmic_change"),
            "rejected_overrides": rejected,
        }
    if template == "algolab_plugin":
        return _materialize_plugin(hypothesis)
    if template in ("perturb_mutate", "improve_mutate", "lift_mutate", "basin_hop", "dinkelbach_lse"):
        default = TEMPLATE_FILES.get(template)
        kind_map = {
            "perturb_mutate": "perturb", "improve_mutate": "improve", "lift_mutate": "lift",
            "basin_hop": "basin_hop", "dinkelbach_lse": "improve",
        }
        ov = dict(hypothesis.get("config_overrides") or {})
        ov.setdefault("KIND", kind_map[template])
        return _materialize_plugin({**hypothesis, "config_overrides": ov}, default_plugin=default)
    path = TEMPLATE_FILES.get(template or "")
    if path is None:
        return {"executable": False, "queued": False, "hypothesis_id": hid,
                "reason": f"unknown template {template!r}", "rejected_overrides": []}
    if not (ROOT / path).exists():
        return {"executable": False, "queued": False, "hypothesis_id": hid,
                "reason": f"optimizer file missing: {path}", "rejected_overrides": []}
    try:
        cfg, rejected = sanitize_overrides(template, hypothesis.get("config_overrides"))
    except (TypeError, ValueError) as exc:
        return {"executable": False, "queued": False, "hypothesis_id": hid,
                "reason": f"{type(exc).__name__}: {exc}", "rejected_overrides": []}
    return {
        "executable": True, "queued": False, "structural": False,
        "hypothesis_id": hid, "role": hypothesis.get("role"), "title": hypothesis.get("title"),
        "template": template, "optimizer": path, "config": cfg, "rejected_overrides": rejected,
        "claim": hypothesis.get("claim"), "expected_gain": hypothesis.get("expected_gain"),
        "falsification": hypothesis.get("falsification"),
    }


def default_hypotheses_for_mock() -> list[dict]:
    """Distinct role-coloured proposals used by the mock LLM (and as schema examples)."""
    return [
        {
            "hypothesis_id": "H-math-symmetry",
            "role": "mathematical",
            "title": "Exploit equioscillation: guided blocks at near-max autocorrelation",
            "claim": "Concentrating block moves on the near-max set of c=f*f yields larger gain than uniform probe.",
            "mechanism": "At an equioscillating kink, only moves that reshape the active max set can improve C; GUIDED placement targets that set.",
            "algorithmic_change": "Run blockshift with GUIDED=0.85 and slightly tighter GAP_REL.",
            "template": "blockshift_mutate",
            "config_overrides": {"GUIDED": 0.85, "GAP_REL": 3e-7, "REFRESH_S": 20.0},
            "expected_gain": 2e-8, "expected_gain_lo": 5e-9, "expected_gain_hi": 5e-8,
            "risk": 0.4,
            "assumptions": ["Incumbent remains near-equioscillating", "GUIDED uses top of c"],
            "falsification": "Mean gain over matched seeds ≤ probe mean (+3.9e-9) under same 60s budget.",
            "compute_jobs": 3, "novelty": 0.45,
        },
        {
            "hypothesis_id": "H-num-dgrid",
            "role": "numerical",
            "title": "Finer D-grid and golden refinement for exact block steps",
            "claim": "Increasing D_GRID and GOLDEN_ITERS improves step selection enough to beat default blockshift.",
            "mechanism": "Each proposal already has an exact quartic in D; denser grid + golden search reduces step suboptimality.",
            "algorithmic_change": "blockshift with D_GRID=18, GOLDEN_ITERS=12, slightly wider D range.",
            "template": "blockshift_mutate",
            "config_overrides": {"D_GRID": 18, "GOLDEN_ITERS": 12, "D_LOG10_LO": -13.0, "D_LOG10_HI": -2.5},
            "expected_gain": 1.8e-8, "expected_gain_lo": 8e-9, "expected_gain_hi": 4e-8,
            "risk": 0.35,
            "assumptions": ["Per-proposal overhead still leaves > default proposal rate"],
            "falsification": "Proposals/s drops so far that mean gain < default blockshift mean on matched seeds.",
            "compute_jobs": 3, "novelty": 0.35,
        },
        {
            "hypothesis_id": "H-evo-probe-blocks",
            "role": "evolutionary",
            "title": "Larger stochastic probe blocks with longer polish",
            "claim": "Increasing BLOCK_LOG2_MAX and polish fraction diversifies moves enough to close part of the gap to blockshift.",
            "mechanism": "Heavier-tailed block sizes explore different supports; longer polish reduces late noise.",
            "algorithmic_change": "probe_mutate with BLOCK_LOG2_MAX=14, POLISH_FRAC=0.85.",
            "template": "probe_mutate",
            "config_overrides": {"BLOCK_LOG2_MAX": 14, "POLISH_FRAC": 0.85},
            "expected_gain": 8e-9, "expected_gain_lo": 2e-9, "expected_gain_hi": 2e-8,
            "risk": 0.5,
            "assumptions": ["Random Gaussian block noise is not fully dominated by exact D steps"],
            "falsification": "Mean gain ≤ baseline probe mean on the same three seeds.",
            "compute_jobs": 3, "novelty": 0.4,
        },
        {
            "hypothesis_id": "H-stat-replicate",
            "role": "statistical",
            "title": "Replicate default blockshift on matched seeds (confirmatory design)",
            "claim": "Before mutating further, re-estimate blockshift vs probe effect with the same protocol to bound uncertainty.",
            "mechanism": "n=3 cannot reject H0 at 5%; a predeclared replication reduces false discovery from adaptive search.",
            "algorithmic_change": "Re-run stock blockshift (no mutation) as candidate A vs stock probe as baseline B.",
            "template": "blockshift",
            "config_overrides": {},
            "expected_gain": 1.37e-8, "expected_gain_lo": 5e-9, "expected_gain_hi": 2.5e-8,
            "risk": 0.2,
            "assumptions": ["Modal noise similar to smoke test", "Same incumbent certificate"],
            "falsification": "Paired mean D (blockshift−probe) ≤ 0 on the matched seeds.",
            "compute_jobs": 3, "novelty": 0.15,
            "notes": "Prefer holdout seeds for true confirmation after this exploratory replication.",
        },
        {
            "hypothesis_id": "H-lit-dinkelbach-queue",
            "role": "literature",
            "title": "Queue Dinkelbach/LSE fractional programming (not executable yet)",
            "claim": "Forum/literature fractional-programming attacks may unlock larger gains than local blocks, but need a vetted implementation.",
            "mechanism": "Rewrite C as a fractional program; iterate λ with inner smooth max via LSE (claimed by ClaudeExplorer et al.).",
            "algorithmic_change": "Park until a reviewed optimizer file exists; do not execute LLM code.",
            "template": "queue_novel",
            "config_overrides": {},
            "expected_gain": 1e-5, "expected_gain_lo": 0.0, "expected_gain_hi": 5e-4,
            "risk": 0.85,
            "assumptions": ["Forum scores re-verify", "GPU/CPU budget available"],
            "falsification": "Vetted re-implementation fails to beat blockshift mean under equal 60s CPU budget.",
            "compute_jobs": 0, "novelty": 0.9,
            "literature_refs": [
                {"title": "AlphaEvolve / autocorrelation context", "id": "arXiv:2511.02864",
                 "note": "Related constants; definition checks required"},
                {"title": "EinsteinArena forum claims on Dinkelbach for this problem",
                 "id": "digest:second-autocorrelation-inequality", "note": "Unverified claims"},
            ],
        },
    ]
