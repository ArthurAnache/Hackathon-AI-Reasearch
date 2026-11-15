"""JSON schemas for structured LLM outputs (OpenAI strict mode compatible).

OpenAI strict structured outputs require, for every object:
  - "additionalProperties": false
  - every key in "properties" also listed in "required"
Optional fields use nullable types (["string","null"], etc.), not omitted properties.
"""
from __future__ import annotations

from typing import Any

ROLES = (
    "mathematical",
    "numerical",
    "evolutionary",
    "statistical",
    "literature",
)
TEMPLATE_KINDS = (
    "probe",
    "blockshift",
    "blockshift_mutate",
    "probe_mutate",
    "hybrid_refresh",
    "queue_novel",  # not executed; parked until implementation is vetted
)
DECISIONS = ("CONTINUE", "PIVOT", "MORE_REPLICATION", "STOP")

# Explicit optimizer CONFIG keys allowed in hypotheses (probe ∪ blockshift SEARCH_SPACE).
CONFIG_OVERRIDE_KEYS = (
    "BLOCK_LOG2_MAX",
    "D_LOG10_LO",
    "D_LOG10_HI",
    "D_GRID",
    "GOLDEN_ITERS",
    "GAP_REL",
    "GUIDED",
    "REFRESH_S",
    "POLISH_FRAC",
)


def _req(obj: dict, keys: list[str]) -> list[str]:
    return [k for k in keys if k not in obj]


def _nullable_number() -> dict:
    return {"type": ["number", "null"]}


def _nullable_integer() -> dict:
    return {"type": ["integer", "null"]}


def _nullable_string() -> dict:
    return {"type": ["string", "null"]}


CONFIG_OVERRIDES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(CONFIG_OVERRIDE_KEYS),
    "properties": {
        # Integers (also accept number from the model; we coerce downstream)
        "BLOCK_LOG2_MAX": _nullable_number(),
        "D_GRID": _nullable_number(),
        "GOLDEN_ITERS": _nullable_number(),
        "D_LOG10_LO": _nullable_number(),
        "D_LOG10_HI": _nullable_number(),
        "GAP_REL": _nullable_number(),
        "GUIDED": _nullable_number(),
        "REFRESH_S": _nullable_number(),
        "POLISH_FRAC": _nullable_number(),
    },
}

LITERATURE_REF_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "id", "note"],
    "properties": {
        "title": {"type": "string"},
        "id": {"type": "string"},
        "note": {"type": "string"},
    },
}

HYPOTHESIS_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "hypothesis_id", "role", "title", "claim", "mechanism", "algorithmic_change",
        "template", "config_overrides", "expected_gain", "expected_gain_lo", "expected_gain_hi",
        "risk", "assumptions", "falsification", "compute_jobs", "novelty",
        "literature_refs", "notes",
    ],
    "properties": {
        "hypothesis_id": {"type": "string"},
        "role": {"type": "string", "enum": list(ROLES)},
        "title": {"type": "string"},
        "claim": {"type": "string"},
        "mechanism": {"type": "string"},
        "algorithmic_change": {"type": "string"},
        "template": {"type": "string", "enum": list(TEMPLATE_KINDS)},
        "config_overrides": CONFIG_OVERRIDES_SCHEMA,
        "expected_gain": {"type": "number"},
        "expected_gain_lo": {"type": "number"},
        "expected_gain_hi": {"type": "number"},
        "risk": {"type": "number"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "falsification": {"type": "string"},
        "compute_jobs": {"type": "integer"},
        "novelty": {"type": "number"},
        "literature_refs": {
            "type": ["array", "null"],
            "items": LITERATURE_REF_SCHEMA,
        },
        "notes": _nullable_string(),
    },
}

CRITIQUE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "reviewer_role", "target_id", "validity", "novelty", "expected_gain",
        "compute_cost", "testability", "objections", "score",
    ],
    "properties": {
        "reviewer_role": {"type": "string"},
        "target_id": {"type": "string"},
        "validity": {"type": "number"},
        "novelty": {"type": "number"},
        "expected_gain": {"type": "number"},
        "compute_cost": {"type": "number"},
        "testability": {"type": "number"},
        "objections": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "number"},
    },
}

REVISION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hypothesis_id", "role", "response", "revised", "changes", "hypothesis"],
    "properties": {
        "hypothesis_id": {"type": "string"},
        "role": {"type": "string"},
        "response": {"type": "string"},
        "revised": {"type": "boolean"},
        "changes": {"type": "array", "items": {"type": "string"}},
        "hypothesis": {"anyOf": [HYPOTHESIS_JSON_SCHEMA, {"type": "null"}]},
    },
}

VOTE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["voter_role", "ranking", "rationale"],
    "properties": {
        "voter_role": {"type": "string"},
        "ranking": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
}

STOP_VOTE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "role", "decision", "additional_jobs", "rationale",
        "expected_information_gain", "remaining_budget_note", "selected_hypothesis_ids",
    ],
    "properties": {
        "role": {"type": "string"},
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "additional_jobs": {"type": "integer"},
        "rationale": {"type": "string"},
        "expected_information_gain": {"type": "number"},
        "remaining_budget_note": {"type": "string"},
        "selected_hypothesis_ids": {
            "type": ["array", "null"],
            "items": {"type": "string"},
        },
    },
}


# One reviewer critiques every other proposal in one request (5 requests per council instead of 20).
CRITIQUE_BATCH_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["critiques"],
    "properties": {"critiques": {"type": "array", "items": CRITIQUE_JSON_SCHEMA}},
}

MANAGER_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["selected_ids", "seeds_per_strategy", "budget_s_request", "rationale", "rejections"],
    "properties": {
        "selected_ids": {"type": "array", "items": {"type": "string"}},
        "seeds_per_strategy": {"type": "integer"},
        "budget_s_request": {"type": "number"},
        "rationale": {"type": "string"},
        "rejections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypothesis_id", "reason"],
                "properties": {"hypothesis_id": {"type": "string"}, "reason": {"type": "string"}},
            },
        },
    },
}


def assert_openai_strict_compatible(schema: dict, path: str = "$") -> list[str]:
    """Return violations of OpenAI strict structured-output rules (empty = ok)."""
    errs: list[str] = []

    def walk(node: Any, p: str) -> None:
        if not isinstance(node, dict):
            return
        # unwrap anyOf / oneOf branches
        for key in ("anyOf", "oneOf", "allOf"):
            if key in node:
                for i, branch in enumerate(node[key]):
                    walk(branch, f"{p}.{key}[{i}]")
        types = node.get("type")
        type_list = types if isinstance(types, list) else ([types] if types else [])
        if "object" in type_list or (types is None and "properties" in node):
            if node.get("additionalProperties") is not False:
                errs.append(f"{p}: object must set additionalProperties false")
            props = node.get("properties") or {}
            required = node.get("required")
            if required is None:
                errs.append(f"{p}: object must list required")
            else:
                missing = sorted(set(props) - set(required))
                extra = sorted(set(required) - set(props))
                if missing:
                    errs.append(f"{p}: properties not in required: {missing}")
                if extra:
                    errs.append(f"{p}: required keys missing from properties: {extra}")
            for name, sub in props.items():
                walk(sub, f"{p}.properties.{name}")
        if "items" in node:
            walk(node["items"], f"{p}.items")

    walk(schema, path)
    return errs


def normalize_config_overrides(raw: Any) -> dict:
    """Drop nulls / unknown keys; keep only supported optimizer parameters."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("config_overrides must be an object")
    out = {}
    for k, v in raw.items():
        if k not in CONFIG_OVERRIDE_KEYS:
            continue
        if v is None:
            continue
        out[k] = v
    return out


def normalize_hypothesis(h: dict) -> dict:
    h = dict(h)
    h["config_overrides"] = normalize_config_overrides(h.get("config_overrides"))
    if h.get("literature_refs") is None:
        h["literature_refs"] = []
    if "notes" not in h:
        h["notes"] = None
    return h


def validate_hypothesis(h: dict) -> list[str]:
    h = normalize_hypothesis(h)
    errs = _req(h, [
        "hypothesis_id", "role", "title", "claim", "mechanism", "algorithmic_change",
        "template", "config_overrides", "expected_gain", "expected_gain_lo", "expected_gain_hi",
        "risk", "assumptions", "falsification", "compute_jobs", "novelty",
    ])
    if h.get("role") not in ROLES:
        errs.append(f"role must be one of {ROLES}")
    if h.get("template") not in TEMPLATE_KINDS:
        errs.append(f"template must be one of {TEMPLATE_KINDS}")
    if not isinstance(h.get("config_overrides"), dict):
        errs.append("config_overrides must be a dict")
    else:
        unknown = sorted(set(h["config_overrides"]) - set(CONFIG_OVERRIDE_KEYS))
        if unknown:
            errs.append(f"config_overrides unknown keys: {unknown}")
    for k in ("expected_gain", "expected_gain_lo", "expected_gain_hi", "risk", "novelty"):
        if k in h and not isinstance(h[k], (int, float)):
            errs.append(f"{k} must be numeric")
    if not isinstance(h.get("assumptions"), list):
        errs.append("assumptions must be a list")
    if not isinstance(h.get("falsification"), str) or not h.get("falsification"):
        errs.append("falsification must be a non-empty string")
    refs = h.get("literature_refs")
    if refs is not None and not isinstance(refs, list):
        errs.append("literature_refs must be a list or null")
    return errs


def validate_critique(c: dict) -> list[str]:
    errs = _req(c, ["reviewer_role", "target_id", "validity", "novelty", "expected_gain",
                     "compute_cost", "testability", "objections", "score"])
    if not isinstance(c.get("objections"), list):
        errs.append("objections must be a list")
    if not isinstance(c.get("score"), (int, float)):
        errs.append("score must be numeric")
    return errs


def validate_revision(r: dict) -> list[str]:
    errs = _req(r, ["hypothesis_id", "role", "response", "revised", "changes"])
    if r.get("revised") and r.get("hypothesis"):
        errs += [f"revised.{e}" for e in validate_hypothesis(r["hypothesis"])]
    return errs


def validate_vote(v: dict) -> list[str]:
    errs = _req(v, ["voter_role", "ranking", "rationale"])
    if not isinstance(v.get("ranking"), list) or len(v.get("ranking") or []) < 1:
        errs.append("ranking must be a non-empty list of hypothesis ids")
    return errs


def validate_stop_vote(v: dict) -> list[str]:
    errs = _req(v, ["role", "decision", "additional_jobs", "rationale",
                     "expected_information_gain", "remaining_budget_note"])
    if v.get("decision") not in DECISIONS:
        errs.append(f"decision must be one of {DECISIONS}")
    aj = v.get("additional_jobs")
    if not isinstance(aj, int) or not 0 <= aj <= 40:
        errs.append("additional_jobs must be int in 0..40")
    return errs


def validate_manager(m: dict) -> list[str]:
    errs = _req(m, ["selected_ids", "seeds_per_strategy", "budget_s_request", "rationale", "rejections"])
    if not isinstance(m.get("selected_ids"), list):
        errs.append("selected_ids must be a list")
    if not isinstance(m.get("seeds_per_strategy"), int):
        errs.append("seeds_per_strategy must be an integer")
    return errs


def ensure(kind: str, obj: Any) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"{kind}: expected object, got {type(obj).__name__}")
    if kind == "hypothesis":
        obj = normalize_hypothesis(obj)
    if kind == "revision" and obj.get("hypothesis"):
        obj = dict(obj)
        obj["hypothesis"] = normalize_hypothesis(obj["hypothesis"])
    if kind == "stop_vote" and obj.get("selected_hypothesis_ids") is None:
        obj = dict(obj)
        obj["selected_hypothesis_ids"] = []
    validators = {
        "hypothesis": validate_hypothesis,
        "critique": validate_critique,
        "revision": validate_revision,
        "vote": validate_vote,
        "stop_vote": validate_stop_vote,
        "manager": validate_manager,
    }
    errs = validators[kind](obj)
    if errs:
        raise ValueError(f"{kind} invalid: " + "; ".join(errs))
    return obj
