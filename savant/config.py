"""SAVANT configuration: paths, budgets, safety caps. No secrets here."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT = Path(
    r"C:\Users\thiba\Desktop\Berkeley\AIRESEARCHACK\ws3-v2-runner"
    r"\runs\ws3\20261115T015342Z-b60\export_20261115T015703Z"
)
RUNS_DIR = ROOT / "runs" / "savant"
PROBLEM = "second-autocorrelation-inequality"
BASELINE_OPTIMIZERS = (
    "optimizers/second-autocorrelation-inequality_probe.py",
    "optimizers/second-autocorrelation-inequality_blockshift.py",
)
MATCHED_SEEDS = (101, 202, 303)
HOLDOUT_SEEDS = (404, 505, 606)  # confirmatory; never used for adaptive tuning
# Ten matched seeds for a wave. 101/202/303 stay comparable with the smoke test.
# 404/505/606 stay holdouts. The other seven are new paired seeds.
CAMPAIGN_SEEDS = (101, 202, 303, 707, 808, 909, 1001, 1102, 1203, 1304)

# List prices (USD / second). Same figures as ws3_bench; not re-checked against modal.com.
PRICE_CPU_CORE_S = 3.94e-5
PRICE_MEM_GIB_S = 6.72e-6
MODAL_STARTUP_S = 45.0


@dataclass
class BudgetCaps:
    max_iterations: int = 10
    max_strategies_per_iteration: int = 4
    max_jobs_per_iteration: int = 40
    max_jobs_total: int = 400
    # 400 jobs at the expanded 300 s budget. The dollar cap can stop a wave earlier.
    max_search_seconds_total: float = 120_000.0
    budget_s_per_job: float = 60.0
    budget_s_expanded: float = 300.0
    seeds_per_strategy: int = 10
    cpu: float = 1.0
    memory_mb: int = 4096
    max_containers: int = 40
    max_llm_usd: float = 50.0
    # High enough that the $50 estimate binds first at the configured token prices.
    max_llm_tokens: int = 40_000_000
    max_modal_usd: float = 50.0
    # conservative alpha spending across up to 10 planned decision points
    alpha_total: float = 0.05
    alpha_spend_per_decision: float = 0.005  # 0.05 / 10


def estimate_modal_usd(n_jobs: int, budget_s: float, caps: BudgetCaps | None = None) -> float:
    """Committed cost of a wave: billed max(request, estimate), startup included. Hard-stop input."""
    caps = caps or BudgetCaps()
    seconds = n_jobs * (budget_s + MODAL_STARTUP_S)
    per_s = caps.cpu * PRICE_CPU_CORE_S + (caps.memory_mb / 1024) * PRICE_MEM_GIB_S
    return seconds * per_s


# USD per 1M tokens, Standard tier. gpt-6-astra: developers.openai.com/api/docs/models/gpt-6-astra
# (checked 2026-11-14). Reasoning tokens are billed as output. Above long_context_tokens of input the whole
# request is billed at the long multipliers.
MODEL_PRICES = {
    "gpt-6-astra": {"input": 10.0, "cached_input": 1.0, "cache_write": 12.5, "output": 50.0,
                    "long_context_tokens": 272_000, "long_input_mult": 2.0, "long_output_mult": 1.5},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "cache_write": 0.40, "output": 1.60,
                     "long_context_tokens": None, "long_input_mult": 1.0, "long_output_mult": 1.0},
}
REASONING_MODELS = ("gpt-6-astra",)

ASTRA_MODEL = "gpt-6-astra"
ASTRA_EFFORT = "max"
# Reasoning + visible JSON share this limit. Model maximum is 128k; 64k leaves room for max effort
# while keeping one request's reservation at about $3.2 + input.
ASTRA_MAX_OUTPUT_TOKENS = 64_000
# Approved by Thibault 2026-11-14: proposals, manager and Continue/Pivot/Stop stay at max; routine council
# stages use xhigh on the same model. Keys are `purpose` prefixes.
ASTRA_EFFORT_BY_PURPOSE = {"critique:": "xhigh", "revise:": "xhigh", "vote:": "xhigh"}


@dataclass
class LLMConfig:
    provider: str = "openai"  # openai | mock | anthropic
    model: str = "gpt-4.1-mini"
    fallback_model: str = "gpt-4.1-mini"  # never used automatically
    temperature: float = 0.4
    max_output_tokens: int = 2500
    reasoning_effort: str | None = None
    # approximate list prices USD / 1M tokens; MODEL_PRICES wins when the model is listed
    input_usd_per_mtok: float = 0.40
    output_usd_per_mtok: float = 1.60
    cached_input_usd_per_mtok: float = 0.10
    request_timeout_s: float = 1800.0
    effort_by_purpose: dict = field(default_factory=dict)

    def effort_for(self, purpose: str | None) -> str | None:
        for prefix, effort in self.effort_by_purpose.items():
            if purpose and purpose.startswith(prefix):
                return effort
        return self.reasoning_effort

    def prices(self) -> dict:
        p = MODEL_PRICES.get(self.model)
        if p:
            return p
        return {"input": self.input_usd_per_mtok, "cached_input": self.cached_input_usd_per_mtok,
                "cache_write": self.input_usd_per_mtok * 1.25, "output": self.output_usd_per_mtok,
                "long_context_tokens": None, "long_input_mult": 1.0, "long_output_mult": 1.0}

    @property
    def is_reasoning(self) -> bool:
        return self.model in REASONING_MODELS or self.reasoning_effort is not None


def astra_config(base: LLMConfig | None = None) -> LLMConfig:
    base = base or LLMConfig()
    return LLMConfig(provider="openai", model=ASTRA_MODEL, fallback_model=ASTRA_MODEL,
                     temperature=base.temperature, max_output_tokens=ASTRA_MAX_OUTPUT_TOKENS,
                     reasoning_effort=ASTRA_EFFORT, request_timeout_s=base.request_timeout_s,
                     effort_by_purpose=dict(ASTRA_EFFORT_BY_PURPOSE))


@dataclass
class SavantConfig:
    problem: str = PROBLEM
    runs_dir: Path = field(default_factory=lambda: RUNS_DIR)
    seed_export: Path = field(default_factory=lambda: DEFAULT_EXPORT)
    matched_seeds: tuple[int, ...] = MATCHED_SEEDS
    campaign_seeds: tuple[int, ...] = CAMPAIGN_SEEDS
    holdout_seeds: tuple[int, ...] = HOLDOUT_SEEDS
    baseline_optimizers: tuple[str, ...] = BASELINE_OPTIMIZERS
    budgets: BudgetCaps = field(default_factory=BudgetCaps)
    llm: LLMConfig = field(default_factory=LLMConfig)
    voting_seed: int = 20261115

    def session_dir(self, session_id: str) -> Path:
        return Path(self.runs_dir) / session_id

    def to_dict(self) -> dict:
        d = asdict(self)
        d["runs_dir"] = str(self.runs_dir)
        d["seed_export"] = str(self.seed_export)
        return d


def load_dotenv(root: Path = ROOT) -> None:
    try:
        from dotenv import load_dotenv as _load
        _load(root / ".env")
    except ImportError:
        pass


def require_openai_key(*, allow_mock: bool = False) -> str:
    load_dotenv()
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        return key
    if allow_mock or os.environ.get("SAVANT_LLM_PROVIDER", "").lower() == "mock":
        return ""
    raise SystemExit(
        "Missing OPENAI_API_KEY. Copy .env.example to .env and set OPENAI_API_KEY=...\n"
        "Or set SAVANT_LLM_PROVIDER=mock for offline dry-runs."
    )


def env_llm_config(base: LLMConfig | None = None) -> LLMConfig:
    load_dotenv()
    cfg = base or LLMConfig()
    return LLMConfig(
        provider=os.environ.get("SAVANT_LLM_PROVIDER", cfg.provider),
        model=os.environ.get("SAVANT_LLM_MODEL", cfg.model),
        fallback_model=os.environ.get("SAVANT_LLM_FALLBACK_MODEL", cfg.fallback_model),
        temperature=float(os.environ.get("SAVANT_LLM_TEMPERATURE", cfg.temperature)),
        max_output_tokens=int(os.environ.get("SAVANT_LLM_MAX_OUTPUT_TOKENS", cfg.max_output_tokens)),
        reasoning_effort=os.environ.get("SAVANT_LLM_REASONING_EFFORT", cfg.reasoning_effort) or None,
        input_usd_per_mtok=cfg.input_usd_per_mtok,
        output_usd_per_mtok=cfg.output_usd_per_mtok,
        cached_input_usd_per_mtok=cfg.cached_input_usd_per_mtok,
        request_timeout_s=cfg.request_timeout_s,
    )
