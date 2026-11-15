"""Schedules, chain specifications, ids, records and cost estimates of the annealing chains. No array code here:
everything in this module runs (and is tested) without torch.

A chain is a list of stages (beta, evaluations). All chains of one batch are evaluated together, one point per
chain and per batched evaluation, so they must have the same total number of evaluations.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

SLUG = "second-autocorrelation-inequality"
VOLUME = "knockout-v2-gpu"

# Modal prices read on https://modal.com/pricing on 2026-11-14 (USD per second).
USD_PER_S = {"H100": 0.001097, "A100-80GB": 0.000694, "cpu_core": 0.0000131, "mem_gib": 0.00000222}

SHARED_FIELDS = ("experiment_id", "hypothesis", "config", "seed", "budget_s", "valid", "score", "best_known_score",
                 "decision", "rationale", "evidence_urls")
GPU_FIELDS = ("run_id", "source_commit", "device", "gpu_type", "dtype", "torch_version", "cuda_version",
              "gpu_objective", "wall_s", "solution_path")


@dataclass(frozen=True)
class Stage:
    beta: float
    evals: int


@dataclass(frozen=True)
class ChainSpec:
    name: str
    start: str                      # "frozen" (certificate, EA solution 2782), "live" (live #1), or "vol:<path>.npy"
    stages: tuple
    floor: float = 0.0              # added to f before the chain, in units of the start's mean non-zero value
    seed: int = 0
    hypothesis: str = ""
    jitter: float = 0.0             # kick before the chain: f_i <- f_i * exp(jitter * N(0, 1)), drawn from `seed`
    reopen: float = 0.0             # periodic opening of closed cells with a positive derivative (chain.py)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stages"] = [[s.beta, s.evals] for s in self.stages]
        return d

    @staticmethod
    def from_dict(d: dict) -> "ChainSpec":
        stages = tuple(Stage(float(b), int(e)) for b, e in d["stages"])
        return ChainSpec(name=d["name"], start=d["start"], stages=stages, floor=float(d.get("floor", 0.0)),
                         seed=int(d.get("seed", 0)), hypothesis=d.get("hypothesis", ""),
                         jitter=float(d.get("jitter", 0.0)), reopen=float(d.get("reopen", 0.0)))


def geometric(beta_start: float, beta_end: float, n: int) -> list[float]:
    """n values from beta_start to beta_end, both included, in geometric progression."""
    if n < 1 or beta_start <= 0 or beta_end <= 0:
        raise ValueError("need n >= 1 and positive betas")
    if n == 1:
        return [float(beta_end)]
    ratio = (beta_end / beta_start) ** (1.0 / (n - 1))
    out = [beta_start * ratio ** k for k in range(n)]
    out[-1] = float(beta_end)
    return out


def split_evals(total: int, parts: int) -> list[int]:
    """`total` evaluations split into `parts` stages as evenly as possible (the first stages get the remainder)."""
    if parts < 1 or total < parts:
        raise ValueError(f"cannot split {total} evaluations into {parts} stages")
    base, rest = divmod(total, parts)
    return [base + (1 if k < rest else 0) for k in range(parts)]


def anneal(beta_start: float, beta_end: float, per_decade: float, evals: int,
           polish: tuple = (), polish_evals: int = 0) -> tuple:
    """A cooling schedule: beta rises geometrically from beta_start to beta_end with `per_decade` stages per
    factor 10, sharing `evals` evaluations equally; then optional polish stages share `polish_evals`."""
    decades = math.log10(beta_end / beta_start)
    if decades < 0:
        raise ValueError("beta_end must be >= beta_start (cooling means beta goes up)")
    n = max(1, int(round(decades * per_decade)) + 1)
    stages = [Stage(b, e) for b, e in zip(geometric(beta_start, beta_end, n), split_evals(evals, n))]
    if polish:
        stages += [Stage(float(b), e) for b, e in zip(polish, split_evals(polish_evals, len(polish)))]
    return tuple(stages)


def total_evals(stages) -> int:
    return sum(s.evals for s in stages)


def boundaries(stages) -> list[int]:
    """Cumulative evaluation count at the end of each stage."""
    out, acc = [], 0
    for s in stages:
        if s.evals < 1:
            raise ValueError("every stage needs at least one evaluation")
        acc += s.evals
        out.append(acc)
    return out


def check_batch(chains: list[ChainSpec]) -> int:
    """All chains of a batch share the evaluation budget and have distinct names. Returns that budget."""
    if not chains:
        raise ValueError("empty batch")
    totals = {total_evals(c.stages) for c in chains}
    if len(totals) != 1:
        raise ValueError(f"chains of one batch must have the same total evaluations, got {sorted(totals)}")
    names = [c.name for c in chains]
    if len(set(names)) != len(names):
        raise ValueError("chain names must be distinct")
    for c in chains:
        if c.floor < 0 or c.jitter < 0 or c.reopen < 0:
            raise ValueError("floor, jitter and reopen must be >= 0")
    return totals.pop()


def experiment_id(run_id: str, chain: str, step: str) -> str:
    """Global id <run_id>/<chain>/<step> (CLAUDE.md, WS3 rule 9). Also the solution path, plus '.npy'."""
    for part in (run_id, chain, step):
        if not part or "/" in part or part.startswith("."):
            raise ValueError(f"bad id part {part!r}")
    return f"{run_id}/{chain}/{step}"


def solution_path(exp_id: str) -> str:
    return exp_id + ".npy"


def step_name(stage_index: int) -> str:
    return f"s{stage_index:03d}"


def container_usd_per_s(gpu: str = "H100", cpu: float = 6.0, memory_gib: float = 32.0) -> float:
    return USD_PER_S[gpu] + cpu * USD_PER_S["cpu_core"] + memory_gib * USD_PER_S["mem_gib"]


def estimate(evals: int, s_per_eval: float, overhead_s: float = 240.0, usd_per_s: float | None = None) -> dict:
    """Wall time and cost ESTIMATE of one container running `evals` batched evaluations."""
    rate = container_usd_per_s() if usd_per_s is None else usd_per_s
    wall = evals * s_per_eval + overhead_s
    return {"wall_s": wall, "usd": wall * rate, "usd_per_s": rate}


def make_record(*, run_id: str, chain: ChainSpec, step: str, score, valid: bool, gpu_objective, gpu_exact,
                start_score: float, best_known_score: float, env: dict, wall_s: float, budget_s: float,
                source_commit: str, extra: dict | None = None, solution: str | None = None,
                evidence_urls: list | None = None) -> dict:
    """One line of the chain table: the shared JSONL fields plus the WS3 rule 9 fields. `score` is the frozen
    verifier's float64 score on CPU and the only number that counts; `gpu_objective` is the smoothed value."""
    exp_id = experiment_id(run_id, chain.name, step)
    gain_start = None if score is None else score - start_score
    gain_live = None if score is None else score - best_known_score
    if not valid or score is None:
        decision, rationale = "discard", "the frozen verifier rejected the candidate"
    elif gain_start > 0:
        decision = "keep"
        rationale = (f"exact score above its start by {gain_start:.3e}; gain over the live #1 {gain_live:.3e} "
                     f"(a record needs >= 1e-5 and the arbiter's verdict)")
    else:
        decision, rationale = "discard", f"exact score not above its start ({gain_start:.3e})"
    rec = {
        "experiment_id": exp_id, "hypothesis": chain.hypothesis, "config": chain.to_dict(), "seed": chain.seed,
        "budget_s": budget_s, "valid": bool(valid), "score": score, "best_known_score": best_known_score,
        "decision": decision, "rationale": rationale, "evidence_urls": list(evidence_urls or []),
        "run_id": run_id, "source_commit": source_commit, "device": env.get("device"),
        "gpu_type": env.get("gpu_type"), "dtype": env.get("dtype"), "torch_version": env.get("torch_version"),
        "cuda_version": env.get("cuda_version"), "gpu_objective": gpu_objective, "wall_s": wall_s,
        "solution_path": solution if solution is not None else solution_path(exp_id),
        "chain": chain.name, "step": step, "parent": chain.start, "gpu_exact": gpu_exact,
        "start_score": start_score, "gain_over_start": gain_start, "gain_over_live": gain_live,
    }
    rec.update(extra or {})
    missing = [k for k in SHARED_FIELDS + GPU_FIELDS if k not in rec]
    if missing:
        raise ValueError(f"record misses fields {missing}")
    return rec


# ---------------------------------------------------------------------------------------------------------------
# Plans: named batches of chains. `evals` is the evaluation budget of every chain of the batch.

POLISH = (1e9, 3e9, 1e10, 3e10, 1e11)


def _with_polish(beta_start: float, beta_end: float, per_decade: float, evals: int, polish_share: float) -> tuple:
    polish_evals = max(len(POLISH), int(evals * polish_share))
    tail = tuple(b for b in POLISH if b > beta_end)
    if not tail:
        return anneal(beta_start, beta_end, per_decade, evals)
    return anneal(beta_start, beta_end, per_decade, evals - polish_evals, tail, polish_evals)


def plan_smoke(start: str, evals: int) -> list[ChainSpec]:
    """Plumbing check: a plain polish and a mild reheat, two stages each."""
    half = evals // 2
    return [
        ChainSpec("polish", start, (Stage(1e9, half), Stage(3e9, evals - half)),
                  hypothesis="control: polish only, no reheat"),
        ChainSpec("mild", start, (Stage(1e7, half), Stage(1e9, evals - half)),
                  hypothesis="plumbing: mild reheat at beta = 1e7, then beta = 1e9"),
    ]


def plan_deep(start: str, evals: int) -> list[ChainSpec]:
    """E1 / E2: deep reheat-then-cool cycles that differ in the starting beta, the cooling speed and whether
    zero cells may open, plus a polish-only control with the same evaluation budget."""
    chains = [ChainSpec("control", start, anneal(1e9, 1e11, 2, evals),
                        hypothesis="control: no reheat, polish from beta = 1e9 to 1e11 with the same budget")]
    for tag, b0 in (("b4", 1e4), ("b5", 1e5), ("b6", 1e6)):
        chains.append(ChainSpec(
            f"{tag}-slow", start, _with_polish(b0, 1e9, 3, evals, 0.25),
            hypothesis=f"deep reheat at beta = {b0:.0e}, one slow cooling to 1e9 (3 stages per decade), "
                       f"then polish to 1e11; support fixed (f = w^2 keeps zero cells at zero)"))
    for tag, b0 in (("b4", 1e4), ("b5", 1e5), ("b6", 1e6)):
        chains.append(ChainSpec(
            f"{tag}-open", start, _with_polish(b0, 1e9, 3, evals, 0.25), floor=1e-3,
            hypothesis=f"deep reheat at beta = {b0:.0e} with zero cells opened (floor 1e-3 of the mean "
                       f"non-zero value), one slow cooling to 1e9, then polish to 1e11"))
    half = evals // 2
    fast = _with_polish(1e5, 1e9, 2, half, 0.25) + _with_polish(1e5, 1e9, 2, evals - half, 0.25)
    chains.append(ChainSpec(
        "b5-open-2cycles", start, fast, floor=1e-6,
        hypothesis="two fast reheat-cool cycles from beta = 1e5 (2 stages per decade), zero cells opened with a "
                   "tiny floor (1e-6 of the mean non-zero value)"))
    return chains


def _cycles(beta_start: float, per_decade: float, evals: int, count: int) -> tuple:
    """`count` reheat-cool cycles in a row, sharing `evals` evaluations."""
    out = ()
    for part in split_evals(evals, count):
        out += _with_polish(beta_start, 1e9, per_decade, part, 0.25)
    return out


def plan_deep16(start: str, evals: int) -> list[ChainSpec]:
    """E1 / E2 on one H100 (E0: one chain already saturates the GPU, so 16 chains cost about the same per chain
    as 1). One polish-only control, five reheat depths with the support fixed, the same depths with zero cells
    opened at three floor levels, and two chains made of four fast cycles."""
    slow = "one slow cooling to 1e9 (3 stages per decade, 75 % of the budget), then polish to 1e11"
    chains = [ChainSpec("control", start, anneal(1e9, 1e11, 2, evals),
                        hypothesis="control: no reheat, polish from beta = 1e9 to 1e11 with the same budget")]
    depths = (("b7", 1e7), ("b6", 1e6), ("b5", 1e5), ("b4", 1e4), ("b3", 1e3))
    for tag, b0 in depths:
        chains.append(ChainSpec(f"{tag}-fixed", start, _with_polish(b0, 1e9, 3, evals, 0.25),
                                hypothesis=f"reheat at beta = {b0:.0e}, {slow}; support fixed (f = w^2 keeps "
                                           f"zero cells at zero)"))
    for tag, b0 in depths[1:]:
        chains.append(ChainSpec(f"{tag}-open3", start, _with_polish(b0, 1e9, 3, evals, 0.25), floor=1e-3,
                                hypothesis=f"reheat at beta = {b0:.0e} with zero cells opened (floor 1e-3 of the "
                                           f"mean non-zero value), {slow}"))
    for tag, b0 in depths[1:4]:
        chains.append(ChainSpec(f"{tag}-open6", start, _with_polish(b0, 1e9, 3, evals, 0.25), floor=1e-6,
                                hypothesis=f"reheat at beta = {b0:.0e} with zero cells opened (floor 1e-6 of the "
                                           f"mean non-zero value), {slow}"))
    chains.append(ChainSpec("b5-open2", start, _with_polish(1e5, 1e9, 3, evals, 0.25), floor=1e-2,
                            hypothesis=f"reheat at beta = 1e5 with zero cells opened wide (floor 1e-2 of the mean "
                                       f"non-zero value), {slow}"))
    for tag, b0 in depths[2:4]:
        chains.append(ChainSpec(f"{tag}-open3-4cycles", start, _cycles(b0, 2, evals, 4), floor=1e-3,
                                hypothesis=f"four fast reheat-cool cycles from beta = {b0:.0e} (2 stages per "
                                           f"decade), zero cells opened (floor 1e-3) before the first cycle"))
    return chains


def plan_kick16(start: str, evals: int) -> list[ChainSpec]:
    """Stochastic reheat. Solutions 2782 and 2783 have the same envelope (mass per block of 1,024 cells within
    1 %) and differ like weakly correlated cell-level noise (log-ratio quartiles 0.79 / 1.24). So: kick every
    cell by a log-normal factor of width sigma, then cool slowly from a moderate beta. Several seeds per sigma
    show the spread of the basins reached; a polish-only control shares the budget."""
    chains = [ChainSpec("control", start, anneal(1e9, 1e11, 2, evals),
                        hypothesis="control: no kick, polish from beta = 1e9 to 1e11 with the same budget")]

    def kick(sigma, b0, seed, floor=0.0):
        tag = f"k{sigma:g}-b{int(round(math.log10(b0)))}-s{seed}" + ("-open3" if floor else "")
        opened = f", zero cells opened (floor {floor:g})" if floor else ""
        return ChainSpec(tag, start, _with_polish(b0, 1e9, 3, evals, 0.25), floor=floor, seed=seed, jitter=sigma,
                         hypothesis=f"log-normal cell kick of width {sigma:g} (seed {seed}){opened}, then one "
                                    f"slow cooling from beta = {b0:.0e} to 1e9 and a polish to 1e11")

    chains += [kick(0.03, 1e5, 1)]
    chains += [kick(0.1, 1e5, s) for s in (1, 2, 3)]
    chains += [kick(0.3, 1e5, s) for s in (1, 2, 3)]
    chains += [kick(1.0, 1e5, s) for s in (1, 2)]
    chains += [kick(0.3, 1e4, s) for s in (1, 2)]
    chains += [kick(0.3, 1e6, s) for s in (1, 2)]
    chains += [kick(0.1, 1e5, 4, floor=1e-3), kick(0.3, 1e5, 4, floor=1e-3)]
    return chains


def plan_reopen16(start: str, evals: int) -> list[ChainSpec]:
    """Support moves driven by the first-order condition, with and without a reheat, plus a few kicks.

    At a point polished with the support fixed, 1 to 2 % of the zero cells of the live #1 have a positive
    derivative (14,000 to 37,000 cells; 23,000 cells appear between solutions 2782 and 2783). `reopen` opens
    exactly those cells, every 500 evaluations, instead of flooring all 1.5 million zero cells."""
    chains = [ChainSpec("control", start, anneal(1e9, 1e11, 2, evals),
                        hypothesis="control: no reheat, support fixed, polish from beta = 1e9 to 1e11")]
    for eta in (1e-4, 1e-3, 1e-2):
        chains.append(ChainSpec(f"polish-reopen{eta:g}", start, anneal(1e9, 1e11, 2, evals), reopen=eta,
                                hypothesis=f"polish from beta = 1e9 to 1e11, closed cells with a positive "
                                           f"derivative opened every 500 evaluations (eta = {eta:g})"))
    for tag, b0, etas in (("b8", 1e8, (1e-3, 1e-2)), ("b7", 1e7, (1e-3, 1e-2)), ("b6", 1e6, (1e-3, 1e-2)),
                          ("b5", 1e5, (1e-2,)), ("b4", 1e4, (1e-2,))):
        for eta in etas:
            chains.append(ChainSpec(
                f"{tag}-reopen{eta:g}", start, _with_polish(b0, 1e9, 3, evals, 0.25), reopen=eta,
                hypothesis=f"reheat at beta = {b0:.0e}, slow cooling to 1e9, polish to 1e11; closed cells with a "
                           f"positive derivative opened every 500 evaluations (eta = {eta:g})"))

    def kick(sigma, seed, eta):
        tag = f"k{sigma:g}-b5-s{seed}" + (f"-reopen{eta:g}" if eta else "")
        return ChainSpec(tag, start, _with_polish(1e5, 1e9, 3, evals, 0.25), seed=seed, jitter=sigma, reopen=eta,
                         hypothesis=f"log-normal cell kick of width {sigma:g} (seed {seed}), slow cooling from "
                                    f"beta = 1e5 to 1e9, polish to 1e11" + (f"; reopen eta = {eta:g}" if eta else ""))

    chains += [kick(0.1, 1, 0.0), kick(0.3, 1, 0.0), kick(0.1, 1, 1e-2), kick(0.3, 1, 1e-2)]
    return chains


PLANS = {"smoke": plan_smoke, "deep": plan_deep, "deep16": plan_deep16, "kick16": plan_kick16,
         "reopen16": plan_reopen16}


def build_plan(name: str, start: str, evals: int) -> list[ChainSpec]:
    if name not in PLANS:
        raise ValueError(f"unknown plan {name!r} (known: {sorted(PLANS)})")
    chains = PLANS[name](start, evals)
    check_batch(chains)
    return chains
