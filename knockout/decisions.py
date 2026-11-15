"""Every decision of the loop as a pure function of logged inputs and a logged seed.

The live loop calls these functions and writes their inputs and outputs to events.jsonl. `knockout.replay` calls the
same functions on the recorded inputs and fails when an output differs. Nothing here reads a clock, a file or an
API, so a decision can always be re-derived from its event.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta as _beta

PHASES = ("explore", "climb", "plateau")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def sign(direction: str) -> float:
    return 1.0 if direction == "maximize" else -1.0


def gain(new: float, old: float, direction: str) -> float:
    return sign(direction) * (new - old)


def improves(new: float | None, old: float | None, eps_rel: float, direction: str) -> bool:
    """`new` beats `old` by more than EPS (relative to max(1, |old|)). A missing `new` never wins."""
    if new is None:
        return False
    if old is None:
        return True
    return gain(new, old, direction) > eps_rel * max(1.0, abs(old))


# ---------------------------------------------------------------------------------------------- bandits (5.6, 5.7)
def thompson_pick(arms: dict[str, list[float]], seed: int, weights: dict[str, float] | None = None) -> dict:
    """Thompson sampling over Beta(alpha, beta) arms. `weights` multiply the draws (island budget shares)."""
    rng = _rng(seed)
    draws = {}
    for name in sorted(arms):
        a, b = arms[name]
        draws[name] = float(rng.beta(max(a, 1e-6), max(b, 1e-6))) * float((weights or {}).get(name, 1.0))
    choice = max(sorted(draws), key=lambda k: draws[k])
    return {"choice": choice, "draws": draws}


def bandit_update(arms: dict[str, list[float]], chosen: str, successes: float, failures: float,
                  discount: float, floor: float = 1.0) -> dict[str, list[float]]:
    """Discounted update: every arm forgets a little, then the chosen arm gets its successes and failures.
    A parameter above `floor` never decays below it, so an arm falls back to Beta(1, 1), not to Beta(0, 0)."""
    def decay(x: float) -> float:
        return x if x <= floor else max(floor, discount * x)

    out = {k: [decay(v[0]), decay(v[1])] for k, v in arms.items()}
    out[chosen] = [out[chosen][0] + successes, out[chosen][1] + failures]
    return out


def select_parent(ranked_ids: list[str], power: float, seed: int) -> dict:
    """Rank-based power law: the member of rank r (0 = best) is drawn with weight (r + 1) ** -power."""
    w = np.array([(r + 1.0) ** -float(power) for r in range(len(ranked_ids))])
    idx = int(_rng(seed).choice(len(ranked_ids), p=w / w.sum()))
    return {"choice": ranked_ids[idx], "rank": idx}


# ---------------------------------------------------------------------------------------------- phases (5.3)
def phase(budget_fraction: float, evals_since_improvement: int, explore_until: float, window: int) -> str:
    if budget_fraction < explore_until:
        return "explore"
    return "climb" if evals_since_improvement < window else "plateau"


# ---------------------------------------------------------------------------------------------- genes (5.3)
def gene_posterior(stats: dict[str, list[float]], current_phase: str, phase_min_obs: int) -> list[float]:
    """Beta parameters used for a gene in a phase. `stats[phase] = [a, b, n]` holds discounted pseudo-counts and
    the number of direct observations; the posterior is Beta(1 + a, 1 + b). With fewer than `phase_min_obs`
    observations in the phase, the counts of all phases are pooled."""
    a, b, n = stats.get(current_phase, [0.0, 0.0, 0])
    if n < phase_min_obs:
        a = sum(v[0] for v in stats.values())
        b = sum(v[1] for v in stats.values())
    return [1.0 + a, 1.0 + b]


def build_genome(pool: list[dict], g_max: int, seed: int) -> dict:
    """Thompson genome construction. `pool` items: {"id", "alpha", "beta", "linked", "incompatible"}.
    Genes are taken by decreasing draw; a gene brings its linked genes with it, and a gene incompatible with one
    already chosen is skipped."""
    rng = _rng(seed)
    by_id = {g["id"]: g for g in sorted(pool, key=lambda g: g["id"])}
    draws = {gid: float(rng.beta(g["alpha"], g["beta"])) for gid, g in by_id.items()}
    chosen: list[str] = []

    def compatible(gid: str) -> bool:
        return all(gid not in by_id[c].get("incompatible", []) and c not in by_id[gid].get("incompatible", [])
                   for c in chosen)

    for gid in sorted(draws, key=lambda k: (-draws[k], k)):
        group = [gid] + [x for x in by_id[gid].get("linked", []) if x in by_id]
        group = [x for x in dict.fromkeys(group) if x not in chosen]
        if not group or len(chosen) + len(group) > g_max or not all(compatible(x) for x in group):
            continue
        chosen += group
        if len(chosen) >= g_max:
            break
    return {"genome": sorted(chosen), "draws": draws}


def credit(a: float, b: float, s: float, gamma: float, weight: float = 1.0) -> list[float]:
    """a <- gamma * a + weight * s, b <- gamma * b + weight * (1 - s). weight = 0.5 ** depth for ancestors."""
    return [gamma * a + weight * s, gamma * b + weight * (1.0 - s)]


def linkage(n_joint: int, joint_rate: float, rate_a: float, rate_b: float, min_cooccurrence: int,
            margin: float) -> str | None:
    if n_joint < min_cooccurrence:
        return None
    best_single = max(rate_a, rate_b)
    if joint_rate > best_single + margin:
        return "linked"
    if joint_rate < best_single - margin:
        return "incompatible"
    return None


# ---------------------------------------------------------------------------------------------- gate (5.2)
def pair_outcome(score_with: float | None, score_without: float | None, eps_rel: float, direction: str) -> str:
    """win: the child with the gene beats the other by more than EPS. loss: the reverse. tie: neither."""
    if score_with is None and score_without is None:
        return "tie"
    if improves(score_with, score_without, eps_rel, direction):
        return "win"
    if improves(score_without, score_with, eps_rel, direction):
        return "loss"
    return "tie"


def gate_verdict(wins: int, losses: int, pairs: int, p_validate: float, p_refute: float, min_pairs: int,
                 max_pairs: int) -> dict:
    """Sequential test under Beta(1 + wins, 1 + losses): p = P(theta > 0.5). `pairs` counts ties too."""
    p = float(_beta.sf(0.5, 1 + wins, 1 + losses))
    verdict = None
    if pairs >= min_pairs:
        if p >= p_validate:
            verdict = "validated"
        elif p <= p_refute:
            verdict = "refuted"
    if verdict is None and pairs >= max_pairs:
        verdict = "retired"
    return {"verdict": verdict, "p": p}


# ---------------------------------------------------------------------------------------------- fidelity (5.4)
def promote(scores: dict[str, float], ratio: float, direction: str) -> list[str]:
    """Successive halving: the top `ratio` of the valid children of a batch, at least one."""
    if not scores:
        return []
    k = max(1, int(math.floor(len(scores) * ratio + 1e-9)))
    order = sorted(scores, key=lambda i: (-sign(direction) * scores[i], i))
    return order[:k]


# ---------------------------------------------------------------------------------------------- control arm (5.10)
def control_round(edit_scores: dict[str, float | None], reference: float | None, eps_rel: float,
                  direction: str) -> dict:
    """Hill Sampling: the best edit is kept if it beats the reference by more than EPS."""
    valid = {k: v for k, v in edit_scores.items() if v is not None}
    if not valid:
        return {"improved": False, "chosen": None}
    # first best in submission order, as the arm does
    best = None
    for k in edit_scores:
        if k in valid and (best is None or gain(valid[k], valid[best], direction) > 0):
            best = k
    ok = improves(valid[best], reference, eps_rel, direction)
    return {"improved": ok, "chosen": best if ok else None}


# ---------------------------------------------------------------------------------------------- lineages (5.8)
def should_freeze(evals_since_improvement: int, window: int) -> bool:
    return evals_since_improvement >= window


# ---------------------------------------------------------------------------------------------- migration (5.7)
def migrants(genes: list[dict], min_posterior: float) -> list[str]:
    """Validated non-formulation genes whose posterior mean exceeds the threshold. Items: {"id", "type",
    "status", "alpha", "beta"}."""
    return sorted(g["id"] for g in genes if g["status"] == "validated" and g["type"] != "formulation"
                  and g["alpha"] / (g["alpha"] + g["beta"]) > min_posterior)


def island_to_replace(islands: list[dict], min_evals: int) -> str | None:
    """The island with the lowest posterior mean among those with at least `min_evals` evaluations (never the
    canonical one). Items: {"id", "alpha", "beta", "evals", "canonical"}."""
    ok = [i for i in islands if i["evals"] >= min_evals and not i["canonical"]]
    if not ok:
        return None
    return min(sorted(ok, key=lambda i: i["id"]), key=lambda i: i["alpha"] / (i["alpha"] + i["beta"]))["id"]


# ---------------------------------------------------------------------------------------------- promotion (5.11)
def promotion(beats_record: bool, verify_ok: bool, independent_ok: bool | None, independent_beats: bool | None) -> str | None:
    """record_claim only when the float score beats the record, the verifier accepts the candidate and the
    independent verifier (when the problem has one) agrees and also beats the record."""
    if not beats_record or not verify_ok:
        return None
    if independent_ok is None:
        return "record_claim"
    return "record_claim" if independent_ok and independent_beats else "numerical_artifact"


# ---------------------------------------------------------------------------------------------- strategist (5.9)
def stagnated(fraction_now: float, fraction_last_improvement: float, fraction_last_trigger: float,
              stagnation_fraction: float) -> bool:
    return fraction_now - max(fraction_last_improvement, fraction_last_trigger) >= stagnation_fraction


def should_revert(progress_before: float, progress_after: float) -> bool:
    return progress_after < progress_before
