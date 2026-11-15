"""Episode plans for the CPU fleet. A plan is a list of episode specs (dicts with the fields of EpisodeSpec).

Strengths are chosen from the score right after the move, measured on solutions 2782 and 2783 (0.96364 and
0.96366). The score is very sensitive: a log-normal noise of sigma = 0.0003 lowers it by 3.5e-4, and sigma = 0.003
by 2.8e-3; splitting 14 runs lowers it by 1.2e-2. Each family gets strengths that lower the score by roughly 1e-6
to 1e-2, so that the grid spans "the polisher undoes it" to "the polisher cannot come back".
"""
from __future__ import annotations

from knockout.v2.episode import EpisodeSpec

# family -> strengths, from the mildest to the strongest
STRENGTHS = {
    "lognormal": [3e-6, 3e-5, 3e-4, 3e-3],
    "region_noise": [3e-5, 3e-4, 3e-3, 3e-2],
    "threshold": [1e-6, 1e-5, 1e-4, 1e-3],
    "open_boundary": [1e-5, 1e-4, 1e-3, 1e-2],
    "swap_support": [1e-6, 1e-5, 1e-4, 1e-3],
    "split_runs": [1e-4, 3e-4, 1e-3, 1e-2],
    "merge_runs": [1e-4, 3e-4, 1e-3, 1e-2],
    "transfer_mass": [1e-5, 1e-4, 1e-3, 1e-2],
    "shift_runs": [1e-5, 1e-4, 1e-3, 1e-2],
    "open_gradient": [1e-4, 1e-3, 1e-2, 1e-1],
}
REHEAT_SCHEDULES = ["a7", "a6", "a5", "mm6", "mm7", "mm8"]


def spec(start, family, strength, seed, schedule, budget) -> dict:
    return {"start": start, "family": family, "strength": float(strength), "seed": int(seed),
            "schedule": schedule, "budget": int(budget)}


def with_baselines(specs: list[dict]) -> list[dict]:
    """The plan plus one "polish alone" episode per (start, budget), without duplicates."""
    out, seen = [], set()
    for s in specs:
        es = EpisodeSpec(**s)
        for e in (es.baseline(), es):
            if e.episode_id not in seen:
                seen.add(e.episode_id)
                out.append(spec(e.start, e.family, e.strength, e.seed, e.schedule, e.budget))
    return out


def smoke() -> list[dict]:
    """About ten short episodes, to check the plumbing and the cost of one evaluation on Modal."""
    out = [spec("sol2783", family, strengths[1], 1, "p0", 100) for family, strengths in STRENGTHS.items()]
    return out + [spec("sol2783", "reheat", 0.0, 0, "mm8", 100)]


def reheats(start: str, budgets=(1500, 3000, 6000, 12000), schedules=REHEAT_SCHEDULES) -> list[dict]:
    """The reheat family: no move, every low-beta schedule, every budget."""
    return [spec(start, "reheat", 0.0, 0, schedule, budget) for budget in budgets for schedule in schedules]


def grid(start: str, families=None, seeds=(1, 2), schedules=("p0", "a7"), budgets=(1500, 3000, 6000)) -> list[dict]:
    """Move family x strength x seed x schedule x budget, from one start. The shortest budget gets every seed and
    schedule; the middle one a single seed; the longest a single seed, the last schedule and the two middle
    strengths."""
    out = []
    for family in families or STRENGTHS:
        strengths = STRENGTHS[family]
        for strength in strengths:
            for schedule in schedules:
                out += [spec(start, family, strength, seed, schedule, budgets[0]) for seed in seeds]
                if len(budgets) > 1:
                    out.append(spec(start, family, strength, seeds[0], schedule, budgets[1]))
        if len(budgets) > 2:
            out += [spec(start, family, strength, seeds[0], schedules[-1], budgets[2]) for strength in strengths[1:3]]
    return out


def e1() -> list[dict]:
    """E1, validation from solution 2782: a basin 2.3e-5 higher is known to exist (solution 2783)."""
    return reheats("sol2782") + grid("sol2782")


def e2_reheat() -> list[dict]:
    """E2, first part: the reheat family from the live #1 (solution 2783)."""
    return reheats("sol2783")


PLANS = {"smoke": smoke, "e1": e1, "e2_reheat": e2_reheat}
