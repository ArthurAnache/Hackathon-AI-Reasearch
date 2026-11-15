"""STAT 201B-style paired hypothesis testing for optimizer comparisons.

H0: E[D] <= 0 where D_i = improvement_A_i - improvement_B_i on matched seed i.
H1: E[D] > 0.

With n=3, exact sign-flip / sign tests cannot reject H0 at alpha=0.05 (min one-sided
p from all-positive signs is 1/8 = 0.125). Reports must say so explicitly.
"""
from __future__ import annotations

import itertools
import math
import random
import statistics
from typing import Sequence


def _mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def _se(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return statistics.stdev(xs) / math.sqrt(len(xs))


def cohens_dz(diffs: Sequence[float]) -> float:
    if len(diffs) < 2:
        return float("nan")
    sd = statistics.stdev(diffs)
    return _mean(diffs) / sd if sd > 0 else float("nan")


def sign_test_one_sided(diffs: Sequence[float]) -> dict:
    """Exact binomial sign test, one-sided P(X >= k) under H0: P(+)=1/2. Zeros dropped."""
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    k = sum(d > 0 for d in nonzero)
    if n == 0:
        return {"n": 0, "positive": 0, "p": 1.0, "note": "all zeros"}
    # P(X >= k) = sum_{i=k}^n C(n,i) / 2^n
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return {"n": n, "positive": k, "p": p,
            "note": "With n=3, minimum attainable one-sided p when all signs positive is 0.125."}


def sign_flip_one_sided(diffs: Sequence[float], *, max_exact: int = 20) -> dict:
    """Exact one-sided sign-flip / paired permutation test on the mean difference."""
    d = list(diffs)
    n = len(d)
    if n == 0:
        return {"n": 0, "p": 1.0, "n_perm": 0}
    obs = _mean(d)
    if n <= max_exact:
        count = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            total += 1
            if _mean([s * x for s, x in zip(signs, d)]) >= obs - 1e-18:
                count += 1
        p = count / total
        return {"n": n, "p": p, "n_perm": total, "exact": True,
                "note": "With n=3, all-positive yields p=1/8=0.125; cannot be < 0.05."}
    # Monte Carlo for larger n
    rng = random.Random(0)
    B = 10_000
    count = 1  # count observed
    for _ in range(B - 1):
        m = _mean([x if rng.random() < 0.5 else -x for x in d])
        if m >= obs - 1e-18:
            count += 1
    return {"n": n, "p": count / B, "n_perm": B, "exact": False}


def bootstrap_ci(diffs: Sequence[float], *, alpha: float = 0.05, B: int = 5000,
                 seed: int = 0) -> dict:
    """Percentile bootstrap CI for mean(D). Exploratory uncertainty only."""
    d = list(diffs)
    if not d:
        return {"lo": float("nan"), "hi": float("nan"), "B": 0}
    rng = random.Random(seed)
    n = len(d)
    means = []
    for _ in range(B):
        sample = [d[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    lo_i = int(math.floor(alpha / 2 * B))
    hi_i = int(math.ceil((1 - alpha / 2) * B)) - 1
    return {"lo": means[max(0, lo_i)], "hi": means[min(B - 1, hi_i)], "B": B,
            "note": "Exploratory; with n=3 the bootstrap CI is unstable."}


def student_t_one_sided(diffs: Sequence[float]) -> dict:
    """Classical paired t (normality assumption). Reported as exploratory for small n."""
    n = len(diffs)
    if n < 2:
        return {"t": float("nan"), "df": 0, "p_approx": float("nan")}
    mean, se = _mean(diffs), _se(diffs)
    if se == 0:
        return {"t": float("inf") if mean > 0 else 0.0, "df": n - 1, "p_approx": 0.0 if mean > 0 else 1.0}
    t = mean / se
    # crude normal approximation for p (avoid scipy dependency for core path)
    # better: use math.erf for t~normal when df large; for df=2 note unreliability
    p_norm = 0.5 * (1 - math.erf(t / math.sqrt(2)))
    return {"t": t, "df": n - 1, "p_approx_normal": max(0.0, min(1.0, p_norm)),
            "note": "Normal approximation; unreliable for n=3. Prefer exact sign-flip."}


def mde_paired(n: int, alpha: float, power: float, sd: float) -> float:
    """Rough two-sided-z MDE for paired mean (approx). Directional one-sided uses z_alpha."""
    if n < 2 or sd <= 0:
        return float("nan")
    # one-sided z approx
    z_a = {0.05: 1.64485, 0.025: 1.95996, 0.005: 2.57583, 0.125: 1.15035}.get(
        round(alpha, 5), 1.64485)
    z_b = {0.8: 0.84162, 0.9: 1.28155}.get(round(power, 2), 0.84162)
    return (z_a + z_b) * sd / math.sqrt(n)


def sequential_alpha(decision_index: int, alpha_total: float = 0.05,
                     max_decisions: int = 10, spend: float | None = None) -> dict:
    """Predeclared conservative alpha spending: equal spend across planned decisions."""
    per = spend if spend is not None else alpha_total / max_decisions
    spent = min(decision_index + 1, max_decisions) * per
    return {
        "decision_index": decision_index,
        "alpha_this_look": per,
        "alpha_spent_cumulative": min(spent, alpha_total),
        "alpha_total": alpha_total,
        "max_decisions": max_decisions,
        "policy": "equal_spend_bonferroni_style",
    }


def paired_comparison(gains_a: Sequence[float], gains_b: Sequence[float], *,
                      label_a: str = "A", label_b: str = "B",
                      seeds: Sequence[int] | None = None,
                      alpha: float = 0.05,
                      decision_index: int = 0,
                      alpha_spend: float = 0.005,
                      practical_delta: float = 1e-5) -> dict:
    """Compare improvements of A vs B on matched seeds.

    gains_* are improvements over the same incumbent (not raw scores).
    Dependent within-job evaluations are NOT treated as independent samples.
    """
    if len(gains_a) != len(gains_b):
        raise ValueError("paired gains must have equal length")
    diffs = [a - b for a, b in zip(gains_a, gains_b)]
    n = len(diffs)
    mean_d, se = _mean(diffs), _se(diffs)
    sign = sign_test_one_sided(diffs)
    flip = sign_flip_one_sided(diffs)
    boot = bootstrap_ci(diffs, alpha=alpha)
    tstat = student_t_one_sided(diffs)
    seq = sequential_alpha(decision_index, alpha_total=0.05, spend=alpha_spend)
    alpha_look = seq["alpha_this_look"]
    # claim allowed only if exact/sign-flip p < alpha_look AND n is large enough for that alpha
    claim = bool(flip["p"] < alpha_look and n >= 5)
    sd = statistics.stdev(diffs) if n >= 2 else float("nan")
    mde = mde_paired(n, alpha_look, 0.8, sd) if n >= 2 and sd == sd else float("nan")
    practical = abs(mean_d) >= practical_delta if mean_d == mean_d else False
    if n < 5:
        interpretation = (
            f"n={n} paired seeds: exploratory only. Exact one-sided sign-flip p={flip['p']:.4f}; "
            f"with n=3 the smallest possible all-positive p is 0.125, which cannot beat "
            f"alpha_look={alpha_look}. Do not claim significance. "
            f"{'Mean |D| reaches practical delta.' if practical else 'Mean |D| is far below practical delta '
             f'(record threshold {practical_delta}).'}"
        )
    elif claim:
        interpretation = (
            f"Reject H0 at predeclared alpha_look={alpha_look} (sign-flip p={flip['p']:.4g}). "
            "Still require holdout seeds before promoting a strategy."
        )
    else:
        interpretation = (
            f"Fail to reject H0 at alpha_look={alpha_look} (sign-flip p={flip['p']:.4g}). "
            "This is NOT proof of no improvement."
        )
    return {
        "label_a": label_a, "label_b": label_b, "seeds": list(seeds) if seeds else list(range(n)),
        "gains_a": list(gains_a), "gains_b": list(gains_b), "diffs": diffs,
        "n": n, "mean_diff": mean_d, "se": se, "sd": sd if n >= 2 else float("nan"),
        "cohens_dz": cohens_dz(diffs),
        "sign_positive": sign["positive"], "sign_test": sign, "sign_flip": flip,
        "p_sign_flip_one_sided": flip["p"],
        "bootstrap_ci_mean": boot, "t_exploratory": tstat,
        "sequential": seq, "alpha_look": alpha_look,
        "claim_allowed": claim, "practical_vs_record_delta": practical,
        "practical_delta": practical_delta,
        "mde_80pct_at_alpha_look": mde,
        "outliers": _outlier_notes(diffs),
        "interpretation": interpretation,
        "warnings": [
            "Within-job optimizer evaluations are dependent; only matched seeds are samples.",
            "Do not run naive repeated unadjusted tests after every adaptive iteration.",
            "Nonsignificance ≠ proof of zero effect.",
        ],
    }


def _outlier_notes(diffs: Sequence[float]) -> list[str]:
    if len(diffs) < 3:
        return []
    mean, sd = _mean(diffs), statistics.stdev(diffs)
    if sd == 0:
        return ["zero variance"]
    notes = []
    for i, d in enumerate(diffs):
        z = (d - mean) / sd
        if abs(z) >= 1.5:
            notes.append(f"index {i}: z={z:.2f}")
    return notes


def report_markdown(rep: dict) -> str:
    lines = [
        f"# Statistical report: {rep['label_a']} vs {rep['label_b']}",
        "",
        f"H0: E[D] <= 0 with D_i = gain({rep['label_a']})_i - gain({rep['label_b']})_i on matched seeds.",
        f"n = {rep['n']}, seeds = {rep['seeds']}",
        "",
        f"- Mean D = {rep['mean_diff']:+.6e}, SE = {rep['se']:.6e}, Cohen dz = {rep['cohens_dz']:.3g}",
        f"- Sign test: {rep['sign_test']['positive']}/{rep['sign_test']['n']} positive, p = {rep['sign_test']['p']:.4g}",
        f"- Exact sign-flip (one-sided): p = {rep['p_sign_flip_one_sided']:.4g}",
        f"- Bootstrap CI (exploratory): [{rep['bootstrap_ci_mean']['lo']:+.6e}, {rep['bootstrap_ci_mean']['hi']:+.6e}]",
        f"- Alpha this look (predeclared spend) = {rep['alpha_look']}",
        f"- Claim of significance allowed: **{rep['claim_allowed']}**",
        f"- Practical vs record delta {rep['practical_delta']}: **{rep['practical_vs_record_delta']}**",
        f"- Approx MDE (80% power at alpha_look): {rep['mde_80pct_at_alpha_look']}",
        "",
        "## Interpretation",
        "",
        rep["interpretation"],
        "",
        "## Warnings",
        "",
    ]
    lines += [f"- {w}" for w in rep["warnings"]]
    return "\n".join(lines) + "\n"
