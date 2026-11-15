"""Level schedules, batch specifications and cost estimates of the coarse-to-fine runs. No array code and no
torch: everything here is tested locally.

A batch is one ladder of resolutions shared by all its rows (the rows live in one tensor, so they need the same
length at every moment):

    start (n cells) -> zero-pad to `pad_to` -> row perturbation `pre` -> sum-pool by `pool`
    level 0: optimize `level_evals[0]` evaluations at pad_to / pool cells
    for each factor k of `ups`: repeat-upsample by k, row perturbation `post`, optimize at the finer level

Rows differ by their seed, their perturbations, their floors and their beta schedule. `pool = 1, ups = ()` is a
single-level run at full resolution (used for the redraws at a fixed envelope).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from knockout.v2.gpu import schedule as sch

SLUG = sch.SLUG
VOLUME = "knockout-v2-multires"
APP = "knockout-v2-multires"
CAP = 2_000_000                       # EinsteinArena accepts at most 2,000,000 values per solution
POLISH = (3e9, 1e10, 3e10, 1e11)

# Measured on an H100 in float64 by knockout/v2/gpu (E0 and its 30,000-evaluation runs): 16 chains of 2,000,000
# cells take 0.0334 s per batched evaluation, i.e. about 1.05e-9 s per chain, cell and evaluation.
S_PER_CELL_EVAL = 1.05e-9
S_PER_EVAL_FLOOR = 0.0035             # a batched evaluation never costs less than this (measured: 0.0033)


@dataclass(frozen=True)
class RowSpec:
    name: str
    seed: int = 0
    pre: tuple = ()                   # perturbations at the start resolution, before pooling
    post: tuple = ()                  # perturbations after every upsampling
    floor: float = 0.0                # level 0: added to every cell, in units of the mean non-zero value
    up_floor: float = 0.0             # same, at the levels that follow an upsampling
    beta0: float = 1e6                # every level cools from beta0 ...
    beta1: float = 1e9                # ... to beta1; the last level then polishes up to 1e11
    beta0_up: float = 0.0             # first beta of the levels that follow an upsampling (0 = beta0)
    per_decade: float = 3.0
    polish_share: float = 0.25
    hypothesis: str = ""
    coarse: tuple = ()                # perturbations of the pooled vector, before level 0
    fine_stages: tuple = ()           # explicit (beta, share of the evaluations) stages after an upsampling

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("pre", "post", "coarse"):
            d[key] = [dict(p) for p in getattr(self, key)]
        d["fine_stages"] = [list(s) for s in self.fine_stages]
        return d

    @staticmethod
    def from_dict(d: dict) -> "RowSpec":
        d = dict(d)
        for key in ("pre", "post", "coarse"):
            d[key] = tuple(dict(p) for p in d.get(key, ()))
        d["fine_stages"] = tuple((float(b), float(s)) for b, s in d.get("fine_stages", ()))
        return RowSpec(**d)


@dataclass(frozen=True)
class BatchSpec:
    name: str
    experiment: str                   # "E1" ... "E4", "smoke"
    start: str                        # "frozen" (EA solution 2782), "live" (live #1) or "vol:<path>.npy"
    pool: int
    ups: tuple
    level_evals: tuple
    rows: tuple
    pad_to: int = CAP

    def to_dict(self) -> dict:
        return {"name": self.name, "experiment": self.experiment, "start": self.start, "pool": self.pool,
                "ups": list(self.ups), "level_evals": list(self.level_evals), "pad_to": self.pad_to,
                "rows": [r.to_dict() for r in self.rows]}

    @staticmethod
    def from_dict(d: dict) -> "BatchSpec":
        return BatchSpec(name=d["name"], experiment=d["experiment"], start=d["start"], pool=int(d["pool"]),
                         ups=tuple(int(k) for k in d["ups"]), level_evals=tuple(int(e) for e in d["level_evals"]),
                         rows=tuple(RowSpec.from_dict(r) for r in d["rows"]), pad_to=int(d["pad_to"]))


def level_sizes(batch: BatchSpec) -> list[int]:
    """Number of cells at each level, coarse to fine."""
    sizes = [batch.pad_to // batch.pool]
    for k in batch.ups:
        sizes.append(sizes[-1] * k)
    return sizes


def check_batch(batch: BatchSpec) -> None:
    if batch.pool < 1 or batch.pad_to % batch.pool:
        raise ValueError(f"{batch.name}: pad_to {batch.pad_to} is not a multiple of the pooling factor {batch.pool}")
    if any(k < 1 for k in batch.ups):
        raise ValueError(f"{batch.name}: upsampling factors must be >= 1 (1 = a cycle at the same resolution)")
    if batch.pool % math.prod(batch.ups):
        raise ValueError(f"{batch.name}: the upsampling factors {batch.ups} do not divide the pooling factor")
    if len(batch.level_evals) != len(batch.ups) + 1:
        raise ValueError(f"{batch.name}: need one evaluation budget per level")
    if level_sizes(batch)[-1] > CAP:
        raise ValueError(f"{batch.name}: the finest level exceeds the submission cap of {CAP} values")
    names = [r.name for r in batch.rows]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{batch.name}: rows need distinct names")
    for r in batch.rows:
        if (r.floor < 0 or r.up_floor < 0 or not 0 <= r.polish_share < 1 or r.beta0 > r.beta1
                or r.beta0_up > r.beta1 or r.beta0_up < 0):
            raise ValueError(f"{batch.name}/{r.name}: bad floors, polish share or betas")
        for level in range(len(batch.level_evals)):
            stages(batch, r, level)       # raises if a level has fewer evaluations than stages


def stages(batch: BatchSpec, row: RowSpec, level: int) -> tuple:
    """The (beta, evaluations) stages of `row` at `level`: one cooling from beta0 to beta1, and on the last level
    a polish up to 1e11 that takes `polish_share` of the level's evaluations. A row with beta0 = beta1 >= 1e9 is
    the polish-only control."""
    evals = batch.level_evals[level]
    last = level == len(batch.level_evals) - 1
    if level > 0 and row.fine_stages:
        total = sum(share for _, share in row.fine_stages)
        counts = [max(1, int(evals * share / total)) for _, share in row.fine_stages]
        counts[-1] += evals - sum(counts)
        return tuple(sch.Stage(float(b), c) for (b, _), c in zip(row.fine_stages, counts))
    beta0 = row.beta0_up if level > 0 and row.beta0_up else row.beta0
    if not last:
        return sch.anneal(beta0, row.beta1, row.per_decade, evals)
    if beta0 >= 1e9:
        return sch.anneal(beta0, max(row.beta1, 1e11), 2, evals)
    tail = tuple(b for b in POLISH if b > row.beta1)
    polish_evals = max(len(tail), int(evals * row.polish_share)) if tail else 0
    return sch.anneal(beta0, row.beta1, row.per_decade, evals - polish_evals, tail, polish_evals)


def estimate(batches, overhead_s: float = 150.0, score_s_per_batch: float = 45.0) -> dict:
    """Wall time and cost ESTIMATE of one container that runs `batches` one after the other."""
    rate = sch.container_usd_per_s()
    loop = 0.0
    for b in batches:
        for n, evals in zip(level_sizes(b), b.level_evals):
            loop += evals * max(S_PER_EVAL_FLOOR, len(b.rows) * n * S_PER_CELL_EVAL)
    wall = loop + overhead_s + score_s_per_batch * len(batches)
    return {"loop_s": loop, "wall_s": wall, "usd": wall * rate, "usd_per_s": rate}


# ---------------------------------------------------------------------------------------------------------------
# Plans. `evals` is the evaluation budget of every level.

def _row(name, seed=0, hypothesis="", **kw) -> RowSpec:
    return RowSpec(name=name, seed=seed, hypothesis=hypothesis, **kw)


def _doublings(pool: int) -> tuple:
    d = int(round(math.log2(pool)))
    if 2 ** d != pool:
        raise ValueError("this plan needs a power-of-two pooling factor")
    return (2,) * d


def plan_smoke(start: str, evals: int, pools=(4,)) -> list[BatchSpec]:
    """Plumbing: one two-level V-cycle with four rows, and one single-level redraw batch with two rows."""
    rows = (_row("plain", beta0=1e6), _row("ln0.3-s1", 1, post=({"kind": "lognormal", "sigma": 0.3},), beta0=1e6),
            _row("polish", beta0=1e9, beta1=1e9), _row("open", beta0=1e6, floor=1e-3, up_floor=1e-6))
    redraw = (_row("control", beta0=1e9, beta1=1e9),
              _row("shift128", 1, pre=({"kind": "shift", "block": 128, "max_shift": 2, "frac": 0.1},), beta0=1e6))
    return [BatchSpec("smoke-v4", "smoke", start, 4, (2, 2), (evals,) * 3, rows),
            BatchSpec("smoke-redraw", "smoke", start, 1, (), (evals,), redraw)]


def plan_e1(start: str, evals: int, pools=(2, 4, 8, 16, 32, 64)) -> list[BatchSpec]:
    """E1, descent curve: the start pooled by each factor, then optimized at that level only. Four rows per
    level: polish only, two cooling depths with the pooled support fixed, and one cooling with zero cells opened.
    (In knockout/v2/gpu's kicked chains, the lower the first beta, the closer a chain came back to its start.)"""
    out = []
    for pool in pools:
        what = f"start pooled by {pool} and optimized at that level"
        rows = (_row("polish", beta0=1e9, beta1=1e9, hypothesis=f"{what}: polish only (beta 1e9 to 1e11)"),
                _row("b5", beta0=1e5, hypothesis=f"{what}: cooling from beta = 1e5, then polish"),
                _row("b3", beta0=1e3, hypothesis=f"{what}: cooling from beta = 1e3, then polish"),
                _row("b3-open", beta0=1e3, floor=1e-3,
                     hypothesis=f"{what}: cooling from beta = 1e3 with zero cells opened (floor 1e-3)"))
        out.append(BatchSpec(f"e1-p{pool}", "E1", start, pool, (), (evals,), rows))
    return out


def plan_e2(start: str, evals: int, pools=(2, 4)) -> list[BatchSpec]:
    """E2, V-cycles: pool by 2^d, optimize, then d times (repeat x2, perturb, optimize) back to 2,000,000 cells.
    Sixteen rows per depth: no noise at four reheat depths, log-normal noise of three widths with several seeds,
    the uniform all-cell noise of the public velocity cycles, and two rows that let zero cells open."""
    out = []
    for pool in pools:
        what = f"V-cycle of depth {pool} (x2 per level)"

        def ln(sigma, seed, b0=1e4, **kw):
            return _row(f"ln{sigma:g}-b{int(round(math.log10(b0)))}-s{seed}", seed,
                        post=({"kind": "lognormal", "sigma": sigma},), beta0=b0,
                        hypothesis=f"{what}: log-normal cell noise of width {sigma:g} after every upsampling "
                                   f"(seed {seed}), cooling from beta = {b0:.0e} at every level", **kw)

        rows = [_row(f"plain-b{int(round(math.log10(b0)))}", beta0=b0,
                     hypothesis=f"{what}: no noise, cooling from beta = {b0:.0e} at every level")
                for b0 in (1e3, 1e4, 1e5, 1e6)]
        rows += [ln(0.1, s) for s in (1, 2, 3)] + [ln(0.3, s) for s in (1, 2, 3)] + [ln(1.0, s) for s in (1, 2)]
        rows += [ln(0.3, 1, b0=1e3), ln(0.3, 1, b0=1e5)]
        rows.append(_row("uni1e-8-b4", 1, post=({"kind": "uniform", "scale": 1e-8},), beta0=1e4,
                         hypothesis=f"{what}: uniform noise in [0, 1e-8 max f] on every cell after every "
                                    f"upsampling (the public velocity-cycle noise), cooling from beta = 1e4"))
        rows.append(_row("open6-b4", beta0=1e4, floor=1e-6, up_floor=1e-6,
                         hypothesis=f"{what}: zero cells opened at every level (floor 1e-6), cooling from 1e4"))
        out.append(BatchSpec(f"e2-v{pool}", "E2", start, pool, _doublings(pool), (evals,) * (len(_doublings(pool)) + 1),
                             tuple(rows)))
    return out


def plan_e2_blend(start: str, evals: int, pools=(4, 16, 64, 256)) -> list[BatchSpec]:
    """E2, partial V-cycles at full resolution: f <- (1 - t) f + t (f pooled by k then repeated), for four block
    sizes and four strengths; t = 1 is the full V-cycle without any coarse optimization. Block masses are kept
    exactly and zero cells next to the support open in proportion to the local mass."""
    rows = []
    for k in pools:
        for t in (0.01, 0.1, 0.3, 1.0):
            rows.append(_row(f"blend{k}-t{t:g}", pre=({"kind": "blend", "block": k, "t": t},), beta0=1e4,
                             hypothesis=f"partial V-cycle: blend with the block-smoothed start (blocks of {k} "
                                        f"cells, t = {t:g}), then one cooling from beta = 1e4 and a polish"))
    return [BatchSpec("e2-blend", "E2", start, 1, (), (evals,), tuple(rows))]


def plan_e3(start: str, evals: int, pools=(32, 128, 512)) -> list[BatchSpec]:
    """E3, micro-structure redraw at a fixed envelope: the mass of every block of 32, 128 or 512 cells is kept
    and the arrangement inside the blocks is redrawn (shift, shuffle, bootstrap redraw, log-normal resample), on
    all the blocks or on a small share of them, then one cooling and a polish at full resolution. Measured on
    CPU before the run: every such redraw costs 2e-2 to 5e-1 at once, even on 1 % of the blocks."""
    rows = [_row("control", beta0=1e9, beta1=1e9, hypothesis="control: no redraw, polish only")]

    def add(kind, block, seed, frac=1.0, b0=1e4, **params):
        spec = {"kind": kind, "block": block, "frac": frac, **params}
        tag = f"{kind}{block}" + (f"-f{frac:g}" if frac < 1 else "") + f"-s{seed}"
        rows.append(_row(tag, seed, pre=(spec,), beta0=b0,
                         hypothesis=f"redraw at fixed envelope: {kind} inside blocks of {block} cells "
                                    f"({frac:.1%} of the blocks, seed {seed}), cooling from beta = {b0:.0e}"))

    for block in pools:
        add("shift", block, 1, max_shift=2)
        add("shift", block, 1, frac=0.01, max_shift=2)
        add("shuffle", block, 1, frac=0.01)
        add("resample", block, 1, sigma=0.3)
    mid = pools[min(1, len(pools) - 1)]
    add("shift", mid, 1, frac=0.001, max_shift=2)
    add("redraw", mid, 1, frac=0.01)
    add("shuffle", mid, 1, frac=0.1)
    return [BatchSpec("e3-redraw", "E3", start, 1, (), (evals,), tuple(rows[:16]))]


def plan_edge(start: str, evals: int, pools=(1, 2, 4, 8)) -> list[BatchSpec]:
    """E3b, support-edge opening. Between solutions 2782 and 2783 about 23,000 cells were born and 25,000 died,
    and 75 % of the born cells sit within 2 cells of the old support: the comb teeth moved their edges. A chain
    with f = w^2 can close a cell but never open one. So: open the zero cells within `radius` cells of the
    support at a small share t of the neighbouring value, then polish, with or without a mild reheat.
    Measured on CPU before the run: opening costs about t x 1.3 (radius 1) to t x 8 (radius 8) at once, because
    any added mass raises some lag of the flat maximum; the larger t, the lower the first beta of the row."""
    rows = [_row("control", beta0=1e9, beta1=1e9, hypothesis="control: no opening, polish only")]

    def add(radius, t, b0):
        tag = f"edge{radius}-t{t:g}-b{int(round(math.log10(b0)))}"
        how = "polish only" if b0 >= 1e9 else f"cooling from beta = {b0:.0e}, then polish"
        rows.append(_row(tag, pre=({"kind": "dilate", "radius": radius, "t": t},), beta0=b0,
                         hypothesis=f"support-edge opening: zero cells within {radius} cells of the support "
                                    f"open at {t:g} of the neighbouring value; {how}"))

    for radius in pools:
        add(radius, 1e-6, 1e7)
        add(radius, 1e-5, 1e6)
        add(radius, 1e-4, 1e5)
    mid = pools[min(1, len(pools) - 1)]
    add(mid, 1e-7, 1e9)
    add(mid, 1e-6, 1e6)
    add(mid, 1e-5, 1e5)
    tag = start.replace("vol:", "").replace("/", "_").replace(".npy", "")[-24:]
    return [BatchSpec(f"e3b-edge-{tag}", "E3b", start, 1, (), (evals,), tuple(rows[:16]))]


def plan_vdeep(start: str, evals: int, pools=(2,)) -> list[BatchSpec]:
    """E2b, thorough V-cycles, set after E1: at a coarse level a cooling from beta = 1e3 ends far above one from
    1e5 (pool 2: 0.96354 against 0.96239), and at 2,000,000 cells a cooling from 1e4 or higher loses little
    (knockout/v2/gpu: -1.3e-6 from 1e4, -7e-8 from 1e5 on the incumbent). So each row cools from a low beta at the
    coarse level and from a higher one after every upsampling. Rows also vary the pooling phase (the start rolled
    by pool / 2 cells, which is score-neutral: the padded last cell is zero), the cooling speed, and a log-normal
    noise after the upsampling.

    Five rows follow the recipe a second model proposed to the team (unverified advice, relayed by the
    coordinator): kick the coarse vector in square-root coordinates (sigma 0.05), optimize it, repeat, kick on the
    fine grid (sigma 0.03), then a fine schedule beta = 1e7, 1e8, 1e9, 1e10 with 10 / 20 / 30 / 40 % of the
    evaluations. One row uses that fine schedule without any kick."""
    advised = ((1e7, 0.1), (1e8, 0.2), (1e9, 0.3), (1e10, 0.4))
    out = []
    for pool in pools:
        ups = _doublings(pool)
        what = f"thorough V-cycle of depth {pool}"
        rows = []

        def add(tag, c0, f0, seed=0, pre=(), post=(), per_decade=3.0, note=""):
            rows.append(_row(f"{tag}-c{int(round(math.log10(c0)))}-f{int(round(math.log10(f0)))}", seed, pre=pre,
                             post=post, beta0=c0, beta0_up=f0, per_decade=per_decade,
                             hypothesis=f"{what}: cooling from beta = {c0:.0e} at the coarsest level and from "
                                        f"{f0:.0e} after every upsampling{note}"))

        for c0 in (1e2, 1e3):
            for f0 in (1e4, 1e5, 1e6):
                add("plain", c0, f0)
        add("plain", 1e1, 1e4)
        roll = ({"kind": "roll", "by": pool // 2},)
        for f0 in (1e4, 1e5):
            add("phase", 1e3, f0, pre=roll, note=f"; pooling phase shifted by {pool // 2} cell(s)")
        add("ln0.1-s1", 1e3, 1e4, seed=1, post=({"kind": "lognormal", "sigma": 0.1},),
            note="; log-normal noise of width 0.1 after every upsampling (seed 1)")
        add("slow", 1e3, 1e4, per_decade=6.0, note="; 6 stages per decade")
        rows.append(_row("advised-fine-c3", beta0=1e3, fine_stages=advised,
                         hypothesis=f"{what}: cooling from beta = 1e3 at the coarsest level, then the advised "
                                    f"fine schedule (beta 1e7 to 1e10, growing stages), no kick"))
        for seed in (1, 2, 3, 4):
            rows.append(_row(f"advised-kick-s{seed}", seed, beta0=1e3, fine_stages=advised,
                             coarse=({"kind": "sqrt_gauss", "sigma": 0.05},),
                             post=({"kind": "sqrt_gauss", "sigma": 0.03},),
                             hypothesis=f"{what}, advised recipe: Gaussian kick in square-root coordinates on "
                                        f"the coarse vector (sigma 0.05, seed {seed}), cooling from beta = 1e3, "
                                        f"repeat, kick on the fine grid (sigma 0.03), advised fine schedule"))
        tag = start.replace("vol:", "").replace("/", "_").replace(".npy", "")[-24:]
        out.append(BatchSpec(f"e2b-v{pool}-{tag}", "E2b", start, pool, ups, (evals,) * (len(ups) + 1),
                             tuple(rows)))
    return out


def plan_grow(start: str, evals: int, pools=(1, 2, 4)) -> list[BatchSpec]:
    """E3c, gradient-guided support growth, in cycles at full resolution. Level 0 polishes at beta = 1e9; then,
    three times: compute the gain d log C_beta / d f_i of every zero cell at the current point, open the zero
    cells within `radius` cells of the support whose gain is positive (at t times the neighbouring value), and
    cool again from the beta at which the gain was taken. This is the active-set step that a chain with f = w^2
    cannot take; unlike the blind opening of E3b it adds mass only where the smoothed objective says it helps.
    Measured on CPU on solution 2782: the gain must be taken at beta <= 1e7 (at 1e8 and above every edge cell
    has a positive gain, because the smoothed maximum then follows a handful of lags, and opening them all costs
    as much as the blind opening); at beta = 1e6 it selects 9,550 of the 32,499 cells next to the support, and
    opening them at t = 1e-4 costs 8e-7 where the blind opening costs 1.3e-4."""
    rows = [_row("control", beta0=1e9, beta1=1e9, hypothesis="control: the same four polish levels, no opening")]

    def add(radius, t, at):
        tag = f"grow{radius}-t{t:g}-g{int(round(math.log10(at)))}"
        rows.append(_row(tag, post=({"kind": "grow", "radius": radius, "t": t, "beta": at},), beta0=1e9,
                         beta0_up=at, hypothesis=f"support growth: zero cells within {radius} cells of the "
                                                 f"support with a positive gain at beta = {at:.0e} open at "
                                                 f"{t:g} of the neighbouring value, then cooling from that "
                                                 f"beta; three cycles"))

    for at in (1e6, 1e7):
        for radius in pools:
            for t in (1e-5, 1e-4):
                add(radius, t, at)
    mid = pools[min(1, len(pools) - 1)]
    add(mid, 1e-3, 1e6)
    add(mid, 1e-3, 1e7)
    add(mid, 1e-4, 1e5)
    tag = start.replace("vol:", "").replace("/", "_").replace(".npy", "")[-24:]
    return [BatchSpec(f"e3c-grow-{tag}", "E3c", start, 1, (1, 1, 1), (evals,) * 4, tuple(rows[:16]))]


def plan_relink(start: str, evals: int, pools=()) -> list[BatchSpec]:
    """E5, path relinking between the two public solutions. The public velocity cycles restart from
    f_best + 0.3 (f_best - f_prev) (momentum along the last improvement); here the last improvement is the step
    from solution 2782 to the live #1. Rows start from points on the line through both: inside the segment
    (t > 0, towards 2782) and beyond the live #1 (t < 0), then cool and polish. Measured on CPU: the exact score
    has a kink at each end, it loses 1.1e-4 at 1 % inside the segment and 4.3e-4 at 0.5 % beyond the live #1.
    The start must be "live"; the other end is the frozen certificate."""
    rows = [_row("control", beta0=1e9, beta1=1e9, hypothesis="control: the live #1, polish only")]
    plan = ((0.5, 1e3), (0.5, 1e4), (0.25, 1e3), (0.25, 1e4), (0.1, 1e4), (0.1, 1e5), (0.03, 1e4), (0.03, 1e5),
            (0.01, 1e5), (0.01, 1e6), (0.005, 1e6), (-0.002, 1e5), (-0.005, 1e4), (-0.005, 1e5), (-0.01, 1e4))
    for t, b0 in plan:
        where = f"{t:.1%} of the way to solution 2782" if t > 0 else f"{-t:.1%} beyond the live #1"
        rows.append(_row(f"mix{t:g}-b{int(round(math.log10(b0)))}", pre=({"kind": "mix", "other": "frozen", "t": t},),
                         beta0=b0, hypothesis=f"path relinking: start {where} on the line through both public "
                                              f"solutions, cooling from beta = {b0:.0e}, then polish"))
    return [BatchSpec("e5-relink", "E5", start, 1, (), (evals,), tuple(rows))]


PLANS = {"smoke": plan_smoke, "e1": plan_e1, "e2": plan_e2, "e2-blend": plan_e2_blend, "e3": plan_e3,
         "edge": plan_edge, "vdeep": plan_vdeep, "grow": plan_grow, "relink": plan_relink}


def build(names, start: str, evals: int, pools=None) -> list[BatchSpec]:
    """The batches of the plans `names` (comma-separated string or list), checked. A plan may carry its own
    factors after colons, e.g. "e1,e2:64:4" (E1 with its default factors, then V-cycles of depth 64 and 4);
    `pools` applies to the plans that carry none. A plan may also carry its own start after "@" and its own
    evaluations per level after "=", e.g. "relink@live=12000,grow@frozen=4000"."""
    if isinstance(names, str):
        names = [x for x in names.split(",") if x]
    out = []
    for token in names:
        token, _, own_evals = token.partition("=")
        token, _, own_start = token.partition("@")
        name, *own = token.split(":")
        if name not in PLANS:
            raise ValueError(f"unknown plan {name!r} (known: {sorted(PLANS)})")
        factors = tuple(int(x) for x in own) or pools
        begin, budget = own_start or start, int(own_evals) if own_evals else evals
        out += PLANS[name](begin, budget, factors) if factors else PLANS[name](begin, budget)
    seen = [b.name for b in out]
    if len(set(seen)) != len(seen):
        raise ValueError("batch names must be distinct")
    for b in out:
        check_batch(b)
    return out
