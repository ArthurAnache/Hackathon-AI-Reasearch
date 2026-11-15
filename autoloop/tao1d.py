"""Problem plug-in: Tao constant 1d (signed autoconvolution, upper bound, lower is better).

Everything specific to the problem lives here: the float score, the vector id, the executors (seed with the WS2
optimizer, refine with attack/c1d_refine.py, certify with attack/c1d_round.py + attack/c1d_exact.py), the live
reference reader, the claim threshold and the refine schedule. A second problem plugs in by providing a module with
the same names (see autoloop/problem.py).
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import resource
import subprocess
import sys
import urllib.request
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.fft as sfft

from .safety import clean_env, safe_err

MODAL_ENV = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE")

NAME = "tao-1d"
LOWER_IS_BETTER = True
SEED_CELLS = 800
SPLIT_UP = 4                                   # refinement up a resolution level: 800 -> 3200 -> 12800 -> 51200 ...
P0_SCHEDULE = [2.0, 4.0, 8.0, 16.0, 64.0, 512.0, 1024.0, 4096.0]   # restarts, soft (small p0) to hard; 2-8 gave the best 51k certificates today
LOCAL_MAX_CELLS = 12800
MODAL_KINDS = dict(seed="seed1d", refine="refine1d")   # job kinds of autoloop/modal_app.py
METHOD_NOTES = ("Method knowledge from our hand-run chain (E12, 2026-11-14): at the high-resolution stage a soft restart "
                "(small p0, 4 and 8) is what moved lineages; the rank before that stage did not predict the winner, so "
                "do not cut to a handful of lineages before trying soft restarts on a wide set. Seeds are cheap: a 45 s "
                "seed already ranks well.")
WIDE_P0_EARLY = [8.0]                          # refine_wide up to the level below the top
WIDE_P0_FINAL = [4.0, 8.0]                     # entering the top level: soft restarts (E12)
SOFT_RESTARTS = [4.0, 8.0, 2.0, 16.0]          # restarts at the top level, softest first (E12)
REPO = Path(__file__).resolve().parent.parent
ATTACK = REPO / "attack"
PY = sys.executable
REF_PAGE = "https://teorth.github.io/optimizationproblems/constants/1d.html"
REF_REPO = "teorth/optimizationproblems"


def score(v) -> float:
    """Float score 2N max|v*v| / (sum v)^2 (lower is better), same formula as attack/c1d_refine.py."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    L = sfft.next_fast_len(2 * n - 1, real=True)
    c = sfft.irfft(sfft.rfft(v, L) ** 2, L)[: 2 * n - 1]
    return float(2 * n * np.abs(c).max() / v.sum() ** 2)


def vec_id(v) -> str:
    return hashlib.sha256(np.asarray(v, dtype=np.float64).tobytes()).hexdigest()


def load_vec(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    d = json.load(open(path))
    for key in ("best_values", "solution", "values"):
        if isinstance(d, dict) and key in d:
            d = d[key]
    if isinstance(d, dict):
        d = d["values"]
    return np.asarray([float(x) for x in d], dtype=np.float64)


WORKSPACE = __import__("os").environ.get("AUTOLOOP_WORKSPACE", str((lambda q: q[2] if len(q) > 2 else q[-1])(__import__("pathlib").Path(__file__).resolve().parents)))  # folder that holds repo/, wt-compare/, shared/ and .env
INDEPENDENT = Path(WORKSPACE + "/repo/records/1d/highres/check_independent.py")
INDEPENDENT_SHA256 = "f44ad31b6313eec67fa51b330a5170b196a61b1d1c1dd6d2da2f4698acd2ac5c"   # pinned 2026-11-15 (Lupa N6)


def pinned_checker_source() -> str:
    """The independent checker's source, read once and checked against its pinned sha256 (else refuse). The bytes
    that were hashed are the bytes that run (passed on stdin), so the file cannot change between check and use."""
    raw = INDEPENDENT.read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    if h != INDEPENDENT_SHA256:
        raise RuntimeError(f"independent checker hash {h[:16]} differs from the pinned {INDEPENDENT_SHA256[:16]}")
    return raw.decode()


RUN_PINNED = ("import sys; src=sys.stdin.read(); sys.argv=['check', sys.argv[1]]; "
              "g={'__name__': '__main__', '__file__': 'check_independent.py'}; "
              "exec(compile(src, 'check_independent.py', 'exec'), g); ")


def independent_exact(int_json):
    """Run the coordinator's independent checker (Kronecker on P - Q, direct sum at the deciding knot, spot checks)
    in a clean subprocess and read back its exact fraction. Returns (Fraction or None, cpu_s)."""
    code = RUN_PINNED + "s=g['score']; print('FRACTION', s.numerator, s.denominator)"
    c0 = _cpu_children()
    r = subprocess.run([PY, "-I", "-c", code, int_json], input=pinned_checker_source(), capture_output=True, text=True,
                       timeout=3600, env=clean_env())
    cpu = _cpu_children() - c0
    for line in r.stdout.splitlines():
        if line.startswith("FRACTION "):
            _, p, q = line.split()
            return Fraction(int(p), int(q)), cpu
    return None, cpu


def valid_vec(v) -> bool:
    return len(v) >= 50 and bool(np.isfinite(v).all()) and v.sum() != 0


def _cpu_children() -> float:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def _run_parallel(cmds, timeout):
    """Run commands in parallel; return their results and the CPU-seconds they used (children rusage delta)."""
    c0 = _cpu_children()
    procs = [subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(REPO), env=clean_env()) for c in cmds]
    out = []
    for c, p in zip(cmds, procs):
        try:
            so, se = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill(); so, se = p.communicate()
        out.append(subprocess.CompletedProcess(c, p.returncode, so, se))
    return out, _cpu_children() - c0


# ----------------------------------------------------------------------------------------------- executors
class LocalExecutor:
    spent_usd = 0.0
    """Real runs on this machine: seed_worker (WS2 optimizer), attack/c1d_refine.py, c1d_round.py + c1d_exact.py."""
    real = True
    usd_per_cpu_s = 0.0

    def __init__(self, workdir: Path, max_cells: int = LOCAL_MAX_CELLS):
        self.dir = Path(workdir); self.max_cells = max_cells
        (self.dir / "certs").mkdir(parents=True, exist_ok=True)

    def seed(self, n, budget_s, cells, seed0):
        outs = [str(self.dir / "certs" / f"seed_s{seed0 + i}_n{cells}") for i in range(n)]
        cmds = [[PY, "-m", "autoloop.seed_worker", "--cells", str(cells), "--budget", str(budget_s),
                 "--seed", str(seed0 + i), "--out", o] for i, o in enumerate(outs)]
        res, cpu = [], 0.0
        w = getattr(self, "workers", 6)
        for k in range(0, len(cmds), w):                  # at most `workers` seeds at once on this machine
            r, c = _run_parallel(cmds[k:k + w], timeout=budget_s * 2 + 120); res += r; cpu += c
        rows = []
        for i, (o, r) in enumerate(zip(outs, res)):
            if r.returncode != 0 or not Path(o + ".npy").exists():
                rows.append(dict(ok=False, error=safe_err(r.stderr))); continue
            rows.append(dict(ok=True, path=o + ".npy", reported=json.load(open(o + ".json"))["best"],
                             params=dict(seed=seed0 + i, cells=cells, budget_s=budget_s, optimizer="wt-compare 1d_tuned")))
        return rows, cpu

    def refine(self, parent_path, split, p0s, budget_s, tag):
        outs = [str(self.dir / "certs" / f"{tag}_x{split}_p{p0:g}") for p0 in p0s]
        cmds = [[PY, str(ATTACK / "c1d_refine.py"), "--start", parent_path, "--out", o, "--split", str(split),
                 "--p0", str(p0), "--budget", str(budget_s), "--threads", "1", "--quiet"] for p0, o in zip(p0s, outs)]
        res, cpu = _run_parallel(cmds, timeout=budget_s + 600)
        rows = []
        for p0, o, r in zip(p0s, outs, res):
            if r.returncode != 0 or not Path(o + ".npy").exists():
                rows.append(dict(ok=False, error=safe_err(r.stderr), params=dict(split=split, p0=p0))); continue
            j = json.load(open(o + ".json"))
            rows.append(dict(ok=True, path=o + ".npy", reported=j["best"],
                             params=dict(split=split, p0=p0, budget_s=budget_s, nev=j["nev"], seconds=j["seconds"],
                                         start_score=j["start_score"])))
        return rows, cpu

    workers = 6

    def refine_many(self, parent_paths, split, p0s, budget_s, tag):
        """Wide refine: every parent x every p0, run `workers` at a time. Rows carry `pi`, the parent's index."""
        rows, cpu = [], 0.0
        jobs = [(i, pp, p0) for i, pp in enumerate(parent_paths) for p0 in p0s]
        for k in range(0, len(jobs), self.workers):
            chunk = jobs[k:k + self.workers]
            outs = [str(self.dir / "certs" / f"{tag}_{i:02d}_x{split}_p{p0:g}") for i, _, p0 in chunk]
            cmds = [[PY, str(ATTACK / "c1d_refine.py"), "--start", pp, "--out", o, "--split", str(split), "--p0", str(p0),
                     "--budget", str(budget_s), "--threads", "1", "--quiet"] for (_, pp, p0), o in zip(chunk, outs)]
            res, c = _run_parallel(cmds, timeout=budget_s + 600); cpu += c
            for (i, _, p0), o, r in zip(chunk, outs, res):
                if r.returncode != 0 or not Path(o + ".npy").exists():
                    rows.append(dict(ok=False, pi=i, error=safe_err(r.stderr), params=dict(split=split, p0=p0))); continue
                j = json.load(open(o + ".json"))
                rows.append(dict(ok=True, pi=i, path=o + ".npy", reported=j["best"],
                                 params=dict(split=split, p0=p0, budget_s=budget_s, nev=j["nev"], seconds=j["seconds"],
                                             start_score=j["start_score"], executor="local")))
        return rows, cpu

    def certify(self, parent_path, tag):
        out = str(self.dir / "certs" / f"{tag}_int56.json")
        (r,), cpu = _run_parallel([[PY, str(ATTACK / "c1d_round.py"), parent_path, out]], timeout=3600)
        if r.returncode != 0:
            return dict(ok=False, error=safe_err(r.stderr)), cpu
        rounded = json.loads(r.stdout.strip().splitlines()[-1])
        sys.path.insert(0, str(ATTACK))
        import c1d_exact  # second, in-process exact check of the same integer file (also gives the fraction)
        ints = json.load(open(out))["values"]
        chk = c1d_exact.check(ints)
        if abs(float(chk["score"]) - rounded["exact"]) > 1e-15:
            return dict(ok=False, error=f"c1d_round and c1d_exact disagree: {rounded['exact']} vs {float(chk['score'])}"), cpu
        f2, cpu2 = independent_exact(out)          # second implementation, shares no code with c1d_exact
        if f2 is None or f2 != chk["score"]:
            return dict(ok=False, error=f"independent exact check disagrees or failed: {safe_err(str(f2))}"), cpu + cpu2
        return dict(ok=True, path=out, ints=ints, exact=chk["score"], knot=chk["knot"],
                    checks=["c1d_round", "c1d_exact", "check_independent"]), cpu + cpu2


class HybridExecutor(LocalExecutor):
    """Local up to LOCAL_MAX_CELLS output cells; above, Modal CPUs through attack/modal_c1d.py (run unchanged, launched
    with `modal run --detach` so a dropped connection cannot stop the app). Results come back through the script's
    local entrypoint (attack/c1d/modal/<tag>/), or from the Modal volume if the local client died."""
    MODAL_USD_PER_JOB_S = 1.0 * 0.0000131 + 3072 / 1024 * 0.00000222   # same prices as attack/modal_c1d.py

    def __init__(self, workdir, max_cells, modal_budget_s=3000, usd_cap=30.0, run_name="run"):
        super().__init__(workdir, max_cells)
        self.run_name = re.sub(r"[^A-Za-z0-9_-]", "-", run_name)
        self.modal_budget_s, self.usd_cap = modal_budget_s, usd_cap
        self.spent_usd = 0.0

    def refine(self, parent_path, split, p0s, budget_s, tag):
        v = load_vec(parent_path)
        if len(v) * split <= LOCAL_MAX_CELLS:
            return super().refine(parent_path, split, p0s, budget_s, tag)
        rows, cpu = self._modal([parent_path], split, p0s, self.modal_budget_s, tag)
        return rows, cpu

    def refine_many(self, parent_paths, split, p0s, budget_s, tag):
        """Wide refine on Modal when the output is above the local limit or there are more jobs than local workers."""
        cells = len(load_vec(parent_paths[0])) * split
        if cells <= LOCAL_MAX_CELLS and len(parent_paths) * len(p0s) <= self.workers:
            return super().refine_many(parent_paths, split, p0s, budget_s, tag)
        return self._modal(parent_paths, split, p0s, budget_s, tag)

    def _modal(self, parent_paths, split, p0s, B, tag):
        starts = []
        for i, pp in enumerate(parent_paths):
            s = self.dir / "certs" / f"{tag}_{i:02d}_start.npy"; np.save(s, load_vec(pp)); starts.append(s)
        B = int(min(B, 3300))                                  # modal_c1d.py refuses budgets above 3300 s
        n = len(starts) * len(p0s)
        est = n * (B + 120) * self.MODAL_USD_PER_JOB_S
        if self.spent_usd + est > self.usd_cap:
            return [dict(ok=False, pi=i, error=f"Modal estimate USD {est:.2f} would pass the cap {self.usd_cap}",
                         params=dict(split=split, p0=p)) for i in range(len(starts)) for p in p0s], 0.0
        mtag = f"autoloop_{self.run_name}_{tag}"
        cmd = [PY, "-m", "modal", "run", "--detach", str(ATTACK / "modal_c1d.py"), "--start", ",".join(map(str, starts)),
               "--p0s", ",".join(f"{p:g}" for p in p0s), "--splits", str(split), "--budget", str(int(B)),
               "--tag", mtag, "--max-usd", f"{self.usd_cap - self.spent_usd:.2f}"]
        log = open(self.dir / f"{mtag}.log", "w")
        waves = -(-n // 100)                                   # modal_c1d.py: at most 100 containers at once
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO), env=clean_env(allow=MODAL_ENV), timeout=waves * (B + 600) + 1800)
        self.spent_usd += est
        outd = ATTACK / "c1d" / "modal" / mtag
        rows = []
        for i, start in enumerate(starts):
          for p0 in p0s:
            name = f"{start.stem}_x{split}_p{p0:g}_r0"
            f = self.dir / "certs" / f"{mtag}_{name}.npy"   # from the volume (final save of c1d_refine)
            subprocess.run([PY, "-m", "modal", "volume", "get", "--force", "revolutionaries-c1d",
                            f"{mtag}/{name}.npy", str(f)], capture_output=True, cwd=str(REPO), env=clean_env(allow=MODAL_ENV), timeout=600)
            if not f.exists() and (outd / f"{name}.npy").exists():   # our own tag folder written by the entrypoint
                f = outd / f"{name}.npy"
            if not f.exists():
                rows.append(dict(ok=False, pi=i, error=f"modal: no result for {name} (exit {p.returncode})",
                                 params=dict(split=split, p0=p0))); continue
            rows.append(dict(ok=True, pi=i, path=str(f), reported=score(np.load(f)),
                             params=dict(split=split, p0=p0, budget_s=B, executor="modal", modal_tag=mtag,
                                         usd_estimate=round(est / n, 4))))
        return rows, n * B


class FakeExecutor:
    """Dry run: no optimizer, deterministic fake scores, files are tiny real vectors. Seconds for a whole loop."""
    real = False
    usd_per_cpu_s = 0.001          # pretend cost, so the USD cap can be tested
    spent_usd = 0.0

    def __init__(self, workdir: Path, max_cells: int = LOCAL_MAX_CELLS, seed: int = 0):
        self.dir = Path(workdir); self.max_cells = max_cells
        (self.dir / "certs").mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

    def _save(self, name, cells):
        v = self.rng.normal(size=min(cells, 64)) + 1.0
        p = self.dir / "certs" / f"{name}.npy"; np.save(p, v)
        return str(p)

    def seed(self, n, budget_s, cells, seed0):
        return [dict(ok=True, path=self._save(f"seed{seed0 + i}", cells), reported=1.47 + 0.01 * self.rng.random(),
                     params=dict(seed=seed0 + i, cells=cells, budget_s=budget_s, fake_cells=cells)) for i in range(n)], 1.0 * n

    def refine(self, parent_path, split, p0s, budget_s, tag, parent_score=1.47):
        rows = []
        for p0 in p0s:
            gain = (3e-3 if split > 1 else 2e-5) * self.rng.random()
            rows.append(dict(ok=True, path=self._save(f"{tag}_x{split}_p{p0:g}", 1), reported=parent_score * (1 - gain),
                             params=dict(split=split, p0=p0, budget_s=budget_s)))
        return rows, 1.0 * len(p0s)

    def refine_many(self, parent_paths, split, p0s, budget_s, tag, parent_scores=None):
        rows = []
        for i, pp in enumerate(parent_paths):
            r, _ = self.refine(pp, split, p0s, budget_s, f"{tag}_{i:02d}", parent_score=parent_scores[i])
            rows += [dict(x, pi=i) for x in r]
        return rows, 1.0 * len(rows)

    def certify(self, parent_path, tag, parent_score=1.47):
        ex = Fraction(parent_score).limit_denominator(10 ** 15)
        return dict(ok=True, path=parent_path, ints=None, exact=ex, knot=0), 0.1


# ----------------------------------------------------------------------------------------------- reference
NUM = re.compile(r"1\.4[0-9]{4,}")


def fetch_reference() -> dict:
    """Best 1d upper bound, read live; fails closed (see autoloop/refread.py)."""
    from .refread import read
    return read("1d", REF_PAGE, NUM, 1.40, 1.50)


def claim_threshold(ref_value: str) -> Fraction:
    """A decimal claim d with k digits may stand for any value in [d - 5e-(k+1), d + ...): be conservative."""
    k = len(ref_value.split(".")[1])
    return Fraction(ref_value) - Fraction(5, 10 ** (k + 1))
