"""Fixed constants, frozen-file check, CLI and the Karpathy-mode entry point. DO NOT MODIFY.

This is the `prepare.py` of Karpathy's autoresearch. Together with problems/ and
interfaces.py it is hash-locked in frozen.lock: an experiment run while any of
them differs from the lock is recorded as `void`.

Karpathy mode: search.py (an optimizer file) ends with `run(__file__)`, so
    python search.py "my hypothesis" > run.log 2>&1
runs one experiment through interfaces.run_experiment and prints a summary block.

CLI:
    python prepare.py list                                  # problems and records
    python prepare.py verify <problem> <solution.json> [--exact]
    python prepare.py check                                 # frozen files intact?
    python prepare.py lock                                  # humans only, after a deliberate edit
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------- fixed constants
TIME_BUDGET = 60             # wall-clock seconds an optimizer may use, per experiment
SEED = 42                    # fixed seed: every experiment sees the same randomness
OVER_BUDGET_SLACK = 1.10     # an optimizer may overrun by 10% (+1s) before the result is void
HARD_KILL_FACTOR = 2.0       # watchdog: abort at 2x budget + 30s
MIN_IMPROVEMENT_REL = 1e-9   # keep only if better than the best kept score by more than this (relative)

ROOT = Path(__file__).resolve().parent
LOCK_FILE = ROOT / "frozen.lock"
SOLUTIONS_DIR = ROOT / "solutions"


def time_budget() -> float:
    """TIME_BUDGET, or AR_TIME_BUDGET (e.g. for tests). The value used is stored in every record."""
    return float(os.environ.get("AR_TIME_BUDGET", TIME_BUDGET))


# ---------------------------------------------------------------- frozen files
# WS2's truth layer. WS1's EinsteinArena problems (problems/<slug>/, catalog.py, _frozen.py) are frozen by
# WS1 itself (problems/MANIFEST.json + tests/test_ws1_frozen_problems.py), so they are not listed here.
WS2_PROBLEM_FILES = ("__init__.py", "common.py", "p1a.py", "p1b.py", "p1d.py", "p10c.py", "p11a.py")


def frozen_files() -> list[Path]:
    return ([ROOT / "prepare.py", ROOT / "interfaces.py", ROOT / "certificates.py"]
            + [ROOT / "problems" / f for f in WS2_PROBLEM_FILES])


def compute_hashes() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_files()}


def write_lock() -> None:
    LOCK_FILE.write_text(json.dumps(compute_hashes(), indent=2, sort_keys=True) + "\n")


def frozen_status() -> str:
    """'ok' or a description of what changed."""
    if not LOCK_FILE.exists():
        return "missing frozen.lock"
    expected, actual = json.loads(LOCK_FILE.read_text()), compute_hashes()
    changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
    return "ok" if not changed else "MODIFIED: " + ", ".join(changed)


# ---------------------------------------------------------------- Karpathy-mode entry point
def run(optimizer_path: str) -> dict:
    """Run the optimizer file as one experiment; the hypothesis is the command-line text."""
    import interfaces as I

    opt = I.FileOptimizer(optimizer_path)
    problem = I.get_problem(opt.problem_id)
    hypothesis = " ".join(sys.argv[1:]).strip() or "(no hypothesis given)"
    r = I.run_experiment(problem, opt, I.ExperimentConfig(hypothesis=hypothesis), I.JsonlStore(),
                         agent=os.environ.get("AR_AGENT", "coding-agent"))
    # Summary block, in the spirit of train.py's: `grep "^decision:\|^score:" run.log`.
    print("---")
    if r["score"] is not None:
        print(f"score:          {r['score']:.15g}")
    for key in ("status", "experiment_id", "problem", "best_before", "beats_record", "error",
                "budget_s", "duration_s", "seeds", "optimizer_hash", "certificate_path"):
        if r.get(key) is not None:
            print(f"{key + ':':<15} {r[key]}")
    print(f"{'direction:':<15} {problem.direction}")
    print(f"{'best_known:':<15} {problem.best_known}")
    print(f"{'decision:':<15} {'keep' if r['status'] == 'keep' else 'reset'}")
    if r["status"] == "crash":
        raise SystemExit(1)
    return r


# ---------------------------------------------------------------- CLI
def _cli(argv: list[str]) -> int:
    import interfaces as I
    import problems
    from certificates import CertificateStore

    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "list":
        for pid in problems.PROBLEMS:
            p = I.get_problem(pid)
            print(f"{p.id:4} {p.direction:8} best known {p.best_known:.15g}  {p.name}\n     {p.url}\n     {p.solution_format}")
        return 0
    if cmd == "lock":
        write_lock()
        print("frozen.lock written")
        return 0
    if cmd == "check":
        st = frozen_status()
        print(f"frozen: {st}")
        return 0 if st == "ok" else 1
    if cmd == "verify":
        if len(argv) < 3:
            print("usage: python prepare.py verify <problem> <solution.json> [--exact]")
            return 2
        p = I.get_problem(argv[1])
        src = argv[2]
        if not Path(src).exists() and "/" in src:      # a certificate id such as 1d/3f2a9c1d0b4e
            print(f"certificate {src}: re-verified on load")
            src = CertificateStore().load(p, src).path
        data = json.loads(Path(src).read_text())
        cand = data.get("solution", data)   # a solutions/<id>/*.json file or a bare candidate
        if not p.verify(cand):
            print(f"invalid:      {p.why_invalid(cand)}")
            return 1
        score = p.score(cand)
        print(f"score:        {score:.15g}")
        if "--exact" in argv:
            exact = p.certify(cand)
            if exact is None:
                print("exact:        (not available for this problem; checked in float64)")
            else:
                print(f"exact:        {float(exact):.17g}  (= {exact.numerator.bit_length()}-bit rational)")
                score = float(exact)
        print(f"best_known:   {p.best_known:.15g} ({p.direction})")
        print(f"beats_record: {p.beats_record(score)}")
        return 0
    print(f"unknown command {cmd!r}\n{__doc__}")
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
