"""WS1 arbiter: verify a claimed improvement. It never submits anything (submission is integrations/submit.py and
needs Arthur's --confirm).

    .venv/bin/python selection/verify_claim.py SLUG CANDIDATE.json [--claimed-score X] [--by AGENT]
                                               [--provenance LABEL]

CANDIDATE.json holds the solution object exactly as it would be submitted. --provenance records where the candidate
comes from (for example "milo_published_copy" or "clean"); it is logged as given and does not change the verdict.
SLUG is a problem of problems/catalog.py, or a problem frozen as a proposal (PROPOSED_SLUGS), which is loaded by
path because it is not registered in the catalog until Arthur accepts it. The arbiter:
  1. checks that the frozen problem is intact (verifier sha256 = frozen value) and that the live verifier is still
     byte-identical to it;
  2. runs the PINNED verifier in a clean process (python -I, the server's harness), plus the server schema check;
  3. runs the INDEPENDENT verifier (problems/<slug>/independent.py: different code, exact where possible);
  4. re-reads the LIVE #1 (leaderboard) and the live delta;
  5. returns one verdict and appends it to selection/claims.jsonl:
       invalid                 the schema, the pinned verifier or the independent verifier rejects the candidate
       verifier_disagreement   both accept it but differ by more than the independent tolerance
       valid_below_record      valid, but gain over the live #1 < delta (or a tie)
       beats_record            valid, both verifiers agree, and gain >= delta, gain > 0, gain > tolerance (for both
                               the pinned and the independent score; exactly, in Fractions, when the independent
                               score is exact)
     plus beats_record_by (the gain) and the gap between the two verifiers.
Exit code: 0 beats_record, 1 valid_below_record, 2 invalid or verifier_disagreement.

A verdict is about the candidate as given: the claim only becomes a record once the server scores the submission.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from integrations import einstein_arena as ea  # noqa: E402
from problems.catalog import PROBLEMS_DIR, SLUGS, load  # noqa: E402

CLAIMS = ROOT / "claims.jsonl"
EVAL_RUNNER = ROOT / "eval_runner.py"
# Frozen in problems/<slug>/ as a proposal (selection/freeze_problems.py, role "proposed_main"). problems/catalog.py
# is a frozen shared file and does not list them until Arthur accepts the proposal; then the slug moves to
# catalog.SLUGS and leaves this tuple.
PROPOSED_SLUGS = ("third-autocorrelation-inequality",)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def load_problem(slug: str):
    """problems/<slug>/problem.py: through the catalog, or by path for a proposed problem (same module name)."""
    if slug in SLUGS:
        return load(slug)
    if slug not in PROPOSED_SLUGS:
        raise KeyError(f"unknown frozen problem {slug!r}; known: {list(SLUGS + PROPOSED_SLUGS)}")
    name = "ws1_problem_" + slug.replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PROBLEMS_DIR / slug / "problem.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def run_pinned(verifier: Path, candidate_path: Path) -> dict:
    """Pinned verifier in a fresh, isolated interpreter (same harness as the server)."""
    cmd = [sys.executable, "-I", str(EVAL_RUNNER), str(verifier), str(candidate_path), "1", "600"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        return {"ok": False, "error": f"no output (rc={proc.returncode}): {proc.stderr[-400:]}"}
    return json.loads(lines[-1])


def arbitrate(slug: str, candidate_path: Path, claimed: float | None, by: str | None,
              provenance: str | None = None) -> dict:
    raw = candidate_path.read_bytes()
    rec: dict = {"timestamp": now(), "slug": slug, "candidate_file": str(candidate_path),
                 "candidate_sha256": hashlib.sha256(raw).hexdigest(), "claimed_score": claimed, "claimed_by": by,
                 "provenance": provenance}
    p = load_problem(slug)
    rec["frozen_verifier_sha256"] = p.VERIFIER_SHA256
    candidate = json.loads(raw)
    rec["json_chars"] = len(json.dumps(candidate, separators=(",", ":")))
    rec["submission_path"] = p.submission_path(candidate)

    # 1. live verifier and delta (the frozen definition must still be what the server runs)
    try:
        live = ea.get_problem(slug)
        rec["live_verifier_identical"] = hashlib.sha256(live["verifier"].encode()).hexdigest() == p.VERIFIER_SHA256
        delta = max(float(live["minImprovement"]), p.DELTA)
        rec["delta_live"] = float(live["minImprovement"])
    except ea.ArenaError as exc:
        rec["live_verifier_identical"] = None
        rec["live_error"] = str(exc)
        delta = p.DELTA
    rec["delta"] = delta

    # 2. schema + pinned verifier (clean process)
    schema_error = None
    try:
        p.check_schema(candidate)
    except p.InvalidSolution as exc:
        schema_error = str(exc)
    t0 = time.perf_counter()
    pinned = run_pinned(REPO_ROOT / "problems" / slug / "verifier_pinned.py", candidate_path)
    pinned_ok = bool(pinned.get("ok")) and schema_error is None
    pinned_score = pinned.get("score") if pinned.get("ok") else None
    if pinned_score is not None and not (pinned_score == pinned_score and abs(pinned_score) != float("inf")):
        pinned_ok = False
    rec["pinned"] = {"ok": pinned_ok, "score": pinned_score, "score_repr": pinned.get("score_repr"),
                     "schema_error": schema_error, "error": pinned.get("error"),
                     "seconds": round(time.perf_counter() - t0, 2)}

    # 3. independent verifier
    t0 = time.perf_counter()
    try:
        ind = p.independent_score(candidate)
        frac = ind.get("fraction")
        rec["independent"] = {"ok": True, "score": ind["score"], "tolerance_abs": ind["tolerance_abs"],
                              "exact": f"{frac.numerator}/{frac.denominator}" if frac is not None else None,
                              "method": ind["method"], "seconds": round(time.perf_counter() - t0, 2)}
    except p.InvalidSolution as exc:
        ind, frac = None, None
        rec["independent"] = {"ok": False, "error": str(exc), "seconds": round(time.perf_counter() - t0, 2)}

    # 4. live #1
    best = p.best_known_score(max_age_s=0)
    rec["live_best"] = best
    rec["target"] = best["value"] + delta if p.DIRECTION == "maximize" else best["value"] - delta

    # 5. verdict
    reasons = []
    if not pinned_ok or not rec["independent"]["ok"]:
        verdict = "invalid"
        if not pinned_ok and rec["independent"]["ok"]:
            reasons.append("pinned verifier rejects but the independent verifier accepts: investigate")
    else:
        tol = rec["independent"]["tolerance_abs"]
        gap = abs(pinned_score - ind["score"])
        rec["verifier_gap"] = gap
        gain = p.gain(pinned_score, best["value"])
        gain_ind = p.gain(ind["score"], best["value"])
        rec["beats_record_by"] = gain
        rec["beats_record_by_independent"] = gain_ind
        if frac is not None:
            g = (frac - Fraction(best["value"])) if p.DIRECTION == "maximize" else (Fraction(best["value"]) - frac)
            rec["beats_record_by_exact"] = float(g)   # exact rational gain, rounded only for the log
        if gap > tol:
            verdict = "verifier_disagreement"
            reasons.append(f"|pinned - independent| = {gap:.3e} > tolerance {tol:.1e}")
        elif all(g >= delta and g > 0 and g > tol for g in (gain, gain_ind)) and (frac is None or (
                (frac > Fraction(best["value"])) if p.DIRECTION == "maximize" else (frac < Fraction(best["value"])))):
            verdict = "beats_record"
        else:
            verdict = "valid_below_record"
            reasons.append(f"gain {gain:.3e} vs delta {delta:.1e} (ties never win)")
    if claimed is not None and pinned_score is not None:
        rec["claimed_minus_pinned"] = claimed - pinned_score
        if abs(claimed - pinned_score) > 1e-12 * max(1.0, abs(claimed)):
            reasons.append(f"claimed score {claimed!r} differs from the pinned score {pinned_score!r}")
    if rec.get("live_verifier_identical") is False:
        reasons.append("the live verifier changed since the freeze: the frozen definition may no longer apply")
    rec["verdict"] = verdict
    rec["reasons"] = reasons
    with CLAIMS.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="WS1 arbiter: pinned + independent verification against the live #1.")
    ap.add_argument("slug", choices=SLUGS + PROPOSED_SLUGS)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--claimed-score", type=float)
    ap.add_argument("--by", help="who claims it (agent / experiment id)")
    ap.add_argument("--provenance", help="where the candidate comes from, logged as given (e.g. milo_published_copy)")
    args = ap.parse_args()
    r = arbitrate(args.slug, args.candidate, args.claimed_score, args.by, args.provenance)
    print(f"{r['slug']}: {r['verdict'].upper()}")
    if r["provenance"]:
        print(f"  provenance  = {r['provenance']} (as declared; sha256 {r['candidate_sha256']})")
    print(f"  pinned      = {r['pinned']['score']!r} ({'ok' if r['pinned']['ok'] else 'REJECTED: ' + str(r['pinned']['schema_error'] or r['pinned']['error'])})")
    ind = r["independent"]
    print(f"  independent = {ind.get('score')!r} ({'ok' if ind['ok'] else 'REJECTED: ' + ind['error']})")
    if "verifier_gap" in r:
        print(f"  verifier gap = {r['verifier_gap']:.3e} (tolerance {ind['tolerance_abs']:.1e})")
    b = r["live_best"]
    print(f"  live #1 = {b['value']!r} by {b['agent']} ({'live' if b['live'] else 'SNAPSHOT'}); delta = {r['delta']}; "
          f"target = {r['target']!r}")
    if "beats_record_by" in r:
        print(f"  beats_record_by = {r['beats_record_by']:.3e} (needs >= {r['delta']:.1e} and > 0)")
    for reason in r["reasons"]:
        print(f"  note: {reason}")
    print(f"  logged to {CLAIMS.relative_to(REPO_ROOT)}; nothing was submitted")
    return {"beats_record": 0, "valid_below_record": 1}.get(r["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
