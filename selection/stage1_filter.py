"""Stage 1 of problem selection: fast screening of EinsteinArena problems.

Reproducible: clones EinsteinArena and the AlphaEvolve problem repository at
pinned commits, extracts each problem's metadata and verifier, applies the
screening rules below and writes one JSONL record per problem.

Usage: python selection/stage1_filter.py [--out selection/registry_stage1.jsonl]
"""
import argparse
import datetime
import json
import pathlib
import re
import subprocess
import tempfile
import urllib.request

EA_API_PROBLEMS = "https://einsteinarena.com/api/problems"
EA_REPO = "https://github.com/vinid/einstein-arena"
EA_SHA = "5baf6361a3e0e2123edb98f676ef44f7ab081ad6"
AE_REPO = "https://github.com/google-deepmind/alphaevolve_repository_of_problems"
AE_SHA = "8f447457957deac61e28bf1676746f0753b3b2f8"

# Problems whose record is an all-or-nothing integer jump (one more point,
# sphere, codeword...). Kept as at most one moonshot, never as main problem.
DISCRETE = {
    "kissing-number-d11", "kissing-number-d11-605", "kissing-number-d12",
    "kissing-number-d12-842", "kissing-number-d16", "no-three-in-line-75",
    "snake-in-the-box-13", "sorting-network-16", "two-deletion-code-16",
    "shannon-capacity-c7-5", "spencer-discrepancy", "sidon-45-set",
    "hadamard-det-51",
}

# Manual eliminations: each one needs a rationale and evidence.
ELIMINATED = {
    "erdos-142": (
        "Evaluation mode is 'proof' (Lean), not a construction with a numeric verifier.",
        [f"{EA_REPO}/blob/{EA_SHA}/web/src/lib/problems/erdos-142.ts"],
    ),
    "edges-vs-triangles": (
        "Underlying constant is known exactly (AlphaEvolve status 'matched_optimal', "
        "problem 46); the EinsteinArena score is a proxy, so no mathematical record "
        "can be beaten.",
        [f"{AE_REPO}/blob/{AE_SHA}/status.json"],
    ),
    "prime-number-theorem": (
        "Verifier is stochastic (1e7 random samples of the constraint), so a score is "
        "not a rigorous certificate; the target limit is known to equal 1 (PNT), and "
        "AlphaEvolve status is 'worse_than_record'. To double-check: Diamond-Erdos "
        "result on Chebyshev-type bounds approaching 1.",
        [f"{EA_REPO}/blob/{EA_SHA}/web/src/lib/problems/prime-number-theorem.ts",
         f"{AE_REPO}/blob/{AE_SHA}/status.json"],
    ),
}

# Known screening caveats to carry into stage 2.
NOTES = {
    "hexagon-packing": "Verifier accepts overlaps up to 1e-9: any gain near that scale is a tolerance artifact.",
    "ring-loading-15": "Exact rational arithmetic (fractions.Fraction): ideal for independent verification.",
    "third-autocorrelation-inequality": "Check normalisation against Tao constant 1d before comparing records.",
    "difference-bases": "Integer construction; score moves in discrete but fine-grained steps.",
    "uncertainty-principle": "AlphaEvolve status 'worse_than_record': check the human record first.",
    "sum-difference-2": "AlphaEvolve status 'worse_than_record': check the human record first.",
}


def live_slugs() -> set:
    """Active problems = those served by the live API (source of truth).

    The `hidden` flag in the source code does not match the live site in either direction (checked 2026-11-14),
    so it is recorded for information only.
    """
    req = urllib.request.Request(EA_API_PROBLEMS, headers={"User-Agent": "ws1-stage1-filter/1.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return {p["slug"] for p in json.load(resp)}


def clone(url: str, sha: str, dest: pathlib.Path) -> None:
    subprocess.run(["git", "clone", "-q", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", sha], check=True)


def field(text: str, pattern: str, default=None):
    m = re.search(pattern, text, re.S)
    return m.group(1) if m else default


def ae_status(ae_dir: pathlib.Path) -> dict:
    status = json.loads((ae_dir / "status.json").read_text())
    return {n: label for label, nums in status.items() for n in nums}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="selection/registry_stage1.jsonl")
    args = parser.parse_args()

    active_slugs = live_slugs()
    tmp = pathlib.Path(tempfile.mkdtemp())
    clone(EA_REPO, EA_SHA, tmp / "ea")
    clone(AE_REPO, AE_SHA, tmp / "ae")
    statuses = ae_status(tmp / "ae")
    prob_dir = tmp / "ea" / "web" / "src" / "lib" / "problems"
    verifier_dir = pathlib.Path(args.out).parent / "verifiers"
    verifier_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.date.today().isoformat()
    records = []
    for path in sorted(prob_dir.glob("*.ts")):
        if any(k in path.name for k in ("test", "index", "types", "lean-sum")):
            continue
        text = path.read_text()
        slug = field(text, r'slug:\s*"([^"]+)"')
        verifier = field(text, r"verifier:\s*`(.*?)`,\s*\n", "")
        verifier = verifier.replace("\\`", "`").replace("\\\\", "\\")
        if verifier:
            (verifier_dir / f"{slug}.py").write_text(verifier)
        reference = field(text, r'reference:\s*"([^"]+)"', "")
        ae_num = (field(reference, r"Problem 6\.(\d+) of https://arxiv.org/abs/2511.02864")
                  or field(reference, r"alphaevolve_repository_of_problems/problems/(\d+)\.html"))
        source_hidden = field(text, r"hidden:\s*(\w+)", "false") == "true"
        active = slug in active_slugs
        mode = field(text, r'evaluationMode:\s*"([^"]+)"', "construction")

        if slug in ELIMINATED:
            decision, (rationale, evidence) = "eliminate", ELIMINATED[slug]
        elif slug in DISCRETE:
            decision = "moonshot_only"
            rationale = "All-or-nothing integer record: too risky as main problem."
            evidence = [f"{EA_REPO}/blob/{EA_SHA}/web/src/lib/problems/{path.name}"]
        elif not active:
            decision = "reserve"
            rationale = ("Continuous score but not served by the live EinsteinArena API: no live "
                         "leaderboard, record must be established from the literature.")
            evidence = [f"{EA_REPO}/blob/{EA_SHA}/web/src/lib/problems/{path.name}", EA_API_PROBLEMS]
        else:
            decision = "keep_stage2"
            rationale = "Active, construction-based, continuous score, public verifier."
            evidence = [f"{EA_REPO}/blob/{EA_SHA}/web/src/lib/problems/{path.name}", EA_API_PROBLEMS]

        records.append({
            "stage": 1,
            "problem_id": f"einsteinarena/{slug}",
            "title": field(text, r'title:\s*"([^"]+)"'),
            "direction": field(text, r'scoring:\s*"([^"]+)"'),
            "min_improvement": field(text, r"minImprovement:\s*([^,\n]+)", "1e-4 (default)"),
            "evaluation_mode": mode,
            "active_on_platform": active,
            "active_source": EA_API_PROBLEMS,
            "source_hidden_flag": source_hidden,
            "alphaevolve_problem": int(ae_num) if ae_num else None,
            "alphaevolve_status": statuses.get(int(ae_num)) if ae_num else None,
            "verifier_uses_decimal": "Decimal" in verifier,
            "verifier_is_stochastic": bool(re.search(r"random|rng", verifier)),
            "decision": decision,
            "rationale": rationale,
            "notes": NOTES.get(slug, ""),
            "evidence_urls": evidence,
            "checked_on": today,
        })

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in records:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    print(f"{len(records)} problems screened -> {out}: {counts}")


if __name__ == "__main__":
    main()
