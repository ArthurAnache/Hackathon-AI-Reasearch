"""Run Knockout Evolution: the Hill Sampling control arm, plus the main arm (genes, gate, islands) when the LLM
editor is used.

    .venv/bin/python -m knockout.run --problem toy --wall-clock 120
    .venv/bin/python -m knockout.run --problem toy --set control.edits_per_round=8 --set fidelity.f0_s=3
    .venv/bin/python -m knockout.run --problem second-autocorrelation-inequality --profile large-warm-start \
        --init live --editor openai --wall-clock 3600 --max-usd 10

On Modal the same `run()` is called by the orchestrator function in knockout/modal_app.py.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from knockout.budget import BudgetGovernor
from knockout.config import Strategy
from knockout.context import RunContext, host_info
from knockout.control import HillSamplingArm
from knockout.editor import Editor, OfflineParamEditor
from knockout.executor import Executor, make_executor
from knockout.ledger import Ledger, validate_run
from knockout.problems_api import TOY, load_problem

ROOT = Path(__file__).resolve().parent
SEEDS = {TOY: ROOT / "seeds" / "toy_circles.py",
         "second-autocorrelation-inequality": ROOT / "seeds" / "second_autocorrelation.py"}

# Named sets of strategy overrides. Explicit --set values win over a profile.
PROFILES = {
    # A problem searched from one large warm start (2M values): one construct call per evaluation, gains
    # accumulated through the warm-start ratchet, and only the candidates the run depends on are stored.
    "large-warm-start": {"fidelity.f0_s": 60.0, "tuner.calls": 1, "tuner.top_k": 1, "control.ratchet": True,
                         "control.edits_per_round": 6, "executor.workers": 7, "selection.eps_rel": 1.0e-11,
                         "sandbox.timeout_slack_s": 25, "ledger.save_solutions": "selected",
                         # main arm: one fidelity (a longer run from the same warm start is what the next batch
                         # does anyway), 6 children per batch of which up to 2 gate pairs
                         "main.batch_size": 6, "gate.pairs_per_batch": 2, "fidelity.levels": 1,
                         "strategist.min_progress": 1.0e-7},
}


def load_init(problem, spec: str | None) -> tuple[object, dict]:
    """The warm start and where it comes from. `spec`: None (the frozen certificate), "live" (the live #1 of
    EinsteinArena, cached under selection/.cache/live/), or a path to a .json / .json.gz candidate."""
    if spec is None:
        if problem.initial_candidate is None:
            return None, {"source": "none"}
        return problem.initial_candidate(), {"source": "frozen certificate"}
    if spec == "live":
        from integrations import einstein_arena as ea
        from problems.catalog import load

        sol = ea.get_best_solutions(load(problem.id).PROBLEM_ID, limit=1)[0]
        cand = sol["data"]
        raw = json.dumps(cand, separators=(",", ":")).encode()
        path = ROOT.parent / "selection" / ".cache" / "live" / problem.id / f"solution_{sol['id']}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(gzip.compress(raw, 6))
        return cand, {"source": "live #1", "solution_id": sol["id"], "agent": sol.get("agentName"),
                      "displayed_score": sol.get("score"), "sha256": hashlib.sha256(raw).hexdigest(),
                      "path": str(path.relative_to(ROOT.parent))}
    path = Path(spec)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(raw), {"source": "file", "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def make_editor(ctx: RunContext) -> Editor:
    kind = ctx.strategy.get("editor.kind", "offline")
    if kind == "offline":
        return OfflineParamEditor()
    if kind == "openai":
        from knockout.llm_editor import OpenAIEditor

        if float(ctx.strategy["run.max_usd"]) <= 0:
            raise SystemExit("the LLM editor needs a spend cap: --max-usd > 0")
        editor = OpenAIEditor(ctx)
        editor.preflight()
        return editor
    raise SystemExit(f"unknown editor.kind {kind!r}")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def new_run_id(problem_id: str, commit: str | None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{problem_id}-{(commit or 'nogit')[:7]}"


def run(problem_id: str, *, runs_root: str | Path = "runs", run_id: str | None = None,
        overrides: dict | None = None, executor: Executor | None = None, source_commit: str | None = None,
        modal_fn=None, on_round=print, init_spec: str | None = None, editor: Editor | None = None) -> dict:
    strategy = Strategy.load(overrides=overrides)
    problem = load_problem(problem_id)
    seed_path = SEEDS.get(problem.id)
    if seed_path is None:
        raise SystemExit(f"no seed program for {problem.id} yet (knockout/seeds/)")
    source_commit = source_commit or _git_commit()
    run_id = run_id or new_run_id(problem.id, source_commit)
    run_dir = Path(runs_root) / run_id
    ledger = Ledger(run_dir, run_id)
    (run_dir / "strategy.yaml").write_text(yaml.safe_dump(strategy.snapshot(), sort_keys=False))
    governor = BudgetGovernor(strategy["run.wall_clock_s"], strategy["run.max_usd"], strategy["budget_shares"])
    executor = executor or make_executor(strategy, modal_fn=modal_fn)
    init, init_info = load_init(problem, init_spec)
    # The warm start is scored here, in the trusted process, by the frozen verifier (an invalid one stops the run).
    init_score = float(problem.score(init)) if init is not None else None
    ctx = RunContext(run_id, problem, strategy, ledger, governor, executor, source_commit,
                     device="cpu", init=init, init_score=init_score)
    rng = np.random.default_rng(int(strategy["run.seed"]))
    editor = editor or make_editor(ctx)
    seed_source = seed_path.read_text()
    control = HillSamplingArm(ctx, editor, seed_source, rng)
    # The main arm (genes, gate, islands) needs the LLM. It has its own RNG stream and its own warm start, so
    # the control arm stays isolated from it (section 5.10).
    arms_cfg = strategy.get("run.arms", "auto")
    main = strategist = None
    if ctx.llm is not None and arms_cfg in ("auto", "both"):
        from knockout.main_arm import MainArm
        from knockout.meta import Strategist, scout

        main = MainArm(ctx, ctx.llm, seed_source, np.random.default_rng(int(strategy["run.seed"]) + 1))
        strategist = Strategist(main)
    ledger.event("run_start", seed=int(strategy["run.seed"]), problem=problem.id, executor=executor.name,
                 source_commit=source_commit, host=host_info(), arms=["control"] + (["main"] if main else []),
                 editor=editor.name, best_known=ctx.best_known(), best_known_source=problem.best_known_source,
                 init={**init_info, "score": init_score})
    t0 = time.monotonic()
    stop_reason = "error"
    steps = {"control": 0, "main": 0}
    try:
        base = control.baseline()
        on_round(f"control baseline: valid={base.valid} score={base.score} ({base.elapsed_s:.1f}s)")
        if main is not None:
            mb = main.start()
            on_round(f"main baseline: valid={mb.valid} score={mb.score}; {len(main.genes.genes) - 1} seed genes")
            if not mb.valid:
                raise SystemExit(f"the seed program is invalid in the main arm: {mb.error}")
            scout(main, "start")
        empty = {"control": 0, "main": 0}
        while True:
            if governor.exhausted() or governor.remaining_s() < float(strategy["fidelity.f0_s"]) + 5:
                stop_reason = "budget"
                break
            # The control arm gets its fixed share (20 %); everything else goes to the main arm.
            name = "control" if main is None or governor.control_owed() else "main"
            if name == "control":
                info = control.step()
                on_round(f"control round {info['round']}: {info['valid']}/{info['n']} valid, best child "
                         f"{info['best_child']}{'  (program improved)' if info['improved'] else ''}, best "
                         f"{info['best']}{'  (new best)' if info['new_best'] else ''}, usd {governor.total_usd():.3f}")
            else:
                info = main.step()
                on_round(f"main batch {info['batch']} [{info['island']}, {info['phase']}]: {info['written']}/"
                         f"{info['n']} written, {info['valid']} valid, {info['kept']} beat their parent, best "
                         f"{info['best']}{'  (new best)' if info['new_best'] else ''}, genes {info['genes']}, "
                         f"usd {governor.total_usd():.3f}")
                meta_out = strategist.step()
                if meta_out:
                    on_round(f"strategist: {meta_out}")
            steps[name] += 1
            empty[name] = 0 if info["n_edits"] else empty[name] + 1
            if ctx.llm is not None and ctx.llm.budget_stopped:
                stop_reason = "usd cap reached (LLM calls no longer fit)"
                break
            if empty[name] >= 2:  # two steps without a single usable program: stop instead of burning the budget
                stop_reason = f"{name} arm got no program twice: {getattr(ctx.llm, 'last_error', None)}"
                break
    finally:
        # Best verified candidate of the run, over both arms.
        arms = {"control": {"best": control.best_score, "best_experiment": control.best_experiment,
                            "best_solution_path": control.best_solution_path, "steps": steps["control"],
                            "spend": governor.spend("control")}}
        if main is not None:
            arms["main"] = {"best": main.best_score, "best_experiment": main.best_experiment,
                            "best_solution_path": main.best_solution_path, "steps": steps["main"],
                            "spend": main._spend(), "evaluations": main.evals}
            main.save_state()
        scored = [(k, v) for k, v in arms.items() if v["best"] is not None]
        best_arm = None
        for k, v in scored:
            if best_arm is None or problem.gain(v["best"], arms[best_arm]["best"]) > 0:
                best_arm = k
        top = arms[best_arm] if best_arm else {"best": None, "best_experiment": None, "best_solution_path": None}
        claims = [e for e in ledger.events() if e["type"] == "promotion"
                  and e["payload"]["outputs"]["decision"] == "record_claim"]
        summary = {"run_id": run_id, "problem": problem.id, "editor": editor.name, "stop_reason": stop_reason,
                   "init_score": init_score, "baseline": base.score if "base" in locals() else None,
                   "best": top["best"], "best_arm": best_arm, "best_experiment": top["best_experiment"],
                   "best_solution_path": top["best_solution_path"], "arms": arms,
                   "program_incumbent": control.incumbent.program_id, "rounds": control.rounds,
                   "wall_s": round(time.monotonic() - t0, 1), "budget": governor.summary(),
                   "control_curve": control.curve, "main_curve": main.curve if main is not None else [],
                   "best_known": ctx.best_known(),
                   # True only when the promotion gate logged a record_claim (section 5.11); still to be
                   # confirmed by selection/verify_claim.py.
                   "record_claims": [e["refs"][0] for e in claims], "beats_record": bool(claims)}
        ledger.event("run_end", seed=None, refs=[top["best_experiment"]] if top["best_experiment"] else [],
                     **{k: v for k, v in summary.items() if k not in ("control_curve", "main_curve")})
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if executor is not None:
            executor.shutdown()
        try:
            from knockout.report import write as write_report

            write_report(run_dir)
        except Exception as exc:  # noqa: BLE001  (the report must never hide the run's own error)
            on_round(f"report failed: {type(exc).__name__}: {exc}")
    summary["validation"] = validate_run(run_dir)
    return summary


def _parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, raw = item.partition("=")
        out[key] = yaml.safe_load(raw)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="toy")
    ap.add_argument("--wall-clock", type=float, help="seconds (overrides run.wall_clock_s)")
    ap.add_argument("--max-usd", type=float, help="USD cap, compute + LLM (overrides run.max_usd)")
    ap.add_argument("--executor", choices=("local", "modal"))
    ap.add_argument("--editor", choices=("offline", "openai"), help="who writes the edits (overrides editor.kind)")
    ap.add_argument("--init", help='warm start: "live" (live #1 of EinsteinArena) or a .json/.json.gz candidate; '
                                   "default: the frozen certificate")
    ap.add_argument("--profile", choices=sorted(PROFILES), help="a named set of strategy overrides")
    ap.add_argument("--runs-root", default=str(ROOT.parent / "runs"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a strategy.yaml key")
    a = ap.parse_args(argv)
    overrides = {**(PROFILES[a.profile] if a.profile else {}), **_parse_set(a.set)}
    if a.wall_clock is not None:
        overrides["run.wall_clock_s"] = a.wall_clock
    if a.max_usd is not None:
        overrides["run.max_usd"] = a.max_usd
    if a.executor:
        overrides["executor.kind"] = a.executor
    if a.editor:
        overrides["editor.kind"] = a.editor
    summary = run(a.problem, runs_root=a.runs_root, overrides=overrides, init_spec=a.init)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("control_curve", "main_curve")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
