"""The autonomous loop: build the state from the ledger, ask the policy for a move, run it under contracts, write the
decision and its result, apply the claim gate, repeat until a cap or a stop.

    python -m autoloop.loop --run autoloop/runs/NAME [--memory SRC ...] [--policy rules|llm] [--executor fake|local]
                            [--max-wall-s S] [--max-decisions N] [--max-usd X] [--llm-usd-cap 10] ...

Claim gate: a certificate is called "below the reference" only if its exact fraction (certify) is strictly below the
conservative threshold of a reference read less than 30 minutes earlier. It writes a claim record. Nothing is ever
submitted anywhere.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
import subprocess
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

from . import problem as problem_mod
from .ledger import Ledger, import_memory
from .moves import ContractError, Moves
from .safety import safe_err
from .policy import compact_args, LLMPolicy, RulesPolicy, legal_moves, lineages, load_env, pick_provider

REF_MAX_AGE_S = 1800
WORKSPACE = __import__("os").environ.get("AUTOLOOP_WORKSPACE", str((lambda q: q[2] if len(q) > 2 else q[-1])(__import__("pathlib").Path(__file__).resolve().parents)))  # folder that holds repo/, wt-compare/, shared/ and .env
DEFAULT_ENV = WORKSPACE + "/.env"


def best_cert(certs):
    c = [x for x in certs if x["stage"] != "certified"]
    return min(c, key=lambda x: x["score"]) if c else None


def build_state(L, prob, cfg, t_start, moves, policy, max_cells):
    certs = L.certs()
    refs = [r for r in L.of_kind("reference") if r.get("ok")]
    ref = refs[-1] if refs else None
    return dict(prob=prob, cfg=cfg, certs=certs, decisions=L.decisions(), results=L.of_kind("result"),
                best=best_cert(certs), ref=ref, ref_age_s=(time.time() - ref["t"]) if ref else None,
                wall_left_s=cfg["max_wall_s"] - (time.time() - t_start),
                decisions_left=cfg["max_decisions"] - cfg["_n_dec"], usd_left=cfg["max_usd"] - moves.usd,
                llm_usd_left=cfg["llm_usd_cap"] - getattr(policy, "usd", 0.0), max_cells=max_cells,
                memory_best=(L.of_kind("memory_baseline") or [{}])[-1].get("best_exact_float"), executor=cfg["executor"],
                guidelines=cfg.get("_guidelines"))


def claim_gate(L, prob, cert):
    refs = [r for r in L.of_kind("reference") if r.get("ok")]
    if not cert.get("exact"):
        return None
    if not refs or time.time() - refs[-1]["t"] > REF_MAX_AGE_S:
        return L.append(dict(kind="note", text=f"claim gate: no reference younger than {REF_MAX_AGE_S} s for {cert['id'][:10]}; "
                                               "no claim written"))
    ref = refs[-1]
    ex = Fraction(cert["exact"])
    thr = prob.claim_threshold(ref["value"])
    below_ref = ex < thr
    mem = L.of_kind("memory_baseline")
    mem_best = Fraction(mem[-1]["best_exact"]) if mem else None
    below_mem = mem_best is None or ex < mem_best
    below = below_ref and below_mem
    txt = ("BELOW the live reference" if below_ref else "not below the live reference")
    if mem_best is not None:
        txt += (" and BELOW" if below_mem else " and not below") + f" the memory's best exact {float(mem_best):.16f}"
    return L.append(dict(kind="claim", cert=cert["id"], exact=cert["exact"], exact_float=float(ex), reference=ref["value"],
                         reference_source=ref["source"], reference_age_s=round(time.time() - ref["t"]),
                         threshold=float(thr), below_reference=below_ref,
                         memory_best_exact=float(mem_best) if mem_best is not None else None,
                         below_memory=below_mem, below_target=below, submitted=False,
                         wording="below the snapshot, own check",
                         text=(f"BELOW THE REFERENCE SNAPSHOT read {round(time.time() - ref['t'])} s ago, OWN CHECK "
                               "(our exact checks only; not yet verified by the arbiter; nothing submitted): " if below else "")
                              + txt + " (exact check passed)"))


def run(cfg):
    prob = problem_mod.load(cfg["problem"])
    rundir = Path(cfg["run"]); rundir.mkdir(parents=True, exist_ok=True)
    L = Ledger(rundir / "ledger.jsonl")
    if cfg["executor"] == "fake":
        ex = prob.FakeExecutor(rundir, max_cells=cfg["max_cells"], seed=cfg["fake_seed"])
    elif cfg["executor"] == "hybrid":
        if cfg["max_usd"] >= 1e9:
            raise SystemExit("--max-usd is required with --executor hybrid")
        ex = prob.HybridExecutor(rundir, max_cells=cfg["max_cells"], modal_budget_s=cfg["modal_budget_s"],
                                 usd_cap=cfg["max_usd"], run_name=rundir.name)
    elif cfg["executor"] == "modal":
        if cfg["max_usd"] >= 1e9:
            raise SystemExit("--max-usd is required with --executor modal")
        from .modal_exec import ModalExecutor
        ex = ModalExecutor(rundir, prob, max_cells=cfg["max_cells"], usd_cap=cfg["max_usd"], run_name=rundir.name)
    else:
        ex = prob.LocalExecutor(rundir, max_cells=min(cfg["max_cells"], prob.LOCAL_MAX_CELLS))
    fetch = cfg.get("_ref_fetcher")
    if fetch is None and cfg["executor"] == "fake":
        fetch = lambda: dict(ok=True, value="1.4546733414", source="fake reference (dry run)")  # noqa: E731
    moves = Moves(L, ex, prob, refine_budget_s=cfg["refine_budget_s"], ref_fetcher=fetch,
                  modal_budget_s=cfg["modal_budget_s"], seed_base=cfg["seed_base"])
    ex.workers = cfg["workers"]
    if cfg["policy"] == "llm":
        load_env(cfg["env_file"])
        provider = pick_provider(cfg["llm_provider"])
        model = cfg["llm_model"] or (os.environ.get("OPENAI_MODEL") if provider == "openai" else None)
        if provider in ("openai", "anthropic") and not model:
            raise SystemExit(f"--llm-model (or OPENAI_MODEL) is required for provider {provider}")
        policy = LLMPolicy(provider, model, rundir / "llm", cfg["llm_price_in"], cfg["llm_price_out"], cfg["llm_usd_cap"])
    else:
        policy = RulesPolicy()
    for src in cfg["memory"]:
        n = import_memory(L, src, prob)
        L.append(dict(kind="note", text=f"memory: imported {n} certificates from {src}"))
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=str(Path(__file__).parent)).stdout.strip()  # noqa: E731
    L.append(dict(kind="run_start", policy=policy.name, executor=cfg["executor"], commit=os.environ.get("AUTOLOOP_COMMIT") or git("rev-parse", "HEAD"),
                  autoloop_dirty=(os.environ.get("AUTOLOOP_COMMIT_VERIFIED") != "1") and bool(git("status", "--porcelain", "--", ".")),
                  code_dir=str(Path(__file__).resolve().parent),
                  pastdev_read=bool(cfg.get("pastdev")), pastdev_scope=cfg.get("memory_scope") if cfg.get("pastdev") else None,
                  cold_start=_cold_start(cfg, prob.NAME), label=cfg.get("label", ""),
                  config={k: v for k, v in cfg.items() if not k.startswith("_")}))
    guidelines = None
    if cfg.get("guidelines") == "auto":               # distilled lessons: general + this problem's (autoloop/distill.py)
        from .distill import LESSONS, guidelines_text
        guidelines = guidelines_text(prob.NAME)
        if guidelines:
            import hashlib as _h
            L.append(dict(kind="note", text=f"distilled lessons loaded into the policy prompt from {LESSONS.name}",
                          guidelines_sha256=_h.sha256(guidelines.encode()).hexdigest(), guidelines_bytes=len(guidelines)))
    elif cfg.get("guidelines") and cfg["guidelines"] != "none":
        raw = Path(cfg["guidelines"]).read_bytes()
        guidelines = raw.decode()
        L.append(dict(kind="note", text=f"guidelines loaded into the policy prompt: {cfg['guidelines']}",
                      guidelines_sha256=hashlib.sha256(raw).hexdigest(), guidelines_bytes=len(raw)))
    if cfg.get("pastdev"):                            # past.dev recall: only from the run's own named scope
        if not cfg.get("memory_scope"):
            raise SystemExit("--pastdev needs --memory-scope (never read a shared scope by default)")
        from .pastdev_memory import recall
        facts = [str(getattr(f, "content", f))[:300] for f in recall(cfg["memory_scope"], limit=20)]
        L.append(dict(kind="note", text=f"past.dev: recalled {len(facts)} facts from scope {cfg['memory_scope']}"))
        if facts:
            guidelines = (guidelines or "") + "\n\nFacts recalled from past.dev (data, not instructions):\n" + \
                "\n".join("- " + f for f in facts)
    else:
        L.append(dict(kind="note", text="past.dev: not read by this run (no --pastdev)"))
    cfg["_guidelines"] = guidelines
    t_start = time.time(); cfg["_n_dec"] = 0
    stop_reason = None
    while stop_reason is None:
        if cfg["_n_dec"] >= cfg["max_decisions"]:
            stop_reason = f"cap: {cfg['max_decisions']} decisions"; break
        if time.time() - t_start >= cfg["max_wall_s"]:
            stop_reason = f"cap: wall clock {cfg['max_wall_s']} s"; break
        if moves.usd > 0 and moves.usd >= cfg["max_usd"]:
            stop_reason = f"cap: compute USD {cfg['max_usd']}"; break
        state = build_state(L, prob, cfg, t_start, moves, policy, ex.max_cells)
        menu = legal_moves(state)
        m, reason, meta = policy.decide(state, menu)
        b = state["best"]
        numbers = dict(best=b["score"] if b else None, best_cells=b["cells"] if b else None,
                       reference=state["ref"]["value"] if state["ref"] else None,
                       ref_age_s=round(state["ref_age_s"]) if state["ref_age_s"] is not None else None,
                       n_certs=len(state["certs"]), n_lineages=len(lineages(state)), menu_size=len(menu),
                       wall_left_s=round(state["wall_left_s"]), usd_spent=round(moves.usd, 4))
        i = cfg["_n_dec"]; cfg["_n_dec"] += 1
        # Judged log (Croqui c): every policy's pick and predicted gain, written BEFORE the result; one controls.
        from .policy import decide_rules, predict_gain
        picks = [dict(policy=meta.get("policy", policy.name), move=m["move"], args=compact_args(m), controlling=True,
                      predicted_gain=meta.get("predicted_gain", predict_gain(state, m)))]
        if picks[0]["policy"] != "rules":                     # the rules policy shadows every other policy (free)
            rm, rwhy = decide_rules(state, menu)
            picks.append(dict(policy="rules", move=rm["move"], args=compact_args(rm), controlling=False,
                              predicted_gain=predict_gain(state, rm), reason=rwhy[:200]))
        L.append(dict(kind="picks", i=i, picks=picks))
        L.append(dict(kind="decision", i=i, options=[f"{x['move']} {json.dumps(x['args'])}" for x in menu],
                      move=m["move"], args=m["args"], reason=reason, numbers=numbers, **meta))
        t0 = time.time()
        res = dict(kind="result", i=i, move=m["move"])
        try:
            if m["move"] == "stop":
                stop_reason = m["args"].get("reason", "policy stop") + f" -- {reason}"
                res["summary"] = "stopped"
            elif m["move"] == "check_reference":
                r = moves.check_reference()
                L.append(dict(kind="reference", **r))
                res["summary"] = f"reference {r.get('value')} from {r.get('source')}" if r.get("ok") else "reference read failed"
                if r.get("ok"):          # re-gate certified results that never had a fresh reference
                    gated = {c["cert"] for c in L.of_kind("claim")}
                    for c in [c for c in L.certs() if c["stage"] == "certified" and c["id"] not in gated
                              and c.get("origin") == "autoloop"]:
                        cl = claim_gate(L, prob, c)
                        if cl and cl.get("below_target"):
                            (rundir / "CLAIM_PASSED.json").write_text(json.dumps(cl, indent=1))
                            print(f"CLAIM GATE PASSED: {cl['text']}", flush=True)
                            res["summary"] += f"; re-gated {c['id'][:8]}: {cl['text']}"
            else:
                fn = getattr(moves, m["move"])
                new, info = fn(**m["args"])
                for c in new:
                    if L.cert(c["id"]) is None:
                        L.append(c)
                res.update(info)
                res["new"] = [c["id"] for c in new]
                if new:
                    top = min(new, key=lambda c: c["score"])
                    res["best_new"] = top["score"]; res["root"] = top.get("root")
                    if m["move"] == "refine":
                        par = L.cert(m["args"]["cert"])
                        res["rel_gain"] = (par["score"] - top["score"]) / par["score"]
                    if m["move"] in ("refine", "refine_wide"):     # realized: the largest parent-to-child drop
                        res["realized_gain"] = max(L.cert(c["parent"])["score"] - c["score"] for c in new
                                                   if c.get("parent") and L.cert(c["parent"]))
                    res["summary"] = (f"{len(new)} certificate(s), best {top['score']:.13f} at {top['cells']} cells"
                                      + (f", gain {res['rel_gain']:.2e}" if "rel_gain" in res else ""))
                    if m["move"] == "certify":
                        cl = claim_gate(L, prob, top)
                        res["summary"] = f"exact {top['score']:.15f}; " + (cl.get("text", "") if cl else "")
                        if cl and cl.get("below_target"):
                            (rundir / "CLAIM_PASSED.json").write_text(json.dumps(cl, indent=1))
                            print(f"CLAIM GATE PASSED: {cl['text']}", flush=True)
                else:
                    res["summary"] = "no certificate produced"
        except ContractError as e:
            res.update(error=safe_err(f"contract: {e}"), summary=safe_err(f"contract violation: {e}"))
        except Exception as e:  # noqa: BLE001
            res.update(error=safe_err(repr(e)), trace=safe_err(traceback.format_exc(), n_lines=6, limit=800), summary=("error: " + safe_err(repr(e)))[:200])
        res["seconds"] = round(time.time() - t0, 1)
        res["best_after"] = (best_cert(L.certs()) or {}).get("score")
        L.append(res)
        if not cfg["quiet"]:
            print(f"[{i:03d}] {m['move']:<15} {res.get('summary', '')}  ({meta['policy']}: {reason[:90]})", flush=True)
    b = best_cert(L.certs())
    certified = [c for c in L.certs() if c["stage"] == "certified"]
    end = dict(kind="run_end", reason=stop_reason, decisions=cfg["_n_dec"], wall_s=round(time.time() - t_start, 1),
               cpu_s=round(moves.cpu_s, 1), usd=round(moves.usd, 4), llm_usd=round(getattr(policy, "usd", 0.0), 4),
               best=b["score"] if b else None,
               best_exact=min((c["score"] for c in certified), default=None),
               claims_below=sum(1 for c in L.of_kind("claim") if c.get("below_target", c["below_reference"])))
    L.append(end)
    if cfg.get("distill") and cfg["executor"] != "fake":
        try:
            from .distill import distill
            d = distill([L.path], outdir=rundir / "llm", pastdev=True)
            L.append(dict(kind="note", text=f"lessons distilled: generation {d['generation']}, {len(d['applied'])} changes "
                                            f"({', '.join(a['action'] + ' ' + str(a['id']) for a in d['applied'])}), "
                                            f"{d['rejected']} rejected, USD {d['usd']}"))
        except Exception as e:  # noqa: BLE001
            L.append(dict(kind="note", text="distiller failed: " + safe_err(repr(e))))
    if not cfg["quiet"]:
        print(f"stopped: {stop_reason}; best {end['best']}, best exact {end['best_exact']}", flush=True)
    return end


def _cold_start(cfg, problem):
    """Cold = no imported memory, no past.dev, and no knowledge of THIS problem: only general lessons (R14: a run
    from its problem's lessons or a guidelines file is exploitation)."""
    if cfg["memory"] or cfg.get("pastdev"):
        return False
    g = cfg.get("guidelines")
    if g == "auto":
        from .distill import load
        return not any(x["scope"] == problem and x["status"] != "dropped" for x in load())
    return g in (None, "none")


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (ledger.jsonl, certs/, llm/)")
    ap.add_argument("--problem", default="tao-1d")
    ap.add_argument("--memory", action="append", default=[], help="ledger.jsonl, directory or file of solutions")
    ap.add_argument("--policy", choices=["rules", "llm"], default="rules")
    ap.add_argument("--llm-provider", choices=["auto", "openai", "anthropic", "claude-cli", "fake"], default="auto")
    ap.add_argument("--llm-model", default=None, help="model name (default: OPENAI_MODEL for openai)")
    ap.add_argument("--llm-usd-cap", type=float, default=10.0)
    ap.add_argument("--llm-price-in", type=float, default=0.0, help="USD per 1M input tokens (openai/anthropic)")
    ap.add_argument("--llm-price-out", type=float, default=0.0, help="USD per 1M output tokens (openai/anthropic)")
    ap.add_argument("--llm-price-note", default="", help="where the prices come from (e.g. an assumption)")
    ap.add_argument("--env-file", default=DEFAULT_ENV, help="loaded at run time for the LLM keys only")
    ap.add_argument("--executor", choices=["fake", "local", "hybrid", "modal"],
                    help="hybrid = local up to 12,800 cells, Modal above (modal run clients); modal = every seed and refine on the deployed app autoloop/modal_app.py", default="local")
    ap.add_argument("--max-cells", type=int, default=12800)
    ap.add_argument("--max-wall-s", type=float, default=1200)
    ap.add_argument("--max-decisions", type=int, default=100)
    ap.add_argument("--max-usd", type=float, default=1e9, help="compute USD cap (required for hybrid)")
    ap.add_argument("--modal-budget-s", type=float, default=3000, help="seconds per Modal restart (hybrid)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed-budget-s", type=float, default=60)
    ap.add_argument("--refine-budget-s", type=float, default=120)
    ap.add_argument("--min-roots", type=int, default=4)
    ap.add_argument("--promote", type=int, default=3)
    ap.add_argument("--menu-lineages", type=int, default=6)
    ap.add_argument("--ref-refresh-s", type=float, default=1500)
    ap.add_argument("--certify-gain", type=float, default=1e-6, help="certify again when the best gains this much")
    ap.add_argument("--pastdev", action="store_true", help="recall past.dev facts into the prompt (off by default)")
    ap.add_argument("--memory-scope", default=None, help="the past.dev scope --pastdev reads (required with it)")
    ap.add_argument("--guidelines", default="auto",
                    help="'auto' = distilled lessons (general + this problem), a text file, or 'none' (sha256 logged)")
    ap.add_argument("--no-distill", dest="distill", action="store_false",
                    help="do not run the lessons distiller at the end of the run")
    ap.add_argument("--seed-sizes", type=lambda s: [int(x) for x in s.split(",")], default=[],
                    help="extra seed batch sizes offered in the menu, e.g. 64,200")
    ap.add_argument("--wide-sizes", type=lambda s: [int(x) for x in s.split(",")], default=None,
                    help="lineage counts offered for refine_wide, e.g. 64,30,16 (default: --wide-top)")
    ap.add_argument("--final-width", type=int, default=30, help="rules: lineages taken into the top level")
    ap.add_argument("--seed-base", type=int, default=1000, help="first random seed of this run's seeds")
    ap.add_argument("--wide-top", type=int, default=16, help="lineages in a top-level soft-restart round")
    ap.add_argument("--label", default="", help="plain description of the run's starting knowledge, written in run_start")
    ap.add_argument("--recipe", choices=["wide", "promote", "d7"], default="wide",
                    help="rules: wide (all lineages up, then soft restarts on a wide set) or promote (top few)")
    ap.add_argument("--fake-seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    return ap


def cfg_from(argv):
    return {k.replace("-", "_"): v for k, v in vars(parser().parse_args(argv)).items()}


if __name__ == "__main__":
    run(cfg_from(sys.argv[1:]))
