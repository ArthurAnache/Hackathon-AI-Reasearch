"""Model-written stage (T5b): a model writes one optimizer stage from the verifier and the format alone; it is kept
only if the verifiers confirm a gain.

    python -m autoloop.stage_writer --run autoloop/runs/t5b_circles --problem ea:circles-rectangle \
        --start-ledger autoloop/runs/t5_circles/ledger.jsonl --provider openai [--model M] --budget-s 600 --jobs 8

1. Prompt: the pinned verifier's source, the solution format, the direction, the dimension. No scores, no reference,
   no method notes, no earlier runs. The model answers {"code": "..."} defining
       refine(v: np.ndarray, score, budget_s: float, rng: np.random.Generator) -> np.ndarray
   where score(v) is the pack's score (lower is better, +inf when the verifier rejects).
2. The code is saved (sha256 in the ledger) and runs ONLY on Modal (job kind pack_stage:<slug>): the container mounts
   no secret and no local disk. Locally a stage could read .env (Lupa, N6), so it never runs here.
3. Each job starts from the best certificate of --start-ledger (or the public #1), with its own seed. The container
   keeps the stage's output only if the verifier scores it strictly better than its start.
4. Back here, the best output is certified with the pinned AND the independent exact verifier. Ledger: a decision
   (move "model_stage"), a result, and a certificate only for a confirmed gain. Nothing is ever submitted.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np

from .ledger import Ledger, cert_record
from .policy import call_llm, load_env
from .problem import load as load_problem
from .safety import safe_err

SYSTEM = ("You write one numerical optimization stage in Python (numpy and scipy available, no network, no files). "
          "Answer with one JSON object {\"code\": \"<python source>\"} only.")


def prompt(P, noise, history):
    src = (Path(P.REPO) / "problems" / P.SLUG / "verifier_pinned.py").read_text()
    t = (f"Problem: {P.SLUG}. Direction: {P._P.DIRECTION}. A solution is the JSON {{\"{P._KEY}\": array of shape "
         f"{list(P._SHAPE)}}}; you work on it flattened as a float64 vector of length {P.SEED_CELLS}.\n"
         f"The exact verifier (the only definition of the problem):\n```python\n{src}\n```\n"
         "Write `def refine(v, score, budget_s, rng):` returning a vector of that length. score(x) is the verifier's "
         "value turned into a minimization (lower is better; +inf when the verifier rejects x). The verifier allows "
         "no tolerance, so keep validity strict. Stop before budget_s seconds and return the best valid vector found. "
         "Only numpy/scipy/stdlib, no files, no network.\n")
    t += ("v is None: no solution is given. Build a valid solution from scratch, then improve it as far as you can."
          if noise else "v is a valid, already strong solution: improve it, using the structure of the verifier "
          "(constraints, active contacts). Return v itself if nothing better is found.")
    for h in history:
        t += (f"\n\nYour version {h['version']}:\n```python\n{h['code']}\n```\nWhat the verifier measured, per job "
              f"(gain = start score minus stage score; positive is better; errors shown): {json.dumps(h['feedback'])}\n"
              "Write the next version: keep what gained, fix what failed.")
    return t


def best_start(P, ledger_path):
    if ledger_path:
        recs = [json.loads(x) for x in Path(ledger_path).read_text().splitlines() if x.strip()]
        certs = [r for r in recs if r["kind"] == "certificate" and r.get("problem") == P.NAME]
        if certs:
            c = min(certs, key=lambda r: r["score"])
            p = Path(c["path"])
            if p.suffix == ".json":                                      # a certified solution file
                return np.asarray(json.loads(p.read_text())[P._KEY], dtype=np.float64).ravel(), c["id"]
            return P.load_vec(p), c["id"]
    v, _ = P.public_best()
    return v, "public#1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--problem", required=True)
    ap.add_argument("--start", choices=["noise", "public", "ledger"], default="ledger"); ap.add_argument("--start-ledger")
    ap.add_argument("--versions", type=int, default=1, help="rewrite loop: the model sees its last version's gains")
    ap.add_argument("--provider", default="openai"); ap.add_argument("--model")
    ap.add_argument("--budget-s", type=float, default=600); ap.add_argument("--jobs", type=int, default=8)
    WORKSPACE = __import__("os").environ.get("AUTOLOOP_WORKSPACE", str((lambda q: q[2] if len(q) > 2 else q[-1])(__import__("pathlib").Path(__file__).resolve().parents)))  # folder that holds repo/, wt-compare/, shared/ and .env
    ap.add_argument("--max-usd", type=float, default=2.0); ap.add_argument("--env-file", default=WORKSPACE + "/.env")
    ap.add_argument("--price-in", type=float, default=0.0); ap.add_argument("--price-out", type=float, default=0.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--resume", action="store_true", help="continue a crashed run in the same ledger: best kept "
                    "certificate, last version's code and measured gains, next version number")
    ap.add_argument("--isolated", action="store_true", help="run stages in modal_stage_app.py (no problems/, no volume, "
                    "no network, no secret; outputs by value) instead of modal_app.py's run_job")
    a = ap.parse_args()
    P = load_problem(a.problem)
    run = Path(a.run); (run / "stages").mkdir(parents=True, exist_ok=True); (run / "certs").mkdir(exist_ok=True)
    L = Ledger(run / "ledger.jsonl")
    load_env(a.env_file)
    model = a.model or (os.environ.get("OPENAI_MODEL") if a.provider == "openai" else None)
    noise = a.start == "noise"
    from . import modal_stage_app as _iso
    iso = dict(_iso.ISOLATION, enforced=True, how="Modal block_network=True; image without problems/; no volume; no "
               "secret; stage subprocess env PATH only; hard timeout; outputs by value; in-container self-check per job") \
        if a.isolated else dict(enforced=False, how="modal_app.run_job: image holds problems/ (incl. the #1 certificate), "
                                "volume mounted, network open; 'from noise' rests on code inspection only")
    old = [json.loads(x) for x in (run / "ledger.jsonl").read_text().splitlines() if x.strip()] if a.resume else []
    if a.resume:
        L.append(dict(kind="note", text="resumed after a crash (the run had no run_end); same config, same isolation: "
                      + json.dumps(iso)[:200]))
    else:
      L.append(dict(kind="run_start", isolation=iso, policy=f"model_stage:{a.provider}:{model or 'default'}", executor="modal",
                  commit=os.environ.get("AUTOLOOP_COMMIT", "?"), cold_start=noise, pastdev_read=False,
                  label=a.label or ("stages written by a model from the verifier and the format alone; "
                                    + ("from noise: no solution given" if noise else f"start: {a.start}")), config=vars(a)))
    first, history0 = 1, []
    if a.resume and any(r["kind"] == "certificate" for r in old):
        c = [r for r in old if r["kind"] == "certificate"][-1]
        kept = next(j["path"] for r in old if r["kind"] == "result" for j in r["jobs"]          # the job output certified
                    if j.get("ok") and P.vec_id(P.load_vec(j["path"])) == c["id"])
        v0, parent, s0 = P.load_vec(kept), c["id"], float(c["score"])
        lastd = [r for r in old if r["kind"] == "decision"][-1]; lastr = [r for r in old if r["kind"] == "result"][-1]
        first = lastd["args"]["version"] + 1 if lastr["i"] == lastd["i"] else lastd["args"]["version"]
        if lastr["i"] == lastd["i"]:
            code0 = next(run.glob(f"stages/v{lastd['args']['version']}_*.py")).read_text(); ps = lastd["numbers"]["best"]
            fb0 = [dict(job=j["job"], error=j["error"]) if not j["ok"] else dict(
                job=j["job"], valid=bool(np.isfinite(j["summary"].get("stage_score", np.inf))), error=j["summary"].get("error"),
                gain=(ps - j["summary"]["stage_score"]) if ps is not None and np.isfinite(j["summary"].get("stage_score", np.inf)) else None,
                seconds=round(j["summary"].get("seconds", 0))) for j in lastr["jobs"]]
            history0 = [dict(version=lastd["args"]["version"], code=code0, feedback=fb0)]
    elif noise:
        v0, parent, s0 = None, None, float("inf")
    else:
        v0, parent = best_start(P, a.start_ledger if a.start == "ledger" else None); s0 = P.score(v0)
    import modal
    from .modal_exec import USD_PER_JOB_S
    fn = (modal.Function.from_name(_iso.APP_NAME, "run_stage") if a.isolated
          else modal.Function.from_name("revolutionaries-autoloop", "run_job"))
    vsrc = (Path(P.REPO) / "problems" / P.SLUG / "verifier_pinned.py").read_text()
    spent = sum(r.get("usd_est", 0) for r in old if r["kind"] == "result")
    llm_usd = sum((r.get("llm_usage") or {}).get("usd") or 0 for r in old if r["kind"] == "decision")
    history = history0
    for ver in range(first, a.versions + 1):
        est = a.jobs * (a.budget_s + 300) * USD_PER_JOB_S
        if spent + est > a.max_usd:
            L.append(dict(kind="note", text=f"version {ver} not run: Modal estimate would pass USD {a.max_usd}")); break
        pr = prompt(P, noise and v0 is None, history)
        out = None
        for attempt in (1, 2):
            try:
                out, usage = call_llm(a.provider, model, SYSTEM, pr, a.price_in, a.price_out); llm_usd += usage.get("usd") or 0
                break
            except Exception as e:  # noqa: BLE001
                L.append(dict(kind="note", text=f"version {ver}: model call failed (attempt {attempt}): " + safe_err(repr(e))))
        if out is None:
            break
        try:
            code = json.loads(out[out.find("{"): out.rfind("}") + 1])["code"]
        except Exception as e:  # noqa: BLE001
            L.append(dict(kind="note", text=f"version {ver}: unparseable model answer: " + safe_err(repr(e)))); continue
        sha = hashlib.sha256(code.encode()).hexdigest()
        (run / "stages" / f"v{ver}_{sha[:16]}.py").write_text(code)
        L.append(dict(kind="decision", i=ver - 1, move="model_stage", policy=f"model_stage:{a.provider}", args=dict(
            version=ver, parent=parent, code_sha256=sha, jobs=a.jobs, budget_s=a.budget_s, from_noise=v0 is None),
            numbers=dict(best=s0 if np.isfinite(s0) else None), llm_usage=usage,
            rationale=f"version {ver}: written from the verifier and the format" + (" and its last version's measured gains" if history else "")))
        start = None
        if v0 is not None:
            buf = io.BytesIO(); np.save(buf, v0); start = buf.getvalue()
        if a.isolated:
            calls = [fn.spawn(dict(job=f"v{ver}_{sha[:8]}_{k}", key=P._KEY, shape=list(P._SHAPE), direction=P._P.DIRECTION,
                                   seed=1000 * ver + k, budget=float(a.budget_s)), vsrc, code, start) for k in range(a.jobs)]
        else:
            calls = [fn.spawn(dict(kind=f"pack_stage:{P.SLUG}", key=f"v{ver}_{sha[:8]}_{k}", run=run.name, seed=1000 * ver + k,
                                   budget=float(a.budget_s), code=code), start) for k in range(a.jobs)]
        spent += est
        res = []
        for c in calls:
            try:
                res.append(c.get(timeout=a.budget_s * 2 + 900))
            except Exception as e:  # noqa: BLE001
                res.append(dict(ok=False, err_tail=safe_err(repr(e))))
        rows, fb = [], []
        for k, r in enumerate(res):
            sm = r.get("summary") or {}
            if not r.get("ok"):
                rows.append(dict(job=k, ok=False, error=safe_err(r.get("err_tail", "")))); fb.append(dict(job=k, error=rows[-1]["error"]))
                continue
            pth = run / "certs" / f"v{ver}_{sha[:8]}_{k}.npy"; pth.write_bytes(r["npy"])
            sc = P.score(P.load_vec(pth))
            rows.append(dict(job=k, ok=True, path=str(pth), score=sc, summary=sm, self_check=r.get("self_check")))
            g = (s0 - sm["stage_score"]) if np.isfinite(s0) and np.isfinite(sm.get("stage_score", np.inf)) else None
            fb.append(dict(job=k, valid=bool(np.isfinite(sm.get("stage_score", np.inf))), gain=g, error=sm.get("error"),
                           seconds=round(sm.get("seconds", 0))))
        good = [x for x in rows if x["ok"] and x["score"] < s0]
        summ = f"version {ver}: {len(good)}/{a.jobs} jobs improved on the start by the verifier"
        if good:
            b = min(good, key=lambda x: x["score"])
            r, _ = P.LocalExecutor(run).certify(b["path"], f"v{ver}_{sha[:8]}")
            if r["ok"] and float(r["exact"]) < s0:
                cid = P.vec_id(P.load_vec(b["path"]))
                L.append(cert_record(problem=P.NAME, cid=cid, cells=P.SEED_CELLS, score_=float(r["exact"]), parent=parent,
                                     stage="certified", params=dict(model_stage=sha, version=ver, checks=r["checks"]),
                                     cpu_s=a.jobs * a.budget_s, path=r["path"], exact=f"{r['exact'].numerator}/{r['exact'].denominator}",
                                     origin="model_stage"))
                summ += (f"; KEPT: exact {float(r['exact'])!r}" + (f" (gain {s0 - float(r['exact']):.3e} over the start)"
                                                                     if np.isfinite(s0) else " (first valid solution)"))
                v0, s0, parent = P.load_vec(b["path"]), float(r["exact"]), cid
            else:
                summ += "; NOT kept: " + (r.get("error") or "exact check shows no gain")
        else:
            summ += "; NOT kept"
        L.append(dict(kind="result", i=ver - 1, move="model_stage", summary=summ, jobs=rows, usd_est=round(est, 3)))
        print(summ, flush=True)
        history = [dict(version=ver, code=code, feedback=fb)]          # the next version sees this one and its gains
    L.append(dict(kind="run_end", reason=f"{a.versions} versions or the cap", best=s0 if np.isfinite(s0) else None,
                  best_exact=s0 if np.isfinite(s0) else None, usd=round(spent, 3), llm_usd=round(llm_usd, 4)))


if __name__ == "__main__":
    main()
