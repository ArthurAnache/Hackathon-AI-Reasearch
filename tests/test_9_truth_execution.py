"""Process 9: L1 certificates (verified truth, the only warm starts) and L2 execution (Local / Modal)."""
import json
import textwrap

import pytest

P1 = pytest.mark.process("9a. Certificates (L1 truth)")
P2 = pytest.mark.process("9b. Execution layer (L2)")


@pytest.fixture
def truth(workspace, monkeypatch):
    import importlib, sys
    monkeypatch.chdir(workspace)
    monkeypatch.syspath_prepend(str(workspace))
    names = ("prepare", "interfaces", "certificates", "execution", "importers", "problems")
    sub = lambda k: any(k.startswith(n + ".") for n in ("problems", "execution"))
    saved = {k: v for k, v in sys.modules.items() if k in names or sub(k)}
    for k in saved:
        del sys.modules[k]
    I = importlib.import_module("interfaces")
    C = importlib.import_module("certificates")
    E = importlib.import_module("execution")
    yield I, C, E, workspace
    for k in [k for k in sys.modules if k in names or sub(k)]:
        del sys.modules[k]
    sys.modules.update(saved)


class InProcess:
    """Executor that runs jobs in this process (fast, for tests)."""
    name = "inprocess"

    def submit(self, job):
        return self.map([job])[0]

    def map(self, jobs):
        from dataclasses import asdict
        import execution as E
        return [E.JobResult(**E.run_job(asdict(j))) for j in jobs]


# ---------------------------------------------------------------- certificates
@P1
def test_certificate_roundtrip_and_reverification(truth, record):
    I, C, _, root = truth
    p = I.get_problem("1a")
    store = C.CertificateStore(root / "certificates")
    cert = store.put(p, {"values": [1.0, 2.0, 1.0]}, 2.25, {"by": "test"})
    again = store.load(p, cert.id)
    record(id=cert.id, score=again.score, path=str(cert.path.relative_to(root)))
    assert again.score == 2.25 and again.solution == {"values": [1.0, 2.0, 1.0]}
    with pytest.raises(C.CertificateError):
        store.put(p, {"values": [1.0, 2.0, 1.0]}, 1.0)            # wrong claimed score is refused


@P1
@pytest.mark.parametrize("tamper", ["score", "solution", "problem"])
def test_tampered_certificate_is_rejected(truth, record, tamper):
    """Editing a stored certificate (score, solution or problem) makes it fail on load; best() skips it."""
    I, C, _, root = truth
    p = I.get_problem("1a")
    store = C.CertificateStore(root / "certificates")
    good = store.put(p, {"values": [1.0, 1.0]}, 2.0)
    bad = store.put(p, {"values": [1.0, 2.0, 1.0]}, 2.25)
    data = json.loads(bad.path.read_text())
    if tamper == "score":
        data["score"] = 1.0                       # would be the new best if trusted
    elif tamper == "solution":
        data["solution"]["values"][0] = 5.0
    else:
        data["problem"] = "1d"
    bad.path.write_text(json.dumps(data))
    with pytest.raises(C.CertificateError) as e:
        store.load(p, bad.id)
    best = store.best(p)
    record(tamper=tamper, error=str(e.value)[:120], best_after=best.id)
    assert best.id == good.id


@P1
def test_only_certificates_are_warm_starts(truth, record):
    """run_experiment refuses a free-form INIT path and accepts a (re-verified) certificate id."""
    I, C, _, root = truth
    p = I.get_problem("1d")
    store = I.JsonlStore(root / "e.jsonl")
    certs = C.CertificateStore(root / "certificates")
    opt = I.FileOptimizer(root / "optimizers" / "1d_tuned.py")
    first = I.run_experiment(p, opt, I.ExperimentConfig("cold"), store, budget_s=0.5, certs=certs)
    path = certs.load(p, first["certificate_path"]).path
    free = I.run_experiment(p, opt, I.ExperimentConfig("free path", {"INIT": str(path)}), store, budget_s=0.5, certs=certs)
    warm = I.run_experiment(p, opt, I.ExperimentConfig("cert", {"INIT": first["certificate_path"]}), store,
                            budget_s=0.5, certs=certs)
    record(cold=first["score"], free_path_status=free["status"], free_path_error=free["error"],
           warm_status=warm["status"], warm_score=warm["score"], warm_start=warm["warm_start"])
    assert free["status"] == "invalid" and "warm start rejected" in free["error"]
    assert warm["valid"] and warm["warm_start"] == first["certificate_path"] and warm["score"] <= first["score"]


# ---------------------------------------------------------------- execution
@P2
def test_job_is_self_contained_and_hash_checked(truth, record):
    I, C, E, root = truth
    p = I.get_problem("1a")
    job = E.make_job(p, root / "optimizers" / "1a_baseline.py", {"N_CELLS": 9}, 7, 0.2, job_id="j1", arm="test")
    ok = E.run_job(E.to_dict(job))
    job.optimizer_source += "\n# tampered\n"
    bad = E.run_job(E.to_dict(job))
    record(fields=sorted(E.to_dict(job)), ok_status=ok["status"], n_values=len(ok["candidate"]["values"]),
           tampered=bad["error"][:80], meta=ok["meta"])
    assert ok["status"] == "ok" and len(ok["candidate"]["values"]) == 9 and ok["meta"] == {"arm": "test"}
    assert bad["status"] == "crash" and "optimizer_hash" in bad["error"]


@P2
def test_local_executor_parallel(truth, record):
    """LocalExecutor runs a batch in separate processes; results come back in job order."""
    import time
    I, C, E, root = truth
    p = I.get_problem("1a")
    jobs = [E.make_job(p, root / "optimizers" / "1a_baseline.py", {}, s, 1.0, job_id=f"s{s}") for s in range(3)]
    t0 = time.monotonic()
    res = E.LocalExecutor(workers=3).map(jobs)
    wall = time.monotonic() - t0
    record(wall_s=round(wall, 2), workers=sorted({r.worker_id for r in res}), ids=[r.job_id for r in res])
    assert [r.job_id for r in res] == ["s0", "s1", "s2"] and all(r.status == "ok" for r in res)
    assert wall < 3 * 1.0 + 2                 # ran concurrently, not one after another


@P2
def test_warm_start_ships_verified_certificate(truth, record):
    I, C, E, root = truth
    p = I.get_problem("1d")
    certs = C.CertificateStore(root / "certificates")
    cert = certs.put(p, {"values": [1.0, 0.5, 0.5, 1.0]}, p.score({"values": [1.0, 0.5, 0.5, 1.0]}))
    job = E.make_job(p, root / "optimizers" / "1d_tuned.py", {"N_CELLS": 8, "INIT": "ignored"}, 0, 0.3, cert.id, certs)
    res = E.run_job(E.to_dict(job))
    record(cert=cert.id, parent=cert.score, result=p.score(res["candidate"]), config_sent=job.config)
    assert "INIT" not in job.config and job.warm_start_solution == cert.solution
    assert p.score(res["candidate"]) <= cert.score + 1e-12          # never worse than the parent


@P2
def test_modal_executor_contract_with_stand_in(truth, record):
    """ModalExecutor drives a deployed function's .map(); remote exceptions become crash results."""
    I, C, E, root = truth

    class FakeModalFunction:
        def map(self, payloads, return_exceptions=False):
            for i, job in enumerate(payloads):
                yield RuntimeError("container lost") if i == 1 else E.run_job(job)

    p = I.get_problem("1a")
    jobs = [E.make_job(p, root / "optimizers" / "1a_baseline.py", {}, s, 0.2, job_id=f"m{s}") for s in range(3)]
    res = E.ModalExecutor(function=FakeModalFunction()).map(jobs)
    record(statuses=[r.status for r in res], error=res[1].error)
    assert [r.status for r in res] == ["ok", "crash", "ok"] and "container lost" in res[1].error


@P2
def test_importer_reads_pilot_results_back(truth, record):
    """WS3 pilot results (JSONL + *_best.json) enter the one store; bad certificates are rejected."""
    I, C, _, root = truth
    import importers
    d = root / "results"
    d.mkdir()
    run = "ws2_1a_test_20261114T000000Z"
    rows = [{"experiment_id": "exp_0000", "problem": "1a", "variant": "tuned", "seed": 42, "budget_s": 10.0,
             "score": 2.0, "valid": True, "status": "keep", "config": {}, "algorithm_version": "abc", "git_commit": "def"},
            {"experiment_id": "exp_0000", "problem": "1a", "variant": "baseline", "seed": 42, "budget_s": 10.0,
             "score": 2.25, "valid": True, "status": "keep", "config": {}}]
    (d / f"{run}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (d / f"{run}_tuned_best.json").write_text(json.dumps({"problem": "1a", "experiment_id": "exp_0000", "score": 2.0,
                                                          "solution": {"values": [1.0, 1.0]}}))
    (d / f"{run}_baseline_best.json").write_text(json.dumps({"problem": "1a", "experiment_id": "exp_0000",
                                                             "score": 1.0, "solution": {"values": [1.0, 2.0, 1.0]}}))
    store, certs = I.JsonlStore(root / "e.jsonl"), C.CertificateStore(root / "certificates")
    rep = importers.import_pilot(d, store, certs)
    again = importers.import_pilot(d, store, certs)
    recs = store.read()
    record(report=rep, reimport_skipped=again["skipped_existing"], arms=[r["arm"] for r in recs])
    assert rep["records"] == 2 and len(rep["certificates"]) == 1 and len(rep["rejected"]) == 1
    assert again["records"] == 0 and {r["arm"] for r in recs} == {"pilot:tuned", "pilot:baseline"}
    assert next(r for r in recs if r["arm"] == "pilot:tuned")["certificate_path"] == rep["certificates"][0]
