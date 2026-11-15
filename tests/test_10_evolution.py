"""Process 10: the evolutionary research loop (L3 genes/selection, L4 bandit/islands/migration).

Uses a stub optimizer for problem 1a whose behaviour is known exactly:
  HELP >= 2  -> a better candidate (score 2.0 instead of 2.25)   = a gene that really helps
  JUNK       -> no effect                                         = a gene that should not survive
Seeds add a tiny perturbation, so the multi-seed statistics are not degenerate.
"""
import math
import textwrap

import numpy as np
import pytest

P = pytest.mark.process("10. Evolution (genes, selection, knock-out, bandit, islands)")

STUB = textwrap.dedent('''
    from prepare import run
    PROBLEM = "1a"
    CONFIG = {"HELP": 1, "JUNK": 1.0, "INIT": None}
    def search(budget_s, seed, config=CONFIG):
        vals = [1.0, 1.0] if config["HELP"] >= 2 else [1.0, 2.0, 1.0]
        vals[0] += 1e-4 * (seed % 7)
        return {"values": vals}
    if __name__ == "__main__":
        run(__file__)
''')


@pytest.fixture
def evo(workspace, monkeypatch):
    import importlib, sys
    for f in (workspace / "optimizers").glob("1a_*.py"):
        f.unlink()                                       # only the stub: no operator genes from real files
    (workspace / "optimizers" / "1a_stub.py").write_text(STUB)
    monkeypatch.chdir(workspace)
    monkeypatch.syspath_prepend(str(workspace))
    pref = ("problems", "execution", "evolution", "research", "integrations")
    names = ("prepare", "interfaces", "certificates", "execution", "evolution", "research", "problems", "integrations")
    monkeypatch.setenv("INTEGRATIONS_ENV_FILE", str(workspace / "no.env"))       # never read the real .env
    monkeypatch.setenv("PASTDEV_LOG_PATH", str(workspace / "pastdev_log.jsonl"))
    for v in ("EXA_API_KEYS", "EXA_API_KEY", "PAST_API_KEYS", "PAST_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    sub = lambda k: any(k.startswith(n + ".") for n in pref)
    saved = {k: v for k, v in sys.modules.items() if k in names or sub(k)}
    for k in saved:
        del sys.modules[k]
    I = importlib.import_module("interfaces")
    EV = importlib.import_module("evolution")
    yield I, EV, workspace
    for k in [k for k in sys.modules if k in names or sub(k)]:
        del sys.modules[k]
    sys.modules.update(saved)


class InProcess:
    name = "inprocess"

    def map(self, jobs):
        from dataclasses import asdict
        import execution as E
        return [E.JobResult(**E.run_job(asdict(j))) for j in jobs]

    def submit(self, job):
        return self.map([job])[0]


# ---------------------------------------------------------------- units
@P
def test_real_margin_needs_more_than_noise(evo, record):
    I, EV, _ = evo
    p = I.get_problem("1a")                                   # minimise
    clear = EV.real_margin(p, [2.0, 2.001, 1.999], [2.25])
    noisy = EV.real_margin(p, [2.2, 2.4, 2.0], [2.25])
    record(clear=clear, noisy=noisy)
    assert clear[0] and not noisy[0]


@P
def test_genes_express_and_conflicts(evo, record):
    I, EV, _ = evo
    G = EV.Gene
    lr, lr2 = G.make("config", "LR=0.1", LR=0.1), G.make("config", "LR=0.2", LR=0.2)
    up = G.make("formulation", "upscale_x2", knob="N_CELLS", factor=2)
    op = G.make("operator", "other", optimizer="optimizers/other.py")
    opt, cfg = EV.express([up, lr, op], "optimizers/base.py", {"N_CELLS": 150, "LR": 0.01})
    record(optimizer=opt, config=cfg, compatible_pair=EV.compatible([lr, lr2]))
    assert opt == "optimizers/other.py" and cfg == {"N_CELLS": 300, "LR": 0.1}
    assert not EV.compatible([lr, lr2]) and EV.compatible([lr, up, op])


@P
def test_thompson_crossover_prefers_genes_that_helped(evo, record):
    """Over many children, a gene with a strong record is inherited far more often than a weak one."""
    I, EV, _ = evo
    pool = EV.GenePool()
    good, bad = EV.Gene.make("config", "A=1", A=1), EV.Gene.make("config", "B=1", B=1)
    pool.add(good); pool.add(bad)
    for _ in range(8):
        pool.credit(good.id, True); pool.credit(bad.id, False)
    from evolution.operators import crossover
    rng = np.random.default_rng(0)
    kids = [crossover([good], [bad], pool, rng) for _ in range(400)]
    rate = lambda g: sum(g in k for k in kids) / len(kids)
    record(inherit_good=rate(good), inherit_bad=rate(bad), bad_state=pool.state[bad.id])
    assert rate(good) > 0.75 and rate(bad) < 0.25
    assert pool.state[bad.id] == "graveyard"                  # kept losing -> buried, never proposed again


@P
def test_bandit_keeps_control_and_follows_success(evo, record):
    I, EV, _ = evo
    b = EV.MetaBandit(["hill_sampling", "gene_evolution", "formulation"], seed=0)
    for _ in range(20):
        b.credit("gene_evolution", True); b.credit("formulation", False); b.credit("hill_sampling", False)
    alloc = [b.allocate(6) for _ in range(50)]
    mean = {a: sum(x[a] for x in alloc) / 50 for a in alloc[0]}
    record(mean_allocation=mean)
    assert all(x["hill_sampling"] >= 1 for x in alloc) and mean["gene_evolution"] > 3
    assert all(sum(x.values()) == 6 for x in alloc)


# ---------------------------------------------------------------- end to end
@P
def test_survival_of_the_fittest_end_to_end(evo, record):
    """Cold start -> each gene tested alone -> the helpful gene is kept, passes the knock-out gate,
    is validated, migrates to the other island; the useless gene does not survive; every step is logged."""
    I, EV, root = evo
    from certificates import CertificateStore
    store = I.JsonlStore(root / "e.jsonl")
    orch = EV.Orchestrator("1a", "optimizers/1a_stub.py", InProcess(), store, CertificateStore(root / "certificates"),
                           islands=2, seeds=3, budget_s=0.2, candidates=4, seed=1)
    reports = orch.run(4)
    recs = store.read()
    states = [dict(isl.pool.state) for isl in orch.islands]
    record(rounds=[[r[1:4] for r in rep.results] for rep in reports], gene_states=states,
           best=[(i.best_cert, i.best_score) for i in orch.islands], bandit=orch.bandit.summary(),
           migrated=[rep.migrated for rep in reports])
    assert all(set(r) == set(I.RECORD_FIELDS) for r in recs)
    assert all(len(r["seeds"]) == 3 for r in recs)
    assert any(s.get("config:HELP=2") == "validated" for s in states)               # validated by knock-out
    assert all(s.get("config:HELP=2") == "validated" for s in states)                # ... and migrated
    assert not any(s.get("config:JUNK=2.0") == "validated" for s in states)          # no effect -> never validated
    assert all(i.best_score == pytest.approx(2.0, abs=1e-3) for i in orch.islands)   # best certificate shared
    ko = [r for r in recs if r["arm"] == "knockout"]
    assert ko and any(r["status"] == "keep" and "side=with" in r["rationale"] for r in ko)
    assert {r["island"] for r in recs} == {"island0", "island1"}
    paired = [r for r in recs if r["arm"] != "fresh_seeds"]
    assert len({tuple(r["seeds"]) for r in paired}) == 2                             # own seeds per island
    fresh = [tuple(r["seeds"]) for r in recs if r["arm"] == "fresh_seeds"]
    assert fresh and len(set(fresh)) == len(fresh) and not set(sum(fresh, ())) & set(sum(map(list, {tuple(r["seeds"]) for r in paired}), []))


@P
def test_migration_moves_only_validated_genes_and_best_certificate(evo, record):
    """A gene validated on island0 (knock-out 'with' record, status keep) is imported by island1;
    a gene whose knock-out failed, or a 'without' record, never migrates. The best certificate is shared."""
    I, EV, root = evo
    from certificates import CertificateStore
    from research import Space
    p = I.get_problem("1a")
    opt = I.FileOptimizer(root / "optimizers" / "1a_stub.py")
    space = Space.from_optimizer(opt)
    isl = [EV.Island(f"island{i}", p, "optimizers/1a_stub.py", {"HELP": 1, "JUNK": 1.0}, space, root, i) for i in range(2)]
    good = EV.Gene.make("config", "HELP=2", HELP=2).to_dict()
    junk = EV.Gene.make("config", "JUNK=9.0", JUNK=9.0).to_dict()
    recs = [I.new_record(experiment_id="a", problem="1a", arm="knockout", island="island0", status="keep", genes=[good],
                         rationale="knockout=config:HELP=2; side=with; parent=x"),
            I.new_record(experiment_id="b", problem="1a", arm="knockout", island="island0", status="discard", genes=[junk],
                         rationale="knockout=config:JUNK=9.0; side=with; parent=x"),
            I.new_record(experiment_id="c", problem="1a", arm="knockout", island="island0", status="keep", genes=[junk],
                         rationale="knockout=config:JUNK=9.0; side=without; parent=x")]
    certs = CertificateStore(root / "certificates")
    cert = certs.put(p, {"values": [1.0, 1.0]}, 2.0)
    moved = EV.migrate(isl, recs, certs, p)
    record(moved=moved, island1_states={k: v for k, v in isl[1].pool.state.items()},
           island1_note=isl[1].pool.notes["config:HELP=2"])
    assert moved["genes"] == [("island0", "island1", "config:HELP=2")]
    assert isl[1].pool.state["config:HELP=2"] == "validated" and "island0" in isl[1].pool.notes["config:HELP=2"]
    assert "config:JUNK=9.0" not in isl[1].pool.genes
    assert all(i.best_cert == cert.id for i in isl)


# ---------------------------------------------------------------- research tools (Exa, past.dev)
class FakeExa:
    """Stands in for integrations.evidence.ExaEvidence: one literature hit, on the first round only."""
    def __init__(self, I):
        self.I, self.calls = I, 0

    def gather(self, problem, history):
        self.calls += 1
        if self.calls > 1:
            return []
        return [self.I.Evidence(source="exa", content="Paper: a better relaxation for autocorrelation bounds",
                                citation="https://arxiv.org/abs/0000.00000", author="exa-scout")]


@P
def test_tools_share_verdicts_between_agents(evo, record):
    """Agent A validates HELP=2 and buries useless genes; agent B (another process, same past.dev memory)
    starts with HELP=2 favoured and A's graveyard already buried. Exa literature reaches the LLM context,
    and every record says which tools were on."""
    I, EV, root = evo
    from certificates import CertificateStore
    from integrations.pastdev_client import LocalMemory
    memory = LocalMemory(root / "memory.jsonl")
    tools_a = EV.ResearchTools(exa=FakeExa(I), memory=memory, agent_id="agent-a")
    a = EV.Orchestrator("1a", "optimizers/1a_stub.py", InProcess(), I.JsonlStore(root / "a.jsonl"),
                        CertificateStore(root / "certs_a"), islands=1, seeds=3, budget_s=0.2, candidates=4, seed=1,
                        tools=tools_a)
    reps = a.run(4)
    shared = [r["metadata"] for r in map(__import__("json").loads, (root / "memory.jsonl").read_text().splitlines())]
    verdicts = {m["gene"]["id"]: m["verdict"] for m in shared if m.get("kind") == "gene_verdict"}

    tools_b = EV.ResearchTools(memory=memory, agent_id="agent-b")
    b = EV.Orchestrator("1a", "optimizers/1a_stub.py", InProcess(), I.JsonlStore(root / "b.jsonl"),
                        CertificateStore(root / "certs_b"), islands=1, seeds=3, budget_s=0.2, candidates=4, seed=2,
                        tools=tools_b)
    got = tools_b.before_round(b)
    pool_b = b.islands[0].pool
    recs = I.JsonlStore(root / "a.jsonl").read()
    record(verdicts_shared=verdicts, b_imported=got, b_help_prior=pool_b.ab.get("config:HELP=2"),
           b_graveyard=[g.id for g, _ in pool_b.graveyard()], literature=[e.citation for e in a.islands[0].evidence],
           agent_tag=recs[0]["agent"], tools_per_round=[r.tools for r in reps])
    assert verdicts.get("config:HELP=2") == "validated"
    assert pool_b.ab["config:HELP=2"] == [3.0, 1.0] and "agent-a" in pool_b.notes["config:HELP=2"]
    buried_by_a = {g for g, v in verdicts.items() if v == "graveyard"}
    assert buried_by_a <= {g.id for g, _ in pool_b.graveyard()}
    assert a.islands[0].evidence and a.islands[0].evidence[0].citation.startswith("https://arxiv.org")
    assert recs[0]["agent"] == "evolution@agent-a[exa,pastdev]" and set(recs[0]) == set(I.RECORD_FIELDS)


@P
def test_tool_hints_become_genes_and_are_cited(evo, record):
    """A config another agent kept (past.dev) becomes a gene tested on its own; its record cites the source."""
    I, EV, root = evo
    from certificates import CertificateStore
    from integrations.pastdev_client import LocalMemory
    memory = LocalMemory(root / "memory.jsonl")
    memory.write("Experiment exp_0003 on problem 1a by laptop-b: keep", author="laptop-b", kind="experiment",
                 metadata={"problem": "1a", "kind": "experiment", "status": "keep", "config": {"HELP": 3, "INIT": "x"}})
    tools = EV.ResearchTools(memory=memory, agent_id="evo")
    orch = EV.Orchestrator("1a", "optimizers/1a_stub.py", InProcess(), I.JsonlStore(root / "e.jsonl"),
                           CertificateStore(root / "certs"), islands=1, seeds=3, budget_s=0.2, candidates=6, seed=1,
                           tools=tools)
    orch.run(2)
    recs = I.JsonlStore(root / "e.jsonl").read()
    hinted = [r for r in recs if any(g["id"] == "config:HELP=3" for g in r["genes"])]
    record(origin=orch.islands[0].pool.genes["config:HELP=3"].origin, evidence=hinted[0]["evidence"] if hinted else None)
    assert orch.islands[0].pool.genes["config:HELP=3"].origin == "past.dev:laptop-b"
    assert hinted and hinted[0]["evidence"][0]["source"] == "past.dev" and hinted[0]["evidence"][0]["author"] == "laptop-b"


LOTTERY = textwrap.dedent('''
    from prepare import run
    PROBLEM = "1a"
    CONFIG = {"INIT": None}
    def search(budget_s, seed, config=CONFIG):
        # a warm start can only polish (tiny gain); a cold start lands in a random basin, sometimes a much better one
        if config.get("INIT"):
            import json
            v = json.load(open(config["INIT"]))["solution"]["values"]
            return {"values": v}
        return {"values": [1.0, 1.0] if seed % 5 == 0 else [1.0, 2.0, 1.0]}
    if __name__ == "__main__":
        run(__file__)
''')


@P
def test_fresh_seeds_finds_better_basin_and_is_judged_on_best(evo, record):
    """WS3's lesson: new bests come from fresh seeds (best-of-N), not from polishing the best chain.
    On a landscape where refinement is stuck, only fresh_seeds improves; it is credited on its best seed
    even though its mean is not significantly better."""
    I, EV, root = evo
    from certificates import CertificateStore
    (root / "optimizers" / "1a_lottery.py").write_text(LOTTERY)
    (root / "optimizers" / "1a_stub.py").unlink()
    store = I.JsonlStore(root / "e.jsonl")
    orch = EV.Orchestrator("1a", "optimizers/1a_lottery.py", InProcess(), store, CertificateStore(root / "certs"),
                           islands=1, seeds=3, budget_s=0.2, candidates=3, seed=1)
    orch.run(3)
    recs = store.read()
    fresh = [r for r in recs if r["arm"] == "fresh_seeds"]
    winners = [r for r in fresh if r["status"] == "keep"]
    record(fresh=[(r["seeds"], r["scores"], r["status"]) for r in fresh], best=orch.islands[0].best_score,
           bandit=orch.bandit.summary())
    assert winners and orch.islands[0].best_score == 2.0
    p = I.get_problem("1a")
    assert not EV.real_margin(p, winners[0]["scores"], [winners[0]["best_before"]])[0]   # mean test would miss it
    assert not any(r["status"] == "keep" for r in recs if r["arm"] == "hill_sampling" and r["best_before"] is not None)
