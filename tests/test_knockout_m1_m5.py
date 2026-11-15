"""Tests for Knockout Evolution M1-M5: decisions, genes, gate, filters, strategist patch, promotion gate, and one
recorded toy run of both arms with a fake LLM (ledger schema, report, replay determinism).

    .venv/bin/python -m pytest tests/test_knockout_m1_m5.py -q
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from knockout import decisions as D
from knockout import filters as F
from knockout.budget import BudgetGovernor
from knockout.config import Strategy
from knockout.context import Evaluation, RunContext
from knockout.genes import GeneStore, render
from knockout.ledger import Ledger, validate_run
from knockout.problems_api import ProblemHandle

ROOT = Path(__file__).resolve().parent.parent
SEED_SRC = (ROOT / "knockout" / "seeds" / "toy_circles.py").read_text()
GATE = {"p_validate": 0.90, "p_refute": 0.10, "min_pairs": 3, "max_pairs": 8}


# ------------------------------------------------------------------ gate (5.2)
def test_gate_three_straight_wins_validate():
    assert D.gate_verdict(2, 0, 2, **GATE)["verdict"] is None            # no verdict before the third pair
    out = D.gate_verdict(3, 0, 3, **GATE)
    assert out["verdict"] == "validated" and out["p"] == pytest.approx(1 - 0.5**4)


def test_gate_refutes_retires_and_waits():
    assert D.gate_verdict(0, 3, 3, **GATE)["verdict"] == "refuted"
    assert D.gate_verdict(2, 1, 3, **GATE)["verdict"] is None            # p = 0.6875
    assert D.gate_verdict(4, 4, 8, **GATE)["verdict"] == "retired"       # inconclusive at the cap
    assert D.gate_verdict(0, 0, 8, **GATE)["verdict"] == "retired"       # 8 ties: dropped, but they hit the cap
    assert D.gate_verdict(5, 1, 6, **GATE)["verdict"] == "validated"     # p = 0.9375


def test_pair_outcome_needs_more_than_eps():
    assert D.pair_outcome(1.0 + 1e-3, 1.0, 1e-6, "maximize") == "win"
    assert D.pair_outcome(1.0 + 1e-9, 1.0, 1e-6, "maximize") == "tie"
    assert D.pair_outcome(1.0, 1.0 + 1e-3, 1e-6, "maximize") == "loss"
    assert D.pair_outcome(None, 1.0, 1e-6, "maximize") == "loss" and D.pair_outcome(1.0, None, 1e-6, "maximize") == "win"
    assert D.pair_outcome(0.9, 1.0, 1e-6, "minimize") == "win"


# ------------------------------------------------------------------ credit, phases, genome (5.3)
def test_credit_update_and_clade_weight():
    assert D.credit(2.0, 1.0, 1.0, 0.97) == pytest.approx([0.97 * 2 + 1, 0.97 * 1])
    assert D.credit(2.0, 1.0, 0.0, 0.97) == pytest.approx([0.97 * 2, 0.97 * 1 + 1])
    assert D.credit(0.0, 0.0, 1.0, 0.97, weight=0.25) == pytest.approx([0.25, 0.0])   # ancestor at depth 2


def test_gene_posterior_pools_phases_until_three_observations():
    stats = {"explore": [2.0, 0.0, 2], "climb": [0.0, 3.0, 3]}
    assert D.gene_posterior(stats, "explore", 3) == [3.0, 4.0]     # pooled: 1 + 2, 1 + 3
    assert D.gene_posterior(stats, "climb", 3) == [1.0, 4.0]       # its own phase
    assert D.phase(0.1, 0, 0.2, 30) == "explore" and D.phase(0.5, 3, 0.2, 30) == "climb"
    assert D.phase(0.5, 30, 0.2, 30) == "plateau"


def test_build_genome_respects_cap_links_and_incompatibilities():
    pool = [{"id": f"g{i}", "alpha": 1.0, "beta": 1.0, "linked": [], "incompatible": []} for i in range(12)]
    out = D.build_genome(pool, 8, seed=7)
    assert len(out["genome"]) == 8 and out == D.build_genome(pool, 8, seed=7)       # seeded, replayable
    strong = {"id": "a", "alpha": 500.0, "beta": 1.0, "linked": ["b"], "incompatible": ["c"]}
    weak = {"id": "b", "alpha": 1.0, "beta": 500.0, "linked": ["a"], "incompatible": []}
    rival = {"id": "c", "alpha": 100.0, "beta": 1.0, "linked": [], "incompatible": ["a"]}
    genome = D.build_genome([strong, weak, rival], 8, seed=1)["genome"]
    assert genome == ["a", "b"]                                    # b rides with a, c is never co-selected with a


def test_linkage_rule():
    assert D.linkage(6, 0.9, 0.5, 0.6, 6, 0.15) == "linked"
    assert D.linkage(6, 0.3, 0.5, 0.6, 6, 0.15) == "incompatible"
    assert D.linkage(5, 0.9, 0.5, 0.6, 6, 0.15) is None and D.linkage(6, 0.7, 0.5, 0.6, 6, 0.15) is None


def test_gene_store_dedupes_and_renders():
    store = GeneStore()
    a = store.add("I1", "structure", "Warm start from the incumbent and never return a worse candidate.", "seed")
    assert a is not None and a.status == "candidate"
    assert store.add("I1", "structure", "Warm start from the incumbent, never return a worse candidate!", "diff") is None
    assert store.add("I2", "structure", a.text, "migration") is not None      # status is local to an island
    long = store.add("I1", "numeric", " ".join(f"w{i}" for i in range(90)), "seed")
    assert len(long.text.split()) == 60
    a.status = "refuted"
    text = render([long], store.negatives("I1", 5))
    assert "Apply these tested insights" in text and "Already refuted, do not retry" in text


# ------------------------------------------------------------------ fidelity, bandits
def test_promote_top_third_and_bandit_discount():
    scores = {f"e{i}": float(i) for i in range(9)}
    assert D.promote(scores, 1 / 3, "maximize") == ["e8", "e7", "e6"]
    assert D.promote(scores, 1 / 3, "minimize") == ["e0", "e1", "e2"]
    assert D.promote({"only": 1.0}, 1 / 3, "maximize") == ["only"]
    arms = D.bandit_update({"a": [1.0, 1.0], "b": [1.0, 1.0]}, "a", 2.0, 0.0, 0.95)
    assert arms == {"a": [3.0, 1.0], "b": [1.0, 1.0]}                  # nothing decays below Beta(1, 1)
    arms = D.bandit_update({"a": [3.0, 1.0], "b": [1.0, 5.0]}, "b", 0.0, 1.0, 0.95)
    assert arms == {"a": [pytest.approx(2.85), 1.0], "b": [1.0, pytest.approx(5.75)]}
    pick = D.thompson_pick({"a": [50.0, 1.0], "b": [1.0, 50.0]}, seed=3)
    assert pick["choice"] == "a" and pick == D.thompson_pick({"a": [50.0, 1.0], "b": [1.0, 50.0]}, seed=3)


# ------------------------------------------------------------------ anti-cycling filters (5.5)
def test_filters_duplicates_and_recycled_lines():
    consts = "".join(f"C{i} = {100 + i}\n" for i in range(40))      # enough tokens for a realistic Jaccard
    parent = "import numpy as np\n" + consts + "\ndef construct(params, seed, budget_s, init=None):\n" \
             "    grid = np.linspace(0, 1, 7)\n    radius = 0.07\n    return {'grid': grid.tolist(), 'r': radius}\n"
    same = parent.replace("    grid", "    # a comment only\n    grid")
    hashes, tokens = {F.norm_hash(parent)}, [F.token_set(parent)]
    assert F.check(same, parent, hashes, tokens, set(), 0.95, 0.5)["reason"] == "exact duplicate"
    near = parent.replace("0.07", "0.08")
    assert F.check(near, parent, hashes, tokens, set(), 0.95, 0.5)["reason"] == "near duplicate"
    # the lineage once deleted two lines; a child that brings both back (2 of its 3 new lines) is cycling
    deleted = {"offset = seed % 3", "scale = 1.0 / (offset + 1)"}
    child = parent.replace("    radius = 0.07\n", "    offset = seed % 3\n    scale = 1.0 / (offset + 1)\n"
                                                    "    radius = 0.07 * scale\n")
    hit = F.check(child, parent, hashes, tokens, deleted, 0.95, 0.5)
    assert hit["reason"] == "recycled lines" and hit["ratio"] == pytest.approx(2 / 3, abs=1e-3)
    assert F.check(child, parent, hashes, tokens, set(), 0.95, 0.5) is None        # new lines elsewhere: fine
    added, removed = F.diff_lines(parent, child)
    assert "radius = 0.07" in removed and "offset = seed % 3" in added


# ------------------------------------------------------------------ strategist allow-list (5.9)
def test_strategist_patch_is_restricted_and_validated():
    from knockout.meta import validate_patch

    st = Strategy.load()
    clean, errors = validate_patch({"selection.parent_power": 2.0, "run.max_usd": 1000, "genes.g_max": 99,
                                    "fidelity.f0_s": 4.0, "sandbox.deny_network": False,
                                    "seed_genes": [{"type": "structure", "text": "Try a hexagonal layout first."}]},
                                   st, ["I1"])
    assert set(clean) == {"selection.parent_power", "fidelity.f0_s", "seed_genes"}
    assert len(errors) == 3 and any("run.max_usd" in e for e in errors) and any("sandbox" in e for e in errors)
    with pytest.raises(Exception):
        st.apply_patch({"run.max_usd": 1000})
    clean, errors = validate_patch({"fidelity.f0_s": 8.0, "fidelity.f1_s": 5.0}, st, ["I1"])      # f0 > f1
    assert not any(k.startswith("fidelity.f") for k in clean) and errors


# ------------------------------------------------------------------ promotion gate (5.11)
def _fake_problem(independent_score):
    return ProblemHandle(
        id="fake", direction="maximize", min_margin=0.1, verify=lambda c: True, score=lambda c: c["v"],
        invalid_exc=ValueError, guide_ref="x:y", best_known=lambda: 1.0,
        independent=lambda c: {"score": independent_score, "tolerance_abs": 1e-9, "method": "test"})


def _ctx(tmp_path, problem):
    st = Strategy.load(overrides={"ledger.save_solutions": "selected"})
    return RunContext("r", problem, st, Ledger(tmp_path / "r", "r"), BudgetGovernor(60, 0, st["budget_shares"]), None)


def _eval(score):
    return Evaluation("ok", True, score, {"v": score}, {}, None, None, 1, "none", 0.1, 0.0)


def test_promotion_gate_claims_only_with_the_independent_verifier(tmp_path):
    assert D.promotion(True, True, None, None) == "record_claim" and D.promotion(True, False, True, True) is None
    ctx = _ctx(tmp_path, _fake_problem(2.0))
    ev = _eval(2.0)
    ctx._promotion_gate(ev)
    assert ev.claim == "record_claim" and ev.valid
    rec = ctx.log_experiment(experiment_id="exp_000001", ev=ev, source="x = 1\n", program_id="p_x", chain="main",
                             agent="t", hypothesis="h", rationale="r", decision="keep", status="keep", seed=0,
                             budget_s=1.0, best_before=None, save_solution=False)
    assert rec["decision"] == "record_claim" and rec["solution_path"]           # certificate always stored
    assert [e["payload"]["outputs"]["decision"] for e in ctx.ledger.events()] == ["record_claim"]
    below = _eval(1.05)                                                          # above the record, below the margin
    ctx._promotion_gate(below)
    assert below.claim is None


def test_promotion_gate_flags_a_numerical_artifact(tmp_path):
    ctx = _ctx(tmp_path, _fake_problem(0.5))             # the independent verifier disagrees with the float score
    ev = _eval(2.0)
    ctx._promotion_gate(ev)
    assert ev.claim == "numerical_artifact" and not ev.valid and ev.status == "void"
    rec = ctx.log_experiment(experiment_id="exp_000001", ev=ev, source="x = 1\n", program_id="p_x", chain="main",
                             agent="t", hypothesis="h", rationale="r", decision="void", status="void", seed=0,
                             budget_s=1.0, best_before=None)
    assert rec["decision"] == "numerical_artifact" and not rec["beats_record"]


# ------------------------------------------------------------------ recorded toy run: both arms, fake LLM
class _FakeLLMClient:
    """Deterministic stand-in for the OpenAI client. Programs wrap the seed program and scale its radii: by 1.0
    when the prompt carries the good gene, 0.8 with the bad gene, 0.9 otherwise. So the good gene must win its
    knock-out pairs and the bad gene must lose them."""

    def __init__(self):
        self.n = 0
        self.roles: list[str] = []
        self.responses = self
        self.models = type("Models", (), {"retrieve": staticmethod(lambda name: {"id": name})})()

    def _program(self, prompt: str) -> dict:
        insights = prompt.split("# Insights", 1)[-1].split("# Parent program", 1)[0].split("Already refuted", 1)[0]
        q = 0.8 if "shrink all radii" in insights else 1.0 if "full radii" in insights else 0.9
        tags = ", ".join(f"'variant_{self.n}_token_{k}'" for k in range(30))
        source = SEED_SRC + (f"\n\n_Q_{self.n} = {q}\n_TAGS_{self.n} = [{tags}]\n_base_{self.n} = construct\n\n\n"
                             f"def construct(params, seed, budget_s, init=None):\n"
                             f"    out = _base_{self.n}(params, seed, budget_s, init)\n"
                             f"    out['circles'] = [[x, y, r * _Q_{self.n}] for x, y, r in out['circles']]\n"
                             f"    return out\n")
        return {"hypothesis": f"variant {self.n}: radii scaled by {q}", "rationale": "test", "source": source}

    def create(self, **kwargs):
        self.n += 1
        role = kwargs["text"]["format"]["name"]
        self.roles.append(role)
        if role in ("program", "program_edit"):
            answer = self._program(kwargs["input"])
        elif role == "program_reformulate":      # same object, identity adapter: enough to open an island
            words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
            answer = {**self._program(kwargs["input"]), "formulation": f"Formulation {self.n}: describe the "
                      f"layout through {words[self.n % 8]} {words[(self.n // 8) % 8]} coordinates number {self.n}.",
                      "adapter_source": "def to_canonical(obj):\n    return obj\n"}
        elif role == "seed_genes":
            answer = {"genes": [{"type": "structure", "text": "Keep the full radii that the solver returns, never scale them."},
                                {"type": "numeric", "text": "Always shrink all radii a lot to stay far from any overlap."}]}
        elif role == "review_diff":
            answer = {"genes": []}
        elif role == "strategy_patch":
            answer = {"rationale": "favour better parents", "patch": [
                {"key": "selection.parent_power", "value_json": "2.0"}, {"key": "run.max_usd", "value_json": "1000"}]}
        else:
            raise AssertionError(f"unexpected role {role}")
        usage = type("Usage", (), {"input_tokens": 1000, "output_tokens": 2000})()
        return type("Response", (), {"id": f"resp_{self.n}", "usage": usage, "output_text": json.dumps(answer)})()


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory):
    import knockout.meta
    from knockout.llm_editor import OpenAIEditor
    from knockout.run import run

    tmp = tmp_path_factory.mktemp("ko")
    client = _FakeLLMClient()
    real_scout, knockout.meta.scout = knockout.meta.scout, lambda arm, reason: 0     # no Exa call in tests
    overrides = {"run.wall_clock_s": 55, "run.max_usd": 100.0, "fidelity.f0_s": 1.0, "fidelity.f1_s": 2.0,
                 "fidelity.levels": 2, "tuner.calls": 1, "tuner.top_k": 1, "executor.workers": 6,
                 "control.edits_per_round": 2, "main.batch_size": 6, "gate.pairs_per_batch": 3,
                 "editor.kind": "openai", "strategist.stagnation_fraction": 0.25, "selection.eps_rel": 1e-6,
                 "islands.max_islands": 3, "islands.migrate_every": 6, "islands.min_evals_before_replace": 6}

    class Editor(OpenAIEditor):          # the control arm's editor, on the fake client
        def __init__(self, ctx):
            super().__init__(ctx, client=client)

    import knockout.run as run_module

    real_make = run_module.make_editor
    run_module.make_editor = lambda ctx: Editor(ctx)
    try:
        summary = run("toy", runs_root=tmp, run_id="rec", overrides=overrides, on_round=lambda *_: None)
    finally:
        run_module.make_editor = real_make
        knockout.meta.scout = real_scout
    return tmp / "rec", summary, client


def test_recorded_run_uses_both_arms_and_gates_genes(recorded_run):
    run_dir, summary, client = recorded_run
    assert summary["arms"]["main"]["steps"] >= 2 and summary["arms"]["control"]["steps"] >= 1
    assert summary["validation"]["experiments.jsonl"] > 10 and not summary["beats_record"]
    assert validate_run(run_dir) == summary["validation"]
    state = json.loads((run_dir / "state.json").read_text())
    genes = {g["text"][:9]: g for g in state["genes"]}
    assert genes["Keep the "]["status"] == "validated"            # three straight wins
    assert genes["Always sh"]["status"] == "refuted"              # three straight losses: a negative gene
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    types = {e["type"] for e in events}
    assert {"gene_born", "gate_pair", "gate_verdict", "genome_built", "operator_pick", "island_pick",
            "credit_update", "fidelity_promotion", "llm_call", "control_round"} <= types
    verdicts = [e["payload"] for e in events if e["type"] == "gate_verdict" and e["payload"]["outputs"]["verdict"]]
    assert verdicts[0]["inputs"]["wins"] == 3 and verdicts[0]["outputs"]["verdict"] == "validated"
    # a validated gene reaches genomes; a candidate gene never does outside its gate pairs
    rows = [json.loads(line) for line in (run_dir / "experiments.jsonl").read_text().splitlines()]
    validated = next(g["id"] for g in state["genes"] if g["text"].startswith("Keep the"))
    assert any(validated in r["config"].get("genome", []) for r in rows if r["config"].get("arm") == "main")
    refuted = next(g["id"] for g in state["genes"] if g["text"].startswith("Always sh"))
    assert not any(refuted in r["config"].get("genome", []) for r in rows)
    assert {"seed_genes", "program", "program_edit"} <= set(client.roles)
    # formulation islands: opened by `reformulate`, never more than the cap, genes migrate as candidates only
    assert "island_opened" in types and 2 <= len(state["islands"]) <= 3
    assert all(g["status"] != "validated" or g["origin"] != "migration" or g["evidence"] for g in state["genes"])
    # the control arm stays at its fixed share: it ran, and the main arm got most of the spend
    assert 0 < summary["arms"]["control"]["spend"] < summary["arms"]["main"]["spend"]


def test_report_is_written_and_claims_nothing(recorded_run):
    run_dir, _, _ = recorded_run
    report = (run_dir / "report.md").read_text()
    for heading in ("## Summary", "## Control arm versus main arm", "## Genealogy", "## Gene table",
                    "## Strategy patches"):
        assert heading in report
    assert "does not beat the record" in report and "record_claim**" not in report


def test_replay_is_deterministic_and_detects_tampering(recorded_run, tmp_path):
    import shutil

    from knockout.replay import ReplayError, replay

    run_dir, _, _ = recorded_run
    out = replay(run_dir)
    assert out == replay(run_dir)
    for kind in ("gate_pair", "gate_verdict", "genome_built", "operator_pick", "island_pick", "credit_update",
                 "control_round", "program", "gene_born"):
        assert out["checked"].get(kind, 0) > 0, kind

    def tampered(edit) -> Path:
        d = tmp_path / f"copy{len(list(tmp_path.iterdir()))}"
        shutil.copytree(run_dir, d)
        edit(d)
        return d

    def flip_gate(d):
        path = d / "events.jsonl"
        text = path.read_text()
        assert '"outcome":"win"' in text
        path.write_text(text.replace('"outcome":"win"', '"outcome":"loss"', 1))

    def change_genome(d):
        path = d / "events.jsonl"
        lines = path.read_text().splitlines()
        i = next(k for k, line in enumerate(lines) if '"type":"genome_built"' in line and '"pool":[{' in line)
        e = json.loads(lines[i])
        e["payload"]["outputs"]["genome"] = []
        lines[i] = json.dumps(e)
        path.write_text("\n".join(lines) + "\n")

    def drop_cache(d):
        next((d / "llm_cache").glob("*.json")).unlink()

    def swap_program(d):
        path = d / "experiments.jsonl"
        lines = path.read_text().splitlines()
        i = next(k for k, line in enumerate(lines) if '"llm_request":"' in line)
        lines[i] = re.sub(r'"algorithm_version":"[0-9a-f]{12}"', '"algorithm_version":"000000000000"', lines[i])
        path.write_text("\n".join(lines) + "\n")

    for edit in (flip_gate, change_genome, drop_cache, swap_program):
        with pytest.raises(ReplayError):
            replay(tampered(edit))
