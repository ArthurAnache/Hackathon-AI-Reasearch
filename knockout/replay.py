"""Re-derive every logged decision of a run from its events and its LLM cache, without calling any API.

    .venv/bin/python -m knockout.replay runs/<run_id>

Each decision event stores the inputs, the outputs and the seed of one pure function of knockout.decisions. This
module calls the same function on the recorded inputs and fails loudly when an output differs. It also checks that:
- every answered `llm_call` has its response in runs/<run_id>/llm_cache/;
- every program the LLM wrote is the program that was evaluated (hash of the cached source = `algorithm_version`);
- every gene born from an LLM answer appears in that cached answer;
- every strategist patch touches allow-listed keys only;
- the gate counters in each `gate_verdict` equal the `gate_pair` outcomes logged before it;
- a `record_claim` or `numerical_artifact` decision in experiments.jsonl has its `promotion` event, and vice versa.
It does not re-run programs: scores are taken from the ledger (the saved solutions can be re-scored with
selection/verify_claim.py).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

from knockout import decisions as D
from knockout.ledger import validate_run


class ReplayError(AssertionError):
    pass


def _same(a, b, tol: float = 1e-12) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    return a == b


def _strip_fence(source: str) -> str:
    from knockout.llm_editor import strip_fence

    return strip_fence(source)


def _derive(e: dict) -> tuple[dict, dict] | None:
    """(recomputed outputs, recorded outputs restricted to the same keys), or None if the event has no decision."""
    t, p, seed = e["type"], e["payload"], e["seed"]
    i, o = p.get("inputs"), p.get("outputs")
    if i is None or o is None:
        return None
    if t == "island_pick":
        return D.thompson_pick(i["arms"], seed, i.get("weights")), o
    if t == "operator_pick":
        if p.get("kind") in ("parent", "parent_b"):
            return D.select_parent(i["ranked_ids"], i["power"], seed), o
        return D.thompson_pick(i["arms"], seed), o
    if t == "genome_built":
        built = D.build_genome(i["pool"], i["g_max"], seed)
        return {"genome": built["genome"], "draws": built["draws"], "phase": D.phase(**i["phase"])}, o
    if t == "gate_pair":
        return {"outcome": D.pair_outcome(**i)}, o
    if t == "gate_verdict":
        return D.gate_verdict(**i), o
    if t == "fidelity_promotion":
        return {"promoted": D.promote(i["scores"], i["ratio"], i["direction"])}, o
    if t == "credit_update":
        s = int(D.improves(i["child_score"], i["parent_score"], i["eps_rel"], i["direction"]))
        after = [D.credit(u["before"][0], u["before"][1], float(s), i["gamma"], u["weight"]) for u in i["updates"]]
        return {"s": s, "after": after}, o
    if t == "linkage":
        return {"verdict": D.linkage(**i)}, o
    if t == "migration":
        return {"migrants": D.migrants(i["genes"], i["min_posterior"])}, o
    if t == "lineage_frozen":
        return {"freeze": D.should_freeze(**i)}, o
    if t == "strategy_patch":
        return {"stagnated": D.stagnated(**i), "keys": o["keys"]}, o
    if t == "strategy_revert":
        return {"revert": D.should_revert(**i)}, o
    if t == "promotion":
        return {"decision": D.promotion(**i)}, o
    if t == "control_round":
        return D.control_round(i["edit_scores"], i["reference"], i["eps_rel"], i["direction"]), o
    return None


def replay(run_dir: str | Path) -> dict:
    """Returns counts per event type. Raises ReplayError on the first divergence."""
    run_dir = Path(run_dir)
    validate_run(run_dir)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines() if line.strip()]
    experiments = [json.loads(line) for line in (run_dir / "experiments.jsonl").read_text().splitlines()
                   if line.strip()]
    cache = run_dir / "llm_cache"
    strategy_file = run_dir / "strategy.yaml"
    allow = None
    if strategy_file.exists():
        import yaml

        allow = set(yaml.safe_load(strategy_file.read_text()).get("strategist_allow_list", []))
    checked: dict[str, int] = {}
    gate: dict[tuple[str, bool], dict[str, int]] = {}

    def fail(e: dict, msg: str):
        raise ReplayError(f"{run_dir.name} {e.get('event_id', e.get('experiment_id'))} ({e.get('type', 'experiment')}): {msg}")

    def cached(h: str | None) -> dict | None:
        path = cache / f"{h}.json"
        return json.loads(path.read_text()) if h and path.exists() else None

    for e in events:
        t, p = e["type"], e["payload"]
        derived = _derive(e)
        if derived is not None:
            new, old = derived
            if not _same(new, {k: old.get(k) for k in new}):
                fail(e, f"decision diverges: recorded {old}, re-derived {new}")
            checked[t] = checked.get(t, 0) + 1
        if t == "llm_call" and p.get("ok") and cached(p.get("request_hash")) is None:
            fail(e, f"no cached response for request {p.get('request_hash')}")
        if t == "gene_born" and p.get("request_hash"):
            c = cached(p["request_hash"])
            if c is None:
                fail(e, "the LLM answer this gene comes from is not in the cache")
            texts = [" ".join(str(g.get("text", "")).split()) for g in _genes_in(json.loads(c["text"]))]
            if not any(t_.startswith(p["text"]) or p["text"].startswith(t_[:40]) for t_ in texts):
                fail(e, "the gene text is not in the cached LLM answer")
            checked["gene_born"] = checked.get("gene_born", 0) + 1
        if t == "gate_pair":
            g = gate.setdefault((p["gene"], bool(p.get("audit"))), {"wins": 0, "losses": 0, "pairs": 0})
            g["pairs"] += 1
            g["wins"] += p["outputs"]["outcome"] == "win"
            g["losses"] += p["outputs"]["outcome"] == "loss"
        if t == "gate_verdict":
            g = gate.get((p["gene"], bool(p.get("audit"))), {})
            if any(g.get(k) != p["inputs"][k] for k in ("wins", "losses", "pairs")):
                fail(e, f"gate counters {p['inputs']} do not match the gate_pair events {g}")
            if p["outputs"]["verdict"] is not None:
                gate.pop((p["gene"], bool(p.get("audit"))), None)
        if t == "strategy_patch" and allow is not None:
            bad = [k for k in p.get("patch", {}) if k not in allow]
            if bad:
                fail(e, f"patched keys outside the allow-list: {bad}")
    promoted = {r for e in events if e["type"] == "promotion" for r in e["refs"]}
    for r in experiments:
        h = r["config"].get("llm_request") or (r["config"].get("edit_meta") or {}).get("request_hash")
        if h:
            c = cached(h)
            if c is None:
                fail(r, f"no cached LLM answer for request {h}")
            source = _strip_fence(str(json.loads(c["text"]).get("source", "")))
            if hashlib.sha256(source.encode()).hexdigest()[:12] != r["algorithm_version"]:
                fail(r, "the evaluated program is not the program in the cached LLM answer")
            checked["program"] = checked.get("program", 0) + 1
        claimed = r["decision"] in ("record_claim", "numerical_artifact", "exploratory_gate")
        if claimed != (r["experiment_id"] in promoted):
            fail(r, "record_claim / numerical_artifact decisions and promotion events do not match")
    return {"events": len(events), "experiments": len(experiments), "checked": checked}


def _genes_in(answer: dict) -> list[dict]:
    genes = list(answer.get("genes") or [])
    for item in answer.get("patch") or []:       # strategist seed genes
        if item.get("key") == "seed_genes":
            try:
                genes += json.loads(item["value_json"])
            except (ValueError, TypeError):
                pass
    return [g for g in genes if isinstance(g, dict)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    try:
        print(json.dumps(replay(sys.argv[1])))
    except ReplayError as exc:
        print(f"REPLAY FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
