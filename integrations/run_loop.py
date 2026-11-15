"""Run the WS2 research loop with the WS4 integrations plugged in.

    python -m integrations.run_loop --problem 1a -n 5 --budget 20 --agent-id laptop-a --exa --pastdev
    python -m integrations.run_loop --problem 1a -n 5 --budget 20 --agent-id laptop-b --pastdev

--exa           literature search (record audit, then ideas on plateau)
--pastdev       read other agents' facts as evidence and mirror every experiment
                to shared memory, attributed to --agent-id
--memory-scope  isolate this comparison from earlier runs: facts are tagged with the
                scope and only facts with the same tag are read. The first use of a
                scope on this machine records its start time, and only facts written
                after it are requested from past.dev (--memory-since overrides it,
                e.g. for Modal workers that share a scope started elsewhere).
                Scopes named import-* (integrations.import_results) use the tag only.
                Without a scope, agents read the accumulated memory of every run.
IMPORTANT for comparisons: give every arm its own --store file. The agent reads
the experiment history from the store, so a shared store leaks results between
arms even with every sponsor switched off.
--idea ID       run every experiment of this run under one idea of the research map
                (integrations.ideas): the rationale gets `idea=ID` next to WS2's
                `strategy=...; parent=...` tags. A dead end, or an idea about another
                problem, is refused before any compute is spent.
--ideas         search over ideas: before each experiment a bandit picks the idea
                (integrations.idea_chooser; dead ends never), then that idea's own knob
                tuner picks the config. Only ideas with a runner can be picked:
                python -m integrations.ideas runner ID --optimizer optimizers/x.py
--idea-problem SLUG   the map's name for the problem of this run, when it differs from
                --problem (WS2 says "1a", the map says "second-autocorrelation-inequality")
--delta X       the problem's delta, when the problem module does not carry it
--seed N        the seed of every experiment of this run (default: WS2's fixed prepare.SEED).
                It is stored in each record, so two arms run with the same seeds can be
                compared like for like, and a repeat with another seed is not a "repeat".
                With --idea or --ideas, the idea's verdict (reproduced / refuted by us,
                experiment ids, gain/delta) is updated after every experiment
                (integrations.verdicts). Import ideas first: python -m integrations.import_knowledge
Every experiment record gets an "integrations" field with the switches in force.
Without flags it behaves exactly like `python loop.py` (plus that field).
"""
from __future__ import annotations

import argparse
import sys

import interfaces as I
import prepare
from loop import _print, research_loop
from research import FeedbackAgent

from . import ideas, verdicts
from .evidence import PastDevMirrorStore, TaggedStore, from_flags
from .idea_agent import IdeaAgent
from .idea_chooser import IdeaChooser, IdeaSearchAgent
from .pastdev_client import get_memory, scope_start


def seeded_loop(problem, optimizer, agent, store, evidence_source=None, n=5, budget_s=None, on_result=None,
                seed=None) -> list:
    """WS2's research_loop (loop.py), line for line, except that every experiment is run
    with the given seed. WS2's function has no seed argument and its file is not ours to
    change; run_experiment() itself takes the seed, so the protocol is the same."""
    evidence_source = evidence_source or I.NoEvidence()
    results = []
    for _ in range(n):
        history = store.read()
        evidence = evidence_source.gather(problem, history)
        exp = agent.propose(history, evidence)
        opt = optimizer if exp.optimizer is None else I.FileOptimizer(exp.optimizer)
        result = I.run_experiment(problem, opt, exp, store, budget_s=budget_s, seed=seed, agent=agent.name,
                                  evidence=evidence)
        results.append(result)
        if on_result:
            on_result(result)
        if agent.review(result) == "stop":
            break
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--optimizer")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--store", default=str(prepare.ROOT / "experiments.jsonl"))
    ap.add_argument("--agent-id", default="agent", help="identity used in shared memory")
    ap.add_argument("--exa", action="store_true")
    ap.add_argument("--pastdev", action="store_true")
    ap.add_argument("--memory-scope", default=None, help="isolate shared memory for a fair comparison")
    ap.add_argument("--memory-since", default=None, help="ISO time; read only facts written after it")
    ap.add_argument("--no-warm-start", action="store_true")
    ap.add_argument("--idea", default=None, help="idea id from the research map; tags every experiment")
    ap.add_argument("--ideas", action="store_true", help="let the idea chooser pick the idea of each experiment")
    ap.add_argument("--idea-problem", default=None, help="the map's slug for this run's problem")
    ap.add_argument("--delta", type=float, default=None, help="delta of the problem (gains are reported in delta)")
    ap.add_argument("--seed", type=int, default=None, help="seed of every experiment (default: prepare.SEED)")
    ap.add_argument("--ideas-file", default=None, help="idea store (default: %s)" % ideas.default_path())
    a = ap.parse_args(argv)

    scope = a.memory_scope
    # import-* scopes hold facts written before this run (integrations.import_results): tag filter only
    since = a.memory_since or (scope_start(scope) if scope and not scope.startswith("import-") else None)
    problem = I.get_problem(a.problem)
    opt = I.FileOptimizer(a.optimizer or prepare.ROOT / "optimizers" / f"{a.problem}_baseline.py")
    if a.idea and a.ideas:
        ap.error("--idea (one idea) and --ideas (the chooser picks) are exclusive")
    agent = None
    if not a.ideas:          # with --ideas each idea brings its own optimizer: the default file may not exist
        agent = FeedbackAgent(problem, opt, warm_start=False if a.no_warm_start else None)
        agent.name = f"{agent.name}@{a.agent_id}"     # experiment records show which machine/agent ran it
    idea_store = ideas.IdeaStore(a.ideas_file) if (a.idea or a.ideas) else None
    slug = a.idea_problem or problem.id                       # the problem's name on the map
    aliases = {problem.id: slug} if slug != problem.id else None
    ref = verdicts.reference_from_problem(problem)
    if a.delta is not None:
        ref.delta = a.delta
    try:
        if a.idea:
            agent = IdeaAgent(agent, idea_store, a.idea, problems={slug})
        elif a.ideas:
            def tuner(node):                      # one knob tuner per idea, on the idea's own optimizer
                path = (node.get("runner") or {}).get("optimizer")
                t = FeedbackAgent(problem, I.FileOptimizer(path) if path else opt,
                                  warm_start=False if a.no_warm_start else None)
                t.name = f"{t.name}@{a.agent_id}"
                return t
            chooser = IdeaChooser(idea_store, lambda node: ref, problems={slug}, aliases=aliases)
            if not chooser.candidates():
                waiting = [n["id"] for n in chooser.frontier()]
                ap.error("no runnable idea for %s: %d live ideas have no runner (%s). Give one a runner: "
                         "python -m integrations.ideas runner ID --optimizer optimizers/x.py"
                         % (slug, len(waiting), ", ".join(waiting[:8]) or "none"))
            agent = IdeaSearchAgent(idea_store, chooser, tuner, name=f"ideas@{a.agent_id}",
                                    sign=ref.sign, record_problems={problem.id})
    except (KeyError, ValueError) as e:           # unknown id, not testable, wrong problem, dead end, corrupt store
        ap.error(str(e.args[0] if e.args else e))

    memory = get_memory() if a.pastdev else None
    base = PastDevMirrorStore(a.store, memory, author=a.agent_id, scope=scope) if a.pastdev \
        else I.JsonlStore(a.store)
    store = TaggedStore(base, {"exa": a.exa, "pastdev": a.pastdev, "memory_scope": scope,
                               "memory_since": since, "agent_id": a.agent_id,
                               "idea": a.idea or ("chooser" if a.ideas else None), "seed": a.seed,
                               "memory_backend": memory.name if memory else None})
    if idea_store is not None:                    # outermost: the verdict reads the record as stored
        store = verdicts.VerdictStore(store, idea_store, ref, memory=memory, aliases=aliases,
                                      hash_root=prepare.ROOT, label=verdicts.store_label(a.store))
    evidence = from_flags(a.agent_id, a.exa, a.pastdev, memory=memory, scope=scope, since=since)
    print(f"agent: {agent.name} | exa: {a.exa} | past.dev: {a.pastdev} | scope: {scope} since {since}"
          f" | idea: {a.idea or ('chooser' if a.ideas else None)} | seed: {a.seed if a.seed is not None else 'default'}")

    def show(r):
        _print(r)
        tagged = ideas.idea_of(r)
        if tagged and idea_store is not None:
            try:
                node = idea_store.get(tagged) or {}
            except ideas.CorruptStore as e:
                node = {}
                print(f"           {e}")
            v = node.get("verdict") or {}
            print(f"           idea {tagged}: {v.get('status', 'no verdict')}, best gain/delta "
                  f"{v.get('best_gain_over_delta')}, {v.get('trials', 0)} experiments")
        for e in r.get("evidence") or []:
            e = e if isinstance(e, dict) else e.__dict__
            print(f"           evidence [{e.get('source')}] by {e.get('author')}: {e.get('citation')}")

    if a.seed is None:
        research_loop(problem, opt, agent, store, evidence_source=evidence, n=a.n, budget_s=a.budget, on_result=show)
    else:
        seeded_loop(problem, opt, agent, store, evidence_source=evidence, n=a.n, budget_s=a.budget, on_result=show,
                    seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
