"""Research tools for the evolutionary loop: Exa (literature) and past.dev (shared memory between agents).

Both come from WS4's adapters in integrations/ and are optional: with no key, Exa is skipped and memory
falls back to a local file, so the loop runs the same way, only less informed.

Where each tool enters the loop (once per round, in the parent process, never inside a job):

  Diversity: a gene that comes from another agent is given to ONE island (see _home); burials go to all.

  before_round   Exa       on the first round (record audit) and whenever every island has gone
                           `plateau` rounds without a better certificate (idea mining). Findings go
                           to the LLM editor's prompt and are written to shared memory.
                 past.dev  other agents' gene verdicts for this problem:
                             validated elsewhere -> new candidate gene here, with a favourable prior
                                                    (it must still pass THIS island's knock-out gate)
                             buried elsewhere    -> buried here too, never retried
                           and configs other agents kept -> new config genes (tested on their own first).
  after_round    past.dev  publish this run's new verdicts (validated / graveyard) and new best scores.
                           Solutions stay in the certificate store; only scores and genes are shared.

Every gene that comes from a tool carries origin "exa:<url>" or "past.dev:<author>", and every record
of a candidate that uses such a gene cites it in its `evidence` field, so a decision can be traced back
to the search result or the agent that suggested it. Tool output is untrusted data: it only ever becomes
a gene (which must survive selection like any other) or text in a prompt; it is never executed.
"""
from __future__ import annotations

import json
import re

from .genes import Gene

HINT = re.compile(r"\{.*\}", re.S)
SHARED_KIND = "gene_verdict"


def config_gene(hint: dict, space, origin: str) -> Gene | None:
    """A config hint ({knob: value}) -> one config gene, keeping only knobs the optimizer has, in range."""
    params = {}
    for k, v in sorted(hint.items()):
        knob = space.knobs.get(k)
        if knob is None or not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        v = int(round(v)) if knob.integer else float(f"{float(v):.6g}")
        if knob.low <= v <= knob.high:
            params[k] = v
    if not params:
        return None
    return Gene.make("config", ",".join(f"{k}={v}" for k, v in params.items()), origin=origin, **params)


def genes_from_evidence(evidence, space) -> list[Gene]:
    out = []
    for e in evidence:
        m = HINT.search(e.content or "")
        try:
            hint = json.loads(m.group(0)).get("config") if m else None
        except (json.JSONDecodeError, AttributeError):
            hint = None
        if isinstance(hint, dict):
            g = config_gene(hint, space, f"{e.source}:{e.author or e.citation}")
            if g is not None:
                out.append(g)
    return out


def _home(gene: Gene, islands):
    """The one island a shared gene goes to. WS3: with N chains reading the same memory, hints pull every
    chain to the same config, while diversity is what pays; so each hint seeds one island only, and it
    reaches the others only by winning there (knock-out gate, then migration). Burials go everywhere."""
    import hashlib
    return islands[int(hashlib.sha256(gene.id.encode()).hexdigest(), 16) % len(islands)]


def cite(genes: list[dict]) -> list[dict]:
    """Evidence entries for the record of a candidate whose genome contains tool-suggested genes."""
    out = []
    for d in genes:
        src, _, who = (d.get("origin") or "").partition(":")
        if src in ("exa", "past.dev"):
            out.append({"source": src, "content": f"suggested gene {d['id']}", "citation": who, "author": who})
    return out


class ResearchTools:
    """Exa + past.dev for the orchestrator. Every part is optional (None = switched off)."""

    def __init__(self, exa=None, memory=None, agent_id: str = "evolution", scope: str | None = None,
                 since: str | None = None, plateau: int = 2):
        self.exa, self.memory = exa, memory
        self.agent_id, self.scope, self.since, self.plateau = agent_id, scope, since, plateau
        self.literature: list = []         # Exa evidence, shown to the LLM editor
        self.rounds: list[dict] = []       # one row per round, for Exa's plateau trigger
        self.published: set = set()
        self.imported: set = set()

    @classmethod
    def from_flags(cls, exa: bool, pastdev: bool, agent_id: str = "evolution", scope=None, since=None):
        if not (exa or pastdev):
            return cls(agent_id=agent_id)
        from integrations.evidence import ExaEvidence
        from integrations.pastdev_client import get_memory
        memory = get_memory() if pastdev else None
        return cls(ExaEvidence(memory=memory, plateau=2, scope=scope) if exa else None, memory, agent_id, scope, since)

    @property
    def tag(self) -> str:
        on = [n for n, x in (("exa", self.exa), ("pastdev", self.memory)) if x is not None]
        return f"evolution@{self.agent_id}" + (f"[{','.join(on)}{';scope=' + self.scope if self.scope else ''}]" if on else "")

    # ------------------------------------------------------------------ before a round
    def before_round(self, orch) -> dict:
        added = {"literature": 0, "genes": [], "buried": []}
        if self.exa is not None:
            try:
                found = self.exa.gather(orch.problem, self.rounds)
            except Exception:  # noqa: BLE001  a tool that fails never stops the loop
                found = []
            self.literature = (found + self.literature)[:8]
            added["literature"] = len(found)
            for isl in orch.islands:
                isl.evidence = list(self.literature)
        if self.memory is not None:
            added.update(self._import_memory(orch))
        return added

    def _import_memory(self, orch) -> dict:
        from integrations.evidence import PastDevEvidence
        added = {"genes": [], "buried": []}
        try:
            facts = self.memory.read(f"problem {orch.problem.id} gene verdicts and kept configs", reader=self.agent_id,
                                     limit=20, scope=self.scope, since=self.since)
        except Exception:  # noqa: BLE001
            facts = []
        space = orch.islands[0].space
        for f in facts:
            meta = f.get("metadata") or {}
            if meta.get("problem") != orch.problem.id or f.get("author") == self.agent_id:
                continue
            if meta.get("kind") == SHARED_KIND and isinstance(meta.get("gene"), dict):
                key = (meta["gene"].get("id"), meta.get("verdict"))
                if key in self.imported:
                    continue
                self.imported.add(key)
                g = Gene.from_dict({**meta["gene"], "origin": f"past.dev:{f.get('author')}"})
                home = _home(g, orch.islands)
                for isl in orch.islands:
                    if meta.get("verdict") == "validated" and isl is home and (isl.pool.add(g) or (
                            isl.pool.state[g.id] == "candidate" and isl.pool.ab[g.id] == [1.0, 1.0])):
                        isl.pool.ab[g.id] = [3.0, 1.0]           # favourable prior; still gated here
                        isl.pool.notes[g.id] = f"validated by {f.get('author')} (past.dev {f.get('fact_id')})"
                        added["genes"].append((isl.id, g.id))
                    elif meta.get("verdict") == "graveyard" and isl.pool.state.get(g.id, "candidate") == "candidate" \
                            and g.id not in isl.tested:
                        isl.pool.add(g)
                        isl.pool.bury(g.id, f"buried by {f.get('author')} (past.dev {f.get('fact_id')})")
                        added["buried"].append((isl.id, g.id))
        # configs that other agents (e.g. the FeedbackAgent loop) kept become config genes, tested alone first
        hints = PastDevEvidence(self.memory, reader=self.agent_id, scope=self.scope, since=self.since)
        try:
            ev = hints.gather(orch.problem, [])
        except Exception:  # noqa: BLE001
            ev = []
        for g in genes_from_evidence(ev, space):
            isl = _home(g, orch.islands)
            if isl.pool.add(g):
                added["genes"].append((isl.id, g.id))
        return added

    # ------------------------------------------------------------------ after a round
    def after_round(self, orch, report, improved: bool) -> int:
        self.rounds.append({"problem": orch.problem.id, "status": "keep" if improved else "discard"})
        if self.memory is None:
            return 0
        sent = 0
        for isl in orch.islands:
            for gid, state in isl.pool.state.items():
                if state not in ("validated", "graveyard") or (gid, state) in self.published:
                    continue
                if isl.pool.notes.get(gid, "").startswith(("validated by", "buried by", "validated on island")):
                    continue                     # learned from others: do not echo it back
                self.published.add((gid, state))
                g = isl.pool.genes[gid]
                try:
                    self.memory.write(f"Gene {gid} on problem {orch.problem.id} was {state} by {self.agent_id} "
                                      f"({isl.pool.notes.get(gid, '')}).", author=self.agent_id, kind=SHARED_KIND,
                                      label=f"{orch.problem.id} {state}",
                                      metadata={"problem": orch.problem.id, "gene": g.to_dict(), "verdict": state,
                                                "note": isl.pool.notes.get(gid, "")},
                                      wait=False, scope=self.scope)
                    sent += 1
                except Exception:  # noqa: BLE001
                    pass
        if improved:
            best = None
            for i in orch.islands:
                if i.best_score is not None and (best is None or orch.problem.better(i.best_score, best.best_score)):
                    best = i
            try:
                self.memory.write(f"Best verified score on problem {orch.problem.id}: {best.best_score} "
                                  f"(certificate {best.best_cert}, agent {self.agent_id}).", author=self.agent_id,
                                  kind="best_score", label=f"{orch.problem.id} best",
                                  metadata={"problem": orch.problem.id, "score": best.best_score,
                                            "certificate": best.best_cert}, wait=False, scope=self.scope)
                sent += 1
            except Exception:  # noqa: BLE001
                pass
        return sent
