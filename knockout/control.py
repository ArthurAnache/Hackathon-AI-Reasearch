"""Hill Sampling control arm (section 5.10; "Hill Sampling", arXiv 2609.25510 as cited by the team brief).

Each round: N edits of the incumbent, all tuned by the same tuner as the main arm, keep the best if it beats the
incumbent by more than EPS. No genes, no archive, isolated from the main population. It owns a fixed budget share.

Warm-start ratchet (`control.ratchet: true`, for problems searched from a warm start): the best verified candidate
of a round becomes `init` for the next round, so gains accumulate. Programs are then compared from the same `init`:
each round also replays the incumbent program, and an edit is kept only if it beats that replay by more than EPS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from knockout.context import Evaluation, RunContext, source_hash
from knockout.editor import EditProposal, Editor

REPLAY = "replay"


@dataclass
class Incumbent:
    program_id: str
    source: str
    score: float | None
    experiment_id: str | None


class HillSamplingArm:
    category = "control"
    chain = "control"
    agent = "knockout/control-hill-sampling"

    def __init__(self, ctx: RunContext, editor: Editor, seed_source: str, rng: np.random.Generator):
        self.ctx = ctx
        self.editor = editor
        self.rng = rng
        self.incumbent = Incumbent(f"p_{source_hash(seed_source)}", seed_source, None, None)
        self.rounds = 0
        self.curve: list[tuple[float, float]] = []      # (cumulative arm spend, best score) for the report
        self.eval_seed = int(ctx.strategy["run.seed"])  # common random numbers: every program sees the same seed
        self.ratchet = bool(ctx.strategy.get("control.ratchet", False))
        # Best verified candidate of the arm. With the ratchet it starts at the warm start itself (no experiment).
        self.best_score: float | None = ctx.init_score if self.ratchet else None
        self.best_experiment: str | None = None
        self.best_solution_path: str | None = None

    @property
    def budget_s(self) -> float:
        return float(self.ctx.strategy["fidelity.f0_s"])

    def _spend(self) -> float:
        return self.ctx.governor.spend(self.category)

    def _is_new_best(self, ev: Evaluation) -> bool:
        return ev.valid and (self.best_score is None or self.ctx.problem.gain(ev.score, self.best_score) > 0)

    def _set_best(self, ev: Evaluation, exp_id: str, record: dict) -> None:
        """Record a new best candidate; with the ratchet it also becomes the warm start of the next round."""
        before = self.best_score
        self.best_score, self.best_experiment = ev.score, exp_id
        self.best_solution_path = record.get("solution_path")
        if self.ratchet:
            self.ctx.init, self.ctx.init_score = ev.candidate, ev.score
            self.ctx.ledger.event("warm_start_update", seed=None, refs=[exp_id], score_before=before,
                                  score_after=ev.score, solution_path=self.best_solution_path)

    def baseline(self) -> Evaluation:
        ctx, inc = self.ctx, self.incumbent
        ev = ctx.finish(ctx.submit(inc.source, self.budget_s, self.eval_seed).result(), self.category)
        exp_id = ctx.ledger.next_experiment_id()
        rec = ctx.log_experiment(experiment_id=exp_id, ev=ev, source=inc.source, program_id=inc.program_id,
                                 chain=self.chain, agent=self.agent,
                                 hypothesis="baseline: the seed program as written",
                                 rationale="first evaluation of the control arm's incumbent", decision="baseline",
                                 status="keep" if ev.valid else ev.status, seed=self.eval_seed,
                                 budget_s=self.budget_s, best_before=self.best_score,
                                 extra_config={"arm": "control", "parent_id": None, "fidelity": "F0",
                                               "ref_score": ctx.init_score if self.ratchet else None})
        if ev.valid:
            inc.score, inc.experiment_id = ev.score, exp_id
            if self._is_new_best(ev):
                self._set_best(ev, exp_id, rec)
            self.curve.append((self._spend(), self.best_score))
        return ev

    def step(self) -> dict:
        """One round. Returns a short summary for the console."""
        ctx, inc = self.ctx, self.incumbent
        round_seed = int(self.rng.integers(2**31 - 1))
        proposals = self.editor.propose(inc.source, int(ctx.strategy["control.edits_per_round"]),
                                        np.random.default_rng(round_seed))
        n_edits = len(proposals)
        if self.ratchet:  # the incumbent program runs again, from the same warm start as its edits
            proposals = [EditProposal(inc.source, "replay: the incumbent program from the current warm start",
                                      REPLAY)] + proposals
        init_score = ctx.init_score
        futures = [(p, ctx.submit(p.source, self.budget_s, self.eval_seed)) for p in proposals]
        children = []
        for p, fut in futures:
            ev = ctx.finish(fut.result(), self.category)
            children.append((p, ev, ctx.ledger.next_experiment_id(), f"p_{source_hash(p.source)}"))
        # Reference the edits must beat: the incumbent's stored score, or its replay from the current warm start
        # (the warm start's own score if the replay failed).
        ref = inc.score
        if self.ratchet:
            replay = children[0][1]
            ref = replay.score if replay.valid else init_score
        best, top = None, None
        for child in children:
            p, ev = child[0], child[1]
            if not ev.valid:
                continue
            if top is None or ctx.problem.gain(ev.score, top[1].score) > 0:
                top = child                                   # best candidate of the round, replay included
            if p.operator != REPLAY and (best is None or ctx.problem.gain(ev.score, best[1].score) > 0):
                best = child                                  # best edit of the round
        eps = float(ctx.strategy["selection.eps_rel"])
        improved = best is not None and ctx.problem.improves(best[1].score, ref, eps)
        # New best candidate of the arm: with the ratchet, any candidate above the best so far (the replay counts);
        # without it, the kept edit.
        winner = top if self.ratchet else best
        new_best = (top is not None and self._is_new_best(top[1])) if self.ratchet else improved
        best_before = self.best_score
        for p, ev, exp_id, pid in children:
            chosen = improved and exp_id == best[2]
            is_top = new_best and exp_id == winner[2]
            if not ev.valid:
                decision, status = ev.status, ev.status
                rationale = f"no valid candidate: {ev.error}"
            elif p.operator == REPLAY:
                decision, status = "baseline", "keep"
                rationale = f"reference of round {self.rounds}: incumbent program scored {ev.score:.15g} from the " \
                            f"current warm start ({init_score})"
            elif chosen:
                decision = status = "keep"
                rationale = (f"best of {n_edits} edits; tuned score {ev.score:.15g} beats the reference "
                             f"{ref} by more than eps_rel={eps:g}")
            else:
                decision = status = "discard"
                rationale = (f"tuned score {ev.score:.15g} vs reference {ref}: "
                             + ("not the best edit of the round" if improved else f"no gain above eps_rel={eps:g}"))
            if is_top:
                rationale += "; best candidate so far" + (", becomes the warm start" if self.ratchet else "")
            rec = ctx.log_experiment(
                experiment_id=exp_id, ev=ev, source=p.source, program_id=pid, chain=self.chain, agent=self.agent,
                hypothesis=p.description, rationale=rationale, decision=decision, status=status,
                seed=self.eval_seed, budget_s=self.budget_s, best_before=best_before,
                save_solution=chosen or is_top,
                extra_config={"arm": "control", "parent_id": inc.program_id, "parent_experiment": inc.experiment_id,
                              "operator": p.operator, "editor": "none" if p.operator == REPLAY else self.editor.name,
                              "fidelity": "F0", "round": self.rounds, "ref_score": ref, "edit_meta": p.meta})
            if is_top:
                self._set_best(ev, exp_id, rec)
        before = inc.score
        if improved:
            p, ev, exp_id, pid = best
            self.incumbent = Incumbent(pid, p.source, ev.score, exp_id)
        elif self.ratchet and ref is not None:
            inc.score = ref
        self.curve.append((self._spend(), self.best_score))
        ctx.ledger.event("control_round", seed=round_seed, refs=[c[2] for c in children],
                         round=self.rounds, incumbent_before=before, incumbent_after=self.incumbent.score,
                         improved=improved, chosen=best[2] if improved else None, reference=ref,
                         best_before=best_before, best_after=self.best_score, n_edits=n_edits,
                         inputs={"edit_scores": {c[2]: c[1].score if c[1].valid else None for c in children
                                                 if c[0].operator != REPLAY},
                                 "reference": ref, "eps_rel": eps, "direction": ctx.problem.direction},
                         outputs={"improved": improved, "chosen": best[2] if improved else None},
                         child_scores={c[2]: c[1].score for c in children})
        self.rounds += 1
        return {"round": self.rounds, "n": len(children), "n_edits": n_edits,
                "valid": sum(c[1].valid for c in children), "best_child": best[1].score if best else None,
                "incumbent": self.incumbent.score, "improved": improved, "best": self.best_score,
                "new_best": new_best}
