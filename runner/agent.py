"""The agent step: read history -> propose ONE hypothesis + edits to strategy/ only.

Two agents share one interface:
    propose(context) -> Proposal(hypothesis, rationale_from_history, files)

* ClaudeAgent     calls the Claude API (needs ANTHROPIC_API_KEY, e.g. in .env).
* HeuristicAgent  offline fallback; picks the next untried idea from a fixed
                  playbook using the history. Lets the loop run with no key.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Proposal:
    hypothesis: str
    rationale_from_history: str
    files: dict[str, str] = field(default_factory=dict)  # path -> full new content
    agent: str = ""
    idea_key: str | None = None  # heuristic agent only: dedup key


@dataclass
class Context:
    root: Path
    history: list[dict]
    best: dict | None
    strategy_files: dict[str, str]
    program: str


# ---------------------------------------------------------------- heuristic
class HeuristicAgent:
    """Deterministic playbook. Each idea: (key, applies(cfg), edit(cfg), hypothesis)."""

    name = "heuristic"

    PLAYBOOK = [
        ("hill_climb", lambda c: c.get("method") != "hill_climb",
         lambda c: {**c, "method": "hill_climb"},
         "Switching from random sampling to hill climbing (perturb one point, keep if min-distance does not drop) "
         "will raise the mean score, because local refinement exploits structure that independent samples ignore."),
        ("more_samples", lambda c: int(c.get("n_samples", 0)) < 20000,
         lambda c: {**c, "n_samples": int(c.get("n_samples", 1000)) * 5},
         "Giving the current method 5x more iterations will raise the score, since runs finish far below the time budget."),
        ("smaller_step", lambda c: c.get("method") == "hill_climb" and float(c.get("step_size", 0.05)) > 0.011,
         lambda c: {**c, "step_size": round(float(c.get("step_size", 0.05)) / 2, 4)},
         "Halving the hill-climb step size will raise the score by letting points settle more precisely near the optimum."),
        ("larger_step", lambda c: c.get("method") == "hill_climb" and float(c.get("step_size", 0.05)) < 0.2,
         lambda c: {**c, "step_size": round(float(c.get("step_size", 0.05)) * 2, 4)},
         "Doubling the hill-climb step size will raise the score by escaping poor local optima early on."),
    ]

    def propose(self, ctx: Context) -> Proposal:
        cfg = yaml.safe_load(ctx.strategy_files["strategy/config.yaml"]) or {}
        tried = {r.get("idea_key") for r in ctx.history}
        rejected = [r for r in ctx.history if r["decision"] == "reject"]
        for key, applies, edit, hyp in self.PLAYBOOK:
            # an idea can be retried only if the config it applies to has changed since
            tag = f"{key}@{json.dumps(cfg, sort_keys=True)}"
            if tag in tried or not applies(cfg):
                continue
            new_cfg = edit(cfg)
            best = ctx.best
            rationale = (
                f"{len(ctx.history)} experiments so far; best score {best['score']:.6f} from {best['id']} "
                f"(config: {json.dumps(cfg, sort_keys=True)}). "
                if best else "No scored experiment yet. "
            )
            if rejected:
                last = rejected[-1]
                rationale += f"Last rejected: {last['id']} ({last['hypothesis'][:80]}...). "
            rationale += f"Untried playbook idea '{key}' applies to the current config."
            return Proposal(hyp, rationale, {"strategy/config.yaml": yaml.safe_dump(new_cfg, sort_keys=False)},
                            self.name, idea_key=tag)
        raise RuntimeError("heuristic playbook exhausted: set ANTHROPIC_API_KEY to use the Claude agent")


# ---------------------------------------------------------------- Claude
SYSTEM_PROMPT = """You are the hypothesis step of an autonomous research loop that tries to improve a \
mathematical optimisation record. You may only change the files under strategy/ that you are given. \
The problem definition, verifier and runner are frozen and hash-checked; never try to change them or to \
game the verifier. Every experiment is run with the same seeds and time budget and scored by an \
independent verifier, so propose one focused, testable change per experiment and learn from the history \
(especially from rejected and failed experiments).

Reply with ONLY one JSON object, no prose around it:
{"hypothesis": "<one falsifiable sentence: what you change and what score effect you expect>",
 "rationale_from_history": "<which past experiments / numbers led you here>",
 "files": {"strategy/search.py": "<full new file content, or omit>",
           "strategy/config.yaml": "<full new file content, or omit>"}}"""


def load_dotenv(root: Path) -> None:
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _history_digest(rows: list[dict]) -> str:
    keep = ("id", "decision", "score", "best_before", "hypothesis", "failure_reason", "edited_files")
    return "\n".join(json.dumps({k: r.get(k) for k in keep}) for r in rows)


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("agent reply contained no JSON object")
    return json.loads(m.group(0))


class ClaudeAgent:
    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high"):
        import anthropic  # imported lazily so the heuristic path needs no install

        self.client = anthropic.Anthropic()
        self.model, self.effort = model, effort

    def _call(self, user: str) -> str:
        kwargs = dict(
            model=self.model, max_tokens=16000, system=SYSTEM_PROMPT,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": user}],
        )
        try:  # server-side refusal fallback; drop it if the installed SDK predates it
            resp = self.client.beta.messages.create(
                **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        except TypeError:
            resp = self.client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            raise RuntimeError("model refused the request")
        return "".join(b.text for b in resp.content if b.type == "text")

    def propose(self, ctx: Context) -> Proposal:
        best = ctx.best
        user = (
            f"# program.md\n{ctx.program}\n\n"
            f"# Best so far\n{json.dumps({k: best[k] for k in ('id', 'score', 'hypothesis')}) if best else 'none'}\n\n"
            f"# History (oldest first)\n{_history_digest(ctx.history) or '(empty)'}\n\n"
            + "".join(f"# Current {p}\n```\n{c}\n```\n\n" for p, c in ctx.strategy_files.items())
            + "Propose the next experiment."
        )
        data = _extract_json(self._call(user))
        return Proposal(
            hypothesis=str(data["hypothesis"]),
            rationale_from_history=str(data.get("rationale_from_history", "")),
            files={str(k): str(v) for k, v in (data.get("files") or {}).items()},
            agent=f"claude:{self.model}",
        )


def make_agent(kind: str, root: Path, model: str, effort: str):
    load_dotenv(root)
    if kind == "auto":
        kind = "claude" if os.environ.get("ANTHROPIC_API_KEY") else "heuristic"
    if kind == "claude":
        return ClaudeAgent(model, effort)
    return HeuristicAgent()
