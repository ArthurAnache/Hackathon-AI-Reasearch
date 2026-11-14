# Sponsor tools: why we use each one and how we measure it

Track 1 asks for an autonomous research loop that beats a best-known math
record, with every decision traceable. Each sponsor tool covers one stage of
that loop:

```
 Exa          ->  Claude        ->  Devin        ->  Modal          ->  Verifier  ->  past.dev
 (reads the       (proposes the     (implements      (runs the          (exact     (remembers what
  literature)      next idea)        new methods)     experiments)       check)     was learned)
      ^                                                                                  |
      +----------------- the next iteration starts from what was learned ----------------+
```

**Principle:** a tool counts only if it changes a decision or a result that
shows up in the log. Every metric below is measured as an **ablation**: the
same loop, same seeds, same compute budget, with the tool switched on and off.
Every tool sits behind an adapter with a fallback, so switching one off never
breaks the loop.

Status key: ✅ implemented · 🟡 partially implemented · ⬜ planned

---

## 1. Exa: knowing where the finish line is

**Why it matters most.** Track 1 is judged on beating the record, so we first
have to know what the true current record is. Leaderboards go stale. Announcing
a "record" that someone already published is the worst outcome possible, and
the team's own rule forbids claiming a record from an outdated leaderboard.
Exa makes the target sourced rather than assumed.

**How the loop uses it**

| Use | What happens | Status |
|---|---|---|
| Record audit | Before any experiment, search for the best published value and its source; set the target from it | ✅ adapter, 🟡 wired into the loop |
| Warm start | Find published constructions (coordinates, coefficients) and start optimizing from them instead of from scratch | ⬜ |
| Pre-claim check | Before announcing an improvement, search again for equal or better published results | ⬜ |
| Idea mining | When the loop plateaus, search papers for techniques and turn them into hypotheses with `evidence_urls` | ⬜ |

**Metrics**

| Metric | How it is measured | What it proves |
|---|---|---|
| Time to reach the record | Warm start from Exa vs cold start, equal budget | Exa speeds up research |
| Best score at a fixed budget | Same comparison | Exa takes the search further |
| Target corrections | Times Exa showed our target was wrong (e.g. leaderboard said X, a paper says Y > X) | Exa prevents chasing the wrong goal |
| False records prevented | Candidate "records" that the pre-claim check found already published | Exa prevents a public embarrassment |
| ROI | Exa spend (USD) vs compute saved on Modal by not rediscovering the record | Exa pays for itself |
| Context: warm-start rate | Share of runs that started from a sourced construction | Usage, reported as context only |

**Safety:** the stop loss is in code (`EXA_BUDGET_USD`, default $5, and
`EXA_MAX_CALLS`, default 200). Identical queries are cached and cost nothing.
Every call is logged with its query, latency, cost and URLs in
`runs/provenance.jsonl`.

---

## 2. Modal: spending compute only where it pays off

**Why it matters.** Records fall to many experiments. A laptop runs about 8 at
a time, Modal runs 50 or more. The track explicitly rewards *progress per unit
of compute*, not raw usage, so the value comes from the loop deciding where to
spend.

**How the loop uses it**

| Use | What happens | Status |
|---|---|---|
| Parallel fan-out | One container per (config, seed), results re-verified locally | ✅ (WS3) |
| Sieve (successive halving) | Many ideas for 10 s, then the top ~20% for 1 min, then the top 2 for 10 min on several seeds, plus a random repechage | ⬜ (WS3) |
| Local first, cloud when promising | Cheap rounds run locally, only promising ideas scale out | ⬜ |
| Sandbox for generated code | Optimizer code written by an agent runs isolated | ⬜ |
| Cost caps | At most 50 containers and 600 s per experiment | ✅ |

**Metrics**

| Metric | How it is measured | What it proves |
|---|---|---|
| Time to reach the record | Laptop only vs laptop + Modal | Modal speeds up discovery |
| Experiments per hour | Count from the experiment log | More exploration in the same time |
| Cost per improvement | USD spent ÷ verified improvements | Efficient use, not waste |
| Sieve savings | Sieve vs a uniform long budget for every idea, at equal final quality | The loop knows where to spend |

---

## 3. past.dev: memory shared between agents

**Why it matters.** A loop without memory repeats its own mistakes and forgets
everything when it restarts. past.dev lets several agents write and read the
same source-backed memory, and it marks a fact as superseded when a newer one
replaces it, so the loop's beliefs visibly update.

**How the loop uses it**

| Use | What happens | Status |
|---|---|---|
| Shared facts | A scout writes record claims with sources, workers write lessons, the planner reads both and cites their ids in its rationale | ✅ adapter, 🟡 demo |
| Failed-idea graveyard | Every rejected idea becomes a fact, and the planner checks it before proposing | ⬜ |
| Warm restart | After a restart the loop recalls what worked instead of starting over | ⬜ |
| Belief updates | A new record or lesson supersedes the old one (`supersededAt`) | ⬜ |

**Metrics**

| Metric | How it is measured | What it proves |
|---|---|---|
| Repeated experiments avoided | Proposals blocked because memory showed the idea already failed | Memory saves compute |
| Cross-agent fact use | Decisions whose rationale cites a fact written by another agent | The sponsor requirement, measured |
| Recovery after restart | Iterations needed to get back to the best score, with vs without memory | Memory makes the loop durable |
| Recall latency | Seconds per read, from the provenance log | The memory is fast enough for the loop |

The sponsor requirement (more than one agent writes, more than one reads, and an
agent uses a fact produced by another) is checked from the `used_fact_ids` and
`used_fact_authors` fields in `runs/decisions.jsonl`.

---

## 4. Claude: the brain that chooses the next experiment

**Why it matters.** Track 1 demands *autonomous selection of the next
experiment*, not brute force. Claude reads the history and the evidence and
proposes what to try next. A stronger model plans, and a cheaper model handles
volume. Keep or reject is always decided by the exact verifier's numbers, never
by the model's opinion.

**How the loop uses it**

| Use | What happens | Status |
|---|---|---|
| Planner | Proposes the next hypothesis and config with a written rationale | ✅ (WS2, with a deterministic heuristic fallback) |
| Skeptic | A second agent tries to refute each improvement (seed luck, numerical error) before it is accepted | ⬜ |
| Readable log | Each decision is explained in plain language | ✅ (WS2) |

**Metrics**

| Metric | How it is measured | What it proves |
|---|---|---|
| Keep rate | Share of proposals that improve the score: Claude planner vs heuristic fallback | Claude picks better ideas |
| Improvement per LLM dollar | Verified score gain ÷ API spend | The planner is worth its cost |
| False improvements caught | Improvements rejected by the skeptic on re-verification | Rigor, not luck |

---

## 5. Devin: turning a paper into working code

**Why it matters.** When Exa surfaces a technique we have not implemented,
Devin can implement it as a new optimizer. A human reviews every change before
it is merged. This closes the loop from literature to experiment.

**How the loop uses it**

| Use | What happens | Status |
|---|---|---|
| Paper to optimizer | The planner opens a task, Devin implements it in a PR, a human reviews, and the loop picks it up | ⬜ |
| Scaffolding and tests | Faster delivery of supporting code | ⬜ |

**Metrics**

| Metric | How it is measured | What it proves |
|---|---|---|
| Idea to running experiment | Time from task opened to first verified run | Devin speeds up iteration |
| Methods delivered | New optimizers merged after review | Real capability added |
| Review pass rate | Share of Devin PRs that pass tests on the first try | Quality of generated code |

---

## Evaluation protocol (all tools)

1. Fix the problem, verifier, seeds and compute budget.
2. Run the full loop, then the same loop with one tool disabled at a time.
3. Report the metrics above from the logs (`runs/experiments.jsonl`,
   `runs/provenance.jsonl`, `runs/decisions.jsonl`), never from estimates.
4. Report results honestly, including tools that made no measurable difference.

| Configuration | Best score | Time to record | Cost (USD) | Repeated ideas |
|---|---|---|---|---|
| Full loop | | | | |
| Without Exa | | | | |
| Without past.dev | | | | |
| Without the Modal sieve | | | | |
| Heuristic instead of Claude | | | | |

## Code in this branch

| File | Purpose |
|---|---|
| `autoresearch/adapters/search.py` | Exa search with a stop loss, cache and provenance log; `NullSearch` fallback |
| `autoresearch/adapters/memory.py` | past.dev ingest/recall; `LocalMemory` JSONL fallback with the same interface |
| `autoresearch/adapters/common.py` | `.env` loading, HTTP retries with backoff, provenance log |
| `.env.example` | Which keys exist; copy it to `.env`, which is git-ignored |

```python
from autoresearch.adapters.common import load_env
from autoresearch.adapters.search import get_search
from autoresearch.adapters.memory import get_memory

load_env()
hits = get_search().search("best known packing of 26 circles sum of radii",
                           agent="scout", purpose="record audit")
memory = get_memory()
fact_id = memory.write("Record claim: ...", author="scout", kind="record")
facts = memory.read("current record", reader="planner")
```
