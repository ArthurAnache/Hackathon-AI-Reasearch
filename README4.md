# README4 — WS4: cloud, agent tools and sponsor integrations

Owner: Murilo. Team: The Revolutionaries, SF AI Research Hackathon (14–15 Nov 2026), Track 1.

WS4 connects the research loop to external tools, and every tool has to change a decision or a
result that shows up in the logs. Nothing here is decorative: each integration plugs into the shared
contract in `interfaces.py` (WS2), and the loop runs unchanged when no key is configured.

Every result below is labelled **verified** (we ran it), **hypothesis** or **limitation**.

Last updated: 2026-11-14.

---

## 1. What WS4 delivers

| Contract slot (`INTERFACES.md`) | WS4 implementation | What it does in the loop |
|---|---|---|
| `evidence_source.gather(problem, history)` | `integrations.evidence.ExaEvidence` | Exa literature search, only when it can change a decision: a record audit the first time a problem is seen, then idea mining each time the loop plateaus |
| `evidence_source.gather(problem, history)` | `integrations.evidence.PastDevEvidence` | Recalls what **other** agents learned from past.dev. A config another agent kept becomes an actionable hint (`{"config": {...}}`) that `FeedbackAgent`'s `EvidenceHints` strategy acts on |
| `experiment_store.append(result)` | `integrations.evidence.PastDevMirrorStore` | The local JSONL store stays the source of truth; every experiment is also mirrored to shared memory, attributed to the agent that ran it |
| — | `integrations.exa_client.search` | Drop-in replacement for the WS1 Exa client (same signature, return shape and log file), plus multi-key failover, a stop loss and a cache |
| — | `integrations.pastdev_client` | past.dev ingest/recall with multi-key failover and a same-project guard; `LocalMemory` fallback |
| — | `integrations.sponsor_metrics` | The sponsor metrics, computed from the logs |

Every `Evidence` carries `source`, `citation` (URL or memory fact id) and `author`, and
`run_experiment()` stores it in the experiment record, so each decision traces back to the search
result or the agent that informed it.

## 2. Setup

```bash
cp .env.example .env        # .env is git-ignored; fill in your keys
python tests/test_integrations.py                 # offline: fake local server, no keys, no network
python -m pytest -q                               # whole suite incl. WS2 tests
```

No extra dependency: the integrations use the standard library only (plain HTTPS). `exa-py` and
`python-dotenv` are no longer required for Exa.

### Keys

- **Several keys per service**, tried in order with automatic failover:
  `EXA_API_KEYS=murilo:key1,arthur:key2`. A single `EXA_API_KEY=...` still works. A key that fails
  authentication or runs out of credit is disabled and remembered in `.cache/integrations/`.
- **past.dev: one project for the whole team.** Different projects are different memories. The client
  binds to the first project it sees and refuses a key from another project.
- Keys are never printed or logged. Logs show the key's **name** only (for "who spent what").
- Modal containers get keys through Modal Secrets, not through `.env`.

### Cost safety (verified on the dashboards, 2026-11-14)

| Service | Protection |
|---|---|
| Exa | No card on file, so spend is capped by the prepaid balance. In code: stop loss for all keys together (`EXA_BUDGET_USD`, default $5; `EXA_MAX_CALLS`, default 200), and identical queries are cached and free |
| past.dev | Pay as you go with no card on file: "nothing is billed after the fact"; with no credits left, writes and recall pause. The plan allows 1 project, so every key shares one memory |
| Rule | Do not click "Buy credits" or "Upgrade" on either dashboard: that saves a card and enables auto top-up |

## 3. Running the loop with the integrations

```bash
# agent A: Exa + shared memory
python -m integrations.run_loop --problem 1a -n 3 --budget 8 --agent-id laptop-a --exa --pastdev
# agent B, another machine or store: shared memory only
python -m integrations.run_loop --problem 1a -n 2 --budget 8 --agent-id laptop-b --pastdev --store experiments_b.jsonl
python -m integrations.sponsor_metrics --store experiments_b.jsonl
```

Without `--exa` and `--pastdev`, `run_loop` behaves exactly like `python loop.py`.

Logs (no secrets):

| File | Content |
|---|---|
| `selection/exa_log.jsonl` | Every Exa call: query, parameters, decision served, duration, URLs, cost, key name, cached or blocked |
| `logs/pastdev_log.jsonl` | Every memory write and read: author or reader, fact ids, latency |
| `experiments.jsonl` | WS2 records; the `evidence` field lists the sources, citations and authors each experiment was given |

## 4. Results

### Verified: live run, 2026-11-14 (problem 1a, 8 s per experiment)

Run on a checkout of `feature/ws2-autoresearch-loop` plus this branch, with real Exa and past.dev keys.

1. **Agent A** (`--exa --pastdev`). The record audit queried Exa once ($0.007) and returned Tao's page
   for 1a (`teorth.github.io/optimizationproblems/constants/1a.html`) plus two record trackers. The
   findings were written to past.dev as a `literature` fact by `exa-scout`.
2. **Agent B** (`--pastdev`, separate experiment store, so it has none of A's history locally).
   - `exp_0000`: baseline, score 1.677567317.
   - `exp_0001`: proposed by the `evidence` strategy, *"suggested by past.dev (experiment-923994a447)"*,
     a fact written by **laptop-a**. Kept, score **1.666475062** (problem 1a minimizes).
3. `sponsor_metrics`: "2 experiments used a fact written by another agent, 2 kept".

This meets the past.dev requirement end to end: more than one agent writes (exa-scout, laptop-a,
laptop-b), more than one reads (laptop-a, laptop-b), and an agent used another agent's fact in a
decision that improved its score.

- **Limitation:** budgets were tiny (8 s) and the run is short. It shows the mechanism, not a record.
  Problem 1a's best known value is 1.5027; these scores are nowhere near it.
- **Limitation:** Exa results were not yet turned into config hints. They inform the rationale and the
  shared memory, but the agent only acts directly on structured `{"config": ...}` hints.

### Verified: tests

- `tests/test_integrations.py`: 7 offline tests (fake server): key failover, spend per key name,
  cache, stop loss, missing key, past.dev same-project guard, mirror store with a cross-agent hint,
  Exa trigger logic (audit once, ideas once per plateau), no secret in any log.
- On the combined checkout (WS2 branch plus this branch): **94 passed, 3 skipped**, WS2's own tests
  included, frozen files intact.

## 5. Why each sponsor tool, and how we measure it

The principle: measure each tool as an **ablation**, with the same loop, seeds and budget, tool on vs off.

| Rank | Tool | Role in the loop | Key metrics | Status |
|---|---|---|---|---|
| 1 | **Exa** | Know the true record (audit), start from published constructions, check before claiming, mine ideas on plateau | Time to record with vs without Exa warm start; best score at fixed budget; target corrections; false records prevented; ROI (Exa $ vs compute saved) | Audit and ideas ✅; warm start and pre-claim check ⬜ |
| 2 | **Modal** | Spend compute only where it pays off: sieve (successive halving) with repechage; local first, cloud when promising | Time to record laptop vs Modal; experiments per hour; cost per improvement; sieve savings vs uniform budget | WS3 |
| 3 | **past.dev** | Memory shared by agents: kept configs, failed ideas, literature; durable across restarts | Experiments informed by another agent's fact (and kept); repeated experiments avoided; recovery after restart | ✅ (live, see §4) |
| 4 | **Claude** | Proposes the next experiment; a skeptic agent tries to refute each improvement | Keep rate, Claude planner vs heuristic `FeedbackAgent`; gain per LLM dollar; false improvements caught | WS2 |
| 5 | **Devin** | Turns a technique found by Exa into a new optimizer file, reviewed by a human | Idea-to-running-experiment time; optimizers merged; review pass rate | ⬜ |

Ablation table to fill from the logs:

| Configuration | Best score | Time to record | Cost (USD) | Repeated ideas |
|---|---|---|---|---|
| Full loop | | | | |
| Without Exa | | | | |
| Without past.dev | | | | |
| Without the Modal sieve | | | | |
| Heuristic instead of Claude | | | | |

## 6. Merge notes for the team

- `integrations/exa_client.py` **replaces** the WS1 version from `ws1-selection`. It keeps the same
  `search(query, since, n, *, decision, search_type)` signature, the same return shape, the same
  `ExaUnavailable` exception and the same log file, so `selection/literature_check.py` works unchanged.
  Expect a merge conflict on that file: take this version.
- `integrations/__init__.py` is byte-identical to the WS1 one.
- `.env.example` and `.gitignore` are the union of the WS1, WS2 and WS4 versions.
- `integrations/run_loop.py` and `integrations/evidence.py` import WS2's `interfaces`, `loop` and
  `research`, so they need the WS2 branch merged. The adapters themselves have no dependency.

## 7. Status checklist

| Requirement | Status | Evidence |
|---|---|---|
| Working Exa call | ✅ verified | `selection/exa_log.jsonl`, §4 |
| past.dev: >1 agent writes, >1 reads, an agent uses another's fact | ✅ verified | `logs/pastdev_log.jsonl`, `experiments_b.jsonl`, §4 |
| `.env.example`, no credentials committed | ✅ | `.gitignore`, secret scan before each push |
| Adapter interfaces behind clean fallbacks | ✅ | the loop runs with no key (tests) |
| Visible provenance trace | ✅ | the logs in §3 and the `evidence` field of every experiment |
| Retries, failover, budget caps | ✅ | `integrations/_common.py`, `exa_client.py` |
| Coding-agent permissions and review workflow | 🟡 | rules in `CLAUDE.md` (frozen verifier, no push without Arthur's approval); Devin not configured |
| Dashboard / deployed logs | ⬜ | `python analysis.py --json` (WS2) is the starting point |
