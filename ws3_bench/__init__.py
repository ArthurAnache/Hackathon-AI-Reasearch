"""WS3 benchmark runner for the WS2 V2 engine (optimizer benchmarks, no evolution loop).

    python -m ws3_bench plan                       # dry run: jobs, image contents, payload sizes, cost estimate
    python -m ws3_bench run --executor local ...   # local run through the same worker (tests, small budgets)
    python -m ws3_bench run --executor modal --confirm-modal ...
    python -m ws3_bench collect --run-id ID [--from modal]   # rebuild exports from the persisted job files

Layers reused from V2, unchanged: execution.make_job / run_job / ModalExecutor, certificates.candidate_sha,
interfaces.new_record / JsonlStore, and WS1's frozen problems (pinned verifier, frozen certificate).
WS3 only adds: an allowlisted Modal image, a worker that persists every job to a Volume, and the exports.

This is an OPTIMIZER benchmark (fixed optimizer files, fixed configs, paired seeds). It does not compare
research agents; that needs both agents (WS2 evolution vs Arthur's reflection architecture) run with the
same budget, the same incumbent and the same executor.
"""
