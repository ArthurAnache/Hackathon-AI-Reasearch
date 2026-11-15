"""The small interface a problem plug-in provides (autoloop/tao1d.py is the first one).

Module attributes:  NAME, LOWER_IS_BETTER, SEED_CELLS, SPLIT_UP, P0_SCHEDULE, LOCAL_MAX_CELLS
Functions:          score(v) -> float, vec_id(v) -> str, load_vec(path) -> array, valid_vec(v) -> bool,
                    fetch_reference() -> {ok, value (decimal string), source, candidates},
                    claim_threshold(value) -> Fraction  (a certified exact score must be strictly below it)
Executors:          LocalExecutor(workdir, max_cells), FakeExecutor(workdir, max_cells, seed), each with
                    .real, .usd_per_cpu_s, .max_cells, seed(n, budget_s, cells, seed0), refine(path, split, p0s,
                    budget_s, tag), certify(path, tag).
The loop, the ledger, the policies and the report only use these names.
"""
import importlib

PROBLEMS = {"tao-1d": "autoloop.tao1d", "tao-1a": "autoloop.tao1a", "ea-first-autocorrelation": "autoloop.ea_first_autocorr",
            "ea-third-autocorrelation": "autoloop.ea_third_autocorr"}


def load(name: str):
    if name.startswith("ea:"):                       # default pack: any frozen EinsteinArena problem (autoloop/ea_default.py)
        return importlib.import_module("autoloop.ea_default").configure(name[3:])
    return importlib.import_module(PROBLEMS[name])
