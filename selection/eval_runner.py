"""Run one EinsteinArena verifier on one solution file, in a clean process, and time it.

Usage: python -I eval_runner.py VERIFIER.py SOLUTION.json [MAX_REPEATS] [TIME_BUDGET_S]

Mirrors the server harness (web/src/app/api/evaluate/route.ts): the verifier source is prefixed with
"import json", then `evaluate(data)` is called on the submitted `solution` dict. Prints one JSON object
on stdout with the score (repr of the float, as the server prints it) and the per-call wall times.
"""

import json
import sys
import time
import traceback


def main() -> None:
    verifier_path, solution_path = sys.argv[1], sys.argv[2]
    max_repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    time_budget_s = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0

    with open(verifier_path) as f:
        source = "import json\n" + f.read()
    with open(solution_path) as f:
        raw = f.read()

    namespace: dict = {"__name__": "einstein_verifier"}
    t0 = time.perf_counter()
    exec(compile(source, verifier_path, "exec"), namespace)
    import_s = time.perf_counter() - t0
    evaluate = namespace["evaluate"]

    scores, times = [], []
    try:
        while len(times) < max_repeats:
            data = json.loads(raw)  # fresh copy: a verifier may mutate its input
            t = time.perf_counter()
            score = evaluate(data)
            times.append(time.perf_counter() - t)
            scores.append(score)
            if sum(times) > time_budget_s:
                break
    except Exception as exc:  # report verifier rejections as data, not crashes
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()[-1500:], "import_s": import_s}))
        return

    print(json.dumps({
        "ok": True,
        "score": float(scores[0]),
        "score_repr": f"{scores[0]}",
        "deterministic": len({repr(s) for s in scores}) == 1,
        "times_s": times,
        "import_s": import_s,
    }))


if __name__ == "__main__":
    main()
