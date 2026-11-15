"""Readable decision log for a judge: time, what the loop knew, what it chose, why, what came out.

    python -m autoloop.report autoloop/runs/NAME/ledger.jsonl [--wide | --table]
"""
import json
import sys
import textwrap


def load(path):
    return [json.loads(x) for x in open(path).read().splitlines() if x.strip()]


def fmt(x, d=10):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def args_str(move, a):
    if move == "refine":
        return f"{a['cert'][:8]} x{a['split']} p0={a['p0']}"
    if move == "certify":
        return a["cert"][:8]
    if move == "seed":
        return f"n={a['n']} {a['budget_s']:g}s"
    return a.get("reason", "") if move == "stop" else ""


def table(path):
    """Short judged table: per decision, the controlling pick, its predicted and realized gain, and shadow picks."""
    recs = load(path)
    res = {r["i"]: r for r in recs if r["kind"] == "result"}
    g = lambda x: "-" if x is None else f"{x:.2e}"  # noqa: E731
    out = [f"{'#':>3} | {'controlling pick':<44} | {'predicted':>9} | {'realized':>9} | shadow picks (policy: move, predicted)"]
    for r in recs:
        if r["kind"] != "picks":
            continue
        c = next(p for p in r["picks"] if p["controlling"])
        x = res.get(r["i"], {})
        sh = "; ".join(f"{p['policy']}: {p['move']} {'(same)' if (p['move'], p['args']) == (c['move'], c['args']) else ''}"
                       f" {g(p['predicted_gain'])}" for p in r["picks"] if not p["controlling"])
        out.append(f"{r['i']:>3} | {(c['policy'] + ': ' + c['move'] + ' ' + c['args'])[:44]:<44} | {g(c['predicted_gain']):>9} | "
                   f"{g(x.get('realized_gain')):>9} | {sh or '-'}")
    return "\n".join(out)


def report(path, wide=False):
    recs = load(path)
    t0 = next((r["t"] for r in recs if r["kind"] == "run_start"), recs[0]["t"] if recs else 0)
    res = {r["i"]: r for r in recs if r["kind"] == "result"}
    out = []
    for r in recs:
        if r["kind"] == "run_start":
            out.append(f"RUN policy={r['policy']} executor={r['executor']} caps: wall {r['config']['max_wall_s']:g} s, "
                       f"decisions {r['config']['max_decisions']}, compute USD {r['config']['max_usd']:g}, "
                       f"LLM USD {r['config']['llm_usd_cap']:g}")
            out.append(f"{'#':>3} {'t(s)':>6} | {'knew: best / cells / ref':<36} | {'move':<28} | {'policy':<16} | why -> result")
        elif r["kind"] == "note":
            out.append(f"    note: {r['text']}")
        elif r["kind"] == "decision":
            n = r["numbers"]
            knew = f"{fmt(n['best'])} / {fmt(n['best_cells'])} / {n['reference'] or '-'}"
            pol = r["policy"] + ("*" if r.get("fallback") else "")
            x = res.get(r["i"], {})
            why = r["reason"] + (f" [fallback: {r['fallback']}]" if r.get("fallback") else "")
            body = f"{why}  ->  {x.get('summary', '?')}" + (f" ({x['seconds']:g} s)" if "seconds" in x else "")
            head = (f"{r['i']:>3} {r['t'] - t0:>6.0f} | {knew:<36} | {(r['move'] + ' ' + args_str(r['move'], r['args']))[:28]:<28} | "
                    f"{pol[:16]:<16} | ")
            lines = textwrap.wrap(body, 200 if wide else 90) or [""]
            out.append(head + lines[0])
            out += [" " * len(head) + ln for ln in lines[1:]]
        elif r["kind"] == "claim":
            out.append(f"    CLAIM GATE: {r['text']}: exact {r['exact_float']:.15f} vs reference {r['reference']} "
                       f"({r['reference_source']}, read {r['reference_age_s']} s before). Not submitted.")
        elif r["kind"] == "run_end":
            out.append(f"END {r['reason']} | decisions {r['decisions']}, wall {r['wall_s']:g} s, CPU {r['cpu_s']:g} s, "
                       f"compute USD {r['usd']:g}, LLM USD {r['llm_usd']:g} | best float {fmt(r['best'], 13)}, "
                       f"best exact {fmt(r['best_exact'], 15)}, claims below reference: {r['claims_below']}")
    return "\n".join(out)


if __name__ == "__main__":
    print(table(sys.argv[1]) if "--table" in sys.argv else report(sys.argv[1], wide="--wide" in sys.argv))
