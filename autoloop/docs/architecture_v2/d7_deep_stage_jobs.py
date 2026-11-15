"""D7 (read-only replay, job accounting): the deep stage of the cold runs. Entry runs versus what followed.
usage: python -I -B d7_deep_stage_jobs.py LEDGER [LEDGER ...]
CPU is taken from the runs' own result lines (jobs x declared seconds), so failed jobs and runs that returned their parent are counted."""
import json, sys, collections
from pathlib import Path

def spearman(x, y):
    def rank(a):
        o = sorted(range(len(a)), key=lambda i: a[i]); r = [0.0] * len(a)
        for pos, i in enumerate(o): r[i] = pos + 1
        return r
    rx, ry = rank(x), rank(y); n = len(x); mx = (n + 1) / 2
    return sum((a - mx) * (b - mx) for a, b in zip(rx, ry)) / sum((a - mx) ** 2 for a in rx)

for lp in sys.argv[1:]:
    recs = [json.loads(x) for x in Path(lp).read_text().splitlines() if x.strip()]
    allc = {r["id"]: r for r in recs if r["kind"] == "certificate"}
    certs = {k: c for k, c in allc.items() if c["stage"] != "certified"}
    run = Path(lp).parent.name
    levels = sorted({c["cells"] for c in certs.values()}); top, prev = levels[-1], levels[-2]
    dec = {r["i"]: r for r in recs if r["kind"] == "decision"}; res = {r["i"]: r for r in recs if r["kind"] == "result"}
    total_cpu = sum((r.get("cpu_s") or 0) for r in res.values())
    # classify the decisions that work at the top level
    entry_cpu = rest_cpu = 0.0; entry_jobs = collections.Counter(); rest_jobs = collections.Counter(); job_s = None
    for i, d in sorted(dec.items()):
        if d["move"] not in ("refine", "refine_wide"): continue
        a = d["args"]; ids = a.get("certs") or [a.get("cert")]
        src = [allc[c] for c in ids if c in allc]
        if not src: continue
        split = a.get("split", 1); p0s = a.get("p0") if isinstance(a.get("p0"), list) else [a.get("p0")]
        dest = src[0]["cells"] * (split if split and split > 1 else 1)
        if dest != top: continue
        cpu = res[i].get("cpu_s") or 0; per = cpu / (len(src) * len(p0s)); job_s = job_s or per
        for c in src:
            (entry_jobs if split > 1 else rest_jobs)[c["root"]] += len(p0s)
        if split > 1: entry_cpu += cpu
        else: rest_cpu += cpu
    deep = entry_cpu + rest_cpu
    bprev = {}; e4 = {}; fin = {}
    for c in certs.values():
        r = c["root"]
        if c["cells"] == prev: bprev[r] = min(bprev.get(r, 9), c["score"])
        if c["cells"] == top:
            fin[r] = min(fin.get(r, 9), c["score"])
            p = c.get("params") or {}
            if p.get("split", 1) > 1 and p.get("p0") == 4.0: e4[r] = min(e4.get(r, 9), c["score"])
    roots = sorted(e4, key=e4.get); n = len(roots); best = min(fin.values()); winner = min(fin, key=fin.get)
    o_prev = sorted(bprev, key=bprev.get); pos_prev = {r: i + 1 for i, r in enumerate(o_prev)}; pos4 = {r: i + 1 for i, r in enumerate(roots)}
    wr = [r for r in roots if rest_jobs[r]]
    print(f"\n===== {run} ({top} cells): CPU-hours of the run {total_cpu/3600:.1f}; deep stage {deep/3600:.1f} ({100*deep/total_cpu:.0f}%): entry {entry_cpu/3600:.1f}, after entry {rest_cpu/3600:.1f} ({100*rest_cpu/total_cpu:.0f}% of the run); {job_s:.0f} s per job")
    print(f"  roots entered {len(entry_jobs)}, with a p0=4 entry certificate {n}, restarted {len(wr)}")
    print(f"  Spearman with the final rank at {top}: rank before the stage {spearman([bprev[r] for r in roots], [fin[r] for r in roots]):+.2f} (n={n}); "
          f"rank after one p0=4 entry run {spearman([e4[r] for r in wr], [fin[r] for r in wr]):+.2f} (n={len(wr)}, restarted roots only)")
    print(f"  the run's best certificate: rank {pos_prev[winner]} of {len(o_prev)} before the stage, rank {pos4[winner]} of {n} after one entry run")
    print(f"  pool best after the entry runs {min(min(c['score'] for c in certs.values() if c['cells']==top and (c.get('params') or {}).get('split',1)>1 and c['root']==r) for r in roots):.7f}; at the end {best:.7f}: everything after entry added {min(min(c['score'] for c in certs.values() if c['cells']==top and (c.get('params') or {}).get('split',1)>1 and c['root']==r) for r in roots) - best:.2e}")
    for m in (1, 2, 3, 5):
        keep = roots[:m]
        b = min([fin[r] for r in keep] + [e4[r] for r in roots[m:]])
        cpu = len(entry_jobs) * job_s + sum((entry_jobs[r] - 1) * job_s + rest_jobs[r] * job_s for r in keep)
        print(f"  plan m={m}: one entry run on every root, the rest only on the top {m} after it: best {b:.7f} (loss {b-best:+.2e}); CPU-h {cpu/3600:5.1f} = {100*cpu/deep:.0f}% of the stage; saves {(deep-cpu)/3600:.1f} CPU-h = {100*(deep-cpu)/total_cpu:.0f}% of the run")
