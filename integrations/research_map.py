"""Render the research map as one static HTML page.

    python -m integrations.research_map                              # ideas + ./experiments.jsonl if present
    python -m integrations.research_map --experiments a.jsonl b.jsonl --out map.html
    python -m integrations.research_map --problem circles-rectangle
    python -m integrations.research_map --live --catalog-root ../wt-attack     # read #1 and delta live (network)
    # options: --ideas PATH  --delta X  --direction maximize|minimize
    #          --record X --record-read-at ISO   (a typed record is labelled as typed, never as read)
    #          --alias 1a=second-autocorrelation-inequality   (WS2 problem id = map slug)
    #          --label B0=scout-021   (an agent's own idea label = map id, for logs made outside the loop)
    #          --min-gain-over-delta 0.01  --refute-after 3  --min-compute 3600  --save-verdicts

The first screen is for someone who has two minutes: what is dead and why, what
was tested and what came out, where the compute went. The tables below hold the
detail: verdicts, the frontier in the order the idea chooser would try it, every
dead end with its reason, compute per idea, every experiment.

It is a pure function of its inputs (the idea store and the experiment files), so
it can be regenerated at any time. Verdicts are recomputed from the experiment
files given; an idea with no experiment in those files shows the verdict stored
in the idea store. Nothing is written except the page, unless --save-verdicts.
Each experiment file is one store: a run is named "<store label>/<experiment id>"
(the label is dropped on the page when there is a single store).

The yardstick. The record moves (the #1 of the main target changed twice on the
first day), so it is never taken from the map: pass --live (WS1's problems.catalog,
which must be importable: see --catalog-root). Without it the page says the record
was not read, and gains are still shown against the best result before each
experiment. --record types one by hand; the page then says it was typed, not read.
delta comes from --delta, --live, else from WS1's record node. The direction comes
from --direction, --live, else from the three WS1 problems this file knows.
An experiment counts for an idea only if it ran on the idea's problem (--alias maps
WS2's short ids to the map's slugs); the others are listed, never counted.
Experiment files written outside the WS2 loop are accepted too (see
verdicts.normalise_record): they name their idea in an "idea" field, and a record
without a problem is taken to be about the page's problem.
All text from the map is HTML-escaped: forum content is data.
A verdict on this page is never a record: only selection/verify_claim.py decides that.
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import ideas, verdicts
from ._common import utc_now
from .idea_chooser import IdeaChooser

MAIN = "second-autocorrelation-inequality"
UNTAGGED = "(no idea tag)"
MAX_ROWS = 300
# WS1's three frozen problems (docs/ws1_handoff.md). Any other problem needs --direction or --live.
KNOWN_DIRECTIONS = {MAIN: "maximize", "circles-rectangle": "maximize", "ring-loading-15": "maximize"}


# ------------------------------------------------------------------ inputs
def load_experiments(paths: List[str], sign: float = 1.0, default_problem: Optional[str] = None,
                     labels: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Records of every file, in order, each carrying its store label in `_store`.
    Records logged outside the WS2 loop are normalised (verdicts.normalise_record)."""
    out = []
    for p in paths:
        path = Path(p)
        label = verdicts.store_label(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue                                  # a torn last line must not hide the rest
            if isinstance(r, dict):
                out.append(dict(verdicts.normalise_record(r, sign, default_problem, labels), _store=label))
    return out


_LIVE_READER = """
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
try:                                  # WS1's client logs every call inside ITS checkout: log in ours instead
    import integrations.einstein_arena as ea
    ea.LOG_PATH = pathlib.Path(sys.argv[3])
except Exception:
    pass
from problems.catalog import load
p = load(sys.argv[2])
best = p.best_known_score(0)
print("LIVE " + json.dumps({"best": best, "delta": p.DELTA, "direction": getattr(p, "DIRECTION", "maximize")}))
"""


def reference_live(slug: str, catalog_root: Optional[str] = None) -> verdicts.Reference:
    """Live #1 and delta from WS1's catalog (problems.catalog.load(slug)): one read of the
    leaderboard. It runs in a separate process started in the catalog's checkout, because
    that checkout has its own `integrations` package (the leaderboard client), which
    cannot be imported next to this one. When the catalog falls back to its frozen
    snapshot (offline), the record is NOT used: a stale record is worse than none.
    Nothing is written in the catalog's checkout: no bytecode, and the client's request
    log goes next to our idea store."""
    root = str(Path(catalog_root or ".").resolve())
    log = ideas.default_path().parent / "arena_log.jsonl"
    try:
        # -B: read-only guest in somebody else's checkout, leave no .pyc behind
        proc = subprocess.run([sys.executable, "-B", "-c", _LIVE_READER, root, slug, str(log)], cwd=root,
                              capture_output=True, text=True, timeout=120)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("LIVE ")), None)
        if line is None:
            raise RuntimeError((proc.stderr.strip().splitlines() or ["no output"])[-1][:200])
        got = json.loads(line[5:])
    except Exception as e:                       # no catalog there, timeout, bad output
        return verdicts.Reference(problem=slug, source="live read failed (%s: %s); record not read"
                                                       % (type(e).__name__, e))
    best = got["best"]
    ref = verdicts.Reference(problem=slug, direction=got["direction"], delta=float(got["delta"]))
    if best.get("live"):
        ref.record, ref.record_read_at = float(best["value"]), str(best.get("read_at"))
        ref.source = "live #1 by %s, read %s" % (best.get("agent"), best.get("read_at"))
    else:
        ref.source = "live read failed, the catalog only has its snapshot of %s; record not read" % best.get("read_at")
    return ref


class _Nodes:
    """The read side of an IdeaStore over nodes already in memory (for the chooser)."""

    def __init__(self, nodes):
        self._nodes = list(nodes)

    def all(self):
        return list(self._nodes)


# ------------------------------------------------------------------ model
def build(nodes: List[Dict[str, Any]], records: List[Dict[str, Any]], ref: verdicts.Reference, problem: str,
          aliases: Optional[Dict[str, str]] = None, min_gain_over_delta: float = verdicts.MIN_GAIN_OVER_DELTA,
          refute_after: int = verdicts.REFUTE_AFTER, min_compute_s: float = verdicts.MIN_COMPUTE_S) -> Dict[str, Any]:
    rule = {"min_gain_over_delta": min_gain_over_delta, "refute_after": refute_after, "min_compute_s": min_compute_s}
    groups = verdicts.by_idea(records)
    known = {n["id"] for n in nodes}
    views, foreign = [], []
    for n in nodes:
        if n["problem"] != problem:
            continue
        foreign += [(n["id"], r) for r in verdicts.records_for(n, groups.get(n["id"], []), aliases)[1]]
        if n["kind"] not in ideas.TESTABLE:
            views.append({"node": n, "verdict": None, "rows": []})
            continue
        rows = verdicts.observe(n, records, ref, aliases, **rule)
        verdict = verdicts.summarise(n, rows, ref.as_dict(), **rule) if rows else n.get("verdict")
        views.append({"node": ideas.with_verdict(n, verdict), "verdict": verdict,
                      "rows": verdict["runs"] if verdict else []})
    ranking = IdeaChooser(_Nodes([v["node"] for v in views]), lambda n: ref, need_runner=False, aliases=aliases,
                          min_gain_over_delta=min_gain_over_delta).ranking(records)
    by_id = {v["node"]["id"]: v for v in views}

    compute = [{"id": v["node"]["id"], "label": v["node"]["content"], "experiments": len(v["rows"]),
                "seconds": sum(x["duration_s"] for x in v["rows"])} for v in views if v["rows"]]
    unknown = {i: rs for i, rs in groups.items() if i is not None and i not in known}
    extra = [(UNTAGGED, "experiments run without an idea (any problem)", groups.get(None, []))]
    extra += [("%s (not on the map)" % i, "idea tag unknown to the map", rs) for i, rs in sorted(unknown.items())]
    extra.append(("(other problem)", "tagged with an idea of this problem but run on another problem",
                  [r for _, r in foreign]))
    for label_id, label, rs in extra:
        if rs:
            compute.append({"id": label_id, "label": label, "experiments": len(rs),
                            "seconds": sum(float(r.get("duration_s") or 0) for r in rs)})
    compute.sort(key=lambda c: -c["seconds"])

    experiments = [dict(x, idea=v["node"]["id"]) for v in views for x in v["rows"]]
    stores = {x["store"] for x in experiments}
    return {
        "problem": problem, "ref": ref, "generated": utc_now(), "rule": rule,
        "single_store": len(stores) <= 1,
        "tested": [v for v in views if v["verdict"]],
        "frontier": [{"rank": i + 1, **r, "view": by_id[r["id"]]} for i, r in enumerate(ranking)],
        "dead": [v for v in views if ideas.is_dead(v["node"])],
        "context": [v for v in views if v["node"]["kind"] not in ideas.TESTABLE],
        "compute": compute, "experiments": experiments,
        "unknown_tags": sorted(unknown),
        "foreign": [(i, str(r.get("experiment_id")), r.get("problem")) for i, r in foreign],
        "other_problems": Counter(n["problem"] for n in nodes if n["problem"] != problem),
        "counts": {"nodes": len(views), "experiments": sum(c["experiments"] for c in compute),
                   "tagged": len(experiments), "compute_s": sum(c["seconds"] for c in compute)},
    }


# ------------------------------------------------------------------ html
def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _num(x: Optional[float], digits: int = 3, signed: bool = True) -> str:
    if x is None:
        return "–"
    return ("%+.*g" if signed else "%.*g") % (digits, x)


def _delta(x: Optional[float]) -> str:
    return "–" if x is None else "%s δ" % _num(x)


def _secs(s: float) -> str:
    if s >= 3600:
        return "%.1f h" % (s / 3600)
    if s >= 60:
        return "%.1f min" % (s / 60)
    return "%.0f s" % s


def _short(text: str, n: int = 105) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1].rsplit(" ", 1)[0] + "…"


_BADGE = {   # status -> (css class, icon, label): state is never shown by colour alone
    "reproduced": ("good", "✓", "reproduced by us"), "refuted": ("critical", "✕", "refuted by us"),
    "inconclusive": ("warning", "…", "inconclusive"), "verified": ("neutral", "◆", "verified by WS1"),
    "claim": ("neutral", "○", "claim, untested by us"), "idea": ("neutral", "○", "untested idea"),
}


def _badge(status: str, label: Optional[str] = None) -> str:
    cls, icon, text = _BADGE.get(status, ("neutral", "○", status))
    return '<span class="badge %s"><span aria-hidden="true">%s</span> %s</span>' % (cls, icon, _e(label or text))


def _idea_cell(node: Dict[str, Any]) -> str:
    cites = "".join('<span class="cite">%s</span>' % _e(c) for c in ["by " + node["author"]] + node["citations"])
    return '<div class="idea"><code>%s</code> %s</div><div class="cites">%s</div>' % (
        _e(node["id"]), _e(node["content"]), cites)


def _table(head: List[str], rows: List[List[str]], empty: str, numeric: tuple = ()) -> str:
    if not rows:
        return '<p class="empty">%s</p>' % _e(empty)
    th = "".join('<th%s>%s</th>' % (' class="num"' if i in numeric else "", _e(h)) for i, h in enumerate(head))
    body = "".join("<tr>%s</tr>" % "".join(
        '<td%s>%s</td>' % (' class="num"' if i in numeric else "", c) for i, c in enumerate(r)) for r in rows)
    return '<div class="scroll"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (th, body)


def _why_dead(v: Dict[str, Any], run_id) -> tuple:
    """(who says it, why) for one dead end, in plain words."""
    n, vd = v["node"], v["verdict"] or {}
    gain = _delta(vd.get("best_gain_over_delta"))
    if n["kind"] != "dead_end":
        against = vd.get("contradicting") or []
        return (_badge("refuted"), "we ran it %d times (%s of compute) and it never gained: best %s"
                % (len(against), _secs(vd.get("negative_compute_s") or 0), gain))
    if n["status"] == "reproduced":
        return (_badge("reproduced", "confirmed by us"),
                "the source reported it does not work, and our %d runs agree: best %s" % (vd.get("trials", 0), gain))
    measured = "according to" in n["source"]
    who = _badge(n["claimed_status"], ("measured by " if measured else "reported by ") + n["author"])
    why = "%s that it does not work" % ("the scout measured" if measured else "the source reports")
    if vd.get("experiments"):
        why += "; our %d runs so far: best %s" % (vd.get("trials", 0), gain)
    return who, why


CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--raised:#f4f3f0;--line:#e3e2dd;--text:#0b0b0b;--text2:#52514e;
--muted:#6f6e69;--bar:#2a78d6;--good:#0b7d0b;--warning:#8a5a00;--critical:#b42d2d;--neutral:#52514e;
--good-bg:#e4f4e4;--warning-bg:#fdf0d0;--critical-bg:#fbe5e5;--neutral-bg:#ecebe7}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--surface:#1a1a19;--raised:#242423;--line:#383835;
--text:#f0efec;--text2:#c3c2b7;--muted:#9c9b92;--bar:#3987e5;--good:#5fce5f;--warning:#fab219;--critical:#f08a8a;
--neutral:#c3c2b7;--good-bg:#17301a;--warning-bg:#3a2e10;--critical-bg:#3b1c1c;--neutral-bg:#2e2e2c}}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;line-height:1.2;margin:0 0 2px;text-wrap:balance}
h2{font-size:18px;margin:40px 0 4px}
h3{font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--text2);margin:0 0 2px;font-weight:600}
p{margin:6px 0}.sub,.note{color:var(--text2)}.sub{font-size:14px}.note{font-size:13px;max-width:78ch}
.lede{font-size:16px;max-width:75ch;margin:2px 0 4px}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--raised);padding:1px 5px;border-radius:4px}
.glance{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin:16px 0 6px}
.card{background:var(--raised);border-radius:12px;padding:14px 16px;min-width:0}
.card code{background:var(--surface)}
.big{font-size:30px;line-height:1.1;font-weight:700;font-variant-numeric:tabular-nums;margin:2px 0 0}
.big small{font-size:14px;font-weight:400;color:var(--text2)}
.card ul{list-style:none;margin:10px 0 0;padding:0}
.card li{padding:7px 0;border-top:1px solid var(--line);font-size:14px;line-height:1.4}
.card li .why{display:block;color:var(--text2);font-size:13px}
.card .more{color:var(--muted);font-size:13px;margin-top:8px}
.scroll{overflow-x:auto;margin-top:10px}
table{border-collapse:collapse;width:100%;font-size:14px}
th{text-align:left;font-weight:600;color:var(--text2);font-size:12px;letter-spacing:.03em;text-transform:uppercase;
border-bottom:1px solid var(--line);padding:6px 10px 6px 0;white-space:nowrap}
td{border-bottom:1px solid var(--line);padding:9px 10px 9px 0;vertical-align:top}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.idea{max-width:62ch}.cites{margin-top:3px}
.cite{display:inline-block;color:var(--muted);font-size:12px;margin-right:8px}
.badge{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12px;font-weight:600;white-space:nowrap}
.badge.good{color:var(--good);background:var(--good-bg)}.badge.warning{color:var(--warning);background:var(--warning-bg)}
.badge.critical{color:var(--critical);background:var(--critical-bg)}.badge.neutral{color:var(--neutral);background:var(--neutral-bg)}
.empty{color:var(--muted);font-style:italic}
.bars{margin-top:12px;display:grid;grid-template-columns:minmax(90px,max-content) 1fr max-content;gap:7px 12px;align-items:center}
.bars .name{font-size:13px;white-space:nowrap}.bars .val{font-size:13px;color:var(--text2);font-variant-numeric:tabular-nums}
.track{height:10px;min-width:40px}.bar{height:10px;background:var(--bar);border-radius:0 4px 4px 0;min-width:2px}
.flag{border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:12px 0;background:var(--raised)}
footer{margin-top:48px;color:var(--muted);font-size:13px}
"""


def _bars(rows: List[Dict[str, Any]], total: float) -> str:
    top = max(x["seconds"] for x in rows) or 1.0
    out = ['<div class="bars" role="table" aria-label="Compute per idea">']
    for x in rows:
        share = 100 * x["seconds"] / (total or 1.0)
        tip = "%s: %s over %d experiments (%.0f%% of the total). %s" % (
            x["id"], _secs(x["seconds"]), x["experiments"], share, x["label"])
        out.append('<div class="name" role="cell"><code>%s</code></div>'
                   '<div class="track" role="cell" title="%s"><div class="bar" style="width:%.2f%%"></div></div>'
                   '<div class="val" role="cell">%s · %d %s · %s</div>'
                   % (_e(x["id"]), _e(tip), 100 * x["seconds"] / top, _e(_secs(x["seconds"])), x["experiments"],
                      "run" if x["experiments"] == 1 else "runs",
                      "&lt;1%" if 0 < share < 0.5 else "%.0f%%" % share))
    out.append("</div>")
    return "".join(out)


def render(m: Dict[str, Any]) -> str:
    ref: verdicts.Reference = m["ref"]
    rule = m["rule"]

    def run_id(key: str) -> str:               # one store: the label adds nothing for the reader
        return key.split("/", 1)[1] if m["single_store"] and "/" in key else key

    def exp_list(keys: List[str], limit: int = 8) -> str:
        if not keys:
            return "–"
        shown = ", ".join("<code>%s</code>" % _e(run_id(k)) for k in keys[:limit])
        return shown + (" +%d more" % (len(keys) - limit) if len(keys) > limit else "")

    out: List[str] = []
    w = out.append
    w('<!doctype html><html lang="en"><head><meta charset="utf-8">'
      '<meta name="viewport" content="width=device-width,initial-scale=1">'
      "<title>Research map</title><style>%s</style></head><body><main>" % CSS)
    w("<h1>Research map: %s</h1>" % _e(m["problem"]))
    w('<p class="lede">Every idea we know for this problem, what happened when we tried it, and what it cost. '
      "The search works on ideas, not on settings: an idea that failed is remembered and never tried again.</p>")
    delta = _e("%g" % ref.delta if ref.delta else "unknown")
    if ref.record is None:
        w('<p class="sub">Goal: %s the score. One δ = <b>%s</b> is the smallest improvement that counts as a new '
          "record. <b>The record was not read</b>: it moves, so this page never copies it from a note. Regenerate "
          "with <code>--live</code> to see scores against the live #1 (%s). Generated %s.</p>"
          % (_e(ref.direction), delta, _e(ref.source), _e(m["generated"])))
    else:
        target = None if ref.delta is None else ref.record + ref.sign * ref.delta
        w('<p class="sub">Goal: %s the score. Record to beat <b>%s</b>; one δ = <b>%s</b> is the smallest '
          "improvement that counts, so the target is %s <b>%s</b> (%s). Generated %s.</p>"
          % (_e(ref.direction), _e(repr(ref.record)), delta, "≤" if ref.direction == "minimize" else "≥",
             _e(repr(target) if target is not None else "unknown"), _e(ref.source), _e(m["generated"])))

    # ---- one screen: dead ends, what we tested, where the compute went
    c = m["counts"]
    w('<div class="glance">')
    w('<section class="card"><h3>Dead ends</h3><p class="big">%d <small>ideas we will not repeat</small></p><ul>'
      % len(m["dead"]))
    ours_first = sorted(m["dead"], key=lambda v: (v["verdict"] is None, v["node"]["kind"] == "dead_end"))
    for v in ours_first[:4]:
        w('<li><code>%s</code> %s<span class="why">Why: %s.</span></li>'
          % (_e(v["node"]["id"]), _e(_short(v["node"]["content"], 90)), _why_dead(v, run_id)[1]))
    w("</ul>%s</section>" % ('<p class="more">+ %d more, each with its reason, below.</p>' % (len(m["dead"]) - 4)
                             if len(m["dead"]) > 4 else ""))
    w('<section class="card"><h3>Tested by us</h3><p class="big">%d <small>ideas, %d experiments</small></p><ul>'
      % (len(m["tested"]), c["tagged"]))
    ranked = sorted(m["tested"], key=lambda v: -(v["verdict"].get("best_gain_over_delta") or -1e300))
    for v in ranked[:4]:
        vd = v["verdict"]
        w('<li><code>%s</code> %s<span class="why">%s best gain %s in %d %s.</span></li>'
          % (_e(v["node"]["id"]), _e(_short(v["node"]["content"], 80)), _badge(vd["status"]),
             _delta(vd.get("best_gain_over_delta")), vd.get("trials", 0), "run" if vd.get("trials") == 1 else "runs"))
    if not m["tested"]:
        w('<li class="empty">No experiment carries an idea yet.</li>')
    w("</ul>")
    nxt = [f for f in m["frontier"] if f["pulls"] == 0][:2]
    w('<p class="more">%d ideas still open. Next in line: %s.</p></section>'
      % (len(m["frontier"]), ", ".join("<code>%s</code> %s" % (_e(f["id"]), _e(_short(f["node"]["content"], 60)))
                                       for f in nxt) or "none untried"))
    w('<section class="card"><h3>Where the compute went</h3><p class="big">%s <small>over %d experiments</small></p>'
      % (_e(_secs(c["compute_s"])), c["experiments"]))
    w(_bars(m["compute"][:6], c["compute_s"]) if m["compute"] else '<p class="empty">No experiment yet.</p>')
    w("</section></div>")
    w('<p class="note">A verdict is what our own experiments say about an idea. It is never a record: a record '
      "exists only when the referee's <code>selection/verify_claim.py</code> prints <code>beats_record</code>.</p>")

    candidates = [(v["node"]["id"], x) for v in m["tested"] for x in (v["verdict"].get("record_candidates") or [])]
    if candidates:
        w('<div class="flag"><b>Send to the referee:</b> %s scored at least one δ above the record shown on this '
          "page (%s). This is not a record until <code>selection/verify_claim.py</code> says so.</div>"
          % (", ".join("<code>%s</code> (idea <code>%s</code>)" % (_e(run_id(x)), _e(i)) for i, x in candidates),
             _e(ref.source)))
    if m["unknown_tags"]:
        w('<div class="flag">Experiments carry idea tags that are not on the map: %s. Import those ideas to '
          "see them here.</div>" % ", ".join("<code>%s</code>" % _e(i) for i in m["unknown_tags"]))
    if m["foreign"]:
        w('<div class="flag">Not counted: %d experiments are tagged with an idea of this problem but ran on '
          "another problem: %s. If it is the same problem under another name, pass <code>--alias</code>.</div>"
          % (len(m["foreign"]), ", ".join("<code>%s</code> (idea <code>%s</code>, problem <code>%s</code>)"
                                         % (_e(x), _e(i), _e(p)) for i, x, p in m["foreign"][:12])))

    # ---- verdicts
    w("<h2>Verdicts: what our experiments say</h2>")
    w('<p class="note">A run counts in favour of an idea when it gains at least %g δ over the best we had: the '
      "record, the other ideas, and the score the run started from (for a claim with a claimed value: when our "
      "score reaches that value). An idea is decided against only after %d runs that neither gained nor improved "
      "on the idea itself, adding up to its fair budget (%s unless its runner says otherwise). The referee column "
      "is filled only from the referee's own file, <code>selection/claims.jsonl</code>.</p>"
      % (rule["min_gain_over_delta"], rule["refute_after"], _secs(rule["min_compute_s"])))
    rows = []
    for v in ranked:
        n, vd = v["node"], v["verdict"]
        has_delta = (vd.get("reference") or {}).get("delta")
        gain = _delta(vd.get("best_gain_over_delta")) if has_delta else _num(vd.get("best_gain"))
        if vd.get("criterion") == "claimed_value":
            gain = "reached %s of claimed %s" % (_e(repr(vd.get("best_score"))), _e(repr(vd.get("claimed_value"))))
        rulings = "<br>".join("%s (%s vs record %s read %s)" % (
            _e(r["verdict"]), _delta(r.get("gain_over_delta")), _e(repr(r.get("record_value"))),
            _e(r.get("record_read_at"))) for r in n.get("referee") or []) or "–"
        retest = n.get("retest")
        note = '<div class="cites"><span class="cite">reopened by %s: %s</span></div>' % (
            _e(retest.get("by")), _e(retest.get("reason"))) if retest else ""
        rows.append([_idea_cell(n) + note, _e(n["kind"].replace("_", " ")),
                     _badge(n["claimed_status"]) + " → " + _badge(vd["status"]),
                     exp_list(vd.get("experiments") or []), gain,
                     _delta(vd.get("best_vs_record_over_delta")),
                     "%d (%d failed)" % (vd.get("trials", 0), vd.get("failed", 0)),
                     "%s (%s against, of %s)" % (_secs(vd.get("compute_s") or 0),
                                                 _secs(vd.get("negative_compute_s") or 0),
                                                 _secs(vd.get("fair_budget_s") or rule["min_compute_s"])),
                     rulings])
    w(_table(["Idea", "Kind", "Claimed → our verdict", "Experiments", "Best gain", "Best vs record", "Runs",
              "Compute", "Referee"],
             rows, "No experiment carries an idea tag yet. Run the loop with --idea or --ideas.", numeric=(4, 5, 6, 7)))

    # ---- frontier
    w("<h2>Frontier: what to try next</h2>")
    w('<p class="note">Live ideas in the order the idea chooser ranks them (ideas never tried first, then by '
      "upper confidence bound on the reward our experiments returned). Dead ends are never listed. The chooser "
      "only runs ideas that have a runner.</p>")
    rows = []
    for f in m["frontier"]:
        n = f["node"]
        runner = n.get("runner")
        rows.append([str(f["rank"]), _idea_cell(n), _e(n["kind"].replace("_", " ")), _badge(n["status"]),
                     "<code>%s</code>" % _e(runner.get("optimizer") or "default optimizer") if runner
                     else '<span class="empty">no runner yet</span>',
                     str(f["pulls"]), _e(f["why"])])
    w(_table(["#", "Idea", "Kind", "Status", "Runner", "Runs", "Why this rank"], rows,
             "Nothing left on the frontier.", numeric=(0, 5)))

    # ---- dead ends
    w("<h2>Dead ends, and why</h2>")
    w('<p class="note">Never proposed again. A dead end is either reported by its source (the forum digest, a '
      "scout's measurement) or an idea our own experiments refuted.</p>")
    rows = []
    for v in ours_first:
        who, why = _why_dead(v, run_id)
        vd = v["verdict"] or {}
        runs = exp_list(vd.get("experiments") or [])
        rows.append([_idea_cell(v["node"]), who, why + ("" if runs == "–" else "<br>runs: " + runs),
                     _secs(vd.get("compute_s") or 0) if vd else "–"])
    w(_table(["Dead end", "Who says", "Why", "Compute"], rows, "No dead end recorded.", numeric=(3,)))

    # ---- compute
    w("<h2>Compute spent per idea</h2>")
    if m["compute"]:
        w('<p class="note">Wall-clock seconds of every experiment run for the idea, failed runs included.</p>')
        w(_bars(m["compute"], c["compute_s"]))
    else:
        w('<p class="empty">No experiment yet.</p>')

    # ---- experiments
    w("<h2>Experiments, by idea</h2>")
    shown = m["experiments"][-MAX_ROWS:]
    if len(m["experiments"]) > len(shown):
        w('<p class="note">Showing the last %d of %d experiments.</p>' % (len(shown), len(m["experiments"])))
    rows = [["<code>%s</code>" % _e(run_id(x["key"])), "<code>%s</code>" % _e(x["idea"]), _e(x["status"]),
             _e(repr(x["score"]) if x["score"] is not None else "–"),
             _delta(x["gain_over_delta"]) if x["gain_over_delta"] is not None else _num(x["gain"]),
             _e(x["bar_source"] or "–"), _delta(x["vs_record_over_delta"]), _e(x["evidence"]),
             _secs(x["duration_s"])] for x in shown]
    w(_table(["Experiment", "Idea", "Loop status", "Score", "Gain", "Measured against", "Vs record", "Evidence",
              "Time"], rows, "No experiment carries an idea tag yet.", numeric=(3, 4, 6, 8)))

    # ---- context
    w("<h2>Records and facts</h2>")
    w('<p class="note">Background written by the sources. A record quoted here is what its author saw on that '
      "day, not the record to beat.</p>")
    rows = [[_idea_cell(v["node"]), _e(v["node"]["kind"]), _badge(v["node"]["status"]), _e(v["node"]["author"])]
            for v in m["context"]]
    w(_table(["Node", "Kind", "Status", "Written by"], rows, "None."))

    other = ", ".join("%s (%d nodes)" % (_e(p), k) for p, k in sorted(m["other_problems"].items()))
    w("<footer>%s Regenerate: <code>python -m integrations.research_map</code>.</footer>"
      % ("Other problems on the map: %s; use <code>--problem</code>." % other if other else ""))
    w("</main></body></html>")
    return "".join(out)


# ------------------------------------------------------------------ cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ideas", help="idea store (default: %s)" % ideas.default_path())
    ap.add_argument("--experiments", nargs="*", help="experiment JSONL files (default: ./experiments.jsonl if present)")
    ap.add_argument("--problem", help="problem slug (default: %s if on the map)" % MAIN)
    ap.add_argument("--out", help="output file (default: next to the idea store, index.html)")
    ap.add_argument("--record", type=float, help="a record typed by hand (shown as typed, never as read)")
    ap.add_argument("--record-read-at", help="when the typed record was read from the leaderboard (ISO time)")
    ap.add_argument("--delta", type=float)
    ap.add_argument("--direction", choices=("maximize", "minimize"))
    ap.add_argument("--live", action="store_true", help="read #1 and delta from problems.catalog (network)")
    ap.add_argument("--catalog-root", help="a checkout that contains problems/catalog.py, for --live")
    ap.add_argument("--alias", action="append", default=[], metavar="ID=SLUG",
                    help="experiment records name the problem ID, the map names it SLUG (repeatable)")
    ap.add_argument("--label", action="append", default=[], metavar="LABEL=ID",
                    help="an experiment log calls an idea LABEL, the map calls it ID (repeatable)")
    ap.add_argument("--min-gain-over-delta", type=float, default=verdicts.MIN_GAIN_OVER_DELTA)
    ap.add_argument("--refute-after", type=int, default=verdicts.REFUTE_AFTER)
    ap.add_argument("--min-compute", type=float, default=verdicts.MIN_COMPUTE_S,
                    help="seconds of negative runs before an idea may be refuted (a runner can set its own)")
    ap.add_argument("--save-verdicts", action="store_true", help="also write the recomputed verdicts to the idea store")
    a = ap.parse_args(argv)

    store = ideas.IdeaStore(a.ideas)
    nodes = store.all()
    if not nodes:
        print("no ideas in %s: run python -m integrations.import_knowledge first" % store.path, file=sys.stderr)
        return 1
    problems = Counter(n["problem"] for n in nodes)
    problem = a.problem or (MAIN if MAIN in problems else problems.most_common(1)[0][0])
    if problem not in problems:
        ap.error("no node for problem %r; on the map: %s" % (problem, ", ".join(sorted(problems))))

    paths = a.experiments
    if paths is None:
        paths = [p for p in (str(Path("experiments.jsonl")),) if Path(p).exists()]

    ref = verdicts.reference_from_nodes(nodes, problem)
    direction_known = problem in KNOWN_DIRECTIONS
    ref.direction = KNOWN_DIRECTIONS.get(problem, "maximize")
    if a.live:
        live = reference_live(problem, a.catalog_root)
        if live.delta is None:                     # the read failed: keep the offline delta, say why
            ref.source = "%s; %s" % (live.source, ref.source)
        else:
            ref, direction_known = live, True
    if a.direction:
        ref.direction, direction_known = a.direction, True
    if not direction_known and paths:              # every sign on the page would be a guess
        ap.error("is %s maximized or minimized? pass --direction, or --live" % problem)
    if a.delta is not None:
        ref.delta, ref.source = a.delta, ref.source + "; delta given on the command line"
    if a.record is not None:
        ref.record = a.record
        ref.record_read_at = a.record_read_at or "never (typed by hand)"
        ref.source = ("record TYPED on the command line, said to be read %s" % a.record_read_at
                      if a.record_read_at else "record TYPED on the command line, not read: it may be stale")
    aliases, labels = {}, {}
    for flag, items, target in (("--alias", a.alias, aliases), ("--label", a.label, labels)):
        for item in items:
            if "=" not in item:
                ap.error("%s expects NAME=VALUE, got %r" % (flag, item))
            k, v = item.split("=", 1)
            target[k] = v
    records = load_experiments(paths, ref.sign, problem, labels)

    rule = {"min_gain_over_delta": a.min_gain_over_delta, "refute_after": a.refute_after,
            "min_compute_s": a.min_compute}
    if a.save_verdicts:
        mine = [r for r in records if ideas.same_problem(problem, r.get("problem"), aliases)]
        verdicts.update_all(store, mine, lambda n: ref, aliases=aliases, **rule)
        nodes = store.all()
    model = build(nodes, records, ref, problem, aliases=aliases, **rule)
    out = Path(a.out) if a.out else store.path.parent / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(model), encoding="utf-8")
    c = model["counts"]
    print("research map for %s: %d nodes, %d on the frontier, %d dead ends, %d ideas tested, "
          "%d experiments (%d tagged), %s of compute -> %s"
          % (problem, c["nodes"], len(model["frontier"]), len(model["dead"]), len(model["tested"]),
             c["experiments"], c["tagged"], _secs(c["compute_s"]), out.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
