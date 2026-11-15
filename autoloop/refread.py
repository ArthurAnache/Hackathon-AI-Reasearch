"""Live reference reader for a Tao constant (teorth/optimizationproblems). Fails CLOSED.

Sources, all read live, all treated as data (only numbers are extracted, nothing is followed):
1. the constant's page, "Known upper bounds" table;
2. every open pull request about the constant (R11): title starting with its id ("1d") or naming C_<id>, or a changed
   file constants/<id>.md. Candidates (any number of decimals, in the plausible range): numbers in the title, in
   bold, and numbers stated as a bound in the body (right after <=, \\le, \\leq, the "less-or-equal" sign, or "bound").
The reference is the smallest candidate in the plausible range (the most demanding one). Fails CLOSED (ok=False, so
the claim gate cannot pass) when: the page or the pull-request list cannot be read; the list may be cut (as many
entries as the limit); or a pull request about the constant whose title claims a result (bound, improve, record,
certificate, construction, better, new) states no readable number (e.g. a file only). Editorial PRs are skipped.
Offline cases: Lupa's wt-referee checks/refread_offline.py (A-J).
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import urllib.request
from fractions import Fraction

from .safety import clean_env, safe_err

REPO = "teorth/optimizationproblems"
DECIMAL = re.compile(r"(?<![0-9.])\d+\.\d+(?![0-9])")
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
STATED = re.compile(r"(?:<=|\u2264|\\leq?\b|\bbound(?:\s+(?:is|of))?:?)\s*\$?\s*((?<![0-9.])\d+\.\d+)(?![0-9])", re.I)
LIMIT = 1000


def read(constant: str, page: str, num: re.Pattern, lo: float, hi: float, gh_fetch=None, page_fetch=None) -> dict:
    cands, errors = [], []
    try:
        raw = (page_fetch or (lambda: urllib.request.urlopen(page, timeout=30).read().decode("utf-8", "replace")))()
        txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))
        if "Known upper bounds" not in txt:
            raise ValueError("page has no 'Known upper bounds' section")
        up = txt.split("Known upper bounds", 1)[1].split("Known lower bounds", 1)[0]
        cands += [dict(value=x, source=page, where="page table") for x in DECIMAL.findall(up)]
    except Exception as e:  # noqa: BLE001
        errors.append("page: " + safe_err(repr(e)))
    try:
        if gh_fetch:
            prs = gh_fetch()
        else:
            p = subprocess.run(["gh", "pr", "list", "--repo", REPO, "--state", "open", "--limit", str(LIMIT),
                                "--json", "number,title,body,files"], capture_output=True, text=True, timeout=90,
                               env=clean_env(allow=("GH_TOKEN", "GITHUB_TOKEN", "GH_CONFIG_DIR")))
            if p.returncode != 0:
                raise RuntimeError("gh exit %d: %s" % (p.returncode, safe_err(p.stderr)))
            prs = json.loads(p.stdout)
        if not isinstance(prs, list):
            raise ValueError("unexpected gh output")
        limit = LIMIT if not gh_fetch else 300
        if len(prs) >= limit:
            raise ValueError(f"{len(prs)} pull requests returned, the limit: the list may be cut")
        cid = re.escape(constant)
        for pr in prs:
            t, body = pr.get("title", ""), pr.get("body") or ""
            paths = [f.get("path", "") for f in pr.get("files") or [] if isinstance(f, dict)]
            about = (re.match(rf"\s*{cid}\b", t) or re.search(rf"C_\{{?{cid}\b", t)
                     or any(re.search(rf"(^|/){cid}\.md$", x) for x in paths))
            if not about:
                continue
            url = f"https://github.com/{REPO}/pull/{pr.get('number', '?')}"
            found = [dict(value=x, source=url, where="title", title=t[:120]) for x in DECIMAL.findall(t)]
            found += [dict(value=x, source=url, where="body (bold)", title=t[:120]) for b in BOLD.findall(body)
                      for x in DECIMAL.findall(b)]
            found += [dict(value=x, source=url, where="body (stated bound)", title=t[:120]) for x in STATED.findall(body)]
            claims = re.search(r"bound|improv|record|certificate|construction|better|new", t, re.I)
            if claims and not [c for c in found if lo < float(c["value"]) < hi]:
                raise ValueError(f"pull request {url} is about {constant} but states no readable value")
            cands += found
    except Exception as e:  # noqa: BLE001
        errors.append("pull requests: " + safe_err(repr(e)))
    vals = [c for c in cands if lo < float(c["value"]) < hi]
    if errors or not vals:
        return dict(ok=False, errors=errors or ["no candidate value found"], n_candidates=len(vals))
    vals.sort(key=lambda c: Fraction(c["value"]))
    best = vals[0]
    return dict(ok=True, value=best["value"], source=best["source"], where=best["where"], n_candidates=len(vals),
                candidates=vals[:5])
