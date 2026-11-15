"""Import idea nodes (WS1's docs/ws1_knowledge.jsonl) into the research map and shared memory.

    # default: WS1's file, read-only from git, nothing is checked out. The only TRUSTED source.
    python -m integrations.import_knowledge
    # same thing, spelled out
    python -m integrations.import_knowledge --git-ref origin/ws1-selection --git-path docs/ws1_knowledge.jsonl
    # a scout's file: it can add ideas and refresh the scout's own, nothing else
    python -m integrations.import_knowledge ../wt-scout/analysis/ideas.jsonl --owner sonar-scout
    # any other file: it can only add new ideas (this is the default for a path or another git ref)
    python -m integrations.import_knowledge some_ideas.jsonl
    # a local copy of WS1's own file: say so explicitly
    python -m integrations.import_knowledge ws1_knowledge.jsonl --trusted
    # options: --dry-run  --memory auto|local|none  --scope import-research-map  --ideas PATH  --problem SLUG

Each line is one node (see integrations.ideas for the schema). A node goes to two places:
- the local idea store (the source of truth of the map, integrations.ideas.IdeaStore);
- shared memory, as one fact of kind "idea" with the stable id "idea-<id>".

Guarantees, following integrations.import_results:
- Idempotent: re-importing changes nothing (store: "unchanged"; memory: same fact
  id, replaced in place). What we learned ourselves survives a re-import: verdicts,
  runners and referee rulings are never taken from a file, whatever the file says.
- Trust is explicit. Only the default source (origin/ws1-selection:docs/ws1_knowledge.jsonl)
  is trusted; a local path or another git ref is UNTRUSTED unless --trusted is typed.
  Nodes written by a trusted import are marked, and no untrusted file can change
  them afterwards, whatever --owner it declares.
- Untrusted files can only ADD ideas: a line whose id
  already exists is skipped, so it cannot rewrite a WS1 node or revive a dead end.
  "verified" is WS1's word: a line of an untrusted file that says "verified" (a scout
  that measured something itself) is imported as "claim", and its source field
  records who said verified. --owner NAME (implies --untrusted) also lets
  the file refresh the nodes NAME wrote before (a scout that rebuilds its file); every
  line is then attributed to NAME, so a file cannot speak in somebody else's name.
- Own scope (default "import-research-map"). Like every import-* scope it is filtered
  by tag only, so it works from any machine, and comparison runs that use another
  --memory-scope never see idea facts. Under another --scope the facts get their own
  ids: a second scope never moves the facts of the first.
- One pass, line by line: a bad line is skipped and reported, the good lines of the
  same file are still imported, and the exit code is then 1. A single line is never
  half-imported.
- Forum text stays data: braces and key=value tokens are neutralised in the memory
  fact, so it can never be parsed as a config hint or as a WS2 tag, and nothing in
  it is executed or followed.

--memory auto   past.dev when a key is configured, else the local file (default)
--memory local  the local file only: no network call
--memory none   only the idea store
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import ideas
from ._common import REPO_ROOT
from .pastdev_client import LocalMemory, get_memory

DEFAULT_REF = "origin/ws1-selection"
DEFAULT_PATH = "docs/ws1_knowledge.jsonl"
SCOPE = ideas.SCOPE


def load_rows(path: Optional[str] = None, git_ref: Optional[str] = None,
              git_path: Optional[str] = None) -> List[Tuple[int, Any]]:
    """[(line number, parsed JSON or the ValueError)] for every non-empty line."""
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = subprocess.run(["git", "show", "%s:%s" % (git_ref, git_path)], cwd=REPO_ROOT,
                              check=True, capture_output=True, text=True).stdout
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append((n, json.loads(line)))
        except ValueError as e:
            rows.append((n, e))
    return rows


def import_nodes(rows: List[Tuple[int, Any]], store: Optional[ideas.IdeaStore], memory=None,
                 scope: str = SCOPE, trusted: bool = True, problem: Optional[str] = None,
                 out=print, owner: Optional[str] = None) -> Dict[str, Any]:
    """One pass: each row is validated and, if valid, written; a bad row is skipped
    and reported. store=None is a dry run. trusted=False: rows can only add new ideas.
    Returns {"new", "updated", "unchanged", "skipped": [(line, reason)], "by_kind", "memory_errors"}."""
    counts: Counter = Counter()
    by_kind: Counter = Counter()
    skipped: List[Tuple[int, str]] = []
    memory_errors = 0
    seen = set()
    for line, raw in rows:
        if isinstance(raw, Exception):
            skipped.append((line, "not JSON: %s" % raw))
            continue
        downgraded = False
        if not trusted and isinstance(raw, dict):
            if owner is not None:
                raw = dict(raw, author=owner)
            if raw.get("status") == "verified":       # measured by the file's author, not by WS1 or the referee
                downgraded = True
                raw = dict(raw, status="claim", source=("%s [status 'verified' according to %s, not WS1]"
                                                        % (raw.get("source") or "", raw.get("author") or "its author")).strip())
        try:
            node = ideas.make_node(raw, allow_verified=trusted)
        except ideas.InvalidIdea as e:
            skipped.append((line, str(e)))
            continue
        if problem and node["problem"] != problem:
            continue
        if node["id"] in seen:
            skipped.append((line, "%s: duplicate id in this file" % node["id"]))
            continue
        seen.add(node["id"])
        if store is None:
            outcome = "dry-run"
        else:
            try:
                outcome = store.upsert(raw, trusted=trusted, owner=owner)
            except ideas.InvalidIdea as e:          # untrusted row on an id that already exists
                skipped.append((line, str(e)))
                continue
            node = store.get(node["id"])            # with any verdict we already reached
        counts[outcome] += 1
        by_kind[node["kind"]] += 1
        out("  %-9s %-10s %-13s %-9s %s%s" % (outcome, node["id"], node["kind"], node["status"],
                                             "(said verified) " if downgraded else "", node["content"][:90]))
        if memory is not None and store is not None:
            try:
                ideas.mirror(memory, node, scope)
            except Exception as e:                 # the store is the source of truth; say so and go on
                memory_errors += 1
                out("    memory write failed for %s: %s" % (node["id"], str(e)[:120]))
    for line, reason in skipped:
        out("  SKIPPED line %d: %s" % (line, reason))
    return {"new": counts["new"], "updated": counts["updated"], "unchanged": counts["unchanged"],
            "dry_run": counts["dry-run"], "skipped": skipped, "by_kind": dict(by_kind),
            "memory_errors": memory_errors}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="local JSONL file (default: read --git-ref:--git-path)")
    ap.add_argument("--git-ref", default=DEFAULT_REF)
    ap.add_argument("--git-path", default=DEFAULT_PATH)
    ap.add_argument("--ideas", help="idea store file (default: %s)" % ideas.default_path())
    ap.add_argument("--scope", default=SCOPE, help="memory scope of the idea facts")
    ap.add_argument("--memory", choices=("auto", "local", "none"), default="auto")
    ap.add_argument("--problem", help="import only the nodes of this problem")
    ap.add_argument("--trusted", action="store_true",
                    help="this path / git ref is WS1's own file: it may write 'verified' and refresh any node")
    ap.add_argument("--untrusted", action="store_true",
                    help="treat even the default source as untrusted (it can only add new ideas)")
    ap.add_argument("--owner", help="untrusted file of this author: it may also refresh that author's nodes")
    ap.add_argument("--dry-run", action="store_true", help="validate and print, write nothing")
    a = ap.parse_args(argv)
    if not a.scope.startswith("import-"):
        ap.error("--scope must start with import- (tag-only scopes; see integrations.run_loop)")

    source = a.path or "%s:%s" % (a.git_ref, a.git_path)
    rows = load_rows(a.path, a.git_ref, a.git_path)
    store = None if a.dry_run else ideas.IdeaStore(a.ideas)
    memory = None
    if not a.dry_run and a.memory != "none":
        memory = LocalMemory() if a.memory == "local" else get_memory()
    print("%d lines from %s -> ideas %s, memory %s, scope %s"
          % (len(rows), source, store.path if store else "(dry run)",
             memory.name if memory else "(none)", a.scope))
    default_source = not a.path and (a.git_ref, a.git_path) == (DEFAULT_REF, DEFAULT_PATH)
    if a.trusted and (a.untrusted or a.owner):
        ap.error("--trusted cannot be combined with --untrusted or --owner")
    trusted = (default_source or a.trusted) and not (a.untrusted or a.owner)
    print("source is %s" % ("TRUSTED (WS1): it may write 'verified' and refresh any node" if trusted else
                            "untrusted: it can only add new ideas%s"
                            % (" and refresh the untrusted nodes of %s" % a.owner if a.owner else "")))
    r = import_nodes(rows, store, memory, scope=a.scope, trusted=trusted, problem=a.problem, owner=a.owner)
    print("new %d, updated %d, unchanged %d, dry-run %d, skipped %d, memory errors %d | by kind: %s"
          % (r["new"], r["updated"], r["unchanged"], r["dry_run"], len(r["skipped"]), r["memory_errors"],
             ", ".join("%s %d" % kv for kv in sorted(r["by_kind"].items()))))
    return 1 if r["skipped"] or r["memory_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
