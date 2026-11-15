"""Import published experiment results (e.g. WS3 Modal JSONL exports) into shared memory.

    # from a local file
    python -m integrations.import_results results/ws2-v1/ws2_1a_....jsonl
    # straight from a git branch, read-only (no checkout, nothing changes locally)
    python -m integrations.import_results --git-ref origin/results/modal-benchmarks \\
        --git-path results/ws2-v1/ws2_1a_131b577d_20261114T212806Z.jsonl
    # options: --dry-run  --author ws3-modal  --scope import-...

Each experiment record (interfaces.RECORD_FIELDS plus WS3 metadata such as
`variant`) becomes one memory fact of kind "experiment", so PastDevEvidence can
turn a kept config into a hint, exactly as for live runs.

Guarantees:
- Idempotent: each fact gets a stable id derived from (git_commit, variant,
  experiment_id, timestamp). Re-importing replaces instead of duplicating. An import
  under a --scope other than the default adds that scope to the id, so it creates its
  own facts and never moves the ones of the default scope.
- Namespaced: experiment ids are stored as "<variant>/<experiment_id>", so two
  workers that both produced exp_0000 do not collide.
- Machine-specific warm-start paths (config INIT, solution_path) are not used as hints.
- Own scope (default "import-<file stem>"). Import scopes are filtered by tag
  only (no time filter), so they work from any machine. Comparison runs with another
  --memory-scope never see imported facts, and runs without --pastdev never
  read memory at all. Only runs that ask for this scope, or that read the
  accumulated memory (no scope), use them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from ._common import REPO_ROOT
from .pastdev_client import get_memory, scope_start

INIT_KEY = "INIT"


def load_records(path: str = None, git_ref: str = None, git_path: str = None) -> List[Dict]:
    if git_ref:
        text = subprocess.run(["git", "show", "%s:%s" % (git_ref, git_path)], cwd=REPO_ROOT,
                              check=True, capture_output=True, text=True).stdout
    else:
        text = Path(path).read_text(encoding="utf-8")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def fact_for(r: Dict, source: str, author: str, scope_key: str = None):
    """scope_key: set when the import goes to a scope other than the file's default one,
    so the same records under two scopes are two facts (a second scope never moves the first)."""
    variant = r.get("variant") or "default"
    eid = "%s/%s" % (variant, r.get("experiment_id"))
    key = "|".join(str(r.get(k)) for k in ("git_commit", "variant", "experiment_id", "timestamp"))
    if scope_key:
        key += "|scope=" + scope_key
    fact_id = "import-" + hashlib.sha256(key.encode()).hexdigest()[:16]
    cfg = {k: v for k, v in (r.get("config") or {}).items() if k != INIT_KEY}
    # No braces in the text: FeedbackAgent's EvidenceHints parses the first {...} as a hint,
    # and PastDevEvidence appends the clean {"config": ...} itself for kept experiments.
    knobs = ", ".join("%s=%s" % (k, cfg[k]) for k in sorted(cfg))
    text = ("Imported experiment %s on problem %s (%s, optimizer %s): %s, score %s (best before %s). "
            "Hypothesis: %s. Knobs: %s."
            % (eid, r.get("problem"), author, r.get("optimizer"), r.get("status"), r.get("score"),
               r.get("best_before"), r.get("hypothesis"), knobs))
    meta = {"problem": r.get("problem"), "experiment_id": eid, "status": r.get("status"),
            "score": r.get("score"), "beats_record": r.get("beats_record"), "config": cfg,
            "optimizer": r.get("optimizer"), "algorithm_version": r.get("algorithm_version"),
            "git_commit": r.get("git_commit"), "seed": r.get("seed"), "budget_s": r.get("budget_s"),
            "variant": variant, "source": source, "imported": True}
    return fact_id, text, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="local JSONL file")
    ap.add_argument("--git-ref", help="read from this git ref instead of a local file")
    ap.add_argument("--git-path", help="path of the JSONL inside --git-ref")
    ap.add_argument("--author", default="ws3-modal", help="identity the facts are attributed to")
    ap.add_argument("--scope", help='memory scope (default "import-<file stem>")')
    ap.add_argument("--dry-run", action="store_true", help="print the facts, write nothing")
    a = ap.parse_args(argv)
    if not a.path and not (a.git_ref and a.git_path):
        ap.error("give a JSONL path, or --git-ref and --git-path")

    source = "%s:%s" % (a.git_ref, a.git_path) if a.git_ref else a.path
    records = load_records(a.path, a.git_ref, a.git_path)
    default_scope = "import-" + Path(a.git_path or a.path).stem[:40]
    scope = a.scope or default_scope
    memory = None if a.dry_run else get_memory()
    if memory is not None:
        scope_start(scope)
    print("%d records from %s -> scope %s, author %s, memory %s"
          % (len(records), source, scope, a.author, memory.name if memory else "(dry run)"))
    for r in records:
        if r.get("status") == "void":
            continue
        fact_id, text, meta = fact_for(r, source, a.author, None if scope == default_scope else scope)
        print("  %s  %s" % (fact_id, text[:150]))
        if memory is not None:
            memory.write(text, author=a.author, kind="experiment",
                         label="%s %s" % (meta["problem"], meta["status"]), metadata=meta,
                         wait=False, scope=scope, fact_id=fact_id)
    if memory is not None:
        print("use them: run_loop --pastdev --memory-scope %s   (or no scope: accumulated memory)" % scope)
    return 0


if __name__ == "__main__":
    sys.exit(main())
