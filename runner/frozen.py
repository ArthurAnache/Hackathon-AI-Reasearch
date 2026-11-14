"""Hash-check the frozen parts of the repo (problem/ and runner/).

`frozen.lock` stores a sha256 per file. The loop refuses to run if any frozen
file was added, removed or changed since the lock was written. Re-lock on
purpose with `python -m runner.cli freeze` after a deliberate human edit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FROZEN_DIRS = ("problem", "runner")
LOCK_NAME = "frozen.lock"


class FrozenViolation(RuntimeError):
    pass


def _frozen_files(root: Path):
    for d in FROZEN_DIRS:
        for p in sorted((root / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".yaml", ".yml", ".json", ".md"}:
                yield p


def compute_hashes(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in _frozen_files(root)
    }


def combined_hash(hashes: dict[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(hashes):
        h.update(f"{k}:{hashes[k]}\n".encode())
    return h.hexdigest()[:16]


def write_lock(root: Path) -> dict[str, str]:
    hashes = compute_hashes(root)
    (root / LOCK_NAME).write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return hashes


def check(root: Path) -> str:
    """Return the combined frozen hash, or raise FrozenViolation."""
    lock = root / LOCK_NAME
    if not lock.exists():
        raise FrozenViolation(f"{LOCK_NAME} missing: run `python -m runner.cli freeze` once")
    expected = json.loads(lock.read_text())
    actual = compute_hashes(root)
    changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
    if changed:
        raise FrozenViolation("frozen files changed: " + ", ".join(changed))
    return combined_hash(actual)
