"""Immutability guard for problem/ (definition + verifier).

frozen.lock stores a sha256 over every file in problem/. The runner checks it
before every experiment and refuses to run if anything changed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBLEM_DIR = ROOT / "problem"
LOCK_FILE = ROOT / "frozen.lock"


class FrozenViolation(RuntimeError):
    pass


def _files(directory: Path):
    for p in sorted(directory.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            yield p


def dir_hash(directory: Path) -> str:
    h = hashlib.sha256()
    for p in _files(directory):
        h.update(p.relative_to(directory).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def problem_hash() -> str:
    return dir_hash(PROBLEM_DIR)


def write_lock() -> str:
    digest = problem_hash()
    LOCK_FILE.write_text(f"sha256 {digest}\n")
    return digest


def read_lock() -> str:
    if not LOCK_FILE.exists():
        raise FrozenViolation(f"{LOCK_FILE.name} missing; run `python -m autoresearch lock` once (humans only)")
    parts = LOCK_FILE.read_text().split()
    if len(parts) != 2 or parts[0] != "sha256":
        raise FrozenViolation(f"{LOCK_FILE.name} is malformed")
    return parts[1]


def check_frozen() -> str:
    expected, actual = read_lock(), problem_hash()
    if expected != actual:
        raise FrozenViolation(
            f"problem/ changed (lock {expected[:12]}, now {actual[:12]}). "
            "The problem and verifier are immutable; restore them or have a human re-lock."
        )
    return actual
