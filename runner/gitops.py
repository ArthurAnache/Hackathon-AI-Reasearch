"""Keep/reject via git: a kept experiment is a commit; a rejected one is reverted."""
from __future__ import annotations

import subprocess
from pathlib import Path


def git(root: Path, *args: str, check: bool = True) -> str:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def dirty_paths(root: Path) -> list[str]:
    lines = git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    return [l[3:].strip() for l in lines if l.strip()]


def diff(root: Path, paths: list[str]) -> str:
    git(root, "add", "-N", *paths, check=False)  # so new files show up in the diff
    return git(root, "diff", "--", *paths)


def revert(root: Path, path: str = "strategy") -> None:
    git(root, "checkout", "HEAD", "--", path, check=False)
    git(root, "clean", "-fdq", "--", path, check=False)


def commit(root: Path, paths: list[str], message: str) -> str:
    git(root, "add", "--", *paths)
    git(root, "commit", "-q", "-m", message, "--", *paths)
    return git(root, "rev-parse", "--short", "HEAD").strip()


def head(root: Path) -> str:
    return git(root, "rev-parse", "--short", "HEAD").strip()
