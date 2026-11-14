import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A throwaway copy of the repo in its own git repo, with a fast protocol."""
    root = tmp_path / "repo"
    root.mkdir()
    for d in ("problem", "strategy", "runner"):
        shutil.copytree(REPO / d, root / d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO / "program.md", root / "program.md")
    cfg_path = root / "runner" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.update(seeds=[0, 1], budget_s=2, agent="heuristic")
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    from runner import frozen
    frozen.write_lock(root)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return root
