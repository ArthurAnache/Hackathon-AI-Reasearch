"""Allowlist of files shipped into Algorithm Lab Modal containers. Never ship .env, credentials or integrations."""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/repo"
PYTHON_VERSION = "3.11"
PIP_PINS = ("numpy==2.2.6", "scipy==1.16.2")
IMAGE_ENV = {
    "PYTHONPATH": REMOTE_ROOT,
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
CODE_FILES = (
    "prepare.py",
    "certificates.py",
    "execution/__init__.py",
    "execution/local.py",
    "problems/__init__.py",
    "problems/common.py",
    "problems/catalog.py",
    "problems/_frozen.py",
    "problems/p1a.py",
    "problems/p1b.py",
    "problems/p1d.py",
    "problems/p10c.py",
    "problems/p11a.py",
    "algolab/__init__.py",
    "algolab/kit.py",
    "algolab/gate.py",
    "algolab/child.py",
    "algolab/image.py",
    "algolab/modal_app.py",
    "optimizers/second-autocorrelation-inequality_algolab.py",
    "ws3_bench/__init__.py",
    "ws3_bench/worker.py",
)
WS1_PROBLEM_FILES = ("meta.json", "problem.py", "verifier_pinned.py", "independent.py", "certificate.json.gz")
FORBIDDEN_GLOBS = (
    ".env", ".env.*", "*.pem", "*.key", ".modal.toml", "*credentials*", "*secret*",
    ".git/*", ".venv/*", "*__pycache__*", "selection/*", "integrations/*", "runs/*",
    "algolab/plugins/*",  # plugins travel inside the job, never baked into the image
)
SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_\-]{20,}|ak-[A-Za-z0-9]{20,}|"
    r"(api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"])", re.IGNORECASE)


def relative_files(slug: str) -> list[str]:
    return list(CODE_FILES) + [f"problems/{slug}/{name}" for name in WS1_PROBLEM_FILES]


def image_files(slug: str, root: Path = ROOT) -> list[tuple[Path, str]]:
    out = []
    for rel in relative_files(slug):
        local = root / rel
        if not local.is_file():
            raise FileNotFoundError(f"algolab image file missing: {rel}")
        out.append((local, f"{REMOTE_ROOT}/{rel}"))
    return out


def audit(slug: str, root: Path = ROOT) -> list[str]:
    issues = []
    for rel in relative_files(slug):
        if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(Path(rel).name, g) for g in FORBIDDEN_GLOBS):
            issues.append(f"forbidden path in image: {rel}")
        path = root / rel
        if not path.is_file():
            issues.append(f"missing: {rel}")
            continue
        if path.suffix in (".py", ".json", ".md", ".txt"):
            if SECRET_PATTERN.search(path.read_text(encoding="utf-8", errors="replace")):
                issues.append(f"secret-looking string in {rel}")
    return issues
