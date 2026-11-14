"""Register an EinsteinArena agent (challenge -> proof of work -> register). Run once, by hand, with Arthur's approval.

    .venv/bin/python integrations/arena_register.py --name AGENT_NAME            # dry run: checks only, no request
    .venv/bin/python integrations/arena_register.py --name AGENT_NAME --confirm  # really registers

The server shows the API key only once. This script never prints or logs it. It:
  - writes it to ~/.config/einsteinarena/credentials.json (directory 0700, file 0600, outside the repo), and refuses
    to overwrite an existing file;
  - copies it to the macOS clipboard (pbcopy), so it can be pasted into .env as EINSTEIN_ARENA_API_KEY=...

Calls are logged to selection/arena_log.jsonl by integrations.einstein_arena, without request or response bodies.
Rate limit: 20 challenge/register calls per hour per IP. The challenge expires after 10 minutes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from integrations import einstein_arena as ea  # noqa: E402

CRED_PATH = Path.home() / ".config" / "einsteinarena" / "credentials.json"
NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{2,30}$")
CHALLENGE_TTL_S = 600
CHUNK = 2_000_000


def pow_ok(challenge: str, nonce: int, difficulty: int) -> bool:
    """Same test as web/src/app/api/agents/register/route.ts (verifyPow)."""
    h = hashlib.sha256(f"{challenge}{nonce}".encode()).hexdigest()
    zeros, extra = difficulty // 4, difficulty % 4
    if h[:zeros] != "0" * zeros:
        return False
    return extra == 0 or int(h[zeros], 16) < (16 >> extra)


def _search(args: tuple[str, int, int, int]) -> int | None:
    challenge, difficulty, start, stop = args
    zeros, extra = difficulty // 4, difficulty % 4
    prefix, limit = "0" * zeros, 16 >> extra
    base = hashlib.sha256(challenge.encode())
    for nonce in range(start, stop):
        h = base.copy()
        h.update(str(nonce).encode())
        d = h.hexdigest()
        if d.startswith(prefix) and (extra == 0 or int(d[zeros], 16) < limit):
            return nonce
    return None


def solve_pow(challenge: str, difficulty: int, deadline: float) -> int:
    workers = max(1, (os.cpu_count() or 2) - 1)
    with mp.Pool(workers) as pool:
        start = 0
        while time.time() < deadline:
            jobs = [(challenge, difficulty, start + i * CHUNK, start + (i + 1) * CHUNK) for i in range(workers)]
            start += workers * CHUNK
            found = [n for n in pool.map(_search, jobs) if n is not None]
            if found:
                return min(found)
    raise TimeoutError("proof of work not found before the challenge deadline")


def save_key(name: str, key: str) -> None:
    CRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(CRED_PATH.parent, 0o700)
    fd = os.open(CRED_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"agent_name": name, "api_key": key,
                   "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   "base_url": ea.base_url()}, f)


def copy_to_clipboard(secret: str) -> bool:
    if not shutil.which("pbcopy"):
        return False
    subprocess.run(["pbcopy"], input=secret.encode(), check=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Register an EinsteinArena agent (one time).")
    ap.add_argument("--name", required=True, help="agent name: 2-30 chars, letters, digits, - and _")
    ap.add_argument("--confirm", action="store_true", help="really send the challenge and register requests")
    args = ap.parse_args()

    problems = []
    if not NAME_RE.fullmatch(args.name):
        problems.append("name must be 2-30 characters: letters, digits, '-' or '_'")
    if ea.has_api_key():
        problems.append("EINSTEIN_ARENA_API_KEY is already set in .env: an agent seems registered already")
    if CRED_PATH.exists():
        problems.append(f"{CRED_PATH} already exists: move it away first (it may hold a live key)")
    if problems:
        sys.exit("Refusing to register:\n  - " + "\n  - ".join(problems))

    if not args.confirm:
        print(f"Dry run OK for agent name {args.name!r} on {ea.base_url()}.\n"
              f"With --confirm: POST /api/agents/challenge, proof of work (~2^25 hashes), POST /api/agents/register,\n"
              f"then the key goes to {CRED_PATH} (0600) and to the clipboard. Nothing was sent.")
        return

    ch = ea.request_challenge(args.name)
    deadline = time.time() + CHALLENGE_TTL_S - 60
    t0 = time.time()
    nonce = solve_pow(ch["challenge"], int(ch["difficulty"]), deadline)
    assert pow_ok(ch["challenge"], nonce, int(ch["difficulty"]))
    print(f"Proof of work found in {time.time() - t0:.1f}s (difficulty {ch['difficulty']}).")
    resp = ea.register(args.name, ch["challenge"], nonce)
    key = resp["agent"]["api_key"]
    save_key(resp["agent"]["name"], key)
    copied = copy_to_clipboard(key)
    del key
    print(f"Registered agent {resp['agent']['name']!r}. API key saved to {CRED_PATH} (mode 0600)"
          + (" and copied to the clipboard." if copied else ".")
          + "\nPaste it into .env as EINSTEIN_ARENA_API_KEY=<key>. The key is not shown here and was not logged.")


if __name__ == "__main__":
    main()
