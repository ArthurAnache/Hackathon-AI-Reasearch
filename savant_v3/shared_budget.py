"""Atomic shared budget ceiling across SAVANT V2/V3.

Canonical file (outside worktrees):
  _coordination/savant/SHARED_BUDGET_CEILING.json

Uses an exclusive lock file so concurrent worktrees cannot double-spend.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

COORD = Path(__file__).resolve().parents[2].parent / "_coordination" / "savant"
# parents[2] = savant-v3 root; .parent = AIRESEARCHACK
if not (COORD / "SHARED_BUDGET_CEILING.json").exists():
    # Fallback absolute path used on this machine
    COORD = Path(r"C:\Users\thiba\Desktop\Berkeley\AIRESEARCHACK\_coordination\savant")

CEILING = COORD / "SHARED_BUDGET_CEILING.json"
LOCK = COORD / "SHARED_BUDGET_CEILING.lock"


@contextmanager
def _lock(timeout_s: float = 30.0):
    COORD.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}\n".encode())
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"budget lock held: {LOCK}")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            LOCK.unlink(missing_ok=True)
        except OSError:
            pass


def load() -> dict:
    with _lock():
        return json.loads(CEILING.read_text(encoding="utf-8"))


def remaining() -> float:
    d = load()
    return float(d["authorized_total_usd"]) - float(d.get("spent_usd") or 0) - float(d.get("reserved_usd") or 0)


def reserve(amount_usd: float, *, purpose: str, campaign: str = "v3") -> str:
    if amount_usd < 0:
        raise ValueError("negative reserve")
    with _lock():
        d = json.loads(CEILING.read_text(encoding="utf-8"))
        rem = float(d["authorized_total_usd"]) - float(d.get("spent_usd") or 0) - float(d.get("reserved_usd") or 0)
        if amount_usd > rem + 1e-9:
            raise RuntimeError(f"shared ceiling exhausted: need ${amount_usd:.4f}, remaining ${rem:.4f}")
        rid = f"{campaign}-{int(time.time())}-{os.getpid()}"
        d["reserved_usd"] = float(d.get("reserved_usd") or 0) + amount_usd
        d.setdefault("reservations", {})[rid] = {
            "amount_usd": amount_usd, "purpose": purpose, "campaign": campaign,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        d["remaining_ceiling_usd"] = float(d["authorized_total_usd"]) - float(d["spent_usd"]) - float(d["reserved_usd"])
        CEILING.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
        return rid


def commit(reservation_id: str, actual_usd: float | None = None) -> dict:
    with _lock():
        d = json.loads(CEILING.read_text(encoding="utf-8"))
        res = (d.get("reservations") or {}).pop(reservation_id, None)
        if res is None:
            return d  # idempotent
        held = float(res["amount_usd"])
        amount = held if actual_usd is None else min(held, max(0.0, float(actual_usd)))
        d["reserved_usd"] = max(0.0, float(d.get("reserved_usd") or 0) - held)
        d["spent_usd"] = float(d.get("spent_usd") or 0) + amount
        d.setdefault("spend_log", []).append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "amount_usd": amount, "reservation_id": reservation_id,
            "purpose": res.get("purpose"), "campaign": res.get("campaign"),
        })
        d["remaining_ceiling_usd"] = float(d["authorized_total_usd"]) - float(d["spent_usd"]) - float(d.get("reserved_usd") or 0)
        CEILING.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
        return d


def release(reservation_id: str) -> dict:
    with _lock():
        d = json.loads(CEILING.read_text(encoding="utf-8"))
        res = (d.get("reservations") or {}).pop(reservation_id, None)
        if res is None:
            return d
        d["reserved_usd"] = max(0.0, float(d.get("reserved_usd") or 0) - float(res["amount_usd"]))
        d["remaining_ceiling_usd"] = float(d["authorized_total_usd"]) - float(d["spent_usd"]) - float(d.get("reserved_usd") or 0)
        CEILING.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
        return d
