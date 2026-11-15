"""Safety helpers: a scrubbed environment for every subprocess, and redacted error text for the ledger.

- clean_env(allow=()) -> copy of os.environ without any variable whose name looks like a secret (KEY, TOKEN, SECRET,
  PASSWORD, CREDENTIAL, AUTH), except the names listed in `allow` (e.g. GH_TOKEN for gh).
- safe_err(text) -> the last lines of an error, with anything key-like replaced by <redacted>, at most 300 chars.
  Raw stderr never goes to the ledger.
"""
import os
import re

SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|AUTH", re.I)
SECRET_VALUE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|past_sk[A-Za-z0-9_\-]*|ak-[A-Za-z0-9]{8,}|as-[A-Za-z0-9]{8,}|gh[pousr]_[A-Za-z0-9]{8,}"
    r"|Bearer\s+\S+|[A-Za-z0-9_\-]{32,})")


def clean_env(allow=()):
    return {k: v for k, v in os.environ.items() if k in allow or not SECRET_NAME.search(k)}


def safe_err(text, n_lines=3, limit=300):
    if not text:
        return ""
    lines = [ln for ln in str(text).strip().splitlines() if ln.strip()][-n_lines:]
    return SECRET_VALUE.sub("<redacted>", " | ".join(lines))[-limit:]
