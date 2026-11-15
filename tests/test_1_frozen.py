"""Process 1: the problem definitions, verifiers and harness are immutable (hash-locked)."""
import json
import subprocess
import sys

import pytest

P = pytest.mark.process("1. Frozen files")


def check(root):
    return subprocess.run([sys.executable, "prepare.py", "check"], cwd=root, capture_output=True, text=True)


@P
def test_lock_matches_repo(record):
    """The committed frozen.lock matches prepare.py and problems/ in this repo."""
    import prepare
    st = prepare.frozen_status()
    record(frozen_status=st, files_locked=sorted(json.loads(prepare.LOCK_FILE.read_text())))
    assert st == "ok"


@P
@pytest.mark.parametrize("target", ["prepare.py", "problems/p1a.py", "problems/common.py"])
def test_tampering_is_detected(workspace, record, target):
    """Editing any frozen file makes `prepare.py check` fail and names the file."""
    assert check(workspace).returncode == 0
    path = workspace / target
    path.write_text(path.read_text() + "\n# tampered\n")
    proc = check(workspace)
    record(file=target, check_output=proc.stdout.strip())
    assert proc.returncode == 1 and target in proc.stdout


@P
def test_lock_scope_ws2_files_only(workspace, record):
    """WS2's lock covers its own truth layer (incl. certificates.py); WS1's problems/<slug>/ files are
    frozen by WS1 (MANIFEST.json), so adding one does not void WS2 runs."""
    import json as _json
    locked = sorted(_json.loads((workspace / "frozen.lock").read_text()))
    (workspace / "problems" / "ws1_new_helper.py").write_text("x = 1\n")
    ok = check(workspace)
    p = workspace / "certificates.py"
    p.write_text(p.read_text() + "\n# tampered\n")
    bad = check(workspace)
    record(locked=locked, after_ws1_file=ok.stdout.strip(), after_certificates_edit=bad.stdout.strip())
    assert "certificates.py" in locked and "problems/catalog.py" not in locked
    assert ok.returncode == 0 and bad.returncode == 1 and "certificates.py" in bad.stdout


@P
def test_missing_lock_is_reported(workspace, record):
    (workspace / "frozen.lock").unlink()
    proc = check(workspace)
    record(check_output=proc.stdout.strip())
    assert proc.returncode == 1 and "missing" in proc.stdout
