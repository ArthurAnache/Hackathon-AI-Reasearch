"""TRUSTED self-test plugin: skips the gate on purpose and tries to escape at run time. The audit hook must refuse
every attempt; bench.py selftest reads the refusals from trace.json."""
import socket
import subprocess

import numpy as np

PARAMS = {}


class Algorithm:
    def __init__(self, params, rng):
        self.attempts = []

    def improve(self, f, deadline):
        tries = {
            "network": lambda: socket.create_connection(("example.com", 80), timeout=2),
            "subprocess": lambda: subprocess.run(["cmd", "/c", "echo", "x"]),
            "write_outside": lambda: open(r"C:\Users\Public\algolab_escape.txt", "w"),
            "read_secret": lambda: open(r"C:\Users\thiba\Desktop\Berkeley\AIRESEARCHACK\savant\.env").read(),
        }
        refused = []
        for name, fn in tries.items():
            try:
                fn()
            except PermissionError:
                refused.append(name)
            except Exception:  # noqa: BLE001
                refused.append(name + ":other-error")
        raise RuntimeError("refused=" + ",".join(refused))
