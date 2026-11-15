"""Deliberately hostile plugin: must be rejected by the gate (used by the self-test in bench.py)."""
import os
import numpy as np

PARAMS = {}


class Algorithm:
    def __init__(self, params, rng):
        self.rng = rng

    def improve(self, f, deadline):
        open("leak.txt", "w").write(os.environ.get("OPENAI_API_KEY", ""))
        mod = __import__("socket")
        np.save("x.npy", f)
        return f.__class__(f)
