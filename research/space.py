"""The parameter space an agent may search: an optimizer's numeric CONFIG knobs.

Optimizers may declare ranges in an optional SEARCH_SPACE dict:
    SEARCH_SPACE = {"N_CELLS": (100, 3000, "log"), "LR": (1e-3, 0.1, "log")}
Knobs without a declaration get [default/8, default*8] on a log scale. Integer knobs
stay integers. Everything is mapped to the unit cube [0,1]^d, so strategies can
work in one geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Non-numeric config keys that carry results between experiments (warm start).
INIT_KEY = "INIT"


@dataclass(frozen=True)
class Knob:
    name: str
    low: float
    high: float
    log: bool
    integer: bool

    def to_unit(self, v: float) -> float:
        v = min(max(float(v), self.low), self.high)
        if self.high == self.low:
            return 0.5
        if self.log:
            return (math.log(v) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (v - self.low) / (self.high - self.low)

    def from_unit(self, u: float):
        u = min(max(float(u), 0.0), 1.0)
        if self.log:
            v = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            v = self.low + u * (self.high - self.low)
        return max(int(self.low), int(round(v))) if self.integer else float(f"{v:.6g}")


class Space:
    def __init__(self, knobs: dict[str, Knob], defaults: dict):
        self.knobs = knobs
        self.defaults = dict(defaults)
        self.names = list(knobs)

    @classmethod
    def from_optimizer(cls, optimizer) -> "Space":
        module = optimizer._load()
        config, declared = dict(module.CONFIG), dict(getattr(module, "SEARCH_SPACE", {}) or {})
        knobs = {}
        for k, v in config.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            integer = isinstance(v, int)
            if k in declared:
                low, high, *scale = declared[k]
                log = (scale[0] if scale else "log") == "log"
            elif v > 0:
                low, high, log = v / 8, v * 8, True
                if integer:
                    low, high = max(1, math.floor(low)), max(2, math.ceil(high))
            else:
                continue                      # zero/negative without declared range: not tunable
            knobs[k] = Knob(k, float(low), float(high), log, integer)
        return cls(knobs, config)

    @property
    def dim(self) -> int:
        return len(self.names)

    def to_unit(self, config: dict) -> np.ndarray:
        return np.array([self.knobs[k].to_unit(config.get(k, self.defaults[k])) for k in self.names])

    def from_unit(self, u, base: dict | None = None) -> dict:
        out = dict(base or {})
        for k, x in zip(self.names, u):
            out[k] = self.knobs[k].from_unit(x)
        return out

    def key(self, config: dict) -> tuple:
        """Identity of a config for de-duplication (knob values after rounding, plus warm-start source)."""
        rounded = tuple(self.knobs[k].from_unit(self.knobs[k].to_unit(config.get(k, self.defaults[k])))
                        for k in self.names)
        return rounded + (config.get(INIT_KEY),)
