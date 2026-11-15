"""Load strategy.yaml and expose dotted-key access. The strategist allow-list is enforced here."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).with_name("strategy.yaml")


class StrategyError(ValueError):
    pass


class Strategy:
    """A mutable view of strategy.yaml. Only allow-listed keys can be patched at run time."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH, overrides: dict[str, Any] | None = None) -> "Strategy":
        data = yaml.safe_load(Path(path).read_text())
        s = cls(data)
        for key, value in (overrides or {}).items():
            s._set(key, value)  # CLI overrides are trusted (a human typed them)
        return s

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, dotted: str) -> Any:
        value = self.get(dotted, _MISSING)
        if value is _MISSING:
            raise KeyError(dotted)
        return value

    def _set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def apply_patch(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a strategist patch. Returns the previous values (for revert). Rejects non-allow-listed keys."""
        allowed = set(self._data.get("strategist_allow_list", []))
        bad = [k for k in patch if k not in allowed]
        if bad:
            raise StrategyError(f"keys not in strategist_allow_list: {bad}")
        _check_portfolio_ranges(patch)
        previous = {k: copy.deepcopy(self.get(k)) for k in patch}
        for k, v in patch.items():
            self._set(k, v)
        return previous

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


_MISSING = object()


def _check_portfolio_ranges(patch: dict[str, Any]) -> None:
    """Range checks of the start portfolio keys the strategist may patch (brief section 6)."""
    if "portfolio.floors" in patch:
        floors = patch["portfolio.floors"]
        if not isinstance(floors, dict) or not all(isinstance(v, (int, float)) and 0.1 <= v <= 0.4
                                                   for v in floors.values()) or sum(floors.values()) > 0.9:
            raise StrategyError("portfolio.floors: each floor in [0.1, 0.4] and their sum at most 0.9")
    if "portfolio.promote_gap" in patch and not (isinstance(patch["portfolio.promote_gap"], (int, float))
                                                 and 0.0 < patch["portfolio.promote_gap"] <= 0.2):
        raise StrategyError("portfolio.promote_gap must be in (0, 0.2]")
    if "portfolio.k_cold" in patch and patch["portfolio.k_cold"] not in (1, 2, 3, 4):
        raise StrategyError("portfolio.k_cold must be an integer in [1, 4]")
