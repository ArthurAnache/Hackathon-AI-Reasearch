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
        previous = {k: copy.deepcopy(self.get(k)) for k in patch}
        for k, v in patch.items():
            self._set(k, v)
        return previous

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


_MISSING = object()
