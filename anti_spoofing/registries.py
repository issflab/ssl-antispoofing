from __future__ import annotations

from typing import Callable, Dict


class Registry:
    """Simple name -> builder registry."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable] = {}

    def register(self, name: str):
        def decorator(builder: Callable):
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = builder
            return builder

        return decorator

    def get(self, name: str) -> Callable:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items))
            raise ValueError(
                f"Unknown {self.kind} '{name}'. Available: {available}"
            ) from exc

    def names(self):
        return sorted(self._items)


MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")

