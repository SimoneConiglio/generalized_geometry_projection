# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Self-registering initial-guess pattern registry.

Mirrors :mod:`ggp.projection.registry`.

Usage
-----
**Register** a pattern::

    @register_pattern("tri2d")
    class Triangulation2D(InitialGuessPattern):
        ...

**Retrieve** a pattern::

    pattern = get_pattern("tri2d")
    skeleton = pattern.generate(box, cell_size=15.0)

**List** available patterns::

    names = list_patterns()  # ["hex_star", "quad_star", "tet3d", "tri2d"]
"""
from __future__ import annotations

from typing import Dict, List, Type

from .base import InitialGuessPattern

_REGISTRY: Dict[str, Type[InitialGuessPattern]] = {}


def register_pattern(name: str):
    """Class decorator registering an :class:`InitialGuessPattern` subclass."""

    def decorator(cls: Type[InitialGuessPattern]) -> Type[InitialGuessPattern]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_pattern(name: str, **kwargs) -> InitialGuessPattern:
    """Instantiate the registered pattern *name*.

    Raises
    ------
    ValueError
        If *name* has not been registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"No initial-guess pattern registered for '{name}'. Available: {available}"
        )
    return _REGISTRY[name](**kwargs)


def is_pattern(name: str) -> bool:
    """Return ``True`` if *name* is a registered mesh pattern."""
    return name in _REGISTRY


def list_patterns() -> List[str]:
    """Return the names of all registered patterns."""
    return sorted(_REGISTRY.keys())
