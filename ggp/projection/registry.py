# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Self-registering mapper registry.

Usage
-----
**Register** a mapper::

    @register_mapper("Free")
    class Free2DMapper(ProjectionMapper):
        ...

**Retrieve** a mapper::

    mapper = get_mapper("Free", num_components=18, ...)

**List** available mappers::

    names = list_mappers()  # ["Free", "ALM", "3D_Free", ...]
"""
from __future__ import annotations

from typing import Dict, List, Type

from .base import ProjectionMapper

_REGISTRY: Dict[str, Type[ProjectionMapper]] = {}


def register_mapper(mode: str):
    """Class decorator that registers a :class:`ProjectionMapper` subclass.

    Parameters
    ----------
    mode : str
        Formulation mode identifier (e.g. ``"Free"``, ``"ALM"``, ``"3D_Free"``).
    """
    def decorator(cls: Type[ProjectionMapper]) -> Type[ProjectionMapper]:
        _REGISTRY[mode] = cls
        return cls
    return decorator


def get_mapper(mode: str, **kwargs) -> ProjectionMapper:
    """Instantiate the registered mapper for *mode*.

    Parameters
    ----------
    mode : str
        Formulation mode (must match a previously registered mapper).
    \*\*kwargs
        Forwarded to the mapper constructor.

    Raises
    ------
    ValueError
        If *mode* has not been registered.
    """
    if mode not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"No mapper registered for mode '{mode}'. Available: {available}"
        )
    return _REGISTRY[mode](**kwargs)


def list_mappers() -> List[str]:
    """Return the names of all registered mappers."""
    return sorted(_REGISTRY.keys())
