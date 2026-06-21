# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Self-registering geometry reader registry.

Usage
-----
**Register** a reader::

    @register_reader("step")
    class STEPReader(GeometryReader):
        ...

**Retrieve** a reader::

    reader = get_reader("step")
    domain = reader.read(geometry_spec)

**List** available readers::

    names = list_readers()  # ["fenics_rectangle", "fenics_box", "step", ...]
"""
from __future__ import annotations

from typing import Dict, List, Type

from .base import GeometryReader

_REGISTRY: Dict[str, Type[GeometryReader]] = {}


def register_reader(geometry_type: str):
    """Class decorator that registers a :class:`GeometryReader` subclass.

    Parameters
    ----------
    geometry_type : str
        The ``GeometrySpec.type`` value this reader handles.
    """
    def decorator(cls: Type[GeometryReader]) -> Type[GeometryReader]:
        _REGISTRY[geometry_type] = cls
        return cls
    return decorator


def get_reader(geometry_type: str) -> GeometryReader:
    """Instantiate the registered reader for *geometry_type*.

    Raises
    ------
    ValueError
        If *geometry_type* has not been registered.
    """
    if geometry_type not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"No reader for geometry type '{geometry_type}'. "
            f"Available: {available}"
        )
    return _REGISTRY[geometry_type]()


def list_readers() -> List[str]:
    """Return the names of all registered geometry readers."""
    return sorted(_REGISTRY.keys())
