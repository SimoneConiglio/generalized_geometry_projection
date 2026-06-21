# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Abstract base class and intermediate representation for geometry I/O.

:class:`DomainRepresentation` is the **format-agnostic** intermediate
object that bridges any geometry source (CAD file, built-in mesh, point
cloud) with any downstream discretiser (FEM, meshless, external solver).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DomainRepresentation:
    """Format-agnostic intermediate representation of a geometry.

    Attributes
    ----------
    vertices : ndarray, shape (N, dim)
        Vertex / node coordinates.  Always available.
    cells : ndarray or None, shape (M, nodes_per_cell)
        Cell connectivity.  ``None`` for point clouds / meshless sources.
    cell_type : str or None
        ``"triangle"``, ``"quad"``, ``"hex"``, ``"tet"``, or ``None``.
    dim : int
        Spatial dimension (2 or 3).
    facets : dict, optional
        Named boundary facet sets, e.g. ``{"left": [0, 1, 5], ...}``.
    metadata : dict, optional
        Arbitrary CAD/file metadata (units, layer info, etc.).
    """
    vertices: np.ndarray
    cells: Optional[np.ndarray] = None
    cell_type: Optional[str] = None
    dim: int = 2
    facets: Optional[Dict[str, List[int]]] = None
    metadata: Optional[Dict[str, Any]] = None


class GeometryReader(ABC):
    """Reads a geometry source into a :class:`DomainRepresentation`.

    Subclasses must implement :meth:`read` and :meth:`supports`.
    """

    @abstractmethod
    def read(self, spec: Any) -> DomainRepresentation:
        """Convert *spec* (a :class:`GeometrySpec`) into a domain representation.

        Parameters
        ----------
        spec : GeometrySpec
            Geometry specification from the problem definition layer.

        Returns
        -------
        DomainRepresentation
        """

    @staticmethod
    @abstractmethod
    def supports(geometry_type: str) -> bool:
        """Return ``True`` if this reader handles *geometry_type*."""
