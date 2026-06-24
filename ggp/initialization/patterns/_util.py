# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Shared helpers for structured initial-guess patterns."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def axis_coords(length: float, cell_size: float) -> Tuple[np.ndarray, int]:
    """Return ``(grid_points, n_cells)`` along one axis at ~``cell_size`` spacing.

    At least one cell is always produced.
    """
    n = max(1, int(round(length / cell_size)))
    return np.linspace(0.0, length, n + 1), n


class NodeTable:
    """Coordinate-deduplicating node accumulator.

    ``add(point)`` returns a stable integer index, reusing the index for points
    that coincide (within rounding) so edges shared between adjacent cells refer
    to the same node.
    """

    def __init__(self, decimals: int = 6) -> None:
        self._map: dict = {}
        self._coords: List[np.ndarray] = []
        self.decimals = decimals

    def add(self, point) -> int:
        p = np.asarray(point, dtype=float)
        key = tuple(np.round(p, self.decimals))
        idx = self._map.get(key)
        if idx is None:
            idx = len(self._coords)
            self._map[key] = idx
            self._coords.append(p)
        return idx

    def nodes(self) -> np.ndarray:
        return np.asarray(self._coords, dtype=float)
