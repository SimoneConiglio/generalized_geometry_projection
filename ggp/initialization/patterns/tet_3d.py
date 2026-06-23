# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Coarse structured tetrahedral mesh of a 3-D box (pattern ``tet3d``)."""
from __future__ import annotations

import numpy as np

from ..base import InitialGuessPattern
from ..registry import register_pattern
from ..skeleton import DomainBox, Skeleton
from ._util import NodeTable, axis_coords

# Freudenthal / Kuhn split of a cube into 6 tetrahedra, all sharing the main
# diagonal 0–7.  Corner local index = dx + 2*dy + 4*dz, (dx,dy,dz) in {0,1}^3.
_TETS = [
    (0, 1, 3, 7),
    (0, 1, 5, 7),
    (0, 4, 5, 7),
    (0, 4, 6, 7),
    (0, 2, 6, 7),
    (0, 2, 3, 7),
]
# unique vertex pairs of a tetrahedron
_TET_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


@register_pattern("tet3d")
class Tet3D(InitialGuessPattern):
    """Split each hex cell into 6 tetrahedra; tet edges seed the bars."""

    dim = 3

    def generate(self, box: DomainBox, *, cell_size: float, **params) -> Skeleton:
        Lz = box.Lz if box.Lz is not None else box.Ly
        xs, nx = axis_coords(box.Lx, cell_size)
        ys, ny = axis_coords(box.Ly, cell_size)
        zs, nz = axis_coords(Lz, cell_size)
        nt = NodeTable()
        edges = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 8 cube corners by local index dx + 2dy + 4dz
                    corner = []
                    for l in range(8):
                        dx, dy, dz = l & 1, (l >> 1) & 1, (l >> 2) & 1
                        corner.append((xs[i + dx], ys[j + dy], zs[k + dz]))
                    for tet in _TETS:
                        for a, b in _TET_EDGES:
                            edges.append((nt.add(corner[tet[a]]), nt.add(corner[tet[b]])))
        sk = Skeleton(nt.nodes(), np.asarray(edges, dtype=int))
        return sk.dedup_edges().drop_zero_length()
