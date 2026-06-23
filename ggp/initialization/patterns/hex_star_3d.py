# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Hex lattice with face- and body-centre spokes (pattern ``hex_star``).

Each hex cell is split into 24 tetrahedra: every face is cut into four triangles
through its face centre, and each face triangle is joined to the body centre.
The union of cube edges, face-centre spokes, and body-centre spokes seeds the
bars.
"""
from __future__ import annotations

import numpy as np

from ..base import InitialGuessPattern
from ..registry import register_pattern
from ..skeleton import DomainBox, Skeleton
from ._util import NodeTable, axis_coords

# 12 cube edges as corner-index pairs (corner = dx + 2dy + 4dz)
_CUBE_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),   # x-direction
    (0, 2), (1, 3), (4, 6), (5, 7),   # y-direction
    (0, 4), (1, 5), (2, 6), (3, 7),   # z-direction
]
# 6 faces, each as its 4 corner indices
_FACES = [
    (0, 2, 6, 4),   # x = 0
    (1, 3, 7, 5),   # x = 1
    (0, 1, 5, 4),   # y = 0
    (2, 3, 7, 6),   # y = 1
    (0, 1, 3, 2),   # z = 0
    (4, 5, 7, 6),   # z = 1
]


@register_pattern("hex_star")
class HexStar3D(InitialGuessPattern):
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
                    corner = []
                    for l in range(8):
                        dx, dy, dz = l & 1, (l >> 1) & 1, (l >> 2) & 1
                        corner.append(np.array([xs[i + dx], ys[j + dy], zs[k + dz]]))
                    body_c = np.mean(corner, axis=0)
                    # cube edges
                    for a, b in _CUBE_EDGES:
                        edges.append((nt.add(corner[a]), nt.add(corner[b])))
                    # body-centre spokes to corners
                    for c in corner:
                        edges.append((nt.add(body_c), nt.add(c)))
                    # face centres: spokes to face corners + body centre
                    for face in _FACES:
                        face_c = np.mean([corner[v] for v in face], axis=0)
                        edges.append((nt.add(body_c), nt.add(face_c)))
                        for v in face:
                            edges.append((nt.add(face_c), nt.add(corner[v])))
        sk = Skeleton(nt.nodes(), np.asarray(edges, dtype=int))
        return sk.dedup_edges().drop_zero_length()
