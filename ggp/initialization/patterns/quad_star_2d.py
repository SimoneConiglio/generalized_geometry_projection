# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Quad lattice with centre spokes (pattern ``quad_star``).

Each quad cell is split into four triangles by connecting its four sides to the
cell centre; the union of quad sides and centre spokes seeds the bars.
"""
from __future__ import annotations

import numpy as np

from ..base import InitialGuessPattern
from ..registry import register_pattern
from ..skeleton import DomainBox, Skeleton
from ._util import NodeTable, axis_coords


@register_pattern("quad_star")
class QuadStar2D(InitialGuessPattern):
    dim = 2

    def generate(self, box: DomainBox, *, cell_size: float, **params) -> Skeleton:
        xs, nx = axis_coords(box.Lx, cell_size)
        ys, ny = axis_coords(box.Ly, cell_size)
        nt = NodeTable()
        edges = []
        for i in range(nx):
            for j in range(ny):
                p00 = (xs[i], ys[j])
                p10 = (xs[i + 1], ys[j])
                p11 = (xs[i + 1], ys[j + 1])
                p01 = (xs[i], ys[j + 1])
                c = (0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1]))
                corners = [p00, p10, p11, p01]
                # quad sides
                for k in range(4):
                    edges.append((nt.add(corners[k]), nt.add(corners[(k + 1) % 4])))
                # centre spokes
                for corner in corners:
                    edges.append((nt.add(c), nt.add(corner)))
        sk = Skeleton(nt.nodes(), np.asarray(edges, dtype=int))
        return sk.dedup_edges().drop_zero_length()
