# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Coarse structured triangulation of a 2-D rectangle (pattern ``tri2d``)."""
from __future__ import annotations

import numpy as np

from ..base import InitialGuessPattern
from ..registry import register_pattern
from ..skeleton import DomainBox, Skeleton
from ._util import NodeTable, axis_coords


@register_pattern("tri2d")
class Triangulation2D(InitialGuessPattern):
    """Split the domain into a grid of cells, each cut into two triangles.

    The triangle edges (horizontal, vertical, and one diagonal per cell) become
    the seed bars.  ``cell_size`` controls the resolution and therefore the
    component count.
    """

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
                # two triangles sharing the p00–p11 diagonal
                for a, b in [(p00, p10), (p10, p11), (p11, p00),
                             (p11, p01), (p01, p00)]:
                    edges.append((nt.add(a), nt.add(b)))
        sk = Skeleton(nt.nodes(), np.asarray(edges, dtype=int))
        return sk.dedup_edges().drop_zero_length()
