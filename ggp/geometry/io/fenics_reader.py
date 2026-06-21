# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""FEniCS built-in geometry reader.

Handles ``fenics_rectangle`` (2-D quadrilateral mesh) and ``fenics_box``
(3-D hexahedral mesh) geometry types by calling FEniCS mesh generators.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import DomainRepresentation, GeometryReader
from .registry import register_reader


@register_reader("fenics_rectangle")
class FenicsRectangleReader(GeometryReader):
    """Creates a structured quadrilateral rectangle via ``dolfin.RectangleMesh``."""

    def read(self, spec: Any) -> DomainRepresentation:
        import dolfin as df

        p = spec.params
        Lx = float(p.get("Lx", 1.0))
        Ly = float(p.get("Ly", 1.0))
        nx = int(p.get("nx", 10))
        ny = int(p.get("ny", 10))

        mesh = df.RectangleMesh.create(
            [df.Point(0.0, 0.0), df.Point(Lx, Ly)],
            [nx, ny],
            df.CellType.Type.quadrilateral,
        )

        vertices = mesh.coordinates()
        cells = np.array([c.entities(0) for c in df.cells(mesh)])

        return DomainRepresentation(
            vertices=vertices,
            cells=cells,
            cell_type="quad",
            dim=2,
            metadata={"dolfin_mesh": mesh, "Lx": Lx, "Ly": Ly, "nx": nx, "ny": ny},
        )

    @staticmethod
    def supports(geometry_type: str) -> bool:
        return geometry_type == "fenics_rectangle"


@register_reader("fenics_box")
class FenicsBoxReader(GeometryReader):
    """Creates a structured hexahedral box via ``dolfin.BoxMesh``."""

    def read(self, spec: Any) -> DomainRepresentation:
        import dolfin as df

        p = spec.params
        Lx = float(p.get("Lx", 1.0))
        Ly = float(p.get("Ly", 1.0))
        Lz = float(p.get("Lz", 1.0))
        nx = int(p.get("nx", 10))
        ny = int(p.get("ny", 10))
        nz = int(p.get("nz", 10))

        mesh = df.BoxMesh.create(
            [df.Point(0.0, 0.0, 0.0), df.Point(Lx, Ly, Lz)],
            [nx, ny, nz],
            df.CellType.Type.hexahedron,
        )

        vertices = mesh.coordinates()
        cells = np.array([c.entities(0) for c in df.cells(mesh)])

        return DomainRepresentation(
            vertices=vertices,
            cells=cells,
            cell_type="hex",
            dim=3,
            metadata={"dolfin_mesh": mesh, "Lx": Lx, "Ly": Ly, "Lz": Lz, "nx": nx, "ny": ny, "nz": nz},
        )

    @staticmethod
    def supports(geometry_type: str) -> bool:
        return geometry_type == "fenics_box"
