# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""cadjoint geometry reader — CAD-defined design domains from a signed distance field.

`cadjoint <https://github.com/andrinr/cadjoint>`_ (Apache-2.0) is a differentiable
CAD system: sketches with constraints are extruded or revolved into an SDF tree,
which it then contours into a mesh.  This reader borrows only its geometry core,
so GGP's design domain no longer has to be an axis-aligned box — it can be any
solid a cadjoint scene describes.

Meshing path
------------
``cadjoint.fem.hexmesh.sdf_to_hex_mesh`` keeps the lattice cells whose centre
lies inside the field (``sdf < 0``), so the result is a *trimmed voxel grid*: a
structured HEX8 mesh with the shape of the CAD solid punched out of it.  Cells
outside the solid are absent rather than marked void, so they cost no DOFs.

``snap`` is **off by default, deliberately**.  With ``snap=True`` cadjoint
Newton-projects boundary vertices onto the zero set, which gives a far better
surface but leaves every boundary cell a different shape.
:class:`~ggp.discretisation.fem.FEMDiscretiser` builds a *single* reference
element stiffness matrix from the first cell of the mesh
(``df.assemble_local(a, next(df.cells(mesh)))``) and reuses it for every
element, which is only valid when all cells are congruent.  Unsnapped, they
are — every cell is the same lattice box — so ``ke_ref`` stays correct.  Turn
snapping on only if you also replace that single-``ke_ref`` assumption.

Requirements
------------
The geometry core needs ``jax``, ``numpy`` and ``optax`` only — not cadjoint's
``fem`` extra, and not FEniCS.  cadjoint is not on PyPI; install it from source
and pin a commit, since its README warns the API is unstable::

    git clone https://github.com/andrinr/cadjoint
    pip install -e cadjoint

**Python version.** cadjoint declares ``requires-python = ">=3.9"`` but imports
:class:`enum.StrEnum`, which is 3.11+.  ``environment.yml`` pins the ``ggp``
environment to Python 3.10, and that pin is load-bearing: ``dolfin-adjoint``
2019.1.0 fails against the 3.11 build of ``dolfin`` (``TypeError: __class__
assignment: 'Mesh' object layout differs``), so the environment cannot simply be
moved to 3.11.

``StrEnum`` is cadjoint's only 3.11-ism on this reader's import path, so either
of these works, both verified against the full test suite:

* Make ``enum.StrEnum`` available on 3.10 — a three-line backport
  (``class StrEnum(str, Enum)``) is enough, and is worth proposing upstream.
* Run cadjoint in its own Python 3.11 environment and hand this reader the
  resulting arrays.

Example
-------
::

    from ggp.problem.spec import GeometrySpec
    from ggp.geometry.io import get_reader
    from cadjoint.sdf import Box, difference

    bracket = difference(Box(size=[60, 60, 10]), Box(size=[30, 30, 12]))
    spec = GeometrySpec(
        type="cadjoint",
        params={"sdf": bracket, "Lx": 60, "Ly": 60, "Lz": 10,
                "nx": 24, "ny": 24, "nz": 4},
    )
    domain = get_reader("cadjoint").read(spec)
"""
from __future__ import annotations

import runpy
import sys
from typing import Any, Callable, Dict, List

import numpy as np

from .base import DomainRepresentation, GeometryReader
from .registry import register_reader

#: cadjoint emits HEX8 connectivity in VTK/meshio corner order
#: (``cadjoint.fem.elements.HEX_CORNER_OFFSETS``); FEniCS/UFC orders the eight
#: corners lexicographically in (x, y, z).  ``ufc = vtk[:, _VTK_TO_UFC_HEX]``.
_VTK_TO_UFC_HEX = np.array([0, 4, 3, 7, 1, 5, 2, 6], dtype=np.int64)

#: Axis-extreme facet names, in the vocabulary ``FEMDiscretiser`` already speaks
#: (``"left"``, ``"top"``, …).  Keyed by (axis, take-the-maximum).
_FACET_NAMES: Dict[tuple, str] = {
    (0, False): "left",   (0, True): "right",
    (1, False): "bottom", (1, True): "top",
    (2, False): "front",  (2, True): "back",
}

# Names a scene module may bind its geometry to, tried in order.
_SCENE_ATTRS = ("sdf", "scene", "geometry", "part", "body")


def _as_batched_sdf(source: Any) -> Callable[[Any], Any]:
    """Adapt any cadjoint geometry source to the calling convention its mesher uses.

    ``sdf_to_hex_mesh`` calls the field two different ways: batched, on ``(M, 3)``
    lattice points, and *per point* on a ``(3,)`` tracer inside
    ``jax.vmap(jax.grad(...))`` when it groups the boundary faces.  A
    :class:`cadjoint.sdf.SDF` node broadcasts over ``(..., 3)`` and serves both.
    A function produced by :func:`cadjoint.functionalize` maps one point to one
    scalar and needs a :func:`jax.vmap` for the batched call.

    The rank of the argument decides which is which on every call — latching the
    decision once would break the per-point path, since a per-point function
    called on ``(3,)`` and on ``(M, 3)`` needs opposite treatment.
    """
    if not callable(source):
        raise TypeError(
            f"cadjoint geometry source must be an SDF or a callable, got {type(source).__name__}."
        )

    import jax
    import jax.numpy as jnp

    # Whether `source` handles a batch itself. Resolved on the first batched
    # call and reused; the per-point path never consults it.
    broadcasts: Dict[str, Any] = {"value": None}

    def evaluate(points):
        pts = jnp.asarray(points)
        if pts.ndim == 1:
            # Single point — every source shape handles this directly.
            return source(pts)

        if broadcasts["value"] is None:
            try:
                values = source(pts)
                broadcasts["value"] = jnp.shape(values)[:1] == (pts.shape[0],)
                if broadcasts["value"]:
                    return values
            except Exception:        # noqa: BLE001 — any failure means "needs vmap"
                broadcasts["value"] = False
        elif broadcasts["value"]:
            return source(pts)

        return jax.vmap(lambda point: jnp.asarray(source(point)).reshape(()))(pts)

    return evaluate


def _import_failure_hint(exc: ImportError) -> str:
    """Explain a failed ``import cadjoint`` rather than re-raising it bare.

    On Python 3.10 the failure is almost always :class:`enum.StrEnum`, which
    cadjoint imports despite declaring ``requires-python = ">=3.9"``.  That is
    worth naming, because the obvious fix — moving the environment to 3.11 —
    breaks ``dolfin-adjoint`` instead.
    """
    hint = (
        f"The 'cadjoint' geometry backend is not importable ({exc}). "
        "Install it from source: "
        "git clone https://github.com/andrinr/cadjoint && pip install -e cadjoint"
    )
    if sys.version_info < (3, 11) and "StrEnum" in str(exc):
        hint += (
            f". Note this environment runs Python "
            f"{sys.version_info.major}.{sys.version_info.minor}: cadjoint imports "
            "enum.StrEnum, which is 3.11+, even though it declares support for "
            "3.9. Moving the ggp environment to 3.11 is not the fix — "
            "dolfin-adjoint 2019.1.0 breaks against the 3.11 build of dolfin. "
            "Backport StrEnum (class StrEnum(str, Enum)) or run cadjoint in a "
            "separate 3.11 environment; see the module docstring."
        )
    return hint


def _sdf_from_path(path: Any) -> Any:
    """Load a cadjoint scene module and return the SDF it declares.

    The module is executed, then searched for one of :data:`_SCENE_ATTRS`.
    """
    namespace = runpy.run_path(str(path))
    for attr in _SCENE_ATTRS:
        if attr in namespace and namespace[attr] is not None:
            return namespace[attr]
    raise ValueError(
        f"cadjoint scene '{path}' defines none of {_SCENE_ATTRS}; "
        "bind the geometry to one of those names, or pass params['sdf'] directly."
    )


def hex_cells_to_dolfin_order(cells: np.ndarray) -> np.ndarray:
    """Reorder VTK/meshio HEX8 connectivity into FEniCS/UFC corner order.

    Parameters
    ----------
    cells : ndarray, shape (C, 8)
        Connectivity as cadjoint emits it.

    Returns
    -------
    ndarray, shape (C, 8)
        The same cells with corners permuted to UFC's lexicographic order.

    Notes
    -----
    On a trimmed lattice this permutation also leaves each row *ascending*,
    which is what DOLFIN requires of cell connectivity — the lattice vertex
    numbering is itself lexicographic in (i, j, k), so UFC corner order and
    ascending global index coincide.
    """
    cells = np.asarray(cells)
    if cells.ndim != 2 or cells.shape[1] != 8:
        raise ValueError(f"Expected HEX8 connectivity of shape (C, 8), got {cells.shape}.")
    return cells[:, _VTK_TO_UFC_HEX]


def build_dolfin_hex_mesh(vertices: np.ndarray, cells: np.ndarray):
    """Build a ``dolfin.Mesh`` of hexahedra from plain arrays.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    cells : ndarray, shape (C, 8)
        VTK/meshio corner order — permuted here, so pass cadjoint's output as-is.

    Returns
    -------
    dolfin.Mesh
    """
    import dolfin as df

    vertices = np.asarray(vertices, dtype=np.float64)
    cells_ufc = hex_cells_to_dolfin_order(cells)

    mesh = df.Mesh()
    editor = df.MeshEditor()
    editor.open(mesh, "hexahedron", 3, 3)
    editor.init_vertices(vertices.shape[0])
    editor.init_cells(cells_ufc.shape[0])
    for index, point in enumerate(vertices):
        editor.add_vertex(index, point)
    for index, cell in enumerate(cells_ufc):
        editor.add_cell(index, np.asarray(cell, dtype=np.uintp))
    editor.close()
    return mesh


def _axis_facets(vertices: np.ndarray, tol: float) -> Dict[str, List[int]]:
    """Name the six axis-extreme vertex sets of the bounding box."""
    facets: Dict[str, List[int]] = {}
    for axis in range(3):
        column = vertices[:, axis]
        for take_max, extreme in ((False, column.min()), (True, column.max())):
            hits = np.where(np.abs(column - extreme) <= tol)[0]
            facets[_FACET_NAMES[(axis, take_max)]] = hits.tolist()
    return facets


@register_reader("cadjoint")
class CadjointReader(GeometryReader):
    """Meshes a cadjoint SDF into a trimmed structured HEX8 design domain.

    Parameters read from ``GeometrySpec.params``
    --------------------------------------------
    sdf : SDF or callable, optional
        The geometry.  Required unless ``GeometrySpec.path`` names a scene.
    Lx, Ly, Lz : float
        Extent of the sampling volume (default 1.0 each).
    nx, ny, nz : int
        Cells per axis over that volume (default 10 each).
    origin : sequence of 3 floats
        Lower corner of the sampling volume (default ``(0, 0, 0)``).
    snap : bool
        Project boundary vertices onto the surface.  Default ``False`` — see
        the module docstring before turning this on.
    """

    def read(self, spec: Any) -> DomainRepresentation:
        try:
            from cadjoint.fem.hexmesh import GridSpec, sdf_to_hex_mesh
        except ImportError as exc:
            raise ImportError(_import_failure_hint(exc)) from exc

        p = dict(getattr(spec, "params", None) or {})

        source = p.get("sdf")
        if source is None:
            if getattr(spec, "path", None) is None:
                raise ValueError(
                    "cadjoint geometry needs either params['sdf'] or a path to a scene file."
                )
            source = _sdf_from_path(spec.path)

        Lx, Ly, Lz = (float(p.get(k, 1.0)) for k in ("Lx", "Ly", "Lz"))
        nx, ny, nz = (int(p.get(k, 10)) for k in ("nx", "ny", "nz"))
        origin = tuple(float(v) for v in p.get("origin", (0.0, 0.0, 0.0)))
        snap = bool(p.get("snap", False))

        grid = GridSpec.from_bounds(bounds=origin, size=(Lx, Ly, Lz), resolution=(nx, ny, nz))
        hex_mesh = sdf_to_hex_mesh(_as_batched_sdf(source), grid, snap=snap)

        # Unsnapped, points and base_points coincide; snapped, base_points are
        # the lattice positions the topology was frozen against.
        vertices = np.asarray(hex_mesh.points, dtype=np.float64)
        cells = np.asarray(hex_mesh.cells, dtype=np.int64)

        spacing = np.asarray(grid.spacing, dtype=np.float64)
        metadata: Dict[str, Any] = {
            "Lx": Lx, "Ly": Ly, "Lz": Lz,
            "nx": nx, "ny": ny, "nz": nz,
            "origin": origin,
            "snapped": snap,
            # All cells are congruent only while snapping is off; FEMDiscretiser's
            # single ke_ref depends on it.
            "uniform_cells": not snap,
            "sdf": source,
            "grid": grid,
            "cadjoint_hex_mesh": hex_mesh,
            "cells_kept": int(cells.shape[0]),
            "cells_in_lattice": nx * ny * nz,
        }

        # FEniCS is optional here: without it the reader still produces a valid
        # DomainRepresentation, and FEMDiscretiser raises its own error later.
        try:
            metadata["dolfin_mesh"] = build_dolfin_hex_mesh(vertices, cells)
        except ImportError:
            metadata["dolfin_mesh_unavailable"] = "dolfin is not importable"

        return DomainRepresentation(
            vertices=vertices,
            cells=cells,
            cell_type="hex",
            dim=3,
            facets=_axis_facets(vertices, tol=1e-9 + 1e-6 * float(spacing.min())),
            metadata=metadata,
        )

    @staticmethod
    def supports(geometry_type: str) -> bool:
        return geometry_type == "cadjoint"
