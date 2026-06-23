# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Serialisable dataclass specifications for GGP optimisation problems.

Every optimisation run is fully described by a single :class:`ProblemSpec`
instance.  This object can be constructed programmatically, loaded from a
YAML file, or built from CLI arguments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


# ── Geometry ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GeometrySpec:
    """A geometry region (design-space or non-design-space).

    Supported *type* values (extensible via the reader registry):

    ====================== ====================================================
    type                   Description
    ====================== ====================================================
    ``box``                Parametric axis-aligned box (2-D or 3-D)
    ``fenics_rectangle``   FEniCS built-in quadrilateral rectangle
    ``fenics_box``         FEniCS built-in hexahedral box
    ``step``               STEP CAD file (requires ``gmsh`` or ``cadquery``)
    ``stl``                STL surface mesh (requires ``meshio``)
    ``brep``               OpenCASCADE BREP file
    ``points``             Raw point cloud (``numpy`` only)
    ====================== ====================================================

    Parameters
    ----------
    type : str
        Geometry source type.
    role : {"design", "non_design"}
        Whether this region participates in the design optimisation.
    path : Path, optional
        File path for file-based formats (STEP, STL, …).
    params : dict
        Format-specific parameters (e.g. ``{"Lx": 60, "Ly": 30, "nx": 60, "ny": 30}``).
    """
    type: str
    role: Literal["design", "non_design"] = "design"
    path: Optional[Path] = None
    params: Dict[str, Any] = field(default_factory=dict)


# ── Boundary Conditions & Loads ───────────────────────────────────────────────

@dataclass(frozen=True)
class BoundaryCondition:
    """A Dirichlet boundary condition applied to a named region.

    Parameters
    ----------
    region : str
        Boundary identifier (e.g. ``"left"``, ``"top"``, ``"fixed_face"``).
        Mapped to concrete selection logic by the discretiser.
    type : {"fixed", "symmetry", "prescribed"}
        Kind of constraint.
    components : list of int, optional
        DOF components to constrain (0 = x, 1 = y, 2 = z).
        ``None`` means all components.
    value : list of float, optional
        Prescribed displacement for ``type="prescribed"``.
    """
    region: str
    type: Literal["fixed", "symmetry", "prescribed"] = "fixed"
    components: Optional[List[int]] = None
    value: Optional[List[float]] = None


@dataclass(frozen=True)
class Load:
    """A mechanical load applied to a named region.

    Parameters
    ----------
    region : str
        Region identifier (e.g. ``"mid_right"``, ``"top_face"``).
    type : {"point", "pressure", "traction", "body"}
        Load application mode.
    value : list of float
        Force/pressure vector.
    """
    region: str
    type: Literal["point", "pressure", "traction", "body"] = "point"
    value: List[float] = field(default_factory=lambda: [0.0, -1.0])


# ── Formulation ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FormulationSpec:
    """GGP projection formulation specification.

    Parameters
    ----------
    mode : {"Free", "ALM", "3D_Free", "3D_ALM"}
        Projection mode.
    num_components : int
        Total number of geometric primitives.
    num_layers : int, optional
        Number of manufacturing layers (ALM modes only).
    comp_per_layer : int, optional
        Components per layer (ALM modes only).
    layer_height : float, optional
        Height of each layer (ALM modes only).
    r_gp : float
        Sampling-window radius for Gauss-point integration.
    Ngp : int
        Number of Gauss points per dimension.
    method : {"GP", "WGP"}
        Geometry projection method variant.
    """
    mode: Literal["Free", "ALM", "3D_Free", "3D_ALM"] = "Free"
    num_components: int = 18
    num_layers: Optional[int] = None
    comp_per_layer: Optional[int] = None
    layer_height: Optional[float] = None
    r_gp: float = 0.5
    Ngp: int = 2
    method: str = "GP"


# ── Optimisation ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ObjectiveSpec:
    """Objective function to minimise.

    Parameters
    ----------
    name : str
        Objective identifier (e.g. ``"compliance"``).
    """
    name: str = "compliance"


@dataclass(frozen=True)
class ConstraintSpec:
    """An optimisation constraint.

    Parameters
    ----------
    name : str
        Constraint identifier (``"volume"``, ``"overhang"``, ``"stress"``, …).
    type : {"ineq", "eq"}
        Inequality or equality.
    bound : float
        Right-hand-side value (constraint ≤ bound for ineq).
    params : dict
        Extra parameters (e.g. ``{"alpha_deg": 45}`` for overhang).
    """
    name: str
    type: Literal["ineq", "eq"] = "ineq"
    bound: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolverSpec:
    """Optimisation solver / algorithm configuration.

    Parameters
    ----------
    algorithm : str
        Optimisation algorithm name (``"MMA"``, ``"SLP"``, ``"CONLIN"``).
    max_iter : int
        Maximum number of outer iterations.
    iterative : bool
        Use PETSc iterative (PCG + GAMG) solver instead of direct LU.
    use_line_search : bool
        Enable line search (SLP / CONLIN only).
    options : dict
        Additional algorithm-specific key-value options.
    """
    algorithm: str = "MMA"
    max_iter: int = 50
    iterative: bool = False
    use_line_search: bool = False
    options: Dict[str, Any] = field(default_factory=dict)


# ── Material ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MaterialSpec:
    """Isotropic material properties.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    """
    E: float = 1.0
    nu: float = 0.3


# ── Initial guess ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InitGuessSpec:
    """Initial-guess (component seeding) configuration.

    Parameters
    ----------
    pattern : str, optional
        Seeding strategy.  ``None`` selects a dimension-aware default
        (``"tri2d"`` for 2-D Free, ``"tet3d"`` for 3-D Free, legacy staircase
        for ALM).  ``"grid"`` reproduces the legacy crossed-bar grid.  Any
        registered mesh pattern name (``"tri2d"``, ``"quad_star"``, ``"tet3d"``,
        ``"hex_star"``) tessellates the domain and seeds one bar per edge.
    cell_size : float, optional
        Tessellation cell size for mesh patterns.  Drives the component count.
        ``None`` selects a dimension-aware default.
    thickness : float, optional
        Fixed initial bar thickness.  ``None`` fits the thickness so the seeded
        design starts near ``volfrac`` (see ``fit_volfrac``).
    membership : float
        Initial membership / density of each seeded bar.
    fit_volfrac : bool
        If ``True`` (and ``thickness`` is ``None``), bisect the thickness so the
        projected initial volume fraction matches the target ``volfrac``.
    params : dict
        Extra pattern-specific parameters.
    """
    pattern: Optional[str] = None
    cell_size: Optional[float] = None
    thickness: Optional[float] = None
    membership: float = 1.0
    fit_volfrac: bool = True
    params: Dict[str, Any] = field(default_factory=dict)



# ── Top-level ProblemSpec ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProblemSpec:
    """Complete, self-contained optimisation problem definition.

    This is the **single source of truth** for any GGP run.  The CLI,
    Python API, and YAML loader all produce a ``ProblemSpec``.

    Parameters
    ----------
    geometries : list of GeometrySpec
        Domain geometry regions (design + non-design).
    boundary_conditions : list of BoundaryCondition
        Dirichlet boundary conditions.
    loads : list of Load
        Applied loads.
    formulation : FormulationSpec
        GGP formulation / projection configuration.
    objective : ObjectiveSpec
        Objective function.
    constraints : list of ConstraintSpec
        Optimisation constraints.
    solver : SolverSpec
        Algorithm and solver configuration.
    volfrac : float
        Target volume fraction.
    material : MaterialSpec
        Material properties.
    """
    geometries: List[GeometrySpec]
    boundary_conditions: List[BoundaryCondition]
    loads: List[Load]
    formulation: FormulationSpec
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    constraints: List[ConstraintSpec] = field(default_factory=lambda: [
        ConstraintSpec(name="volume", type="ineq", bound=0.0)
    ])
    solver: SolverSpec = field(default_factory=SolverSpec)
    volfrac: float = 0.4
    material: MaterialSpec = field(default_factory=MaterialSpec)
    init: InitGuessSpec = field(default_factory=InitGuessSpec)
