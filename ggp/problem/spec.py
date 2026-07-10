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
    type : {"point", "patch", "pressure", "traction", "body"}
        Load application mode. ``"patch"`` spreads the total force uniformly over the
        boundary nodes within ``width`` of the region's reference point — the standard
        way to avoid the point-load stress singularity in stress-constrained problems.
    value : list of float
        Force/pressure vector (total force for ``"point"``/``"patch"``).
    width : float
        Extent of the ``"patch"`` load along the loaded face (ignored otherwise).
    """
    region: str
    type: Literal["point", "patch", "pressure", "traction", "body"] = "point"
    value: List[float] = field(default_factory=lambda: [0.0, -1.0])
    width: float = 0.0


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
    method : {"GP", "WGP", "rect"}
        Geometry projection method / primitive variant. ``"GP"`` is the round-ended
        bar (capsule); ``"rect"`` is a true rectangle primitive (quintic-smoothed,
        width ``L`` x height ``h``).
    init : {"default", "grid"}
        Initial-design layout. ``"default"`` is the Matlab-style crossed-bar grid;
        ``"grid"`` tiles the domain with ``grid_nx`` x ``grid_ny`` rectangles
        (squares when the tile aspect matches the domain) at density = ``volfrac``.
    grid_nx, grid_ny : int, optional
        Number of tiles along x and y for the ``"grid"`` initialisation. Their
        product must equal ``num_components`` and both must be < the mesh resolution.
    """
    mode: Literal["Free", "ALM", "3D_Free", "3D_ALM", "Truss", "2D_Truss",
                  "Truss3D", "3D_Truss"] = "Free"
    num_components: int = 18
    num_layers: Optional[int] = None
    comp_per_layer: Optional[int] = None
    layer_height: Optional[float] = None
    r_gp: float = 0.5
    Ngp: int = 2
    method: str = "GP"
    init: str = "default"
    grid_nx: Optional[int] = None
    grid_ny: Optional[int] = None
    # Minimum member thickness (in physical/mesh units) -> a minimum length scale.
    # Raises the lower bound on the primitive thickness `h` so members are born thick
    # enough to threshold to a clean, connected 0/1 design. None -> framework default (1).
    min_thickness: Optional[float] = None
    # Enforce a design symmetry by mirror-pairing components. "y" -> top-bottom
    # symmetry about Ly/2 (the optimiser controls num_components/2 free components).
    # None -> no symmetry. Free 2D only.
    symmetry: Optional[str] = None
    # Truss mode only: node-pair connection radius for the ground structure. A bar
    # is created for every node pair within this distance. None -> ~1 tile-diagonal
    # (each node connects to its 8 grid neighbours); larger -> denser, approaching
    # the complete graph. grid_nx x grid_ny give the NODE lattice in Truss mode.
    truss_radius: Optional[float] = None
    # 3-D truss only: number of nodes along z (grid_nx x grid_ny x grid_nz lattice).
    grid_nz: Optional[int] = None
    # Fix every component's mass density Mc to 1 (bounds [1, 1]): components are
    # SOLID bars and the design changes only by geometry (move/resize/reorient) --
    # the Lagrangian-native "binary" design space. Essential for active stress
    # constraints: intermediate Mc lets the optimizer fade overstressed material
    # (sigma_hat ~ 1/rho^p rewards removal in the relaxed constraint), while solid
    # bars must relieve stress by reshaping. Free 2D only.
    fix_mc: bool = False


# ── Optimisation ──────────────────────────────────────────────────────────────

# Discipline outputs ("responses") that can be selected as the objective or as a
# constraint. compliance/volume always exist; displacement/stress disciplines are
# built on demand (see GGPPipeline.run) whenever they're needed by either role.
KNOWN_RESPONSES = ("compliance", "volume", "displacement", "stress")


@dataclass(frozen=True)
class ObjectiveSpec:
    """Objective function to minimise or maximise.

    Parameters
    ----------
    name : str
        Response identifier — one of :data:`KNOWN_RESPONSES`
        (``"compliance"``, ``"volume"``, ``"displacement"``, ``"stress"``).
    sense : {"minimize", "maximize"}
        Optimisation direction.
    params : dict
        Extra parameters for responses that need them when used as the objective
        without also being a constraint (e.g. ``P``/``aggregation`` for ``"stress"``,
        ``axis``/``point`` for ``"displacement"``) — mirrors :attr:`ConstraintSpec.params`.
    """
    name: str = "compliance"
    sense: Literal["minimize", "maximize"] = "minimize"
    params: Dict[str, Any] = field(default_factory=dict)


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
        Optimisation algorithm name (``"MMA"``, ``"SLP"``, ``"CONLIN"``,
        ``"GE_SBO"``).
    max_iter : int
        Maximum number of outer iterations.
    iterative : bool
        Deprecated shorthand for ``fem_solver="iterative"``. Kept for
        backward compatibility with existing YAML files and CLI.
    fem_solver : {"direct", "iterative", "amjax"}
        FEM linear-system backend.  ``"direct"`` uses
        :func:`scipy.sparse.linalg.spsolve` (LU); ``"iterative"`` uses
        PETSc CG + GAMG; ``"amjax"`` uses PyAMG + AMJax (JAX-accelerated
        algebraic multigrid, JIT-compiled and GPU-compatible).
    use_line_search : bool
        Enable line search (SLP / CONLIN only).
    options : dict
        Additional algorithm-specific key-value options.
    """
    algorithm: str = "MMA"
    max_iter: int = 50
    iterative: bool = False
    fem_solver: Literal["direct", "iterative", "amjax"] = "direct"
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
