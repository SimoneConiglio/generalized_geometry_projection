# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Load a :class:`ProblemSpec` from a YAML (or JSON) file."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .spec import (
    BoundaryCondition,
    ConstraintSpec,
    FormulationSpec,
    GeometrySpec,
    Load,
    MaterialSpec,
    ObjectiveSpec,
    ProblemSpec,
    SolverSpec,
)


def _build_geometry(raw: Dict[str, Any]) -> GeometrySpec:
    path = Path(raw["path"]) if "path" in raw else None
    return GeometrySpec(
        type=raw["type"],
        role=raw.get("role", "design"),
        path=path,
        params=raw.get("params", {}),
    )


def _build_bc(raw: Dict[str, Any]) -> BoundaryCondition:
    return BoundaryCondition(
        region=raw["region"],
        type=raw.get("type", "fixed"),
        components=raw.get("components"),
        value=raw.get("value"),
    )


def _build_load(raw: Dict[str, Any]) -> Load:
    return Load(
        region=raw["region"],
        type=raw.get("type", "point"),
        value=raw.get("value", [0.0, -1.0]),
    )


def _build_formulation(raw: Dict[str, Any]) -> FormulationSpec:
    return FormulationSpec(
        mode=raw.get("mode", "Free"),
        num_components=raw.get("num_components", 18),
        num_layers=raw.get("num_layers"),
        comp_per_layer=raw.get("comp_per_layer"),
        layer_height=raw.get("layer_height"),
        r_gp=raw.get("r_gp", 0.5),
        Ngp=raw.get("Ngp", 2),
        method=raw.get("method", "GP"),
        init=raw.get("init", "default"),
        grid_nx=raw.get("grid_nx"),
        grid_ny=raw.get("grid_ny"),
        min_thickness=raw.get("min_thickness"),
        symmetry=raw.get("symmetry"),
    )


def _build_constraint(raw: Dict[str, Any]) -> ConstraintSpec:
    return ConstraintSpec(
        name=raw["name"],
        type=raw.get("type", "ineq"),
        bound=raw.get("bound", 0.0),
        params=raw.get("params", {}),
    )


def _build_solver(raw: Dict[str, Any]) -> SolverSpec:
    return SolverSpec(
        algorithm=raw.get("algorithm", "MMA"),
        max_iter=raw.get("max_iter", 50),
        iterative=raw.get("iterative", False),
        fem_solver=raw.get("fem_solver", "direct"),
        use_line_search=raw.get("use_line_search", False),
        options=raw.get("options", {}),
    )


def _build_material(raw: Dict[str, Any]) -> MaterialSpec:
    return MaterialSpec(
        E=raw.get("E", 1.0),
        nu=raw.get("nu", 0.3),
    )


def load_problem(source: Union[str, Path, Dict[str, Any]]) -> ProblemSpec:
    """Load a :class:`ProblemSpec` from a YAML/JSON file or a raw dict.

    Parameters
    ----------
    source : str, Path, or dict
        Path to a YAML/JSON file, or an already-parsed dictionary.

    Returns
    -------
    ProblemSpec
        Fully constructed problem specification.

    Examples
    --------
    >>> spec = load_problem("problem.yaml")
    >>> spec = load_problem({"geometries": [{"type": "fenics_rectangle", ...}], ...})
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    else:
        data = source

    geometries = [_build_geometry(g) for g in data["geometries"]]
    boundary_conditions = [_build_bc(bc) for bc in data.get("boundary_conditions", [])]
    loads = [_build_load(ld) for ld in data.get("loads", [])]
    formulation = _build_formulation(data.get("formulation", {}))

    objective_raw = data.get("objective", {})
    objective = ObjectiveSpec(name=objective_raw.get("name", "compliance"))

    constraints = [_build_constraint(c) for c in data.get("constraints", [])]
    if not constraints:
        constraints = [ConstraintSpec(name="volume", type="ineq", bound=0.0)]

    solver = _build_solver(data.get("solver", {}))
    material = _build_material(data.get("material", {}))
    volfrac = data.get("volfrac", 0.4)

    return ProblemSpec(
        geometries=geometries,
        boundary_conditions=boundary_conditions,
        loads=loads,
        formulation=formulation,
        objective=objective,
        constraints=constraints,
        solver=solver,
        volfrac=volfrac,
        material=material,
    )
