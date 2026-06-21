# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Problem definition module for GGP optimisation."""
from .spec import (
    GeometrySpec,
    BoundaryCondition,
    Load,
    FormulationSpec,
    ObjectiveSpec,
    ConstraintSpec,
    SolverSpec,
    ProblemSpec,
)
from .loader import load_problem

__all__ = [
    "GeometrySpec",
    "BoundaryCondition",
    "Load",
    "FormulationSpec",
    "ObjectiveSpec",
    "ConstraintSpec",
    "SolverSpec",
    "ProblemSpec",
    "load_problem",
]
