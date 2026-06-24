# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Re-evaluate a converged GGP design on a refined grid for clean plots.

The optimiser evaluates the density at the FE element centres.  When the element
size is comparable to a bar's thickness, a bar whose axis runs diagonally to the
mesh aliases into a dotted/broken isosurface — purely a sampling artefact, not a
projection error.  This module re-evaluates the (pure-NumPy) geometry mapper on a
finer structured grid so the rendered geometry is continuous, independent of the
FE resolution used for the optimisation.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import numpy as np

from ggp.problem.spec import ProblemSpec


def refined_density(
    spec: ProblemSpec,
    design_variables: np.ndarray,
    refine: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(rho_E, eval_coords)`` for *design_variables* on a refined grid.

    The design geometry's mesh resolution (``nx``/``ny``/``nz``) is multiplied by
    *refine* and the geometry discipline is rebuilt on that finer mesh.  Only the
    geometry projection is evaluated — there is no FE solve — so this is cheap.

    Parameters
    ----------
    spec : ProblemSpec
        The problem that produced the design (used for the domain + formulation).
    design_variables : array
        The optimal design vector ``x_opt`` (``OptimisationResult.design_variables``).
    refine : int
        Mesh refinement factor for the render grid (>= 1).
    """
    from ggp.geometry.io.registry import get_reader
    from ggp.discretisation.fem import FEMDiscretiser
    from ggp.gemseo_wrappers.geometry_discipline import GGPGeometryDiscipline

    design_geom = next(g for g in spec.geometries if g.role == "design")
    params = dict(design_geom.params)
    for axis in ("nx", "ny", "nz"):
        if axis in params and params[axis]:
            params[axis] = int(params[axis]) * int(refine)
    fine_geom = replace(design_geom, params=params)

    domain = get_reader(fine_geom.type).read(fine_geom)
    analysis = FEMDiscretiser().discretise(domain, replace(spec, geometries=[fine_geom]))

    geom = GGPGeometryDiscipline(
        mesh=analysis.mesh,
        num_components=_num_components(spec, design_variables),
        mode=spec.formulation.mode,
        ka=10.0,
        pp=100.0,
        method=spec.formulation.method,
        r_gp=spec.formulation.r_gp,
    )
    geom.execute({"x_vars": np.asarray(design_variables)})
    rho_E = np.asarray(geom.local_data["rho_E"]).flatten()
    return rho_E, geom.eval_coords.copy()


def _num_components(spec: ProblemSpec, design_variables: np.ndarray) -> int:
    """Recover the component count from the design-vector length."""
    from ggp.projection.registry import get_mapper

    mapper = get_mapper(spec.formulation.mode, num_components=1, r_gp=spec.formulation.r_gp)
    vpc = mapper.num_vars_per_component()
    return int(len(design_variables) // vpc)
