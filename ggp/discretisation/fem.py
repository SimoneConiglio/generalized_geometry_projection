# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Discretisation layer — FEM mesh discretiser."""
from __future__ import annotations

from typing import Any, List

from ggp.problem.spec import ProblemSpec
from ggp.geometry.io.base import DomainRepresentation

import numpy as np


def _closest_dof_on_face(
    dof_coords: np.ndarray,
    candidate_dofs: np.ndarray,
    face_coord: float,
    face_axis: int,
    target: np.ndarray,
    face_tol: float = 1e-3,
) -> int:
    """Return the DOF in *candidate_dofs* that is on the given face and
    closest to *target*.

    Falls back to the globally closest candidate DOF if no DOF lies on the
    face within *face_tol* (e.g. when the mesh spacing is coarser than the
    tolerance).
    """
    on_face = np.abs(dof_coords[:, face_axis] - face_coord) < face_tol
    face_candidates = np.intersect1d(np.where(on_face)[0], candidate_dofs)
    if len(face_candidates) == 0:
        # Fallback: use globally closest candidate DOF
        face_candidates = candidate_dofs
    dists = np.linalg.norm(dof_coords[face_candidates] - target, axis=1)
    return int(face_candidates[np.argmin(dists)])


class AnalysisDomain:
    """Analysis-ready domain containing mesh and BCs."""
    def __init__(self, dim: int, eval_coords: np.ndarray, metadata: dict = None):
        self.dim = dim
        self.num_eval_points = eval_coords.shape[0]
        self.eval_coords = eval_coords
        self.metadata = metadata or {}
        self.mesh = None
        self.function_spaces = {}
        self.bcs_applied = []
        self.point_fixed_dofs: List[int] = []  # extra DOFs fixed via direct indexing
        self.load_vector = None
        self.empty_elements = []
        self.ke_ref = None
        # Reference (unit-E) element stress-extraction operator: cell-average Voigt
        # stress = se_ref @ u_e. Shape (n_stress, dofs_per_cell); n_stress = 3 (2D
        # plane stress: σxx,σyy,τxy) or 6 (3D: σxx,σyy,σzz,τxy,τyz,τxz).
        self.se_ref = None


class FEMDiscretiser:
    """Converts a DomainRepresentation into a FEniCS-ready AnalysisDomain."""

    def discretise(self, domain: DomainRepresentation, spec: ProblemSpec) -> AnalysisDomain:
        import dolfin as df
        import ufl

        mesh = domain.metadata.get("dolfin_mesh")
        if mesh is None:
            raise ValueError("FEMDiscretiser currently requires a FEniCS built-in mesh.")

        V_u = df.VectorFunctionSpace(mesh, "CG", 1)
        V_dg = df.FunctionSpace(mesh, "DG", 0)

        eval_coords = V_dg.tabulate_dof_coordinates()
        analysis = AnalysisDomain(dim=domain.dim, eval_coords=eval_coords, metadata=domain.metadata)
        analysis.mesh = mesh
        analysis.function_spaces["u"] = V_u
        analysis.function_spaces["dg"] = V_dg

        zero_vec = df.Constant((0.0, 0.0, 0.0)) if domain.dim == 3 else df.Constant((0.0, 0.0))
        dof_coords = V_u.tabulate_dof_coordinates()
        Lx = domain.metadata.get("Lx", 60.0)
        Ly = domain.metadata.get("Ly", 30.0)

        # DOF sub-maps (component → DOF indices)
        sub_dofs = {i: V_u.sub(i).dofmap().dofs() for i in range(domain.dim)}

        bcs = []
        point_fixed_dofs: List[int] = []

        def _on_left(x, on_bound):
            return on_bound and df.near(x[0], 0.0)

        def _on_top(x, on_bound):
            return on_bound and df.near(x[1], Ly)

        for bc in spec.boundary_conditions:
            if bc.region == "left":
                if bc.type == "fixed":
                    bcs.append(df.DirichletBC(V_u, zero_vec, _on_left))
                elif bc.type == "symmetry" and bc.components:
                    bcs.append(df.DirichletBC(
                        V_u.sub(bc.components[0]), df.Constant(0.0), _on_left
                    ))

            elif bc.region == "top":
                bcs.append(df.DirichletBC(V_u, zero_vec, _on_top))

            elif bc.region == "bottom_right_corner":
                # DirichletBC on quad meshes often misses a single corner DOF.
                # Apply directly by finding the corner DOF via coordinates.
                target = np.array([Lx, 0.0])
                dists = np.linalg.norm(dof_coords[:, :2] - target, axis=1)
                corner_nodes = np.where(dists < 1e-8)[0]
                if bc.components:
                    for comp in bc.components:
                        comp_dofs = sub_dofs[comp]
                        matched = np.intersect1d(corner_nodes, comp_dofs)
                        point_fixed_dofs.extend(matched.tolist())

        analysis.bcs_applied = bcs
        analysis.point_fixed_dofs = sorted(set(point_fixed_dofs))

        # Apply Loads
        f_vec = np.zeros(V_u.dim())
        y_dofs = sub_dofs[1]

        for ld in spec.loads:
            if ld.region == "mid_right" and ld.type == "point":
                if domain.dim == 3:
                    Lz = domain.metadata.get("Lz", 30.0)
                    target = np.array([Lx, Ly / 2.0, Lz / 2.0])
                else:
                    target = np.array([Lx, Ly / 2.0])
                tip_dof = _closest_dof_on_face(dof_coords, y_dofs, face_coord=Lx, face_axis=0, target=target)
                f_vec[tip_dof] = ld.value[1]

            elif ld.region == "top_left_corner" and ld.type == "point":
                if domain.dim == 3:
                    Lz = domain.metadata.get("Lz", 30.0)
                    target = np.array([0.0, Ly, Lz / 2.0])
                else:
                    target = np.array([0.0, Ly])
                tip_dof = _closest_dof_on_face(dof_coords, y_dofs, face_coord=0.0, face_axis=0, target=target)
                f_vec[tip_dof] = ld.value[1]

        analysis.load_vector = f_vec

        # Non-design empty elements
        for geom in spec.geometries:
            if geom.role == "non_design" and geom.type == "box":
                origin = geom.params.get("origin", [0, 0])
                centers_x = eval_coords[:, 0]
                centers_y = eval_coords[:, 1]
                mask = (centers_x > origin[0]) & (centers_y > origin[1])
                analysis.empty_elements = np.where(mask)[0].tolist()

        # Unit element stiffness matrix
        plane_stress = domain.dim == 2
        mu = 1.0 / 2.6
        if plane_stress:
            lmbda = 0.3 / (1.0 - 0.3 ** 2)
        else:
            lmbda = 0.3 / (1.3 * (1.0 - 2.0 * 0.3))

        def eps_f(u):
            return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

        def sig_f(u):
            return lmbda * ufl.tr(eps_f(u)) * ufl.Identity(domain.dim) + 2.0 * mu * eps_f(u)

        u_trial = df.TrialFunction(V_u)
        v_test = df.TestFunction(V_u)
        a = ufl.inner(sig_f(u_trial), eps_f(v_test)) * df.dx

        cell = next(df.cells(mesh))
        analysis.ke_ref = df.assemble_local(a, cell)

        # Reference stress-extraction operator (unit E), consistent with ke_ref:
        # for each Voigt component k, ∫_cell σ_k(u) dx / |cell| is a linear form in u,
        # so assemble_local returns the row of se_ref acting on the element DOFs.
        if domain.dim == 2:
            voigt_idx = [(0, 0), (1, 1), (0, 1)]                       # σxx, σyy, τxy
        else:
            voigt_idx = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
        vol = cell.volume()
        se_rows = []
        for (ii, jj) in voigt_idx:
            M = np.zeros((domain.dim, domain.dim))
            M[ii, jj] = 1.0
            a_k = ufl.inner(sig_f(u_trial), df.Constant(M)) * df.dx
            se_rows.append(np.asarray(df.assemble_local(a_k, cell)) / vol)
        analysis.se_ref = np.array(se_rows)      # (n_stress, dofs_per_cell)

        return analysis
