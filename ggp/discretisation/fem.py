# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Discretisation layer — FEM mesh discretiser."""
from __future__ import annotations

from typing import Any

from ggp.problem.spec import ProblemSpec
from ggp.geometry.io.base import DomainRepresentation

import numpy as np


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
        self.load_vector = None
        self.empty_elements = []
        self.ke_ref = None


class FEMDiscretiser:
    """Converts a DomainRepresentation into a FEniCS-ready AnalysisDomain."""

    def discretise(self, domain: DomainRepresentation, spec: ProblemSpec) -> AnalysisDomain:
        import dolfin as df
        import ufl
        
        # Recover mesh from the FEniCS geometry reader
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
        
        # Apply Boundary Conditions
        zero_vec = df.Constant((0.0, 0.0, 0.0)) if domain.dim == 3 else df.Constant((0.0, 0.0))
        dof_coords = V_u.tabulate_dof_coordinates()
        
        bcs = []
        Lx = domain.metadata.get("Lx", 60.0)
        Ly = domain.metadata.get("Ly", 30.0)
        
        for bc in spec.boundary_conditions:
            if bc.region == "left":
                def on_boundary(x, on_bound): return on_bound and df.near(x[0], 0.0)
                if bc.type == "fixed":
                    bcs.append(df.DirichletBC(V_u, zero_vec, on_boundary))
                elif bc.type == "symmetry" and bc.components:
                    bcs.append(df.DirichletBC(V_u.sub(bc.components[0]), df.Constant(0.0), on_boundary))
                    
            elif bc.region == "top":
                def on_boundary(x, on_bound): return on_bound and df.near(x[1], Ly)
                bcs.append(df.DirichletBC(V_u, zero_vec, on_boundary))
                
            elif bc.region == "bottom_right_corner":
                def on_boundary(x, on_bound): return on_bound and df.near(x[0], Lx) and df.near(x[1], 0.0)
                if bc.components:
                    bcs.append(df.DirichletBC(V_u.sub(bc.components[0]), df.Constant(0.0), on_boundary))
                    
        analysis.bcs_applied = bcs
        
        # Apply Loads
        f_vec = np.zeros(V_u.dim())
        y_dofs = V_u.sub(1).dofmap().dofs()
        
        for ld in spec.loads:
            if ld.region == "mid_right" and ld.type == "point":
                if domain.dim == 3:
                    Lz = domain.metadata.get("Lz", 30.0)
                    target = np.array([Lx, Ly/2.0, Lz/2.0])
                else:
                    target = np.array([Lx, Ly/2.0])
                dists = np.linalg.norm(dof_coords - target, axis=1)
                tip_dof = np.intersect1d(np.where(dists < 1e-3)[0], y_dofs)[0]
                f_vec[tip_dof] = ld.value[1]
                
            elif ld.region == "top_left_corner" and ld.type == "point":
                if domain.dim == 3:
                    Lz = domain.metadata.get("Lz", 30.0)
                    target = np.array([0.0, Ly, Lz/2.0])
                else:
                    target = np.array([0.0, Ly])
                dists = np.linalg.norm(dof_coords - target, axis=1)
                tip_dof = np.intersect1d(np.where(dists < 1e-3)[0], y_dofs)[0]
                f_vec[tip_dof] = ld.value[1]
                
        analysis.load_vector = f_vec
        
        # Non-design empty elements logic
        for geom in spec.geometries:
            if geom.role == "non_design" and geom.type == "box":
                origin = geom.params.get("origin", [0, 0])
                extent = geom.params.get("extent", [Lx, Ly])
                
                centers_x = eval_coords[:, 0]
                centers_y = eval_coords[:, 1]
                
                mask = (centers_x > origin[0]) & (centers_y > origin[1])
        # Compute Unit Element Stiffness (ke_ref)
        plane_stress = True
        mu = 1.0 / 2.6
        if plane_stress and domain.dim == 2:
            lmbda = 0.3 / (1.0 - 0.3**2)
        else:
            lmbda = 0.3 / (1.3 * (1.0 - 2.0*0.3))
            
        def eps_f(u): return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        def sig_f(u): return lmbda * ufl.tr(eps_f(u)) * ufl.Identity(domain.dim) + 2.0 * mu * eps_f(u)
        
        u_trial = df.TrialFunction(V_u)
        v_test = df.TestFunction(V_u)
        a = ufl.inner(sig_f(u_trial), eps_f(v_test)) * df.dx
        
        # Assemble for the first cell assuming structured grid
        cell = next(df.cells(mesh))
        analysis.ke_ref = df.assemble_local(a, cell)
                
        return analysis
