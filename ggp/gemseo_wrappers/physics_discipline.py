# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Refactored physics discipline using pre-assembled vectors."""
from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sps
from gemseo.core.discipline.discipline import Discipline

class GGPPhysicsDiscipline(Discipline):
    """
    High-performance Physics Discipline using Fast Assembly (petsc4py/scipy).
    Uses directly assembled load vectors and BCs from the FEMDiscretiser.
    """
    def __init__(
        self,
        V_u,
        ke_ref,
        fixed_dofs,
        f_vec,
        mesh_area: float,
        volfrac: float,
        E0: float = 1.0,
        Emin: float = 1e-6,
        p_penalty: float = 3.0,
        name: str = "GGP_Physics",
        iterative: bool = False,
        empty_elements=None,
    ):
        super().__init__(name=name)

        self.V_u = V_u
        self.ke_ref = ke_ref
        self.fixed_dofs = fixed_dofs
        self.f_vec = f_vec
        self.mesh_area = mesh_area
        self.volfrac = volfrac
        self.E0 = E0
        self.Emin = Emin
        self.p_penalty = p_penalty
        self.iterative = iterative
        # Indices of non-design (empty) elements — forced to Emin stiffness
        self.empty_elements = np.array(empty_elements, dtype=int) if empty_elements else np.array([], dtype=int)
        
        self.num_elements = V_u.mesh().num_cells()
        self.dm = self.V_u.dofmap()
        self.cell_dofs = [self.dm.cell_dofs(i) for i in range(self.num_elements)]
        
        # Apply BCs to F (zero out fixed DOFs)
        self.f_vec[self.fixed_dofs] = 0.0
        
        self.input_grammar.update_from_names(["rho_E", "rho_V"])
        self.output_grammar.update_from_names(["compliance", "volume"])
        self.last_u = None
        
        if hasattr(self, 'cache'):
            self.cache = None
        if hasattr(self, 'cache_type'):
            self.cache_type = Discipline.CacheType.NONE

    def _run(self, input_data=None):
        start = time.time()
        if input_data is not None:
            self.local_data.update(input_data)
            
        rho_E = self.local_data["rho_E"].flatten()
        rho_V = self.local_data["rho_V"].flatten()

        # Non-design (empty) elements: force rho_E → 0 so E = Emin (matches Matlab E(emptyelts)=Emin)
        if len(self.empty_elements):
            rho_E = rho_E.copy()
            rho_E[self.empty_elements] = 0.0

        # Penalize Young's Modulus
        E_vals = self.Emin + (rho_E ** self.p_penalty) * (self.E0 - self.Emin)
        
        dofs_per_cell = self.ke_ref.shape[0]
        rows = np.repeat(self.cell_dofs, dofs_per_cell).reshape(self.num_elements, dofs_per_cell, dofs_per_cell)
        cols = np.tile(self.cell_dofs, (1, dofs_per_cell)).reshape(self.num_elements, dofs_per_cell, dofs_per_cell)
        data = np.outer(E_vals, self.ke_ref).reshape(self.num_elements, dofs_per_cell, dofs_per_cell)
        
        K_global = sps.coo_matrix(
            (data.flatten(), (rows.flatten(), cols.flatten())), 
            shape=(self.V_u.dim(), self.V_u.dim())
        ).tocsr()
        
        # Apply Dirichlet BCs
        col_mask = np.isin(K_global.indices, self.fixed_dofs)
        K_global.data[col_mask] = 0.0
        
        for dof in self.fixed_dofs:
            dof_start = K_global.indptr[dof]
            dof_end = K_global.indptr[dof+1]
            K_global.data[dof_start:dof_end] = 0.0
            col_indices = K_global.indices[dof_start:dof_end]
            diag_idx = np.where(col_indices == dof)[0]
            if len(diag_idx) > 0:
                K_global.data[dof_start + diag_idx[0]] = 1.0

        # Pin isolated material-free DOFs. With Emin=0 (the GP method) a node whose
        # every incident element is void (E=0) — e.g. the interior of a forced
        # non-design region such as the L-shape bracket's empty quadrant — has an
        # all-zero stiffness row and column, which makes K singular (spsolve -> NaN).
        # Such DOFs carry neither load nor stiffness, so pin them to zero with an
        # identity row. This keeps the system solvable without floating the void on a
        # fake Emin stiffness, so the load-carrying structure is modelled exactly.
        diag = K_global.diagonal()
        isolated_dofs = np.where(diag == 0.0)[0]
        if isolated_dofs.size:
            K_global = K_global + sps.csr_matrix(
                (np.ones(isolated_dofs.size), (isolated_dofs, isolated_dofs)),
                shape=K_global.shape,
            )

        # Solve K U = F
        if self.iterative:
            from petsc4py import PETSc
            size = self.V_u.dim()
            A = PETSc.Mat().createAIJ(size=(size, size), csr=(K_global.indptr, K_global.indices, K_global.data))
            b = PETSc.Vec().createWithArray(self.f_vec)
            x = PETSc.Vec().createWithArray(np.zeros_like(self.f_vec))
            
            ksp = PETSc.KSP().create()
            ksp.setOperators(A)
            ksp.setType(PETSc.KSP.Type.CG)
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.GAMG)
            ksp.setTolerances(rtol=1e-8)
            ksp.solve(b, x)
            u_vec = x.getArray().copy()
        else:
            from scipy.sparse.linalg import spsolve
            u_vec = spsolve(K_global, self.f_vec)
            
        self.last_u = u_vec
        
        # Compliance
        compliance = np.dot(self.f_vec, u_vec)

        # Volume fraction
        volume = np.sum(rho_V) / self.num_elements

        # Log-transform objective — matches Matlab GGP_main.m: f0val = log(c+1)
        self.local_data["compliance"] = np.array([np.log(compliance + 1.0)])
        # Normalised, ×100-scaled volume constraint — matches Matlab: fval = (v-volfrac)/volfrac * 100
        self.local_data["volume"] = np.array([(volume - self.volfrac) / self.volfrac * 100.0])

        # Gradients
        dE_drho = self.p_penalty * (rho_E ** (self.p_penalty - 1.0)) * (self.E0 - self.Emin)
        grad_C = np.zeros(self.num_elements)
        for i in range(self.num_elements):
            u_e = u_vec[self.cell_dofs[i]]
            grad_C[i] = -dE_drho[i] * np.dot(u_e, np.dot(self.ke_ref, u_e))

        # Suppress gradient from non-design elements (forced rho_E=0 → gradient is fictitious)
        if len(self.empty_elements):
            grad_C[self.empty_elements] = 0.0

        # Log-transform gradient: d(log(C+1))/drho_E = (dC/drho_E) / (C+1)
        self.dj_drhoE = grad_C / (compliance + 1.0)
        # Volume gradient scaled by 100/volfrac to match ×100 constraint
        self.dv_drhoV = np.ones(self.num_elements) * (100.0 / (self.volfrac * self.num_elements))

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        self.jac = {
            "compliance": {"rho_E": self.dj_drhoE.reshape(1, -1), "rho_V": np.zeros((1, self.num_elements))},
            "volume": {"rho_E": np.zeros((1, self.num_elements)), "rho_V": self.dv_drhoV.reshape(1, -1)}
        }
