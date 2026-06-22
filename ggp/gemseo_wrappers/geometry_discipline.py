# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Refactored geometry discipline using projection mappers."""
from __future__ import annotations

import numpy as np
from gemseo.core.discipline.discipline import Discipline

from ggp.projection.registry import get_mapper
from ggp.utils.vectorized_mapping import smooth_saturation_np


class GGPGeometryDiscipline(Discipline):
    """GEMSEO discipline for GGP geometry projection.
    
    Delegates the core maths to a ProjectionMapper.
    """

    def __init__(
        self,
        mesh,
        num_components: int,
        mode: str = "Free",
        ka: float = 10.0,
        pp: float = 10.0,
        gammac: float = 3.0,
        gammav: float = 1.0,
        name: str = "GGP_Geometry",
        **kwargs,
    ):
        super().__init__(name=name)
        
        self.ka = ka
        self.pp = pp
        self.gammac = gammac
        self.gammav = gammav
        
        # Instantiate the correct mapper
        if mode == "ALM" and mesh.geometry().dim() == 2:
            mode = "2D_ALM"
        self.mapper = get_mapper(mode, num_components=num_components, **kwargs)
        
        # Extract evaluation coordinates (element centroids)
        import dolfin as df
        V_dg = df.FunctionSpace(mesh, "DG", 0)
        self.eval_coords = V_dg.tabulate_dof_coordinates()
        self.num_elements = self.eval_coords.shape[0]
        self.dim = self.eval_coords.shape[1]
        
        # Determine domain extents
        Lx = self.eval_coords[:, 0].max() - self.eval_coords[:, 0].min()
        Ly = self.eval_coords[:, 1].max() - self.eval_coords[:, 1].min()
        extents = (Lx, Ly)
        if self.dim == 3:
            Lz = self.eval_coords[:, 2].max() - self.eval_coords[:, 2].min()
            extents = (Lx, Ly, Lz)
            
        self.lb, self.ub = self.mapper.default_bounds(extents, num_components)
        
        # Override with explicit bounds if provided
        if "lb" in kwargs and kwargs["lb"] is not None:
            self.lb = kwargs["lb"]
        if "ub" in kwargs and kwargs["ub"] is not None:
            self.ub = kwargs["ub"]

        # KS Saturation reference value
        xt = kwargs.get('xt', 1.0 + 1.0/ka * np.log((1.0 + (num_components - 1.0)*np.exp(-ka))/num_components))
        s0 = -np.log(np.exp(-pp) + 1.0 / (np.exp(0.0) + 1.0)) / pp
        self.sat_params = (xt, s0)

        self.input_grammar.update_from_names(["x_vars"])
        self.output_grammar.update_from_names(["rho_E", "rho_V"])

        if hasattr(self, 'cache'):
            self.cache = None
        if hasattr(self, 'cache_type'):
            self.cache_type = Discipline.CacheType.NONE

    def _ks_saturation(self, char_funcs: np.ndarray) -> np.ndarray:
        sum_exp = np.mean(np.exp(self.ka * char_funcs), axis=0)
        ks_val = (1.0 / self.ka) * np.log(sum_exp)
        xt, s0 = self.sat_params
        return smooth_saturation_np(ks_val, self.ka, self.pp, xt, s0)

    def _ks_saturation_grad(
        self,
        char_funcs: np.ndarray,
        grads: np.ndarray,
        global_col: bool = False,
    ) -> np.ndarray:
        """Compute dSaturation/dx_vars as a sparse (n_elements, n_vars) CSR matrix.

        Parameters
        ----------
        char_funcs : (num_comp, n_elements)
        grads      : (num_comp, vars_per_comp, n_elements)
        global_col : if True, use column index j directly (continuous ALM mode where
                     grads[i, j, :] is wrt global variable j, not i*vars_per_comp+j)
        """
        sum_exp = np.mean(np.exp(self.ka * char_funcs), axis=0)
        ks_val = (1.0 / self.ka) * np.log(sum_exp)

        dKS_dV = np.exp(self.ka * char_funcs) / (len(char_funcs) * sum_exp)

        xt, s0 = self.sat_params
        inner_exp = np.exp((self.pp * ks_val) / xt)
        ds_dxs = (inner_exp / (inner_exp + 1.0)**2) / (
            xt * (np.exp(-self.pp) + 1.0 / (inner_exp + 1.0))
        ) / (1.0 - s0)

        import scipy.sparse as sps

        num_comp    = grads.shape[0]
        vars_per_comp = grads.shape[1]
        n_cols = vars_per_comp if global_col else num_comp * vars_per_comp

        rows, cols, data = [], [], []

        for i in range(num_comp):
            base_factor = ds_dxs * dKS_dV[i]
            for j in range(vars_per_comp):
                grad_array = base_factor * grads[i, j, :]
                mask = np.abs(grad_array) > 1e-12
                if np.any(mask):
                    nz_idx = np.where(mask)[0]
                    rows.append(nz_idx)
                    col_idx = j if global_col else i * vars_per_comp + j
                    cols.append(np.full_like(nz_idx, col_idx))
                    data.append(grad_array[mask])

        if rows:
            total_jac = sps.coo_matrix(
                (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.num_elements, n_cols),
            )
        else:
            total_jac = sps.coo_matrix((self.num_elements, n_cols))

        return total_jac.tocsr()

    def _run(self, input_data=None):
        if input_data is not None:
            self.local_data.update(input_data)
            
        x_vars = self.local_data["x_vars"].flatten()
        x_unscaled = self.lb + x_vars * (self.ub - self.lb)
        
        # Forward pass
        funcs_E, funcs_V = self.mapper.forward(x_unscaled, self.eval_coords, self.gammac, self.gammav)
        self.local_data["rho_E"] = self._ks_saturation(funcs_E)
        self.local_data["rho_V"] = self._ks_saturation(funcs_V)
        
        # Jacobian pass
        jac_dict = self.mapper.jacobian(x_unscaled, self.eval_coords, self.gammac, self.gammav)
        
        gc = jac_dict.get("is_continuous", False)
        jac_E_unscaled = self._ks_saturation_grad(jac_dict["funcs_E"], jac_dict["grads_E"], global_col=gc)
        jac_V_unscaled = self._ks_saturation_grad(jac_dict["funcs_V"], jac_dict["grads_V"], global_col=gc)
        
        # Chain rule for scaling
        import scipy.sparse as sps
        scale_diag = sps.diags(self.ub - self.lb)
        self.jac_E = jac_E_unscaled.dot(scale_diag)
        self.jac_V = jac_V_unscaled.dot(scale_diag)

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        self.jac = {
            "rho_E": {"x_vars": self.jac_E}, 
            "rho_V": {"x_vars": self.jac_V}
        }
