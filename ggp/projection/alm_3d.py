# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Additive Layer Manufacturing (ALM) 3-D geometry projection mapper.

Maps components layer-by-layer.
6 variables per component: [Xc, Yc, L, W, Theta, Mc].
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ggp.utils.vectorized_mapping import (
    compute_local_characteristic_np as compute_local_char_3d,
)
from ggp.utils.vectorized_mapping_3d import (
    compute_local_characteristic_3d_alm_with_grad_np,
)
from .base import ProjectionMapper
from .registry import register_mapper


@register_mapper("3D_ALM")
class ALM3DMapper(ProjectionMapper):
    """ALM 3-D projection mapper.
    
    Components are arranged in layers along the Z axis.
    Each component uses 6 parameters:
      [Xc, Yc, length, width, angle, density]
    """

    def __init__(
        self,
        num_components: int,
        num_layers: int,
        comp_per_layer: int,
        layer_height: float,
        r_gp: float = 0.5,
        method: str = "GP",
        Ngp: int = 2,
        **kwargs,
    ):
        self._num_components = num_components
        self.num_layers = num_layers
        self.comp_per_layer = comp_per_layer
        self.layer_height = layer_height
        self.r_gp = r_gp
        self.method = method
        self.Ngp = Ngp

        pts, wts = np.polynomial.legendre.leggauss(self.Ngp)
        pts = pts * self.r_gp
        wts = wts * self.r_gp
        gpcx, gpcy, gpcz = np.meshgrid(pts, pts, pts)
        self.gpc_x_rel = gpcx.flatten()
        self.gpc_y_rel = gpcy.flatten()
        self.gpc_z_rel = gpcz.flatten()
        
        w1 = (wts[:, np.newaxis] * wts[np.newaxis, :]).flatten()
        self.gpc_wts = (w1[:, np.newaxis] * wts[np.newaxis, :]).flatten()
        self.gpc_wts_sum = np.sum(self.gpc_wts)

    def num_vars_per_component(self) -> int:
        return 6

    def default_bounds(
        self, domain_extents: Tuple[float, ...], num_components: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Lx, Ly, _ = domain_extents
        N = num_components
        lb = np.zeros(N * 6)
        ub = np.zeros(N * 6)
        diag = np.sqrt(Lx**2 + Ly**2)
        
        # [Xc, Yc, L, W, Theta, Mc]
        lb[0::6] = -1.0;          ub[0::6] = Lx + 1.0
        lb[1::6] = -1.0;          ub[1::6] = Ly + 1.0
        lb[2::6] = 0.0;           ub[2::6] = diag
        lb[3::6] = 0.0;           ub[3::6] = diag
        lb[4::6] = -2.0 * np.pi;  ub[4::6] = 2.0 * np.pi
        lb[5::6] = 0.0;           ub[5::6] = 1.0
        return lb, ub

    def forward(
        self,
        x_vars: np.ndarray,
        eval_coords: np.ndarray,
        power_E: float = 1.0,
        power_V: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        num_elements = eval_coords.shape[0]
        X_eval = eval_coords[:, 0:1] + self.gpc_x_rel[np.newaxis, :]
        Y_eval = eval_coords[:, 1:2] + self.gpc_y_rel[np.newaxis, :]
        Z_eval = eval_coords[:, 2:3] + self.gpc_z_rel[np.newaxis, :]
        X_flat = X_eval.flatten()
        Y_flat = Y_eval.flatten()
        Z_flat = Z_eval.flatten()

        params = x_vars.reshape(self._num_components, 6)
        h_fixed = self.layer_height
        
        char_funcs_E = []
        char_funcs_V = []

        for layer in range(self.num_layers):
            z_fixed = (layer + 0.5) * self.layer_height
            for i in range(self.comp_per_layer):
                idx = layer * self.comp_per_layer + i
                p = params[idx]
                
                # compute_local_char_3d uses X_mesh, Y_mesh, Xc, Yc, L, h, theta
                # with Z_mesh=Z_flat, Z=z_fixed, W_width=p[3]
                W_gp = compute_local_char_3d(
                    X_flat, Y_flat, p[0], p[1], p[2], h_fixed, p[4], self.r_gp, 
                    method=self.method, Z_mesh=Z_flat, Z=z_fixed, W_width=p[3]
                )
                W_el = np.sum(W_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                
                char_funcs_E.append(W_el * (p[5]**power_E))
                char_funcs_V.append(W_el * (p[5]**power_V))
                
        return np.array(char_funcs_E), np.array(char_funcs_V)

    def jacobian(
        self,
        x_vars: np.ndarray,
        eval_coords: np.ndarray,
        power_E: float = 1.0,
        power_V: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        
        num_elements = eval_coords.shape[0]
        X_eval = eval_coords[:, 0:1] + self.gpc_x_rel[np.newaxis, :]
        Y_eval = eval_coords[:, 1:2] + self.gpc_y_rel[np.newaxis, :]
        Z_eval = eval_coords[:, 2:3] + self.gpc_z_rel[np.newaxis, :]
        X_flat = X_eval.flatten()
        Y_flat = Y_eval.flatten()
        Z_flat = Z_eval.flatten()

        params = x_vars.reshape(self._num_components, 6)
        h_fixed = self.layer_height
        
        char_funcs_E = []
        char_funcs_V = []
        grads_E = []
        grads_V = []

        for layer in range(self.num_layers):
            z_fixed = (layer + 0.5) * self.layer_height
            for i in range(self.comp_per_layer):
                idx = layer * self.comp_per_layer + i
                p = params[idx]
                
                res = compute_local_characteristic_3d_alm_with_grad_np(
                    X_flat, Y_flat, Z_flat, p[0], p[1], z_fixed, p[2], p[3], h_fixed, p[4], 
                    self.r_gp, method=self.method
                )
                W_gp, dWdX_gp, dWdY_gp, dWdZ_gp, dWdL_gp, dWdW_width_gp, dWdh_gp, dWdT_gp = res
                
                W_el = np.sum(W_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                dWdX_el = np.sum(dWdX_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                dWdY_el = np.sum(dWdY_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                dWdL_el = np.sum(dWdL_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                dWdW_width_el = np.sum(dWdW_width_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                dWdT_el = np.sum(dWdT_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                
                m_pE = p[5]**power_E
                m_pE_m1 = power_E * (p[5]**(power_E - 1.0)) if power_E > 0 else 0.0
                m_pV = p[5]**power_V
                m_pV_m1 = power_V * (p[5]**(power_V - 1.0)) if power_V > 0 else 0.0
                
                char_funcs_E.append(W_el * m_pE)
                char_funcs_V.append(W_el * m_pV)
                
                grads_E.append([
                    dWdX_el * m_pE, dWdY_el * m_pE, dWdL_el * m_pE, dWdW_width_el * m_pE, dWdT_el * m_pE, W_el * m_pE_m1
                ])
                grads_V.append([
                    dWdX_el * m_pV, dWdY_el * m_pV, dWdL_el * m_pV, dWdW_width_el * m_pV, dWdT_el * m_pV, W_el * m_pV_m1
                ])
                
        return {
            "funcs_E": np.array(char_funcs_E),
            "funcs_V": np.array(char_funcs_V),
            "grads_E": np.array(grads_E), # (num_comp, 6, num_el)
            "grads_V": np.array(grads_V),
        }
