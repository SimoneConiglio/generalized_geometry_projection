# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Free 2-D geometry projection mapper.

Maps 6 variables per component: [Xc, Yc, L, h, Theta, Mc].
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ggp.utils.vectorized_mapping import (
    compute_local_characteristic_np,
    compute_local_characteristic_2d_with_grad_np,
)
from .base import ProjectionMapper
from .registry import register_mapper


@register_mapper("Free")
@register_mapper("2D_Free")
class Free2DMapper(ProjectionMapper):
    """Free 2-D projection mapper.
    
    Each component uses 6 parameters:
      [Xc, Yc, length, thickness, angle, density]
    """

    def __init__(
        self,
        num_components: int,
        r_gp: float = 0.5,
        method: str = "GP",
        Ngp: int = 2,
        **kwargs,
    ):
        self._num_components = num_components
        self.r_gp = r_gp
        self.method = method
        self.Ngp = Ngp

        # Pre-compute local sampling window integration weights
        pts, wts = np.polynomial.legendre.leggauss(self.Ngp)
        pts = pts * self.r_gp
        wts = wts * self.r_gp
        gpcx, gpcy = np.meshgrid(pts, pts)
        self.gpc_x_rel = gpcx.flatten()
        self.gpc_y_rel = gpcy.flatten()
        self.gpc_wts = (wts[:, np.newaxis] * wts[np.newaxis, :]).flatten()
        self.gpc_wts_sum = np.sum(self.gpc_wts)
        self.num_gp_per_el = len(self.gpc_wts)

    def num_vars_per_component(self) -> int:
        return 6

    def default_bounds(
        self, domain_extents: Tuple[float, ...], num_components: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Lx, Ly = domain_extents
        N = num_components
        lb = np.zeros(N * 6)
        ub = np.zeros(N * 6)
        diag = np.sqrt(Lx**2 + Ly**2)
        
        # [Xc, Yc, L, h, Theta, Mc]
        lb[0::6] = -1.0;          ub[0::6] = Lx + 1.0
        lb[1::6] = -1.0;          ub[1::6] = Ly + 1.0
        lb[2::6] = 0.0;           ub[2::6] = diag
        lb[3::6] = 3.0;           ub[3::6] = diag      # minh=3 elements (matches reference)
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
        X_eval_flat = X_eval.flatten()
        Y_eval_flat = Y_eval.flatten()

        params = x_vars.reshape(self._num_components, 6)
        char_funcs_E = []
        char_funcs_V = []

        for i in range(self._num_components):
            p = params[i]
            W_gp = compute_local_characteristic_np(
                X_eval_flat, Y_eval_flat, p[0], p[1], p[2], p[3], p[4], 
                self.r_gp, method=self.method
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
        X_eval_flat = X_eval.flatten()
        Y_eval_flat = Y_eval.flatten()

        params = x_vars.reshape(self._num_components, 6)
        
        char_funcs_E = []
        char_funcs_V = []
        grads_E = []
        grads_V = []

        for i in range(self._num_components):
            p = params[i]
            res = compute_local_characteristic_2d_with_grad_np(
                X_eval_flat, Y_eval_flat, p[0], p[1], p[2], p[3], p[4], 
                self.r_gp, method=self.method
            )
            W_gp, dWdX_gp, dWdY_gp, dWdL_gp, dWdh_gp, dWdT_gp = res
            
            # Integrate over sampling window
            W_el = np.sum(W_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
            dWdX_el = np.sum(dWdX_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
            dWdY_el = np.sum(dWdY_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
            dWdL_el = np.sum(dWdL_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
            dWdh_el = np.sum(dWdh_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
            dWdT_el = np.sum(dWdT_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
            
            m_pE = p[5]**power_E
            m_pE_m1 = power_E * (p[5]**(power_E - 1.0)) if power_E > 0 else 0.0
            
            m_pV = p[5]**power_V
            m_pV_m1 = power_V * (p[5]**(power_V - 1.0)) if power_V > 0 else 0.0
            
            char_funcs_E.append(W_el * m_pE)
            char_funcs_V.append(W_el * m_pV)
            
            grads_E.append([
                dWdX_el * m_pE, dWdY_el * m_pE, dWdL_el * m_pE, 
                dWdh_el * m_pE, dWdT_el * m_pE, W_el * m_pE_m1
            ])
            grads_V.append([
                dWdX_el * m_pV, dWdY_el * m_pV, dWdL_el * m_pV, 
                dWdh_el * m_pV, dWdT_el * m_pV, W_el * m_pV_m1
            ])
            
        return {
            "funcs_E": np.array(char_funcs_E),
            "funcs_V": np.array(char_funcs_V),
            "grads_E": np.array(grads_E), # shape: (num_comp, 6, num_el)
            "grads_V": np.array(grads_V),
        }
