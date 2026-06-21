# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Additive Layer Manufacturing (ALM) 2-D geometry projection mapper.

Maps components layer-by-layer.
Standard mode: 3 variables per component: [Xc, width, Mc].
Extended mode: 4 variables per component + [y0, theta0].
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ggp.utils.vectorized_mapping import (
    compute_local_characteristic_np,
    compute_local_characteristic_2d_with_grad_np,
    compute_continuous_ALM_characteristic_np,
)
from .base import ProjectionMapper
from .registry import register_mapper


@register_mapper("ALM")
@register_mapper("2D_ALM")
class ALM2DMapper(ProjectionMapper):
    """ALM 2-D projection mapper.
    
    Components are arranged in layers.
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
        gpcx, gpcy = np.meshgrid(pts, pts)
        self.gpc_x_rel = gpcx.flatten()
        self.gpc_y_rel = gpcy.flatten()
        self.gpc_wts = (wts[:, np.newaxis] * wts[np.newaxis, :]).flatten()
        self.gpc_wts_sum = np.sum(self.gpc_wts)

    def num_vars_per_component(self) -> int:
        return 3

    def default_bounds(
        self, domain_extents: Tuple[float, ...], num_components: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Lx, Ly = domain_extents
        N = num_components
        lb = np.zeros(N * 3)
        ub = np.zeros(N * 3)
        diag = np.sqrt(Lx**2 + Ly**2)
        
        # [Xc, width, Mc]
        lb[0::3] = -1.0;  ub[0::3] = Lx + 1.0
        lb[1::3] = 0.0;   ub[1::3] = diag
        lb[2::3] = 0.0;   ub[2::3] = 1.0
        return lb, ub

    def _is_extended(self, x_vars: np.ndarray) -> bool:
        return len(x_vars) == (self._num_components * 4 + 2)

    def _is_continuous(self, x_vars: np.ndarray) -> bool:
        return len(x_vars) == (2 * self.comp_per_layer * self.num_layers + self.num_layers + self.comp_per_layer)

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

        char_funcs_E = []
        char_funcs_V = []

        if self._is_continuous(x_vars):
            np_val = self.comp_per_layer
            nY = self.num_layers
            h = x_vars[2*np_val*nY : 2*np_val*nY + nY]
            y_nodes = np.zeros(nY + 1)
            y_nodes[1:] = np.cumsum(h)
            Yk = np.tile(y_nodes[:-1], (np_val, 1)).T
            p_dict = {'method': self.method, 'r': self.r_gp, 'deltamin': 1e-6, 'q': 10.0, 'E0': 1.0, 'Emin': 1e-6, 'gammac': 3.0, 'penalty': 3.0, 'saturation': False}
            
            W, _, _, _ = compute_continuous_ALM_characteristic_np(x_vars, Y_eval_flat, x_vars, p_dict, np_val, nY, Yk, num_elements)
            m = x_vars[2*np_val*nY + nY : ]
            for i in range(np_val):
                W_el = np.sum(W[i, :].reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                char_funcs_E.append(W_el * (m[i]**power_E))
                char_funcs_V.append(W_el * (m[i]**power_V))
        else:
            is_ext = self._is_extended(x_vars)
            if is_ext:
                y0 = x_vars[-2]
                theta0 = x_vars[-1]
                params = x_vars[:-2].reshape(self._num_components, 4)
            else:
                y0 = 0.0
                theta0 = 0.0
                params = x_vars.reshape(self._num_components, 3)

            for layer in range(self.num_layers):
                y_fixed = (layer + 0.5) * self.layer_height
                for i in range(self.comp_per_layer):
                    idx = layer * self.comp_per_layer + i
                    p = params[idx]
                    Xc, width, Mc = p[0], p[1], p[2]
                    hc = p[3] if is_ext else self.layer_height
                    
                    Xc_g = Xc * np.cos(theta0) - y_fixed * np.sin(theta0)
                    Yc_g = y0 + Xc * np.sin(theta0) + y_fixed * np.cos(theta0)
                    
                    W_gp = compute_local_characteristic_np(
                        X_eval_flat, Y_eval_flat, Xc_g, Yc_g, width, hc, theta0, self.r_gp, method='GP'
                    )
                    W_el = np.sum(W_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                    
                    char_funcs_E.append(W_el * (Mc**power_E))
                    char_funcs_V.append(W_el * (Mc**power_V))
                    
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

        char_funcs_E = []
        char_funcs_V = []
        grads_E = []
        grads_V = []

        if self._is_continuous(x_vars):
            np_val = self.comp_per_layer
            nY = self.num_layers
            h = x_vars[2*np_val*nY : 2*np_val*nY + nY]
            y_nodes = np.zeros(nY + 1)
            y_nodes[1:] = np.cumsum(h)
            Yk = np.tile(y_nodes[:-1], (np_val, 1)).T
            p_dict = {'method': self.method, 'r': self.r_gp, 'deltamin': 1e-6, 'q': 10.0, 'E0': 1.0, 'Emin': 1e-6, 'gammac': 3.0, 'penalty': 3.0, 'saturation': False}
            
            W_gp, dWdX_gp, dWdL_gp, dWdh_gp = compute_continuous_ALM_characteristic_np(x_vars, Y_eval_flat, x_vars, p_dict, np_val, nY, Yk, num_elements)
            m = x_vars[2*np_val*nY + nY : ]
            
            for i in range(np_val):
                W_el = np.sum(W_gp[i, :].reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                
                dWdX_el = np.sum(dWdX_gp.reshape(dWdX_gp.shape[0], num_elements, -1) * self.gpc_wts, axis=2) / self.gpc_wts_sum
                dWdL_el = np.sum(dWdL_gp.reshape(dWdL_gp.shape[0], num_elements, -1) * self.gpc_wts, axis=2) / self.gpc_wts_sum
                dWdh_el = np.sum(dWdh_gp.reshape(dWdh_gp.shape[0], num_elements, -1) * self.gpc_wts, axis=2) / self.gpc_wts_sum
                
                m_pE = m[i]**power_E
                m_pE_m1 = power_E * (m[i]**(power_E - 1.0)) if power_E > 0 else 0.0
                m_pV = m[i]**power_V
                m_pV_m1 = power_V * (m[i]**(power_V - 1.0)) if power_V > 0 else 0.0
                
                char_funcs_E.append(W_el * m_pE)
                char_funcs_V.append(W_el * m_pV)
                
                grads_E.append({
                    'dX': dWdX_el * m_pE, 'dL': dWdL_el * m_pE, 'dh': dWdh_el * m_pE, 'dM': W_el * m_pE_m1
                })
                grads_V.append({
                    'dX': dWdX_el * m_pV, 'dL': dWdL_el * m_pV, 'dh': dWdh_el * m_pV, 'dM': W_el * m_pV_m1
                })
            return {
                "funcs_E": np.array(char_funcs_E), "funcs_V": np.array(char_funcs_V),
                "grads_E": grads_E, "grads_V": grads_V, "is_continuous": True
            }
        else:
            is_ext = self._is_extended(x_vars)
            if is_ext:
                y0 = x_vars[-2]
                theta0 = x_vars[-1]
                params = x_vars[:-2].reshape(self._num_components, 4)
            else:
                y0 = 0.0
                theta0 = 0.0
                params = x_vars.reshape(self._num_components, 3)

            for layer in range(self.num_layers):
                y_fixed = (layer + 0.5) * self.layer_height
                for i in range(self.comp_per_layer):
                    idx = layer * self.comp_per_layer + i
                    p = params[idx]
                    Xc, width, Mc = p[0], p[1], p[2]
                    hc = p[3] if is_ext else self.layer_height
                    
                    Xc_g = Xc * np.cos(theta0) - y_fixed * np.sin(theta0)
                    Yc_g = y0 + Xc * np.sin(theta0) + y_fixed * np.cos(theta0)
                    
                    res = compute_local_characteristic_2d_with_grad_np(
                        X_eval_flat, Y_eval_flat, Xc_g, Yc_g, width, hc, theta0, self.r_gp, method=self.method
                    )
                    W_gp, dWdX_gp, dWdY_gp, dWdL_gp, dWdh_gp, dWdT_gp = res
                    
                    an_Xc_gp = dWdX_gp * np.cos(theta0) + dWdY_gp * np.sin(theta0)
                    
                    W_el = np.sum(W_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                    dWdX_el = np.sum(an_Xc_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                    dWdL_el = np.sum(dWdL_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                    
                    m_pE = Mc**power_E
                    m_pE_m1 = power_E * (Mc**(power_E - 1.0)) if power_E > 0 else 0.0
                    m_pV = Mc**power_V
                    m_pV_m1 = power_V * (Mc**(power_V - 1.0)) if power_V > 0 else 0.0
                    
                    char_funcs_E.append(W_el * m_pE)
                    char_funcs_V.append(W_el * m_pV)
                    
                    if is_ext:
                        dWdh_el = np.sum(dWdh_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                        grads_E.append([dWdX_el * m_pE, dWdL_el * m_pE, W_el * m_pE_m1, dWdh_el * m_pE])
                        grads_V.append([dWdX_el * m_pV, dWdL_el * m_pV, W_el * m_pV_m1, dWdh_el * m_pV])
                    else:
                        grads_E.append([dWdX_el * m_pE, dWdL_el * m_pE, W_el * m_pE_m1])
                        grads_V.append([dWdX_el * m_pV, dWdL_el * m_pV, W_el * m_pV_m1])
                        
            return {
                "funcs_E": np.array(char_funcs_E), "funcs_V": np.array(char_funcs_V),
                "grads_E": np.array(grads_E), "grads_V": np.array(grads_V), "is_continuous": False
            }
