# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Additive Layer Manufacturing (ALM) 2-D geometry projection mapper.

Correct variable layout (interleaved, matching GGP_ALM_Matlab reference):
  [Xc_0_0, L_0_0, Xc_0_1, L_0_1, ..., Xc_{nY-1}_{np-1}, L_{nY-1}_{np-1},
   h_0, ..., h_{np-1},
   Mc_0, ..., Mc_{np-1},
   y0, theta0]

  * 2·nY·np vars:  [Xc, L] interleaved, per layer per component (F-order reshape → Xk[k,j])
  * np vars:       h[j] — normalised total height per printed component (0→1, cutoff = Ly·h)
  * np vars:       Mc[j] — membership density per printed component
  * 2 vars:        y0, theta0 — global printing-plane y-offset and rotation angle
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ggp.utils.vectorized_mapping import compute_continuous_ALM_characteristic_np
from .base import ProjectionMapper
from .registry import register_mapper


@register_mapper("ALM")
@register_mapper("2D_ALM")
class ALM2DMapper(ProjectionMapper):
    """ALM 2-D projection mapper using the MNA continuous formulation.

    All design variables follow the interleaved layout described in the module
    docstring.  The mapper always operates in continuous (MNA) mode — the old
    per-component 3-var mode is removed.
    """

    def __init__(
        self,
        num_components: int,
        num_layers: int,
        comp_per_layer: int,
        layer_height: float,
        r_gp: float = 0.5,
        method: str = "MNA",
        Ngp: int = 2,
        **kwargs,
    ):
        self._num_components = num_components
        self.num_layers = num_layers          # nY
        self.comp_per_layer = comp_per_layer  # np_val
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

        # Fixed y-centres of each component in each layer (not design variables)
        nY = self.num_layers
        np_val = self.comp_per_layer
        self._Yk = np.tile(
            np.arange(nY) * layer_height + 0.5 * layer_height,
            (np_val, 1)
        ).T  # shape (nY, np_val), Yk[k,j] = (k+0.5)*layer_height

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def num_vars(self) -> int:
        """Total number of design variables."""
        return 2 * self.num_layers * self.comp_per_layer + 2 * self.comp_per_layer + 2

    def num_vars_per_component(self) -> int:
        """Legacy; returns total vars (not per-component — layout is not uniform)."""
        return self.num_vars()

    def default_bounds(
        self, domain_extents: Tuple[float, ...], num_components: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Lx, Ly = domain_extents[0], domain_extents[1]
        nY = self.num_layers
        np_val = self.comp_per_layer
        n_xl  = 2 * nY * np_val   # [Xc, L] block
        n_hmc = 2 * np_val        # [h, Mc] block
        n_tot = n_xl + n_hmc + 2  # + [y0, theta0]

        lb = np.zeros(n_tot)
        ub = np.zeros(n_tot)

        # [Xc, L] interleaved
        lb[0:n_xl:2] = -1.0;          ub[0:n_xl:2] = Lx + 1.0   # Xc
        lb[1:n_xl:2] = 0.0;           ub[1:n_xl:2] = Lx          # L (width)

        # h: normalised height [0.2, 1.0]
        lb[n_xl       : n_xl + np_val] = 0.2
        ub[n_xl       : n_xl + np_val] = 1.0

        # Mc: membership [0, 1]
        lb[n_xl + np_val : n_xl + 2*np_val] = 0.0
        ub[n_xl + np_val : n_xl + 2*np_val] = 1.0

        # y0: printing-plane y-offset [-Ly/2, Ly/2]
        lb[n_xl + 2*np_val] = -Ly / 2.0
        ub[n_xl + 2*np_val] =  Ly / 2.0

        # theta0: rotation angle [-pi/4, pi/4]
        lb[n_xl + 2*np_val + 1] = -np.pi / 4.0
        ub[n_xl + 2*np_val + 1] =  np.pi / 4.0

        return lb, ub

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_vars(self, x_vars: np.ndarray):
        """Split x_vars into (Xk, Lk, h, Mc, y0, theta0)."""
        nY, np_val = self.num_layers, self.comp_per_layer
        n_xl = 2 * nY * np_val

        Xk = x_vars[0:n_xl:2].reshape((nY, np_val), order='F')
        Lk = x_vars[1:n_xl:2].reshape((nY, np_val), order='F')
        h  = x_vars[n_xl         : n_xl + np_val]
        Mc = x_vars[n_xl + np_val : n_xl + 2*np_val]

        if len(x_vars) >= n_xl + 2*np_val + 2:
            y0     = x_vars[n_xl + 2*np_val]
            theta0 = x_vars[n_xl + 2*np_val + 1]
        else:
            y0 = 0.0
            theta0 = 0.0

        return Xk, Lk, h, Mc, y0, theta0

    def _apply_plane_transform(
        self,
        X_eval: np.ndarray,
        Y_eval: np.ndarray,
        y0: float,
        theta0: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate + translate evaluation coordinates into print-plane frame."""
        c, s = np.cos(theta0), np.sin(theta0)
        X_p = X_eval * c - Y_eval * s
        Y_p = y0 + X_eval * s + Y_eval * c
        return X_p, Y_p

    # ------------------------------------------------------------------
    # ProjectionMapper API
    # ------------------------------------------------------------------

    def forward(
        self,
        x_vars: np.ndarray,
        eval_coords: np.ndarray,
        power_E: float = 1.0,
        power_V: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        nY, np_val = self.num_layers, self.comp_per_layer
        _, _, h, Mc, y0, theta0 = self._parse_vars(x_vars)
        Ly = nY * self.layer_height

        num_elements = eval_coords.shape[0]
        X_eval = eval_coords[:, 0:1] + self.gpc_x_rel[np.newaxis, :]
        Y_eval = eval_coords[:, 1:2] + self.gpc_y_rel[np.newaxis, :]
        X_p, Y_p = self._apply_plane_transform(
            X_eval.flatten(), Y_eval.flatten(), y0, theta0
        )

        p_dict = {'method': 'MNA'}
        W, _, _, _ = compute_continuous_ALM_characteristic_np(
            X_p, Y_p, x_vars, p_dict, np_val, nY, self._Yk, Ly
        )

        char_funcs_E = []
        char_funcs_V = []
        for j in range(np_val):
            W_el = (np.sum(W[j, :].reshape(num_elements, -1) * self.gpc_wts, axis=1)
                    / self.gpc_wts_sum)
            char_funcs_E.append(W_el * (Mc[j] ** power_E))
            char_funcs_V.append(W_el * (Mc[j] ** power_V))

        return np.array(char_funcs_E), np.array(char_funcs_V)

    def jacobian(
        self,
        x_vars: np.ndarray,
        eval_coords: np.ndarray,
        power_E: float = 1.0,
        power_V: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        nY, np_val = self.num_layers, self.comp_per_layer
        _, _, h, Mc, y0, theta0 = self._parse_vars(x_vars)
        Ly = nY * self.layer_height
        n_xl  = 2 * nY * np_val
        n_tot = len(x_vars)

        num_elements = eval_coords.shape[0]
        X_eval = eval_coords[:, 0:1] + self.gpc_x_rel[np.newaxis, :]
        Y_eval = eval_coords[:, 1:2] + self.gpc_y_rel[np.newaxis, :]
        X_flat, Y_flat = X_eval.flatten(), Y_eval.flatten()
        X_p, Y_p = self._apply_plane_transform(X_flat, Y_flat, y0, theta0)

        p_dict = {'method': 'MNA'}
        W, dW_dXc, dW_dL, dW_dh = compute_continuous_ALM_characteristic_np(
            X_p, Y_p, x_vars, p_dict, np_val, nY, self._Yk, Ly
        )

        # Gauss-point integration to element values
        def _gp_to_el(arr_gp):
            return (np.sum(arr_gp.reshape(arr_gp.shape[0], num_elements, -1) * self.gpc_wts,
                           axis=2) / self.gpc_wts_sum)

        W_el     = _gp_to_el(W)              # (np_val, n_elements)
        dXc_el   = _gp_to_el(dW_dXc.reshape(np_val * nY, -1)).reshape(np_val, nY, num_elements)
        dL_el    = _gp_to_el(dW_dL.reshape(np_val * nY, -1)).reshape(np_val, nY, num_elements)
        dh_el    = _gp_to_el(dW_dh)          # (np_val, n_elements)

        # grads[j, var_idx, e] = d(char_func_j) / d(global_var_idx) at element e
        # grads_E includes the Mc^power factor; grads_V similarly
        funcs_E = np.zeros((np_val, num_elements))
        funcs_V = np.zeros((np_val, num_elements))
        grads_E = np.zeros((np_val, n_tot, num_elements))
        grads_V = np.zeros((np_val, n_tot, num_elements))

        has_global = (n_tot >= n_xl + 2*np_val + 2)

        for j in range(np_val):
            m_pE   = Mc[j] ** power_E
            m_pV   = Mc[j] ** power_V
            m_pE_m = power_E * (Mc[j] ** (power_E - 1.0)) if power_E > 0 else 0.0
            m_pV_m = power_V * (Mc[j] ** (power_V - 1.0)) if power_V > 0 else 0.0

            funcs_E[j, :] = W_el[j, :] * m_pE
            funcs_V[j, :] = W_el[j, :] * m_pV

            # [Xc, L] block — interleaved, F-order: Xk[k,j] at global index 2*(k*np_val+j)
            for k in range(nY):
                idx_xc = 2 * (k * np_val + j)
                idx_l  = idx_xc + 1
                grads_E[j, idx_xc, :] = dXc_el[j, k, :] * m_pE
                grads_E[j, idx_l,  :] = dL_el[j, k, :]  * m_pE
                grads_V[j, idx_xc, :] = dXc_el[j, k, :] * m_pV
                grads_V[j, idx_l,  :] = dL_el[j, k, :]  * m_pV

            # h block
            idx_h = n_xl + j
            grads_E[j, idx_h, :] = dh_el[j, :] * m_pE
            grads_V[j, idx_h, :] = dh_el[j, :] * m_pV

            # Mc block
            idx_mc = n_xl + np_val + j
            grads_E[j, idx_mc, :] = W_el[j, :] * m_pE_m
            grads_V[j, idx_mc, :] = W_el[j, :] * m_pV_m

            # y0, theta0 — coordinate-transform chain rule
            if has_global:
                # dX_p/dtheta0 = -X*sin - Y*cos, dY_p/dtheta0 = X*cos - Y*sin
                # dX_p/dy0 = 0,                  dY_p/dy0 = 1
                c, s = np.cos(theta0), np.sin(theta0)
                dXp_dt = -X_flat * s - Y_flat * c
                dYp_dt =  X_flat * c - Y_flat * s

                # dW/dy0  = dW/dY_p (approximated via height-cutoff direction)
                # We approximate: dW/dX_p ≈ -dWt1 + dWt2 (from the x-boundary tests)
                # dW/dY_p ≈ dWt3 (height only) — full chain omitted for now (y0 is small)
                # Full implementation: recompute boundary interpolation wrt Y_p

                # For simplicity we use finite differences for y0, theta0
                eps_fd = 1e-5

                def _W_el_at(y0_, theta0_):
                    Xpp, Ypp = self._apply_plane_transform(X_flat, Y_flat, y0_, theta0_)
                    W_, _, _, _ = compute_continuous_ALM_characteristic_np(
                        Xpp, Ypp, x_vars, p_dict, np_val, nY, self._Yk, Ly
                    )
                    return (_gp_to_el(W_))[j, :]

                dW_dy0    = (_W_el_at(y0 + eps_fd, theta0) - W_el[j, :]) / eps_fd
                dW_dtheta = (_W_el_at(y0, theta0 + eps_fd) - W_el[j, :]) / eps_fd

                idx_y0 = n_xl + 2*np_val
                idx_th = idx_y0 + 1
                grads_E[j, idx_y0, :] = dW_dy0    * m_pE
                grads_E[j, idx_th, :] = dW_dtheta * m_pE
                grads_V[j, idx_y0, :] = dW_dy0    * m_pV
                grads_V[j, idx_th, :] = dW_dtheta * m_pV

        return {
            "funcs_E": funcs_E,
            "funcs_V": funcs_V,
            "grads_E": grads_E,   # (np_val, n_tot, n_elements) — global col indexing
            "grads_V": grads_V,
            "is_continuous": True,
        }
