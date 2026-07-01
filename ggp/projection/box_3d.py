# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""3-D box primitive mapper — solid analogue of the 2-D rectangle.

Each component is an **oriented box** with 9 parameters
``[Xc, Yc, Zc, Lx, Ly, Lz, theta, phi, Mc]``: centre, three side lengths, the (θ,φ)
orientation of its long axis, and the mass density. The characteristic is the product of
six quintic-smoothed faces (:func:`ggp.utils.jax_primitives.box_batch`); gradients come from
``jax.grad`` (no hand-derived rotation/product chain). Physics uses linear stiffness like the
2-D rect (``p_penalty=1``, small ``Emin``) since the box density saturates to a clean 0/1.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ggp.utils.jax_primitives import box_batch
from .base import ProjectionMapper
from .registry import register_mapper


@register_mapper("Box3D")
@register_mapper("3D_Box")
class Box3DMapper(ProjectionMapper):
    """Per-component oriented-box mapper (9 vars/component)."""

    def __init__(self, num_components: int, r_gp: float = 0.5, method: str = "box",
                 Ngp: int = 2, min_thickness: float = 1.0, **kwargs):
        self._num_components = num_components
        self.r_gp = r_gp
        self.method = method
        self.Ngp = Ngp
        self.min_thickness = float(min_thickness) if min_thickness else 1.0

        pts, wts = np.polynomial.legendre.leggauss(self.Ngp)
        pts = pts * self.r_gp
        wts = wts * self.r_gp
        gx, gy, gz = np.meshgrid(pts, pts, pts, indexing="ij")
        self.gpc_x_rel = gx.flatten(); self.gpc_y_rel = gy.flatten(); self.gpc_z_rel = gz.flatten()
        w1 = (wts[:, None] * wts[None, :]).flatten()
        self.gpc_wts = (w1[:, None] * wts[None, :]).flatten()
        self.gpc_wts_sum = float(np.sum(self.gpc_wts))
        self.num_gp_per_el = len(self.gpc_wts)

    def num_vars_per_component(self) -> int:
        return 9

    def default_bounds(self, domain_extents, num_components):
        Lx, Ly, Lz = domain_extents
        N = num_components
        diag = float(np.sqrt(Lx ** 2 + Ly ** 2 + Lz ** 2))
        lb = np.zeros(N * 9); ub = np.zeros(N * 9)
        mt = self.min_thickness if self.min_thickness > 1.0 else 0.0
        lb[0::9] = -1.0;         ub[0::9] = Lx + 1.0        # Xc
        lb[1::9] = -1.0;         ub[1::9] = Ly + 1.0        # Yc
        lb[2::9] = -1.0;         ub[2::9] = Lz + 1.0        # Zc
        lb[3::9] = mt;           ub[3::9] = diag            # Lx
        lb[4::9] = self.min_thickness; ub[4::9] = diag      # Ly
        lb[5::9] = self.min_thickness; ub[5::9] = diag      # Lz
        lb[6::9] = -2.0 * np.pi; ub[6::9] = 2.0 * np.pi     # theta
        lb[7::9] = -2.0 * np.pi; ub[7::9] = 2.0 * np.pi     # phi
        lb[8::9] = 0.0;          ub[8::9] = 1.0             # Mc
        return lb, ub

    def _eval_points(self, eval_coords):
        X = eval_coords[:, 0:1] + self.gpc_x_rel[None, :]
        Y = eval_coords[:, 1:2] + self.gpc_y_rel[None, :]
        Z = eval_coords[:, 2:3] + self.gpc_z_rel[None, :]
        return np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])

    def _integrate(self, arr_gp, num_elements):
        return (np.sum(arr_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1)
                / self.gpc_wts_sum)

    def forward(self, x_vars, eval_coords, power_E=1.0, power_V=1.0):
        num_elements = eval_coords.shape[0]
        pts = self._eval_points(eval_coords)
        params = x_vars.reshape(self._num_components, 9)
        char_E, char_V = [], []
        for p in params:
            W, _, _, _, _ = box_batch(pts, p[0:3], p[3:6], p[6], p[7], self.r_gp)
            W_el = self._integrate(W, num_elements)
            char_E.append(W_el * (p[8] ** power_E))
            char_V.append(W_el * (p[8] ** power_V))
        return np.array(char_E), np.array(char_V)

    def jacobian(self, x_vars, eval_coords, power_E=1.0, power_V=1.0):
        num_elements = eval_coords.shape[0]
        pts = self._eval_points(eval_coords)
        params = x_vars.reshape(self._num_components, 9)
        char_E, char_V, grads_E, grads_V = [], [], [], []
        for p in params:
            W, dC, dS, dT, dP = box_batch(pts, p[0:3], p[3:6], p[6], p[7], self.r_gp)
            W_el = self._integrate(W, num_elements)
            dC_el = [self._integrate(dC[:, k], num_elements) for k in range(3)]
            dS_el = [self._integrate(dS[:, k], num_elements) for k in range(3)]
            dT_el = self._integrate(dT, num_elements)
            dP_el = self._integrate(dP, num_elements)

            mE = p[8] ** power_E; mV = p[8] ** power_V
            mE_m = power_E * (p[8] ** (power_E - 1.0)) if power_E > 0 else 0.0
            mV_m = power_V * (p[8] ** (power_V - 1.0)) if power_V > 0 else 0.0
            char_E.append(W_el * mE); char_V.append(W_el * mV)
            geo = [dC_el[0], dC_el[1], dC_el[2], dS_el[0], dS_el[1], dS_el[2], dT_el, dP_el]
            grads_E.append([g * mE for g in geo] + [W_el * mE_m])
            grads_V.append([g * mV for g in geo] + [W_el * mV_m])

        return {"funcs_E": np.array(char_E), "funcs_V": np.array(char_V),
                "grads_E": np.array(grads_E), "grads_V": np.array(grads_V)}
