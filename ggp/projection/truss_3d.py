# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Node-based 3-D truss projection mapper (lift of :mod:`ggp.projection.truss_2d`).

A 3-D truss is a set of **nodes** (positions are design variables) and a fixed connectivity
of bars joining node pairs; bars sharing a node share its coordinates, so the structure
stays connected by construction. Each bar also carries a thickness ``h`` and mass density
``Mc`` (the topology knobs). The per-bar capsule characteristic and its endpoint gradients
come from :func:`ggp.utils.jax_primitives.capsule_batch` (segment-distance based, so no
azimuth singularity for vertical bars), scattered into the **shared node columns** via the
``is_continuous`` global-column Jacobian convention.

Design vector (length ``3N + 2B`` for ``N`` nodes, ``B`` bars)::

    [ nx_0,ny_0,nz_0, ..., nx_{N-1},ny_{N-1},nz_{N-1},   # node coordinates
      h_0, ..., h_{B-1},   Mc_0, ..., Mc_{B-1} ]
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ggp.utils.jax_primitives import capsule_batch
from .base import ProjectionMapper
from .registry import register_mapper


def _split_passthrough_3d(nodes, bars, tol=1e-6):
    """Split any bar that passes through an intermediate (collinear) node.

    3-D bars generically do not cross, so there is no crossing-node insertion (unlike the
    2-D planarization); only collinear pass-throughs — common on a lattice (axis- and
    face-diagonal bars through an interior node) — are split at the existing node.
    """
    P = np.asarray(nodes, dtype=float)
    new_bars = set()
    for (a, c) in bars:
        pa, pc = P[a], P[c]
        d = pc - pa
        L2 = float(d @ d)
        on = []
        for k in range(len(P)):
            t = float((P[k] - pa) @ d) / L2
            if -tol <= t <= 1.0 + tol and np.linalg.norm(P[k] - (pa + t * d)) < tol:
                on.append((t, k))
        on.sort()
        for (_, k1), (_, k2) in zip(on, on[1:]):
            if k1 != k2:
                new_bars.add((min(k1, k2), max(k1, k2)))
    return np.asarray(sorted(new_bars), dtype=int).reshape(-1, 2)


def build_ground_structure_3d(grid_nx, grid_ny, grid_nz, Lx, Ly, Lz,
                              radius=None, split_passthrough=True):
    """Build a 3-D node lattice and a radius-based bar connectivity.

    Nodes sit on a regular ``grid_nx x grid_ny x grid_nz`` lattice (row-major, x fastest).
    A bar joins every node pair within ``radius`` (default ≈ 1 cell-diagonal, i.e. the
    26-neighbour stencil). Returns ``(nodes (N,3), bars (B,2) int)``.
    """
    xs = np.linspace(0.0, Lx, grid_nx)
    ys = np.linspace(0.0, Ly, grid_ny)
    zs = np.linspace(0.0, Lz, grid_nz)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    # row-major with x fastest: order (z, y, x) so ravel gives x fastest
    nodes = np.column_stack([XX.ravel(order="F"), YY.ravel(order="F"), ZZ.ravel(order="F")])

    dx = Lx / max(grid_nx - 1, 1)
    dy = Ly / max(grid_ny - 1, 1)
    dz = Lz / max(grid_nz - 1, 1)
    if radius is None:
        radius = 1.01 * float(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))

    diff = nodes[:, None, :] - nodes[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    iu, ju = np.triu_indices(len(nodes), k=1)
    mask = dist[iu, ju] <= radius + 1e-9
    bars = np.column_stack([iu[mask], ju[mask]]).astype(int)
    if split_passthrough and len(bars):
        bars = _split_passthrough_3d(nodes, bars)
    return nodes, bars


@register_mapper("Truss3D")
@register_mapper("3D_Truss")
class Truss3DMapper(ProjectionMapper):
    """Node-based 3-D truss mapper (shared-node design variables + per-bar h, Mc)."""

    def __init__(self, num_components: int = 0, nodes=None, bars=None, r_gp: float = 0.5,
                 method: str = "GP", Ngp: int = 2, min_thickness: float = 1.0, **kwargs):
        if nodes is None or bars is None:
            raise ValueError("Truss3DMapper requires 'nodes' (N,3) and 'bars' (B,2); "
                             "build them with build_ground_structure_3d().")
        self._nodes0 = np.asarray(nodes, dtype=float).reshape(-1, 3)
        self._bars = np.asarray(bars, dtype=int).reshape(-1, 2)
        self.num_nodes = self._nodes0.shape[0]
        self.num_bars = self._bars.shape[0]
        self._num_components = self.num_bars
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

    def num_design_vars(self) -> int:
        return 3 * self.num_nodes + 2 * self.num_bars

    def num_vars_per_component(self) -> int:
        return 2       # not meaningful (shared-node); use num_design_vars()

    def _parse(self, x):
        N, B = self.num_nodes, self.num_bars
        return x[: 3 * N].reshape(N, 3), x[3 * N: 3 * N + B], x[3 * N + B: 3 * N + 2 * B]

    def default_bounds(self, domain_extents, num_components):
        Lx, Ly, Lz = domain_extents[0], domain_extents[1], domain_extents[2]
        diag = float(np.sqrt(Lx ** 2 + Ly ** 2 + Lz ** 2))
        N, B = self.num_nodes, self.num_bars
        lb = np.zeros(3 * N + 2 * B); ub = np.zeros(3 * N + 2 * B)
        lb[0:3 * N:3] = -1.0; ub[0:3 * N:3] = Lx + 1.0
        lb[1:3 * N:3] = -1.0; ub[1:3 * N:3] = Ly + 1.0
        lb[2:3 * N:3] = -1.0; ub[2:3 * N:3] = Lz + 1.0
        lb[3 * N:3 * N + B] = self.min_thickness; ub[3 * N:3 * N + B] = diag
        lb[3 * N + B:] = 0.0; ub[3 * N + B:] = 1.0
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
        node_xyz, h, Mc = self._parse(x_vars)
        char_E, char_V = [], []
        for b in range(self.num_bars):
            a, c = self._bars[b]
            W, _, _, _ = capsule_batch(pts, node_xyz[a], node_xyz[c], h[b], self.r_gp)
            W_el = self._integrate(W, num_elements)
            char_E.append(W_el * (Mc[b] ** power_E))
            char_V.append(W_el * (Mc[b] ** power_V))
        return np.array(char_E), np.array(char_V)

    def jacobian(self, x_vars, eval_coords, power_E=1.0, power_V=1.0):
        num_elements = eval_coords.shape[0]
        pts = self._eval_points(eval_coords)
        node_xyz, h, Mc = self._parse(x_vars)
        N, B = self.num_nodes, self.num_bars
        n_tot = 3 * N + 2 * B
        funcs_E = np.zeros((B, num_elements)); funcs_V = np.zeros((B, num_elements))
        grads_E = np.zeros((B, n_tot, num_elements)); grads_V = np.zeros((B, n_tot, num_elements))

        for b in range(B):
            a, c = self._bars[b]
            W, dP1, dP2, dh = capsule_batch(pts, node_xyz[a], node_xyz[c], h[b], self.r_gp)
            W_el = self._integrate(W, num_elements)
            dP1_el = np.stack([self._integrate(dP1[:, k], num_elements) for k in range(3)], axis=0)
            dP2_el = np.stack([self._integrate(dP2[:, k], num_elements) for k in range(3)], axis=0)
            dh_el = self._integrate(dh, num_elements)

            m_pE = Mc[b] ** power_E; m_pV = Mc[b] ** power_V
            m_pE_m = power_E * (Mc[b] ** (power_E - 1.0)) if power_E > 0 else 0.0
            m_pV_m = power_V * (Mc[b] ** (power_V - 1.0)) if power_V > 0 else 0.0
            funcs_E[b, :] = W_el * m_pE; funcs_V[b, :] = W_el * m_pV

            for k in range(3):                          # node a <- endpoint 1
                grads_E[b, 3 * a + k, :] = dP1_el[k] * m_pE
                grads_V[b, 3 * a + k, :] = dP1_el[k] * m_pV
                grads_E[b, 3 * c + k, :] = dP2_el[k] * m_pE   # node c <- endpoint 2
                grads_V[b, 3 * c + k, :] = dP2_el[k] * m_pV
            grads_E[b, 3 * N + b, :] = dh_el * m_pE          # thickness
            grads_V[b, 3 * N + b, :] = dh_el * m_pV
            grads_E[b, 3 * N + B + b, :] = W_el * m_pE_m     # mass density
            grads_V[b, 3 * N + B + b, :] = W_el * m_pV_m

        return {"funcs_E": funcs_E, "funcs_V": funcs_V, "grads_E": grads_E,
                "grads_V": grads_V, "is_continuous": True}
