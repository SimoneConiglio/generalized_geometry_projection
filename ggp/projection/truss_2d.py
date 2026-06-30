# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Node-based 2-D truss projection mapper.

Instead of parametrising each bar independently by its centre, length and angle
(``[Xc, Yc, L, h, Theta, Mc]`` as in :class:`Free2DMapper`), a *truss* is a set
of **nodes** whose positions are the design variables and a fixed **connectivity**
of bars joining node *pairs*.  Bars that meet at a node share that node's
coordinates ("coincident extremities"), so moving one node moves every bar
attached to it — the truss stays connected by construction and the effective
design space is much smaller than a free bar layout.

To preserve the topology-optimization character every bar additionally carries
its own **thickness** ``h`` and **mass density** ``Mc``; a bar vanishes when
either shrinks to its lower bound, so the optimizer can remove members from the
ground structure while it morphs the surviving ones.

Design vector layout (length ``2*N + 2*B`` for ``N`` nodes, ``B`` bars)::

    [ nx_0, ny_0, nx_1, ny_1, ..., nx_{N-1}, ny_{N-1},   # node coordinates
      h_0, h_1, ..., h_{B-1},                            # bar thicknesses
      Mc_0, Mc_1, ..., Mc_{B-1} ]                        # bar mass densities

Each bar's characteristic is computed with the *same* underlying
``compute_local_characteristic_*`` kernels used by the free mapper, after
converting its two endpoints ``(X1,Y1)``, ``(X2,Y2)`` to the centre/length/angle
the kernel expects; the endpoint Jacobian is obtained by chaining through that
conversion and is accumulated into the *shared* node columns, exactly the
``is_continuous`` (global-column) Jacobian convention the geometry discipline
already supports for the continuous-ALM mapper.
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


def _segment_crossing(p1, p2, p3, p4, tol=1e-9):
    """Return the proper interior crossing of segments p1p2 and p3p4, else None.

    'Proper interior' means the segments are not parallel and intersect strictly
    inside *both* (not at a shared endpoint). Endpoint/collinear touches return None
    (those are handled by the on-segment node split, not by adding a new node).
    """
    r = p2 - p1
    s = p4 - p3
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < tol:                       # parallel or collinear
        return None
    qp = p3 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
    if tol < t < 1.0 - tol and tol < u < 1.0 - tol:
        return p1 + t * r
    return None


def _planarize(nodes: np.ndarray, bars: np.ndarray, tol: float = 1e-6):
    """Make the ground structure a *planar* truss.

    Wherever two bars cross, a node is inserted at the crossing and both bars are
    split there; wherever a bar passes through an existing node it is split at that
    node too. The result is a graph in which every bar intersection is a genuine
    joint (so a moving node drags all members meeting there, with no decoupled
    overlapping material). Returns ``(nodes, bars)`` with the added nodes appended.
    """
    pts = [np.asarray(n, float) for n in nodes]

    def get_node(p):
        for i, q in enumerate(pts):
            if np.hypot(*(q - p)) < tol:
                return i
        pts.append(np.asarray(p, float))
        return len(pts) - 1

    # 1. Insert a node at every proper bar-bar crossing.
    for i in range(len(bars)):
        a, c = bars[i]
        for j in range(i + 1, len(bars)):
            e, f = bars[j]
            if len({int(a), int(c), int(e), int(f)}) < 4:
                continue                      # share an endpoint -> not a crossing
            x = _segment_crossing(pts[a], pts[c], pts[e], pts[f])
            if x is not None:
                get_node(x)

    # 2. Re-split every original bar at all nodes lying on it (endpoints, crossings,
    #    and collinear pass-throughs), emitting the chain of primitive sub-segments.
    P = np.asarray(pts)
    new_bars = set()
    for (a, c) in bars:
        pa, pc = P[a], P[c]
        d = pc - pa
        L2 = float(d @ d)
        on = []
        for k in range(len(P)):
            t = float((P[k] - pa) @ d) / L2
            if -tol <= t <= 1.0 + tol and np.hypot(*(P[k] - (pa + t * d))) < tol:
                on.append((t, k))
        on.sort()
        for (_, k1), (_, k2) in zip(on, on[1:]):
            if k1 != k2:
                new_bars.add((min(k1, k2), max(k1, k2)))

    bars_out = np.asarray(sorted(new_bars), dtype=int).reshape(-1, 2)
    return P, bars_out


def build_ground_structure(
    grid_nx: int,
    grid_ny: int,
    Lx: float,
    Ly: float,
    radius: float | None = None,
    planarize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a node grid and a radius-based bar connectivity.

    Nodes are placed on a regular ``grid_nx x grid_ny`` lattice spanning the
    domain (including the boundary).  A bar is created for every distinct node
    pair whose centre-to-centre distance is ``<= radius``.

    Parameters
    ----------
    grid_nx, grid_ny : int
        Number of nodes along x and y.
    Lx, Ly : float
        Domain extents.
    radius : float, optional
        Connection radius.  Defaults to ``1.01`` tile-diagonals, i.e. each node
        connects to its 8 orthogonal + diagonal neighbours (the classic
        ground-structure stencil).  A larger radius adds longer bars and, in the
        limit, yields the complete graph.
    planarize : bool, default True
        Make the structure a planar truss: add a node at every bar-bar crossing and
        split the bars there, and split any bar that passes through an existing node
        (see :func:`_planarize`). This turns each diagonal X into a real 4-way joint
        and removes collinear pass-through overlaps, so every intersection is a
        genuine connection point. Set ``False`` to keep the raw radius graph.

    Returns
    -------
    nodes : ndarray, shape (N, 2)
        Node coordinates, row-major (x fastest); any crossing nodes are appended.
    bars : ndarray, shape (B, 2), int
        Node-index pairs, ``i < j`` for each bar, sorted.
    """
    xs = np.linspace(0.0, Lx, grid_nx)
    ys = np.linspace(0.0, Ly, grid_ny)
    XX, YY = np.meshgrid(xs, ys)            # (grid_ny, grid_nx)
    nodes = np.column_stack([XX.ravel(), YY.ravel()])   # row-major: x fastest

    dx_tile = Lx / max(grid_nx - 1, 1)
    dy_tile = Ly / max(grid_ny - 1, 1)
    if radius is None:
        radius = 1.01 * float(np.hypot(dx_tile, dy_tile))

    # Pairwise distances; keep distinct pairs within the radius.
    diff = nodes[:, None, :] - nodes[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    iu, ju = np.triu_indices(len(nodes), k=1)
    mask = dist[iu, ju] <= radius + 1e-9
    bars = np.column_stack([iu[mask], ju[mask]]).astype(int)

    if planarize and len(bars):
        nodes, bars = _planarize(nodes, bars)
    return nodes, bars


@register_mapper("Truss")
@register_mapper("2D_Truss")
class Truss2DMapper(ProjectionMapper):
    """Node-based 2-D truss mapper.

    The mapper owns the (fixed) connectivity; node positions, bar thicknesses and
    bar mass densities are the design variables.
    """

    def __init__(
        self,
        num_components: int = 0,
        nodes: np.ndarray | None = None,
        bars: np.ndarray | None = None,
        r_gp: float = 0.5,
        method: str = "GP",
        Ngp: int = 2,
        min_thickness: float = 1.0,
        **kwargs,
    ):
        if nodes is None or bars is None:
            raise ValueError(
                "Truss2DMapper requires 'nodes' (N,2) and 'bars' (B,2); build them "
                "with build_ground_structure() and pass via the geometry kwargs."
            )
        self._nodes0 = np.asarray(nodes, dtype=float).reshape(-1, 2)
        self._bars = np.asarray(bars, dtype=int).reshape(-1, 2)
        self.num_nodes = self._nodes0.shape[0]
        self.num_bars = self._bars.shape[0]
        # One characteristic row per bar -> the KS aggregation runs over bars.
        self._num_components = self.num_bars

        self.r_gp = r_gp
        self.method = method
        self.Ngp = Ngp
        self.min_thickness = float(min_thickness) if min_thickness else 1.0

        # Local sampling-window Gauss points (identical to the free mapper).
        pts, wts = np.polynomial.legendre.leggauss(self.Ngp)
        pts = pts * self.r_gp
        wts = wts * self.r_gp
        gpcx, gpcy = np.meshgrid(pts, pts)
        self.gpc_x_rel = gpcx.flatten()
        self.gpc_y_rel = gpcy.flatten()
        self.gpc_wts = (wts[:, np.newaxis] * wts[np.newaxis, :]).flatten()
        self.gpc_wts_sum = np.sum(self.gpc_wts)
        self.num_gp_per_el = len(self.gpc_wts)

    # ------------------------------------------------------------------ #
    # Layout helpers
    # ------------------------------------------------------------------ #
    def num_design_vars(self) -> int:
        """Total design variables: 2 per node + 2 per bar (h, Mc)."""
        return 2 * self.num_nodes + 2 * self.num_bars

    def num_vars_per_component(self) -> int:
        # Not meaningful for a shared-node truss (variables are not per-bar);
        # callers should use num_design_vars().  Kept to satisfy the ABC.
        return 2

    def _parse(self, x_vars: np.ndarray):
        N, B = self.num_nodes, self.num_bars
        node_xy = x_vars[: 2 * N].reshape(N, 2)
        h = x_vars[2 * N : 2 * N + B]
        Mc = x_vars[2 * N + B : 2 * N + 2 * B]
        return node_xy, h, Mc

    def default_bounds(
        self, domain_extents: Tuple[float, ...], num_components: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Lx, Ly = domain_extents[0], domain_extents[1]
        diag = np.sqrt(Lx ** 2 + Ly ** 2)
        N, B = self.num_nodes, self.num_bars
        lb = np.zeros(2 * N + 2 * B)
        ub = np.zeros(2 * N + 2 * B)
        # Node positions — allow a small overshoot past the domain, like Free.
        lb[0 : 2 * N : 2] = -1.0;      ub[0 : 2 * N : 2] = Lx + 1.0   # node x
        lb[1 : 2 * N : 2] = -1.0;      ub[1 : 2 * N : 2] = Ly + 1.0   # node y
        # Bar thickness (minimum length scale) and mass density.
        lb[2 * N : 2 * N + B] = self.min_thickness
        ub[2 * N : 2 * N + B] = diag
        lb[2 * N + B : 2 * N + 2 * B] = 0.0
        ub[2 * N + B : 2 * N + 2 * B] = 1.0
        return lb, ub

    # ------------------------------------------------------------------ #
    # Endpoint <-> (Xc, Yc, L, theta) conversion
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bar_geometry(p1, p2, eps=1e-9):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        L = float(np.sqrt(dx * dx + dy * dy) + eps)
        Xc, Yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        theta = float(np.arctan2(dy, dx))
        return Xc, Yc, L, theta, dx, dy

    def _eval_points(self, eval_coords):
        X_eval = eval_coords[:, 0:1] + self.gpc_x_rel[np.newaxis, :]
        Y_eval = eval_coords[:, 1:2] + self.gpc_y_rel[np.newaxis, :]
        return X_eval.flatten(), Y_eval.flatten()

    def _integrate(self, arr_gp, num_elements):
        return (np.sum(arr_gp.reshape(num_elements, -1) * self.gpc_wts, axis=1)
                / self.gpc_wts_sum)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x_vars: np.ndarray,
        eval_coords: np.ndarray,
        power_E: float = 1.0,
        power_V: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_elements = eval_coords.shape[0]
        Xf, Yf = self._eval_points(eval_coords)
        node_xy, h, Mc = self._parse(x_vars)

        char_E, char_V = [], []
        for b in range(self.num_bars):
            a, c = self._bars[b]
            Xc, Yc, L, theta, _, _ = self._bar_geometry(node_xy[a], node_xy[c])
            W_gp = compute_local_characteristic_np(
                Xf, Yf, Xc, Yc, L, h[b], theta, self.r_gp, method=self.method
            )
            W_el = self._integrate(W_gp, num_elements)
            char_E.append(W_el * (Mc[b] ** power_E))
            char_V.append(W_el * (Mc[b] ** power_V))
        return np.array(char_E), np.array(char_V)

    # ------------------------------------------------------------------ #
    # Jacobian (global-column / is_continuous convention)
    # ------------------------------------------------------------------ #
    def jacobian(
        self,
        x_vars: np.ndarray,
        eval_coords: np.ndarray,
        power_E: float = 1.0,
        power_V: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        num_elements = eval_coords.shape[0]
        Xf, Yf = self._eval_points(eval_coords)
        node_xy, h, Mc = self._parse(x_vars)

        N, B = self.num_nodes, self.num_bars
        n_tot = 2 * N + 2 * B
        funcs_E = np.zeros((B, num_elements))
        funcs_V = np.zeros((B, num_elements))
        grads_E = np.zeros((B, n_tot, num_elements))
        grads_V = np.zeros((B, n_tot, num_elements))

        for b in range(B):
            a, c = self._bars[b]
            Xc, Yc, L, theta, dx, dy = self._bar_geometry(node_xy[a], node_xy[c])
            W_gp, dXc_gp, dYc_gp, dL_gp, dh_gp, dT_gp = (
                compute_local_characteristic_2d_with_grad_np(
                    Xf, Yf, Xc, Yc, L, h[b], theta, self.r_gp, method=self.method
                )
            )
            W_el = self._integrate(W_gp, num_elements)
            dXc = self._integrate(dXc_gp, num_elements)
            dYc = self._integrate(dYc_gp, num_elements)
            dL = self._integrate(dL_gp, num_elements)
            dh = self._integrate(dh_gp, num_elements)
            dT = self._integrate(dT_gp, num_elements)

            # Chain (Xc, Yc, L, theta) -> endpoints (x1,y1,x2,y2).
            #   Xc = (x1+x2)/2 ; Yc = (y1+y2)/2
            #   L  = |P2-P1| ; theta = atan2(dy, dx),  dx=x2-x1, dy=y2-y1
            L2 = L * L
            dW_dx1 = dXc * 0.5 + dL * (-dx / L) + dT * (dy / L2)
            dW_dx2 = dXc * 0.5 + dL * (dx / L) + dT * (-dy / L2)
            dW_dy1 = dYc * 0.5 + dL * (-dy / L) + dT * (-dx / L2)
            dW_dy2 = dYc * 0.5 + dL * (dy / L) + dT * (dx / L2)

            m_pE = Mc[b] ** power_E
            m_pV = Mc[b] ** power_V
            m_pE_m = power_E * (Mc[b] ** (power_E - 1.0)) if power_E > 0 else 0.0
            m_pV_m = power_V * (Mc[b] ** (power_V - 1.0)) if power_V > 0 else 0.0

            funcs_E[b, :] = W_el * m_pE
            funcs_V[b, :] = W_el * m_pV

            # Node-coordinate columns (shared across bars meeting at a node).
            cols = (2 * a, 2 * a + 1, 2 * c, 2 * c + 1)
            ends = (dW_dx1, dW_dy1, dW_dx2, dW_dy2)
            for col, g in zip(cols, ends):
                grads_E[b, col, :] = g * m_pE
                grads_V[b, col, :] = g * m_pV
            # Thickness column.
            idx_h = 2 * N + b
            grads_E[b, idx_h, :] = dh * m_pE
            grads_V[b, idx_h, :] = dh * m_pV
            # Mass-density column.
            idx_mc = 2 * N + B + b
            grads_E[b, idx_mc, :] = W_el * m_pE_m
            grads_V[b, idx_mc, :] = W_el * m_pV_m

        return {
            "funcs_E": funcs_E,
            "funcs_V": funcs_V,
            "grads_E": grads_E,
            "grads_V": grads_V,
            "is_continuous": True,
        }
