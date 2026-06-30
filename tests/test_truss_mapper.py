# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the node-based 2-D truss mapper.

Pure NumPy: the ground-structure builder, the bounds layout and the analytic
endpoint Jacobian don't need FEniCS / GEMSEO.
"""
from __future__ import annotations

import numpy as np
import pytest

from ggp.projection.truss_2d import Truss2DMapper, build_ground_structure
from ggp.projection.registry import get_mapper


# --------------------------------------------------------------------------- #
# Ground-structure connectivity
# --------------------------------------------------------------------------- #
def test_ground_structure_node_grid_is_row_major():
    nodes, bars = build_ground_structure(3, 2, 60.0, 30.0)
    assert nodes.shape == (6, 2)
    # x fastest: first row y=0 at x=0,30,60 ; second row y=30
    np.testing.assert_allclose(nodes[:3], [[0, 0], [30, 0], [60, 0]])
    np.testing.assert_allclose(nodes[3:], [[0, 30], [30, 30], [60, 30]])
    # bars are distinct, ordered pairs i<j
    assert np.all(bars[:, 0] < bars[:, 1])
    assert len(np.unique(bars, axis=0)) == len(bars)


def test_ground_structure_radius_controls_density():
    # tiny radius -> only the closest neighbours; huge radius -> complete graph
    _, sparse = build_ground_structure(3, 3, 60.0, 60.0, radius=21.0)   # only orthogonal (d=30)?
    _, dense = build_ground_structure(3, 3, 60.0, 60.0, radius=1e6)
    n = 9
    assert len(dense) == n * (n - 1) // 2          # complete graph
    assert len(sparse) < len(dense)


def test_default_radius_connects_eight_neighbours():
    # interior node of a 3x3 lattice should reach all 8 neighbours by default
    nodes, bars = build_ground_structure(3, 3, 60.0, 60.0)
    centre = 4                                       # middle node of row-major 3x3
    deg = np.sum((bars[:, 0] == centre) | (bars[:, 1] == centre))
    assert deg == 8


# --------------------------------------------------------------------------- #
# Variable layout & bounds
# --------------------------------------------------------------------------- #
def test_num_design_vars_and_bounds_layout():
    nodes, bars = build_ground_structure(4, 3, 60.0, 30.0)
    N, B = len(nodes), len(bars)
    m = Truss2DMapper(nodes=nodes, bars=bars, min_thickness=3.0)
    assert m.num_design_vars() == 2 * N + 2 * B
    lb, ub = m.default_bounds((60.0, 30.0), B)
    assert lb.shape == ub.shape == (2 * N + 2 * B,)
    # node x in [-1, Lx+1], node y in [-1, Ly+1]
    np.testing.assert_allclose(lb[0 : 2 * N : 2], -1.0)
    np.testing.assert_allclose(ub[0 : 2 * N : 2], 61.0)
    np.testing.assert_allclose(ub[1 : 2 * N : 2], 31.0)
    # bar thickness lower bound = min length scale; Mc in [0,1]
    np.testing.assert_allclose(lb[2 * N : 2 * N + B], 3.0)
    np.testing.assert_allclose(lb[2 * N + B :], 0.0)
    np.testing.assert_allclose(ub[2 * N + B :], 1.0)


def test_registry_resolves_truss_mode():
    nodes, bars = build_ground_structure(3, 2, 60.0, 30.0)
    m = get_mapper("Truss", num_components=len(bars), nodes=nodes, bars=bars)
    assert isinstance(m, Truss2DMapper)
    m2 = get_mapper("2D_Truss", num_components=len(bars), nodes=nodes, bars=bars)
    assert isinstance(m2, Truss2DMapper)


def test_missing_connectivity_raises():
    with pytest.raises(ValueError):
        Truss2DMapper(num_components=5)


# --------------------------------------------------------------------------- #
# Analytic endpoint Jacobian vs finite differences
# --------------------------------------------------------------------------- #
def _design_vector(nodes, bars, rng):
    N, B = len(nodes), len(bars)
    x = np.empty(2 * N + 2 * B)
    x[: 2 * N] = nodes.ravel() + rng.uniform(-3, 3, 2 * N)   # generic node positions
    x[2 * N : 2 * N + B] = rng.uniform(3.0, 6.0, B)          # thicknesses
    x[2 * N + B :] = rng.uniform(0.3, 0.8, B)                # mass densities
    return x


@pytest.mark.parametrize("method", ["GP", "rect"])
def test_truss_jacobian_matches_fd(method):
    rng = np.random.default_rng(0)
    nodes, bars = build_ground_structure(3, 2, 60.0, 30.0)
    m = Truss2DMapper(nodes=nodes, bars=bars, r_gp=1.0, method=method,
                      Ngp=2, min_thickness=2.0)
    x = _design_vector(nodes, bars, rng)

    ex, ey = np.meshgrid(np.linspace(2, 58, 10), np.linspace(2, 28, 5))
    eval_coords = np.column_stack([ex.ravel(), ey.ravel()])

    pE, pV = 3.0, 1.0
    jac = m.jacobian(x, eval_coords, pE, pV)
    assert jac["is_continuous"] is True
    gE = jac["grads_E"]                       # (B, n_tot, n_el)
    assert gE.shape == (len(bars), m.num_design_vars(), eval_coords.shape[0])

    # forward and jacobian funcs agree
    fE, _ = m.forward(x, eval_coords, pE, pV)
    np.testing.assert_allclose(fE, jac["funcs_E"])

    eps = 1e-6
    for k in range(m.num_design_vars()):
        xp = x.copy(); xp[k] += eps
        xm = x.copy(); xm[k] -= eps
        fEp, _ = m.forward(xp, eval_coords, pE, pV)
        fEm, _ = m.forward(xm, eval_coords, pE, pV)
        fd = (fEp - fEm) / (2 * eps)
        np.testing.assert_allclose(gE[:, k, :], fd, atol=1e-5, rtol=1e-3)


def test_shared_node_couples_multiple_bars():
    # Moving a shared node must change every bar incident to it -> its column is
    # non-zero in several bar-rows of the Jacobian.
    rng = np.random.default_rng(1)
    nodes, bars = build_ground_structure(3, 2, 60.0, 30.0)
    m = Truss2DMapper(nodes=nodes, bars=bars, r_gp=1.0, method="GP", Ngp=2)
    x = _design_vector(nodes, bars, rng)
    ex, ey = np.meshgrid(np.linspace(2, 58, 8), np.linspace(2, 28, 4))
    eval_coords = np.column_stack([ex.ravel(), ey.ravel()])
    gE = m.jacobian(x, eval_coords, 1.0, 1.0)["grads_E"]

    # pick a node touched by >1 bar
    node = 1
    incident = np.where((bars[:, 0] == node) | (bars[:, 1] == node))[0]
    assert len(incident) > 1
    col_x = 2 * node
    rows_nonzero = [b for b in range(len(bars)) if np.any(np.abs(gE[b, col_x, :]) > 1e-12)]
    # exactly the incident bars react to that node's x-coordinate
    assert set(rows_nonzero) == set(incident.tolist())
