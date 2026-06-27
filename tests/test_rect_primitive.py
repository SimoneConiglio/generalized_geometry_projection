# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the rectangle primitive and the grid-fill initial design.

Pure-NumPy: the rectangle characteristic, the auto-grid factorisation and the
grid-fill initialiser don't need FEniCS/GEMSEO.
"""
from __future__ import annotations

import numpy as np
import pytest

from ggp.utils.vectorized_mapping import (
    compute_local_characteristic_np,
    compute_local_characteristic_2d_with_grad_np,
)
from ggp.projection.free_2d import Free2DMapper
from ggp.optimization.pipeline import GGPPipeline, _auto_grid


# --------------------------------------------------------------------------- #
# Rectangle characteristic
# --------------------------------------------------------------------------- #
def test_rect_characteristic_is_a_rectangle():
    # A rectangle centred at origin, width L=10 (x in [-5,5]), height h=4 (y in [-2,2]).
    L, h, r_gp = 10.0, 4.0, 0.5
    pts = np.array([
        [0.0, 0.0],     # centre -> inside
        [4.0, 1.5],     # inside
        [6.0, 0.0],     # outside in x
        [0.0, 3.0],     # outside in y
    ])
    W = compute_local_characteristic_np(pts[:, 0], pts[:, 1], 0.0, 0.0, L, h, 0.0,
                                        r_gp, method="rect")
    assert W[0] > 0.99 and W[1] > 0.99      # interior ~1
    assert W[2] < 0.01 and W[3] < 0.01      # exterior ~0
    # 'rect' and 'AMNA' are the same characteristic
    W_amna = compute_local_characteristic_np(pts[:, 0], pts[:, 1], 0.0, 0.0, L, h, 0.0,
                                            r_gp, method="AMNA")
    np.testing.assert_allclose(W, W_amna)


def test_rect_characteristic_gradient_matches_fd():
    rng = np.random.default_rng(0)
    X = rng.uniform(-8, 8, 40); Y = rng.uniform(-5, 5, 40)
    Xc, Yc, L, h, T, r_gp = 1.0, -0.5, 9.0, 5.0, 0.3, 0.5

    def W_of(params):
        return compute_local_characteristic_np(X, Y, *params, r_gp, method="rect")

    W, dX, dY, dL, dh, dT = compute_local_characteristic_2d_with_grad_np(
        X, Y, Xc, Yc, L, h, T, r_gp, method="rect")
    base = [Xc, Yc, L, h, T]
    eps = 1e-6
    for k, analytic in enumerate((dX, dY, dL, dh, dT)):
        pp = list(base); pp[k] += eps
        pm = list(base); pm[k] -= eps
        fd = (W_of(pp) - W_of(pm)) / (2 * eps)
        np.testing.assert_allclose(analytic, fd, rtol=1e-3, atol=1e-4)


# --------------------------------------------------------------------------- #
# Auto-grid factorisation
# --------------------------------------------------------------------------- #
def test_auto_grid_square_tiles():
    # 60x30 domain, 18 comps -> 6x3 so tiles are 10x10 (square)
    assert _auto_grid(18, 60.0, 30.0) == (6, 3)
    # square domain -> near-square factor count
    assert _auto_grid(16, 30.0, 30.0) == (4, 4)
    gnx, gny = _auto_grid(12, 60.0, 30.0)
    assert gnx * gny == 12 and gnx >= gny      # wider domain -> more x tiles


# --------------------------------------------------------------------------- #
# Grid-fill initial design
# --------------------------------------------------------------------------- #
def test_grid_fill_init_layout():
    Lx, Ly, gnx, gny, vf = 60.0, 30.0, 6, 3, 0.4
    nc = gnx * gny
    lb, ub = Free2DMapper(nc).default_bounds((Lx, Ly), nc)
    x = GGPPipeline._make_init("Free", nc * 6, Lx=Lx, Ly=Ly, lb=lb, ub=ub,
                               init="grid", grid_nx=gnx, grid_ny=gny, volfrac=vf)
    assert x.shape == (nc * 6,)
    assert np.all((x >= 0.0) & (x <= 1.0))
    # Mc == volfrac for every component
    np.testing.assert_allclose(x[5::6], vf)
    # de-normalise positions and check they are the tile centres
    Xc = lb[0::6] + x[0::6] * (ub[0::6] - lb[0::6])
    Yc = lb[1::6] + x[1::6] * (ub[1::6] - lb[1::6])
    Lc = lb[2::6] + x[2::6] * (ub[2::6] - lb[2::6])
    hc = lb[3::6] + x[3::6] * (ub[3::6] - lb[3::6])
    np.testing.assert_allclose(np.sort(np.unique(np.round(Xc, 6))), [5, 15, 25, 35, 45, 55])
    np.testing.assert_allclose(np.sort(np.unique(np.round(Yc, 6))), [5, 15, 25])
    np.testing.assert_allclose(Lc, Lx / gnx)   # 10-wide rectangles
    np.testing.assert_allclose(hc, Ly / gny)   # 10-tall rectangles


def test_grid_fill_init_rejects_mismatched_counts():
    lb, ub = Free2DMapper(18).default_bounds((60.0, 30.0), 18)
    with pytest.raises(ValueError):
        GGPPipeline._make_init("Free", 18 * 6, Lx=60.0, Ly=30.0, lb=lb, ub=ub,
                               init="grid", grid_nx=5, grid_ny=3, volfrac=0.4)  # 5*3 != 18
