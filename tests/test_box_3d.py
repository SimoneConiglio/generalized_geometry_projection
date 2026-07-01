# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the 3-D box primitive mapper (JAX autodiff gradients)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from ggp.projection.box_3d import Box3DMapper
from ggp.projection.registry import get_mapper
from ggp.utils.jax_primitives import box_batch


def test_box_characteristic_is_a_box():
    # axis-aligned box centred at origin, sizes 10 x 6 x 4
    pts = np.array([[0, 0, 0], [4, 2, 1],          # interior
                    [6, 0, 0], [0, 4, 0], [0, 0, 3]], float)  # outside x / y / z
    W, *_ = box_batch(pts, np.zeros(3), np.array([10., 6., 4.]), 0.0, 0.0, 0.4)
    assert W[0] > 0.99 and W[1] > 0.99
    assert W[2] < 0.01 and W[3] < 0.01 and W[4] < 0.01


def test_box_rotation_theta_90_swaps_extent():
    # rotate 90° about z: the long (x) axis now lies along y
    L = np.array([12., 4., 4.])
    W_along_y, *_ = box_batch(np.array([[0., 5., 0.]]), np.zeros(3), L, np.pi / 2, 0.0, 0.4)
    W_along_x, *_ = box_batch(np.array([[5., 0., 0.]]), np.zeros(3), L, np.pi / 2, 0.0, 0.4)
    assert W_along_y[0] > 0.9      # point at y=5 is inside the rotated long axis
    assert W_along_x[0] < 0.1      # point at x=5 is now outside (half-extent 2 in x)


def test_registry_resolves_box3d():
    assert isinstance(get_mapper("3D_Box", num_components=3), Box3DMapper)
    assert isinstance(get_mapper("Box3D", num_components=3), Box3DMapper)


def test_box3d_bounds_layout():
    m = Box3DMapper(4, min_thickness=3.0)
    lb, ub = m.default_bounds((30., 20., 20.), 4)
    assert lb.shape == (4 * 9,)
    np.testing.assert_allclose(lb[4::9], 3.0)      # Ly min thickness
    np.testing.assert_allclose(lb[5::9], 3.0)      # Lz min thickness
    np.testing.assert_allclose(lb[8::9], 0.0)      # Mc
    np.testing.assert_allclose(ub[8::9], 1.0)


def test_box3d_jacobian_matches_fd():
    rng = np.random.default_rng(0)
    N = 3
    m = Box3DMapper(N, r_gp=0.5, Ngp=2, min_thickness=2.0)
    x = np.empty(N * 9)
    for i in range(N):
        x[i * 9:i * 9 + 3] = rng.uniform(5, 25, 3)
        x[i * 9 + 3:i * 9 + 6] = rng.uniform(6, 12, 3)
        x[i * 9 + 6] = rng.uniform(0, 1)
        x[i * 9 + 7] = rng.uniform(-0.5, 0.5)
        x[i * 9 + 8] = 0.6
    gx, gy, gz = np.meshgrid(np.linspace(2, 28, 5), np.linspace(2, 18, 3),
                             np.linspace(2, 18, 3), indexing="ij")
    ec = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    pE, pV = 3.0, 1.0
    jac = m.jacobian(x, ec, pE, pV)
    gE = jac["grads_E"]
    assert gE.shape == (N, 9, ec.shape[0])
    fEf, _ = m.forward(x, ec, pE, pV)
    np.testing.assert_allclose(fEf, jac["funcs_E"])

    eps = 1e-6
    for k in rng.choice(N * 9, 18, replace=False):
        xp = x.copy(); xp[k] += eps
        xm = x.copy(); xm[k] -= eps
        fEp, _ = m.forward(xp, ec, pE, pV)
        fEm, _ = m.forward(xm, ec, pE, pV)
        fd = (fEp - fEm) / (2 * eps)
        comp, loc = k // 9, k % 9
        np.testing.assert_allclose(gE[comp, loc, :], fd[comp], atol=1e-6, rtol=1e-3)
