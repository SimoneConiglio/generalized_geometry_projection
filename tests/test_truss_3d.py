# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the node-based 3-D truss mapper (JAX capsule primitive)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from ggp.projection.truss_3d import (
    Truss3DMapper, build_ground_structure_3d, _split_passthrough_3d,
)
from ggp.projection.registry import get_mapper
from ggp.utils.jax_primitives import capsule_batch


def _passthrough_count(nodes, bars, tol=1e-6):
    bad = 0
    for (a, c) in bars:
        pa, pc = nodes[a], nodes[c]
        d = pc - pa
        L2 = float(d @ d)
        for k in range(len(nodes)):
            if k in (a, c):
                continue
            t = float((nodes[k] - pa) @ d) / L2
            if t <= tol or t >= 1.0 - tol:
                continue
            if np.linalg.norm(nodes[k] - (pa + t * d)) < tol:
                bad += 1
                break
    return bad


def test_ground_structure_3d_lattice_and_no_passthrough():
    nodes, bars = build_ground_structure_3d(3, 2, 2, 30.0, 20.0, 20.0)
    assert nodes.shape == (12, 3)
    assert np.all(bars[:, 0] < bars[:, 1]) and len(np.unique(bars, axis=0)) == len(bars)
    # collinear pass-through bars are split out
    assert _passthrough_count(nodes, bars) == 0
    # a denser (complete) graph would have pass-throughs before splitting
    raw_n, raw_b = build_ground_structure_3d(3, 3, 1, 20.0, 20.0, 0.0,
                                             radius=1e6, split_passthrough=False)
    assert _passthrough_count(raw_n, raw_b) > 0


def test_registry_resolves_3d_truss():
    nodes, bars = build_ground_structure_3d(3, 2, 2, 30.0, 20.0, 20.0)
    m = get_mapper("3D_Truss", num_components=len(bars), nodes=nodes, bars=bars)
    assert isinstance(m, Truss3DMapper)
    assert get_mapper("Truss3D", num_components=len(bars), nodes=nodes, bars=bars) is not None


def test_num_design_vars_and_bounds():
    nodes, bars = build_ground_structure_3d(3, 2, 2, 30.0, 20.0, 20.0)
    N, B = len(nodes), len(bars)
    m = Truss3DMapper(nodes=nodes, bars=bars, min_thickness=3.0)
    assert m.num_design_vars() == 3 * N + 2 * B
    lb, ub = m.default_bounds((30.0, 20.0, 20.0), B)
    assert lb.shape == (3 * N + 2 * B,)
    np.testing.assert_allclose(lb[3 * N:3 * N + B], 3.0)     # min thickness
    np.testing.assert_allclose(lb[3 * N + B:], 0.0)          # Mc lower
    np.testing.assert_allclose(ub[3 * N + B:], 1.0)


def test_vertical_bar_has_finite_gradient():
    # a z-aligned bar would be singular under a (theta,phi) chain; the segment
    # capsule keeps gradients finite
    P1 = np.array([5.0, 5.0, 1.0]); P2 = np.array([5.0, 5.0, 9.0])
    pts = np.random.default_rng(0).uniform(2, 8, (40, 3))
    W, dP1, dP2, dh = capsule_batch(pts, P1, P2, 3.0, 0.5)
    assert np.all(np.isfinite(dP1)) and np.all(np.isfinite(dP2)) and np.all(np.isfinite(dh))


def test_truss_3d_jacobian_matches_fd():
    rng = np.random.default_rng(1)
    nodes, bars = build_ground_structure_3d(3, 2, 2, 30.0, 20.0, 20.0)
    N, B = len(nodes), len(bars)
    m = Truss3DMapper(nodes=nodes, bars=bars, r_gp=1.0, Ngp=2, min_thickness=2.0)
    ntot = m.num_design_vars()
    x = np.empty(ntot)
    x[:3 * N] = nodes.ravel() + rng.uniform(-2, 2, 3 * N)   # generic node positions
    x[3 * N:3 * N + B] = 4.0
    x[3 * N + B:] = 0.6
    gx, gy, gz = np.meshgrid(np.linspace(2, 28, 5), np.linspace(2, 18, 3),
                             np.linspace(2, 18, 3), indexing="ij")
    ec = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    pE, pV = 3.0, 1.0
    jac = m.jacobian(x, ec, pE, pV)
    assert jac["is_continuous"] is True
    gE = jac["grads_E"]
    assert gE.shape == (B, ntot, ec.shape[0])
    fEf, _ = m.forward(x, ec, pE, pV)
    np.testing.assert_allclose(fEf, jac["funcs_E"])

    eps = 1e-6
    for k in rng.choice(ntot, 18, replace=False):
        xp = x.copy(); xp[k] += eps
        xm = x.copy(); xm[k] -= eps
        fEp, _ = m.forward(xp, ec, pE, pV)
        fEm, _ = m.forward(xm, ec, pE, pV)
        fd = (fEp - fEm) / (2 * eps)
        np.testing.assert_allclose(gE[:, k, :], fd, atol=1e-6, rtol=1e-3)


def test_shared_node_couples_incident_bars():
    rng = np.random.default_rng(2)
    nodes, bars = build_ground_structure_3d(3, 2, 2, 30.0, 20.0, 20.0)
    N, B = len(nodes), len(bars)
    m = Truss3DMapper(nodes=nodes, bars=bars, r_gp=1.0, Ngp=2)
    x = np.empty(m.num_design_vars())
    x[:3 * N] = nodes.ravel() + rng.uniform(-1, 1, 3 * N)
    x[3 * N:3 * N + B] = 4.0
    x[3 * N + B:] = 0.6
    gx, gy, gz = np.meshgrid(np.linspace(2, 28, 4), np.linspace(2, 18, 3),
                             np.linspace(2, 18, 3), indexing="ij")
    ec = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    gE = m.jacobian(x, ec, 1.0, 1.0)["grads_E"]
    node = 1
    incident = set(np.where((bars[:, 0] == node) | (bars[:, 1] == node))[0].tolist())
    col = 3 * node
    rows = set(b for b in range(B) if np.any(np.abs(gE[b, col, :]) > 1e-12))
    # a node's coordinate may only affect bars incident to it (never others), and it
    # does affect at least one of them (those whose characteristic is nonzero here)
    assert rows.issubset(incident)
    assert len(rows) > 0
