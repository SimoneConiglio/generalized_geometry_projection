# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the AMJax FEM linear-system backend.

These tests are skipped automatically when amjax / pyamg / jax are not
installed so they never break environments that only have FEniCS.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps


# ── helpers ───────────────────────────────────────────────────────────────────

def _poisson_1d(n: int):
    """Return (K, f) for a 1-D Poisson problem with Dirichlet BCs applied."""
    # Tri-diagonal K
    diag = np.full(n, 2.0)
    off = np.full(n - 1, -1.0)
    K = sps.diags([off, diag, off], [-1, 0, 1], format="csr")
    # Pin DOF 0 and DOF n-1
    for i in (0, n - 1):
        K.data[K.indptr[i]:K.indptr[i + 1]] = 0.0
        K[i, i] = 1.0
    f = np.ones(n, dtype=np.float64)
    f[0] = 0.0
    f[n - 1] = 0.0
    return K.astype(np.float64), f


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def amjax_available():
    pytest.importorskip("amjax", reason="amjax not installed")
    pytest.importorskip("pyamg", reason="pyamg not installed")
    pytest.importorskip("jax", reason="jax not installed")


def test_solve_amjax_small_system(amjax_available):
    """AMJax solve should return a numpy array satisfying K u ≈ f."""
    from ggp.physics.amjax_solver import solve_amjax

    n = 50
    K, f = _poisson_1d(n)
    u = solve_amjax(K, f, tol=1e-10, maxiter=200)

    assert isinstance(u, np.ndarray), "solve_amjax must return a numpy array"
    assert u.shape == (n,)
    residual = np.linalg.norm(K @ u - f) / (np.linalg.norm(f) + 1e-15)
    assert residual < 1e-6, f"Residual too large: {residual:.2e}"


def test_solve_amjax_matches_direct(amjax_available):
    """AMJax and scipy spsolve should agree to within solver tolerance."""
    from scipy.sparse.linalg import spsolve
    from ggp.physics.amjax_solver import solve_amjax

    n = 80
    K, f = _poisson_1d(n)
    u_ref = spsolve(K, f)
    u_amjax = solve_amjax(K, f, tol=1e-10, maxiter=300)

    np.testing.assert_allclose(u_amjax, u_ref, rtol=1e-5, atol=1e-8,
                               err_msg="AMJax and direct solver solutions differ")


def test_solve_amjax_v_cycle(amjax_available):
    """V, W, and F cycles should all converge."""
    from ggp.physics.amjax_solver import solve_amjax

    n = 60
    K, f = _poisson_1d(n)
    for cycle in ("V", "W", "F"):
        u = solve_amjax(K, f, tol=1e-9, maxiter=200, cycle=cycle)
        residual = np.linalg.norm(K @ u - f) / (np.linalg.norm(f) + 1e-15)
        assert residual < 1e-5, f"Cycle {cycle}: residual {residual:.2e}"


def test_solve_amjax_output_dtype(amjax_available):
    """Output must be float64 regardless of JAX internal precision."""
    from ggp.physics.amjax_solver import solve_amjax

    K, f = _poisson_1d(30)
    u = solve_amjax(K, f)
    assert u.dtype == np.float64


def test_solver_spec_fem_solver_field():
    """SolverSpec.fem_solver should default to 'direct' and accept 'amjax'."""
    from dataclasses import replace
    from ggp.problem.spec import SolverSpec

    spec = SolverSpec()
    assert spec.fem_solver == "direct"

    spec_amjax = replace(spec, fem_solver="amjax")
    assert spec_amjax.fem_solver == "amjax"
