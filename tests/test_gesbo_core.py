# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the GEMSEO-free GE-SBO core (:mod:`scp_uno.gesbo_core`).

These tests exercise the gradient-enhanced kriging surrogate, the
active-subspace reduction, the batch acquisition and the trust-region driver
on analytic problems; no FEniCS or GEMSEO is required.
"""
import numpy as np
import pytest

from scp_uno.gesbo_core import (
    GESBOConfig,
    GESBOptimizer,
    GradientEnhancedGP,
    active_subspace,
    gesbo_minimize,
)


# --------------------------------------------------------------------------- #
# Analytic test problems (x -> (f, grad_f, cons, jac_cons))
# --------------------------------------------------------------------------- #
def sphere_problem(d, xstar=0.3):
    target = np.full(d, xstar)

    def evaluate(x):
        r = x - target
        return float(r @ r), 2.0 * r, np.zeros(0), np.zeros((0, d))

    return evaluate


def constrained_problem():
    """min (x-1)^2 + (y-1)^2  s.t.  x + y <= 1; optimum (0.5, 0.5), f = 0.5."""

    def evaluate(x):
        f = (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2
        g = np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 1.0)])
        return float(f), g, np.array([x[0] + x[1] - 1.0]), np.array([[1.0, 1.0]])

    return evaluate


# --------------------------------------------------------------------------- #
# Gradient-enhanced GP
# --------------------------------------------------------------------------- #
def _quad(X):
    return (X[:, 0] - 0.3) ** 2 + 2.0 * (X[:, 1] + 0.1) ** 2


def _quad_grad(X):
    return np.column_stack([2.0 * (X[:, 0] - 0.3), 4.0 * (X[:, 1] + 0.1)])


def test_gek_interpolates_values_and_beats_value_only_kriging():
    rng = np.random.default_rng(0)
    Z = rng.random((8, 2)) * 2.0 - 1.0
    Y = _quad(Z)[:, None]
    G = _quad_grad(Z)[:, :, None]

    gek = GradientEnhancedGP().fit(Z, Y, G)
    # interpolation of the training values
    mean_train, _ = gek.predict(Z)
    assert np.max(np.abs(mean_train[:, 0] - _quad(Z))) < 1e-3

    # gradient enhancement improves interior prediction over value-only kriging
    Zq = rng.random((200, 2)) * 1.6 - 0.8
    rmse_ge = np.sqrt(np.mean((gek.predict(Zq)[0][:, 0] - _quad(Zq)) ** 2))
    value_only = GradientEnhancedGP(use_gradients=False).fit(Z, Y)
    rmse_v = np.sqrt(np.mean((value_only.predict(Zq)[0][:, 0] - _quad(Zq)) ** 2))
    assert rmse_ge < rmse_v


def test_gek_analytic_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    Z = rng.random((10, 3))
    Y = np.column_stack([np.sin(Z @ np.array([1.0, 2.0, -1.0])), Z[:, 0] ** 2])
    G = np.stack(
        [
            np.column_stack(
                [
                    np.cos(Z[i] @ np.array([1.0, 2.0, -1.0]))
                    * np.array([1.0, 2.0, -1.0]),
                    np.array([2.0 * Z[i, 0], 0.0, 0.0]),
                ]
            )
            for i in range(len(Z))
        ]
    )
    gek = GradientEnhancedGP().fit(Z, Y, G)

    z0 = np.array([0.4, 0.5, 0.6])
    mean, sigma, dmean, dsig = gek.predict_std_single(z0)
    eps = 1e-6
    for k in range(3):
        zp, zm = z0.copy(), z0.copy()
        zp[k] += eps
        zm[k] -= eps
        mp, sp = gek.predict_std(zp[None])
        mm, sm = gek.predict_std(zm[None])
        fd_mean = (mp[0] - mm[0]) / (2 * eps)
        fd_sig = (sp[0] - sm[0]) / (2 * eps)
        assert np.allclose(dmean[k], fd_mean, rtol=1e-4, atol=1e-6)
        assert np.isclose(dsig[k], fd_sig, rtol=1e-3, atol=1e-6)


def test_gek_multi_output_shares_kernel():
    rng = np.random.default_rng(2)
    Z = rng.random((12, 2))
    Y = np.column_stack([Z[:, 0] + Z[:, 1], Z[:, 0] - Z[:, 1]])
    G = np.stack([np.array([[1.0, 1.0], [1.0, -1.0]]).T for _ in range(12)])
    gek = GradientEnhancedGP().fit(Z, Y, G)
    mean, std = gek.predict(rng.random((5, 2)))
    assert mean.shape == (5, 2)
    assert std.shape == (5, 2)
    assert np.all(std >= 0.0)


# --------------------------------------------------------------------------- #
# Active subspace
# --------------------------------------------------------------------------- #
def test_active_subspace_recovers_gradient_span():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((3, 40))          # f(x) = h(Ax): gradients in span(A^T)
    gradients = rng.standard_normal((30, 3)) @ A
    W = active_subspace(gradients, max_dim=5, rng=rng)
    assert W.shape == (40, 5)
    # orthonormal columns
    assert np.allclose(W.T @ W, np.eye(5), atol=1e-10)
    # the true 3-dimensional gradient span is captured
    residual = np.linalg.norm(A @ W @ W.T - A) / np.linalg.norm(A)
    assert residual < 1e-10


def test_active_subspace_identity_in_low_dimension():
    rng = np.random.default_rng(4)
    W = active_subspace(rng.standard_normal((5, 6)), max_dim=10, rng=rng)
    assert np.array_equal(W, np.eye(6))


def test_active_subspace_zero_gradients_fallback():
    rng = np.random.default_rng(5)
    W = active_subspace(np.zeros((4, 30)), max_dim=6, rng=rng)
    assert W.shape == (30, 6)
    assert np.allclose(W.T @ W, np.eye(6), atol=1e-10)


# --------------------------------------------------------------------------- #
# Driver: batch behaviour
# --------------------------------------------------------------------------- #
def test_each_iteration_acquires_a_batch_of_distinct_points():
    d, q = 6, 3
    records = []
    cfg = GESBOConfig(batch_size=q, max_evals=40, max_outer_iter=6, seed=0)
    opt = GESBOptimizer(
        sphere_problem(d), np.zeros(d), np.ones(d), cfg,
        on_iteration=records.append,
    )
    result = opt.run(np.full(d, 0.8))

    assert len(records) >= 2
    for rec in records:
        idx = rec["batch_indices"]
        assert 1 <= len(idx) <= q
        pts = [opt.X[i] for i in idx]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                assert np.linalg.norm(pts[a] - pts[b]) > 1e-8
    # full batches whenever the budget allowed it
    assert sum(len(r["batch_indices"]) for r in records) + cfg.batch_size + 1 \
        <= result.n_evals + q


def test_evaluation_budget_is_respected():
    d = 4
    cfg = GESBOConfig(batch_size=4, max_evals=17, seed=0)
    result = gesbo_minimize(
        sphere_problem(d), np.full(d, 0.9), np.zeros(d), np.ones(d), cfg
    )
    assert result.n_evals <= 17


# --------------------------------------------------------------------------- #
# Driver: convergence
# --------------------------------------------------------------------------- #
def test_converges_on_20d_sphere():
    d = 20
    result = gesbo_minimize(
        sphere_problem(d),
        np.full(d, 0.8),
        np.zeros(d),
        np.ones(d),
        GESBOConfig(max_evals=120, seed=1),
    )
    f0 = sphere_problem(d)(np.full(d, 0.8))[0]
    assert result.f_opt < 1e-2 * f0
    assert result.is_feasible


def test_converges_on_constrained_problem():
    result = gesbo_minimize(
        constrained_problem(),
        np.array([0.1, 0.1]),
        np.zeros(2),
        np.ones(2),
        GESBOConfig(max_evals=80, seed=2),
    )
    assert result.is_feasible
    assert result.constraints[0] <= 1e-6
    assert result.f_opt < 0.62          # true optimum: 0.5 at (0.5, 0.5)


def test_scales_to_high_dimension_with_low_effective_rank():
    """200 design variables, 5 effective directions: the active subspace makes
    the surrogate tractable and the optimizer converges within 100 evaluations."""
    d = 200
    rng = np.random.default_rng(6)
    B = rng.standard_normal((d, 5)) / np.sqrt(d)

    def evaluate(x):
        t = B.T @ (x - 0.5)
        return float(t @ t), 2.0 * B @ t, np.zeros(0), np.zeros((0, d))

    x0 = np.full(d, 0.9)
    f0 = evaluate(x0)[0]
    result = gesbo_minimize(
        evaluate, x0, np.zeros(d), np.ones(d),
        GESBOConfig(max_evals=100, seed=3),
    )
    assert result.f_opt < 1e-6 * f0
    assert result.n_evals <= 100


def test_value_only_mode_still_optimizes():
    d = 3
    result = gesbo_minimize(
        sphere_problem(d),
        np.full(d, 0.9),
        np.zeros(d),
        np.ones(d),
        GESBOConfig(max_evals=60, use_gradients=False, seed=4),
    )
    f0 = sphere_problem(d)(np.full(d, 0.9))[0]
    assert result.f_opt < 0.2 * f0


def test_result_fields_are_consistent():
    d = 5
    result = gesbo_minimize(
        sphere_problem(d),
        np.full(d, 0.6),
        np.zeros(d),
        np.ones(d),
        GESBOConfig(max_evals=30, seed=5),
    )
    assert result.x_opt.shape == (d,)
    assert np.all(result.x_opt >= 0.0) and np.all(result.x_opt <= 1.0)
    assert result.n_iter == len(result.history)
    assert result.status
    r = result.x_opt - 0.3
    assert np.isclose(result.f_opt, float(r @ r), rtol=1e-10)
