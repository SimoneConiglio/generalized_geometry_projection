# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the GEMSEO-free MLS-SBO core (:mod:`scp_uno.mls_sbo_core`).

Exercises the gradient-enhanced Moving Least Squares surrogate (linear-exact
Hermite fit, diffuse derivatives, density-proxy sigma), the error-adaptive
length scale and the trust-region batch driver on analytic problems; no
FEniCS or GEMSEO required.
"""
import numpy as np
import pytest

from scp_uno.mls_sbo_core import (
    AnchoredSeparableQuadratic,
    MLSSBOConfig,
    MLSSBOptimizer,
    MovingLeastSquares,
    mls_sbo_minimize,
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
# Moving Least Squares surrogate
# --------------------------------------------------------------------------- #
def _fit_linear(h, rng, use_gradients=True):
    a, b = 1.7, np.array([0.5, -2.0, 3.0])
    X = rng.random((12, 3))
    Y = (a + X @ b)[:, None]
    G = np.tile(b[:, None], (12, 1, 1)).astype(float)
    mls = MovingLeastSquares(lengthscale=h, use_gradients=use_gradients)
    return mls.fit(X, Y, G), a, b


@pytest.mark.parametrize("h", [0.05, 0.3, 2.0])
def test_mls_reproduces_linear_functions_exactly(h):
    """Linear exactness at any length scale: values and diffuse gradients."""
    rng = np.random.default_rng(0)
    mls, a, b = _fit_linear(h, rng)
    Xq = rng.random((20, 3))
    mean, _ = mls.predict(Xq)
    assert np.allclose(mean[:, 0], a + Xq @ b, atol=1e-6)
    m0, _, dmean, _ = mls.predict_std_single(Xq[0])
    assert np.allclose(mls._ym[0] + mls._ys[0] * m0[0], a + Xq[0] @ b, atol=1e-6)
    assert np.allclose(mls._ys[0] * dmean[:, 0], b, rtol=1e-5, atol=1e-6)


def test_gradient_enhancement_improves_curved_fit():
    rng = np.random.default_rng(1)

    def f(X):
        return np.sin(3 * X[:, 0]) + (X[:, 1] - 0.4) ** 2

    def g(X):
        return np.column_stack([3 * np.cos(3 * X[:, 0]), 2 * (X[:, 1] - 0.4)])

    X = rng.random((10, 2))
    Y = f(X)[:, None]
    G = g(X)[:, :, None]
    Xq = rng.random((200, 2))
    with_g = MovingLeastSquares(0.4).fit(X, Y, G)
    without_g = MovingLeastSquares(0.4, use_gradients=False).fit(X, Y, G)
    rmse_g = np.sqrt(np.mean((with_g.predict(Xq)[0][:, 0] - f(Xq)) ** 2))
    rmse_v = np.sqrt(np.mean((without_g.predict(Xq)[0][:, 0] - f(Xq)) ** 2))
    assert rmse_g < rmse_v


def test_small_lengthscale_interpolates_nearest_sample():
    rng = np.random.default_rng(2)
    X = rng.random((8, 2))
    Y = rng.random((8, 1)) * 5
    G = rng.standard_normal((8, 2, 1))
    mls = MovingLeastSquares(lengthscale=1e-3).fit(X, Y, G)
    mean, _ = mls.predict(X)
    assert np.allclose(mean[:, 0], Y[:, 0], atol=1e-4)


def test_sigma_density_proxy_and_gradients():
    rng = np.random.default_rng(3)
    X = rng.random((15, 3))
    Y = rng.random((15, 2))
    G = rng.standard_normal((15, 3, 2))
    mls = MovingLeastSquares(lengthscale=0.2).fit(X, Y, G)
    # ~0 at a training point, -> 1 far from all data
    _, sig_at = mls.predict_std(X[:1])
    assert sig_at[0] < 0.05
    _, sig_far = mls.predict_std(np.full((1, 3), 25.0))
    assert sig_far[0] > 0.999

    x0 = np.array([0.4, 0.5, 0.6])
    _, s0, _, dsig = mls.predict_std_single(x0)
    eps = 1e-6
    for k in range(3):
        xp, xm = x0.copy(), x0.copy()
        xp[k] += eps
        xm[k] -= eps
        fd = (mls.predict_std(xp[None])[1][0] - mls.predict_std(xm[None])[1][0]) \
            / (2 * eps)
        assert np.isclose(dsig[k], fd, rtol=1e-4, atol=1e-7)


def test_multi_output_shapes():
    rng = np.random.default_rng(4)
    mls = MovingLeastSquares(0.3).fit(
        rng.random((9, 4)), rng.random((9, 3)), rng.standard_normal((9, 4, 3))
    )
    mean, std = mls.predict(rng.random((5, 4)))
    assert mean.shape == (5, 3)
    assert std.shape == (5, 3)
    assert np.all(std >= 0.0)


# --------------------------------------------------------------------------- #
# Anchored (first-order consistent) model
# --------------------------------------------------------------------------- #
def _anchored_from_data(X, Y, G, k, h=0.4):
    """Build the anchored model exactly as the driver does, anchored at row k."""
    return AnchoredSeparableQuadratic(X[k], Y[k], G[k], X, G, h)


def test_anchored_model_is_first_order_consistent_at_center():
    rng = np.random.default_rng(10)
    X = rng.random((9, 3))
    Y = rng.standard_normal((9, 2)) * 5.0
    G = rng.standard_normal((9, 3, 2))
    k = 4
    anchored = _anchored_from_data(X, Y, G, k)
    val, grad = anchored.value_and_slope(X[k])
    assert np.allclose(val, Y[k], atol=1e-12)
    assert np.allclose(grad, G[k], atol=1e-12)


def test_anchored_model_gradient_is_consistent_with_value():
    """The gradient handed to the subproblem solver must be the true
    derivative of the model value (this is what the per-query diffuse MLS
    derivative violated)."""
    rng = np.random.default_rng(12)
    X = rng.random((8, 3))
    Y = rng.standard_normal((8, 2))
    G = rng.standard_normal((8, 3, 2))
    anchored = _anchored_from_data(X, Y, G, 2)
    x0 = np.array([0.35, 0.55, 0.45])
    _, grad = anchored.value_and_slope(x0)
    eps = 1e-6
    for k in range(3):
        xp, xm = x0.copy(), x0.copy()
        xp[k] += eps
        xm[k] -= eps
        fd = (anchored.value_and_slope(xp)[0]
              - anchored.value_and_slope(xm)[0]) / (2 * eps)
        assert np.allclose(grad[k], fd, rtol=1e-6, atol=1e-8)


def test_anchored_model_recovers_separable_quadratic():
    """On a separable quadratic the secant-fitted diagonal Hessian is exact,
    so the model reproduces the function and its minimizer."""
    rng = np.random.default_rng(11)
    d = 4
    a = np.array([1.0, 2.0, 0.5, 3.0])
    b = np.array([0.3, 0.4, 0.6, 0.5])

    def f(x):
        return float(a @ (x - b) ** 2), 2.0 * a * (x - b)

    X = 0.5 + 0.2 * rng.standard_normal((10, d))
    Y = np.array([[f(x)[0]] for x in X])
    G = np.stack([f(x)[1][:, None] for x in X])
    anchored = _anchored_from_data(X, Y, G, 0)
    assert np.allclose(anchored.q[:, 0], 2.0 * a, rtol=1e-6)
    xq = rng.random(d)
    val, grad = anchored.value_and_slope(xq)
    assert np.isclose(val[0], f(xq)[0], rtol=1e-6, atol=1e-8)
    assert np.allclose(grad[:, 0], f(xq)[1], rtol=1e-6, atol=1e-8)


# --------------------------------------------------------------------------- #
# Length-scale adaptation
# --------------------------------------------------------------------------- #
def test_lengthscale_tracks_min_distance_in_trust_region():
    """The length scale must equal ``ls_factor * d_min`` (clipped) at every
    iteration, and localize as the run converges and samples cluster."""
    d = 6
    cfg = MLSSBOConfig(max_evals=100, seed=0)
    records = []
    mls_sbo_minimize(
        sphere_problem(d), np.full(d, 0.8), np.zeros(d), np.ones(d), cfg,
        on_iteration=records.append,
    )
    assert records
    for rec in records:
        expected = float(np.clip(cfg.ls_factor * rec["min_dist"],
                                 cfg.ls_min, cfg.ls_max))
        assert np.isclose(rec["lengthscale"], expected, rtol=1e-12)
    # converging run: the fit localizes with the sample spacing
    assert records[-1]["lengthscale"] < records[0]["lengthscale"]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def test_each_iteration_acquires_a_batch_and_budget_respected():
    d, q = 6, 3
    records = []
    cfg = MLSSBOConfig(batch_size=q, max_evals=40, max_outer_iter=6, seed=0)
    opt = MLSSBOptimizer(
        sphere_problem(d), np.zeros(d), np.ones(d), cfg,
        on_iteration=records.append,
    )
    result = opt.run(np.full(d, 0.8))
    assert result.n_evals <= 40
    assert records
    for rec in records:
        assert 1 <= len(rec["batch_indices"]) <= q


def test_converges_on_sphere():
    d = 10
    f0 = sphere_problem(d)(np.full(d, 0.8))[0]
    result = mls_sbo_minimize(
        sphere_problem(d), np.full(d, 0.8), np.zeros(d), np.ones(d),
        MLSSBOConfig(max_evals=120, seed=1),
    )
    assert result.f_opt < 1e-2 * f0
    assert result.is_feasible


def test_converges_on_constrained_problem():
    result = mls_sbo_minimize(
        constrained_problem(), np.array([0.1, 0.1]), np.zeros(2), np.ones(2),
        MLSSBOConfig(max_evals=80, seed=2),
    )
    assert result.is_feasible
    assert result.constraints[0] <= 1e-6
    assert result.f_opt < 0.65          # true optimum: 0.5 at (0.5, 0.5)


def test_tractable_at_topology_optimization_dimension():
    """Linear-basis MLS must stay cheap at ~100 variables."""
    d = 100
    rng = np.random.default_rng(5)
    B = rng.standard_normal((d, 6)) / np.sqrt(d)

    def evaluate(x):
        t = B.T @ (x - 0.5)
        return float(t @ t), 2.0 * B @ t, np.zeros(0), np.zeros((0, d))

    x0 = np.full(d, 0.85)
    f0 = evaluate(x0)[0]
    result = mls_sbo_minimize(
        evaluate, x0, np.zeros(d), np.ones(d),
        MLSSBOConfig(max_evals=60, seed=3),
    )
    assert result.n_evals <= 60
    assert result.f_opt < 0.2 * f0


def test_sequential_mode_spends_budget_and_converges():
    """batch_size=1 (the MMA-fair regime) must not collapse prematurely: the
    driver either reaches tight tolerance or uses most of the budget."""
    d = 10
    f0 = sphere_problem(d)(np.full(d, 0.8))[0]
    records = []
    result = mls_sbo_minimize(
        sphere_problem(d), np.full(d, 0.8), np.zeros(d), np.ones(d),
        MLSSBOConfig(max_evals=120, batch_size=1, max_outer_iter=200, seed=6),
        on_iteration=records.append,
    )
    assert result.f_opt < 1e-4 * f0 or result.n_evals >= 96
    assert result.f_opt < 1e-3 * f0
    assert any(r["subproblem"] == "slsqp" for r in records)


def test_sequential_mode_constrained():
    result = mls_sbo_minimize(
        constrained_problem(), np.array([0.1, 0.1]), np.zeros(2), np.ones(2),
        MLSSBOConfig(max_evals=80, batch_size=1, max_outer_iter=120, seed=7),
    )
    assert result.is_feasible
    assert result.constraints[0] <= 1e-6
    assert result.f_opt < 0.6


def test_resets_keep_spending_budget_on_flat_function():
    """A flat function rejects every step; the driver must reset the trust
    region instead of quitting, and terminate cleanly."""
    d = 3

    def flat(x):
        return 1.0, np.zeros(d), np.zeros(0), np.zeros((0, d))

    records = []
    result = mls_sbo_minimize(
        flat, np.full(d, 0.5), np.zeros(d), np.ones(d),
        MLSSBOConfig(max_evals=60, batch_size=1, max_outer_iter=300,
                     n_resets=3, seed=8),
        on_iteration=records.append,
    )
    assert result.n_evals <= 60
    assert max(r["resets"] for r in records) >= 1
    assert result.status


def test_result_fields_are_consistent():
    d = 5
    result = mls_sbo_minimize(
        sphere_problem(d), np.full(d, 0.6), np.zeros(d), np.ones(d),
        MLSSBOConfig(max_evals=30, seed=4),
    )
    assert result.x_opt.shape == (d,)
    assert np.all(result.x_opt >= 0.0) and np.all(result.x_opt <= 1.0)
    assert result.n_iter == len(result.history)
    assert result.status
    r = result.x_opt - 0.3
    assert np.isclose(result.f_opt, float(r @ r), rtol=1e-10)
