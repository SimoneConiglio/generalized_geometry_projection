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
    PlanarMLSModel,
    ProductHermiteSurrogate,
    TangentPlaneSurrogate,
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
def _anchored_from_data(X, Y, G, k, h=0.4, intermediate="linear", asy=0.5):
    """Build the anchored model exactly as the driver does, anchored at row k."""
    return AnchoredSeparableQuadratic(X[k], Y[k], G[k], X, G, h,
                                      intermediate=intermediate, asy=asy, Y=Y)


def test_function_values_shape_the_curvature_fit():
    """Hermite fit: samples whose *gradients* carry no curvature information
    but whose *values* do must still bend the model (this is what a
    gradient-residual-only secant misses)."""
    rng = np.random.default_rng(15)
    d, n = 2, 12
    a = 3.0
    X = 0.5 + 0.2 * rng.standard_normal((n, d))
    k = 0
    S = X - X[k]
    g_k = np.array([0.7, -0.4])
    # values from a curved function, gradients all equal to the center's
    Y = (0.0 + S @ g_k + 0.5 * a * np.sum(S * S, axis=1))[:, None]
    G = np.tile(g_k[:, None], (n, 1, 1)).astype(float)
    anchored = AnchoredSeparableQuadratic(X[k], Y[k], G[k], X, G, 0.4, Y=Y)
    # value rows demand q = a; gradient rows demand q = 0. The mixed LSQ
    # must land clearly away from zero.
    assert np.all(anchored.q[:, 0] > 0.2 * a)


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


def test_mma_intermediates_are_consistent_and_anchored():
    """MMA-variable model: exact anchor at the center and FD-consistent
    gradients everywhere inside the asymptotes."""
    rng = np.random.default_rng(13)
    X = 0.5 + 0.15 * rng.standard_normal((8, 3))
    Y = rng.standard_normal((8, 2))
    G = rng.standard_normal((8, 3, 2)) + 0.5   # mixed gradient signs
    k = 3
    anchored = _anchored_from_data(X, Y, G, k, intermediate="mma")
    val, grad = anchored.value_and_slope(X[k])
    assert np.allclose(val, Y[k], atol=1e-10)
    assert np.allclose(grad, G[k], atol=1e-10)
    x0 = X[k] + 0.08 * rng.standard_normal(3)
    _, grad = anchored.value_and_slope(x0)
    eps = 1e-6
    for j in range(3):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += eps
        xm[j] -= eps
        fd = (anchored.value_and_slope(xp)[0]
              - anchored.value_and_slope(xm)[0]) / (2 * eps)
        assert np.allclose(grad[j], fd, rtol=1e-5, atol=1e-7)


def test_mma_intermediates_fit_reciprocal_functions_better():
    """On a compliance-like reciprocal response the MMA-variable model must
    predict better than the plain quadratic in x."""
    rng = np.random.default_rng(14)
    d = 3
    a = np.array([1.0, 2.0, 0.5])

    def f(x):
        return float(np.sum(a / (x + 0.3))), -a / (x + 0.3) ** 2

    X = 0.45 + 0.25 * rng.random((12, d))
    Y = np.array([[f(x)[0]] for x in X])
    G = np.stack([f(x)[1][:, None] for x in X])
    mdl_mma = _anchored_from_data(X, Y, G, 0, intermediate="mma", asy=0.75)
    mdl_lin = _anchored_from_data(X, Y, G, 0, intermediate="linear")
    Xq = 0.45 + 0.25 * rng.random((100, d))
    err_mma = err_lin = 0.0
    for xq in Xq:
        truth = f(xq)[0]
        err_mma += (mdl_mma.value_and_slope(xq)[0][0] - truth) ** 2
        err_lin += (mdl_lin.value_and_slope(xq)[0][0] - truth) ** 2
    assert err_mma < err_lin


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


def test_product_shape_functions_full_hermite_interpolation_any_spacing():
    """Product-form construction: exact interpolation of every sample's
    value AND gradient — including a jammed pair at distance 1e-3
    (cardinality by product zeros holds for ANY spacing and d_max)."""
    rng = np.random.default_rng(30)
    X = np.vstack([rng.random((5, 3)),
                   [0.5 * np.ones(3)], [0.5 * np.ones(3) + 1e-3]])
    Y = rng.standard_normal((7, 2)) * 2.0
    G = rng.standard_normal((7, 3, 2))
    surr = ProductHermiteSurrogate(X, Y, G, h=0.3, support_factor=3.0)
    for i in range(7):
        val, grad = surr.value_and_slope(X[i])
        assert np.allclose(val, Y[i], atol=1e-9)
        assert np.allclose(grad, G[i], atol=1e-7)


def test_oa_milp_finds_nearest_plane_global_minimum():
    """The MILP must return the exact global minimum of the nearest-plane
    model over the box - verified against a dense brute-force grid."""
    from scp_uno.mls_sbo_core import MLSSBOptimizer, OATangentPlanes

    rng = np.random.default_rng(4)
    X = rng.random((5, 2))
    Y = rng.standard_normal((5, 1))
    G = rng.standard_normal((5, 2, 1))
    oa = OATangentPlanes(X, Y, G)

    def fobj(x):
        return 0.0, np.zeros(2), np.zeros(0), np.zeros((0, 2))

    opt = MLSSBOptimizer(fobj, np.zeros(2), np.ones(2),
                         MLSSBOConfig(max_evals=2, seed=0))
    lo, hi = np.zeros(2), np.ones(2)
    x_star, tag = opt._solve_oa_milp(oa, np.full(2, 0.5), lo, hi)
    assert tag == "oa-milp"
    gx, gy = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    brute = float(np.min(oa.values(grid)[:, 0]))
    got = float(oa.value_and_slope(x_star)[0][0])
    assert got <= brute + 1e-6


def test_oa_driver_converges_on_sphere():
    d = 4

    def sphere(x):
        return float(np.sum((x - 0.3) ** 2)), 2 * (x - 0.3), \
            np.zeros(0), np.zeros((0, d))

    result = mls_sbo_minimize(
        sphere, np.full(d, 0.8), np.zeros(d), np.ones(d),
        MLSSBOConfig(max_evals=60, batch_size=1, model="oa", seed=1,
                     max_outer_iter=200),
    )
    assert result.f_best < 1e-2


def test_product_delta_coupling_preserves_interpolation():
    """rho_i = nn_factor * max(d_nn, delta): cardinality is
    radius-independent, so exact Hermite interpolation must survive the
    trust-region coupling."""
    rng = np.random.default_rng(33)
    X = rng.random((6, 3))
    Y = rng.standard_normal((6, 2))
    G = rng.standard_normal((6, 3, 2))
    surr = ProductHermiteSurrogate(X, Y, G, h=0.2, delta=0.3)
    assert np.all(surr.rho >= 2.5 * 0.3 - 1e-12)      # floor active
    for i in range(6):
        val, grad = surr.value_and_slope(X[i])
        assert np.allclose(val, Y[i], atol=1e-9)
        assert np.allclose(grad, G[i], atol=1e-7)


def test_product_shape_functions_analytic_gradient_matches_fd():
    rng = np.random.default_rng(31)
    X = rng.random((8, 4))
    Y = rng.standard_normal((8, 2))
    G = rng.standard_normal((8, 4, 2))
    surr = ProductHermiteSurrogate(X, Y, G, h=0.25, support_factor=3.0)
    x0 = rng.random(4)
    _, grad = surr.value_and_slope(x0)
    eps = 1e-6
    for k in range(4):
        xp, xm = x0.copy(), x0.copy()
        xp[k] += eps
        xm[k] -= eps
        fd = (surr.value_and_slope(xp)[0] - surr.value_and_slope(xm)[0]) / (2 * eps)
        assert np.allclose(grad[k], fd, rtol=1e-5, atol=1e-7)


def test_product_shape_functions_compact_support_and_fallback():
    X = np.array([[0.2, 0.2], [0.3, 0.2]])
    Y = np.array([[1.0], [4.0]])
    G = np.zeros((2, 2, 1))
    surr = ProductHermiteSurrogate(X, Y, G, h=0.03, support_factor=3.0)
    # far outside every support: nearest sample's tangent plane
    val, grad = surr.value_and_slope(np.array([0.95, 0.95]))
    assert np.isclose(val[0], 4.0)
    assert np.allclose(grad[:, 0], 0.0)


def test_loo_selects_a_sensible_support_radius():
    """The gradient-enhanced LOO selector must pick a support factor whose
    TRUE held-out error is close to the best in the grid (likelihood
    analogue, computed blind)."""
    from scp_uno.mls_sbo_core import loo_select_support
    rng = np.random.default_rng(40)
    d = 2

    def f(x):
        return float(np.sin(3 * x[0]) + (x[1] - 0.4) ** 2), \
            np.array([3 * np.cos(3 * x[0]), 2 * (x[1] - 0.4)])

    X = rng.random((12, d))
    Y = np.array([[f(x)[0]] for x in X])
    G = np.stack([f(x)[1][:, None] for x in X])
    h = 0.15
    factors = (1.0, 2.0, 3.0, 4.5, 6.0)
    fac = loo_select_support(X, Y, G, h, factors=factors)
    assert fac in factors
    # true generalization error of each candidate on a probe set
    probes = rng.random((150, d))
    def true_err(fc):
        s = ProductHermiteSurrogate(X, Y, G, h, support_factor=fc)
        return np.sqrt(np.mean([(s.value_and_slope(p)[0][0] - f(p)[0]) ** 2
                                for p in probes]))
    errs = {fc: true_err(fc) for fc in factors}
    assert errs[fac] <= 1.5 * min(errs.values())


def test_product_model_drives_the_sequential_optimizer():
    d = 6
    f0 = sphere_problem(d)(np.full(d, 0.8))[0]
    result = mls_sbo_minimize(
        sphere_problem(d), np.full(d, 0.8), np.zeros(d), np.ones(d),
        MLSSBOConfig(max_evals=100, batch_size=1, max_outer_iter=200,
                     model="product", seed=32),
    )
    assert result.f_opt < 0.05 * f0


def test_wendland_weights_are_cardinal_with_bounded_gradients():
    """Separation-aware Wendland bumps: exact value+gradient interpolation
    at every node with SMOOTH bounded weights (no singularity) — including
    heterogeneous spacing, where global-scale Shepard degenerates."""
    rng = np.random.default_rng(26)
    X = np.vstack([rng.random((6, 3)),
                   [[0.05, 0.05, 0.05]], [[0.9, 0.9, 0.9]]])  # uneven spread
    Y = rng.standard_normal((8, 2)) * 2.0
    G = rng.standard_normal((8, 3, 2))
    surr = TangentPlaneSurrogate(X, Y, G, h=0.4, weighting="wendland")
    for i in range(8):
        val, grad = surr.value_and_slope(X[i])
        assert np.allclose(val, Y[i], atol=1e-10)
        assert np.allclose(grad, G[i], atol=1e-8)
    # bounded weight gradients at a generic point (no pole anywhere)
    xq = np.array([0.4, 0.5, 0.6])
    _, grad = surr.value_and_slope(xq)
    assert np.all(np.isfinite(grad))


def test_thinning_enforces_separation_and_keeps_priorities():
    d = 3

    def flat(x):
        return 1.0, np.zeros(d), np.zeros(0), np.zeros((0, d))

    opt = MLSSBOptimizer(flat, np.zeros(d), np.ones(d), MLSSBOConfig(seed=0))
    pts = [np.full(d, 0.5), np.full(d, 0.5) + 1e-4,          # jammed pair
           np.full(d, 0.6), np.full(d, 0.8)]
    for p in pts:
        opt._eval(p)
    keep = np.arange(4)
    sel = opt._thin(keep, np.full(d, 0.5), sep=0.05, center_i=0)
    assert 0 in sel and 1 not in sel                          # cluster collapsed
    assert 2 in sel and 3 in sel
    for a in range(len(sel)):
        for b in range(a + 1, len(sel)):
            assert np.linalg.norm(opt.X[sel[a]] - opt.X[sel[b]]) >= 0.05


@pytest.mark.parametrize("weighting", ["softmax", "shepard", "wendland"])
def test_tangent_surrogate_gradient_is_exact(weighting):
    """The analytic gradient of the tangent-plane blend must equal the
    finite-difference derivative of its value — the surrogate is handed
    to SLSQP with moving weights, so this consistency is load-bearing."""
    rng = np.random.default_rng(20)
    X = rng.random((9, 4))
    Y = rng.standard_normal((9, 2)) * 3.0
    G = rng.standard_normal((9, 4, 2))
    surr = TangentPlaneSurrogate(X, Y, G, h=0.3, weighting=weighting)
    x0 = rng.random(4)
    _, grad = surr.value_and_slope(x0)
    eps = 1e-6
    for k in range(4):
        xp, xm = x0.copy(), x0.copy()
        xp[k] += eps
        xm[k] -= eps
        fd = (surr.value_and_slope(xp)[0] - surr.value_and_slope(xm)[0]) / (2 * eps)
        assert np.allclose(grad[k], fd, rtol=1e-5, atol=1e-8)


def test_shepard_weights_interpolate_values_and_gradients_at_finite_h():
    """Hermite–Shepard cardinal weights: the blend interpolates every
    sample's value AND gradient exactly at FINITE length scale (the sheet's
    full constraint set: alpha_i(x_k)=delta_ik, beta_ij(x_k)=0,
    grad alpha=0 and grad beta_ij(x_k)=delta_ik e_j at all nodes)."""
    rng = np.random.default_rng(25)
    X = rng.random((7, 3))
    Y = rng.standard_normal((7, 2)) * 2.0
    G = rng.standard_normal((7, 3, 2))
    surr = TangentPlaneSurrogate(X, Y, G, h=0.4, weighting="shepard")
    for i in range(7):
        val, grad = surr.value_and_slope(X[i])
        assert np.allclose(val, Y[i], atol=1e-9)
        assert np.allclose(grad, G[i], atol=1e-7)


def test_shepard_weights_have_compact_support_and_far_field_guard():
    """alpha_i = 0 beyond the support radius; where NO sample is in range,
    the nearest sample's plane takes over (minimal support extension)."""
    X = np.array([[0.1, 0.1], [0.2, 0.1]])
    Y = np.array([[1.0], [5.0]])
    G = np.zeros((2, 2, 1))
    surr = TangentPlaneSurrogate(X, Y, G, h=0.05, weighting="shepard",
                                 support_factor=3.0)
    # far query: outside both supports -> nearest plane (sample 2)
    val, _ = surr.value_and_slope(np.array([0.9, 0.9]))
    assert np.isclose(val[0], 5.0)


@pytest.mark.parametrize("h", [0.05, 0.5, 3.0])
def test_tangent_surrogate_exact_on_affine(h):
    """All tangent planes of an affine function coincide, so the blend is
    exact for any length scale (value and gradient)."""
    rng = np.random.default_rng(21)
    a, b = 1.3, np.array([0.5, -2.0, 1.0])
    X = rng.random((8, 3))
    Y = (a + X @ b)[:, None]
    G = np.tile(b[:, None], (8, 1, 1)).astype(float)
    surr = TangentPlaneSurrogate(X, Y, G, h=h)
    xq = rng.random(3)
    val, grad = surr.value_and_slope(xq)
    assert np.isclose(val[0], a + xq @ b, atol=1e-9)
    assert np.allclose(grad[:, 0], b, atol=1e-9)


def test_tangent_surrogate_interpolates_as_h_shrinks():
    """h -> 0: the softmax becomes a nearest-plane indicator, so the blend
    interpolates each sample's value AND gradient."""
    rng = np.random.default_rng(22)
    X = rng.random((7, 2))
    Y = rng.standard_normal((7, 1)) * 2.0
    G = rng.standard_normal((7, 2, 1))
    surr = TangentPlaneSurrogate(X, Y, G, h=5e-3)
    for i in range(7):
        val, grad = surr.value_and_slope(X[i])
        assert np.isclose(val[0], Y[i, 0], atol=1e-8)
        assert np.allclose(grad[:, 0], G[i, :, 0], atol=1e-6)


def test_global_subproblem_phase_finds_the_better_valley():
    """The tangent blend is multimodal; the LHS global phase must find a
    subproblem candidate at least as good as the SQP-only solve."""
    rng = np.random.default_rng(24)
    d = 3
    # two 'basins': planes tilting toward two different corners
    X = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])
    Y = np.array([[0.0], [-1.0]])
    G = np.stack([np.full((d, 1), 2.0), np.full((d, 1), -2.0)])

    def evaluate(x):
        return 1.0, np.zeros(d), np.zeros(0), np.zeros((0, d))

    def run(n_global):
        opt = MLSSBOptimizer(evaluate, np.zeros(d), np.ones(d),
                             MLSSBOConfig(n_global=n_global, seed=24))
        opt._eval(np.full(d, 0.2))
        surr = TangentPlaneSurrogate(X, Y, G, h=0.15)
        cand, _ = opt._solve_subproblem(
            surr, np.full(d, 0.2), np.full(d, 0.2),
            np.zeros(d), np.ones(d))
        return float(surr.value_and_slope(cand)[0][0])

    assert run(256) <= run(0) + 1e-9


def test_tangent_model_drives_the_sequential_optimizer():
    d = 6
    f0 = sphere_problem(d)(np.full(d, 0.8))[0]
    result = mls_sbo_minimize(
        sphere_problem(d), np.full(d, 0.8), np.zeros(d), np.ones(d),
        MLSSBOConfig(max_evals=100, batch_size=1, max_outer_iter=200,
                     model="tangent", seed=23),
    )
    assert result.f_opt < 0.05 * f0


def test_planar_model_blends_values_and_gradients():
    """The planar Hermite MLS model must be consistent (grad = true slope of
    the plane) and reproduce a linear function exactly."""
    rng = np.random.default_rng(16)
    a, b = 0.7, np.array([1.0, -2.0, 0.5])
    X = rng.random((10, 3))
    Y = (a + X @ b)[:, None]
    G = np.tile(b[:, None], (10, 1, 1)).astype(float)
    mls = MovingLeastSquares(0.3).fit(X, Y, G)
    planar = PlanarMLSModel(mls, X[4])
    xq = rng.random(3)
    val, grad = planar.value_and_slope(xq)
    assert np.isclose(val[0], a + xq @ b, atol=1e-6)
    assert np.allclose(grad[:, 0], b, atol=1e-6)


def test_planar_model_drives_the_sequential_optimizer():
    d = 6
    f0 = sphere_problem(d)(np.full(d, 0.8))[0]
    result = mls_sbo_minimize(
        sphere_problem(d), np.full(d, 0.8), np.zeros(d), np.ones(d),
        MLSSBOConfig(max_evals=100, batch_size=1, max_outer_iter=200,
                     model="planar", seed=9),
    )
    assert result.f_opt < 0.05 * f0


def test_sequential_mode_constrained():
    result = mls_sbo_minimize(
        constrained_problem(), np.array([0.1, 0.1]), np.zeros(2), np.ones(2),
        MLSSBOConfig(max_evals=80, batch_size=1, max_outer_iter=120, seed=7),
    )
    assert result.is_feasible
    assert result.constraints[0] <= 1e-6
    assert result.f_opt < 0.6


def test_flat_function_spends_budget_sampling_the_region():
    """A flat function never improves; under the hold-region policy the
    driver keeps sampling inside the fixed region (each failure is an
    interpolation point) and terminates cleanly within budget."""
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
    assert result.n_evals >= 40          # budget spent on region samples
    assert result.status
    # classical mode still resets as before
    records2 = []
    result2 = mls_sbo_minimize(
        flat, np.full(d, 0.5), np.zeros(d), np.ones(d),
        MLSSBOConfig(max_evals=60, batch_size=1, max_outer_iter=300,
                     n_resets=3, seed=8, hold_region=False),
        on_iteration=records2.append,
    )
    assert max(r["resets"] for r in records2) >= 1


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
