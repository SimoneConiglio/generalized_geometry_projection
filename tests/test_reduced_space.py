# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the reduced-space (2D) optimizers: GEK2D and TRANSFORMER_2D.

The GEK2D tests are pure NumPy/SciPy; transformer tests require JAX. The key
genericity checks: the transformer is trained on generic families only, then
evaluated on the **held-out toy-SIMP family** (the compliance-like structure
it has never seen) and across problem dimensions with the same weights.
"""
import numpy as np
import pytest

from scp_uno.mma_teacher import TeacherTask
from scp_uno.reduced_space import (
    ReducedSpaceConfig,
    build_frame,
    gek2d_minimize,
    ks_aggregate,
)


def wrap(task):
    def evaluate(x):
        f, g, c, gc = task.evaluate(x)
        if c is None:
            return f, g, np.zeros(0), np.zeros((0, len(x)))
        return f, g, np.array([c]), gc[None, :]
    return evaluate


def sphere_problem(d, xstar=0.3):
    target = np.full(d, xstar)

    def evaluate(x):
        r = x - target
        return float(r @ r), 2.0 * r, np.zeros(0), np.zeros((0, d))

    return evaluate


# --------------------------------------------------------------------------- #
# Frame and aggregation
# --------------------------------------------------------------------------- #
def test_frame_is_orthonormal_and_descent_aligned():
    rng = np.random.default_rng(0)
    g = rng.standard_normal(20)
    gc = rng.standard_normal(20)
    e1, e2 = build_frame(g, gc, None, rng)
    assert np.isclose(np.linalg.norm(e1), 1.0)
    assert np.isclose(np.linalg.norm(e2), 1.0)
    assert abs(e1 @ e2) < 1e-12
    assert np.isclose(e1 @ g, -np.linalg.norm(g))     # steepest descent


def test_frame_fallbacks_when_degenerate():
    rng = np.random.default_rng(1)
    g = rng.standard_normal(10)
    # constraint gradient parallel to g: falls back to prev step, then random
    e1, e2 = build_frame(g, 3.0 * g, None, rng)
    assert abs(e1 @ e2) < 1e-10
    prev = rng.standard_normal(10)
    e1b, e2b = build_frame(g, None, prev, rng)
    assert abs(e1b @ e2b) < 1e-10


def test_ks_aggregate_matches_single_and_bounds_max():
    J = np.array([[1.0, 0.0], [0.0, 2.0]])
    c = np.array([-0.5, 0.2])
    c1, g1 = ks_aggregate(c[:1], J[:1])
    assert c1 == -0.5 and np.allclose(g1, J[0])
    cagg, gagg = ks_aggregate(c, J, rho=100.0)
    assert cagg >= np.max(c) - 1e-12
    assert cagg <= np.max(c) + np.log(2) / 100.0 + 1e-12
    assert gagg.shape == (2,)


# --------------------------------------------------------------------------- #
# GEK2D
# --------------------------------------------------------------------------- #
def test_gek2d_unconstrained_quadratic_converges():
    task = TeacherTask("quad_free", 32, np.random.default_rng(2))
    x0 = np.random.default_rng(3).uniform(0.1, 0.9, task.d)
    f0 = task.evaluate(x0)[0]
    res = gek2d_minimize(wrap(task), x0, np.zeros(task.d), np.ones(task.d),
                         ReducedSpaceConfig(max_evals=200, seed=0))
    # low-rank quadratics can be very ill-conditioned; 2 % of f0 is already a
    # meaningful reduction for 200 evaluations of a 2D-plane method
    assert res.f_opt < 0.02 * f0


def test_gek2d_constrained_feasible_near_optimum():
    task = TeacherTask("quad_lin", 48, np.random.default_rng(4))
    x0 = np.random.default_rng(5).uniform(0.1, 0.9, task.d)
    res = gek2d_minimize(wrap(task), x0, np.zeros(task.d), np.ones(task.d),
                         ReducedSpaceConfig(max_evals=200, seed=0))
    f_star = task.evaluate(task.x_star)[0]
    f0 = task.evaluate(x0)[0]
    assert res.is_feasible
    assert (res.f_opt - f_star) <= 0.05 * abs(f0 - f_star)


def test_gek2d_toy_simp_feasible():
    task = TeacherTask("toy_simp", 64, np.random.default_rng(6))
    x0 = np.random.default_rng(7).uniform(0.1, 0.9, task.d)
    res = gek2d_minimize(wrap(task), x0, np.zeros(task.d), np.ones(task.d),
                         ReducedSpaceConfig(max_evals=200, seed=1))
    f_star = task.evaluate(task.x_star)[0]
    assert res.is_feasible
    assert res.f_opt <= 1.3 * f_star


def test_gek2d_respects_budget():
    res = gek2d_minimize(sphere_problem(8), np.full(8, 0.9), np.zeros(8),
                         np.ones(8), ReducedSpaceConfig(max_evals=37))
    assert res.n_evals <= 37


# --------------------------------------------------------------------------- #
# TRANSFORMER_2D (requires JAX)
# --------------------------------------------------------------------------- #
jax = pytest.importorskip("jax")

from scp_uno.rs_transformer import (  # noqa: E402
    RSPolicyConfig,
    forward,
    generate_training_batch,
    init_params,
    load_params,
    rs_transformer_minimize,
    save_params,
    train_policy,
)

TINY = RSPolicyConfig(context=8, d_model=32, n_layers=1, n_attn_heads=2,
                      d_mlp=64)


@pytest.fixture(scope="module")
def tiny_trained_policy():
    losses = []
    params = train_policy(TINY, steps=250, batch_states=48, seed=0,
                          on_step=lambda i, l: losses.append(l))
    return params, losses


def test_rs_forward_shapes_and_bounds():
    rng = np.random.default_rng(0)
    tokens, masks, targets = generate_training_batch(rng, TINY, n_states=4)
    assert tokens.shape[1:] == (TINY.context, TINY.n_features)
    assert targets.shape[1:] == (2,)
    assert np.all(np.abs(targets) <= 1.0)
    params = {k: jax.numpy.asarray(v) for k, v in init_params(TINY).items()}
    out = np.asarray(forward(params, jax.numpy.asarray(tokens[0]),
                             jax.numpy.asarray(masks[0]), TINY))
    assert out.shape == (2,)
    assert np.all(np.abs(out) <= 1.0)


def test_rs_save_load_roundtrip(tmp_path):
    params = init_params(TINY, seed=1)
    path = tmp_path / "rs.npz"
    save_params(path, params, TINY)
    loaded, cfg = load_params(path)
    assert cfg == TINY
    for k, v in params.items():
        assert np.allclose(loaded[k], v)


def test_rs_training_reduces_loss(tiny_trained_policy):
    _, losses = tiny_trained_policy
    assert np.mean(losses[-10:]) < 0.6 * np.mean(losses[:10])


def test_rs_transformer_dimension_agnostic_same_weights(tiny_trained_policy):
    """One model, 3 and 300 variables: the reduced space hides the dimension."""
    params, _ = tiny_trained_policy
    for d in (3, 300):
        f0 = sphere_problem(d)(np.full(d, 0.8))[0]
        res = rs_transformer_minimize(
            sphere_problem(d), np.full(d, 0.8), np.zeros(d), np.ones(d),
            params, TINY, ReducedSpaceConfig(max_evals=100, seed=0),
        )
        assert res.f_opt < 1e-3 * f0, f"insufficient progress in d={d}"


def test_rs_transformer_heldout_family_feasible(tiny_trained_policy):
    """toy-SIMP is excluded from training: starting feasible, the zero-shot
    policy must stay feasible while improving the objective."""
    params, _ = tiny_trained_policy
    task = TeacherTask("toy_simp", 48, np.random.default_rng(8))
    x0 = 0.8 * task.x_star            # strictly feasible start (v.x0 = 0.8 V)
    f0 = task.evaluate(x0)[0]
    res = rs_transformer_minimize(
        wrap(task), x0, np.zeros(task.d), np.ones(task.d), params, TINY,
        ReducedSpaceConfig(max_evals=150, seed=0),
    )
    assert res.is_feasible
    assert res.f_opt < f0


def test_rs_transformer_one_eval_per_iteration(tiny_trained_policy):
    params, _ = tiny_trained_policy
    records = []
    res = rs_transformer_minimize(
        sphere_problem(6), np.full(6, 0.7), np.zeros(6), np.ones(6),
        params, TINY, ReducedSpaceConfig(max_evals=40, seed=0),
        on_iteration=records.append,
    )
    # zero inner cost: evaluations = iterations + the initial point
    assert res.n_evals == len(records) + 1


def test_packaged_rs_weights_and_gemseo():
    gemseo = pytest.importorskip("gemseo")
    from scp_uno.reduced_space_gemseo import RS_DEFAULT_WEIGHTS, Transformer2D

    if not RS_DEFAULT_WEIGHTS.exists():
        pytest.skip("reduced-space default weights not trained yet")

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdo_functions.mdo_function import MDOFunction

    d = 7
    design_space = DesignSpace()
    design_space.add_variable("x", d, lower_bound=np.zeros(d),
                              upper_bound=np.ones(d), value=np.full(d, 0.8))
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(
        lambda x: np.array([float(np.sum((x - 0.3) ** 2))]),
        "obj", jac=lambda x: 2.0 * (x - 0.3),
    )
    result = Transformer2D().execute(problem, max_iter=80, seed=0)
    assert float(np.sum((result.x_opt - 0.3) ** 2)) < 1e-3


def test_gek2d_gemseo_runs():
    gemseo = pytest.importorskip("gemseo")
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdo_functions.mdo_function import MDOFunction

    from scp_uno.reduced_space_gemseo import GEK2D

    d = 6
    design_space = DesignSpace()
    design_space.add_variable("x", d, lower_bound=np.zeros(d),
                              upper_bound=np.ones(d), value=np.full(d, 0.8))
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(
        lambda x: np.array([float(np.sum((x - 0.3) ** 2))]),
        "obj", jac=lambda x: 2.0 * (x - 0.3),
    )
    problem.add_constraint(
        MDOFunction(lambda x: np.array([float(np.sum(x) - 0.5 * d)]),
                    "con", jac=lambda x: np.ones((1, d))),
        constraint_type="ineq",
    )
    result = GEK2D().execute(problem, max_iter=120, seed=0)
    assert float(np.sum(result.x_opt) - 0.5 * d) <= 1e-4