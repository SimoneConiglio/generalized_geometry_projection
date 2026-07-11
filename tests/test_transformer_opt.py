# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the transformer learned optimizer v2 (per-variable tokens).

Requires JAX (skipped otherwise); the GEMSEO integration test additionally
skips without GEMSEO. Training-dependent tests train a tiny policy for a few
hundred steps — enough to validate the learning signal without turning the
suite into a training job.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from scp_uno.mma_teacher import (  # noqa: E402
    AsymptoteTracker,
    TeacherTask,
    mma_step,
)
from scp_uno.transformer_opt_core import (  # noqa: E402
    PolicyConfig,
    TransformerOptConfig,
    build_var_tokens,
    forward,
    generate_training_batch,
    init_params,
    load_params,
    replay_trajectory_states,
    save_params,
    train_policy,
    transformer_opt_minimize,
)

TINY = PolicyConfig(n_heads_out=4, d_model=32, n_layers=1, n_attn_heads=2,
                    d_mlp=64)


@pytest.fixture(scope="module")
def tiny_trained_policy():
    losses = []
    params = train_policy(TINY, steps=600, dims=(12, 32), batch_states=32,
                          seed=0, on_step=lambda i, l: losses.append(l))
    return params, losses


def sphere_problem(d, xstar=0.3):
    target = np.full(d, xstar)

    def evaluate(x):
        r = x - target
        return float(r @ r), 2.0 * r, np.zeros(0), np.zeros((0, d))

    return evaluate


# --------------------------------------------------------------------------- #
# MMA teacher
# --------------------------------------------------------------------------- #
def test_mma_teacher_solves_toy_simp_exactly():
    task = TeacherTask("toy_simp", 40, np.random.default_rng(0))
    d = task.d
    lb, ub = np.zeros(d), np.ones(d)
    x = np.random.default_rng(1).uniform(0.1, 0.9, d)
    tracker = AsymptoteTracker(lb, ub, asy_init=0.1, asy_min=1e-3, asy_max=0.2)
    tracker.update(x)
    for _ in range(300):
        f, g, c, gc = task.evaluate(x)
        x = np.clip(x + mma_step(x, g, lb, ub, tracker.width, 0.08, c, gc), lb, ub)
        tracker.update(x)
    f_end, _, c_end, _ = task.evaluate(x)
    f_star = task.evaluate(task.x_star)[0]
    assert c_end <= 1e-8
    assert abs(f_end - f_star) <= 1e-6 * abs(f_star)


def test_mma_teacher_unconstrained_descends():
    task = TeacherTask("valley", 24, np.random.default_rng(2))
    d = task.d
    lb, ub = np.zeros(d), np.ones(d)
    x = np.random.default_rng(3).uniform(0.2, 0.8, d)
    f0 = task.evaluate(x)[0]
    tracker = AsymptoteTracker(lb, ub, asy_init=0.1, asy_min=1e-3, asy_max=0.2)
    tracker.update(x)
    for _ in range(200):
        f, g, _, _ = task.evaluate(x)
        x = np.clip(x + mma_step(x, g, lb, ub, tracker.width, 0.08), lb, ub)
        tracker.update(x)
    assert task.evaluate(x)[0] < 1e-2 * f0


# --------------------------------------------------------------------------- #
# Model mechanics
# --------------------------------------------------------------------------- #
def test_forward_shapes_and_bounds():
    rng = np.random.default_rng(0)
    tokens, targets = generate_training_batch(rng, TINY, d=17, n_states=3)
    assert tokens.shape[1:] == (17, TINY.n_features)
    assert targets.shape[1:] == (TINY.n_heads_out, 17)
    params = {k: jax.numpy.asarray(v) for k, v in init_params(TINY).items()}
    out = np.asarray(forward(params, jax.numpy.asarray(tokens[0]), TINY))
    assert out.shape == (TINY.n_heads_out, 17)
    assert np.all(np.abs(out) <= 1.0)


def test_policy_is_permutation_equivariant_over_variables():
    """Permuting the design variables must permute the proposed steps."""
    rng = np.random.default_rng(1)
    tokens, _ = generate_training_batch(rng, TINY, d=13, n_states=1)
    t = tokens[0]
    params = {k: jax.numpy.asarray(v) for k, v in init_params(TINY).items()}
    out = np.asarray(forward(params, jax.numpy.asarray(t), TINY))
    perm = np.random.default_rng(2).permutation(len(t))
    out_p = np.asarray(forward(params, jax.numpy.asarray(t[perm]), TINY))
    assert np.allclose(out[:, perm], out_p, atol=1e-5)


def test_tokens_are_scale_invariant():
    """Affine objective rescaling must not change the encoding."""
    rng = np.random.default_rng(4)
    d = 9
    x = rng.random(d)
    g = rng.standard_normal(d)
    gc = rng.uniform(0.5, 1.5, d)
    lb, ub = np.zeros(d), np.ones(d)
    width = np.full(d, 0.005)
    last = rng.uniform(-0.01, 0.01, d)
    osc = np.sign(rng.standard_normal(d))
    t1 = build_var_tokens(x, g, 0.2, gc, width, last, osc, lb, ub, 0.01)
    t2 = build_var_tokens(x, 250.0 * g, 0.2, gc, width, last, osc, lb, ub, 0.01)
    assert np.allclose(t1, t2, atol=1e-12)


def test_save_load_roundtrip(tmp_path):
    params = init_params(TINY, seed=5)
    path = tmp_path / "policy.npz"
    save_params(path, params, TINY)
    loaded, cfg = load_params(path)
    assert cfg == TINY
    for k, v in params.items():
        assert np.allclose(loaded[k], v)


def test_replay_trajectory_states_shapes():
    rng = np.random.default_rng(6)
    n, d = 12, 7
    X = np.cumsum(rng.uniform(-0.005, 0.01, (n, d)), axis=0) + 0.5
    G0 = rng.standard_normal((n, d))
    C = rng.uniform(-0.2, 0.2, n)
    G1 = rng.uniform(0.5, 1.5, (n, d))
    toks, tgts = replay_trajectory_states(X, G0, C, G1, 0.01, TINY)
    assert toks.shape == (n - 1, d, TINY.n_features)
    assert tgts.shape == (n - 1, TINY.n_heads_out, d)
    assert np.all(np.abs(tgts) <= 1.0)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def test_training_reduces_imitation_loss(tiny_trained_policy):
    _, losses = tiny_trained_policy
    assert np.mean(losses[-10:]) < 0.6 * np.mean(losses[:10])


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #
def test_trained_policy_optimizes_sphere(tiny_trained_policy):
    params, _ = tiny_trained_policy
    d = 10
    f0 = sphere_problem(d)(np.full(d, 0.85))[0]
    res = transformer_opt_minimize(
        sphere_problem(d), np.full(d, 0.85), np.zeros(d), np.ones(d),
        params, TINY, TransformerOptConfig(max_evals=120, move=0.05, seed=1),
    )
    assert res.f_opt < 0.05 * f0
    assert res.n_evals <= 120


def test_trained_policy_toy_simp_feasible_near_optimum(tiny_trained_policy):
    """The compliance-like family: must end feasible and near the optimum."""
    params, _ = tiny_trained_policy
    task = TeacherTask("toy_simp", 32, np.random.default_rng(7))
    d = task.d

    def evaluate(x):
        f, g, c, gc = task.evaluate(x)
        return f, g, np.array([c]), gc[None, :]

    x0 = np.random.default_rng(8).uniform(0.1, 0.9, d)
    res = transformer_opt_minimize(
        evaluate, x0, np.zeros(d), np.ones(d), params, TINY,
        TransformerOptConfig(max_evals=150, move=0.05, seed=2),
    )
    f_star = task.evaluate(task.x_star)[0]
    assert res.is_feasible
    assert res.f_opt <= 1.25 * f_star


def test_sequential_mode_moves_every_evaluation(tiny_trained_policy):
    params, _ = tiny_trained_policy
    d = 6
    records = []
    res = transformer_opt_minimize(
        sphere_problem(d), np.full(d, 0.7), np.zeros(d), np.ones(d),
        params, TINY,
        TransformerOptConfig(max_evals=30, move=0.05, eval_heads=1, seed=3),
        on_iteration=records.append,
    )
    assert res.n_evals <= 30
    # sequential mode: each iteration costs at most 1 + n_backtracks evals
    assert len(records) >= (res.n_evals - 1) / 3.0


def test_batch_mode_evaluates_multiple_heads(tiny_trained_policy):
    params, _ = tiny_trained_policy
    d = 6
    records = []
    res = transformer_opt_minimize(
        sphere_problem(d), np.full(d, 0.7), np.zeros(d), np.ones(d),
        params, TINY,
        TransformerOptConfig(max_evals=29, move=0.05, eval_heads=4, seed=4),
        on_iteration=records.append,
    )
    assert res.n_evals <= 29
    assert records
    # 4 evaluations per iteration (plus the initial one and any backtracks)
    assert res.n_evals >= 4 * len(records) - 3


def test_dimension_agnostic_same_weights(tiny_trained_policy):
    params, _ = tiny_trained_policy
    for d in (3, 60):
        f0 = sphere_problem(d)(np.full(d, 0.8))[0]
        res = transformer_opt_minimize(
            sphere_problem(d), np.full(d, 0.8), np.zeros(d), np.ones(d),
            params, TINY, TransformerOptConfig(max_evals=100, move=0.05, seed=5),
        )
        assert res.f_opt < 0.5 * f0, f"no progress in d={d}"


# --------------------------------------------------------------------------- #
# Packaged weights + GEMSEO integration
# --------------------------------------------------------------------------- #
def test_packaged_default_weights_optimize():
    from scp_uno.transformer_opt import DEFAULT_WEIGHTS

    if not DEFAULT_WEIGHTS.exists():
        pytest.skip("default weights not trained yet")
    try:
        params, cfg = load_params(DEFAULT_WEIGHTS)
    except ValueError as exc:
        pytest.skip(str(exc))
    d = 20
    f0 = sphere_problem(d)(np.full(d, 0.8))[0]
    res = transformer_opt_minimize(
        sphere_problem(d), np.full(d, 0.8), np.zeros(d), np.ones(d),
        params, cfg, TransformerOptConfig(max_evals=120, move=0.05, seed=0),
    )
    assert res.f_opt < 1e-2 * f0


def test_gemseo_library_runs():
    gemseo = pytest.importorskip("gemseo")
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdo_functions.mdo_function import MDOFunction

    from scp_uno.transformer_opt import DEFAULT_WEIGHTS, TransformerOpt

    if not DEFAULT_WEIGHTS.exists():
        pytest.skip("default weights not trained yet")
    try:
        load_params(DEFAULT_WEIGHTS)
    except ValueError as exc:
        pytest.skip(str(exc))

    d = 5
    design_space = DesignSpace()
    design_space.add_variable(
        "x", d, lower_bound=np.zeros(d), upper_bound=np.ones(d),
        value=np.full(d, 0.8),
    )
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(
        lambda x: np.array([float(np.sum((x - 0.3) ** 2))]),
        "obj",
        jac=lambda x: 2.0 * (x - 0.3),
    )
    result = TransformerOpt().execute(problem, max_iter=80, move=0.05, seed=0)
    assert float(np.sum((result.x_opt - 0.3) ** 2)) < 0.05
