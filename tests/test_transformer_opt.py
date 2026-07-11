# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the transformer learned optimizer (requires JAX; GEMSEO optional).

The training-dependent tests train a *tiny* policy for a few hundred steps —
enough to validate the learning signal and the rollout loop without turning
the test suite into a training job. The packaged default weights are covered
by ``test_packaged_default_weights_optimize`` when present.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from scp_uno.transformer_opt_core import (  # noqa: E402  (after importorskip)
    PolicyConfig,
    TransformerOptConfig,
    build_tokens,
    forward,
    generate_training_batch,
    init_params,
    latent_basis,
    load_params,
    save_params,
    train_policy,
    transformer_opt_minimize,
)

TINY = PolicyConfig(latent_dim=8, context=8, n_heads_out=3,
                    d_model=32, n_layers=1, n_attn_heads=2, d_mlp=64)


@pytest.fixture(scope="module")
def tiny_trained_policy():
    """A tiny policy trained for a short but meaningful number of steps."""
    losses = []
    params = train_policy(TINY, steps=250, seed=0,
                          on_step=lambda i, l: losses.append(l))
    return params, losses


def sphere_problem(d, xstar=0.3):
    target = np.full(d, xstar)

    def evaluate(x):
        r = x - target
        return float(r @ r), 2.0 * r, np.zeros(0), np.zeros((0, d))

    return evaluate


# --------------------------------------------------------------------------- #
# Model mechanics
# --------------------------------------------------------------------------- #
def test_forward_shapes_and_bounds():
    rng = np.random.default_rng(0)
    tokens, masks, _ = generate_training_batch(rng, TINY, n_tasks=1, rollout_len=3)
    params = {k: jax.numpy.asarray(v) for k, v in init_params(TINY).items()}
    out = np.asarray(forward(params, jax.numpy.asarray(tokens[0]),
                             jax.numpy.asarray(masks[0]), TINY))
    assert out.shape == (TINY.n_heads_out, TINY.latent_dim)
    assert np.all(np.abs(out) <= 1.0)


def test_policy_is_permutation_invariant_over_history():
    rng = np.random.default_rng(1)
    tokens, masks, _ = generate_training_batch(rng, TINY, n_tasks=1, rollout_len=6)
    t, m = tokens[-1], masks[-1]
    params = {k: jax.numpy.asarray(v) for k, v in init_params(TINY).items()}
    out = np.asarray(forward(params, jax.numpy.asarray(t),
                             jax.numpy.asarray(m), TINY))
    perm = np.random.default_rng(2).permutation(len(t))
    out_p = np.asarray(forward(params, jax.numpy.asarray(t[perm]),
                               jax.numpy.asarray(m[perm]), TINY))
    assert np.allclose(out, out_p, atol=1e-5)


def test_latent_basis_pads_small_dimensions():
    rng = np.random.default_rng(3)
    W = latent_basis(rng.standard_normal((4, 5)), d=5, r=8, rng=rng)
    assert W.shape == (5, 8)
    assert np.allclose(W[:, :5], np.eye(5))
    assert np.allclose(W[:, 5:], 0.0)


def test_build_tokens_is_scale_invariant():
    """Multiplying the objective by a constant must not change the encoding."""
    rng = np.random.default_rng(4)
    d, n = 6, 7
    X = rng.random((n, d))
    F = rng.random(n) * 5
    G = rng.standard_normal((n, d))
    V = np.full(n, -1.0)
    W = latent_basis(G, d, TINY.latent_dim, rng)
    t1, m1 = build_tokens(X, F, G, V, n - 1, W, 0.2, TINY)
    t2, m2 = build_tokens(X, 100.0 * F, 100.0 * G, V, n - 1, W, 0.2, TINY)
    assert np.array_equal(m1, m2)
    assert np.allclose(t1, t2, atol=1e-10)


def test_save_load_roundtrip(tmp_path):
    params = init_params(TINY, seed=5)
    path = tmp_path / "policy.npz"
    save_params(path, params, TINY)
    loaded, cfg = load_params(path)
    assert cfg == TINY
    for k, v in params.items():
        assert np.allclose(loaded[k], v)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def test_training_reduces_imitation_loss(tiny_trained_policy):
    _, losses = tiny_trained_policy
    start = np.mean(losses[:10])
    end = np.mean(losses[-10:])
    assert end < 0.7 * start


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #
def test_trained_policy_optimizes_sphere(tiny_trained_policy):
    params, _ = tiny_trained_policy
    d = 10
    f0 = sphere_problem(d)(np.full(d, 0.85))[0]
    res = transformer_opt_minimize(
        sphere_problem(d), np.full(d, 0.85), np.zeros(d), np.ones(d),
        params, TINY, TransformerOptConfig(max_evals=90, seed=1),
    )
    assert res.f_opt < 0.05 * f0
    assert res.n_evals <= 90


def test_batches_are_multi_point_and_budget_respected(tiny_trained_policy):
    params, _ = tiny_trained_policy
    d = 5
    records = []
    res = transformer_opt_minimize(
        sphere_problem(d), np.full(d, 0.7), np.zeros(d), np.ones(d),
        params, TINY,
        TransformerOptConfig(max_evals=25, seed=2),
        on_iteration=records.append,
    )
    assert res.n_evals <= 25
    assert records
    full = [r for r in records if len(r["batch_indices"]) == TINY.n_heads_out]
    assert full, "at least one full multi-point batch expected"


def test_dimension_agnostic_same_weights(tiny_trained_policy):
    """One trained model must run on both d < latent_dim and d >> latent_dim."""
    params, _ = tiny_trained_policy
    for d in (3, 60):
        f0 = sphere_problem(d)(np.full(d, 0.8))[0]
        res = transformer_opt_minimize(
            sphere_problem(d), np.full(d, 0.8), np.zeros(d), np.ones(d),
            params, TINY, TransformerOptConfig(max_evals=80, seed=3),
        )
        assert res.f_opt < 0.5 * f0, f"no progress in d={d}"


def test_constrained_problem_feasible(tiny_trained_policy):
    params, _ = tiny_trained_policy

    def evaluate(x):
        f = (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2
        g = np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 1.0)])
        return float(f), g, np.array([x[0] + x[1] - 1.0]), np.array([[1.0, 1.0]])

    res = transformer_opt_minimize(
        evaluate, np.array([0.1, 0.1]), np.zeros(2), np.ones(2),
        params, TINY, TransformerOptConfig(max_evals=60, seed=4),
    )
    assert res.is_feasible
    assert res.f_opt < 1.62  # merit-guided: below f(x0)=1.62, ideally near 0.5


# --------------------------------------------------------------------------- #
# Packaged weights + GEMSEO integration
# --------------------------------------------------------------------------- #
def test_packaged_default_weights_optimize():
    from scp_uno.transformer_opt import DEFAULT_WEIGHTS

    if not DEFAULT_WEIGHTS.exists():
        pytest.skip("default weights not trained yet")
    params, cfg = load_params(DEFAULT_WEIGHTS)
    d = 20
    f0 = sphere_problem(d)(np.full(d, 0.8))[0]
    res = transformer_opt_minimize(
        sphere_problem(d), np.full(d, 0.8), np.zeros(d), np.ones(d),
        params, cfg, TransformerOptConfig(max_evals=120, seed=0),
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
    result = TransformerOpt().execute(problem, max_iter=60, seed=0)
    assert float(np.sum((result.x_opt - 0.3) ** 2)) < 0.05
