# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Unit tests for the global-search strategies (improved local minima).

These tests exercise the *orchestration* logic only -- best-of-N selection,
continuation warm-start chaining, Metropolis acceptance, deflation bookkeeping and
the random-restart / deflation maths -- via a lightweight fake pipeline. They do
NOT require FEniCS or GEMSEO, so they run in any CI that has numpy.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ggp.optimization import global_search as gs
from ggp.optimization.results import OptimisationResult


# --------------------------------------------------------------------------- #
# Fake pipeline
# --------------------------------------------------------------------------- #
def _make_spec(mode="Free", num_components=4, num_layers=None, comp_per_layer=None):
    return SimpleNamespace(
        formulation=SimpleNamespace(
            mode=mode,
            num_components=num_components,
            num_layers=num_layers,
            comp_per_layer=comp_per_layer,
        )
    )


class FakePipeline:
    """Drop-in stand-in for GGPPipeline that records its inputs and returns a
    deterministic OptimisationResult from a simple quadratic landscape."""

    calls = []                      # list of recorded call dicts
    num_vars = 24                   # 6 * 4 components

    def __init__(self, spec, x0=None, overrides=None, deflation=None):
        self.spec = spec
        self.x0 = None if x0 is None else np.asarray(x0, float).flatten()
        self.overrides = overrides
        self.deflation = deflation

    def run(self):
        x_in = self.x0 if self.x0 is not None else np.full(self.num_vars, 0.5)
        # "Local optimisation": nudge the design a touch (so warm-start chaining
        # is observable) and evaluate a quadratic compliance well at x=0.5.
        x_opt = np.clip(x_in - 0.01, 0.0, 1.0)
        compliance = 10.0 * float(np.sum((x_opt - 0.5) ** 2)) + 1.0
        FakePipeline.calls.append(
            dict(x0=None if self.x0 is None else self.x0.copy(),
                 x_opt=x_opt.copy(),
                 overrides=self.overrides,
                 deflation=self.deflation)
        )
        return OptimisationResult(
            problem_name="fake",
            algorithm="MMA",
            status="success",
            iterations=10,
            objective_value=float(np.log(compliance + 1.0)),
            max_constraint_violation=0.0,
            design_variables=x_opt,
            history={},
            execution_time_s=0.01,
        )


@pytest.fixture(autouse=True)
def _reset_calls():
    FakePipeline.calls = []
    yield
    FakePipeline.calls = []


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_compliance_of_inverts_log_transform():
    res = OptimisationResult(
        problem_name="x", algorithm="MMA", status="ok", iterations=1,
        objective_value=float(np.log(76.0)), max_constraint_violation=0.0,
        design_variables=np.zeros(6), history={},
    )
    assert gs.compliance_of(res) == pytest.approx(75.0)


def test_compliance_of_handles_nan():
    res = OptimisationResult(
        problem_name="x", algorithm="MMA", status="ok", iterations=1,
        objective_value=float("nan"), max_constraint_violation=0.0,
        design_variables=np.zeros(6), history={},
    )
    assert gs.compliance_of(res) == float("inf")


def test_random_initial_design_free_bounds():
    rng = np.random.default_rng(0)
    x = gs.random_initial_design(24, "Free", rng)
    assert x.shape == (24,)
    assert np.all((x >= 0.0) & (x <= 1.0))
    # thickness slice (index 3 mod 6) must be small (thin bars)
    assert np.all(x[3::6] <= 0.10)
    # Mc slice near 0.5
    assert np.all((x[5::6] >= 0.4) & (x[5::6] <= 0.6))


def test_random_initial_design_3d_and_fallback():
    rng = np.random.default_rng(1)
    x3 = gs.random_initial_design(16, "3D_Free", rng)
    assert x3.shape == (16,) and np.all((x3 >= 0) & (x3 <= 1))
    xf = gs.random_initial_design(7, "ALM", rng)   # odd length -> fallback
    assert xf.shape == (7,) and np.all((xf >= 0.3) & (xf <= 0.7))


def test_perturb_design_clips_and_preserves_length():
    rng = np.random.default_rng(2)
    x = np.array([0.0, 1.0, 0.5, 0.99, 0.01])
    xp = gs.perturb_design(x, step=0.5, rng=rng)
    assert xp.shape == x.shape
    assert np.all((xp >= 0.0) & (xp <= 1.0))


# --------------------------------------------------------------------------- #
# Continuation
# --------------------------------------------------------------------------- #
def test_continuation_warm_start_chaining():
    spec = _make_spec()
    schedule = [{"r_gp": 1.5}, {"r_gp": 1.0}, {"r_gp": 0.5}]
    out = gs.continuation(spec, schedule, pipeline_cls=FakePipeline)

    assert out.method == "continuation"
    assert len(out.attempts) == 3
    # phase 0 starts from default (x0 is None), later phases warm-start from the
    # previous phase's optimum.
    assert FakePipeline.calls[0]["x0"] is None
    for i in range(1, 3):
        np.testing.assert_allclose(
            FakePipeline.calls[i]["x0"], FakePipeline.calls[i - 1]["x_opt"]
        )
    # overrides were threaded through verbatim
    assert FakePipeline.calls[2]["overrides"] == {"r_gp": 0.5}


def test_continuation_requires_schedule():
    with pytest.raises(ValueError):
        gs.continuation(_make_spec(), [], pipeline_cls=FakePipeline)


# --------------------------------------------------------------------------- #
# Multi-start
# --------------------------------------------------------------------------- #
def test_multi_start_best_of_n_consistency():
    spec = _make_spec()
    out = gs.multi_start(spec, n_starts=4, seed=3, pipeline_cls=FakePipeline)
    # default + 4 restarts
    assert len(out.attempts) == 5
    assert out.attempts[0].label == "default"
    # best compliance equals the minimum over all attempts, and best_result agrees
    comps = [a.compliance for a in out.attempts]
    assert out.best_compliance == pytest.approx(min(comps))
    assert gs.compliance_of(out.best_result) == pytest.approx(out.best_compliance)


def test_multi_start_without_default():
    out = gs.multi_start(_make_spec(), n_starts=3, seed=0,
                         include_default=False, pipeline_cls=FakePipeline)
    assert len(out.attempts) == 3
    assert all(a.label.startswith("restart") for a in out.attempts)


# --------------------------------------------------------------------------- #
# Basin hopping
# --------------------------------------------------------------------------- #
def test_basin_hopping_rejects_worse_at_low_temperature():
    spec = _make_spec()
    # Start AT the global minimum (x = 0.5 + 0.01 so that x_opt -> 0.5). Any
    # perturbation strictly worsens compliance; with T -> 0 all hops are rejected.
    x_min = np.full(FakePipeline.num_vars, 0.51)
    out = gs.basin_hopping(spec, n_hops=4, step=0.2, temperature=1e-9,
                           seed=5, x0=x_min, pipeline_cls=FakePipeline)
    assert len(out.attempts) == 5
    # incumbent (hop0) stays best; later hops are rejected
    assert out.attempts[0].label == "hop0"
    assert out.best_compliance == pytest.approx(out.attempts[0].compliance)
    assert all(not a.accepted for a in out.attempts[1:])


def test_basin_hopping_accepts_improvement():
    spec = _make_spec()
    # Start far from the minimum: perturbations can improve, so at least one hop
    # should be accepted under any temperature.
    x_far = np.zeros(FakePipeline.num_vars)
    out = gs.basin_hopping(spec, n_hops=6, step=0.3, temperature=1.0,
                           seed=7, x0=x_far, pipeline_cls=FakePipeline)
    assert any(a.accepted for a in out.attempts[1:])


# --------------------------------------------------------------------------- #
# Deflation
# --------------------------------------------------------------------------- #
def test_deflated_search_accumulates_roots():
    spec = _make_spec()
    out = gs.deflated_search(spec, n_solutions=3, pipeline_cls=FakePipeline)
    assert len(out.attempts) == 3
    # first solve has no deflation; subsequent solves carry a growing root set
    assert FakePipeline.calls[0]["deflation"] is None
    assert len(FakePipeline.calls[1]["deflation"]["roots"]) == 1
    assert len(FakePipeline.calls[2]["deflation"]["roots"]) == 2
    assert FakePipeline.calls[2]["deflation"]["power"] == 2.0


def test_num_vars_matches_modes():
    assert gs._num_vars(_make_spec(mode="Free", num_components=18)) == 108
    assert gs._num_vars(_make_spec(mode="3D_Free", num_components=10)) == 80
    alm = _make_spec(mode="2D_ALM", num_layers=10, comp_per_layer=5)
    assert gs._num_vars(alm) == 2 * 10 * 5 + 2 * 5 + 2


# --------------------------------------------------------------------------- #
# GlobalSearchResult helpers
# --------------------------------------------------------------------------- #
def test_global_search_result_summary_and_best_index():
    out = gs.multi_start(_make_spec(), n_starts=2, seed=0, pipeline_cls=FakePipeline)
    s = out.summary()
    assert "best compliance" in s
    assert out.best_index == out.attempts.index(
        min(out.attempts, key=lambda a: a.compliance)
    )


# --------------------------------------------------------------------------- #
# Deflation discipline maths (requires GEMSEO; skipped otherwise)
# --------------------------------------------------------------------------- #
def test_deflated_objective_gradient_matches_finite_difference():
    pytest.importorskip("gemseo")
    from ggp.optimization.pipeline import _DeflatedObjectiveDiscipline

    rng = np.random.default_rng(0)
    n = 8
    roots = [rng.uniform(0, 1, n), rng.uniform(0, 1, n)]
    disc = _DeflatedObjectiveDiscipline(roots, num_vars=n, shift=1.0, power=2.0)

    x = rng.uniform(0.2, 0.8, n)
    j = 3.0
    m0, grad = disc._factor_and_grad(x)

    # finite-difference check of dM/dx
    eps = 1e-6
    fd = np.zeros(n)
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        fd[i] = (disc._factor_and_grad(xp)[0] - disc._factor_and_grad(xm)[0]) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-6)

    # objective = J * M, d/dJ = M, d/dx = J * dM/dx
    disc._run({"compliance": np.array([j]), "x_vars": x})
    assert disc.local_data["compliance_deflated"][0] == pytest.approx(j * m0)
    disc._compute_jacobian()
    np.testing.assert_allclose(disc.jac["compliance_deflated"]["x_vars"][0], j * grad)
    assert disc.jac["compliance_deflated"]["compliance"][0, 0] == pytest.approx(m0)
