# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""GEMSEO integration tests for the ``GE_SBO`` algorithm (skipped without GEMSEO)."""
import numpy as np
import pytest

gemseo = pytest.importorskip("gemseo")

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction

from scp_uno.gesbo import GradientEnhancedSBO
from scp_uno.settings import GESBOSettings


def _quadratic_problem(d=5):
    """min ||x - 0.3||^2  s.t.  sum(x) <= 0.5 d  on [0, 1]^d."""
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
    problem.add_constraint(
        MDOFunction(
            lambda x: np.array([float(np.sum(x) - 0.5 * d)]),
            "con",
            jac=lambda x: np.ones((1, d)),
        ),
        constraint_type="ineq",
    )
    return problem


def test_algorithm_is_described():
    infos = GradientEnhancedSBO.ALGORITHM_INFOS["GE_SBO"]
    assert infos.require_gradient
    assert infos.handle_inequality_constraints
    assert infos.Settings is GESBOSettings


def test_ge_sbo_solves_gemseo_problem():
    problem = _quadratic_problem(d=5)
    library = GradientEnhancedSBO("GE_SBO")
    result = library.execute(problem, max_iter=80, batch_size=3, seed=0)
    x_opt = result.x_opt
    assert np.sum((x_opt - 0.3) ** 2) < 0.05
    assert float(np.sum(x_opt)) <= 0.5 * 5 + 1e-4
