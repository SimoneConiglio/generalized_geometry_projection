# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests of the UNO GEMSEO library (scp_uno/uno_wrapper.py)."""

import pytest

pytest.importorskip("unopy")

from gemseo import execute_algo  # noqa: E402
from gemseo.problems.optimization.power_2 import Power2  # noqa: E402

F_OPT = 2.192090802


@pytest.mark.parametrize("preset", ["ipopt", "filtersqp", "funnelsqp", "filterslp"])
def test_presets_solve_constrained_problem(preset):
    problem = Power2()
    result = execute_algo(problem, "opt", algo_name="UNO", max_iter=200, preset=preset)
    assert result.is_feasible
    assert result.f_opt == pytest.approx(F_OPT, rel=1e-4)


def test_max_iter_bounds_the_evaluations():
    problem = Power2()
    result = execute_algo(problem, "opt", algo_name="UNO", max_iter=3, preset="ipopt")
    assert len(problem.database) == 3
    assert "Maximum number of iterations" in result.message


def test_failed_evaluation_rejects_the_step():
    problem = Power2()
    objective = problem.objective.func

    def failing_objective(x):
        if x[0] < 0.85:
            raise RuntimeError("Factor is exactly singular")
        return objective(x)

    problem.objective.func = failing_objective
    result = execute_algo(problem, "opt", algo_name="UNO", max_iter=100, preset="filtersqp")
    assert result.is_feasible
    assert result.x_opt[0] >= 0.85
