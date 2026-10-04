# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""GEMSEO optimization library wrapping the Uno solver (``unopy``, Uno >= 2.x).

Uno is a modular solver for nonlinearly constrained optimization: a *preset*
fixes the combination of ingredients (``ipopt``: primal-dual interior point +
filter line search; ``filtersqp``: SQP + filter trust region; ``funnelsqp``:
SQP + funnel; ``filterslp``: SLP + filter trust region). Since GGP only
provides first derivatives, the Lagrangian Hessian is approximated by the
``hessian_model`` option (``LBFGS`` by default; ``identity`` and ``zero`` are
also available).

Usage (once ``scp_uno`` is registered as a GEMSEO plugin, see pyproject.toml)::

    scenario.execute(algo_name="UNO", max_iter=300, preset="filtersqp",
                     uno_options={"TR_radius": 0.05})

Every function evaluation goes through the GEMSEO database, so ``max_iter``
bounds the number of FE analyses (Uno's trial points included), exactly as for
the other GEMSEO optimizers. Python exceptions raised inside unopy callbacks
are swallowed by Uno, so GEMSEO's stopping exceptions (e.g. max_iter reached)
are captured, Uno is stopped through its termination callback and the
exception is re-raised once Uno has returned.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from gemseo.algos.design_space_utils import get_value_and_bounds
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription

from scp_uno.settings import UnoSettings

try:
    import unopy
except ImportError:  # pragma: no cover - optional dependency
    unopy = None

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem

LOGGER = logging.getLogger(__name__)


class UnoOpt(BaseOptimizationLibrary[UnoSettings]):
    """GEMSEO wrapper for the Uno solver."""

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "UNO": OptimizationAlgorithmDescription(
            algorithm_name="UNO",
            internal_algorithm_name="UNO",
            library_name="UNO",
            description="Uno: unified nonlinear optimization (SQP / interior point)",
            Settings=UnoSettings,
            require_gradient=True,
            handle_inequality_constraints=True,
            handle_equality_constraints=True,
        )
    }

    def __init__(self, algo_name: str = "UNO"):
        super().__init__(algo_name)

    def _run(self, problem: OptimizationProblem, **options: Any) -> tuple[str, int]:
        if unopy is None:
            raise ImportError("The UNO algorithm requires the 'unopy' package.")
        settings = self._settings
        x_0, lb, ub = get_value_and_bounds(
            problem.design_space, settings.normalize_design_space
        )
        n = x_0.size
        constraints = list(problem.constraints)
        sizes = [np.atleast_1d(c.evaluate(x_0)).size for c in constraints]
        m = int(sum(sizes))

        # Python exceptions raised in callbacks do not cross the C++ boundary,
        # so the first one is stored and Uno is told to stop.
        stop: list[BaseException] = []

        def guarded(func, fallback):
            def wrapper(*args):
                if stop:
                    return fallback(*args)
                try:
                    return func(*args)
                except BaseException as error:  # noqa: BLE001
                    stop.append(error)
                    return fallback(*args)

            return wrapper

        # unopy passes views of Uno's internal buffers, which Uno later
        # overwrites: copy them before they reach the GEMSEO database.
        # Uno has no automatic NLP scaling: constant factors are applied here.
        f_scale = settings.objective_scale
        c_scale = settings.constraint_scale

        def objective(x):
            value = problem.objective.evaluate(np.array(x))
            return f_scale * float(np.real(value).ravel()[0])

        def objective_gradient(x, out):
            out[:] = f_scale * np.asarray(problem.objective.jac(np.array(x))).ravel()

        def constraint_values(x, out):
            x = np.array(x)
            out[:] = c_scale * np.concatenate(
                [np.atleast_1d(c.evaluate(x)).ravel() for c in constraints]
            )

        def constraint_jacobian(x, out):
            # dense, row-major (matches the sparsity pattern declared below)
            x = np.array(x)
            out[:] = c_scale * np.concatenate(
                [np.atleast_2d(c.jac(x)).reshape(-1, n).ravel() for c in constraints]
            )

        def no_value(*args):
            return math.inf

        def no_fill(x, out):
            out[:] = 0.0

        model = unopy.Model(unopy.PROBLEM_NONLINEAR, n, unopy.ZERO_BASED_INDEXING)
        model.set_variables_lower_bounds(
            [v if np.isfinite(v) else -math.inf for v in lb]
        )
        model.set_variables_upper_bounds(
            [v if np.isfinite(v) else math.inf for v in ub]
        )
        model.set_objective(
            unopy.MINIMIZE,
            guarded(objective, no_value),
            guarded(objective_gradient, no_fill),
        )
        if m:
            c_lb, c_ub = [], []
            for constraint, size in zip(constraints, sizes):
                c_lb += [0.0 if constraint.f_type == "eq" else -math.inf] * size
                c_ub += [0.0] * size
            model.set_constraints(
                m,
                guarded(constraint_values, no_fill),
                c_lb,
                c_ub,
                m * n,
                np.repeat(np.arange(m), n).tolist(),
                np.tile(np.arange(n), m).tolist(),
                guarded(constraint_jacobian, no_fill),
            )
        model.set_initial_primal_iterate(x_0.tolist())

        solver = unopy.UnoSolver()
        solver.set_preset(settings.preset)
        solver.set_option("hessian_model", settings.hessian_model)
        solver.set_option("quasi_newton_memory_size", settings.quasi_newton_memory_size)
        solver.set_option("max_iterations", settings.max_uno_iterations)
        solver.set_option("logger", settings.logger)
        for key, value in settings.uno_options.items():
            solver.set_option(key, value)
        solver.set_termination_callback(lambda *args: bool(stop))

        result = solver.optimize(model)
        message = (
            f"Uno ({settings.preset}): {result.optimization_status.name}, "
            f"{result.solution_status.name}, {result.number_iterations} iterations"
        )
        LOGGER.info(message)
        if stop:
            raise stop[0]
        return message, int(result.optimization_status.value)
