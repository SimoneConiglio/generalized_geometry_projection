# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Implementation of the CONLIN approximation in the SCP framework."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction

from scp_uno.scp import SequentialConvexProgramming
from scp_uno.settings import SCPSettings

if TYPE_CHECKING:
    from numpy import ndarray

LOGGER = logging.getLogger(__name__)

class CONLIN(SequentialConvexProgramming):
    """CONLIN implemented in GEMSEO using Uno as the subproblem solver."""

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "CONLIN_UNO": OptimizationAlgorithmDescription(
            algorithm_name="CONLIN_UNO",
            internal_algorithm_name="CONLIN_UNO",
            library_name="CONLIN_UNO",
            description="CONLIN implementation using Uno as the subproblem solver",
            Settings=SCPSettings,
            require_gradient=True,
            handle_inequality_constraints=True,
        )
    }

    def __init__(self):
        super().__init__("CONLIN_UNO")

    def _build_subproblem(
        self,
        original_problem: OptimizationProblem,
        x_k: ndarray,
        f_k: ndarray,
        grad_f_k: ndarray,
        con_k: list[ndarray],
        grad_con_k: list[ndarray]
    ) -> OptimizationProblem:
        """Create the CONLIN approximate subproblem with move limits."""

        n = len(x_k)
        lb_orig = original_problem.design_space.get_lower_bounds()
        ub_orig = original_problem.design_space.get_upper_bounds()
        s = ub_orig - lb_orig

        # Move-limited design space (same pattern as MMA)
        move = self._settings.max_optimization_step
        lb_move = np.maximum(lb_orig, x_k - move * s)
        ub_move = np.minimum(ub_orig, x_k + move * s)
        # Ensure x_k_safe stays positive so reciprocal terms don't blow up
        lb_move = np.maximum(lb_move, 1e-8)

        from gemseo.algos.design_space import DesignSpace
        sub_ds = DesignSpace()
        for var in original_problem.design_space.variable_names:
            size = original_problem.design_space.variable_sizes[var]
            sub_ds.add_variable(var, size, lower_bound=lb_move, upper_bound=ub_move, value=x_k)

        def build_conlin_approx(val_k, grad_k, name):
            x_k_safe = np.maximum(x_k, 1e-6)
            p = np.where(grad_k > 0, grad_k, 0.0)
            q = np.where(grad_k < 0, -grad_k * (x_k_safe ** 2), 0.0)
            c_term = val_k - np.sum(p * x_k) - np.sum(q / x_k_safe)

            def func(x):
                return c_term + np.sum(p * x) + np.sum(q / np.maximum(x, 1e-8))

            def jac(x):
                return (p - q / np.maximum(x, 1e-8) ** 2).reshape(1, -1)

            return MDOFunction(func, name, jac=jac)

        sub_problem = OptimizationProblem(sub_ds)
        sub_problem.objective = build_conlin_approx(f_k[0], grad_f_k.flatten(), "conlin_obj")

        full_grad_con = np.concatenate(grad_con_k, axis=0) if grad_con_k else np.empty((0, n))
        full_con_k = np.concatenate(con_k) if con_k else np.empty(0)

        for i in range(len(full_con_k)):
            sub_problem.add_constraint(
                build_conlin_approx(full_con_k[i], full_grad_con[i], f"conlin_con_{i}"),
                constraint_type="ineq",
            )

        return sub_problem
