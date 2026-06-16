# Copyright (c) 2026 Charlie Vanaret
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Uno optimization library wrapper for GEMSEO."""

from __future__ import annotations

import logging
import sys
import os
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.optimization_result import OptimizationResult

from scp_uno.settings import UnoSettings

# Ensure unopy can be imported
sys.path.append(os.path.join(os.getcwd(), "Uno/build"))
try:
    import unopy
except ImportError:
    pass

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem

LOGGER = logging.getLogger(__name__)

class UnoOpt(BaseOptimizationLibrary[UnoSettings]):
    """GEMSEO wrapper for the Uno optimization solver."""

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "UNO": OptimizationAlgorithmDescription(
            algorithm_name="UNO",
            internal_algorithm_name="UNO",
            library_name="UNO",
            description="The Uno solver for nonlinearly constrained optimization",
            Settings=UnoSettings,
            require_gradient=True,
            handle_inequality_constraints=True,
            handle_equality_constraints=True,
        )
    }

    def __init__(self):
        super().__init__("UNO")

    def _run(
        self,
        problem: OptimizationProblem,
        **options: Any,
    ) -> tuple[str, int]:
        """Run the Uno solver on a GEMSEO problem."""
        
        preset = self._settings.preset
        solver_name = self._settings.solver
        max_iter = self._settings.max_iter
        hessian = self._settings.hessian

        x0 = problem.design_space.get_current_value()
        lb = problem.design_space.get_lower_bounds()
        ub = problem.design_space.get_upper_bounds()
        num_vars = len(x0)

        def obj_func(x):
            try:
                res = problem.objective.evaluate(np.array(x))
                if isinstance(res, (list, np.ndarray)):
                    res = float(res[0])
                else:
                    res = float(res)
                if np.isnan(res) or np.isinf(res): return 1e10
                return res
            except Exception:
                return 1e10

        def obj_grad(x, out):
            try:
                grad = problem.objective.jac(np.array(x))
                out[:] = grad.flatten()
            except Exception:
                out[:] = 0.0

        num_cons = 0
        con_lbs, con_ubs = [], []
        for constraint in problem.constraints:
            val = constraint.evaluate(x0)
            size = val.size
            num_cons += size
            if constraint.f_type == "ineq":
                con_lbs.extend([-1e10] * size)
                con_ubs.extend([0.0] * size)
            else:
                con_lbs.extend([0.0] * size)
                con_ubs.extend([0.0] * size)

        def all_cons_func(x, out):
            idx = 0
            for constraint in problem.constraints:
                val = constraint.evaluate(np.array(x))
                if isinstance(val, (list, np.ndarray)): val = np.array(val).flatten()
                else: val = np.array([val])
                out[idx : idx + len(val)] = val.tolist()
                idx += len(val)

        def all_jac_func(x, out):
            idx_row = 0
            for constraint in problem.constraints:
                jac = constraint.jac(np.array(x))
                if jac.ndim == 1: jac = jac.reshape(1, -1)
                size_con = jac.shape[0]
                out[idx_row * num_vars : (idx_row + size_con) * num_vars] = jac.flatten()
                idx_row += size_con

        model = unopy.Model("GEMSEO_SUBPROBLEM", num_vars, unopy.ZERO_BASED_INDEXING)
        model.set_variables_lower_bounds(lb.tolist())
        model.set_variables_upper_bounds(ub.tolist())
        model.set_objective(unopy.MINIMIZE if problem.minimize_objective else unopy.MAXIMIZE, obj_func, obj_grad)
        if num_cons > 0:
            nnz = num_cons * num_vars
            row_indices = np.repeat(np.arange(num_cons), num_vars).tolist()
            col_indices = np.tile(np.arange(num_vars), num_cons).tolist()
            model.set_constraints(num_cons, all_cons_func, con_lbs, con_ubs, nnz, row_indices, col_indices, all_jac_func)
        model.set_initial_primal_iterate(x0.tolist())

        uno_solver = unopy.UnoSolver()
        uno_solver.set_preset(preset)
        uno_solver.set_option("subproblem_solver", solver_name)
        uno_solver.set_option("hessian_model", hessian)
        uno_solver.set_option("max_iterations", max_iter)
        uno_solver.set_option("logger", self._settings.logger)
        result_unopy = uno_solver.optimize(model)
        
        x_opt = np.clip(np.array(result_unopy.primal_solution)[:num_vars], lb, ub)
        LOGGER.info(f"   Inner Uno: status={result_unopy.optimization_status}, f_opt={result_unopy.solution_objective:.4e}")
        problem.design_space.set_current_value(x_opt)
        return "Uno finished", int(result_unopy.optimization_status)

    def _get_result(self, problem, message, status):
        from gemseo.algos.optimization_result import OptimizationResult
        x_opt = problem.design_space.get_current_value()
        f_opt = problem.objective.evaluate(x_opt)
        return OptimizationResult(x_0=x_opt, x_opt=x_opt, f_opt=f_opt, optimizer_name=self.algo_name, message=message, status=status, n_obj_call=problem.objective.n_calls, is_feasible=True)
