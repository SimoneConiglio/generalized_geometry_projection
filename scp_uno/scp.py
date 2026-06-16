# Copyright (c) 2026 Charlie Vanaret
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Base classes for Sequential Programming and Sequential Convex Programming in GEMSEO."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.optimization_result import OptimizationResult

from scp_uno.uno_wrapper import UnoOpt
from scp_uno.settings import SCPSettings

if TYPE_CHECKING:
    from numpy import ndarray

LOGGER = logging.getLogger(__name__)

class SequentialProgramming(BaseOptimizationLibrary[SCPSettings]):
    """Generic base class for Sequential Programming (SCP, SBO, TR)."""

    def __init__(self, algo_name: str):
        super().__init__(algo_name)

    def _run(
        self,
        problem: OptimizationProblem,
        **options: Any,
    ) -> OptimizationResult:
        """Execute the sequential loop."""
        max_iter = self._settings.max_iter
        xtol_rel = self._settings.xtol_rel
        ftol_rel = self._settings.ftol_rel

        x_k = problem.design_space.get_current_value()
        
        # Outer loop
        for it in range(max_iter):
            # 1. Evaluate current point (Objective and Constraints)
            f_k = problem.objective.evaluate(x_k)
            grad_f_k = problem.objective.jac(x_k)
            
            con_k = []
            grad_con_k = []
            for constraint in problem.constraints:
                con_k.append(constraint.evaluate(x_k).flatten())
                grad_con_k.append(constraint.jac(x_k))
            
            # Log progress
            LOGGER.info(f"Outer Iter {it}: f = {f_k[0]:.4e}, max_con = {np.max(np.concatenate(con_k)) if con_k else 0.0:.4e}")

            # 2. Build the approximation (Subproblem)
            subproblem = self._build_subproblem(problem, x_k, f_k, grad_f_k, con_k, grad_con_k)

            # 3. Solve the subproblem
            x_next = self._solve_subproblem(subproblem)

            # 4. Check convergence
            dx = np.linalg.norm(x_next - x_k)
            threshold = xtol_rel * np.linalg.norm(x_k) + 1e-12
            LOGGER.info(f"Design change: dx = {dx:.4e} (threshold: {threshold:.4e})")
            if dx < threshold:
                LOGGER.info("Converged: small design change.")
                break
            
            x_k = x_next
            problem.design_space.set_current_value(x_k)

        return "SCP finished", 0

    @abstractmethod
    def _build_subproblem(
        self,
        original_problem: OptimizationProblem,
        x_k: ndarray,
        f_k: ndarray,
        grad_f_k: ndarray,
        con_k: list[ndarray],
        grad_con_k: list[ndarray],
    ) -> OptimizationProblem:
        """Create an approximate optimization problem at the current point."""

    @abstractmethod
    def _solve_subproblem(self, subproblem: OptimizationProblem) -> ndarray:
        """Solve the approximate subproblem."""

class SequentialConvexProgramming(SequentialProgramming):
    """Specialization for Convex approximations using Uno as the engine."""

    def _solve_subproblem(self, subproblem: OptimizationProblem) -> ndarray:
        """Use an inner solver to optimize the convex subproblem."""
        algo_name = self._settings.inner_solver
        
        if self._settings.inner_library.upper() == "UNO":
            uno = UnoOpt()
            inner_options = {
                "preset": self._settings.inner_preset,
                "solver": algo_name,
                "max_iter": self._settings.inner_max_iter,
                "hessian": "identity"
            }
            result = uno.execute(subproblem, **inner_options)
        else:
            # Idiomatic GEMSEO way to run an optimizer
            from gemseo import execute_algo
            # Use keyword argument to avoid algo_type positional conflict
            result = execute_algo(subproblem, algo_name=algo_name, max_iter=self._settings.inner_max_iter)
            
        return result.x_opt
