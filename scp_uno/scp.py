# Copyright (c) 2026 Simone Coniglio
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
    """Generic base class for Sequential Programming (SCP, SBO, TR).

    Subclasses must implement:
      - ``_build_subproblem``: construct an approximate problem at the current iterate.
      - ``_solve_subproblem``: solve that approximate problem and return x_opt.

    Optionally, the monotone backtracking line-search can be enabled via
    ``use_line_search=True`` in the settings.  It evaluates the l1 penalty
    merit function

        φ(x) = f(x) + μ · Σᵢ max(0, cᵢ(x))

    and backs off the step x_k → x_next until sufficient decrease is achieved.
    """

    def __init__(self, algo_name: str):
        super().__init__(algo_name)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(
        self,
        problem: OptimizationProblem,
        **options: Any,
    ) -> OptimizationResult:
        """Execute the sequential loop."""
        max_iter = self._settings.max_iter
        xtol_rel = self._settings.xtol_rel

        x_k = problem.design_space.get_current_value()

        # Outer loop
        for it in range(max_iter):
            # 1. Evaluate current point
            f_k = problem.objective.evaluate(x_k)
            grad_f_k = problem.objective.jac(x_k)

            con_k = []
            grad_con_k = []
            for constraint in problem.constraints:
                con_k.append(constraint.evaluate(x_k).flatten())
                grad_con_k.append(constraint.jac(x_k))

            max_con = np.max(np.concatenate(con_k)) if con_k else 0.0
            LOGGER.info(
                f"Outer Iter {it}: f = {f_k[0]:.4e}, max_con = {max_con:.4e}"
            )

            # 2. Build the approximation subproblem
            subproblem = self._build_subproblem(
                problem, x_k, f_k, grad_f_k, con_k, grad_con_k
            )

            # 3. Solve the subproblem → candidate next iterate
            x_next = self._solve_subproblem(subproblem)

            # 4. Optional: monotone backtracking line-search
            if self._settings.use_line_search:
                x_next = self._line_search(
                    problem, x_k, x_next, f_k, grad_f_k, con_k
                )

            # 5. Convergence check
            dx = np.linalg.norm(x_next - x_k)
            threshold = xtol_rel * np.linalg.norm(x_k) + 1e-12
            LOGGER.info(
                f"Design change: dx = {dx:.4e} (threshold: {threshold:.4e})"
            )
            if dx < threshold:
                LOGGER.info("Converged: small design change.")
                break

            x_k = x_next
            problem.design_space.set_current_value(x_k)

        return "SCP finished", 0

    # ------------------------------------------------------------------
    # Monotone backtracking line-search
    # ------------------------------------------------------------------

    def _merit(
        self,
        problem: OptimizationProblem,
        x: ndarray,
        mu: float,
    ) -> float:
        """l1 penalty merit function φ(x) = f(x) + μ·Σᵢ max(0, cᵢ(x))."""
        f_val = problem.objective.evaluate(x)[0]
        penalty = 0.0
        for constraint in problem.constraints:
            c_val = constraint.evaluate(x).flatten()
            penalty += np.sum(np.maximum(0.0, c_val))
        return f_val + mu * penalty

    def _line_search(
        self,
        problem: OptimizationProblem,
        x_k: ndarray,
        x_next: ndarray,
        f_k: ndarray,
        grad_f_k: ndarray,
        con_k: list[ndarray],
    ) -> ndarray:
        """Armijo backtracking line-search on the l1 merit function.

        Searches along the direction d = x_next - x_k for the largest step
        α ∈ {1, β, β², …} satisfying:

            φ(x_k + α·d) ≤ φ(x_k) - c1·α·|φ(x_k) - φ(x_k + d)|

        where φ is the l1 penalty merit.  Falls back to α = β^max_trials if
        no sufficient-decrease step is found (avoids stagnation).
        """
        mu = self._settings.line_search_penalty
        c1 = self._settings.line_search_c1
        beta = self._settings.line_search_beta
        max_trials = self._settings.line_search_max_trials

        d = x_next - x_k
        phi_k = self._merit(problem, x_k, mu)
        phi_next_full = self._merit(problem, x_next, mu)

        # Directional decrease estimate (use full-step merit difference)
        decrease_est = phi_k - phi_next_full

        # If the full step already gives sufficient decrease, accept it
        if decrease_est >= 0:
            LOGGER.info(f"Line-search: full step accepted (φ decrease = {decrease_est:.3e})")
            return x_next

        # Backtrack
        alpha = 1.0
        lb = problem.design_space.get_lower_bounds()
        ub = problem.design_space.get_upper_bounds()

        for trial in range(max_trials):
            alpha *= beta
            x_trial = np.clip(x_k + alpha * d, lb, ub)
            phi_trial = self._merit(problem, x_trial, mu)
            improvement = phi_k - phi_trial

            LOGGER.info(
                f"Line-search trial {trial + 1}: α={alpha:.4f}, "
                f"φ={phi_trial:.4e}, improvement={improvement:.3e}"
            )

            if improvement >= c1 * alpha * abs(decrease_est):
                LOGGER.info(f"Line-search: accepted at α={alpha:.4f}")
                return x_trial

        # No improvement found — accept the smallest tried step
        x_fallback = np.clip(x_k + alpha * d, lb, ub)
        LOGGER.warning(
            f"Line-search: no sufficient decrease after {max_trials} trials; "
            f"accepting α={alpha:.4f} (fallback)."
        )
        return x_fallback

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

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
    """Specialisation for convex approximations (MMA, CONLIN, SLP)."""

    def _solve_subproblem(self, subproblem: OptimizationProblem) -> ndarray:
        """Use an inner solver to optimise the convex subproblem."""
        algo_name = self._settings.inner_solver

        if self._settings.inner_library.upper() == "UNO":
            uno = UnoOpt()
            inner_options = {
                "preset": self._settings.inner_preset,
                "solver": algo_name,
                "max_iter": self._settings.inner_max_iter,
                "hessian": "identity",
            }
            result = uno.execute(subproblem, **inner_options)
        else:
            from gemseo import execute_algo
            result = execute_algo(
                subproblem,
                algo_name=algo_name,
                max_iter=self._settings.inner_max_iter,
            )

        return result.x_opt
