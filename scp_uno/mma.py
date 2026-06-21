# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Implementation of the Method of Moving Asymptotes (MMA) in the SCP framework."""

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

class MMA(SequentialConvexProgramming):
    """Method of Moving Asymptotes (MMA) using Uno as subproblem solver."""

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "MMA_UNO": OptimizationAlgorithmDescription(
            algorithm_name="MMA_UNO",
            internal_algorithm_name="MMA_UNO",
            library_name="MMA_UNO",
            description="MMA implemented in GEMSEO using Uno as the subproblem solver",
            Settings=SCPSettings,
            require_gradient=True,
            handle_inequality_constraints=True,
        )
    }

    def __init__(self):
        super().__init__("MMA_UNO")
        self.L = None
        self.U = None
        self.x_old1 = None
        self.x_old2 = None

    def _build_subproblem(
        self,
        original_problem: OptimizationProblem,
        x_k: ndarray,
        f_k: ndarray,
        grad_f_k: ndarray,
        con_k: list[ndarray],
        grad_con_k: list[ndarray]
    ) -> OptimizationProblem:
        """Create the MMA approximate subproblem."""
        
        n = len(x_k)
        lb_orig = original_problem.design_space.get_lower_bounds()
        ub_orig = original_problem.design_space.get_upper_bounds()
        
        s = ub_orig - lb_orig

        # 1. Update asymptotes
        if self.L is None:
            # First iteration: conservative initialization from baseline
            dist = self._settings.initial_asymptotes_distance
            self.L = x_k - dist * s
            self.U = x_k + dist * s
        elif self.x_old1 is not None and self.x_old2 is not None:
            # Svanberg 1987 update rules with configurable coefficients
            gamma_exp = self._settings.asymptotes_distance_amplification_coefficient
            gamma_red = self._settings.asymptotes_distance_reduction_coefficient
            
            for j in range(n):
                test = (x_k[j] - self.x_old1[j]) * (self.x_old1[j] - self.x_old2[j])
                if test > 0: # Oscillations absent: expand
                    self.L[j] = x_k[j] - gamma_exp * (self.x_old1[j] - self.L[j])
                    self.U[j] = x_k[j] + gamma_exp * (self.U[j] - self.x_old1[j])
                elif test < 0: # Oscillations detected: shrink
                    self.L[j] = x_k[j] - gamma_red * (self.x_old1[j] - self.L[j])
                    self.U[j] = x_k[j] + gamma_red * (self.U[j] - self.x_old1[j])
                else: # Neutral
                    self.L[j] = x_k[j] - (self.x_old1[j] - self.L[j])
                    self.U[j] = x_k[j] + (self.U[j] - self.x_old1[j])
        
        # Clip asymptotes using baseline bounds
        max_dist = self._settings.max_asymptote_distance
        min_dist = self._settings.min_asymptote_distance
        
        for j in range(n):
            self.L[j] = np.maximum(self.L[j], x_k[j] - max_dist * s[j])
            self.L[j] = np.minimum(self.L[j], x_k[j] - min_dist * s[j])
            self.U[j] = np.minimum(self.U[j], x_k[j] + max_dist * s[j])
            self.U[j] = np.maximum(self.U[j], x_k[j] + min_dist * s[j])

        self.x_old2 = self.x_old1
        self.x_old1 = x_k.copy()

        # 2. Build Approximations
        def build_mma_approx(val_k, grad_k, name):
            # Reciprocal coefficients
            # Ensure denominators are non-zero
            du = np.maximum(self.U - x_k, 1e-12)
            dl = np.maximum(x_k - self.L, 1e-12)
            
            p = (du**2) * np.maximum(grad_k, 0.0)
            q = (dl**2) * np.maximum(-grad_k, 0.0)
            
            # Constant term
            c_term = val_k - np.sum(p / du) - np.sum(q / dl)

            def func(x):
                # Clamp x to be inside asymptotes to prevent nan/inf
                dx_u = np.maximum(self.U - x, 1e-10)
                dx_l = np.maximum(x - self.L, 1e-10)
                return c_term + np.sum(p / dx_u) + np.sum(q / dx_l)
            
            def jac(x):
                dx_u = np.maximum(self.U - x, 1e-10)
                dx_l = np.maximum(x - self.L, 1e-10)
                return (p / (dx_u**2) - q / (dx_l**2)).reshape(1, -1)

            return MDOFunction(func, name, jac=jac)

        # 3. Create Subproblem
        sub_problem = OptimizationProblem(original_problem.design_space)
        sub_problem.objective = build_mma_approx(f_k[0], grad_f_k.flatten(), "mma_obj")
        
        full_grad_con = np.concatenate(grad_con_k, axis=0) if grad_con_k else np.empty((0, n))
        full_con_k = np.concatenate(con_k) if con_k else np.empty(0)
        for i in range(len(full_con_k)):
            sub_problem.add_constraint(build_mma_approx(full_con_k[i], full_grad_con[i], f"mma_con_{i}"), constraint_type="ineq")

        # 4. Move limits (strictly inside asymptotes)
        move = self._settings.max_optimization_step
        lb_m = x_k - move * s
        ub_m = x_k + move * s
        
        # Standard MMA safeguard: move limits cannot go closer than 10% to asymptotes
        lb_a = self.L + 0.1 * (x_k - self.L)
        ub_a = self.U - 0.1 * (self.U - x_k)
        
        lb_move = np.maximum(lb_orig, np.maximum(lb_m, lb_a))
        ub_move = np.minimum(ub_orig, np.minimum(ub_m, ub_a))
        
        from gemseo.algos.design_space import DesignSpace
        sub_ds = DesignSpace()
        for var in original_problem.design_space.variable_names:
            size = original_problem.design_space.variable_sizes[var]
            sub_ds.add_variable(var, size, lower_bound=lb_move, upper_bound=ub_move, value=x_k)
        
        sub_problem.design_space = sub_ds
        return sub_problem
