# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Sequential Linear Programming (SLP) in the SCP framework.

At each outer iteration the objective and constraints are approximated by their
first-order Taylor expansion:

    f̃(x) = f(xₖ) + ∇f(xₖ)ᵀ (x − xₖ)
    c̃ᵢ(x) = cᵢ(xₖ) + ∇cᵢ(xₖ)ᵀ (x − xₖ)

The resulting **Linear Program** (LP) is solved over a move-limited box

    xₖ − δ·s ≤ x ≤ xₖ + δ·s,   s = ub − lb

where δ = ``max_optimization_step`` (default 0.05).

Compared to CONLIN / MMA:
  * No reciprocal terms → globally valid (no singularity near x = 0).
  * The subproblem is the cheapest possible (linear objective + linear
    constraints + box bounds) and can be solved with any LP or NLP solver.
  * Convergence can be slower because the linear model underestimates
    curvature.  Pairing with the monotone line-search (``use_line_search=True``)
    typically helps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_linear_function import MDOLinearFunction

from scp_uno.scp import SequentialConvexProgramming
from scp_uno.settings import SCPSettings

if TYPE_CHECKING:
    from numpy import ndarray

LOGGER = logging.getLogger(__name__)


class SLP(SequentialConvexProgramming):
    """Sequential Linear Programming (SLP) using GEMSEO as the inner LP solver.

    The subproblem is a box-constrained LP solved by SLSQP (or any inner solver
    configured via ``inner_solver``).  Because the subproblem is purely linear,
    the inner solver converges in very few iterations.

    Optionally enable the monotone backtracking line-search via
    ``use_line_search=True`` to improve outer-loop stability.
    """

    # Default inner solver: HiGHS dual simplex (exact LP, no jac calls needed
    # beyond the linear coefficients already embedded in MDOLinearFunction).
    # Override with inner_solver='INTERIOR_POINT' or inner_solver='SLSQP' if needed.
    _DEFAULT_INNER_SOLVER = "DUAL_SIMPLEX"

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "SLP": OptimizationAlgorithmDescription(
            algorithm_name="SLP",
            internal_algorithm_name="SLP",
            library_name="SLP",
            description=(
                "Sequential Linear Programming: linearise objective and constraints "
                "at each outer iterate, solve the resulting LP (HiGHS dual simplex "
                "by default) over a move-limited box."
            ),
            Settings=SCPSettings,
            require_gradient=True,
            handle_inequality_constraints=True,
        )
    }

    def __init__(self) -> None:
        super().__init__("SLP")
        # Override SCPSettings default so callers don't need to specify it
        # (still overridable via the options dict passed to execute()).
        self._default_inner_solver = self._DEFAULT_INNER_SOLVER

    def _build_subproblem(
        self,
        original_problem: OptimizationProblem,
        x_k: ndarray,
        f_k: ndarray,
        grad_f_k: ndarray,
        con_k: list[ndarray],
        grad_con_k: list[ndarray],
    ) -> OptimizationProblem:
        """Build the LP subproblem at xₖ with move limits.

        The linearisation is:

            f̃(x) = (f_k − gᵀxₖ) + gᵀx      [g = grad_f_k.flatten()]
            c̃ᵢ(x) = (cᵢ − Jᵢxₖ) + Jᵢx     [Jᵢ = ith constraint Jacobian row]

        Written as functions of x so that any GEMSEO inner solver can consume them.
        """
        n = len(x_k)
        lb_orig = original_problem.design_space.get_lower_bounds()
        ub_orig = original_problem.design_space.get_upper_bounds()
        s = ub_orig - lb_orig

        # ---- Move-limited design space ----
        move = self._settings.max_optimization_step
        lb_move = np.maximum(lb_orig, x_k - move * s)
        ub_move = np.minimum(ub_orig, x_k + move * s)

        from gemseo.algos.design_space import DesignSpace
        sub_ds = DesignSpace()
        for var in original_problem.design_space.variable_names:
            size = original_problem.design_space.variable_sizes[var]
            sub_ds.add_variable(
                var, size, lower_bound=lb_move, upper_bound=ub_move, value=x_k
            )

        # ---- Linear approximation factory (using MDOLinearFunction) ----
        # MDOLinearFunction(coefficients, name, value_at_zero=offset) represents
        #   f̃(x) = coefficients @ x + value_at_zero
        # which GEMSEO's LP solvers (DUAL_SIMPLEX / INTERIOR_POINT) require.
        # The offset is chosen so that f̃(xₖ) = val_k (Taylor consistency).
        def build_linear_approx(val_k: float, grad_k: ndarray, name: str) -> MDOLinearFunction:
            """Return f̃(x) = gᵀx + c₀  with c₀ = val_k − gᵀxₖ."""
            g = grad_k.flatten()
            c0 = float(val_k - g @ x_k)  # scalar offset
            return MDOLinearFunction(g, name, value_at_zero=c0)

        # ---- Assemble subproblem ----
        sub_problem = OptimizationProblem(sub_ds)
        sub_problem.objective = build_linear_approx(
            float(f_k[0]), grad_f_k.flatten(), "slp_obj"
        )

        full_grad_con = (
            np.concatenate(grad_con_k, axis=0) if grad_con_k else np.empty((0, n))
        )
        full_con_k = np.concatenate(con_k) if con_k else np.empty(0)

        for i in range(len(full_con_k)):
            sub_problem.add_constraint(
                build_linear_approx(
                    float(full_con_k[i]), full_grad_con[i], f"slp_con_{i}"
                ),
                constraint_type="ineq",
            )

        return sub_problem
