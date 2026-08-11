# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Moving-Least-Squares SBO (MLS-SBO) as a GEMSEO library.

Registers the ``MLS_SBO`` algorithm: a trust-region-managed, batch,
surrogate-based optimizer in which the objective and the constraints are
approximated by **gradient-enhanced Moving Least Squares** with an
**error-adaptive length scale** (see :mod:`scp_uno.mls_sbo_core`). This
module only adapts a GEMSEO
:class:`~gemseo.algos.optimization_problem.OptimizationProblem` to the
GEMSEO-free core driver, in the same spirit as :mod:`scp_uno.gesbo`.

Usage::

    scenario.execute(algo_name="MLS_SBO", max_iter=200, batch_size=4)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from gemseo.algos.opt.base_optimization_library import (
    BaseOptimizationLibrary,
    OptimizationAlgorithmDescription,
)
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.optimization_result import OptimizationResult

from scp_uno.mls_sbo_core import MLSSBOConfig, mls_sbo_minimize
from scp_uno.settings import MLSSBOSettings

if TYPE_CHECKING:
    from numpy import ndarray

LOGGER = logging.getLogger(__name__)


class MLSSurrogateSBO(BaseOptimizationLibrary[MLSSBOSettings]):
    """GEMSEO wrapper of the MLS-SBO trust-region batch surrogate optimizer."""

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "MLS_SBO": OptimizationAlgorithmDescription(
            algorithm_name="MLS_SBO",
            internal_algorithm_name="MLS_SBO",
            library_name="MLS_SBO",
            description=(
                "Surrogate-based optimization on gradient-enhanced Moving "
                "Least Squares (Hermite MLS, linear basis) for objective and "
                "constraints, with an error-adaptive length scale, "
                "trust-region model management and multi-point (batch) "
                "acquisition at each iteration."
            ),
            Settings=MLSSBOSettings,
            require_gradient=True,
            handle_inequality_constraints=True,
        )
    }

    def __init__(self, algo_name: str = "MLS_SBO") -> None:
        super().__init__(algo_name)

    def _run(
        self,
        problem: OptimizationProblem,
        **options: Any,
    ) -> OptimizationResult:
        """Adapt the GEMSEO problem to the core driver and run it."""
        design_space = problem.design_space
        x0 = design_space.get_current_value()
        lb = design_space.get_lower_bounds()
        ub = design_space.get_upper_bounds()
        n = len(x0)
        constraints = list(problem.constraints)

        def evaluate(x: ndarray):
            f = float(np.asarray(problem.objective.evaluate(x)).flatten()[0])
            grad = np.asarray(problem.objective.jac(x), float).flatten()
            if constraints:
                vals, jacs = [], []
                for constraint in constraints:
                    c = np.asarray(constraint.evaluate(x), float).flatten()
                    jacs.append(
                        np.asarray(constraint.jac(x), float).reshape(len(c), n)
                    )
                    vals.append(c)
                c_all, j_all = np.concatenate(vals), np.vstack(jacs)
            else:
                c_all, j_all = np.zeros(0), np.zeros((0, n))
            return f, grad, c_all, j_all

        s = self._settings
        config = MLSSBOConfig(
            batch_size=s.batch_size,
            n_init=(s.n_init_doe or None),
            max_evals=s.max_iter,
            max_outer_iter=s.max_outer_iter,
            max_points=s.max_points,
            regularization=s.regularization,
            ls_factor=s.ls_factor,
            ls_min=s.ls_min,
            ls_max=s.ls_max,
            anchor_center=s.anchor_center,
            subproblem_maxiter=s.subproblem_maxiter,
            n_global=s.n_global,
            weighting=s.weighting,
            support_factor=s.support_factor,
            radius=s.radius,
            nn_factor=s.nn_factor,
            auto_support=s.auto_support,
            min_sep_frac=s.min_sep_frac,
            intermediate=s.intermediate,
            asy_init=s.asy_init,
            min_fit_neighbors=s.min_fit_neighbors,
            fit_values=s.fit_values,
            model=s.model,
            n_resets=s.n_resets,
            hold_region=s.hold_region,
            region_patience=s.region_patience,
            oa_time_limit=s.oa_time_limit,
            alpha_safety=s.alpha_safety,
            alpha_mode=s.alpha_mode,
            tunnel=s.tunnel,
            tunnel_radius_factor=s.tunnel_radius_factor,
            tunnel_candidates=s.tunnel_candidates,
            tr_init=s.tr_init,
            tr_min=s.tr_min,
            tr_max=s.tr_max,
            tr_shrink=s.tr_shrink,
            tr_expand=s.tr_expand,
            kappa_base=s.kappa_base,
            kappa_growth=s.kappa_growth,
            repulsion_weight=s.repulsion_weight,
            n_restarts=s.acq_n_restarts,
            penalty=s.penalty,
            constraint_tol=s.constraint_tol,
            stall_limit=s.stall_limit,
            seed=s.seed,
        )

        result = mls_sbo_minimize(evaluate, x0, lb, ub, config)
        LOGGER.info(
            "MLS-SBO: %s after %d evaluations (%d batch iterations), f_opt=%.6e",
            result.status, result.n_evals, result.n_iter, result.f_opt,
        )
        design_space.set_current_value(result.x_opt)
        return f"MLS-SBO finished: {result.status}", 0
