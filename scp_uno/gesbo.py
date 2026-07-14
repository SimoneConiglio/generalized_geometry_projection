# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Gradient-Enhanced Surrogate-Based Optimization (GE-SBO) as a GEMSEO library.

This registers the ``GE_SBO`` algorithm: a trust-region-managed, batch,
gradient-enhanced surrogate-based optimizer for expensive problems whose
gradients are available (adjoint sensitivities). The heavy lifting lives in
the GEMSEO-free :mod:`scp_uno.gesbo_core`; this module only adapts a GEMSEO
:class:`~gemseo.algos.optimization_problem.OptimizationProblem` to the core
driver, in the same spirit as :mod:`scp_uno.scp`.

Highlights (see :mod:`scp_uno.gesbo_core` for the full description):

* **gradient-enhanced kriging** — the surrogate is conditioned on values and
  gradients, so each expensive evaluation contributes ``1 + r`` observations;
* **active-subspace reduction** — for high-dimensional designs the surrogate
  operates on a low-dimensional subspace identified from the gradients, which
  keeps the kriging system small regardless of the design-space size;
* **multi-point (batch) acquisition** — every outer iteration proposes
  ``batch_size`` candidate points (penalized exploitation + an LCB
  exploration ladder with diversity repulsion) that are conceptually
  independent and could be evaluated in parallel;
* **trust-region model management** — a classical ratio test on an l1 merit
  function governs step acceptance and radius updates.

Usage::

    scenario.execute(algo_name="GE_SBO", max_iter=200, batch_size=4)
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

from scp_uno.gesbo_core import GESBOConfig, gesbo_minimize
from scp_uno.settings import GESBOSettings

if TYPE_CHECKING:
    from numpy import ndarray

LOGGER = logging.getLogger(__name__)


class GradientEnhancedSBO(BaseOptimizationLibrary[GESBOSettings]):
    """GEMSEO wrapper of the GE-SBO trust-region batch surrogate optimizer."""

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "GE_SBO": OptimizationAlgorithmDescription(
            algorithm_name="GE_SBO",
            internal_algorithm_name="GE_SBO",
            library_name="GE_SBO",
            description=(
                "Gradient-Enhanced Surrogate-Based Optimization: trust-region "
                "managed gradient-enhanced kriging with active-subspace "
                "reduction for high-dimensional problems and multi-point "
                "(batch) acquisition at each iteration."
            ),
            Settings=GESBOSettings,
            require_gradient=True,
            handle_inequality_constraints=True,
        )
    }

    def __init__(self, algo_name: str = "GE_SBO") -> None:
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
        config = GESBOConfig(
            batch_size=s.batch_size,
            n_init=(s.n_init_doe or None),
            max_evals=s.max_iter,
            max_outer_iter=s.max_outer_iter,
            max_latent_dim=s.max_latent_dim,
            max_points=s.max_points,
            max_grad_points=s.max_grad_points,
            use_gradients=s.use_gradients,
            regularization=s.regularization,
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

        result = gesbo_minimize(evaluate, x0, lb, ub, config)
        LOGGER.info(
            "GE-SBO: %s after %d evaluations (%d batch iterations), f_opt=%.6e",
            result.status, result.n_evals, result.n_iter, result.f_opt,
        )
        design_space.set_current_value(result.x_opt)
        return f"GE-SBO finished: {result.status}", 0
