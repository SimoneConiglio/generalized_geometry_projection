# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""GEMSEO libraries for the reduced-space (2D) optimizers.

Registers two algorithms built on :mod:`scp_uno.reduced_space`:

* ``GEK2D`` — at each iterate, a gradient-enhanced kriging model of the
  objective and aggregated constraint is fitted over the 2D frame spanned by
  ``-grad f`` and the orthogonalized constraint gradient, and the trust-region
  subproblem is solved on the surrogate (a few true evaluations per iteration).
* ``TRANSFORMER_2D`` — the transformer proposer of
  :mod:`scp_uno.rs_transformer` predicts the 2D step directly from the
  iteration history at zero inner-evaluation cost. The packaged weights are
  trained **only on generic synthetic families** (no topology-optimization
  data), and the reduced space makes the policy independent of the number of
  design variables by construction.

Both are local-descent methods for multimodal problems: they aim at reaching
a good local minimum efficiently, with no optimality claim.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from gemseo.algos.opt.base_optimization_library import (
    BaseOptimizationLibrary,
    OptimizationAlgorithmDescription,
)
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.optimization_result import OptimizationResult

from scp_uno.settings import GEK2DSettings, RSTransformerSettings

if TYPE_CHECKING:
    from numpy import ndarray

LOGGER = logging.getLogger(__name__)

RS_DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "rs_transformer_default.npz"


def _adapt_problem(problem: OptimizationProblem):
    """Wrap a GEMSEO problem into the reduced-space evaluation contract."""
    design_space = problem.design_space
    x0 = design_space.get_current_value()
    lb = design_space.get_lower_bounds()
    ub = design_space.get_upper_bounds()
    n = len(x0)
    constraints = list(problem.constraints)

    def evaluate(x: "ndarray"):
        f = float(np.asarray(problem.objective.evaluate(x)).flatten()[0])
        grad = np.asarray(problem.objective.jac(x), float).flatten()
        if constraints:
            vals, jacs = [], []
            for constraint in constraints:
                c = np.asarray(constraint.evaluate(x), float).flatten()
                jacs.append(np.asarray(constraint.jac(x), float).reshape(len(c), n))
                vals.append(c)
            return f, grad, np.concatenate(vals), np.vstack(jacs)
        return f, grad, np.zeros(0), np.zeros((0, n))

    return evaluate, x0, lb, ub


def _config_from(s) -> "ReducedSpaceConfig":
    from scp_uno.reduced_space import ReducedSpaceConfig

    return ReducedSpaceConfig(
        max_evals=s.max_iter,
        subspace_dim=s.subspace_dim,
        delta_init=s.delta_init,
        delta_min=s.delta_min,
        delta_max=s.delta_max,
        shrink=s.shrink,
        expand=s.expand,
        penalty=s.penalty,
        constraint_tol=s.constraint_tol,
        ks_rho=s.ks_rho,
        stall_limit=s.stall_limit,
        n_backtracks=s.n_backtracks,
        n_resets=s.n_resets,
        seed=s.seed,
        n_inner=getattr(s, "n_inner", 4),
        n_candidates=getattr(s, "n_candidates", 4096),
    )


class GEK2D(BaseOptimizationLibrary[GEK2DSettings]):
    """Reduced-space SCP with a gradient-enhanced kriging 2D subproblem."""

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "GEK2D": OptimizationAlgorithmDescription(
            algorithm_name="GEK2D",
            internal_algorithm_name="GEK2D",
            library_name="GEK2D",
            description=(
                "Sequential 2D reduced-space optimization: trust-region "
                "subproblem in the plane of the objective gradient and the "
                "orthogonalized aggregated-constraint gradient, solved on a "
                "gradient-enhanced kriging surrogate fitted from a few true "
                "evaluations per iteration."
            ),
            Settings=GEK2DSettings,
            require_gradient=True,
            handle_inequality_constraints=True,
        )
    }

    def __init__(self, algo_name: str = "GEK2D") -> None:
        super().__init__(algo_name)

    def _run(self, problem: OptimizationProblem, **options: Any
             ) -> OptimizationResult:
        from scp_uno.reduced_space import gek2d_minimize

        evaluate, x0, lb, ub = _adapt_problem(problem)
        result = gek2d_minimize(evaluate, x0, lb, ub, _config_from(self._settings))
        LOGGER.info("GEK2D: %s after %d evaluations (%d iterations), f_opt=%.6e",
                    result.status, result.n_evals, result.n_iter, result.f_opt)
        problem.design_space.set_current_value(result.x_opt)
        return f"GEK2D finished: {result.status}", 0


class Transformer2D(BaseOptimizationLibrary[RSTransformerSettings]):
    """Reduced-space optimization with the generic transformer proposer."""

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "TRANSFORMER_2D": OptimizationAlgorithmDescription(
            algorithm_name="TRANSFORMER_2D",
            internal_algorithm_name="TRANSFORMER_2D",
            library_name="TRANSFORMER_2D",
            description=(
                "Sequential 2D reduced-space optimization where a transformer "
                "trained on generic synthetic problems (dimension-agnostic by "
                "construction) predicts the trust-region step from the "
                "iteration history at zero inner-evaluation cost."
            ),
            Settings=RSTransformerSettings,
            require_gradient=True,
            handle_inequality_constraints=True,
        )
    }

    def __init__(self, algo_name: str = "TRANSFORMER_2D") -> None:
        super().__init__(algo_name)

    def _run(self, problem: OptimizationProblem, **options: Any
             ) -> OptimizationResult:
        from scp_uno.rs_transformer import load_params, rs_transformer_minimize

        s = self._settings
        weights = Path(s.weights_path) if s.weights_path else RS_DEFAULT_WEIGHTS
        if not weights.exists():
            raise FileNotFoundError(
                f"TRANSFORMER_2D policy weights not found: {weights}. Train "
                "them with scripts/train_rs_transformer.py or pass weights_path."
            )
        params, policy_cfg = load_params(weights)
        evaluate, x0, lb, ub = _adapt_problem(problem)
        result = rs_transformer_minimize(
            evaluate, x0, lb, ub, params, policy_cfg, _config_from(s),
            use_policy_radius=s.use_policy_radius,
        )
        LOGGER.info(
            "TRANSFORMER_2D: %s after %d evaluations (%d iterations), f_opt=%.6e",
            result.status, result.n_evals, result.n_iter, result.f_opt)
        problem.design_space.set_current_value(result.x_opt)
        return f"TRANSFORMER_2D finished: {result.status}", 0
