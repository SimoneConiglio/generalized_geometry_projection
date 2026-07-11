# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Transformer learned-optimizer as a GEMSEO library (``TRANSFORMER_OPT``).

A small set-transformer policy, trained offline by imitating a privileged
teacher on synthetic tasks (:mod:`scp_uno.transformer_opt_core`), reads the
recent optimization history and proposes the next **batch** of query points;
a classical trust-region loop safeguards acceptance. Gradients are consumed
as policy features (and for the active-subspace latent), so the algorithm is
gradient-enhanced and dimension-agnostic.

The packaged default weights (``scp_uno/weights/transformer_opt_default.npz``)
are trained by ``scripts/train_transformer_opt.py``; pass ``weights_path`` to
use custom ones. Requires JAX (available in the ``ggp`` conda environment).

Usage::

    scenario.execute(algo_name="TRANSFORMER_OPT", max_iter=200)
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

from scp_uno.settings import TransformerOptSettings

if TYPE_CHECKING:
    from numpy import ndarray

LOGGER = logging.getLogger(__name__)

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "transformer_opt_default.npz"


class TransformerOpt(BaseOptimizationLibrary[TransformerOptSettings]):
    """GEMSEO wrapper of the transformer learned optimizer."""

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "TRANSFORMER_OPT": OptimizationAlgorithmDescription(
            algorithm_name="TRANSFORMER_OPT",
            internal_algorithm_name="TRANSFORMER_OPT",
            library_name="TRANSFORMER_OPT",
            description=(
                "Learned optimizer: a per-variable-token transformer trained "
                "to imitate MMA's step map (plus far-sighted multi-scale "
                "heads) steps in full dimension on every evaluation, inside "
                "a merit backtracking safeguard."
            ),
            Settings=TransformerOptSettings,
            require_gradient=True,
            handle_inequality_constraints=True,
        )
    }

    def __init__(self, algo_name: str = "TRANSFORMER_OPT") -> None:
        super().__init__(algo_name)

    def _run(
        self,
        problem: OptimizationProblem,
        **options: Any,
    ) -> OptimizationResult:
        # JAX (and with it the core) is imported lazily so that the GEMSEO
        # factory can scan this module on installations without JAX.
        from scp_uno.transformer_opt_core import (
            TransformerOptConfig,
            load_params,
            transformer_opt_minimize,
        )

        s = self._settings
        weights = Path(s.weights_path) if s.weights_path else DEFAULT_WEIGHTS
        if not weights.exists():
            raise FileNotFoundError(
                f"TRANSFORMER_OPT policy weights not found: {weights}. "
                "Train them with scripts/train_transformer_opt.py or pass "
                "weights_path."
            )
        params, policy_cfg = load_params(weights)

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

        config = TransformerOptConfig(
            max_evals=s.max_iter,
            move=s.move,
            eval_heads=s.eval_heads,
            n_backtracks=s.n_backtracks,
            stall_limit=s.stall_limit,
            accept_mode=s.accept_mode,
            nonmonotone_window=s.nonmonotone_window,
            penalty=s.penalty,
            constraint_tol=s.constraint_tol,
            seed=s.seed,
        )
        result = transformer_opt_minimize(
            evaluate, x0, lb, ub, params, policy_cfg, config
        )
        LOGGER.info(
            "TRANSFORMER_OPT: %s after %d evaluations (%d batch iterations), "
            "f_opt=%.6e",
            result.status, result.n_evals, result.n_iter, result.f_opt,
        )
        design_space.set_current_value(result.x_opt)
        return f"TRANSFORMER_OPT finished: {result.status}", 0
