# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Continuation (homotopy) benchmark on the short cantilever.

Runs a warm-started sequence of sharpening phases — soft KS aggregation /
saturation first (``ka``/``pp`` overrides, wide merged basins), baseline
sharpness last — with the TOTAL true-evaluation budget split across phases,
so results are iso-function-calls with the single-phase study. Only the
final phase (baseline sharpness) is comparable with the study numbers.

Usage (inside the ``ggp`` conda environment)::

    python benchmarks/mls_continuation.py --config quadratic --seed 0
    python benchmarks/mls_continuation.py --config wendland  --seed 1
    python benchmarks/mls_continuation.py --config mma
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ggp.problem.loader import load_problem
from ggp.optimization.pipeline import GGPPipeline

PRESET = Path(__file__).resolve().parents[1] / "ggp" / "cli" / "presets" / "short_cantilever.yaml"

# smooth -> sharp: soft aggregation/saturation AND soft penalties first
# (wide merged basins, gray material free), full penalization last - gray
# elements at convergence indicate the penalty was not ramped, since under
# GP's linear stiffness (p=1) field-level gray is structurally optimal.
# Budgets sum to --max-evals.
DEFAULT_SCHEDULE = [
    # Measured winner: continue ONLY the geometry sharpness (ka/pp) and keep
    # the material penalty at full strength (gammac=3) in every phase.
    # Penalty ramps backfire on rho_V grayness and compliance alike:
    # gammac 1->2->3 gives 195-225 with gray ~0.62 (the soft-gammac phase
    # creates spread-out gray-mass sheets the later phases cannot
    # consolidate); stacking p_penalty 1->3 on top gives 203-357 with zero
    # solid elements (effective Mc^9 collapses mid-gray stiffness). The
    # ka/pp-only schedule + tr_init=0.05 gives 127/155/206 (rho_V gray 0.44).
    ({"ka": 3.0, "pp": 20.0}, 0.35),
    ({"ka": 6.0, "pp": 50.0}, 0.30),
    ({}, 0.35),
]


def solver_options(config: str, seed: int, tr_init=None) -> tuple[str, dict]:
    if config == "mma":
        return "MMA", None                # keep the preset's MMA options
    common = {"seed": seed, "batch_size": 1, "max_outer_iter": 400}
    if tr_init is not None:
        common["tr_init"] = tr_init
    if config == "quadratic":
        return "MLS_SBO", {**common, "model": "quadratic", "fit_values": "none",
                           "min_fit_neighbors": 1, "intermediate": "linear",
                           "n_global": 0}
    if config == "wendland":
        return "MLS_SBO", {**common, "model": "tangent", "weighting": "wendland"}
    if config == "product":                    # Deparis nn-radii (default)
        return "MLS_SBO", {**common, "model": "product"}
    if config == "product_loo":                # global radius, LOO-selected
        return "MLS_SBO", {**common, "model": "product",
                           "radius": "global", "auto_support": True}
    if config == "oa":                         # outer approximation (MILP)
        return "MLS_SBO", {**common, "model": "oa"}
    if config == "alpha":                      # alphaBB underestimator
        return "MLS_SBO", {**common, "model": "alpha"}
    raise SystemExit(f"unknown config {config}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        choices=["quadratic", "wendland", "product",
                                 "product_loo", "oa", "alpha", "mma"],
                        default="quadratic")
    parser.add_argument("--max-evals", type=int, default=200,
                        help="TOTAL budget across all phases.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tr-init", type=float, default=None)
    parser.add_argument("--n-init", type=int, default=None,
                        help="LHS DOE size inside the initial trust region "
                             "(first phase only; later phases warm-start).")
    parser.add_argument("--plot-dir", type=Path, default=None)
    args = parser.parse_args()

    spec = load_problem(PRESET)
    algo, options = solver_options(args.config, args.seed, args.tr_init)
    t0 = time.time()
    x = None
    result = None
    for i, (overrides, frac) in enumerate(DEFAULT_SCHEDULE):
        iters = max(1, int(round(args.max_evals * frac)))
        phase_opts = spec.solver.options if options is None else dict(options)
        if options is not None and args.n_init and i == 0:
            phase_opts = {**phase_opts, "n_init_doe": args.n_init}
        s = replace(spec, solver=replace(
            spec.solver, algorithm=algo, max_iter=iters,
            fem_solver="direct",
            options=phase_opts,
        ))
        result = GGPPipeline(s, x0=x, overrides=dict(overrides)).run()
        x = np.asarray(result.design_variables, float).flatten()
        compliance = float(np.expm1(result.objective_value))
        tag = ",".join(f"{k}={v}" for k, v in overrides.items()) or "baseline"
        print(f"[phase {i} {tag}] iters<={iters} compliance={compliance:.3f} "
              f"(comparable only for the final baseline phase)")

    # Fair reporting: the final phase may run sharpened physics (p_penalty=3),
    # so re-evaluate the final design once under the BASELINE model - at a
    # black/white design the two coincide, and the gray fraction shows how
    # well the penalty ramp binarized the field.
    s_eval = replace(spec, solver=replace(
        spec.solver, algorithm=algo, max_iter=1, fem_solver="direct",
        options=(spec.solver.options if options is None else dict(options)),
    ))
    base_eval = GGPPipeline(s_eval, x0=x).run()
    compliance = float(np.expm1(base_eval.objective_value))
    rho = np.asarray(result.density_field, float)
    gray = float(np.mean((rho > 0.1) & (rho < 0.9)))
    solid = float(np.mean(rho >= 0.9))
    print(f"[CONTINUATION {args.config}] final compliance={compliance:.3f} "
          f"gray_frac={gray:.3f} solid_frac={solid:.3f} "
          f"time={time.time() - t0:.1f}s")
    if args.plot_dir and result.density_field is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        from ggp.visualization.plot import save_density_plot_2d
        save_density_plot_2d(
            result.density_field, result.eval_coords,
            args.plot_dir / f"continuation_{args.config}_s{args.seed}.png",
            title=f"continuation {args.config} seed {args.seed}")
        np.savez(args.plot_dir / f"continuation_{args.config}_s{args.seed}.npz",
                 x_opt=result.design_variables, density=result.density_field,
                 objective=result.objective_value)


if __name__ == "__main__":
    main()
