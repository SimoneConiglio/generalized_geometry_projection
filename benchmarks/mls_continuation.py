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

# smooth -> sharp: soft aggregation/saturation first (wide, merged basins),
# baseline (ka=10, pp=100) last. Budgets sum to --max-evals.
DEFAULT_SCHEDULE = [
    ({"ka": 3.0, "pp": 20.0}, 0.35),
    ({"ka": 6.0, "pp": 50.0}, 0.30),
    ({}, 0.35),
]


def solver_options(config: str, seed: int) -> tuple[str, dict]:
    if config == "mma":
        return "MMA", None                # keep the preset's MMA options
    common = {"seed": seed, "batch_size": 1, "max_outer_iter": 400}
    if config == "quadratic":
        return "MLS_SBO", {**common, "model": "quadratic", "fit_values": "none",
                           "min_fit_neighbors": 1, "intermediate": "linear",
                           "n_global": 0}
    if config == "wendland":
        return "MLS_SBO", {**common, "model": "tangent", "weighting": "wendland"}
    raise SystemExit(f"unknown config {config}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=["quadratic", "wendland", "mma"],
                        default="quadratic")
    parser.add_argument("--max-evals", type=int, default=200,
                        help="TOTAL budget across all phases.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-dir", type=Path, default=None)
    args = parser.parse_args()

    spec = load_problem(PRESET)
    algo, options = solver_options(args.config, args.seed)
    t0 = time.time()
    x = None
    result = None
    for i, (overrides, frac) in enumerate(DEFAULT_SCHEDULE):
        iters = max(1, int(round(args.max_evals * frac)))
        s = replace(spec, solver=replace(
            spec.solver, algorithm=algo, max_iter=iters,
            fem_solver="direct",
            options=(spec.solver.options if options is None
                     else dict(options)),
        ))
        result = GGPPipeline(s, x0=x, overrides=dict(overrides)).run()
        x = np.asarray(result.design_variables, float).flatten()
        compliance = float(np.expm1(result.objective_value))
        tag = ",".join(f"{k}={v}" for k, v in overrides.items()) or "baseline"
        print(f"[phase {i} {tag}] iters<={iters} compliance={compliance:.3f} "
              f"(comparable only for the final baseline phase)")

    compliance = float(np.expm1(result.objective_value))
    print(f"[CONTINUATION {args.config}] final compliance={compliance:.3f} "
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
