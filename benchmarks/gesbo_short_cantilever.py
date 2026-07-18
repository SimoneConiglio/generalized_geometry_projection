# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Benchmark GE_SBO / TRANSFORMER_OPT on the short-cantilever GGP preset.

Runs the built-in ``short_cantilever`` problem (18 free bars, 108 design
variables, compliance objective, 40% volume constraint) with the selected
surrogate/learned optimizer and reports compliance, volume feasibility and
timing. Optionally runs the preset MMA baseline with the same evaluation
budget for comparison.

Usage (inside the ``ggp`` conda environment)::

    python benchmarks/gesbo_short_cantilever.py --max-evals 200 --batch-size 4
    python benchmarks/gesbo_short_cantilever.py --algo TRANSFORMER_OPT
    python benchmarks/gesbo_short_cantilever.py --with-mma-baseline
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


def run_algo(spec, algo: str, max_evals: int, seed: int, fem_solver=None,
             **options):
    run_spec = replace(
        spec,
        solver=replace(
            spec.solver,
            algorithm=algo,
            max_iter=max_evals,
            **({"fem_solver": fem_solver} if fem_solver else {}),
            # the preset's options are MMA-specific (asymptotes, move limits):
            # replace them with the selected algorithm's settings.
            options={"seed": seed, **options},
        ),
    )
    t0 = time.time()
    result = GGPPipeline(run_spec).run()
    return result, time.time() - t0


def run_mma(spec, max_iter: int):
    mma_spec = replace(spec, solver=replace(spec.solver, max_iter=max_iter))
    t0 = time.time()
    result = GGPPipeline(mma_spec).run()
    return result, time.time() - t0


def report(tag: str, result, elapsed: float):
    # the pipeline's compliance objective is log(C + 1)
    compliance = float(np.expm1(result.objective_value))
    print(
        f"[{tag}] status={result.status} iters={result.iterations} "
        f"objective(log(C+1))={result.objective_value:.6f} compliance={compliance:.3f} "
        f"time={elapsed:.1f}s"
    )
    return compliance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-evals", type=int, default=200,
                        help="True-evaluation budget (default 200).")
    parser.add_argument(
        "--algo",
        choices=["GE_SBO", "TRANSFORMER_OPT", "GEK2D", "TRANSFORMER_2D",
                 "MLS_SBO"],
        default="GE_SBO", help="Algorithm to benchmark.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Points acquired per GE_SBO iteration (default 4).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-latent-dim", type=int, default=None,
                        help="Active-subspace size (default: GESBOSettings default).")
    parser.add_argument("--max-grad-points", type=int, default=None)
    parser.add_argument("--tr-init", type=float, default=None)
    parser.add_argument("--max-outer-iter", type=int, default=None,
                        help="Cap on outer (batch) iterations (GE_SBO/MLS_SBO).")
    parser.add_argument("--intermediate", type=str, default=None,
                        choices=["mma", "linear"],
                        help="MLS_SBO anchored-model intermediate variables.")
    parser.add_argument("--min-fit-neighbors", type=int, default=None,
                        help="MLS_SBO anchored-fit bandwidth floor (samples).")
    parser.add_argument("--fit-values", type=str, default=None,
                        choices=["none", "constraints", "all"],
                        help="MLS_SBO: outputs whose fit includes value rows.")
    parser.add_argument("--model", type=str, default=None,
                        choices=["tangent", "quadratic", "planar"],
                        help="MLS_SBO exploitation-subproblem model.")
    parser.add_argument("--weighting", type=str, default=None,
                        choices=["wendland", "shepard", "softmax"],
                        help="MLS_SBO tangent-blend weight family.")
    parser.add_argument("--ls-factor", type=float, default=None,
                        help="MLS_SBO length-scale factor h = ls_factor*d_min "
                             "(<1 interpolating regime, >>1 averaging).")
    parser.add_argument("--n-global", type=int, default=None,
                        help="MLS_SBO global subproblem candidates (0=off).")
    parser.add_argument("--fem-solver", type=str, default=None,
                        choices=["direct", "iterative", "amjax"],
                        help=("Override the preset's FEM linear solver. "
                              "'direct' (scipy SuperLU) is bit-deterministic "
                              "across processes; the preset default 'amjax' "
                              "(JAX/XLA) carries ~1e-13 per-process noise "
                              "that chaotic optimizer trajectories amplify."))
    parser.add_argument("--kappa-base", type=float, default=None)
    parser.add_argument("--n-inner", type=int, default=None,
                        help="GEK2D: true evaluations on the subspace per iteration.")
    parser.add_argument("--subspace-dim", type=int, default=None,
                        help="GEK2D/TRANSFORMER_2D: reduced-space directions.")
    parser.add_argument("--delta-init", type=float, default=None,
                        help="GEK2D/TRANSFORMER_2D: initial trust radius.")
    parser.add_argument("--no-policy-radius", action="store_true",
                        help="TRANSFORMER_2D: use the driver radius rule instead "
                             "of the learned multiplier.")
    parser.add_argument("--gek-refresh", type=int, default=None,
                        help="TRANSFORMER_2D: hybrid schedule, GEK iteration "
                             "every k-th step.")
    parser.add_argument("--weights", type=str, default=None,
                        help="TRANSFORMER_2D: policy weights path override.")
    parser.add_argument("--with-mma-baseline", action="store_true",
                        help="Also run the preset MMA with the same evaluation budget.")
    parser.add_argument("--plot-dir", type=Path, default=None,
                        help="Directory for density plots of the optimized designs.")
    args = parser.parse_args()

    spec = load_problem(PRESET)
    print(f"short_cantilever: mode={spec.formulation.mode} "
          f"components={spec.formulation.num_components} volfrac={spec.volfrac}")

    if args.algo == "GE_SBO":
        extra = {
            k: v for k, v in {
                "batch_size": args.batch_size,
                "max_latent_dim": args.max_latent_dim,
                "max_grad_points": args.max_grad_points,
                "tr_init": args.tr_init,
                "kappa_base": args.kappa_base,
            }.items() if v is not None
        }
    elif args.algo == "MLS_SBO":
        extra = {
            k: v for k, v in {
                "batch_size": args.batch_size,
                "tr_init": args.tr_init,
                "kappa_base": args.kappa_base,
                "max_outer_iter": args.max_outer_iter,
                "intermediate": args.intermediate,
                "min_fit_neighbors": args.min_fit_neighbors,
                "fit_values": args.fit_values,
                "model": args.model,
                "ls_factor": args.ls_factor,
                "n_global": args.n_global,
                "weighting": args.weighting,
            }.items() if v is not None
        }
    elif args.algo == "TRANSFORMER_OPT":
        extra = {k: v for k, v in {"tr_init": args.tr_init}.items()
                 if v is not None}
    else:                                       # GEK2D / TRANSFORMER_2D
        extra = {k: v for k, v in {
            "n_inner": args.n_inner if args.algo == "GEK2D" else None,
            "subspace_dim": args.subspace_dim,
            "delta_init": args.delta_init,
            "use_policy_radius": (False if (args.no_policy_radius
                                            and args.algo == "TRANSFORMER_2D")
                                  else None),
            "gek_refresh": (args.gek_refresh
                            if args.algo == "TRANSFORMER_2D" else None),
            "weights_path": (args.weights
                             if args.algo == "TRANSFORMER_2D" else None),
        }.items() if v is not None}
    result, elapsed = run_algo(spec, args.algo, args.max_evals, args.seed,
                               fem_solver=args.fem_solver, **extra)
    report(args.algo, result, elapsed)
    if args.plot_dir and result.density_field is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        from ggp.visualization.plot import save_density_plot_2d
        tag = args.algo.lower()
        save_density_plot_2d(result.density_field, result.eval_coords,
                             args.plot_dir / f"short_cantilever_{tag}.png",
                             title=f"short_cantilever {args.algo}")
        # Trust-region trajectories are chaotic (FP-level nondeterminism in
        # the FEM solves amplifies), so a good run may not be reproducible:
        # always keep the optimal design vector and density field.
        np.savez(args.plot_dir / f"short_cantilever_{tag}_design.npz",
                 x_opt=result.design_variables,
                 density=result.density_field,
                 objective=result.objective_value)

    if args.with_mma_baseline:
        base, base_elapsed = run_mma(spec, args.max_evals)
        report("MMA", base, base_elapsed)
        if args.plot_dir and base.density_field is not None:
            from ggp.visualization.plot import save_density_plot_2d
            save_density_plot_2d(base.density_field, base.eval_coords,
                                 args.plot_dir / "short_cantilever_mma.png",
                                 title="short_cantilever MMA")


if __name__ == "__main__":
    main()
