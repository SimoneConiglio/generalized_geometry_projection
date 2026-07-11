# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Record GEMSEO-MMA trajectories on the short-cantilever GGP problem.

Runs the ``short_cantilever`` preset with its exact MMA configuration from
*random* initial component layouts (the default deterministic start is left
out so the benchmark run is not part of the training data) and stores, for
every iterate: the design vector, the objective gradient, the volume
constraint value and its gradient. The resulting ``.npz`` is consumed by
``scripts/train_transformer_opt.py --ggp-data`` to fine-tune the
TRANSFORMER_OPT policy on the real problem family.

Usage (inside the ``ggp`` conda environment)::

    python scripts/collect_ggp_mma_trajectories.py --runs 3 --max-iter 200 \
        --out scp_uno/data/ggp_mma_trajectories.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PRESET = REPO / "ggp" / "cli" / "presets" / "short_cantilever.yaml"


def run_and_capture(spec, x0):
    """Run the pipeline while capturing the GEMSEO scenario it builds."""
    import ggp.optimization.pipeline as pl
    from ggp.optimization.pipeline import GGPPipeline

    captured = {}
    original = pl.create_scenario

    def hook(*args, **kwargs):
        scenario = original(*args, **kwargs)
        captured["scenario"] = scenario
        return scenario

    pl.create_scenario = hook
    try:
        GGPPipeline(spec, x0=x0).run()
    finally:
        pl.create_scenario = original
    return captured["scenario"]


def extract_trajectory(scenario):
    """Pull (X, G0, C, G1) in iteration order from the scenario database."""
    problem = scenario.formulation.optimization_problem
    database = problem.database

    X, G0, C, G1 = [], [], [], []
    for x_vect, outputs in database.items():
        x = np.asarray(x_vect.unwrap() if hasattr(x_vect, "unwrap") else x_vect,
                       float).flatten()
        if "@compliance" not in outputs or "@volume" not in outputs:
            continue
        X.append(x)
        G0.append(np.asarray(outputs["@compliance"], float).flatten())
        C.append(float(np.asarray(outputs["volume"], float).flatten()[0]))
        G1.append(np.asarray(outputs["@volume"], float).flatten())
    return (np.asarray(X, np.float32), np.asarray(G0, np.float32),
            np.asarray(C, np.float32), np.asarray(G1, np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=REPO / "scp_uno" / "data" / "ggp_mma_trajectories.npz")
    args = parser.parse_args()

    from dataclasses import replace
    from ggp.problem.loader import load_problem
    from ggp.optimization.global_search import random_initial_design

    spec = load_problem(PRESET)
    spec = replace(spec, solver=replace(spec.solver, max_iter=args.max_iter))
    num_vars = 6 * spec.formulation.num_components
    rng = np.random.default_rng(args.seed)

    data = {}
    for run in range(args.runs):
        x0 = random_initial_design(num_vars, spec.formulation.mode, rng, spec=spec)
        t0 = time.time()
        scenario = run_and_capture(spec, x0)
        X, G0, C, G1 = extract_trajectory(scenario)
        print(f"run {run}: {len(X)} iterates in {time.time() - t0:.0f}s, "
              f"final compliance(log)={float(np.expm1(0)) if not len(X) else 'n/a'}")
        print(f"  volume range [{C.min():.3f}, {C.max():.3f}]" if len(C) else "  EMPTY")
        for name, arr in (("X", X), ("G0", G0), ("C", C), ("G1", G1)):
            data[f"run{run}_{name}"] = arr

    data["move"] = np.asarray(0.01)   # the preset's MMA move limit
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"saved {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
