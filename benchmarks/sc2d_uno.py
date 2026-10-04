# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Uno solvers on the Short Cantilever 2D (SC 2D) benchmark.

Runs the GGP short-cantilever preset (objective ``log(C+1)``, volume constraint)
with Uno's own algorithms -- presets ``ipopt`` (interior point + filter line
search), ``filtersqp`` (SQP + filter trust region), ``funnelsqp`` (SQP + funnel
trust region) and ``filterslp`` (SLP + filter trust region) -- through the
``UNO`` GEMSEO library (``scp_uno/uno_wrapper.py``), and compares them with the
preset's MMA at the same budget of FE analyses.

Run inside the conda ``ggp`` environment::

    python benchmarks/sc2d_uno.py                      # all configurations
    python benchmarks/sc2d_uno.py --configs mma filtersqp_tr0.05
    python benchmarks/sc2d_uno.py --max-iter 600 --list
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

import ggp.optimization.pipeline as pipeline_module
from ggp.optimization.pipeline import GGPPipeline
from benchmarks.sc2d_local_minima import _load_spec

OUT = Path(__file__).resolve().parent

# name -> solver options (``None`` keeps the preset's MMA configuration)
PRESETS = ["ipopt", "filtersqp", "funnelsqp", "filterslp"]
CS = {"constraint_scale": 0.01}   # volume response is in percent: bring it to O(1)
CONFIGS: dict[str, dict | None] = {"mma": None}
# 1. untuned presets (L-BFGS Hessian, Uno defaults otherwise)
CONFIGS.update({p: {"preset": p} for p in PRESETS})
# 2. same, volume constraint scaled to O(1)
CONFIGS.update({f"{p}_cs": {"preset": p, **CS} for p in PRESETS})
# 3. trust-region radius = move limit in the normalised [0, 1] design space
for p in ["filtersqp", "funnelsqp", "filterslp"]:
    for r in [0.1, 0.02]:
        CONFIGS[f"{p}_cs_tr{r}"] = {"preset": p, **CS, "uno_options": {"TR_radius": r}}
# 4. Hessian model / memory, interior-point barrier
CONFIGS["filtersqp_cs_m20"] = {"preset": "filtersqp", **CS, "quasi_newton_memory_size": 20}
CONFIGS["filtersqp_cs_id"] = {"preset": "filtersqp", **CS, "hessian_model": "identity"}
CONFIGS["ipopt_cs_mu0.01"] = {"preset": "ipopt", **CS,
                              "uno_options": {"barrier_initial_parameter": 0.01}}
CONFIGS["ipopt_cs_m20"] = {"preset": "ipopt", **CS, "quasi_newton_memory_size": 20}
# 5. constraint scale around the O(1) choice
CONFIGS["filtersqp_cs0.1"] = {"preset": "filtersqp", "constraint_scale": 0.1}
CONFIGS["filtersqp_cs0.001"] = {"preset": "filtersqp", "constraint_scale": 0.001}


def run_config(name, options, max_iter):
    spec = _load_spec(max_iter=max_iter)
    if options is not None:
        solver = dataclasses.replace(
            spec.solver, algorithm="UNO", options=dict(options), max_iter=max_iter
        )
        spec = dataclasses.replace(spec, solver=solver)

    captured = {}
    create_scenario = pipeline_module.create_scenario

    def capture(*args, **kwargs):
        captured["scenario"] = create_scenario(*args, **kwargs)
        return captured["scenario"]

    pipeline_module.create_scenario = capture
    t0 = time.time()
    try:
        result = GGPPipeline(spec).run()
    finally:
        pipeline_module.create_scenario = create_scenario
    elapsed = time.time() - t0

    problem = captured["scenario"].formulation.optimization_problem
    f_hist = np.array(problem.database.get_function_history("compliance")).ravel()
    v_hist = np.array(problem.database.get_function_history("volume")).ravel()
    opt = problem.solution
    volume = float(np.atleast_1d(opt.constraint_values["volume"])[0])
    C = float(np.expm1(opt.f_opt))
    feasible_C = np.expm1(f_hist[: len(v_hist)][v_hist <= 1e-6]) if len(v_hist) else []
    return {
        "config": name,
        "options": json.dumps(options) if options else "preset MMA",
        "compliance": C,
        "volume_con": volume,
        "feasible": bool(opt.is_feasible),
        "n_evals": int(len(problem.database)),
        "time_s": elapsed,
        "message": str(opt.message)[:120],
        "f_hist": f_hist.tolist(),
        "v_hist": v_hist.tolist(),
        "best_feasible_C": float(np.min(feasible_C)) if len(feasible_C) else float("inf"),
        "x_opt": np.asarray(opt.x_opt).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--max-iter", type=int, default=320,
                        help="Budget of FE analyses (GEMSEO max_iter), same for all.")
    parser.add_argument("--tag", default="", help="Suffix of the output files.")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for k, v in CONFIGS.items():
            print(f"{k:24s} {v}")
        return

    rows = []
    stem = OUT / f"sc2d_uno_results{args.tag}"
    for name in args.configs:
        print(f"=== {name} ===", flush=True)
        row = run_config(name, CONFIGS[name], args.max_iter)
        print(f"    C={row['compliance']:.4f} vol_con={row['volume_con']:+.2e} "
              f"feasible={row['feasible']} evals={row['n_evals']} "
              f"t={row['time_s']:.0f}s | {row['message']}", flush=True)
        rows.append(row)
        with open(f"{stem}.json", "w") as fh:
            json.dump(rows, fh)

    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "compliance", "volume_con", "feasible", "n_evals",
                    "time_s", "options", "message"])
        for r in rows:
            w.writerow([r["config"], f"{r['compliance']:.6f}", f"{r['volume_con']:.3e}",
                        r["feasible"], r["n_evals"], f"{r['time_s']:.1f}",
                        r["options"], r["message"]])


if __name__ == "__main__":
    main()
