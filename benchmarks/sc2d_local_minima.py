# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Benchmark of improved-local-minima strategies on the Short Cantilever 2D (SC 2D).

Compares the single-start baseline against four global-search strategies
(continuation, multi-start, basin hopping, deflation) on the GGP short-cantilever
problem, and emits a results table (CSV + Markdown), per-method density figures and
a scatter of the local-minima spread.

Run inside the conda ``ggp`` environment::

    python benchmarks/sc2d_local_minima.py                 # reduced breadth (default)
    python benchmarks/sc2d_local_minima.py --full          # more restarts/hops
    python benchmarks/sc2d_local_minima.py --methods baseline multi_start
    python benchmarks/sc2d_local_minima.py --n-starts 8 --seed 1

Every individual local solve uses the *preset's exact MMA configuration* (algorithm,
asymptotes, move limit, 320 iterations) -- a strategy never changes the local solver,
only how the search is seeded / homotopied / repelled. ``--reduced`` vs ``--full``
controls only the *breadth* of the global search (number of restarts, phases and
hops), not the per-run solver setup.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import time
from pathlib import Path

import numpy as np

from ggp.problem.loader import load_problem
from ggp.optimization.pipeline import GGPPipeline
from ggp.optimization import global_search as gs

PRESET = Path(__file__).resolve().parents[1] / "ggp" / "cli" / "presets" / "short_cantilever.yaml"

ALL_METHODS = ["baseline", "continuation", "multi_start", "basin_hopping", "deflated_search"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class BenchConfig:
    # NB: the per-run *local solver* configuration (MMA algorithm, asymptotes,
    # move limit, max_iter) always comes from the preset and is NEVER changed by a
    # strategy -- every individual local solve is exactly the example's setup. The
    # fields below only control the *breadth* of the global search (how many
    # restarts / phases / hops) and the geometry-homotopy schedule.
    n_starts: int
    n_hops: int
    n_solutions: int
    continuation_schedule: list
    hop_step: float
    hop_temperature: float
    deflation_shift: float
    deflation_power: float
    seed: int


def reduced_config(seed: int) -> BenchConfig:
    return BenchConfig(
        n_starts=4,
        n_hops=4,
        n_solutions=3,
        # r_gp continuation: smooth -> sharp, warm-started. The smooth first phase
        # (r_gp=2.0) is what lets the bars reorganise into a better basin before the
        # landscape is sharpened to the target r_gp=0.5 (== baseline) -- this is the
        # phase whose compliance is comparable to the baseline.
        continuation_schedule=[{"r_gp": 2.0}, {"r_gp": 1.5}, {"r_gp": 1.0}, {"r_gp": 0.5}],
        hop_step=0.08,
        hop_temperature=2.0,
        deflation_shift=1.0,
        deflation_power=2.0,
        seed=seed,
    )


def full_config(seed: int) -> BenchConfig:
    return BenchConfig(
        n_starts=8,
        n_hops=8,
        n_solutions=5,
        continuation_schedule=[{"r_gp": 2.0}, {"r_gp": 1.5}, {"r_gp": 1.0}, {"r_gp": 0.5}],
        hop_step=0.06,
        hop_temperature=2.0,
        deflation_shift=1.0,
        deflation_power=2.0,
        seed=seed,
    )


def _load_spec(max_iter=None):
    """Load the SC 2D preset *unchanged* so every local solve uses the example's
    exact MMA configuration (asymptotes, move limit, 320 iters). ``max_iter`` is an
    optional escape hatch for quick local checks only -- the default keeps the
    preset's solver setup verbatim.
    """
    spec = load_problem(str(PRESET))
    if max_iter is not None:
        opts = dict(spec.solver.options)
        opts.pop("max_iter", None)
        spec = dataclasses.replace(
            spec, solver=dataclasses.replace(spec.solver, max_iter=max_iter, options=opts)
        )
    return spec


# --------------------------------------------------------------------------- #
# Strategy runners
# --------------------------------------------------------------------------- #
def run_baseline(spec):
    t0 = time.time()
    result = GGPPipeline(spec).run()
    attempt = gs._attempt_from_result("baseline", result)
    return gs.GlobalSearchResult(
        method="baseline",
        best_result=result,
        best_compliance=gs.compliance_of(result),
        attempts=[attempt],
        total_time_s=time.time() - t0,
    )


def run_method(name: str, spec, cfg: BenchConfig) -> gs.GlobalSearchResult:
    log = lambda *a: print("   ", *a, flush=True)
    if name == "baseline":
        return run_baseline(spec)
    if name == "continuation":
        return gs.continuation(
            spec, cfg.continuation_schedule,
            on_phase=lambda i, r: log(f"phase {i}: C={gs.compliance_of(r):.4g}"),
        )
    if name == "multi_start":
        return gs.multi_start(
            spec, n_starts=cfg.n_starts, seed=cfg.seed,
            on_start=lambda i, r: log(f"start {i}: C={gs.compliance_of(r):.4g}"),
        )
    if name == "basin_hopping":
        return gs.basin_hopping(
            spec, n_hops=cfg.n_hops, step=cfg.hop_step,
            temperature=cfg.hop_temperature, seed=cfg.seed,
            on_hop=lambda i, r, acc: log(
                f"hop {i}: C={gs.compliance_of(r):.4g} {'accept' if acc else 'reject'}"),
        )
    if name == "deflated_search":
        return gs.deflated_search(
            spec, n_solutions=cfg.n_solutions, shift=cfg.deflation_shift,
            power=cfg.deflation_power, seed=cfg.seed,
            on_solution=lambda i, r: log(f"solution {i}: C={gs.compliance_of(r):.4g}"),
        )
    raise ValueError(f"unknown method {name}")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "best_compliance", "n_runs", "total_time_s",
                    "best_iterations", "all_compliances"])
        for r in rows:
            w.writerow([
                r["method"], f"{r['best_compliance']:.6f}", r["n_runs"],
                f"{r['total_time_s']:.1f}", r["best_iterations"],
                ";".join(f"{c:.4f}" for c in r["all_compliances"]),
            ])


def write_markdown(rows, baseline_C, path):
    lines = [
        "| Method | Best compliance | Improvement vs baseline | Runs | Time (s) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        impr = (baseline_C - r["best_compliance"]) / baseline_C * 100.0 if baseline_C else 0.0
        lines.append(
            f"| {r['method']} | {r['best_compliance']:.4f} | {impr:+.2f}% "
            f"| {r['n_runs']} | {r['total_time_s']:.0f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot_spread(rows, baseline_C, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(rows):
        cs = r["all_compliances"]
        ax.scatter([i] * len(cs), cs, alpha=0.5, s=40, color="tab:gray",
                   label="individual runs" if i == 0 else None)
        ax.scatter([i], [r["best_compliance"]], color="tab:red", s=90, zorder=3,
                   marker="*", label="best" if i == 0 else None)
    ax.axhline(baseline_C, ls="--", color="tab:blue", lw=1, label="baseline best")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["method"] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("Compliance C")
    ax.set_title("SC 2D: local-minima spread by strategy (lower is better)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="SC 2D local-minima strategy benchmark")
    p.add_argument("--full", action="store_true", help="publication-fidelity sweep (slow)")
    p.add_argument("--methods", nargs="+", default=ALL_METHODS,
                   choices=ALL_METHODS, help="which strategies to run")
    p.add_argument("--n-starts", type=int, default=None, help="override multi-start count")
    p.add_argument("--max-iter", type=int, default=None,
                   help="escape hatch for quick local checks ONLY; by default every "
                        "local solve uses the preset's exact MMA setup (320 iters).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None,
                   help="default: docs/_static for figures, benchmarks/ for tables")
    args = p.parse_args()

    cfg = full_config(args.seed) if args.full else reduced_config(args.seed)
    if args.n_starts is not None:
        cfg.n_starts = args.n_starts

    root = Path(__file__).resolve().parents[1]
    fig_dir = Path(args.output_dir) if args.output_dir else root / "docs" / "_static"
    tab_dir = Path(args.output_dir) if args.output_dir else root / "benchmarks"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    spec = _load_spec(max_iter=args.max_iter)   # default: preset's exact MMA setup
    print(f"SC 2D local-minima benchmark | breadth={'FULL' if args.full else 'REDUCED'} | "
          f"solver={spec.solver.algorithm} max_iter={spec.solver.max_iter} (from preset) | "
          f"methods={args.methods}", flush=True)

    rows = []
    results = {}
    for name in args.methods:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        out = run_method(name, spec, cfg)
        results[name] = out
        rows.append(dict(
            method=name,
            best_compliance=out.best_compliance,
            n_runs=len(out.attempts),
            total_time_s=out.total_time_s,
            best_iterations=out.best_result.iterations,
            all_compliances=[a.compliance for a in out.attempts],
        ))
        print(out.summary(), flush=True)
        print(f"   ({time.time() - t0:.1f}s)", flush=True)

    baseline_C = next((r["best_compliance"] for r in rows if r["method"] == "baseline"),
                      min(r["best_compliance"] for r in rows))

    write_csv(rows, tab_dir / "sc2d_local_minima_results.csv")
    write_markdown(rows, baseline_C, tab_dir / "sc2d_local_minima_results.md")
    plot_spread(rows, baseline_C, fig_dir / "sc2d_local_minima_spread.png")

    # Per-method best density figures.
    from ggp.visualization.plot import save_density_plot_2d
    for name, out in results.items():
        br = out.best_result
        if br.density_field is not None and br.eval_coords is not None:
            save_density_plot_2d(
                br.density_field, br.eval_coords,
                fig_dir / f"sc2d_local_minima_{name}.png",
                title=f"SC 2D best ({name}): C={out.best_compliance:.3f}",
            )

    print("\n=== SUMMARY ===", flush=True)
    for r in rows:
        impr = (baseline_C - r["best_compliance"]) / baseline_C * 100.0 if baseline_C else 0.0
        print(f"  {r['method']:<16} C={r['best_compliance']:.4f}  ({impr:+.2f}%)  "
              f"{r['total_time_s']:.0f}s", flush=True)
    print(f"\nTables -> {tab_dir}\nFigures -> {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
