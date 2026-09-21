# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Box-subdivision outer approximation on the Short Cantilever 2D (SC 2D).

Drives the GGP short-cantilever problem with
`gemseo-box-subdivision <https://github.com/SimoneConiglio/gemseo-box-subdivision>`_
instead of a single MMA run: the design space is cut into a Cartesian grid of
boxes, a mixed-integer master decides which box to look into, and MMA (the
preset's own local solver) solves the sub-problem inside it.

The GGP design vector is ``num_components x 6`` normalised variables
``[Xc, Yc, L, h, theta, Mc]`` per bar. Subdividing all of them is pointless --
the number of boxes is the Cartesian product -- so the vector is split in two
design-space variables:

``x_split``
    the variables the subdivision resolves (``--subdivide`` selects which
    per-component slots, ``--n-components`` how many components);
``x_free``
    everything else, an ordinary variable of the sub-problem, starting from the
    preset's grid initialisation.

A :class:`_Scatter` discipline re-assembles the two into the ``x_vars`` the
geometry discipline consumes, with a constant (permutation) Jacobian, so the
GGP geometry/physics code is untouched.

Run inside the conda ``ggp`` environment::

    # plumbing check: 2 variables, 2 subdivisions, tiny budgets
    python benchmarks/sc2d_box_subdivision.py --smoke

    # the layout experiment: the centres of 4 bars, quartered
    python benchmarks/sc2d_box_subdivision.py --subdivide centers \\
        --n-components 4 --n-subdivisions 4 --master-max-iter 12

    # the same budget spent on plain MMA, for comparison
    python benchmarks/sc2d_box_subdivision.py --baseline --max-iter 320

Cost is reported both as FE solves and as *distinct designs analysed*. The
second is the one to compare on: the GGP disciplines disable their caches, so a
plain MMA run solves the same system twice per iteration (once for the value,
once for the gradient) while the chain built by the box-subdivision scenario
caches it, and the raw solve counts are therefore not the same unit.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from gemseo import create_design_space
from gemseo.core.discipline.discipline import Discipline

from ggp.discretisation.fem import FEMDiscretiser
from ggp.gemseo_wrappers.geometry_discipline import GGPGeometryDiscipline
from ggp.gemseo_wrappers.physics_discipline import GGPPhysicsDiscipline
from ggp.geometry.io.registry import get_reader
from ggp.optimization.pipeline import GGPPipeline
from ggp.problem.loader import load_problem

PRESET = Path(__file__).resolve().parents[1] / "ggp" / "cli" / "presets" / "short_cantilever.yaml"

VPC = 6
"""Design variables per GGP component: [Xc, Yc, L, h, theta, Mc]."""

SLOTS = {
    "centers": (0, 1),      # Xc, Yc  -- where the bar sits
    "x": (0,),
    "y": (1,),
    "lengths": (2,),        # L
    "thickness": (3,),      # h
    "angles": (4,),         # theta
    "mc": (5,),             # Mc, the mass density of the component
    "layout": (0, 1, 4),    # Xc, Yc, theta -- the pose of the bar
    "all": tuple(range(VPC)),
}
"""Which per-component slots ``--subdivide`` resolves."""

SPLIT = "x_split"
"""The name of the subdivided design-space variable."""

FREE = "x_free"
"""The name of the variable left whole to the sub-problem."""


# --------------------------------------------------------------------------- #
# Problem assembly
# --------------------------------------------------------------------------- #
class _Scatter(Discipline):
    """Re-assemble ``x_split`` and ``x_free`` into the GGP vector ``x_vars``.

    The map is a permutation, so the Jacobian is constant and GEMSEO composes
    the GGP gradients through it unchanged.
    """

    def __init__(self, split_indices: np.ndarray, num_vars: int, x_init: np.ndarray):
        super().__init__(name="Scatter")
        self.split_indices = np.asarray(split_indices, dtype=int)
        self.free_indices = np.setdiff1d(np.arange(num_vars), self.split_indices)
        self.num_vars = num_vars

        n_split = self.split_indices.size
        n_free = self.free_indices.size
        self.input_grammar.update_from_names([SPLIT, FREE])
        self.output_grammar.update_from_names(["x_vars"])
        self.default_input_data = {
            SPLIT: x_init[self.split_indices].copy(),
            FREE: x_init[self.free_indices].copy(),
        }

        eye = np.eye(num_vars)
        self._jac = {
            "x_vars": {
                SPLIT: eye[:, self.split_indices],
                FREE: eye[:, self.free_indices],
            }
        }
        if hasattr(self, "cache"):
            self.cache = None
        if hasattr(self, "cache_type"):
            self.cache_type = Discipline.CacheType.NONE

    def assemble(self, x_split: np.ndarray, x_free: np.ndarray) -> np.ndarray:
        """Return the full ``x_vars`` vector."""
        x = np.empty(self.num_vars)
        x[self.split_indices] = np.asarray(x_split, float).flatten()
        x[self.free_indices] = np.asarray(x_free, float).flatten()
        return x

    def _run(self, input_data=None):
        if input_data is not None:
            self.local_data.update(input_data)
        self.local_data["x_vars"] = self.assemble(
            self.local_data[SPLIT], self.local_data[FREE]
        )

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        self.jac = self._jac


class _History:
    """Every (design, compliance, volume) the run evaluated.

    One entry is one FE solve. A solve is not the same unit on both sides of the
    comparison: the GGP disciplines disable their caches, so a plain MMA run
    re-solves the same system for the gradient, while the chain the
    box-subdivision scenario builds caches it. :attr:`.unique` counts the
    *distinct designs analysed*, which is the same unit for both.
    """

    def __init__(self, geometry: Discipline, physics: Discipline):
        self.x: list[np.ndarray] = []
        self.compliance: list[float] = []
        self.volume: list[float] = []
        self.time: list[float] = []
        self._t0 = time.time()
        original_run = physics._run

        def _run(input_data=None):
            out = original_run(input_data)
            self.x.append(np.asarray(geometry.local_data["x_vars"]).flatten().copy())
            self.compliance.append(float(np.ravel(physics.local_data["compliance"])[0]))
            self.volume.append(float(np.ravel(physics.local_data["volume"])[0]))
            self.time.append(time.time() - self._t0)
            return out

        physics._run = _run

    def __len__(self) -> int:
        return len(self.compliance)

    @property
    def unique(self) -> int:
        """The number of distinct designs analysed."""
        return len({x.round(12).tobytes() for x in self.x})

    def best(self, tolerance: float = 1e-6) -> tuple[float, np.ndarray, int]:
        """Return the best *feasible* compliance, its design and its index."""
        compliance = np.asarray(self.compliance)
        feasible = np.asarray(self.volume) <= tolerance
        if not feasible.any():
            msg = "No feasible design was evaluated."
            raise RuntimeError(msg)

        masked = np.where(feasible, compliance, np.inf)
        index = int(np.argmin(masked))
        return float(compliance[index]), self.x[index], index


def load_spec(max_iter: int | None = None, fem_solver: str = "direct"):
    """Load the SC 2D preset, pinning the FE solver for reproducibility.

    The preset's ``amjax`` backend is an iterative solver whose result is not
    bit-reproducible; ``direct`` solves the same system exactly, which is what a
    comparison between two search strategies needs.
    """
    spec = load_problem(str(PRESET))
    replacements = {"fem_solver": fem_solver}
    if max_iter is not None:
        options = dict(spec.solver.options)
        options.pop("max_iter", None)
        replacements["max_iter"] = max_iter
        replacements["options"] = options

    return dataclasses.replace(
        spec, solver=dataclasses.replace(spec.solver, **replacements)
    )


def build_disciplines(spec):
    """Build the GGP geometry and physics disciplines of the SC 2D problem.

    This is what :meth:`GGPPipeline.run` does before it creates its scenario,
    restricted to the Free 2D case the preset describes.

    Returns:
        The geometry discipline, the physics discipline, the normalised starting
        point of the preset, and the dimensions of the domain.
    """
    geometry_spec = spec.geometries[0]
    domain = get_reader(geometry_spec.type).read(geometry_spec)
    analysis = FEMDiscretiser().discretise(domain, spec)

    Lx = domain.metadata.get("Lx", 1.0)
    Ly = domain.metadata.get("Ly", 1.0)

    geometry = GGPGeometryDiscipline(
        mesh=analysis.mesh,
        num_components=spec.formulation.num_components,
        mode=spec.formulation.mode,
        ka=10.0,
        pp=100.0,
        method=spec.formulation.method,
        r_gp=spec.formulation.r_gp,
        Ngp=spec.formulation.Ngp,
        min_thickness=spec.formulation.min_thickness,
    )

    fixed_dofs = list(analysis.point_fixed_dofs)
    for bc in analysis.bcs_applied:
        fixed_dofs.extend(bc.get_boundary_values().keys())

    physics = GGPPhysicsDiscipline(
        V_u=analysis.function_spaces["u"],
        ke_ref=analysis.ke_ref,
        fixed_dofs=sorted(set(fixed_dofs)),
        f_vec=analysis.load_vector,
        mesh_area=Lx * Ly,
        volfrac=spec.volfrac,
        fem_solver=spec.solver.fem_solver,
        p_penalty=1.0,     # GP characteristic function: linear stiffness
        Emin=0.0,          # matches the Free GP reference (no stiffness floor)
        E0=1.0,
    )

    num_vars = VPC * spec.formulation.num_components
    x_init = GGPPipeline._make_init(
        spec.formulation.mode,
        num_vars,
        Lx=Lx,
        Ly=Ly,
        lb=geometry.lb,
        ub=geometry.ub,
        init=spec.formulation.init,
        grid_nx=spec.formulation.grid_nx,
        grid_ny=spec.formulation.grid_ny,
        volfrac=spec.volfrac,
    )
    return geometry, physics, x_init, (Lx, Ly)


def split_indices(slots: str, n_components: int, num_components: int) -> np.ndarray:
    """Return the indices of ``x_vars`` the subdivision resolves."""
    components = range(min(n_components, num_components))
    return np.array(
        [c * VPC + slot for c in components for slot in SLOTS[slots]], dtype=int
    )


# --------------------------------------------------------------------------- #
# The two runs being compared
# --------------------------------------------------------------------------- #
def run_box_subdivision(spec, args) -> dict:
    """Solve SC 2D with the box-subdivision outer approximation."""
    from gemseo_box_subdivision import BoxSubdivisionScenario
    from gemseo_box_subdivision.diagnostics import read_margin_report
    from gemseo_box_subdivision.settings import BoxSubdivisionSettings
    from gemseo_box_subdivision.settings import SweptBoxSubdivisionSettings

    geometry, physics, x_init, _ = build_disciplines(spec)
    history = _History(geometry, physics)

    indices = split_indices(args.subdivide, args.n_components, spec.formulation.num_components)
    scatter = _Scatter(indices, x_init.size, x_init)

    design_space = create_design_space()
    design_space.add_variable(
        SPLIT,
        size=indices.size,
        lower_bound=0.0,
        upper_bound=1.0,
        value=x_init[indices],
    )
    design_space.add_variable(
        FREE,
        size=x_init.size - indices.size,
        lower_bound=0.0,
        upper_bound=1.0,
        value=x_init[scatter.free_indices],
    )

    sub_problem_settings = dict(spec.solver.options)
    sub_problem_settings.pop("max_iter", None)
    common = {
        "trust_region_radius": args.trust_region_radius,
        "n_parallel_points": args.n_parallel_points,
        "max_iter": args.master_max_iter,
        "sub_problem_max_iter": args.sub_max_iter,
        "sub_problem_algo_name": args.sub_algo,
        "sub_problem_algo_settings": sub_problem_settings,
    }
    if args.convexity is None:
        settings = SweptBoxSubdivisionSettings(**common)
    else:
        settings = BoxSubdivisionSettings(convexity_margin=args.convexity, **common)

    scenario = BoxSubdivisionScenario(
        [scatter, geometry, physics],
        "compliance",
        design_space,
        n_subdivisions={SPLIT: args.n_subdivisions},
        formulation=args.formulation,
        settings=settings,
    )
    # A box may hold no design meeting the volume constraint; declaring it at the
    # main level turns such a sub-problem into a feasibility cut for the master
    # instead of a stalled iteration.
    scenario.formulation.add_constraint(
        "volume", constraint_type="ineq", value=0.0, main_level=True
    )

    start = time.time()
    scenario.execute()
    elapsed = time.time() - start

    # What the master made of the boxes it solved: how many were admitted, how
    # many cut on feasibility, and the spread of the objective over them, which
    # is the scale the swept convexity reads.
    report = read_margin_report(scenario.formulation.optimization_problem)

    compliance, x_best, index = history.best()
    return {
        "method": (
            f"box[{args.subdivide} of {args.n_components} bars, "
            f"k={args.n_subdivisions}, {args.formulation}, "
            f"{args.sub_max_iter} it/box]"
        ),
        "settings": {
            "subdivide": args.subdivide,
            "n_components": args.n_components,
            "n_subdivisions": args.n_subdivisions,
            "formulation": args.formulation,
            "sub_problem_max_iter": args.sub_max_iter,
            "sub_problem_algo": args.sub_algo,
            "master_max_iter": args.master_max_iter,
            "n_parallel_points": args.n_parallel_points,
            "trust_region_radius": args.trust_region_radius,
            "convexity_margin": args.convexity,
        },
        "compliance": compliance,
        "evaluations": len(history),
        "unique_designs": history.unique,
        "evaluations_to_best": index + 1,
        "time_s": elapsed,
        "x_best": x_best.tolist(),
        "n_subdivided_variables": int(indices.size),
        "n_boxes": float(args.n_subdivisions) ** int(indices.size),
        "boxes_solved": report.n_solved,
        "boxes_admitted": report.n_feasible,
        "objective_spread_over_boxes": report.spread,
        "margin_report": report.describe(),
        "history": history.compliance,
        "volume_history": history.volume,
    }


def run_baseline(spec) -> dict:
    """Solve SC 2D with the preset's single MMA run, for comparison."""
    geometry, physics, x_init, _ = build_disciplines(spec)
    history = _History(geometry, physics)

    from gemseo import create_scenario

    design_space = create_design_space()
    design_space.add_variable(
        "x_vars", size=x_init.size, lower_bound=0.0, upper_bound=1.0, value=x_init
    )
    scenario = create_scenario(
        [geometry, physics],
        objective_name="compliance",
        design_space=design_space,
        formulation_name="MDF",
    )
    scenario.add_constraint("volume", constraint_type="ineq", positive=False, value=0.0)

    options = dict(spec.solver.options)
    options.setdefault("max_iter", spec.solver.max_iter)
    options.setdefault("algo_name", spec.solver.algorithm)

    start = time.time()
    scenario.execute(**options)
    elapsed = time.time() - start

    compliance, x_best, index = history.best()
    return {
        "method": f"MMA, {options['max_iter']} iterations",
        "settings": {"max_iter": options["max_iter"], "algo": options["algo_name"]},
        "compliance": compliance,
        "evaluations": len(history),
        "unique_designs": history.unique,
        "evaluations_to_best": index + 1,
        "time_s": elapsed,
        "x_best": x_best.tolist(),
        "history": history.compliance,
        "volume_history": history.volume,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def save_outputs(result: dict, spec, out_dir: Path, tag: str) -> None:
    """Write the run's record and the density field of its best design."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2))

    geometry, _, _, _ = build_disciplines(spec)
    geometry.execute({"x_vars": np.asarray(result["x_best"])})
    density = np.asarray(geometry.local_data["rho_E"]).flatten()

    from ggp.visualization.plot import save_density_plot_2d

    save_density_plot_2d(
        density,
        geometry.eval_coords,
        out_dir / f"{tag}.png",
        title=f"{result['method']} -- C = {result['compliance']:.4f}",
    )


def summarize(out_dir: Path) -> str:
    """Collect the runs written to *out_dir* into a table, and plot their traces."""
    records = [json.loads(path.read_text()) for path in sorted(out_dir.glob("*.json"))]
    records.sort(key=lambda record: record["compliance"])

    lines = [
        "| Run | Best feasible compliance | Distinct designs | FE solves | Time (s) |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['method']} | {r['compliance']:.4f} | {r.get('unique_designs', '--')} "
        f"| {r['evaluations']} | {r['time_s']:.0f} |"
        for r in records
    ]
    table = "\n".join(lines)
    (out_dir / "summary.md").write_text(table + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(9, 5))
    for record in records:
        compliance = np.asarray(record["history"])
        volume = np.asarray(record["volume_history"])
        best = np.minimum.accumulate(np.where(volume <= 1e-6, compliance, np.inf))
        # The traces are recorded per FE solve, and a run spends a fixed number
        # of solves per design it analyses: two for plain MMA, which re-solves
        # for the gradient, one for a box run, whose chain caches. Dividing by
        # that ratio puts every trace on the axis the runs share.
        per_design = len(record["history"]) / record.get("unique_designs", len(record["history"]))
        designs = np.arange(1, best.size + 1) / per_design
        axes.plot(designs, best, label=record["method"])

    axes.set_xlabel("distinct designs analysed")
    axes.set_ylabel("best feasible compliance")
    axes.set_yscale("log")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "convergence.png", dpi=150)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", action="store_true",
                        help="collect the runs already written to --out into a table")
    parser.add_argument("--baseline", action="store_true", help="run plain MMA instead")
    parser.add_argument("--smoke", action="store_true", help="tiny budgets, plumbing check")
    parser.add_argument("--subdivide", choices=sorted(SLOTS), default="centers")
    parser.add_argument("--n-components", type=int, default=4,
                        help="how many components have their variables subdivided")
    parser.add_argument("--n-subdivisions", type=int, default=4)
    parser.add_argument("--formulation", choices=("normalized", "constraint"),
                        default="normalized",
                        help="how a sub-problem is confined to its box: rewritten in "
                             "the normalized variables of the box, or bounded by a "
                             "constraint, which starts it from the current design "
                             "projected into the box rather than from the box centre")
    parser.add_argument("--trust-region-radius", type=int, default=2)
    parser.add_argument("--n-parallel-points", type=int, default=2)
    parser.add_argument("--master-max-iter", type=int, default=12)
    parser.add_argument("--sub-max-iter", type=int, default=80)
    parser.add_argument("--sub-algo", default="MMA")
    parser.add_argument("--convexity", type=float, default=None,
                        help="convexity margin; omitted, the master sweeps a ladder")
    parser.add_argument("--max-iter", type=int, default=None, help="baseline MMA iterations")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/box_subdivision"))
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    if args.summarize:
        print(summarize(args.out))
        return

    if args.smoke:
        args.subdivide = "centers"
        args.n_components = 1
        args.n_subdivisions = 2
        args.master_max_iter = 2
        args.n_parallel_points = 1
        args.sub_max_iter = 5
        args.max_iter = args.max_iter or 5

    spec = load_spec(max_iter=args.max_iter)
    result = run_baseline(spec) if args.baseline else run_box_subdivision(spec, args)

    tag = args.tag or ("baseline" if args.baseline else
                       f"{args.subdivide}_{args.n_components}c_k{args.n_subdivisions}")
    save_outputs(result, spec, args.out, tag)

    print("\n" + "=" * 62)
    print(f"  {result['method']}")
    print("=" * 62)
    print(f"  Best feasible compliance : {result['compliance']:.6f}")
    print(f"  FE solves                : {result['evaluations']}")
    print(f"  Distinct designs         : {result['unique_designs']}")
    print(f"  ... to reach the best    : {result['evaluations_to_best']}")
    print(f"  Time (s)                 : {result['time_s']:.1f}")
    if "n_boxes" in result:
        print(f"  Subdivided variables     : {result['n_subdivided_variables']}")
        print(f"  Boxes                    : {result['n_boxes']:.3g}")
        print(f"  Master                   : {result['margin_report']}")
    print(f"  Written to               : {args.out}/{tag}.json|png")


if __name__ == "__main__":
    main()
