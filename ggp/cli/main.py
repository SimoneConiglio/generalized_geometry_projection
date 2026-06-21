# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""GGP Command-Line Interface.

After ``pip install .`` (or ``pip install -e .``), run::

    ggp --help
    ggp optimize --preset short_cantilever
    ggp optimize --config my_problem.yaml
    ggp info --presets
"""
from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Optional

import click

from ggp.problem.loader import load_problem


# ── Helpers ───────────────────────────────────────────────────────────────────

_PRESETS_DIR = Path(__file__).parent / "presets"


def _list_presets() -> list[str]:
    """Return available preset names (without ``.yaml`` extension)."""
    return sorted(p.stem for p in _PRESETS_DIR.glob("*.yaml"))


def _resolve_preset(name: str) -> Path:
    """Resolve a preset name to its YAML path."""
    path = _PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(_list_presets())
        raise click.BadParameter(
            f"Unknown preset '{name}'. Available: {available}"
        )
    return path


# ── Main CLI Group ────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.2.0", prog_name="ggp")
def cli():
    """GGP — Generalized Geometry Projection for Topology Optimisation."""


# ── optimize ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--config", "-c", "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to a YAML problem definition file.",
)
@click.option(
    "--preset", "-p", "preset_name",
    type=str,
    help="Name of a built-in preset (e.g. short_cantilever, mbb, l_shape, alm_cantilever).",
)
@click.option("--max-iter", type=int, default=None, help="Override max iterations.")
@click.option("--algorithm", type=str, default=None, help="Override algorithm (MMA, SLP, CONLIN).")
@click.option("--volfrac", type=float, default=None, help="Override volume fraction.")
@click.option("--iterative", is_flag=True, default=False, help="Use PETSc iterative solver.")
@click.option("--use-line-search", is_flag=True, default=False, help="Enable line search (SLP/CONLIN).")
def optimize(
    config_path: Optional[Path],
    preset_name: Optional[str],
    max_iter: Optional[int],
    algorithm: Optional[str],
    volfrac: Optional[float],
    iterative: bool,
    use_line_search: bool,
):
    """Run a topology optimisation from a YAML config or built-in preset.

    Examples::

        ggp optimize --preset short_cantilever
        ggp optimize --config problem.yaml --max-iter 100
        ggp optimize --preset mbb --algorithm SLP --use-line-search
    """
    # Resolve source
    if config_path and preset_name:
        raise click.UsageError("Specify --config OR --preset, not both.")
    if not config_path and not preset_name:
        raise click.UsageError("Provide --config <path> or --preset <name>.")

    yaml_path = config_path if config_path else _resolve_preset(preset_name)
    spec = load_problem(yaml_path)

    # Apply CLI overrides via dataclass replacement (frozen → reconstruct)
    if max_iter is not None or algorithm is not None or use_line_search or iterative:
        from dataclasses import replace
        solver_overrides = {}
        if max_iter is not None:
            solver_overrides["max_iter"] = max_iter
        if algorithm is not None:
            solver_overrides["algorithm"] = algorithm
        if use_line_search:
            solver_overrides["use_line_search"] = True
        if iterative:
            solver_overrides["iterative"] = True
        spec = replace(spec, solver=replace(spec.solver, **solver_overrides))

    if volfrac is not None:
        from dataclasses import replace
        spec = replace(spec, volfrac=volfrac)

    # Display summary
    click.echo(click.style("\n═══════════════════════════════════════════════", fg="cyan", bold=True))
    click.echo(click.style("  GGP Topology Optimisation", fg="cyan", bold=True))
    click.echo(click.style("═══════════════════════════════════════════════\n", fg="cyan", bold=True))

    click.echo(f"  Formulation : {spec.formulation.mode}")
    click.echo(f"  Components  : {spec.formulation.num_components}")
    click.echo(f"  Algorithm   : {spec.solver.algorithm}")
    click.echo(f"  Max iter    : {spec.solver.max_iter}")
    click.echo(f"  Vol. frac.  : {spec.volfrac}")
    click.echo(f"  Iterative   : {spec.solver.iterative}")
    click.echo(f"  Line search : {spec.solver.use_line_search}")
    click.echo()

    # Delegate to the legacy runner for now (Phase 2 will replace this)
    _run_legacy(spec)


def _run_legacy(spec):
    """Bridge: convert ProblemSpec → legacy runner call.

    This function will be removed once the full pipeline is wired up.
    """
    from ggp.cli.runners.free_runner import run_main_ggp

    # Extract geometry params
    geom = spec.geometries[0]
    L = geom.params.get("Lx")
    H = geom.params.get("Ly")
    nelx = geom.params.get("nx")
    nely = geom.params.get("ny")

    # Determine bc_type from boundary conditions (heuristic for legacy bridge)
    bc_regions = {bc.region for bc in spec.boundary_conditions}
    if "top" in bc_regions:
        bc_type = "L-shape"
    elif "left" in bc_regions and len(bc_regions) == 1:
        bc_type = "Short_Cantilever"
    else:
        bc_type = "MBB"

    run_main_ggp(
        bc_type=bc_type,
        max_iter=spec.solver.max_iter,
        mode=spec.formulation.mode,
        algorithm=spec.solver.algorithm,
        use_line_search=spec.solver.use_line_search,
        L_opt=L,
        H_opt=H,
        nelx_opt=nelx,
        nely_opt=nely,
        volfrac_opt=spec.volfrac,
        iterative=spec.solver.iterative,
    )


# ── info ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--presets", is_flag=True, help="List available built-in presets.")
@click.option("--mappers", is_flag=True, help="List available projection mappers.")
@click.option("--backends", is_flag=True, help="List available analysis backends.")
def info(presets: bool, mappers: bool, backends: bool):
    """Show available presets, mappers, and analysis backends."""
    if not (presets or mappers or backends):
        presets = mappers = backends = True

    if presets:
        click.echo(click.style("\nAvailable Presets:", fg="green", bold=True))
        for name in _list_presets():
            click.echo(f"  • {name}")

    if mappers:
        click.echo(click.style("\nAvailable Projection Mappers:", fg="green", bold=True))
        try:
            from ggp.projection.registry import list_mappers
            for name in list_mappers():
                click.echo(f"  • {name}")
        except ImportError:
            # Registry not yet implemented
            click.echo("  • Free (2D)")
            click.echo("  • Free (3D)")
            click.echo("  • ALM (2D)")
            click.echo("  • ALM (3D)")

    if backends:
        click.echo(click.style("\nAvailable Analysis Backends:", fg="green", bold=True))
        click.echo("  • FEniCS Direct (LU)")
        click.echo("  • PETSc Iterative (CG + GAMG)")

    click.echo()


# ── Entry point (for python -m ggp.cli.main) ─────────────────────────────────

if __name__ == "__main__":
    cli()
