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
@click.option(
    "--init-pattern", type=str, default=None,
    help="Initial-guess pattern: grid, tri2d, quad_star, tet3d, hex_star.",
)
@click.option(
    "--init-cell-size", type=float, default=None,
    help="Cell size for mesh init patterns (drives the component count).",
)
@click.option(
    "--output-dir", "-o", "output_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to save the density plot (PNG) after optimization.",
)
def optimize(
    config_path: Optional[Path],
    preset_name: Optional[str],
    max_iter: Optional[int],
    algorithm: Optional[str],
    volfrac: Optional[float],
    iterative: bool,
    use_line_search: bool,
    init_pattern: Optional[str],
    init_cell_size: Optional[float],
    output_dir: Optional[Path],
):
    """Run a topology optimisation from a YAML config or built-in preset.

    Examples::

        ggp optimize --preset short_cantilever
        ggp optimize --preset short_cantilever --output-dir docs/_static
        ggp optimize --config problem.yaml --max-iter 150
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

    if init_pattern is not None or init_cell_size is not None:
        from dataclasses import replace
        init_overrides = {}
        if init_pattern is not None:
            init_overrides["pattern"] = init_pattern
        if init_cell_size is not None:
            init_overrides["cell_size"] = init_cell_size
        spec = replace(spec, init=replace(spec.init, **init_overrides))

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

    from ggp.optimization.pipeline import GGPPipeline

    pipeline = GGPPipeline(spec)
    result = pipeline.run()

    click.echo(click.style("\n═══════════════════════════════════════════════", fg="green", bold=True))
    click.echo(click.style("  Optimisation Completed", fg="green", bold=True))
    click.echo(click.style("═══════════════════════════════════════════════", fg="green", bold=True))
    click.echo(f"  Status       : {result.status}")
    click.echo(f"  Iterations   : {result.iterations}")
    click.echo(f"  Objective    : {result.objective_value:.6e}")
    click.echo(f"  Time (s)     : {result.execution_time_s:.2f}")
    click.echo()

    if output_dir is not None and result.density_field is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        name = preset_name if preset_name else (config_path.stem if config_path else "result")
        plot_path = output_dir / f"{name}_optimized.png"
        from ggp.visualization.plot import save_density_plot_2d, save_density_plot_3d
        if result.dim == 3:
            save_density_plot_3d(result.density_field, result.eval_coords, plot_path, title=name)
        else:
            save_density_plot_2d(result.density_field, result.eval_coords, plot_path, title=name)
        click.echo(click.style(f"  Density plot : {plot_path}", fg="cyan"))
        click.echo()


# ── info ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--presets", is_flag=True, help="List available built-in presets.")
@click.option("--mappers", is_flag=True, help="List available projection mappers.")
@click.option("--patterns", is_flag=True, help="List available initial-guess patterns.")
@click.option("--backends", is_flag=True, help="List available analysis backends.")
def info(presets: bool, mappers: bool, patterns: bool, backends: bool):
    """Show available presets, mappers, init patterns, and analysis backends."""
    if not (presets or mappers or patterns or backends):
        presets = mappers = patterns = backends = True

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

    if patterns:
        click.echo(click.style("\nAvailable Initial-Guess Patterns:", fg="green", bold=True))
        from ggp.initialization import list_patterns
        click.echo("  • grid  (legacy crossed-bar seed)")
        for name in list_patterns():
            click.echo(f"  • {name}")

    if backends:
        click.echo(click.style("\nAvailable Analysis Backends:", fg="green", bold=True))
        click.echo("  • FEniCS Direct (LU)")
        click.echo("  • PETSc Iterative (CG + GAMG)")

    click.echo()


# ── Entry point (for python -m ggp.cli.main) ─────────────────────────────────

if __name__ == "__main__":
    cli()
