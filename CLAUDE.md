# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **GGP-Topo** (`ggp-topo`), a Python/FEniCS framework for structural Topology Optimization using the **Generalized Geometry Projection (GGP)** method. It supports 2D/3D Free and Additive Layer Manufacturing (ALM) formulations, driven by GEMSEO for gradient-based optimization (MMA, SLP, CONLIN).

## Environment & Installation

FEniCS 2019.1.0 **must** be installed via Conda — it cannot be installed with pip.

```bash
conda env create -f environment.yml
conda activate ggp
pip install -e .          # editable install for development
```

The `environment.yml` pins `fenics=2019.1.0`, `dolfin-adjoint=2019.1.0`, and installs `gemseo-mma` via pip.

### Bootstrapping conda in a fresh sandbox (Claude Code remote sessions)

Fresh containers have no conda. Install **Miniforge** and build the `ggp`
environment from `environment.yml` — activating it is what provides GEMSEO
(and FEniCS, dolfin-adjoint, gemseo-mma):

```bash
curl -sSL -o /tmp/miniforge.sh \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash /tmp/miniforge.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh

conda env create -f environment.yml   # ~5-10 min (FEniCS stack)
conda activate ggp
pip install -e .                      # editable install; registers the scp_uno GEMSEO plugin
```

If the sandbox's network policy blocks the GitHub-hosted Miniforge installer
(403 from the proxy), fall back to Miniconda restricted to conda-forge —
functionally equivalent to Miniforge:

```bash
curl -sSL -o /tmp/miniconda.sh \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
bash /tmp/miniconda.sh -b -p ~/miniconda3
printf 'channels:\n  - conda-forge\n' > ~/miniconda3/.condarc   # drop 'defaults' (avoids the Anaconda ToS gate)
source ~/miniconda3/etc/profile.d/conda.sh
# if environment.yml still lists 'defaults', create from a conda-forge-only copy:
grep -v '^  - defaults$' environment.yml > /tmp/environment_forge.yml
conda env create -f /tmp/environment_forge.yml
conda activate ggp
pip install -e .
```

Behind the sandbox HTTPS proxy, point conda at the proxy CA bundle before
creating the environment: `conda config --set ssl_verify /root/.ccr/ca-bundle.crt`
(pip already honours `PIP_CERT`). **Every test/CLI command below assumes the
`ggp` environment is active** (`conda activate ggp`).

## Commands

```bash
# Run all tests (coverage threshold: 80%)
pytest tests/

# Run a single test file
pytest tests/test_architecture.py

# Run a single test by name
pytest tests/test_architecture.py::test_problem_spec_defaults

# Run tests with coverage report
pytest --cov=ggp --cov-report=json tests/

# CLI — list available presets
ggp info --presets

# CLI — run a built-in preset
ggp optimize --preset short_cantilever
ggp optimize --preset mbb --algorithm SLP --use-line-search
ggp optimize --preset alm_cantilever --max-iter 30

# CLI — run from a YAML config file
ggp optimize --config my_problem.yaml --volfrac 0.5 --iterative
```

Built-in presets: `short_cantilever`, `mbb`, `l_shape`, `alm_cantilever` (YAML files in `ggp/cli/presets/`).

## Architecture

### Data Flow

```
YAML / ProblemSpec
      ↓
GGPPipeline (ggp/optimization/pipeline.py)
      ├─ GeometryReader registry  → DomainRepresentation
      ├─ FEMDiscretiser           → AnalysisDomain (mesh, BCs, ke_ref, f_vec)
      ├─ GGPGeometryDiscipline   ← ProjectionMapper (via registry)
      ├─ GGPPhysicsDiscipline    (FEniCS LinearElasticitySolver)
      └─ GEMSEO MDAChain → create_scenario (MDF) → OptimisationResult
```

### Key Modules

**`ggp/problem/`** — Frozen dataclasses (`ProblemSpec`, `GeometrySpec`, `FormulationSpec`, `SolverSpec`, …). These are the single source of truth for any run; the CLI and YAML loader both produce a `ProblemSpec`. All fields are immutable; use `dataclasses.replace()` to derive variants.

**`ggp/geometry/io/`** — Self-registering geometry reader registry. To add a new mesh source, subclass `GeometryReader` and decorate with `@register_reader("my_type")`. No changes elsewhere needed.

**`ggp/projection/`** — Self-registering projection mapper registry. Concrete mappers (`Free2DMapper`, `Free3DMapper`, `ALM2DMapper`, `ALM3DMapper`) implement `ProjectionMapper.forward()` and `ProjectionMapper.jacobian()` — pure NumPy, no FEniCS. To add a new formulation, subclass `ProjectionMapper` and decorate with `@register_mapper("My_Mode")`.

**`ggp/discretisation/fem.py`** — `FEMDiscretiser` converts a `DomainRepresentation` into an `AnalysisDomain` containing the FEniCS mesh, function spaces, applied BCs, load vector, and reference element stiffness matrix (`ke_ref`).

**`ggp/gemseo_wrappers/`** — Two geometry discipline implementations exist:
- `GGPGeometryDiscipline` (in `geometry_discipline.py`) — **current/preferred**: lightweight GEMSEO wrapper that delegates to a `ProjectionMapper` from the registry.
- `GGPVectorizedGeometryDiscipline` (in `modular_disciplines.py`) — **legacy**: monolithic implementation with inline mapping logic. Still used by some older runners.

**`ggp/physics/`** — `LinearElasticitySolver` uses FEniCS + SIMP penalization (`E = Emin + rho^p * (E0 - Emin)`). Supports a direct MUMPS solver (default) or iterative CG + GAMG/hypre_amg for large 3D problems (`--iterative` flag).

**`ggp/optimization/pipeline.py`** — `GGPPipeline.run()` wires everything together and returns an `OptimisationResult`. This is the canonical entry point for programmatic use.

**`ggp/utils/`** — Vectorized NumPy implementations of the geometric mapping math (`vectorized_mapping.py`, `vectorized_mapping_3d.py`), KS aggregation (`Aggregation_Pi.py`), ALM overhang constraints (`alm_utils.py`), and mathematical helpers.

### Design Conventions

- **Registry pattern** throughout: geometry readers, projection mappers, and CLI presets are all self-registering via decorator. Extending any of these never requires touching core loops.
- **`ProblemSpec` is frozen** (`@dataclass(frozen=True)`); use `dataclasses.replace(spec, solver=replace(spec.solver, max_iter=100))` for overrides.
- **Projection mappers are pure functions** (no FEniCS, no side effects) — this is what enables future meshless backends.
- **GEMSEO MDF formulation**: geometry and physics are separate `Discipline` subclasses chained in an `MDAChain`; the optimizer drives `x_vars → rho_E, rho_V → compliance, volume`.
