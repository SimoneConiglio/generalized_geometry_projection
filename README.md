# Generalized Geometry Projection (GGP) for Additive Manufacturing

[![Tests](https://github.com/SimoneConiglio/generalized_geometry_projection/actions/workflows/tests.yml/badge.svg)](https://github.com/SimoneConiglio/generalized_geometry_projection/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FSimoneConiglio%2Fgeneralized_geometry_projection%2Fgh-pages%2Fbadges%2Fcoverage.json&cacheSeconds=3600)](https://simoneconiglio.github.io/generalized_geometry_projection/)
[![Docs](https://github.com/SimoneConiglio/generalized_geometry_projection/actions/workflows/docs.yml/badge.svg)](https://simoneconiglio.github.io/generalized_geometry_projection/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a modular Python framework for Topology Optimization using the **Generalized Geometry Projection (GGP)** method, built on top of **FEniCS** (via `dolfin` and `petsc4py`) and **GEMSEO**.

This framework focuses on extending structural topology optimization to include Additive Layer Manufacturing (ALM) constraints, specifically focusing on layer-by-layer parameterization and overhang angle restrictions to ensure 3D printability.

## 📚 Citations

This framework is a Python/FEniCS re-implementation and extension of the original MATLAB framework.
- **Original Paper:** Coniglio, S., Morlier, J., Gogu, C. et al. *Generalized Geometry Projection: A Unified Approach for Geometric Feature Based Topology Optimization*. Arch Computat Methods Eng 27, 1573–1610 (2020). [https://doi.org/10.1007/s11831-019-09362-8](https://doi.org/10.1007/s11831-019-09362-8)
- **Original Code:** The foundational logic is heavily inspired by the associated [GGP-Matlab](https://github.com/topggp/GGP-Matlab) repository provided by the original authors.
- **AMJax FEM Solver:** The `amjax` backend implements the algebraic multigrid preconditioning approach introduced by [AMJax](https://github.com/vboussange/AMJax) (Boussange, V.), using PyAMG + SciPy CG.

## 📖 Documentation

Comprehensive documentation, including mathematical background and detailed code examples, is automatically generated using Sphinx and hosted on GitHub Pages:

👉 **[Read the Documentation Here](https://simoneconiglio.github.io/generalized_geometry_projection/)**

## 🚀 Features

- **Config-Driven Optimization:** Every run is fully described by a frozen `ProblemSpec` dataclass, loadable from YAML. Built-in presets cover the standard benchmarks; custom problems only need a YAML file.
- **Self-Registering Extension Points:** New geometry readers (`@register_reader`) and projection mappers (`@register_mapper`) integrate without touching core loops.
- **Modular MDAO Architecture:** `GGPGeometryDiscipline` and `GGPPhysicsDiscipline` are decoupled GEMSEO disciplines chained in an `MDAChain` under an MDF scenario.
- **Vectorized Geometric Mapping:** Pure-NumPy projection mappers for Free (2D/3D) and ALM (2D/3D) provide analytic Jacobians with no FEniCS dependency.
- **FEniCS Physics Solver:** Linear elasticity with SIMP penalization, supporting a direct MUMPS solver (2D) or scalable iterative CG + GAMG (3D/large meshes).
- **Topology Optimization Benchmarks:** Built-in presets for Short Cantilever, MBB Beam, L-Shape Bracket, and ALM Cantilever.
- **Robust Test Coverage:** Unit and regression tests covering 3D domains, iterative solvers, and overhang constraints with **>80% coverage**.

## 🛠️ Installation

The framework relies heavily on FEniCS, which is best installed via Conda.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SimoneConiglio/generalized_geometry_projection.git
   cd generalized_geometry_projection
   ```

2. **Create the Conda Environment:**
   An `environment.yml` is provided to ensure all FEniCS dependencies and scientific libraries are correctly resolved.
   ```bash
   conda env create -f environment.yml
   conda activate ggp
   ```

3. **Verify Installation:**
   ```bash
   pytest tests/
   ```

## 🧠 Running Optimizations via CLI

After installation (`pip install -e .`), the `ggp` command is available directly:

```bash
# List available presets
ggp info --presets

# Run a built-in benchmark preset
ggp optimize --preset short_cantilever
ggp optimize --preset mbb --algorithm SLP --use-line-search
ggp optimize --preset alm_cantilever --max-iter 30

# Run from a custom YAML problem definition
ggp optimize --config my_problem.yaml --volfrac 0.5 --iterative

# Override algorithm or iteration count on any preset
ggp optimize --preset l_shape --algorithm CONLIN --max-iter 100
```

For more details on CLI options, see the [CLI User Guide](https://simoneconiglio.github.io/generalized_geometry_projection/cli_guide.html).

## 🔎 Escaping local minima (global search)

Lagrangian / MMA topology optimization is a local solver on a non-convex landscape, so
single-start runs land in whatever basin the initial component layout sits in. GGP-Topo
ships four state-of-the-art strategies for reaching *improved* local minima —
**continuation/homotopy, multi-start, basin hopping** and **deflation** — implemented in
[`ggp/optimization/global_search.py`](ggp/optimization/global_search.py):

```bash
# Best-of-N from diverse random component layouts
ggp search --preset short_cantilever --strategy multi_start --n-starts 8

# Warm-started sharpness continuation (r_gp ramp)
ggp search --preset short_cantilever --strategy continuation

# Benchmark all strategies vs. the single-start baseline on SC 2D
python benchmarks/sc2d_local_minima.py            # reduced (fast)
python benchmarks/sc2d_local_minima.py --full     # publication fidelity
```

See the [state-of-the-art review and benchmark](docs/local_minima_review.md) for the
techniques, references, and results.

## 🏗️ Architecture

The `ggp/` package is organized into seven layers that separate concerns cleanly:

| Layer | Package | Responsibility |
|---|---|---|
| Problem spec | `ggp/problem/` | Frozen dataclasses (`ProblemSpec`, `GeometrySpec`, `SolverSpec`, …) and YAML loader |
| Geometry I/O | `ggp/geometry/io/` | Self-registering reader registry; `@register_reader("type")` to extend |
| Discretisation | `ggp/discretisation/` | `FEMDiscretiser` → `AnalysisDomain` (FEniCS mesh, BCs, stiffness reference, load vector) |
| Projection mappers | `ggp/projection/` | Pure-NumPy mappers (`Free2DMapper`, `Free3DMapper`, `ALM2DMapper`, `ALM3DMapper`); `@register_mapper("mode")` to extend |
| GEMSEO disciplines | `ggp/gemseo_wrappers/` | `GGPGeometryDiscipline` (wraps mapper) + `GGPPhysicsDiscipline` (FEniCS elasticity) |
| Optimization pipeline | `ggp/optimization/` | `GGPPipeline` wires all layers; `OptimisationResult` captures outputs |
| CLI | `ggp/cli/` | Click commands (`optimize`, `info`); YAML presets in `ggp/cli/presets/` |

**Data flow:**

```
YAML / ProblemSpec
      ↓
GGPPipeline
  ├─ GeometryReader  →  DomainRepresentation
  ├─ FEMDiscretiser  →  AnalysisDomain
  ├─ GGPGeometryDiscipline  (ProjectionMapper)
  ├─ GGPPhysicsDiscipline   (LinearElasticitySolver)
  └─ GEMSEO MDAChain → MDF Scenario → OptimisationResult
```
