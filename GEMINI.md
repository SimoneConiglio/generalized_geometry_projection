# SAMO Project: Simulation and Analysis of Manufacturing Operations

This project is a research and development environment focused on **Additive Manufacturing (ALM)** process modeling and **Multidisciplinary Design Analysis and Optimization (MDAO)**. It provides a modular Python framework for Topology Optimization using the Generalized Geometry Projection (GGP) method.

## Project Overview

- **Domain:** Computational Mechanics, Additive Manufacturing, MDAO.
- **Core Package:** `ggp`
- **Core Technologies:**
  - **Simulation:** [FEniCS](https://fenicsproject.org/) (`dolfin`, `dolfin_adjoint`) for finite element analysis and adjoint-based optimization.
  - **Optimization Orchestration:** [GEMSEO](https://gemseo.readthedocs.io/) for multi-disciplinary optimization and MMA (Moving Asymptotes) algorithms.
  - **Environment:** Conda-based Linux/WSL environment for stable FEniCS/MPI management.

## Key Components

- **`ggp.geometry`**: Implements GGP mapping strategies (Free mapping and ALM Layer-by-Layer mapping).
- **`ggp.physics`**: Implements FEniCS-based PDE solvers (Linear Elasticity).
- **`ggp.gemseo_wrappers`**: Contains the `GGPMacroDiscipline`, a monolithic wrapper that leverages `dolfin-adjoint` for highly efficient analytical gradient computation within GEMSEO scenarios.
- **`examples/`**: Provides validated benchmark optimizations (Short Cantilever, MBB, L-Shape, and ALM-constrained designs).
- **`docs/`**: Automated Sphinx documentation with interactive Jupyter Notebook tutorials.

## Development Standards

- **Mathematical Integrity:** All mapping functions must be differentiable. Regularized Heaviside and Saturated KS functions are preferred for smooth assembly of geometric features.
- **Performance:** Performance must be monitored via the `GGPMacroDiscipline`. Avoid modular GEMSEO chains for field-to-field mappings to prevent dense Jacobian bottlenecks.
- **Verification:** Any new feature or mapper must include unit tests in the `tests/` directory. Coverage must be maintained above 80% as enforced by the CI pipeline.
- **Performance Tracking:** Use `python profile_suite.py` to track code speed. Any significant regression (>10%) in the average iteration time should be justified and documented in the performance history.

## Getting Started

Refer to the [Installation Guide](https://simoneconiglio.github.io/generalized_geometry_projection/installation.html) for detailed setup instructions.

```bash
conda env create -f environment.yml
conda activate samo_agents
export PYTHONPATH=$PYTHONPATH:.
pytest tests/
```
