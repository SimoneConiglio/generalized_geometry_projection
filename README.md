# Generalized Geometry Projection (GGP) for Additive Manufacturing

[![Tests](https://github.com/SimoneConiglio/generalized_geometry_projection/actions/workflows/tests.yml/badge.svg)](https://github.com/SimoneConiglio/generalized_geometry_projection/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FSimoneConiglio%2Fgeneralized_geometry_projection%2Fgh-pages%2Fbadges%2Fcoverage.json&cacheSeconds=3600)](https://simoneconiglio.github.io/generalized_geometry_projection/)
[![Docs](https://github.com/SimoneConiglio/generalized_geometry_projection/actions/workflows/docs.yml/badge.svg)](https://simoneconiglio.github.io/generalized_geometry_projection/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a modular Python framework for Topology Optimization using the **Generalized Geometry Projection (GGP)** method, built on top of **FEniCS** (`dolfin-adjoint`) and **GEMSEO**.

This framework focuses on extending structural topology optimization to include Additive Layer Manufacturing (ALM) constraints, specifically focusing on layer-by-layer parameterization and overhang angle restrictions to ensure 3D printability.

## 📚 Citations

This framework is a Python/FEniCS re-implementation and extension of the original MATLAB framework.
- **Original Paper:** Coniglio, S., Morlier, J., Gogu, C. et al. *Generalized Geometry Projection: A Unified Approach for Geometric Feature Based Topology Optimization*. Arch Computat Methods Eng 27, 1573–1610 (2020). [https://doi.org/10.1007/s11831-019-09362-8](https://doi.org/10.1007/s11831-019-09362-8)
- **Original Code:** The foundational logic is heavily inspired by the associated [GGP-Matlab](https://github.com/topggp/GGP-Matlab) repository provided by the original authors.

## 📖 Documentation

Comprehensive documentation, including mathematical background and detailed code examples, is automatically generated using Sphinx and hosted on GitHub Pages:

👉 **[Read the Documentation Here](https://simoneconiglio.github.io/generalized_geometry_projection/)**

*Note: If the link returns a 404, please ensure that GitHub Pages is enabled in your repository settings (Settings -> Pages -> Build and deployment -> Source: `gh-pages` branch).*

## 🚀 Features

- **Object-Oriented Architecture:** Clean separation of Geometry Mapping (`GGP2DMapper`) and Physics Solvers (`LinearElasticitySolver`) using Factory patterns.
- **High Performance Adjoint Gradients:** Utilizes a monolithic GEMSEO `MacroDiscipline` to leverage `dolfin-adjoint`'s highly efficient analytical chain rule, avoiding massive dense Jacobian assemblies.
- **Topology Optimization Benchmarks:** Includes standard benchmarks out-of-the-box (Short Cantilever, MBB Beam, L-Shape Bracket).

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
   conda activate samo_agents
   ```

3. **Verify Installation:**
   ```bash
   pytest tests/
   ```

## 🧠 Running Examples

Three classical topology optimization examples are provided in the `examples/` directory.

```bash
# Run the Short Cantilever benchmark
python examples/ex01_short_cantilever.py

# Run the MBB Beam benchmark
python examples/ex02_mbb_beam.py

# Run the L-Shape Bracket benchmark
python examples/ex03_l_shape_bracket.py

# Run the ALM Cantilever with Overhang Constraints
python examples/ex04_alm_cantilever.py
```

## 🏗️ Architecture

The code is structured into the `ggp/` package:
- `geometry/`: Contains the `GGP2DMapper` which maps design primitives to a FEniCS density field using Saturated Kreisselmeier-Steinhauser (KS) aggregation and regularized Heaviside functions.
- `physics/`: Contains the `LinearElasticitySolver` with SIMP penalization.
- `gemseo_wrappers/`: Contains the `GGPMacroDiscipline`, bridging the FEniCS world with GEMSEO's optimization algorithms (e.g., `NLOPT_MMA`).
