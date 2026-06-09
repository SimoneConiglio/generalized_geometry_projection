# Generalized Geometry Projection (GGP) for Additive Manufacturing

This repository contains a modular Python framework for Topology Optimization using the **Generalized Geometry Projection (GGP)** method, built on top of **FEniCS** (`dolfin-adjoint`) and **GEMSEO**.

This framework focuses on extending structural topology optimization to include Additive Layer Manufacturing (ALM) constraints, specifically focusing on layer-by-layer parameterization and overhang angle restrictions to ensure 3D printability.

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
python examples/01_short_cantilever.py

# Run the MBB Beam benchmark
python examples/02_mbb_beam.py

# Run the L-Shape Bracket benchmark
python examples/03_l_shape_bracket.py
```

## 🏗️ Architecture

The code is structured into the `samo_ggp/` package:
- `geometry/`: Contains the `GGP2DMapper` which maps design primitives to a FEniCS density field using Saturated Kreisselmeier-Steinhauser (KS) aggregation and regularized Heaviside functions.
- `physics/`: Contains the `LinearElasticitySolver` with SIMP penalization.
- `gemseo_wrappers/`: Contains the `GGPMacroDiscipline`, bridging the FEniCS world with GEMSEO's optimization algorithms (e.g., `NLOPT_MMA`).
