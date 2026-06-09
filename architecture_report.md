# Architecture Benchmark Report

## Phase C & D: Macro-Discipline vs. Modular Chain

This report concludes the benchmarking phase comparing the two proposed GEMSEO integration architectures for the GGP Additive Manufacturing framework.

### 1. Macro-Discipline (Monolithic) - **SELECTED**
**Implementation:** `samo_ggp/gemseo_wrappers/macro_discipline.py`

This approach wraps both the Geometry Mapping (`GGP2DMapper`) and the Physics Solver (`LinearElasticitySolver`) into a single GEMSEO `Discipline`. 

**Results:**
- **Performance:** Highly performant. The first iteration takes longer due to FFC Just-In-Time (JIT) compilation, but subsequent iterations solve the PDE and compute exact gradients in fractions of a second.
- **Tape Management:** We successfully solved the `dolfin-adjoint` shadowing issues by explicitly initializing FEniCS-adjoint `Constant` and `RectangleMesh` objects, and clearing the working tape at the start of every `_run()` execution to prevent memory leaks.
- **Gradient Computation:** `dolfin-adjoint` calculates the exact coupled derivatives ($\frac{\partial J}{\partial x}$) analytically using the chain rule over the computational graph, entirely bypassing the need to compute the intermediate density Jacobian.

### 2. Modular Sub-Discipline Chain - **REJECTED**
**Implementation:** `samo_ggp/gemseo_wrappers/modular_disciplines.py`

This approach attempts to split the process into a `GGPGeometryDiscipline` (Outputs Density) and a `GGPPhysicsDiscipline` (Inputs Density, Outputs Compliance).

**Findings:**
- **The "Dense Jacobian" Bottleneck:** In a modular GEMSEO `MDAChain`, GEMSEO requires the explicit Jacobian of each discipline to apply the chain rule. This means the `GGPGeometryDiscipline` must return the derivative of every element's density with respect to every design variable ($\frac{\partial \rho_j}{\partial x_i}$).
- **Dimensionality Explosion:** For a modest $50 \times 50$ mesh (2500 elements) and 50 design variables, this results in a dense $2500 \times 50$ Jacobian matrix being passed through the GEMSEO memory space every iteration.
- **Conclusion:** Modularizing the physics from the geometry mapping breaks the inherent efficiency of the `dolfin-adjoint` tape, which is designed to compute a scalar functional's derivative directly without assembling intermediate dense Jacobians. 

## Next Steps
We will proceed exclusively with the **Macro-Discipline** architecture for integrating Additive Layer Manufacturing (ALM) overhang constraints.
