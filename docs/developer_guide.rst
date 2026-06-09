Developer Guide
===============

This guide outlines the architecture and design decisions behind the `samo_ggp` framework.

Architecture Overview
---------------------

The `samo_ggp` framework uses a highly modular Object-Oriented design integrated with **GEMSEO** for MDAO (Multidisciplinary Design Analysis and Optimization).

### Package Structure

- **`samo_ggp.geometry`**: Contains mapping logic. The `GGP2DMapper` takes discrete primitives (e.g., coordinates, widths, angles) and projects them into a continuous FEniCS density field ($\rho$) using regularized Heaviside and Saturated KS functions.
- **`samo_ggp.physics`**: Contains PDE solvers. The `LinearElasticitySolver` solves the forward mechanics problem and computes the compliance objective.
- **`samo_ggp.gemseo_wrappers`**: Contains the GEMSEO Discipline wrappers.

The Monolithic "Macro-Discipline"
---------------------------------

Early benchmarking revealed that separating the Geometry Mapping and Physics into a modular GEMSEO `MDAChain` led to a severe bottleneck. GEMSEO requires explicitly sized dense Jacobian matrices for chain rule computation. Generating the Jacobian of a full mesh density field with respect to primitive variables ($\frac{\partial \rho}{\partial x}$) is highly inefficient.

Instead, we use a single **`GGPMacroDiscipline`**. This discipline executes the mapper and the solver sequentially within the same `dolfin-adjoint` tape. FEniCS-adjoint can then compute the analytical gradient of the scalar objective directly with respect to the design variables ($\frac{\partial J}{\partial x}$), bypassing the dense density Jacobian entirely.

### Tape Management

To ensure `dolfin-adjoint` does not leak memory or track disconnected variables across multiple GEMSEO iterations, the `_run()` method in the Macro-Discipline follows a strict protocol:

1. **Clear Tape**: `get_working_tape().clear_tape()` is called at the beginning of each iteration.
2. **Fresh Constants**: `dolfin_adjoint.Constant` objects are instantiated *fresh* inside `_run()` based on the current GEMSEO design variables.

Adding New Physics or Geometries
--------------------------------

To extend the framework:
1. Create a new class inheriting from `BaseMapper` or `BaseSolver`.
2. Register it in `GeometryFactory` or `PhysicsFactory`.
3. Instantiate the `GGPMacroDiscipline` passing your new mapper and solver. No modifications to the GEMSEO wrapper are required.
