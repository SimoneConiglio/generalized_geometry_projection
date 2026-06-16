# SCP-Uno: Sequential Convex Programming for GEMSEO

This package implements a robust Sequential Convex Programming (SCP) framework within the GEMSEO environment. It is specifically designed to handle highly non-linear and non-convex problems (like Topology Optimization) by solving a sequence of simplified, convex subproblems.

## Architecture

The framework is built on a hierarchy of abstractions to ensure maintainability and future extensibility:

1.  **`SequentialProgramming` (Base)**: A generic engine for solving problems via a sequence of approximations. It manages the design variable history, outer optimization loop, and convergence criteria. This base can be extended for:
    *   **Sequential Convex Programming (SCP)**
    *   **Surrogate-Based Optimization (SBO)** (with surrogate retraining)
    *   **Trust-Region methods**
2.  **`SequentialConvexProgramming` (Subclass)**: Specializes the engine for cases where the approximations are analytically convex. It introduces a configurable **Inner Solver** interface.
3.  **Algorithms (`MMA`, `CONLIN`)**: Concrete implementations of approximation rules:
    *   **MMA**: Implements the Method of Moving Asymptotes (Svanberg 1987) with reciprocal approximations and dynamic asymptote updates.
    *   **CONLIN**: Implements the Convex Linearization rule (linear for positive gradients, inverse for negative).

## Inner Solvers

The framework supports multiple engines for solving the convex subproblems:
*   **Uno**: High-performance C++ engine for nonlinearly constrained optimization (via `unopy`).
*   **Scipy (SLSQP)**: Reliable Sequential Least Squares Programming.

## Results: The "Rock Solid" State

The framework was validated on a standard GGP (Generalized Geometry Projection) Topology Optimization problem (Short Cantilever).

| Metric | Result |
| :--- | :--- |
| **Algorithm** | MMA |
| **Inner Solver** | Scipy SLSQP |
| **Initial Compliance** | 42,358 |
| **Final Compliance** | **74.5** (Reached target < 75.0) |
| **Volume Fraction** | **39.99%** (Constraint: 40%) |
| **Stability** | Monotonic objective decrease, no numerical explosions. |

## Key Features

*   **Asymptote Safeguards**: Strict enforcement of asymptote distances (default: 0.05) and move limits (default: 0.02) to ensure FEA stability.
*   **Error Trapping**: The `UnoOpt` wrapper includes robust exception handling and clipping to handle potential numerical singularites in the subproblems.
*   **GEMSEO Integration**: Uses standard `OptimizationProblem`, `DesignSpace`, and `MDOFunction` objects for seamless integration with the GEMSEO ecosystem.
