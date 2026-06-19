# SCP-Uno: Sequential Convex Programming for GEMSEO

This package implements a robust Sequential Convex Programming (SCP) framework within the GEMSEO environment. It is specifically designed to handle highly non-linear and non-convex problems (like Topology Optimization) by solving a sequence of simplified, convex subproblems.

## Architecture

The framework is built on a hierarchy of abstractions to ensure maintainability and future extensibility:

1.  **`SequentialProgramming` (Base)**: A generic engine for solving problems via a sequence of approximations. It manages the design variable history, outer optimization loop, and convergence criteria. This base can be extended for:
    *   **Sequential Convex Programming (SCP)**
    *   **Surrogate-Based Optimization (SBO)** (with surrogate retraining)
    *   **Trust-Region methods**
2.  **`SequentialConvexProgramming` (Subclass)**: Specializes the engine for cases where the approximations are analytically convex. It introduces a configurable **Inner Solver** interface.
3.  **Algorithms (`MMA`, `CONLIN`, `SLP`)**: Concrete implementations of approximation rules:
    *   **MMA**: Method of Moving Asymptotes (Svanberg 1987) with reciprocal approximations and dynamic asymptote updates. (Captures curvature best).
    *   **CONLIN**: Convex Linearization rule (linear for positive gradients, reciprocal for negative). Incorporates strict move limits to prevent reciprocal singularities.
    *   **SLP**: Sequential Linear Programming. Purely linear approximations of objective and constraints solved exactly at each step via a HiGHS dual-simplex LP solver over a move-limited box.

## Monotone Backtracking Line-Search

A key feature available to *all* algorithms in this framework is an optional **Monotone Backtracking Line-Search**. Enabled via `use_line_search=True` in the settings, this mechanism:
1. Evaluates the $L_1$ penalty merit function: $\phi(x) = f(x) + \mu \sum_i \max(0, c_i(x))$
2. Checks if taking the proposed step $x_{k+1}$ provides an Armijo sufficient decrease in $\phi(x)$.
3. If not, it backtracks along the direction $d = x_{k+1} - x_k$ (e.g. trying $0.5d$, $0.25d$, etc.) until a step that reduces the merit function is found.

This acts as a powerful stabilizing safeguard—completely eliminating the wild objective spikes and severe constraint violations that can occur when approximations (like pure SLP or CONLIN) become too optimistic.

## Inner Solvers

The framework supports multiple engines for solving the convex subproblems:
*   **HiGHS (Dual Simplex / Interior Point)**: Exact LP solver utilized natively by SLP.
*   **Scipy (SLSQP)**: Reliable Sequential Least Squares Programming.
*   **Uno**: High-performance C++ engine for nonlinearly constrained optimization (via `unopy`).

## Results: Validation on GGP Benchmark

The framework was validated on a standard GGP (Generalized Geometry Projection) Topology Optimization problem (Short Cantilever) targeting a volume fraction of 40%.

| Metric | MMA | CONLIN | SLP | SLP + Line-Search |
| :--- | :--- | :--- | :--- | :--- |
| **Inner Solver** | Scipy SLSQP | Scipy SLSQP | HiGHS DUAL_SIMPLEX | HiGHS DUAL_SIMPLEX |
| **Initial Compliance**| 42,358 | 42,358 | 42,358 | 42,358 |
| **Final Compliance** | **74.5** | 126.75 | 113.23 | 552.37 |
| **Final Volume** | 39.99% | 38.3% | 43.0% | 41.6% |
| **Behavior** | Smooth monotonic descent | Fast drop, then chatters | Fast drop, severe spikes | No spikes, gets stuck in local minima |

> **Note**: While SLP with line-search safely eliminates spikes, it terminates at a higher compliance because the purely linear subproblems often direct the optimizer exactly along constraint boundaries where no strictly linearly-feasible descent direction exists. MMA naturally avoids this by adding curvature via asymptotes, steering iterates efficiently away from boundaries.

## Key Features

*   **Move Limits & Asymptote Safeguards**: Strict enforcement of asymptote distances and move limits (default: 0.05 and 0.02) to ensure FEA stability and prevent singular inverse terms.
*   **Merit-based Stabilization**: The $L_1$ penalty line-search guarantees monotonic progress even when approximations fail.
*   **GEMSEO Integration**: Uses standard `OptimizationProblem`, `DesignSpace`, and `MDOFunction` (and `MDOLinearFunction` for SLP) objects for seamless integration with the GEMSEO ecosystem.
