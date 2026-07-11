# SCP-Uno: Sequential Convex Programming for GEMSEO

This package implements a robust Sequential Convex Programming (SCP) framework within the GEMSEO environment. It is specifically designed to handle highly non-linear and non-convex problems (like Topology Optimization) by solving a sequence of simplified, convex subproblems.

## Architecture

The framework is built on a hierarchy of abstractions to ensure maintainability and future extensibility:

1.  **`SequentialProgramming` (Base)**: A generic engine for solving problems via a sequence of approximations. It manages the design variable history, outer optimization loop, and convergence criteria. This base can be extended for:
    *   **Sequential Convex Programming (SCP)**
    *   **Surrogate-Based Optimization (SBO)** (with surrogate retraining)
    *   **Trust-Region methods**
2.  **`SequentialConvexProgramming` (Subclass)**: Specializes the engine for cases where the approximations are analytically convex. It introduces a configurable **Inner Solver** interface.
3.  **Algorithms (`MMA`, `CONLIN`, `SLP`, `GE_SBO`)**: Concrete implementations of approximation rules:
    *   **MMA**: Method of Moving Asymptotes (Svanberg 1987) with reciprocal approximations and dynamic asymptote updates. (Captures curvature best).
    *   **CONLIN**: Convex Linearization rule (linear for positive gradients, reciprocal for negative). Incorporates strict move limits to prevent reciprocal singularities.
    *   **SLP**: Sequential Linear Programming. Purely linear approximations of objective and constraints solved exactly at each step via a HiGHS dual-simplex LP solver over a move-limited box.
    *   **GE_SBO**: Gradient-Enhanced Surrogate-Based Optimization (see below).

## Gradient-Enhanced Surrogate-Based Optimization (`GE_SBO`)

`GE_SBO` realizes the SBO extension anticipated by the `SequentialProgramming` design: instead of an analytic convex approximation, each outer iteration fits a **gradient-enhanced kriging (GEK)** surrogate to all samples collected so far — a Gaussian process conditioned on both function *values and gradients* (derivative observations enter the kriging system exactly), so every expensive evaluation contributes `1 + r` observations.

Three features make it practical for expensive, high-dimensional, gradient-available problems such as GGP topology optimization:

*   **Active-subspace dimension reduction**: when the design space is larger than `max_latent_dim`, the dominant subspace `W` (an SVD of the sampled gradients) reduces the surrogate inputs to `z = Wᵀx`. The kriging system size becomes independent of the full dimension `d`, and the subspace is re-identified every iteration so it tracks the optimization trajectory.
*   **Multi-point (batch) acquisition**: every iteration proposes `batch_size` points inside the current trust region — one penalized-exploitation point (surrogate minimum) plus a ladder of lower-confidence-bound points `mean − κⱼ·σ` with geometrically increasing `κⱼ` and a smooth repulsion term that keeps the batch spread out. The batch points are mutually independent, so the expensive model can evaluate them in parallel.
*   **Trust-region model management**: the best batch point (by the same `L1` penalty merit used by the line-search) is accepted or rejected by the classical ratio test between actual and surrogate-predicted merit reduction; the trust-region radius grows/shrinks accordingly, which keeps the method robust when the surrogate is inaccurate.

Inequality constraints are handled by co-kriging every constraint alongside the objective (all outputs share one Cholesky factorization) and penalizing predicted violation inside the acquisition.

```python
scenario.execute(algo_name="GE_SBO", max_iter=200, batch_size=4, max_latent_dim=12)
```

`max_iter` is the total budget of *true model evaluations*; each iteration consumes up to `batch_size` of them. The core engine (`scp_uno.gesbo_core`) is pure NumPy/SciPy and can be used standalone through `gesbo_minimize(evaluate, x0, lb, ub, config)`.

## Transformer Learned Optimizer (`TRANSFORMER_OPT`)

`TRANSFORMER_OPT` is a **learned optimizer that reaches MMA-level performance** on GGP topology optimization. A small per-variable-token transformer (~70k parameters, pure JAX) reads one token per design variable — signed-log gradient magnitudes of objective and constraint, constraint activity, the per-variable asymptote width / previous step / oscillation triplet (the same sufficient statistics MMA's own update rule uses), and bound headroom, all dimensionless — and emits a full-dimension step every iteration. Self-attention across the variable tokens supplies the global coupling that MMA obtains from its dual multiplier, and makes the policy permutation-equivariant and dimension-agnostic.

Training (`scripts/train_transformer_opt.py`) is behaviour cloning of **MMA's step map**: a compact NumPy MMA teacher (`scp_uno/mma_teacher.py`, single-constraint dual bisection with classical oscillation-adaptive asymptotes) is rolled out on synthetic families — most importantly a *toy-SIMP* family `min Σ kᵢ/(xᵢ+ε) s.t. v·x ≤ V` that mirrors the reciprocal/volume structure of compliance problems — and head 0 learns the teacher step in per-variable *asymptote-width units* (O(1) in every convergence regime). Heads 1–3 are far-sighted multi-scale proposals (direction to the task optimum at 1/4/16 move lengths) used by the multi-point batch mode (`eval_heads=k`). Recorded GEMSEO-MMA trajectories of the real problem family (`scripts/collect_ggp_mma_trajectories.py`, random initial layouts) are mixed in to close the domain gap.

**Short cantilever (108 variables, 40 % volume, preset MMA settings as baseline):**

| Evaluations | TRANSFORMER_OPT | MMA (gemseo-mma) |
| :--- | :--- | :--- |
| 200 | 77.4 | 75.7 |
| 320 | **74.6** | ~74.5 (converged reference) |

```python
scenario.execute(algo_name="TRANSFORMER_OPT", max_iter=320)  # requires JAX
```

By default the policy steps sequentially like MMA (`eval_heads=1`, `accept_mode="always"` — MMA never rejects a step; best-so-far bookkeeping guards the reported result). It handles one inequality constraint natively (multiple constraints are reduced to the most active one for the features, while the merit safeguard sees them all).

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
