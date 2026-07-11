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

`TRANSFORMER_OPT` replaces the hand-designed acquisition with a **learned proposer**: a small set-transformer policy (~110k parameters, pure JAX) reads the recent optimization history — sample positions, merit values and gradients, encoded in the same gradient active subspace used by `GE_SBO` — and directly emits the next **batch** of query points (one per output head). A classical trust-region accept/shrink loop safeguards every step, so a bad proposal costs one batch, not the run.

The policy is trained offline (`scripts/train_transformer_opt.py`, a few CPU-minutes) by behaviour cloning of a *privileged teacher* on synthetic tasks (anisotropic quadratics, two-well multimodal functions, curved valleys) of random dimension: the teacher knows each task's optimum and the winner-takes-all loss over the output heads lets heads specialize on distinct basins. Because all features are trust-region-relative, merit-scale-normalized and live in a fixed-size latent, **one trained model transfers across dimensions and objective scales** — the packaged default weights (`scp_uno/weights/transformer_opt_default.npz`) were trained only on toy functions yet run unchanged on the 108-variable GGP cantilever.

```python
scenario.execute(algo_name="TRANSFORMER_OPT", max_iter=200)  # requires JAX
```

Compared with `GE_SBO`: no kriging system to factorize (inference is one forward pass), the same multi-point batch structure, but model quality depends on the training distribution rather than on principled uncertainty — treat it as the experimental, research-grade option of the family.

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
