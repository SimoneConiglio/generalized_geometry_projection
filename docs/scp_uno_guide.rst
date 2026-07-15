Sequential Convex Programming (SCP) Framework
=============================================

The ``scp_uno`` package implements a robust Sequential Convex Programming framework integrated with the GEMSEO environment. It is tailored for highly non-linear and non-convex problems such as Topology Optimization.

Overview
--------

The core idea of SCP is to replace a difficult, non-convex optimization problem with a sequence of simpler, convex approximations (subproblems). Each subproblem is solved, the design variables are updated, and a new approximation is constructed until convergence is reached.

Algorithms
----------

The framework provides seven major algorithms:

1. **Method of Moving Asymptotes (MMA)**:
   The gold standard for topology optimization. It builds a strictly convex approximation by introducing reciprocal variables with dynamically updated asymptotes. This naturally injects curvature, safely preventing reciprocal singularities and steering the optimizer efficiently away from constraint boundaries.

2. **Convex Linearization (CONLIN)**:
   A simpler predecessor to MMA. It uses linear approximations for terms with positive derivatives and reciprocal approximations for negative derivatives. To ensure stability and prevent singularities near zero, strict move-limit boxes and lower bounds are actively enforced on the subproblems.

3. **Sequential Linear Programming (SLP)**:
   The simplest approximation scheme, completely linearizing the objective and constraints at each iterate. It does not capture curvature and can only reliably descend when paired with strict move-limit bounds. By default, our SLP implementation uses GEMSEO's exact HiGHS ``DUAL_SIMPLEX`` LP solver, bypassing Jacobian calls in the inner loop.

4. **Gradient-Enhanced Surrogate-Based Optimization (GE_SBO)**:
   A trust-region-managed surrogate optimizer for *expensive* models whose gradients
   are available (adjoint sensitivities). At each outer iteration it:

   - fits a **gradient-enhanced kriging** surrogate — a Gaussian process conditioned
     on both function values and gradients, so each expensive evaluation contributes
     :math:`1 + r` observations;
   - reduces high-dimensional design spaces through an **active subspace**
     :math:`z = W^{T} x` identified from the sampled gradients (SVD), keeping the
     kriging system size independent of the full dimension;
   - acquires a **batch of** ``batch_size`` **points** inside the trust region: one
     penalized-exploitation point plus a ladder of lower-confidence-bound points
     :math:`\mu - \kappa_j \sigma` with increasing :math:`\kappa_j` and a repulsion
     term for diversity. Batch points are independent and can be evaluated in
     parallel by the caller;
   - accepts/rejects the step and updates the trust-region radius by the classical
     ratio test on the :math:`L_1` penalty merit function.

   Inequality constraints are co-krigged with the objective (one shared Cholesky
   factorization) and penalized inside the acquisition. Invoke it with
   ``scenario.execute(algo_name="GE_SBO", max_iter=200, batch_size=4)``, where
   ``max_iter`` is the total budget of true model evaluations. The engine lives in
   ``scp_uno.gesbo_core`` (pure NumPy/SciPy) and is also usable standalone via
   ``gesbo_minimize``.

5. **Transformer Learned Optimizer (TRANSFORMER_OPT)** *(requires JAX)*:
   A "learning to optimize" approach that reaches **MMA-level performance** on
   GGP problems: one token per design variable (signed-log gradients,
   constraint activity, asymptote width / previous step / oscillation — the
   statistics MMA's own update uses), self-attention for the global dual
   coupling, and a full-dimension step per evaluation. Head 0 is trained by
   behaviour cloning of a NumPy MMA teacher on synthetic constrained families
   (incl. a toy-SIMP family mirroring compliance/volume structure) mixed with
   recorded GEMSEO-MMA trajectories of the real problem
   (``scripts/collect_ggp_mma_trajectories.py``); heads 1..3 provide
   far-sighted multi-scale proposals for the batch mode (``eval_heads``).
   On the short cantilever it reaches compliance 74.6 in 320 evaluations vs
   ~74.5 for the preset MMA (77.4 vs 75.7 at 200). Invoke with
   ``scenario.execute(algo_name="TRANSFORMER_OPT", max_iter=320)``.
   Engine: ``scp_uno.transformer_opt_core`` + ``scp_uno.mma_teacher``.

6. **Reduced-space GEK (GEK2D)**:
   At each iterate, an orthonormal subspace is built from
   :math:`e_1 = -\nabla f/\|\nabla f\|`, the KS-aggregated constraint
   gradient, the previous step (momentum: the span contains the
   conjugate-gradient step) and the gradient difference (secant); the
   trust-region subproblem on that subspace is solved with a
   gradient-enhanced kriging surrogate fitted from a few true evaluations per
   iteration (exact projected directional derivatives), with backtracking on
   rejections and restarts from the incumbent best. A local-descent method
   for multimodal problems — efficient convergence to a local minimum, no
   optimality claim. On the short cantilever: compliance 203 / 136 at
   200 / 320 evaluations (from 42358). Engine: ``scp_uno.reduced_space``.

7. **Reduced-space transformer (TRANSFORMER_2D)** *(requires JAX)*:
   The same formulation with the GEK sub-optimization replaced by a
   transformer that predicts the step and a trust-radius multiplier from the
   iteration history at zero inner-evaluation cost. All features are frame
   projections, so the policy is **independent of the design-space dimension
   by construction**; it is trained only on generic synthetic families
   (``scripts/train_rs_transformer.py`` — no topology-optimization data).
   Zero-shot on the short cantilever (never seen): compliance ~340 at 200
   evaluations. Engine: ``scp_uno.rs_transformer``.

8. **Moving-Least-Squares SBO (MLS_SBO)**:
   The same trust-region batch-SBO frame as GE_SBO, with the kriging surrogate
   replaced by **gradient-enhanced Moving Least Squares** (Hermite MLS, linear
   basis): every prediction solves a small local weighted normal system that
   matches the sampled values *and* gradients, so there is no global
   correlation matrix and the cost per query is :math:`O(n d^2)` — the full
   ~100-variable GGP design space is handled directly, without an active
   subspace. The MLS **length scale evolves with the sampling**:
   :math:`h = \mathrm{ls\_factor} \cdot d_{\min}`, where :math:`d_{\min}` is
   the minimal distance from the trust-region center to a sampled point inside
   the trust region — the fit localizes automatically as the trust region
   shrinks and samples cluster. The acquisition (penalized exploitation + LCB
   ladder with repulsion, with a sample-density proxy standing in for the
   kriging variance) is shared with GE_SBO. The exploitation step (the whole
   step in the sequential ``batch_size=1`` regime) optimizes an
   **anchored separable-quadratic model with weights frozen at the
   trust-region center**: exact value/gradient interpolation of the
   incumbent (the fully-linear-model requirement of trust-region theory)
   plus a diagonal Hessian secant-fitted to the neighbours' gradients —
   MMA-class second-order structure with curvature *measured from samples*
   instead of heuristic asymptotes. (Per-query MLS refits must not be handed
   to an SQP solver: their diffuse derivative is not the true derivative of
   their value, an inconsistency that silently degrades the subproblem
   solve.) The constrained subproblem `min m_0 s.t. m_j <= 0` is solved by
   SLSQP inside the trust region, with restarts from the incumbent best
   until the evaluation budget is spent. Short-cantilever results at 200
   true evaluations (iso function calls with MMA; one FEM + adjoint call per
   evaluation for every method): compliance ~350 sequential
   (``batch_size=1``, the MMA-like regime) and ~286 with ``batch_size=4``
   — versus ~790/~1277 for the kriging GE_SBO under the same protocol, ~203
   for the reduced-space GEK2D, and 75.7 for MMA, whose separable reciprocal
   approximation remains near-exact for compliance-type responses.
   Invoke with
   ``scenario.execute(algo_name="MLS_SBO", max_iter=200, batch_size=4)``;
   engine: ``scp_uno.mls_sbo_core`` (pure NumPy/SciPy), standalone via
   ``mls_sbo_minimize``.

Monotone Backtracking Line-Search
---------------------------------

All approximation algorithms in the framework can optionally enable a **Monotone Backtracking Line-Search** mechanism (``use_line_search=True``).

When enabled, the framework:
1. Evaluates an :math:`L_1` penalty merit function: :math:`\phi(x) = f(x) + \mu \sum_i \max(0, c_i(x))`
2. Checks if the subproblem's proposed step provides an Armijo sufficient decrease in :math:`\phi(x)`.
3. If not, it backtracks along the proposed step direction until the merit function decreases.

This serves as a powerful stabilizing safeguard, completely eliminating wild objective spikes and constraint violations when the local convex (or linear) approximations are too optimistic.

Inner Solvers
-------------

The convex (or linear) subproblems can be solved by various internal engines:
- **Scipy SLSQP**: Reliable SQP engine ideal for MMA and CONLIN subproblems.
- **HiGHS (Dual Simplex)**: High-performance exact LP solver utilized natively by SLP.
- **Uno**: High-performance C++ engine for nonlinearly constrained optimization (via the ``unopy`` wrapper).

Settings Configuration
----------------------

All algorithms share a unified ``SCPSettings`` Pydantic model. Key parameters include:
- ``max_iter``: Maximum number of outer iterations.
- ``max_optimization_step``: Strict move limits enforcing maximum design variable change per iteration.
- ``use_line_search``: Toggle for the monotone backtracking line-search.
- ``line_search_penalty``: The :math:`\mu` multiplier weighing constraint violations against objective changes.
