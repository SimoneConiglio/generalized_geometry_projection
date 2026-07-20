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
   until the evaluation budget is spent.

   **Reproducibility.** With the default preset FEM backend (``amjax``,
   JAX/XLA) results are NOT reproducible: XLA's CPU thread-pool reduction
   order is fixed per process, giving ~1e-13 perturbations at identical
   inputs, and the chaotic trust-region trajectory amplifies them into
   *different local minima* within ~200 evaluations. ``fem_solver:
   direct`` (scipy SuperLU) is bit-deterministic end-to-end (verified by
   diffing per-solve residual fingerprints of full duplicate runs); use it
   for any benchmarking of this optimizer.

   **Short-cantilever study** (200 true evaluations — iso function calls
   with MMA, one FEM + adjoint call each — sequential ``batch_size=1``,
   deterministic direct solver, 3 seeds per config): gradient-only
   quadratic 109/539/562; constraint-value learning
   (``fit_values="constraints"``) 453/576/688; planar Hermite MLS
   (``model="planar"``) 526/555/595; tangent-hyperplane softmax blend
   (``model="tangent"``, the default) 536/612/892. With the global
   subproblem phase (``n_global=256``) and a length-scale regime sweep
   (``ls_factor`` 0.5 / 2 / 8, i.e. interpolating to averaging weights),
   the tangent model gives 545–819 / 542–901 / 608–660; the
   Hermite–Shepard cardinal weights (``weighting="shepard"``) give
   535/636/694, and the separation-aware Wendland cardinal weights with
   window de-jamming (``weighting="wendland"``, the default) give
   558/641/659 — neither the global solve, the weight regime, nor either
   cardinal-weight family separates from the base runs.

   **Density-field diagnosis and continuation.** The low-performing runs
   (~500–900) are not inferior truss layouts but *collapsed* designs: the
   components thin out into a near-void gray haze (always feasible under
   the upper-bound volume constraint), after which their position/angle
   gradients vanish and the run stalls on a flat plateau. Parameter
   continuation (``benchmarks/mls_continuation.py``: warm-started
   ``ka``/``pp`` homotopy, soft→baseline, total budget split 35/30/35 so
   results stay iso-function-calls) counters exactly this — the soft
   saturation keeps faint components' sensitivities alive so they can be
   revived before the sharp phase locks the topology. Deterministic
   results (3 seeds): quadratic 157/266/405 (median 539 → 266), Wendland
   tangent 242/472/811 (median 641 → 472); MMA is collapse-proof already
   and does not benefit (78.2 vs 75.7). An MMA-like initial step cap
   alone (``tr_init=0.05``) helps only mildly (344/453/530) — the
   collapse is driven by the sharp landscape's descent direction, not by
   early overshoot. **Continuation + step cap combined eliminates the
   collapse in every seed: 127/155/206 (median 539 → 155)** — the best
   and most consistent MLS_SBO configuration measured, within ~2x of
   MMA's 75.7 at iso-budget.

   **Surrogate quality is not step quality.** The product-form Hermite
   shape functions (``model="product"``: exact value+gradient
   interpolation at every sample for any spacing, cardinality by product
   zeros; support radii either Deparis nearest-neighbour ``radius="nn"``
   or LOO-selected global ``radius="global", auto_support=True``) are the
   best measured *surrogate* — on a 10-point Branin sampling they beat
   gradient-enhanced kriging (f RMSE 22.9 vs 30.2, max grad err 970 vs
   2084) with exact node interpolation. As the optimizer's *step model*
   under the continuation+cap protocol they nonetheless trail the
   anchored quadratic (LOO-global 349/408/415, Deparis-nn 324/772/416 vs
   127/155/206). NOTE: the model is NOT flat at the centre — the gradient
   there is exact by construction (beta contributes
   ``grad beta_ij(x_i) = e_j``); the step-quality gap is under
   investigation, with the trust-region size relative to the surrogate's
   covered region as the leading hypothesis (outside all supports the
   nearest-plane fallback rules, and its Voronoi kinks can attract the
   subproblem). Default step model: quadratic; use ``model="product"``
   when the surrogate itself is the deliverable.

   **Penalty continuation does NOT help** (measured on the geometric mass
   field ``rho_V``, the honest binarization metric — ``rho_E`` carries
   ``gammac`` and overstates grayness): ramping ``gammac`` 1→2→3 raises
   compliance to 195–225 and the gray fraction to ~0.62 (the soft-gammac
   phase actively *creates* spread-out gray-mass layouts — under linear
   stiffness a variable-thickness sheet is optimal — which the later
   phases cannot consolidate), and stacking ``p_penalty`` 1→3 on top is
   worse still (effective ``Mc^9`` collapses mid-gray stiffness:
   203–357, zero solid elements). Keep the material penalty at full
   strength (``gammac=3``) in every phase and continue only the geometry
   sharpness (``ka``/``pp``). Residual grayness tracks incomplete
   convergence, not penalty miscalibration: the near-binary best run
   (gray 0.16) uses the same baseline penalty as the gray ones. The model
   architectures are
   **statistically indistinguishable at this sample size** — run-to-run
   basin scatter (109–688 within one config) dominates the architecture
   choice, so earlier single-run ablation rankings are not supported once
   solver noise is removed. References under the same protocol: kriging
   GE_SBO ~790–1277, reduced-space GEK2D ~203, MMA 75.7. Practical use of
   MLS_SBO on such landscapes is therefore **best-of-N restarts** (the
   best observed run, 109, approaches MMA territory), which the
   deterministic backend makes exactly reproducible.
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
