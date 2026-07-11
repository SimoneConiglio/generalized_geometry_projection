Sequential Convex Programming (SCP) Framework
=============================================

The ``scp_uno`` package implements a robust Sequential Convex Programming framework integrated with the GEMSEO environment. It is tailored for highly non-linear and non-convex problems such as Topology Optimization.

Overview
--------

The core idea of SCP is to replace a difficult, non-convex optimization problem with a sequence of simpler, convex approximations (subproblems). Each subproblem is solved, the design variables are updated, and a new approximation is constructed until convergence is reached.

Algorithms
----------

The framework provides five major algorithms:

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

5. **Transformer Learned Optimizer (TRANSFORMER_OPT)** *(experimental, requires JAX)*:
   A "learning to optimize" approach: a small set-transformer policy reads the
   recent history (positions, merit values, gradients — encoded in the same
   gradient active subspace as GE_SBO, trust-region-relative and
   scale-normalized) and directly proposes the next **batch** of query points;
   a classical trust-region loop safeguards acceptance. The policy is trained
   offline by imitating a privileged teacher on synthetic tasks of random
   dimension (``scripts/train_transformer_opt.py``); the packaged default
   weights transfer unchanged across problem dimensions and objective scales.
   Invoke with ``scenario.execute(algo_name="TRANSFORMER_OPT", max_iter=200)``.
   Engine: ``scp_uno.transformer_opt_core`` (NumPy + JAX).

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
