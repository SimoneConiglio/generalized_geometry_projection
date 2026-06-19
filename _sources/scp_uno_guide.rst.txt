Sequential Convex Programming (SCP) Framework
=============================================

The ``scp_uno`` package implements a robust Sequential Convex Programming framework integrated with the GEMSEO environment. It is tailored for highly non-linear and non-convex problems such as Topology Optimization.

Overview
--------

The core idea of SCP is to replace a difficult, non-convex optimization problem with a sequence of simpler, convex approximations (subproblems). Each subproblem is solved, the design variables are updated, and a new approximation is constructed until convergence is reached.

Algorithms
----------

The framework provides three major algorithms:

1. **Method of Moving Asymptotes (MMA)**:
   The gold standard for topology optimization. It builds a strictly convex approximation by introducing reciprocal variables with dynamically updated asymptotes. This naturally injects curvature, safely preventing reciprocal singularities and steering the optimizer efficiently away from constraint boundaries.

2. **Convex Linearization (CONLIN)**:
   A simpler predecessor to MMA. It uses linear approximations for terms with positive derivatives and reciprocal approximations for negative derivatives. To ensure stability and prevent singularities near zero, strict move-limit boxes and lower bounds are actively enforced on the subproblems.

3. **Sequential Linear Programming (SLP)**:
   The simplest approximation scheme, completely linearizing the objective and constraints at each iterate. It does not capture curvature and can only reliably descend when paired with strict move-limit bounds. By default, our SLP implementation uses GEMSEO's exact HiGHS ``DUAL_SIMPLEX`` LP solver, bypassing Jacobian calls in the inner loop.

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
