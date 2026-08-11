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
   ``grad beta_ij(x_i) = e_j``). The gap was largely a TRUST-REGION-SIZE
   artifact: the interpolant is only informative inside its covered
   region, and steps that outrun it land in nearest-plane fallback
   territory. Deterministic ``tr_init`` sweep (product-nn, 3 seeds):
   0.02 → 260/269/345, 0.05 → 324/416/772, 0.10 → 292/407/1368 —
   monotone in the worst case, exactly as the coverage argument predicts.
   At the matched ``tr_init=0.02`` the two step models are near parity
   (product median 269 vs quadratic 233; the quadratic's best single run,
   104, is the study's overall best). An initial LHS DOE (20 points in
   the first trust region) reduces variance (382/400/454) but does not
   beat the smaller step: in d=108 no affordable DOE covers a box —
   confining steps to the sampled tube is what works. Coupling the radii
   to the trust region (``rho_i = nn_factor * max(d_nn, delta)``,
   automatic when the driver passes ``delta``) removes the large-step
   blow-ups (worst 472 vs 772/1368) without improving the median — a
   robustness device, not an accelerator; the floor only binds when the
   sampling is denser than the step (seed-0 trajectories at tr=0.02 are
   bit-identical with and without it).

   **Hold-region policy** (``hold_region=True``, default): keep the
   center AND radius fixed until an improved point is found inside the
   region — a rejected candidate is a new interpolation point that
   refines the model exactly where the search happens (null steps sample
   a random interior point; ``region_patience`` failures still shrink as
   a safeguard). Deterministic results: quadratic @0.05 improves to
   126/144/144 (median 155 → 144, the study's best median) and product
   @0.02 to 241/248/275 (median 269 → 248, the tightest band measured);
   in the wrong step regime (product @0.05) holding an uncovered region
   amplifies failures (worst 1573) — the policy rewards, and requires, a
   step size matched to the model. Default step model
   remains quadratic; ``model="product"`` with a small ``tr_init`` is
   competitive and is the best measured surrogate.

   **Outer approximation isolates model bias** (``model="oa"``): the
   nearest-plane model's subproblem is solved EXACTLY as a MILP (HiGHS;
   binaries select the plane, Voronoi cells are big-M polyhedra), so
   every candidate carries a global-optimality certificate — the only
   heuristic-free subproblem in the study. Results: 484/577/612 @0.05,
   394/460/491 @0.02 — last place despite the certificate. Conclusion:
   the binding resource is model QUALITY (curvature), not subproblem
   solver quality; exact solves cannot rescue a biased model, and
   conversely the scan+SLSQP heuristic is not what limits the quadratic
   or product models. Ranking at each model's best config (median):
   quadratic+hold 144 < product+hold 248 < OA 460; MMA reference 75.7.
   The alphaBB repair (``model="alpha"``: max of alpha-lowered tangent
   planes, alpha per output from the pairwise secant bound - continuous,
   curvature-aware, interpolating) confirms the diagnosis only halfway:
   283/436/565 @0.02, 533/679/777 @0.05 - better than plain OA at the
   matched small step but still far behind the quadratic. Isotropic
   worst-case curvature (one alpha over all pairs and directions) is too
   conservative in d=108: one jammed pair inflates alpha for the whole
   window and the model plunges away from the center, making steps
   timid. DIRECTIONAL secant curvature - the anchored quadratic's
   per-coordinate diagonal - remains the decisive ingredient.
   The nonuniform-shift variant (``alpha_mode="diag"``: per-coordinate
   alpha_k from Hermite gradient differences, scaled to the pairwise
   validity bound) rules out mere directionality as the explanation:
   373/513/573 @0.02 - no better than the isotropic alpha (436) and
   still ~3.5x the quadratic. The final form of the finding: the
   quadratic's decisive ingredient is directional curvature ANCHORED at
   the incumbent - one curved model centred where the step is decided -
   not curvature distributed over max-of-pieces envelopes. The validity
   (underestimation) constraint that defines the alphaBB family is
   itself the cost: an interpolating step model does not need to be a
   lower bound, and paying for the certificate buys nothing here.

   **Surrogate tunneling at resets** (``tunnel=True``): a reset spends
   its evaluation on the argmin of the archive-fitted product surrogate
   over a shell around the incumbent, deflated by the current basin
   scale — an aimed valley escape at the same one-FEM-call price as the
   blind random reset it replaces (aimedness unit-tested; double-well
   escape verified in classical-shrink mode). On the cantilever under
   continuation + hold-region it is NEUTRAL: 134/145/165 vs 126/144/144
   without — the hold-region policy makes resets rare and late, and by
   the time one fires the soft continuation phase has already committed
   the basin. Mechanism validated, wrong trigger for this protocol; a
   merit-plateau trigger (tunnel while the budget is still young) is the
   untested lever.

   **Per-variable move limits are the single biggest win of the study**
   (``per_variable_tr=True``, now the DEFAULT). Six-seed head-to-head,
   quadratic @0.05 under continuation + hold-region:

   ==========================  ==========================  ======  ======
   step geometry               seeds                       median  best
   ==========================  ==========================  ======  ======
   scalar delta                126/144/144/201/221/1345       172     126
   per-variable, cap 1         99/159/161/162/253/716         162      99
   per-variable, cap 4         87/92/123/136/181/218          130      87
   cap 4 + profile carried     85/89/107/140/147/173          123      85
   ==========================  ==========================  ======  ======

   Carrying the profile across continuation phases
   (``--carry-profile``, ``mv_state_path``) is the cheapest remaining
   gain: better median, best AND worst (173 vs 218), and - the
   qualitative jump - ALL SIX runs binarize (solid 0.10-0.31, gray
   0.15-0.41) where the resetting variant managed 4 of 6 and the scalar
   step essentially none. The reach a variable earns while the homotopy
   is soft is exactly what it needs to finish the run at 0 or 1; throwing
   it away at each phase boundary was undoing the mechanism.

   Reach is NOT universally good: recast at ``mv_max=4`` (seeds 0-2,
   @0.02) the weaker families do not improve - alphaBB 334 -> 403, OA
   399 -> 385, product 229 -> 233 - because extra room only pays when
   the model points in the right direction; a biased model with reach
   simply travels further along a wrong step. The per-variable bound and
   an accurate anchored model are one mechanism, not two.

   MMA at the same 200-eval protocol: 78.2. The cap matters and the
   tidier choice loses: capping the profile at 1 keeps ``delta`` a true
   bound on every step, but measured worse than letting a
   consistently-moving variable reach BEYOND delta (cap 4). So this is a
   per-variable MOVE LIMIT vector, not a partition of a scalar region -
   which is what MMA's asymptotes are, MMA having no enclosing region to
   respect. Mechanism: density variables that want to run to 0/1 need
   sustained room while jittery boundary variables need clamping; one
   scalar delta grants neither. It also removes the catastrophic tail
   (worst 218 vs 1345) and produces the study's only binarizing designs
   (4/6 runs with solid material, best 87 at gray 0.10 / solid 0.35).
   Recast of every family with the per-variable step (seeds 0-2,
   clamp-only) leaves the ladder order intact but lifts the weaker
   members - alphaBB 436 -> 334, OA 460 -> 399, product 248 -> 229 (its
   seed-1 run, 154 at solid 0.16, is the first binarizing interpolant
   result) - confirming the scalar step had been penalizing them too.

   **Historical note: per-variable trust region closes most of the MMA gap**
   (``per_variable_tr=True``): box half-width_k = delta * w_k, the
   profile grown x1.2 on repeated accepted step direction, clamped x0.7
   on sign flips, bounded [0.2, 4] (MMA's asymptote rule). Iso-budget
   control first: MMA under the exact 3-phase 200-eval protocol scores
   78.2 (solid 0.27), so the scalar-TR driver's 144 was a structural
   gap, not budget. With the per-variable profile: 87/123/181 (best
   86.96, gray 0.10, solid 0.35 - the first surrogate designs that
   BINARIZE like MMA's). Mechanism: density variables that want to run
   to 0/1 need sustained room while jittery boundary variables need
   clamping; one scalar delta cannot grant both, and the directional
   freedom is worth more than any model-family choice measured in this
   study (quadratic vs product vs alphaBB moved the median by ~100-300;
   the TR shape moved the best from 126 to 87). Spread widens (181 on
   seed 0): more freedom, stronger basin lottery.

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
