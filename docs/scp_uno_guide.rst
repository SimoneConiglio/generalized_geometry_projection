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

   **Why the interpolants lose: gradient dilution at the DECISION
   point.** Diagnostic (d=40, 25-sample window clustered at the centre,
   anisotropy 9.6, box delta=0.05, available drop in box 6.911):

   =========  ==========  ==========  =============  =================
   model      pred drop   actual      sign agree     grad angle error
   =========  ==========  ==========  =============  =================
   product         2.838       2.631         67.5%             4.2 deg
   alphaBB         2.863       2.631         67.5%             4.6 deg
   OA-planes       2.863       2.631         67.5%             4.6 deg
   quadratic       6.911       6.911        100.0%             0.0 deg
   =========  ==========  ==========  =============  =================

   All three sample-based models find the SAME point and capture 38% of
   the available improvement. The cause is not curvature and not the
   subproblem solver: their gradient direction at that point is only
   ~4 degrees off, but the box optimum is a CORNER, so every coordinate
   moves a full half-width - and a 4-degree error in high dimension
   flips the sign of a third of the coordinates. 32.5% of the variables
   move the wrong way by the full step. The models are exact AT the
   samples, but the step decision happens at the box boundary, far from
   every sample (0.21 away here), where the value and slope are a BLEND
   of neighbouring samples' planes - each belonging to a different
   point. The anchored quadratic never blends: it extrapolates the
   CENTRE's exact value and gradient plus curvature across the whole
   box, so its sign pattern is the centre's, exactly.

   This one measurement explains the whole ladder: why the exactly
   solved OA MILP gains nothing (the exact minimum of a model with a
   third of its signs wrong is still a step with a third of its signs
   wrong); why alphaBB's curvature repair changes little (curvature is
   not the defect); why the product model wins every accuracy metric
   yet loses every optimization metric (its accuracy lives where the
   samples are, the decision lives where they are not); and why reach
   (mv_max>1) helps only the quadratic (it lengthens whichever step the
   model proposes - a gift to a consistent one, a penalty to a diluted
   one). Interpolation and step quality are optimized at different
   points in space.

   **Outer approximation, correctly defined, and its secant repair.**
   ``model="oa"`` minimizes the tangent-plane UPPER ENVELOPE
   ``max_i [f_i + g_i^T(x-x_i)]`` over the box - an LP, not a MILP.
   Six seeds @0.02: 90/96/279/479/608/663, median 379. Its signature is
   unusual: EVERY run binarizes hard (gray 0.10-0.13, solid 0.29-0.35),
   including the failures - the envelope is decisive, and the lottery is
   whether it is decisive about the right structure. The reason it is
   decisive is that near the incumbent the highest plane is usually the
   incumbent's OWN plane, so the envelope is implicitly anchored:
   centre-exact value and gradient, the property section "gradient
   dilution" identifies as decisive.

   **The GEMSEO bilevel-OA adaptive convexification is the winner**
   (``oa_correction="adaptive"``, the rule implemented in
   gemseo-bilevel-outer-approximation): correct each plane's SLOPE by
   least squares - ``(dX dX^T) delta = [g_k.dX_j - df_j + min_dfk]_+``,
   ``g_k^corr = g_k - dX^T delta`` - so it predicts ``f_j - min_dfk`` at
   the other samples while STILL passing exactly through its own. Six
   seeds @0.02: 91/92/95/96/140/183, median 95.4, best 90.9 - against
   raw OA's 379 and the anchored quadratic's 123, and within 22% of
   MMA's 78.2 at the median. Every run binarizes (solid 0.08-0.33).
   This is the best non-quadratic model in the study and the best
   MEDIAN of any model here.

   Adding the per-variable move limits and the carried width profile -
   the two devices that lifted the quadratic from 172 to 123 - does NOT
   compound with it: 91/95/98/100/109/155, median 99.2 against 95.4
   without, at the same step size (and 0.05 is worse still: 129, 960).
   The tail does tighten (worst 155 vs 183) and the spread narrows, but
   the median does not move. The two mechanisms are not additive
   because they were fixing the same defect from opposite ends: the
   move limits let the quadratic COMMIT variables that a scalar radius
   was throttling, and the convexified envelope already commits by
   itself (it binarized 6/6 even before the move limits existed). Once
   the model stops over-promising, per-variable reach has little left
   to buy.

   Trust-region sweep on the convexified envelope (3 seeds unless
   noted): 0.005 -> 120, 0.01 -> 97 (8 seeds:
   84/90/91/94/100/105/111/180), 0.015 -> 103, 0.02 -> 95.4 (6),
   0.03 -> 167, 0.05 -> 267. The optimum is FLAT between 0.01 and 0.02
   and falls off sharply on both sides; 0.01 holds the best single run
   of the study (84.4, matching the anchored quadratic's best while
   keeping a far better median). Below 0.01 the budget expires before
   the design can move - two of three runs at 0.005 finish with zero
   solid material. Convexification does NOT extend the model's validity
   radius: it improves how the envelope RANKS nearby points, but the
   neighbourhood it can be trusted over is still set by curvature,
   since the model is piecewise linear either way.

   **Apples-to-apples with MMA** (``--deterministic-start``:
   ``n_init_doe=1`` in every phase, so the driver starts from x0 alone,
   exactly the information MMA gets - every other number in this guide
   gives the surrogate a random LHS DOE inside the initial trust region
   that MMA never receives, so its seed spread is partly a sampling
   lottery MMA does not play):

   ===========================================  =========  ===========
   configuration (x0 only, 200 evals)           compliance deterministic
   ===========================================  =========  ===========
   MMA                                               78.2  yes
   oa + adaptive convexification @0.02               90.7  yes
   quadratic + pvtr + carried profile @0.05         120.6  no
   oa + adaptive convexification @0.01              223.8  yes
   product_aniso @0.02                              209.0  no
   ===========================================  =========  ===========

   The OA configuration is EXACTLY deterministic: with no DOE and an LP
   subproblem there is no random draw anywhere, and seeds 0 and 1
   return bit-identical results (90.744 twice at 0.02, 223.762 twice at
   0.01). Its fair single-run number against MMA's 78.2 is therefore
   90.7 - a 16% gap, with no averaging or best-of-N involved.

   The step-size ranking INVERTS without the DOE: 0.02 gives 90.7 while
   0.01 gives 223.8 (and 0.015 gives 323.4). With a random DOE the
   0.01-0.02 plateau was flat; from x0 alone the first steps are the
   only information the model has, and too small a radius leaves the
   envelope built from a cluster of nearly-collinear planes. Sampling
   and step size are not separable knobs.

   **Initial DOE size around x0** (LHS inside the initial trust region,
   x0 always evaluated first - MMA's starting guess). Sweep on the
   convexified envelope @0.02:

   ==========  ==========================================  ======  =====
   DOE points  seeds                                       median   best
   ==========  ==========================================  ======  =====
   1 (x0 only)  90.7 (deterministic, no seed)                90.7   90.7
   2 (default)  91/92/95/96/140/183                          95.4   90.9
   5            83/140/198                                  140     83.5
   10           79/88/93/100/120/136/256/368 (8 seeds)      109.7   78.8
   20           82/101/108/146/243/1012 (6 seeds)           127     81.7
   ==========  ==========================================  ======  =====

   The best single run of the whole study came from a 10-point DOE:
   **78.78 against MMA's 78.21** - a 0.7% difference, with a properly
   binarized design (gray 0.161, solid 0.311). But at eight seeds the
   DOE-10 MEDIAN (109.7) is WORSE than the 2-point default (95.4): a
   DOE raises the ceiling and the floor moves the wrong way, because in
   d=108 ten LHS points cannot describe the trust region - they can
   only spend 5% of the budget and hope the spread of cut directions
   helps. Three seeds had made DOE 10 look like a clear winner (88.1);
   eight seeds show that was sampling luck, the same trap that made raw
   OA look strong at three seeds.

   **The DOE is not a universal improvement** - every family was re-run
   with the same 10-point LHS DOE around x0 (3 seeds, each at its own
   best step size), against its 2-point-DOE baseline:

   ====================  ===================  ==========  ==========
   family                DOE 10 (3 seeds)     median      DOE 2 med.
   ====================  ===================  ==========  ==========
   oa + adaptive          79 / 88 / 93              88.1        95.4
   quadratic + pvtr       80 / 145 / 239           144.5       123
   quadratic plain        173 / 182 / 875          182.2       144
   product_aniso          132 / 155 / 308          155.3       154
   nearest_plane          233 / 276 / 345          275.8       385
   product isotropic      219 / 245 / 506          245.3       233
   lsupport               251 / 350 / 366          350.4       334
   ====================  ===================  ==========  ==========

   Only the two PLANE-ENVELOPE models improve (oa 95.4 -> 88.1,
   nearest_plane 385 -> 276). Every anchored or blended model degrades,
   the plain quadratic worst (144 -> 182, one seed at 875). The
   binarization column tells the mechanism: the quadratic finishes all
   three runs with ZERO solid material at gray 0.52-0.54, i.e. it never
   reaches a committed design, while oa keeps solid 0.31-0.34.

   The split is budget versus cut geometry. A DOE buys an envelope
   something it cannot generate any other way - independent cut
   directions, without which its planes are nearly collinear - so ten
   points are worth their cost. An anchored model already gets its
   curvature from its own trajectory; the DOE only takes 10 of phase
   0's 70 evaluations away from walking, and on this problem those
   evaluations are what carry the design to binarization. DOE size is a
   per-model choice, not a protocol-wide one.
   - 90.7 deterministic from x0 alone, median 95.4 over six DOE seeds,
   best 78.8 with a 10-point DOE. MMA: 78.2.

   Why it works where the intercept version failed: both make the
   envelope less optimistic, but the slope rotation keeps
   ``plane_k(x_k) = f_k``. Anchoring is preserved and only the
   overshoot between samples is removed.

   The convexity margin ``oa_margin`` was swept and is monotonically
   harmful at the median here: 0.00 -> 95.4 (6 seeds), 0.02 -> 98 (3),
   0.05 -> 163 (6: 87/137/153/173/186/902), 0.10 -> 272 (3). It also
   erodes binarization - solid 0.08-0.33 everywhere at margin 0, versus
   two zero-solid runs at 0.05 and two of three at 0.10. The one-sided
   least-squares rotation already removes the overshoot that matters;
   forcing the planes strictly BELOW the secants makes the envelope
   pessimistic between samples, and the LP then prefers the flat gray
   interior over committing to 0/1 - the same failure the intercept
   shift produced, reached from the other direction. The effect is not
   uniform per seed (margin 0.05 produced the study's single best OA
   run, 86.7), so a margin can help an individual trajectory; it loses
   on the median. Default: 0.0.

   **Why the margin transfers badly from the discrete case.** In
   gemseo-bilevel-outer-approximation the master variables are one-hot
   CATEGORICAL: every candidate is a vertex of the hypercube, and the
   master is a MILP. There the convexity margin is nearly free and
   genuinely useful - it guarantees the cut at an evaluated vertex makes
   that vertex strictly worse than its achieved value, which is what
   prevents the master from proposing the same point again and gives
   finite termination; and the pessimism it introduces BETWEEN samples
   costs nothing, because no point between vertices is ever feasible.
   Here the master is an LP over a continuous box: its optimum lives in
   exactly the region the margin distorts. Lowering the planes strictly
   below the secants biases the minimizer toward wherever the model was
   corrected least - the flattest part of the window - which is why the
   margin degrades the median and erodes binarization monotonically.
   The anti-cycling role the margin plays in the discrete method is
   already covered here by the trust region and the ratio test, so the
   continuous setting pays the margin's cost without needing its
   benefit. The SLOPE ROTATION itself transfers perfectly; only the
   strict margin on top of it does not.

   The naive secant correction (``oa_correction="secant"``: lower each plane by
   ``max_j [plane_i(x_j) - f_j]_+ + margin`` so the envelope underestimates
   every sample, making the LP a true relaxation) is theoretically the
   right repair for a nonconvex envelope - and it measured WORSE:
   225/361/417/475/738/8998 (median 446) at margin 0, and 284/329/519 at
   margin 0.02, versus 379 raw. It also destroys binarization completely
   (solid 0.000 in 9 of 9 runs, gray 0.24-0.59) where raw OA binarized
   6/6. Mechanism: the correction lowers the incumbent's own plane too
   (by at least the margin), so the envelope no longer reproduces the
   incumbent's value - exactly the anchoring that made raw OA decisive.
   The model is then dominated by whichever plane was least corrected,
   which is the flattest region of the window, i.e. gray interior. The
   relaxation's asymptotic argument is sound, but at a 200-evaluation
   budget the anchoring loss dominates.

   **Anisotropic product radii work** (``radius="aniso"``: per-variable
   rho_ik = nn_factor * d_nn(i) * s_k with the profile s_k ~ c_k^{-1/2}
   read off the Hermite gradient differences, so anisotropy costs no
   extra fitted parameters; ``auto_support=True`` LOO-selects the radius
   RULE and its multiplier). Six seeds: 117/128/143/164/190/280, median
   154 against 233 for isotropic radii, binarization 4/6 against 1/3 -
   the best interpolant configuration measured.

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
