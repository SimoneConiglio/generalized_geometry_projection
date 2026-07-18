# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Settings for the SCP framework."""

from __future__ import annotations
from gemseo.algos.opt.base_optimization_library import BaseOptimizerSettings
from pydantic import Field

class SCPSettings(BaseOptimizerSettings):
    """Settings for Sequential Convex Programming algorithms."""
    max_iter: int = Field(50, description="Maximum number of outer iterations.")
    inner_library: str = Field("UNO", description="GEMSEO library for inner subproblem.")
    inner_preset: str = Field("ipopt", description="Uno preset for subproblem.")
    inner_solver: str = Field("MMA", description="Uno subproblem solver / Inner algorithm name.")
    inner_max_iter: int = Field(500, description="Max iterations for Uno.")

    # MMA Specific settings (matching baseline)
    max_optimization_step: float = Field(0.05, description="Move limit (fraction of design-space range).")
    max_asymptote_distance: float = Field(0.5, description="Max distance for asymptotes.")
    initial_asymptotes_distance: float = Field(0.05, description="Initial distance for asymptotes.")
    min_asymptote_distance: float = Field(0.001, description="Min distance for asymptotes.")
    asymptotes_distance_amplification_coefficient: float = Field(1.2, description="Expansion factor.")
    asymptotes_distance_reduction_coefficient: float = Field(0.7, description="Shrinkage factor.")

    xtol_rel: float = Field(1e-12, description="Relative tolerance on design variables.")
    ftol_rel: float = Field(1e-12, description="Relative tolerance on objective.")
    xtol_abs: float = Field(1e-12, description="Absolute tolerance on design variables.")
    ftol_abs: float = Field(1e-12, description="Absolute tolerance on objective.")
    kkt_tol_abs: float = Field(1e-6, description="Absolute KKT tolerance.")
    kkt_tol_rel: float = Field(1e-6, description="Relative KKT tolerance.")

    # Monotone line-search (optional, works for any SCP subclass)
    use_line_search: bool = Field(False, description="Enable backtracking line-search on the merit function.")
    line_search_penalty: float = Field(
        100.0,
        description=(
            "Penalty weight μ in φ(x) = f(x) + μ·Σ max(0,cᵢ(x)).  "
            "Should be large enough to dominate constraint violations."
        ),
    )
    line_search_c1: float = Field(
        1e-4,
        description="Armijo sufficient-decrease parameter (0 < c1 < 1).",
    )
    line_search_beta: float = Field(
        0.5,
        description="Step-reduction factor per backtracking trial (0 < β < 1).",
    )
    line_search_max_trials: int = Field(
        10,
        description="Maximum number of backtracking trials before accepting the step.",
    )

class GESBOSettings(BaseOptimizerSettings):
    """Settings for the Gradient-Enhanced Surrogate-Based Optimization (GE_SBO) algorithm.

    ``max_iter`` is the total budget of *true model evaluations* (each outer
    iteration consumes up to ``batch_size`` of them, plus the initial DOE).
    """

    max_iter: int = Field(200, description="Total budget of true model evaluations.")
    max_outer_iter: int = Field(
        100, description="Cap on outer (batch) iterations regardless of the budget."
    )

    # -- batch acquisition --
    batch_size: int = Field(
        4, description="Number of points acquired (and evaluated) per outer iteration."
    )
    n_init_doe: int = Field(
        0,
        description=(
            "Initial DOE size (Latin Hypercube inside the initial trust region). "
            "0 means batch_size + 1."
        ),
    )
    kappa_base: float = Field(
        1.0, description="First exploration weight of the LCB kappa ladder."
    )
    kappa_growth: float = Field(
        2.0, description="Geometric growth of the LCB kappa ladder across the batch."
    )
    repulsion_weight: float = Field(
        1.0, description="Batch-diversity repulsion weight in the acquisition."
    )
    acq_n_restarts: int = Field(
        4, description="Random multistarts per acquisition sub-optimization."
    )

    # -- surrogate / high-dimensional scaling --
    use_gradients: bool = Field(
        True,
        description=(
            "Condition the surrogate on gradients (gradient-enhanced kriging). "
            "False falls back to plain (value-only) kriging."
        ),
    )
    max_latent_dim: int = Field(
        12,
        description=(
            "Active-subspace dimension used when the design space is larger; "
            "the surrogate is built on z = W^T x with W spanned by the sampled "
            "gradients (key to high-dimensional scaling)."
        ),
    )
    max_points: int = Field(
        60, description="Training window: number of nearest samples kept in the surrogate."
    )
    max_grad_points: int = Field(
        25, description="Subset of the window whose gradients enter the kriging system."
    )
    regularization: float = Field(1e-8, description="Kernel nugget regularization.")

    # -- trust region --
    tr_init: float = Field(
        0.25, description="Initial trust-region half-width (fraction of the design box)."
    )
    tr_min: float = Field(1e-5, description="Minimum trust-region half-width (stopping).")
    tr_max: float = Field(0.75, description="Maximum trust-region half-width.")
    tr_shrink: float = Field(0.5, description="Radius shrink factor on rejected steps.")
    tr_expand: float = Field(2.0, description="Radius expansion factor on very good steps.")

    # -- merit / stopping --
    penalty: float = Field(
        100.0, description="mu of the l1 merit function f + mu * sum(max(0, c))."
    )
    constraint_tol: float = Field(1e-6, description="Feasibility tolerance on constraints.")
    stall_limit: int = Field(
        5, description="Consecutive non-improving iterations before declaring a stall."
    )
    seed: int = Field(0, description="Random seed (DOE, acquisition multistarts).")

    # Expected by GEMSEO's base driver (KKT-based stopping hooks).
    kkt_tol_abs: float = Field(1e-6, description="Absolute KKT tolerance.")
    kkt_tol_rel: float = Field(1e-6, description="Relative KKT tolerance.")


class MLSSBOSettings(BaseOptimizerSettings):
    """Settings for the Moving-Least-Squares SBO (MLS_SBO) algorithm.

    ``max_iter`` is the total budget of *true model evaluations* (each outer
    iteration consumes up to ``batch_size`` of them, plus the initial DOE).
    """

    max_iter: int = Field(200, description="Total budget of true model evaluations.")
    max_outer_iter: int = Field(
        100, description="Cap on outer (batch) iterations regardless of the budget."
    )

    # -- batch acquisition (shared with GE_SBO) --
    batch_size: int = Field(
        4, description="Number of points acquired (and evaluated) per outer iteration."
    )
    n_init_doe: int = Field(
        0,
        description=(
            "Initial DOE size (Latin Hypercube inside the initial trust region). "
            "0 means batch_size + 1."
        ),
    )
    kappa_base: float = Field(
        1.0, description="First exploration weight of the LCB kappa ladder."
    )
    kappa_growth: float = Field(
        2.0, description="Geometric growth of the LCB kappa ladder across the batch."
    )
    repulsion_weight: float = Field(
        1.0, description="Batch-diversity repulsion weight in the acquisition."
    )
    acq_n_restarts: int = Field(
        4, description="Random multistarts per acquisition sub-optimization."
    )

    # -- surrogate --
    max_points: int = Field(
        60, description="Training window: number of nearest samples kept."
    )
    regularization: float = Field(
        1e-8, description="Tikhonov jitter on the local MLS normal system."
    )

    # -- length scale: tracks the sample spacing inside the trust region --
    ls_factor: float = Field(
        2.0,
        description=(
            "Length scale h = ls_factor * (minimal distance from the "
            "trust-region center to a sampled point inside the trust region)."
        ),
    )
    ls_min: float = Field(1e-3, description="Minimum length scale (normalized units).")
    ls_max: float = Field(2.0, description="Maximum length scale (normalized units).")

    # -- anchored model / sequential subproblem --
    anchor_center: bool = Field(
        True,
        description=(
            "Anchor the model to the exact value and gradient at the "
            "trust-region center (first-order consistency, as in MMA); the "
            "exploitation step solves the constrained model subproblem."
        ),
    )
    subproblem_maxiter: int = Field(
        100, description="SLSQP iterations for the anchored model subproblem."
    )
    n_global: int = Field(
        256,
        description=(
            "Global phase of the model subproblem: LHS candidates scanned "
            "over the trust-region box before the SLSQP polish (0 disables). "
            "Essential for multimodal surrogates such as model='tangent'."
        ),
    )
    intermediate: str = Field(
        "linear",
        description=(
            "Intermediate variables of the anchored model: 'linear' (default) "
            "keeps the plain separable quadratic in x; 'mma' places a "
            "reciprocal transform on the descent side of each gradient "
            "(Method of Moving Asymptotes / CONLIN style) — benchmarks show "
            "no significant difference between the two."
        ),
    )
    asy_init: float = Field(
        0.5, description="Minimum asymptote distance (normalized units)."
    )
    min_fit_neighbors: int = Field(
        1,
        description=(
            "The anchored-fit bandwidth always covers at least this many "
            "samples. Benchmarks favour the tightest (1); larger values "
            "widen the curvature fit over more history."
        ),
    )
    fit_values: str = Field(
        "constraints",
        description=(
            "Which outputs include function-value rows in the Hermite "
            "curvature fit: 'none', 'constraints' (default - feasibility is "
            "a statement about values, while the objective keeps "
            "quasi-Newton gradient-only secants), or 'all'."
        ),
    )
    model: str = Field(
        "tangent",
        description=(
            "Exploitation-subproblem model: 'tangent' (default - "
            "softmax-weighted sum of tangent hyperplanes, closed form with "
            "exactly consistent analytic gradients), 'quadratic' (anchored "
            "separable quadratic, center-frozen weights) or 'planar' "
            "(center-frozen planar Hermite MLS, no curvature term)."
        ),
    )

    # -- trust region --
    tr_init: float = Field(
        0.25, description="Initial trust-region half-width (fraction of the design box)."
    )
    tr_min: float = Field(1e-5, description="Minimum trust-region half-width (stopping).")
    tr_max: float = Field(0.75, description="Maximum trust-region half-width.")
    tr_shrink: float = Field(0.5, description="Radius shrink factor on rejected steps.")
    tr_expand: float = Field(2.0, description="Radius expansion factor on very good steps.")
    n_resets: int = Field(
        8,
        description=(
            "Trust-region restarts from the incumbent best after a collapse "
            "or stall, until the evaluation budget is exhausted."
        ),
    )

    # -- merit / stopping --
    penalty: float = Field(
        100.0, description="mu of the l1 merit function f + mu * sum(max(0, c))."
    )
    constraint_tol: float = Field(1e-6, description="Feasibility tolerance on constraints.")
    stall_limit: int = Field(
        5, description="Consecutive non-improving iterations before declaring a stall."
    )
    seed: int = Field(0, description="Random seed (DOE, acquisition multistarts).")

    # Expected by GEMSEO's base driver (KKT-based stopping hooks).
    kkt_tol_abs: float = Field(1e-6, description="Absolute KKT tolerance.")
    kkt_tol_rel: float = Field(1e-6, description="Relative KKT tolerance.")


class TransformerOptSettings(BaseOptimizerSettings):
    """Settings for the transformer learned-optimizer (TRANSFORMER_OPT).

    ``max_iter`` is the total budget of true model evaluations; each outer
    iteration consumes one batch (the policy's ``n_heads_out``, typically 4).
    """

    max_iter: int = Field(200, description="Total budget of true model evaluations.")
    weights_path: str = Field(
        "",
        description=(
            "Path to a trained policy .npz (see scripts/train_transformer_opt.py). "
            "Empty selects the packaged default weights."
        ),
    )

    # -- stepping --
    move: float = Field(
        0.01,
        description="Move limit as a fraction of each variable's range (as in MMA).",
    )
    eval_heads: int = Field(
        1,
        description=(
            "Policy heads evaluated per iteration: 1 = sequential MMA-like "
            "stepping (one move per evaluation); k>1 = best-of-k batch of the "
            "MMA-imitation head plus far-sighted multi-scale proposals."
        ),
    )
    n_backtracks: int = Field(
        2, description="Step halvings tried when a proposal worsens the merit."
    )
    stall_limit: int = Field(
        8, description="Consecutive rejected iterations before stopping."
    )
    accept_mode: str = Field(
        "always",
        description=(
            "'always' advances unconditionally like MMA (best-so-far "
            "bookkeeping guards the reported result); 'watchdog' accepts "
            "against the worst of the last nonmonotone_window merits; "
            "'monotone' is strict."
        ),
    )
    nonmonotone_window: int = Field(
        10, description="Window size for accept_mode='watchdog'."
    )

    # -- merit / misc --
    penalty: float = Field(
        100.0, description="mu of the l1 merit function f + mu * sum(max(0, c))."
    )
    constraint_tol: float = Field(1e-6, description="Feasibility tolerance on constraints.")
    seed: int = Field(0, description="Random seed.")

    # Expected by GEMSEO's base driver (KKT-based stopping hooks).
    kkt_tol_abs: float = Field(1e-6, description="Absolute KKT tolerance.")
    kkt_tol_rel: float = Field(1e-6, description="Relative KKT tolerance.")


class _ReducedSpaceBaseSettings(BaseOptimizerSettings):
    """Shared settings of the reduced-space (2D) optimizers."""

    max_iter: int = Field(200, description="Total budget of true model evaluations.")
    subspace_dim: int = Field(
        4,
        description=(
            "Reduced-space directions: steepest descent, orthogonalized "
            "constraint gradient, momentum (previous step), secant "
            "(gradient difference)."
        ),
    )
    delta_init: float = Field(0.1, description="Initial trust radius (normalized box).")
    delta_min: float = Field(1e-6, description="Minimum trust radius (stopping).")
    delta_max: float = Field(0.5, description="Maximum trust radius.")
    shrink: float = Field(0.5, description="Radius shrink factor on rejected steps.")
    expand: float = Field(1.6, description="Radius expansion factor on boundary steps.")
    penalty: float = Field(
        10.0, description="l1 merit safety factor over the multiplier estimate."
    )
    constraint_tol: float = Field(1e-6, description="Feasibility tolerance.")
    ks_rho: float = Field(50.0, description="KS constraint-aggregation parameter.")
    stall_limit: int = Field(10, description="Consecutive rejections before stopping.")
    n_backtracks: int = Field(
        2, description="Step halvings tried when a proposal is rejected."
    )
    n_resets: int = Field(
        6, description="Trust-region restarts from the incumbent best before stopping."
    )
    seed: int = Field(0, description="Random seed (frame fallbacks).")

    # Expected by GEMSEO's base driver (KKT-based stopping hooks).
    kkt_tol_abs: float = Field(1e-6, description="Absolute KKT tolerance.")
    kkt_tol_rel: float = Field(1e-6, description="Relative KKT tolerance.")


class GEK2DSettings(_ReducedSpaceBaseSettings):
    """Settings of the GEK2D reduced-space algorithm."""

    n_inner: int = Field(
        5, description="True evaluations spent on the subspace per iteration."
    )
    n_candidates: int = Field(
        4096, description="Surrogate candidate points for the subspace solve."
    )


class RSTransformerSettings(_ReducedSpaceBaseSettings):
    """Settings of the TRANSFORMER_2D reduced-space algorithm."""

    weights_path: str = Field(
        "",
        description=(
            "Path to trained reduced-space policy weights "
            "(scripts/train_rs_transformer.py); empty selects the packaged default."
        ),
    )
    use_policy_radius: bool = Field(
        True,
        description=(
            "Use the policy's learned trust-radius multiplier; False falls "
            "back to the driver's boundary-expand / interior-shrink rule."
        ),
    )
    gek_refresh: int = Field(
        0,
        description=(
            "Hybrid schedule: every k-th iteration runs a full GEK "
            "sub-optimization (measured refresh) and the policy drives the "
            "iterations in between (0 = pure policy)."
        ),
    )


class UnoSettings(BaseOptimizerSettings):
    """Settings for the Uno solver wrapper."""
    preset: str = Field("filtersmma", description="Uno configuration preset.")
    solver: str = Field("MMA", description="Uno subproblem solver.")
    hessian: str = Field("identity", description="Hessian model.")
    max_iter: int = Field(500, description="Max iterations.")
    logger: str = Field("INFO", description="Uno logging level.")
    kkt_tol_abs: float = Field(1e-6, description="Absolute KKT tolerance.")
    kkt_tol_rel: float = Field(1e-6, description="Relative KKT tolerance.")
