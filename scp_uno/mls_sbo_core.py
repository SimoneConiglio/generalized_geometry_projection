# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Core engine for the Moving-Least-Squares SBO (MLS-SBO) — pure NumPy/SciPy.

Trust-region-managed, batch, surrogate-based optimizer in which **both the
objective and the constraints are approximated by gradient-enhanced Moving
Least Squares** (Hermite MLS): at every query point a *local* weighted linear
model is fitted that matches the sampled function values **and** their
gradients, with Gaussian weights ``w_i = exp(-||x - x_i||^2 / (2 h^2))``.

Compared with the gradient-enhanced kriging of :mod:`scp_uno.gesbo_core`:

* no global correlation system — each prediction solves one small
  ``(d+1) x (d+1)`` normal system assembled analytically in ``O(n d^2)``,
  so the method scales to the ~100-variable GGP regime with a *linear*
  basis (a diagonal-Hessian basis is a possible later extension);
* gradients enter as first-class observations (Hermite fit), so the linear
  MLS reproduces any linear function exactly and inherits first-order
  accuracy everywhere;
* the **length scale ``h`` evolves with the sampling**: at every iteration it
  is set proportional to the minimal distance from the trust-region center to
  a sampled point *inside the trust region* (``h = ls_factor * d_min``,
  clipped to bounds). As the trust region shrinks and the samples cluster,
  the fit localises automatically at the pace of the sample spacing — one
  knob, no error heuristics. The per-batch prediction error is still recorded
  in the iteration history for diagnostics.

The batch acquisition (penalized exploitation + LCB kappa ladder with
diversity repulsion) is **shared with GE-SBO** — the surrogate exposes the
same duck-typed interface, with the sample-density proxy
``sigma(x) = sqrt(max(0, 1 - sum_i w_i(x)))`` standing in for the kriging
variance, and the *diffuse* MLS derivative (the local slope ``b``) standing
in for the mean gradient.

The **exploitation step** (the whole step in the sequential
``batch_size=1`` regime) does not use the per-query MLS directly: it solves
a constrained subproblem on the :class:`AnchoredSeparableQuadratic` model —
the MLS weights frozen at the trust-region center, exact value/gradient
interpolation of the incumbent, and a diagonal Hessian secant-fitted to the
neighbours' gradients (see the class docstring for why per-query refits
must not be handed to an SQP solver). A classical three-zone trust-region
update with restarts from the incumbent best spends the entire evaluation
budget, as a sequential optimizer such as MMA would.

References: Lancaster & Salkauskas (1981) for MLS; Nayroles et al. (1992)
for the diffuse-derivative approximation; Alexandrov et al. (1998) for
trust-region model management.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from scp_uno.gesbo_core import EvaluateFn, propose_batch

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration & result containers
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class MLSSBOConfig:
    """Configuration of the MLS-SBO driver (lengths in normalized [0,1] space)."""

    batch_size: int = 4                # q: points acquired per outer iteration
    n_init: Optional[int] = None       # initial DOE size (default: batch_size + 1)
    max_evals: int = 200               # total budget of true evaluations
    max_outer_iter: int = 100          # cap on outer (batch) iterations

    # -- surrogate --
    max_points: int = 60               # training window: nearest points kept
    regularization: float = 1e-8       # Tikhonov jitter on the local normal system

    # -- length scale: tracks the sample spacing inside the trust region --
    ls_factor: float = 2.0             # h = ls_factor * (min distance to a TR sample)
    ls_min: float = 1e-3               # absolute bounds (normalized units)
    ls_max: float = 2.0

    # -- trust region (same semantics as GE-SBO) --
    tr_init: float = 0.25
    tr_min: float = 1e-5
    tr_max: float = 0.75
    tr_shrink: float = 0.5
    tr_expand: float = 2.0
    eta_accept: float = 1e-4
    eta_shrink: float = 0.25           # rho below this shrinks the radius
    eta_expand: float = 0.5
    n_resets: int = 8                  # TR restarts from the best point on collapse

    # -- anchored model / sequential subproblem --
    anchor_center: bool = True         # exact f, grad interpolation at the center
    subproblem_maxiter: int = 100      # SLSQP iterations for the model subproblem
    # Intermediate variables: on the short cantilever "linear" and "mma" are
    # statistically indistinguishable across seeds (169-300 vs 222-236), so
    # the simpler plain-quadratic model is the default; the reciprocal
    # convexification is available but not the active ingredient.
    intermediate: str = "linear"       # "mma" reciprocal variables or "linear"
    asy_init: float = 0.5              # minimum asymptote distance (normalized)
    # Global phase of the model subproblem: the surrogate is cheap, so scan
    # this many LHS candidates over the trust-region box and polish the best
    # ones with SLSQP (0 disables). Essential for multimodal surrogates such
    # as the tangent-plane blend, whose SQP-only solve finds one local valley.
    n_global: int = 256
    # Ablation on the short cantilever (200 evals, batch_size=1): the local
    # gradient-secant fit with the tightest bandwidth wins decisively
    # (222/236 across seeds) over adding value rows (439), widening the
    # bandwidth to >=10 neighbours (511), or both (322) — as in quasi-Newton
    # practice, curvature is best estimated from gradient differences only,
    # and the most local information gives the best steps. The knobs remain
    # for experimentation.
    min_fit_neighbors: int = 1         # anchored-fit bandwidth covers >= k samples
    # Value rows in the curvature fit: "none", "constraints" (default:
    # feasibility is a statement about VALUES, so constraint models must
    # track sampled values, while the objective keeps quasi-Newton
    # gradient-only secants), or "all".
    fit_values: str = "constraints"
    # Model of the exploitation subproblem:
    #   "product"   (default) product-form Hermite shape functions - the
    #               full cardinal constraint set for ANY node spacing and
    #               any d_max (cardinality by product zeros), closed form,
    #               exactly consistent analytic gradients;
    #   "tangent"   weighted sum of tangent hyperplanes (closed form,
    #               consistent gradients, cardinal only per weighting);
    #   "quadratic" anchored separable quadratic (centre-frozen weights);
    #   "planar"    centre-frozen planar Hermite MLS (no curvature term).
    model: str = "product"
    # Tangent-blend weights:
    #   "wendland" (default) separation-aware smooth cardinal bumps - each
    #              node's support stays clear of every other node, so the
    #              Hermite cardinality conditions hold exactly with bounded
    #              non-singular weights;
    #   "shepard"  singular compact cardinal weights (valid only when the
    #              scale is small relative to the spacing);
    #   "softmax"  Gaussian scores, interpolating only as h -> 0.
    weighting: str = "wendland"
    support_factor: float = 3.0
    # De-jamming: thin the fit window to this minimum pairwise separation
    # (fraction of h) before building the tangent blend - between jammed
    # nodes ANY cardinal basis must swing between delta-values over their
    # spacing, so gradients blow up like 1/spacing unless clusters are
    # collapsed to their best representative.
    min_sep_frac: float = 0.5

    # -- acquisition (duck-typed fields consumed by gesbo_core.propose_batch) --
    kappa_base: float = 1.0
    kappa_growth: float = 2.0
    repulsion_weight: float = 1.0
    n_restarts: int = 4
    acq_maxiter: int = 60
    constraint_penalty_acq: float = 10.0

    # -- merit / stopping --
    penalty: float = 100.0             # mu of the l1 merit f + mu*sum(max(0, c))
    constraint_tol: float = 1e-6
    ftol_abs: float = 1e-10
    stall_limit: int = 5

    seed: int = 0


@dataclasses.dataclass
class MLSSBOResult:
    """Outcome of an MLS-SBO run (raw, unnormalized coordinates)."""

    x_opt: np.ndarray
    f_opt: float
    constraints: np.ndarray
    is_feasible: bool
    n_evals: int
    n_iter: int
    status: str
    history: List[dict]


# --------------------------------------------------------------------------- #
# Gradient-enhanced Moving Least Squares (Hermite MLS, linear basis)
# --------------------------------------------------------------------------- #
class MovingLeastSquares:
    """Multi-output gradient-enhanced MLS with a linear basis.

    At a query point ``x`` the local model ``q(s) = c + b^T s``, ``s = x' - x``,
    is fitted by weighted least squares to the value observations
    (``c + b^T s_i ~ f_i``) and to the gradient observations (``b ~ g_i``),
    all with Gaussian weight ``w_i = exp(-||x - x_i||^2 / (2 h^2))``. The
    prediction mean is ``c`` and the *diffuse derivative* is ``b``.

    The normal equations are assembled analytically::

        [ S_w        S_w s^T          ] [c]   [ S_w f            ]
        [ S_w s   S_w (s s^T + I) + rI] [b] = [ S_w (s f + g)    ]

    (one shared system for all ``m`` outputs — objective and constraints —
    since the weights do not depend on the output). Outputs are standardized
    internally; the duck-typed interface (``predict``, ``predict_std``,
    ``predict_std_single``, ``_m``, ``_ym``, ``_ys``) matches
    :class:`scp_uno.gesbo_core.GradientEnhancedGP` so the GE-SBO batch
    acquisition is reused unchanged. The uncertainty proxy is the sample
    density ``sigma(x) = sqrt(max(0, 1 - sum_i w_i(x)))``.
    """

    def __init__(self, lengthscale: float, regularization: float = 1e-8,
                 use_gradients: bool = True) -> None:
        self.h = float(lengthscale)
        self.reg = float(regularization)
        self.use_gradients = bool(use_gradients)

    def fit(self, X: np.ndarray, Y: np.ndarray,
            G: Optional[np.ndarray] = None) -> "MovingLeastSquares":
        """Store samples: inputs ``X (n, d)``, outputs ``Y (n, m)``, gradients
        ``G (n, d, m)`` (``d(output)/dx``; zeros are used when absent)."""
        X = np.atleast_2d(np.asarray(X, float))
        Y = np.asarray(Y, float)
        if Y.ndim == 1:
            Y = Y[:, None]
        n, d = X.shape
        m = Y.shape[1]
        self._ym = Y.mean(axis=0)
        self._ys = np.maximum(Y.std(axis=0), 1e-12)
        self._X = X.copy()
        self._Yt = (Y - self._ym) / self._ys                    # (n, m)
        if G is None or not self.use_gradients:
            G = np.zeros((n, d, m))
        else:
            G = np.asarray(G, float)
            if G.ndim == 2:
                G = G[:, :, None]
        self._Gt = G / self._ys[None, None, :]                  # (n, d, m)
        self._m = m
        self._d = d
        return self

    # ------------------------------------------------------------- internal
    def _weights(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit weights, numerically shifted so the largest is always 1.

        The weighted normal equations are invariant to a common scaling of the
        weights, so factoring out the maximum (i.e. subtracting the smallest
        squared distance in the exponent) changes nothing analytically while
        preventing underflow at small length scales — this is what makes the
        linear-exactness and interpolation limits hold for any ``h``.
        """
        S = self._X - x[None, :]                                # (n, d)
        r2 = np.sum(S * S, axis=1)
        w = np.exp(-(r2 - float(np.min(r2))) / (2.0 * self.h * self.h))
        return w, S

    def _density(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Unshifted weight sum (sample density) and its gradient d(sum)/dx."""
        S = self._X - x[None, :]
        w = np.exp(-np.sum(S * S, axis=1) / (2.0 * self.h * self.h))
        return float(np.sum(w)), (w @ S) / (self.h * self.h)

    def _solve_local(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return standardized ``(c (m,), B (d, m))`` of the local fit at x."""
        w, S = self._weights(x)
        d = self._d
        sw = float(np.sum(w))
        ws = w @ S                                              # (d,)
        A = np.empty((d + 1, d + 1))
        A[0, 0] = sw
        A[0, 1:] = ws
        A[1:, 0] = ws
        A[1:, 1:] = (S.T * w) @ S + sw * np.eye(d)
        A[np.diag_indices_from(A)] += self.reg + 1e-12 * max(sw, 1.0)
        rhs = np.empty((d + 1, self._m))
        rhs[0] = w @ self._Yt
        rhs[1:] = (S.T * w) @ self._Yt + np.einsum("n,ndm->dm", w, self._Gt)
        try:
            theta = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(A, rhs, rcond=None)[0]
        return theta[0], theta[1:]

    # -------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Raw-unit means ``(q, m)`` and stds ``(q, m)`` at inputs ``X (q, d)``."""
        mt, st = self.predict_std(X)
        return self._ym + self._ys * mt, self._ys[None, :] * st[:, None]

    def predict_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Standardized means ``(q, m)`` and shared density-proxy std ``(q,)``."""
        X = np.atleast_2d(np.asarray(X, float))
        mean = np.empty((len(X), self._m))
        sig = np.empty(len(X))
        for i, x in enumerate(X):
            c, _ = self._solve_local(x)
            mean[i] = c
            dens, _ = self._density(x)
            sig[i] = np.sqrt(max(0.0, 1.0 - dens))
        return mean, sig

    def value_and_slope(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Raw-unit local value ``(m,)`` and diffuse slope ``(d, m)`` at ``x``."""
        c, B = self._solve_local(np.asarray(x, float))
        return self._ym + self._ys * c, B * self._ys[None, :]

    def predict_std_single(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Standardized prediction with derivatives at one point ``x (d,)``.

        Returns ``(mean (m,), sigma, dmean/dx (d, m), dsigma/dx (d,))``. The
        mean derivative is the diffuse MLS derivative (the local slope ``B``);
        the sigma derivative is exact for the density proxy.
        """
        x = np.asarray(x, float)
        c, B = self._solve_local(x)
        dens, ddens = self._density(x)
        sigma = np.sqrt(max(0.0, 1.0 - dens))
        dsig = (-ddens / (2.0 * sigma)) if sigma > 1e-12 else np.zeros(self._d)
        return c, sigma, B, dsig


class AnchoredSeparableQuadratic:
    """Center-frozen anchored model with a diagonal (separable) Hessian.

    Trust-region convergence needs the model to reproduce the incumbent's
    value and gradient exactly (a "fully linear" model), and the subproblem
    solver needs the model's value and gradient to be *mutually consistent*.
    A per-query MLS refit fails the second requirement: its diffuse slope is
    not the true derivative of its value (the weights move with the query),
    so an SQP solver fed that pair mis-converges. The remedy is to freeze
    the MLS weights at the trust-region center each iteration — the "moving"
    of Moving Least Squares happens *across* iterations, as the center and
    its weights move — which makes the model an exact polynomial::

        m(x) = f_k + G_k^T s + 1/2 sum_j q_j s_j^2,   s = x - x_k

    Anchoring is exact by construction (``m(x_k) = f_k``,
    ``dm/dx(x_k) = G_k``). The diagonal curvature ``q (d, m)`` is a
    Gaussian-weighted per-coordinate secant fit to the neighbours' gradient
    residuals (the gradient rows carry d observations per sample and
    decouple coordinate-wise; a full Hessian would need O(d^2) coefficients)::

        q_j = sum_i w_i s_ij (g_i - g_k)_j / ( sum_i w_i s_ij^2 + reg )

    with the same shifted Gaussian weights ``w_i`` and length scale ``h`` as
    the MLS. This is MMA-class structure — a separable second-order model at
    the iterate — with curvature *measured from sampled gradients* instead
    of heuristic asymptote updates.

    With ``intermediate="mma"`` the same construction is carried out in
    **MMA-style intermediate variables**: per output and per coordinate the
    model is quadratic in ``y_j`` with the reciprocal transform placed on
    the *descent* side of the gradient (as in the Method of Moving
    Asymptotes / CONLIN)::

        dfk/dx_j < 0 :  y_j = 1/(x_j - L_j),  L_j = x_kj - asy   (lower asymptote)
        dfk/dx_j > 0 :  y_j = 1/(U_j - x_j),  U_j = x_kj + asy   (upper asymptote)
        dfk/dx_j = 0 :  y_j = x_j                                 (linear)

    so ``m = f_k + sum_j c_j t_j + 1/2 q_j t_j^2`` with ``t_j = y_j - y_j(x_k)``,
    ``c_j = g_kj / (dy_j/dx_j)(x_k)`` (exact gradient anchor) and ``q_j`` the
    weighted secant in y-space. This reproduces the reciprocal-type
    monotonicity of stiffness responses that a symmetric quadratic in ``x``
    cannot represent; the asymptote distance ``asy`` must exceed the
    trust-region radius (the driver passes ``max(asy_init, 1.3*delta)``).
    """

    def __init__(self, x_k: np.ndarray, f_k: np.ndarray, G_k: np.ndarray,
                 X: np.ndarray, G: np.ndarray, h: float,
                 intermediate: str = "linear", asy: float = 0.5,
                 Y: Optional[np.ndarray] = None,
                 value_mask: Optional[np.ndarray] = None) -> None:
        self.xk = np.asarray(x_k, float)
        self.fk = np.asarray(f_k, float)        # (m,)
        self.Gk = np.asarray(G_k, float)        # (d, m)
        self.asy = float(asy)
        d, m = self.Gk.shape
        if intermediate == "mma":
            self._branch = np.sign(self.Gk).astype(int)      # (d, m) in {-1,0,1}
        else:
            self._branch = np.zeros((d, m), dtype=int)

        X = np.atleast_2d(np.asarray(X, float))
        S = X - self.xk[None, :]                             # (n, d)
        r2 = np.sum(S * S, axis=1)
        w = np.exp(-(r2 - float(np.min(r2))) / (2.0 * h * h))
        T, dydx = self._transform(X)                         # (n, d, m) both
        _, dydx0 = self._transform(self.xk[None, :])
        self._dydx0 = dydx0[0]                               # (d, m)
        self._c = self.Gk / self._dydx0                      # (d, m)
        # Hermite curvature fit: BOTH observation types enter the weighted
        # least squares for q —
        #   gradient rows (d per sample, decoupled):  t_ij q_j = h_ij
        #   value rows    (1 per sample, coupled):    1/2 sum_j q_j t_ij^2 = r_i
        # so accumulating samples sharpens the learned shape through their
        # values as well as their slopes.
        Hres = np.asarray(G, float) / dydx - self._c[None, :, :]
        A_diag = np.einsum("n,ndm,ndm->dm", w, T, T)         # (d, m)
        rhs = np.einsum("n,ndm,ndm->dm", w, T, Hres)         # (d, m)
        reg = 1e-10 * (1.0 + float(np.max(A_diag, initial=0.0)))
        self.q = np.empty((d, m))
        # Which outputs learn from function values as well as gradients:
        # default all outputs when Y is given (constraints in particular must
        # track values, since feasibility is a statement about values).
        vmask = (np.ones(m, bool) if value_mask is None
                 else np.asarray(value_mask, bool))
        self.q = rhs / (A_diag + reg)
        if Y is not None and np.any(vmask):
            Y = np.asarray(Y, float)
            Rval = Y - self.fk[None, :] - np.einsum("ndm,dm->nm", T, self._c)
            V = T * T                                        # (n, d, m)
            # Scale balance: gradient equations are O(t) while value
            # equations are O(t^2), so at local distances raw LSQ lets the
            # gradients drown the values out. Normalize each value equation
            # by its sample's ||t|| and give it mass d (one value equation
            # balancing the sample's d gradient equations).
            tnorm2 = np.sum(V, axis=1)                       # (n, m)
            wv = 4.0 * d * w[:, None] / (tnorm2 + 1e-30)     # (n, m)
            for o in np.nonzero(vmask)[0]:
                A = np.diag(A_diag[:, o] + reg)
                Vw = V[:, :, o] * wv[:, o, None]
                A += 0.25 * Vw.T @ V[:, :, o]
                b = rhs[:, o] + 0.5 * Vw.T @ Rval[:, o]
                try:
                    self.q[:, o] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:                # pragma: no cover
                    pass

    def _transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``t_j = y_j(x) - y_j(x_k)`` and ``dy_j/dx`` per output, ``(n, d, m)``.

        Points beyond an asymptote (possible only for far *training* samples;
        the subproblem box stays inside by construction) are clamped to a
        safe margin so the secant fit never sees a singular transform.
        """
        X = np.atleast_2d(np.asarray(X, float))
        S = X - self.xk[None, :]                             # (n, d)
        n, d = S.shape
        m = self.Gk.shape[1]
        Sb = np.repeat(S[:, :, None], m, axis=2)             # (n, d, m)
        T = Sb.copy()
        dydx = np.ones_like(Sb)
        margin = 0.05 * self.asy
        neg = np.broadcast_to(self._branch[None] == -1, Sb.shape)
        pos = np.broadcast_to(self._branch[None] == 1, Sb.shape)
        if np.any(neg):
            # y = 1/(x - L), L = xk - asy  ->  x - L = asy + s
            u = np.maximum(self.asy + Sb, margin)
            T = np.where(neg, 1.0 / u - 1.0 / self.asy, T)
            dydx = np.where(neg, -1.0 / (u * u), dydx)
        if np.any(pos):
            # y = 1/(U - x), U = xk + asy  ->  U - x = asy - s
            u = np.maximum(self.asy - Sb, margin)
            T = np.where(pos, 1.0 / u - 1.0 / self.asy, T)
            dydx = np.where(pos, 1.0 / (u * u), dydx)
        return T, dydx

    def value_and_slope(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Anchored raw-unit values ``(m,)`` and exact gradients ``(d, m)``."""
        T, dydx = self._transform(np.asarray(x, float)[None, :])
        t = T[0]                                             # (d, m)
        val = self.fk + np.sum(self._c * t + 0.5 * self.q * t * t, axis=0)
        grad = (self._c + self.q * t) * dydx[0]
        return val, grad


class TangentPlaneSurrogate:
    """Softmax-weighted sum of tangent hyperplanes.

    .. math::

        \\hat f_o(x) = \\sum_i \\lambda_i(x)\\, T_{i,o}(x), \\qquad
        T_{i,o}(x) = f_{i,o} + g_{i,o}^T (x - x_i), \\qquad
        \\lambda_i(x) = \\operatorname{softmax}_i\\!\\big(-\\|x-x_i\\|^2 / 2h^2\\big)

    Each sample contributes its first-order Taylor plane; the softmax of a
    distance-decreasing score blends them, with the length scale ``h``
    evolving with the sampling (the driver's min-distance rule). Key
    properties:

    * **closed form** — the analytic gradient
      ``d\\hat f/dx = sum_i lam_i g_i + sum_i (dlam_i/dx) T_i(x)`` with
      ``dlam_i/dx = lam_i (s_bar - s_i)/h^2``, ``s_i = x - x_i``,
      ``s_bar = sum_l lam_l s_l``, is *exactly* the derivative of the value:
      the surrogate can be handed to an SQP solver as-is, moving weights and
      all — no centre-freezing, no diffuse-derivative approximation;
    * reproduces affine functions exactly at any ``h`` (all planes coincide);
    * interpolates each sample's value *and* gradient in the ``h -> 0``
      limit (softmax -> nearest-plane indicator);
    * every new sample adds its local plane — accumulating data refines the
      learned shape by construction.

    All ``m`` outputs (objective + constraints) share the weights.

    Two weight families are available (``weighting``):

    * ``"shepard"`` (default) — **Hermite--Shepard cardinal weights**: scores
      ``q_i = p log( (1/d_i - 1/d_max)_+ )`` with ``d_i = ||x - x_i||^2``
      and support ``d_max = (support_factor * h)^2``. The singularity at
      ``d_i -> 0`` makes the blend a TRUE Hermite interpolant at finite
      length scale: ``alpha_i(x_k) = delta_ik``,
      ``beta_ij(x) = lambda_i (x - x_i)_j`` vanishing at every node,
      ``grad alpha_i = 0`` and ``grad beta_ij(x_k) = delta_ik e_j`` at all
      nodes (flat-spot property, p >= 1), partition of unity, monotone
      decay in the own distance, and compact support — the full set of
      Hermite shape-function constraints.
    * ``"softmax"`` — Gaussian scores ``q_i = -d_i / (2 h^2)``: everywhere
      smooth and positive, interpolating only in the ``h -> 0`` limit.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, G: np.ndarray,
                 h: float, weighting: str = "wendland", p: float = 2.0,
                 support_factor: float = 3.0) -> None:
        self.X = np.atleast_2d(np.asarray(X, float))         # (n, d)
        Y = np.asarray(Y, float)
        self.Y = Y if Y.ndim == 2 else Y[:, None]            # (n, m)
        G = np.asarray(G, float)
        self.G = G if G.ndim == 3 else G[:, :, None]         # (n, d, m)
        self.h = float(h)
        self.weighting = weighting
        self.p = float(p)
        self.dmax = (float(support_factor) * self.h) ** 2
        if weighting == "wendland":
            # Separation-aware per-point radii: each bump's support must not
            # reach any other node, so cardinality holds exactly with SMOOTH
            # bounded weights - no singularity, no boundary-layer flat spots.
            n = len(self.X)
            if n > 1:
                D = np.linalg.norm(
                    self.X[:, None, :] - self.X[None, :, :], axis=2)
                D[np.diag_indices(n)] = np.inf
                r_nn = D.min(axis=1)
            else:
                r_nn = np.array([1.0])
            self.rho = 0.9 * np.minimum(
                r_nn, float(support_factor) * self.h)        # (n,)

    # ---------------------------------------------------------------- scores
    def _scores(self, S: np.ndarray):
        """Scores ``q`` and their gradients ``dq`` for offsets ``S = x - x_i``.

        Returns ``(q (..., n), dq (..., n, d))`` with ``lambda = softmax(q)``.
        Rows outside the Shepard support get ``q = -inf`` (zero weight); if
        ALL rows fall outside, the nearest sample is kept (minimal support
        extension so the blend is defined everywhere in the box).
        """
        d2 = np.sum(S * S, axis=-1)                          # (..., n)
        if self.weighting == "softmax":
            q = -d2 / (2.0 * self.h * self.h)
            dq = -S / (self.h * self.h)
            return q, dq
        if self.weighting == "wendland":
            # Wendland C2 bump b(t) = (1-t)^4 (4t+1), t = r/rho_i, flat at
            # the centre (b'(0) = 0) and compactly supported at t = 1.
            r = np.sqrt(d2)
            t = r / self.rho                                 # (..., n)
            inside = t < 1.0
            if not np.all(np.any(inside, axis=-1)):          # empty coverage
                nearest = np.argmin(d2, axis=-1)
                idx = np.expand_dims(nearest, -1)
                np.put_along_axis(inside, idx, True, axis=-1)
            tc = np.where(inside, np.minimum(t, 1.0 - 1e-12), 0.0)
            b = (1.0 - tc) ** 4 * (4.0 * tc + 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where(inside, np.log(np.maximum(b, 1e-300)), -np.inf)
                # db/dx = -20 (1-t)^3 / rho^2 * s ; dq = db/(b dx)
                coef = np.where(
                    inside,
                    -20.0 / ((1.0 - tc) * (4.0 * tc + 1.0)
                             * self.rho * self.rho),
                    0.0,
                )
            dq = coef[..., None] * S
            return q, dq
        with np.errstate(divide="ignore", invalid="ignore"):
            d2c = np.maximum(d2, 1e-30)
            R = 1.0 / d2c - 1.0 / self.dmax
            inside = R > 0.0
            if not np.all(np.any(inside, axis=-1)):          # empty support
                nearest = np.argmin(d2, axis=-1)
                idx = np.expand_dims(nearest, -1)
                np.put_along_axis(inside, idx, True, axis=-1)
                R = np.maximum(R, 1e-300)
            Rc = np.where(inside, np.maximum(R, 1e-300), 1.0)
            q = np.where(inside, self.p * np.log(Rc), -np.inf)
            # dq = p/R * dR/dx = p/R * (-1/d^2) * 2 s (zero outside support)
            coef = np.where(inside, -2.0 * self.p / (Rc * d2c * d2c), 0.0)
        dq = coef[..., None] * S
        return q, dq

    def value_and_slope(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Exact values ``(m,)`` and analytic gradients ``(d, m)`` at ``x``."""
        x = np.asarray(x, float)
        S = x[None, :] - self.X                              # (n, d) = x - x_i
        q, dq = self._scores(S)
        q = q - np.max(q)
        w = np.exp(q)
        lam = w / np.sum(w)                                  # (n,)
        T = self.Y + np.einsum("nd,ndm->nm", S, self.G)      # (n, m) planes
        val = lam @ T                                        # (m,)
        dq_bar = lam @ dq                                    # (d,)
        dlam = lam[:, None] * (dq - dq_bar[None, :])         # (n, d)
        grad = np.einsum("n,ndm->dm", lam, self.G) + dlam.T @ T
        return val, grad

    def values(self, Xq: np.ndarray) -> np.ndarray:
        """Vectorized values ``(q, m)`` at query points ``Xq (q, d)`` — the
        cheap batch evaluation used by the global subproblem scan."""
        Xq = np.atleast_2d(np.asarray(Xq, float))
        S = Xq[:, None, :] - self.X[None, :, :]              # (q, n, d)
        q, _ = self._scores(S)
        q = q - q.max(axis=1, keepdims=True)
        w = np.exp(q)
        lam = w / w.sum(axis=1, keepdims=True)               # (q, n)
        T = self.Y[None, :, :] + np.einsum("qnd,ndm->qnm", S, self.G)
        return np.einsum("qn,qnm->qm", lam, T)               # (q, m)


class ProductHermiteSurrogate:
    """Product-form Hermite shape functions (Coniglio construction).

    .. math::

        \\hat f(x) = \\sum_i \\hat\\alpha_i(x) f_i
                     + \\sum_{i,j} \\beta_{ij}(x)\\, (\\nabla f_i)_j

    built from the cubic smoothstep weight in normalized *squared* distance
    ``W(t) = 2t^3 - 3t^2 + 1``, ``t = d_i/d_max`` (flat at both ends,
    compactly supported) and the truncated quintic
    ``gamma(t) = t (t^2 - 1)^2`` (``gamma(0)=0, gamma'(0)=1``, value and
    slope zero at ``|t| = 1``, zero beyond):

    .. math::

        \\alpha_i = \\frac{W_i(x) \\prod_{l \\ne i} (1 - W_l(x))}
                          {\\prod_{l \\ne i} (1 - W_l(x_i))}, \\qquad
        \\hat\\alpha_i = \\alpha_i / \\textstyle\\sum_l \\alpha_l,

    .. math::

        \\beta_{ij} = r_{max}\\, \\gamma\\!\\Bigl(\\tfrac{x_j - x_{ij}}{r_{max}}\\Bigr)
                      \\, \\frac{W_i(x) \\prod_{l \\ne i}(1 - W_l(x))}
                               {\\prod_{l \\ne i}(1 - W_l(x_i))} .

    The decisive property: **cardinality comes from product zeros, not from
    support exclusion** — each factor ``(1 - W_l)`` vanishes *with zero
    slope* at ``x_l`` (flat top of the smoothstep), so the complete Hermite
    constraint set (``alpha_i(x_k) = delta_ik``, ``beta_ij(x_k) = 0``,
    ``grad alpha_i(x_k) = 0``, ``grad beta_ij(x_k) = delta_ik e_j``,
    monotone decay, compact support ``d_max``, partition of unity where
    covered) holds for ANY node spacing and ANY ``d_max`` — no separation
    radii, no singular weights. Where no weight covers ``x`` the nearest
    sample's tangent plane takes over. All outputs share the shape
    functions.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, G: np.ndarray,
                 h: float, support_factor: float = 3.0) -> None:
        self.X = np.atleast_2d(np.asarray(X, float))         # (n, d)
        Y = np.asarray(Y, float)
        self.Y = Y if Y.ndim == 2 else Y[:, None]            # (n, m)
        G = np.asarray(G, float)
        self.G = G if G.ndim == 3 else G[:, :, None]         # (n, d, m)
        self.rmax = float(support_factor) * float(h)
        self.dmax = self.rmax ** 2
        n = len(self.X)
        # per-node normalizers N_i = prod_{l != i} (1 - W_l(x_i))
        self.N = np.empty(n)
        for i in range(n):
            w = self._W(self.X[i])[0]
            self.N[i] = max(np.prod(np.delete(1.0 - w, i)), 1e-300)

    # ------------------------------------------------------------------ w
    def _W(self, x: np.ndarray):
        """Smoothstep weights and their gradients at one point."""
        S = x[None, :] - self.X                              # (n, d)
        dd = np.sum(S * S, axis=1)
        t = np.clip(dd / self.dmax, 0.0, 1.0)
        w = 2.0 * t ** 3 - 3.0 * t * t + 1.0
        # dW/dx = W'(t)/dmax * 2 (x - x_i);  W'(t) = 6t^2 - 6t
        dw = ((6.0 * t * t - 6.0 * t) / self.dmax)[:, None] * (2.0 * S)
        return w, dw, S

    @staticmethod
    def _loo(u: np.ndarray):
        """Leave-one-out products ``P[i] = prod_{m != i} u_m`` (zero-aware)."""
        zero = np.nonzero(u == 0.0)[0]
        n = len(u)
        if len(zero) == 0:
            Q = np.prod(u)
            return Q / u
        if len(zero) == 1:
            P = np.zeros(n)
            p = zero[0]
            P[p] = np.prod(np.delete(u, p))
            return P
        return np.zeros(n)

    def value_and_slope(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Values ``(m,)`` and exact analytic gradients ``(d, m)`` at ``x``."""
        x = np.asarray(x, float)
        n, d = self.X.shape
        w, dw, S = self._W(x)
        u = 1.0 - w
        P = self._loo(u)                                     # (n,)
        # dP_i/dx = sum_{l != i} (-dw_l) prod_{m not in {i,l}} u_m
        dP = np.zeros((n, d))
        for i in range(n):
            ui = np.delete(u, i)
            dwi = np.delete(dw, i, axis=0)
            Pil = self._loo(ui)                              # (n-1,) leave-two-out
            dP[i] = -(Pil[:, None] * dwi).sum(axis=0)
        alpha = w * P / self.N                               # (n,)
        dalpha = (dw * P[:, None] + w[:, None] * dP) / self.N[:, None]
        Ssum = float(alpha.sum())
        if Ssum <= 1e-300:                                   # no coverage
            i0 = int(np.argmin(np.sum(S * S, axis=1)))
            val = self.Y[i0] + S[i0] @ self.G[i0]
            return val, self.G[i0].copy()
        dSsum = dalpha.sum(axis=0)                           # (d,)
        ah = alpha / Ssum
        dah = (dalpha * Ssum - alpha[:, None] * dSsum[None, :]) / (Ssum * Ssum)
        # beta_ij and gradient
        t = S / self.rmax                                    # (n, d)
        inside = np.abs(t) < 1.0
        g = np.where(inside, t * (t * t - 1.0) ** 2, 0.0)    # gamma(t)
        gp = np.where(inside, (t * t - 1.0) * (5.0 * t * t - 1.0), 0.0)
        # beta_ij = rmax gamma(t_ij) alpha_hat_i: the NORMALIZED alpha is the
        # envelope, so |beta| <= 0.19 rmax is bounded by construction (the
        # raw per-node normalizer 1/N_i amplifies between nodes when the
        # supports overlap; all node conditions are unchanged since
        # gamma(0)=0, gamma'(0)=1, alpha_hat_i(x_i)=1, grad alpha_hat=0).
        B = self.rmax * g * ah[:, None]                      # (n, d) = beta_ij
        val = ah @ self.Y + np.einsum("nj,njm->m", B, self.G)
        # dbeta_ij/dx_k = rmax g_ij dah_ik + gamma'_ij delta_jk ah_i
        grad = dah.T @ self.Y                                # value part (d, m)
        grad += np.einsum("nj,nk,njm->km", self.rmax * g, dah, self.G)
        grad += np.einsum("nj,njm->jm", gp * ah[:, None], self.G)
        return val, grad

    def values(self, Xq: np.ndarray) -> np.ndarray:
        """Batch values ``(q, m)`` (loop; used by the global subproblem scan)."""
        Xq = np.atleast_2d(np.asarray(Xq, float))
        return np.stack([self.value_and_slope(xq)[0] for xq in Xq])


class PlanarMLSModel:
    """Center-frozen planar Hermite MLS model (no curvature term).

    The per-query MLS's local weighted *linear* regression, evaluated once
    with the weights frozen at the trust-region center: its value ``c`` and
    slope ``B`` blend the neighbours' sampled **values and gradients**
    (planar Hermite fit), so the plane tilts toward the data trend instead
    of being the incumbent's tangent. Because the frozen fit is a polynomial
    it is exactly value/gradient-consistent for the subproblem solver. It
    does not hard-interpolate the incumbent — trust-region theory only needs
    fully-linear accuracy — and as the length scale shrinks it approaches
    interpolation of the nearest samples.
    """

    def __init__(self, mls: MovingLeastSquares, center: np.ndarray) -> None:
        self.xk = np.asarray(center, float)
        self.fk_model, self.B = mls.value_and_slope(self.xk)   # (m,), (d, m)

    def value_and_slope(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = np.asarray(x, float) - self.xk
        return self.fk_model + s @ self.B, self.B


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
class MLSSBOptimizer:
    """Trust-region-managed batch SBO on a gradient-enhanced MLS surrogate.

    Same evaluation contract as :class:`scp_uno.gesbo_core.GESBOptimizer`
    (``evaluate: x -> (f, grad_f, cons, jac_cons)``, ``c(x) <= 0``).
    """

    def __init__(
        self,
        evaluate: EvaluateFn,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        config: Optional[MLSSBOConfig] = None,
        on_iteration: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.evaluate = evaluate
        self.lb = np.asarray(lower_bounds, float)
        self.ub = np.asarray(upper_bounds, float)
        self.scale = np.maximum(self.ub - self.lb, 1e-30)
        self.cfg = config or MLSSBOConfig()
        self.on_iteration = on_iteration
        self._rng = np.random.default_rng(self.cfg.seed)
        # sample store (normalized coordinates)
        self.X: List[np.ndarray] = []
        self.F: List[float] = []
        self.Gf: List[np.ndarray] = []
        self.C: List[np.ndarray] = []
        self.Jc: List[np.ndarray] = []
        self.n_evals = 0

    # ------------------------------------------------------------- sampling
    def _eval(self, xi: np.ndarray) -> int:
        x = self.lb + self.scale * xi
        f, g, c, J = self.evaluate(x)
        g = np.asarray(g, float).flatten() * self.scale
        c = np.asarray(c, float).flatten()
        J = (np.asarray(J, float).reshape(len(c), len(self.lb))
             * self.scale[None, :])
        self.X.append(np.asarray(xi, float).copy())
        self.F.append(float(f))
        self.Gf.append(g)
        self.C.append(c)
        self.Jc.append(J)
        self.n_evals += 1
        return len(self.X) - 1

    def _merit(self, i: int) -> float:
        viol = float(np.sum(np.maximum(0.0, self.C[i]))) if len(self.C[i]) else 0.0
        return self.F[i] + self.cfg.penalty * viol

    def _violation(self, i: int) -> float:
        return float(np.max(self.C[i])) if len(self.C[i]) else -np.inf

    def _best_index(self) -> int:
        feas = [i for i in range(len(self.X))
                if self._violation(i) <= self.cfg.constraint_tol]
        if feas:
            return min(feas, key=lambda i: self.F[i])
        return min(range(len(self.X)), key=self._violation)

    # ------------------------------------------------------------ surrogate
    def _fit_surrogate(self, center: np.ndarray, h: float,
                       center_i: Optional[int] = None, delta: float = 0.25):
        cfg = self.cfg
        Xa = np.asarray(self.X)
        n = len(Xa)
        order = np.argsort(np.linalg.norm(Xa - center[None, :], axis=1))
        keep = order[: min(cfg.max_points, n)]
        ib = self._best_index()
        if ib not in keep:
            keep = np.concatenate([keep[:-1], [ib]])
        keep = np.array(sorted(set(int(i) for i in keep)))

        m = len(self.C[0])
        Y = np.column_stack(
            [np.asarray(self.F)[keep]]
            + [np.array([self.C[i][c] for i in keep]) for c in range(m)]
        ) if m else np.asarray(self.F)[keep][:, None]
        G = np.stack([
            np.column_stack([self.Gf[i]] + [self.Jc[i][c] for c in range(m)])
            for i in keep
        ])                                                       # (nk, d, m+1)

        mls = MovingLeastSquares(
            lengthscale=h, regularization=cfg.regularization
        ).fit(Xa[keep], Y, G)
        cons_shift = (mls._ym[1:] / mls._ys[1:]) if m else np.empty(0)

        anchored = None
        if cfg.anchor_center and cfg.model == "product":
            anchored = ProductHermiteSurrogate(
                Xa[keep], Y, G, h, support_factor=cfg.support_factor)
        elif cfg.anchor_center and cfg.model == "tangent":
            # De-jam: enforce a minimum pairwise separation in the window
            # (keep the center and the incumbent best, then nearest-first).
            sel = self._thin(keep, center, h * cfg.min_sep_frac,
                             center_i=center_i)
            pos = np.searchsorted(keep, sel)
            anchored = TangentPlaneSurrogate(
                Xa[sel], Y[pos], G[pos], h, weighting=cfg.weighting,
                support_factor=cfg.support_factor)
        elif cfg.anchor_center and cfg.model == "planar":
            ci = center_i if center_i is not None else self._best_index()
            anchored = PlanarMLSModel(mls, self.X[ci])
        elif cfg.anchor_center:
            ci = center_i if center_i is not None else self._best_index()
            f_k = np.concatenate([[self.F[ci]], self.C[ci]])          # (m+1,)
            G_k = np.column_stack([self.Gf[ci]]
                                  + [self.Jc[ci][c] for c in range(m)])  # (d, m+1)
            # Bandwidth floor: cover at least min_fit_neighbors samples, so
            # that accumulating data IMPROVES the learned shape instead of
            # shrinking the fit's support (h alone tracks the *minimum*
            # sample distance, which collapses as sampling densifies).
            dists = np.sort(np.linalg.norm(
                Xa[keep] - np.asarray(self.X[ci])[None, :], axis=1))
            dists = dists[dists > 1e-14]
            k_nn = min(cfg.min_fit_neighbors, len(dists))
            h_fit = max(h, float(dists[k_nn - 1])) if k_nn else h
            fv = cfg.fit_values
            if isinstance(fv, bool):                    # tolerate old configs
                fv = "all" if fv else "none"
            value_mask = np.array(
                [fv == "all"] + [fv in ("all", "constraints")] * m, bool)
            anchored = AnchoredSeparableQuadratic(
                self.X[ci], f_k, G_k, Xa[keep], G, h_fit,
                intermediate=cfg.intermediate,
                asy=max(cfg.asy_init, 1.3 * delta),
                Y=(Y if np.any(value_mask) else None), value_mask=value_mask)
        return mls, cons_shift, anchored

    def _thin(self, keep: np.ndarray, center: np.ndarray, sep: float,
              center_i: Optional[int] = None) -> np.ndarray:
        """Greedy minimum-separation thinning of the window ``keep``.

        Priority order: the trust-region center sample, the incumbent best,
        then remaining points nearest the center first; any point closer
        than ``sep`` to an already-kept one is dropped (jammed clusters
        collapse to their leading representative). Returns sorted indices
        into the global archive.
        """
        Xa = np.asarray(self.X)
        prio: List[int] = []
        if center_i is not None and center_i in keep:
            prio.append(int(center_i))
        ib = self._best_index()
        if ib in keep and ib not in prio:
            prio.append(int(ib))
        rest = [int(i) for i in
                keep[np.argsort(np.linalg.norm(
                    Xa[keep] - center[None, :], axis=1))]
                if int(i) not in prio]
        sel: List[int] = []
        for i in prio + rest:
            xi = Xa[i]
            if all(np.linalg.norm(xi - Xa[j]) >= sep for j in sel):
                sel.append(i)
        return np.array(sorted(sel))

    def _min_dist_in_tr(self, center: np.ndarray, delta: float) -> float:
        """Minimal distance from the center to another sample inside the
        trust-region box (falling back to the nearest sample overall)."""
        Xa = np.asarray(self.X)
        inside = np.all(np.abs(Xa - center[None, :]) <= delta + 1e-15, axis=1)
        dist = np.linalg.norm(Xa[inside] - center[None, :], axis=1)
        dist = dist[dist > 1e-14]
        if len(dist) == 0:
            dist = np.linalg.norm(Xa - center[None, :], axis=1)
            dist = dist[dist > 1e-14]
        return float(np.min(dist)) if len(dist) else self.cfg.ls_min

    def _solve_subproblem(self, anchored: AnchoredSeparableQuadratic,
                          center: np.ndarray,
                          best: np.ndarray, lo: np.ndarray, hi: np.ndarray
                          ) -> Tuple[np.ndarray, str]:
        """Solve ``min m_0(x) s.t. m_j(x) <= 0`` on the anchored model inside
        the trust-region box (the MMA-like exploitation step).

        Returns the candidate and a tag describing which path produced it:
        ``slsqp`` (constrained solve), ``phase1`` (feasibility restoration
        first), or ``penalty`` (L-BFGS-B on the l1-penalized model merit).
        """
        d = len(center)
        m = len(self.C[0])
        cache: dict = {}

        def eval_model(x: np.ndarray):
            key = x.tobytes()
            if key not in cache:
                cache[key] = anchored.value_and_slope(x)
            return cache[key]

        bounds = list(zip(lo, hi))
        starts = [np.clip(center, lo, hi), np.clip(best, lo, hi)]
        starts += [lo + (hi - lo) * self._rng.random(d) for _ in range(2)]

        def model_merit(x):
            v, _ = eval_model(x)
            viol = float(np.sum(np.maximum(0.0, v[1:]))) if m else 0.0
            return float(v[0]) + self.cfg.penalty * viol

        # Global phase: dense candidate scan over the trust-region box (the
        # surrogate is closed-form cheap), best candidates become SLSQP
        # starts. This is what finds the right valley of a multimodal model.
        n_glob = int(getattr(self.cfg, "n_global", 0))
        if n_glob > 0:
            sampler = qmc.LatinHypercube(
                d=d, seed=int(self._rng.integers(2 ** 31)))
            cands = lo + (hi - lo) * sampler.random(n_glob)
            if hasattr(anchored, "values"):
                V = anchored.values(cands)                       # (q, m+1)
            else:
                V = np.array([anchored.value_and_slope(c)[0] for c in cands])
            mer = V[:, 0]
            if m:
                mer = mer + self.cfg.penalty * np.sum(
                    np.maximum(0.0, V[:, 1:]), axis=1)
            starts += [cands[i] for i in np.argsort(mer)[:3]]

        cand, tag = None, "slsqp"
        if m:
            cons = [{
                "type": "ineq",
                "fun": (lambda x, j=j: -float(eval_model(x)[0][1 + j])),
                "jac": (lambda x, j=j: -eval_model(x)[1][:, 1 + j]),
            } for j in range(m)]
        else:
            cons = ()
        for x0 in starts:
            try:
                res = minimize(
                    lambda x: (float(eval_model(x)[0][0]), eval_model(x)[1][:, 0]),
                    x0, jac=True, method="SLSQP", bounds=bounds,
                    constraints=cons,
                    options={"maxiter": self.cfg.subproblem_maxiter,
                             "ftol": 1e-12},
                )
            except Exception:                                # pragma: no cover
                continue
            x_c = np.clip(res.x, lo, hi)
            if cand is None or model_merit(x_c) < model_merit(cand):
                cand = x_c

        # Feasibility restoration when SLSQP found nothing useful: minimize the
        # model violation first, then accept the restored point.
        if m and (cand is None or model_merit(cand) >= model_merit(center)):
            def viol_fg(x):
                v, gr = eval_model(x)
                t = np.maximum(0.0, v[1:])
                return float(np.sum(t * t)), 2.0 * gr[:, 1:] @ t
            res = minimize(viol_fg, np.clip(center, lo, hi), jac=True,
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": self.cfg.subproblem_maxiter})
            x_c = np.clip(res.x, lo, hi)
            if model_merit(x_c) < model_merit(cand if cand is not None else center):
                cand, tag = x_c, "phase1"

        if cand is None or not np.all(np.isfinite(cand)):
            res = minimize(
                lambda x: model_merit(x), np.clip(center, lo, hi),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": self.cfg.subproblem_maxiter})
            cand, tag = np.clip(res.x, lo, hi), "penalty"
        return cand, tag

    def _predicted_merit_drop(self, mls, xi_from, xi_to, anchored=None) -> float:
        if anchored is not None:
            rows = [anchored.value_and_slope(np.asarray(xi_from))[0],
                    anchored.value_and_slope(np.asarray(xi_to))[0]]
        else:
            rows, _ = mls.predict(np.asarray([xi_from, xi_to]))

        def merit(row):
            viol = float(np.sum(np.maximum(0.0, row[1:]))) if len(row) > 1 else 0.0
            return float(row[0]) + self.cfg.penalty * viol

        return merit(rows[0]) - merit(rows[1])

    # ----------------------------------------------------------------- run
    def run(self, x0: np.ndarray) -> MLSSBOResult:
        cfg = self.cfg
        d = len(self.lb)
        xi0 = np.clip((np.asarray(x0, float) - self.lb) / self.scale, 0.0, 1.0)
        delta = cfg.tr_init
        history: List[dict] = []
        status = "max_outer_iter reached"

        # ---- initial DOE: x0 + LHS inside the initial trust region ----
        n_init = cfg.n_init if cfg.n_init is not None else cfg.batch_size + 1
        self._eval(xi0)
        if n_init > 1:
            lo = np.maximum(0.0, xi0 - delta)
            hi = np.minimum(1.0, xi0 + delta)
            S = qmc.LatinHypercube(d=d, seed=cfg.seed).random(n_init - 1)
            for row in S:
                if self.n_evals >= cfg.max_evals:
                    break
                self._eval(lo + (hi - lo) * row)

        center = xi0.copy()
        center_i = 0
        stall = 0
        W = np.eye(d)

        resets = 0
        for it in range(cfg.max_outer_iter):
            if self.n_evals >= cfg.max_evals:
                status = "evaluation budget exhausted"
                break
            if delta < cfg.tr_min or (stall >= cfg.stall_limit
                                      and delta <= cfg.tr_min * 10):
                # Restart from the incumbent best instead of terminating: MMA
                # spends its whole budget, so a fair surrogate driver must too.
                if resets >= cfg.n_resets:
                    status = ("trust region collapsed" if delta < cfg.tr_min
                              else "stalled")
                    break
                resets += 1
                bi = self._best_index()
                center, center_i = self.X[bi].copy(), bi
                # cycle the restart radius (full / half / quarter of tr_init)
                # so successive restarts are not identical, and spend one
                # evaluation on a random point in the new region: the refit
                # then differs from the run that just stalled instead of
                # deterministically reproducing it.
                delta = max(cfg.tr_init / 2.0 ** ((resets - 1) % 3),
                            cfg.tr_min * 100.0)
                stall = 0
                if self.n_evals < cfg.max_evals:
                    lo_r = np.maximum(0.0, center - delta)
                    hi_r = np.minimum(1.0, center + delta)
                    self._eval(lo_r + (hi_r - lo_r) * self._rng.random(d))

            # length scale tracks the sample spacing inside the trust region
            d_min = self._min_dist_in_tr(center, delta)
            h = float(np.clip(cfg.ls_factor * d_min, cfg.ls_min, cfg.ls_max))
            mls, cons_shift, anchored = self._fit_surrogate(
                center, h, center_i, delta)
            lo = np.maximum(0.0, center - delta)
            hi = np.minimum(1.0, center + delta)
            best_i = self._best_index()
            if anchored is not None and cfg.batch_size == 1:
                # sequential (MMA-regime) step: solved constrained subproblem
                x_new, sub_tag = self._solve_subproblem(
                    anchored, center, self.X[best_i], lo, hi)
                if float(np.max(np.abs(x_new - center))) < 1e-12:
                    # model KKT point at the center: no descent is possible in
                    # this radius — classical null step, shrink WITHOUT
                    # spending a true evaluation on a duplicate of the center.
                    delta *= cfg.tr_shrink
                    stall += 1
                    rec = {
                        "iter": it, "n_evals": self.n_evals, "delta": delta,
                        "lengthscale": h, "min_dist": d_min, "pred_error": 0.0,
                        "batch_indices": [], "rho": 0.0,
                        "resets": resets, "subproblem": "null",
                        "merit_center": self._merit(center_i),
                        "best_f": self.F[self._best_index()],
                    }
                    history.append(rec)
                    if self.on_iteration is not None:
                        self.on_iteration(rec)
                    continue
                batch = [x_new]
            else:
                batch = propose_batch(
                    mls, W, center, self.X[best_i], lo, hi, cfg, self._rng,
                    cons_shift,
                )
                sub_tag = "batch"
                if anchored is not None:
                    # exploitation point of the batch also uses the anchored model
                    x_new, tag = self._solve_subproblem(
                        anchored, center, self.X[best_i], lo, hi)
                    if float(np.max(np.abs(x_new - center))) >= 1e-12:
                        batch[0], sub_tag = x_new, tag

            # surrogate predictions recorded *before* evaluating the truth
            pred_std = mls.predict_std(np.asarray(batch))[0][:, 0]

            new_idx = []
            for xi in batch:
                if self.n_evals >= cfg.max_evals:
                    break
                new_idx.append(self._eval(xi))
            if not new_idx:
                status = "evaluation budget exhausted"
                break

            # prediction error over the batch (diagnostics only)
            true_std = (np.asarray([self.F[i] for i in new_idx]) - mls._ym[0]) \
                / mls._ys[0]
            err = float(np.median(np.abs(pred_std[: len(new_idx)] - true_std)))

            # ---- trust-region ratio test on the l1 merit ----
            cand_i = min(new_idx, key=self._merit)
            actual = self._merit(center_i) - self._merit(cand_i)
            pred = self._predicted_merit_drop(mls, self.X[center_i],
                                              self.X[cand_i], anchored)
            rho = actual / max(pred, 1e-14) if pred > 0 else (
                np.inf if actual > 0 else -np.inf)

            step = float(np.max(np.abs(self.X[cand_i] - center)))
            if actual > 0 and rho >= cfg.eta_accept:
                center, center_i = self.X[cand_i].copy(), cand_i
            # classical three-zone radius update: shrink only on poor
            # agreement, hold in the middle band, expand on strong agreement
            # at the trust-region boundary.
            if actual <= 0 or rho < cfg.eta_shrink:
                delta *= cfg.tr_shrink
            elif rho >= cfg.eta_expand and step >= 0.8 * delta:
                delta = min(delta * cfg.tr_expand, cfg.tr_max)

            stall = stall + 1 if actual <= cfg.ftol_abs else 0
            rec = {
                "iter": it, "n_evals": self.n_evals, "delta": delta,
                "lengthscale": h, "min_dist": d_min, "pred_error": err,
                "batch_indices": new_idx, "rho": float(rho),
                "resets": resets, "subproblem": sub_tag,
                "merit_center": self._merit(center_i),
                "best_f": self.F[self._best_index()],
            }
            history.append(rec)
            if self.on_iteration is not None:
                self.on_iteration(rec)
            LOGGER.info(
                "MLS-SBO iter %d: evals=%d f_best=%.6e h=%.3e delta=%.3e "
                "err=%.3f rho=%.2f resets=%d",
                it, self.n_evals, rec["best_f"], h, delta, err, rho, resets,
            )

        ib = self._best_index()
        return MLSSBOResult(
            x_opt=self.lb + self.scale * self.X[ib],
            f_opt=self.F[ib],
            constraints=self.C[ib].copy(),
            is_feasible=self._violation(ib) <= cfg.constraint_tol,
            n_evals=self.n_evals,
            n_iter=len(history),
            status=status,
            history=history,
        )


def mls_sbo_minimize(
    evaluate: EvaluateFn,
    x0: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    config: Optional[MLSSBOConfig] = None,
    on_iteration: Optional[Callable[[dict], None]] = None,
) -> MLSSBOResult:
    """Functional entry point: minimize ``f(x)`` s.t. ``c(x) <= 0``, box bounds."""
    return MLSSBOptimizer(
        evaluate, lower_bounds, upper_bounds, config, on_iteration
    ).run(x0)
