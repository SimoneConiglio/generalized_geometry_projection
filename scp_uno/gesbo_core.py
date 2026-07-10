# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Core engine for Gradient-Enhanced Surrogate-Based Optimization (GE-SBO).

This module is **pure NumPy/SciPy** (no GEMSEO, no FEniCS) so it can be unit
tested in isolation; :mod:`scp_uno.gesbo` provides the thin GEMSEO wrapper that
registers the ``GE_SBO`` algorithm.

Algorithm overview
------------------
GE-SBO is a trust-region-managed, surrogate-based optimizer designed for
*expensive* objective/constraint functions whose **gradients are available**
(e.g. via adjoint sensitivities, as in topology optimization):

1. **Gradient-enhanced Gaussian-process surrogate (GEK).**
   The surrogate is a GP conditioned on both function values *and* gradient
   observations (derivative observations are linear functionals of a GP, so
   they enter the kriging system exactly). Each sample therefore contributes
   ``1 + r`` pieces of information instead of 1, which drastically reduces the
   number of expensive evaluations needed for a given model accuracy.

2. **Active-subspace dimension reduction (high-dimensional scaling).**
   Full GEK in dimension ``d`` costs ``O((n(d+1))^3)``; that does not scale.
   When ``d > max_latent_dim`` the gradients collected so far are stacked and
   an SVD extracts the dominant *active subspace* ``W`` (d x r, r small). The
   surrogate is built on the reduced coordinates ``z = W^T x``; the projected
   gradients ``W^T g`` are exact directional-derivative observations along the
   subspace directions. The subspace is re-identified at every iteration from
   the current gradient set, so it tracks the optimizer trajectory.

3. **Batch (multi-point) acquisition.**
   Each outer iteration proposes ``q = batch_size`` points inside the current
   trust region:

   * point 1 minimizes the (constraint-penalized) surrogate mean —
     exploitation;
   * points 2..q minimize a lower-confidence-bound ``mean - kappa_j * sigma``
     with an increasing ``kappa`` ladder plus a smooth repulsion term from the
     points already selected in the batch — a geometric spread from
     exploitation to exploration.

   The q candidates are independent, so the true model can evaluate them in
   parallel (the driver itself evaluates them through a single ``evaluate``
   callable; parallelism belongs to the caller).

4. **Trust-region model management.**
   The best batch point (by an l1 penalty merit function) is compared with the
   incumbent; the classical ratio test (actual vs. surrogate-predicted merit
   reduction) moves the trust-region center and expands/shrinks its radius.
   This is what makes the method converge reliably even when the surrogate is
   poor far from the data.

References: gradient-enhanced kriging (Morris, Mitchell & Ylvisaker 1993),
GEK with partial-least-squares/high-dim scaling (Bouhlel & Martins 2019,
GE-KPLS), active subspaces (Constantine 2015), q-point lower-confidence-bound
batch strategies (Hutter et al.; Desautels et al. 2014).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import qmc

LOGGER = logging.getLogger(__name__)

# Type of the user-supplied evaluation callable:
#   x (d,) -> (f: float, grad_f: (d,), cons: (m,), jac_cons: (m, d))
# For unconstrained problems return empty arrays of shape (0,) and (0, d).
EvaluateFn = Callable[[np.ndarray], Tuple[float, np.ndarray, np.ndarray, np.ndarray]]


# --------------------------------------------------------------------------- #
# Configuration & result containers
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class GESBOConfig:
    """Configuration of the GE-SBO driver (all lengths in *normalized* [0,1] space)."""

    batch_size: int = 4                # q: points acquired per outer iteration
    n_init: Optional[int] = None       # initial DOE size (default: batch_size + 1)
    max_evals: int = 200               # total budget of true evaluations
    max_outer_iter: int = 100          # cap on outer (batch) iterations

    # -- surrogate / scaling --
    max_latent_dim: int = 12           # active-subspace size when d is large
    max_points: int = 60               # training window: nearest points kept
    max_grad_points: int = 25          # subset of the window carrying gradient info
    use_gradients: bool = True         # False => plain (value-only) kriging
    regularization: float = 1e-8       # kernel nugget
    lengthscale_grid: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)

    # -- trust region --
    tr_init: float = 0.25              # initial half-width (fraction of unit box)
    tr_min: float = 1e-5
    tr_max: float = 0.75
    tr_shrink: float = 0.5
    tr_expand: float = 2.0
    eta_accept: float = 1e-4           # rho >= eta_accept  => move center
    eta_expand: float = 0.5            # rho >= eta_expand  => grow radius

    # -- acquisition --
    kappa_base: float = 1.0            # first exploration kappa
    kappa_growth: float = 2.0          # geometric ladder kappa_j = base*growth^(j-1)
    repulsion_weight: float = 1.0      # batch-diversity penalty weight
    n_restarts: int = 4                # random multistarts per acquisition solve
    acq_maxiter: int = 60              # L-BFGS-B iterations per start
    constraint_penalty_acq: float = 10.0   # smooth hinge weight in the acquisition

    # -- merit / stopping --
    penalty: float = 100.0             # mu of the l1 merit f + mu*sum(max(0, c))
    constraint_tol: float = 1e-6       # feasibility tolerance on raw constraints
    ftol_abs: float = 1e-10            # stall detection on the merit value
    stall_limit: int = 5               # consecutive stalled iterations before stop

    seed: int = 0


@dataclasses.dataclass
class GESBOResult:
    """Outcome of a GE-SBO run (raw, unnormalized coordinates)."""

    x_opt: np.ndarray
    f_opt: float
    constraints: np.ndarray
    is_feasible: bool
    n_evals: int
    n_iter: int
    status: str
    history: List[dict]


# --------------------------------------------------------------------------- #
# Gradient-enhanced Gaussian process (GEK)
# --------------------------------------------------------------------------- #
class GradientEnhancedGP:
    """Multi-output GP with (optional) derivative observations, Gaussian kernel.

    All outputs (objective + constraints) share the training inputs, kernel and
    Cholesky factorization: one factorization serves ``m + 1`` right-hand sides.

    Inputs are standardized internally; each output is standardized to zero
    mean / unit variance, so :meth:`predict_std` returns dimensionless means
    and a shared dimensionless standard deviation — the natural units for
    lower-confidence-bound acquisition.
    """

    def __init__(
        self,
        regularization: float = 1e-8,
        lengthscale_grid: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
        use_gradients: bool = True,
    ) -> None:
        self.reg = float(regularization)
        self.ls_grid = tuple(lengthscale_grid)
        self.use_gradients = bool(use_gradients)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        Z: np.ndarray,
        Y: np.ndarray,
        G: Optional[np.ndarray] = None,
        grad_idx: Optional[np.ndarray] = None,
    ) -> "GradientEnhancedGP":
        """Fit on inputs ``Z (n, r)``, outputs ``Y (n, m)``, gradients ``G``.

        ``G (ng, r, m)`` holds d(output)/dz for the training rows listed in
        ``grad_idx (ng,)`` (default: all rows). With ``use_gradients=False``
        (or ``G is None``) a plain kriging model is fitted.
        """
        Z = np.atleast_2d(np.asarray(Z, float))
        Y = np.asarray(Y, float)
        if Y.ndim == 1:
            Y = Y[:, None]
        n, r = Z.shape
        m = Y.shape[1]

        # ---- standardize inputs and outputs ----
        self._zm = Z.mean(axis=0)
        self._zs = np.maximum(Z.std(axis=0), 1e-12)
        Zt = (Z - self._zm) / self._zs
        self._ym = Y.mean(axis=0)
        self._ys = np.maximum(Y.std(axis=0), 1e-12)
        Yt = (Y - self._ym) / self._ys

        with_grads = self.use_gradients and G is not None and len(G) > 0
        if with_grads:
            G = np.asarray(G, float)
            if G.ndim == 2:                      # (n, r) single output
                G = G[:, :, None]
            gi = (np.arange(n) if grad_idx is None
                  else np.asarray(grad_idx, int))
            # standardized gradients: d(yt)/d(zt) = g * zs / ys
            Gt = G * self._zs[None, :, None] / self._ys[None, None, :]
        else:
            gi = np.empty(0, int)
            Gt = np.empty((0, r, m))

        self._Zt, self._gi, self._r, self._m = Zt, gi, r, m
        Pg = Zt[gi]                              # (ng, r) gradient-carrying inputs
        ng = len(gi)

        # observation vector per output: [values ; gradients (point-major)]
        obs = np.concatenate(
            [Yt, Gt.reshape(ng * r, m)], axis=0
        )                                        # (n + ng*r, m)

        # ---- lengthscale selection by marginal likelihood on output 0 ----
        D2 = _sqdist(Zt, Zt)
        off = D2[~np.eye(n, dtype=bool)]
        med = np.sqrt(np.median(off)) if off.size and np.median(off) > 0 else 1.0
        best = None
        for fac in self.ls_grid:
            ls = med * fac
            K = self._build_K(Zt, Pg, ls)
            factor = _chol_with_jitter(K, self.reg, ls)
            if factor is None:
                continue
            alpha0 = cho_solve(factor, obs[:, 0])
            lml = (-0.5 * float(obs[:, 0] @ alpha0)
                   - float(np.sum(np.log(np.diag(factor[0])))))
            if best is None or lml > best[0]:
                best = (lml, ls, factor)
        if best is None:                          # pathological data: fall back
            ls = med
            K = self._build_K(Zt, Pg, ls)
            K[np.diag_indices_from(K)] += 1e-4 * np.trace(K) / len(K)
            best = (0.0, ls, cho_factor(K, lower=True))

        _, self._ls, self._factor = best
        self._alpha = cho_solve(self._factor, obs)    # (N, m)
        return self

    def _build_K(self, P: np.ndarray, Pg: np.ndarray, ls: float) -> np.ndarray:
        """Joint covariance of [f(P) ; grad f(Pg)] under the Gaussian kernel."""
        n, r = P.shape
        ng = len(Pg)
        l2 = ls * ls
        K_ff = np.exp(-_sqdist(P, P) / (2 * l2))
        if ng == 0:
            return K_ff
        # cov(f(u), df(v)/dv_k) = k(u,v) (u - v)_k / l2
        Kv = np.exp(-_sqdist(P, Pg) / (2 * l2))          # (n, ng)
        Dfg = P[:, None, :] - Pg[None, :, :]             # (n, ng, r)
        K_fg = (Kv[:, :, None] * Dfg / l2).reshape(n, ng * r)
        # cov(df(u)/du_l, df(v)/dv_k) = k [ delta_lk/l2 - (u-v)_l (u-v)_k / l2^2 ]
        Kg = np.exp(-_sqdist(Pg, Pg) / (2 * l2))         # (ng, ng)
        Dgg = Pg[:, None, :] - Pg[None, :, :]            # (ng, ng, r)
        K_gg = (Kg[:, :, None, None]
                * (np.eye(r)[None, None] / l2
                   - Dgg[:, :, :, None] * Dgg[:, :, None, :] / (l2 * l2)))
        K_gg = K_gg.transpose(0, 2, 1, 3).reshape(ng * r, ng * r)
        return np.block([[K_ff, K_fg], [K_fg.T, K_gg]])

    def _cross(self, U: np.ndarray) -> np.ndarray:
        """Cross-covariance k_* between query points U (q, r) and the training set."""
        Zt, Pg, ls = self._Zt, self._Zt[self._gi], self._ls
        q = len(U)
        l2 = ls * ls
        k_f = np.exp(-_sqdist(U, Zt) / (2 * l2))                    # (q, n)
        if len(Pg) == 0:
            return k_f
        kg = np.exp(-_sqdist(U, Pg) / (2 * l2))                     # (q, ng)
        Dq = U[:, None, :] - Pg[None, :, :]                         # (q, ng, r)
        k_g = (kg[:, :, None] * Dq / l2).reshape(q, -1)
        return np.concatenate([k_f, k_g], axis=1)

    # -------------------------------------------------------------- predict
    def predict(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw-unit means ``(q, m)`` and stds ``(q, m)`` at inputs ``Z (q, r)``."""
        mt, st = self.predict_std(Z)
        return self._ym + self._ys * mt, self._ys[None, :] * st[:, None]

    def predict_std(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Standardized means ``(q, m)`` and shared standardized std ``(q,)``."""
        U = (np.atleast_2d(np.asarray(Z, float)) - self._zm) / self._zs
        k = self._cross(U)                                          # (q, N)
        mean = k @ self._alpha                                      # (q, m)
        v = cho_solve(self._factor, k.T)                            # (N, q)
        var = np.maximum(1.0 - np.einsum("qn,nq->q", k, v), 0.0)
        return mean, np.sqrt(var)

    def predict_std_single(
        self, z: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Standardized prediction with analytic gradients at one point ``z (r,)``.

        Returns ``(mean (m,), sigma, dmean/dz (r, m), dsigma/dz (r,))`` where the
        derivatives are with respect to the *unstandardized* latent coordinates.
        """
        u = ((np.asarray(z, float) - self._zm) / self._zs)[None, :]
        Zt, gi, ls, r = self._Zt, self._gi, self._ls, self._r
        Pg = Zt[gi]
        n, ng = len(Zt), len(gi)
        l2 = ls * ls

        k = self._cross(u)[0]                                       # (N,)
        mean = k @ self._alpha                                      # (m,)
        v = cho_solve(self._factor, k)                              # (N,)
        var = max(1.0 - float(k @ v), 0.0)
        sigma = np.sqrt(var)

        # ---- dk/du : (r, N) ----
        diff_f = u[0][None, :] - Zt                                 # (n, r)
        k_f = k[:n]
        dk_f = -(k_f[:, None] * diff_f / l2).T                      # (r, n)
        if ng:
            diff_g = u[0][None, :] - Pg                             # (ng, r)
            kg = np.exp(-np.sum(diff_g ** 2, axis=1) / (2 * l2))    # (ng,)
            # d/du_l [ kg * (u - v)_k / l2 ]
            dk_g = (kg[:, None, None] * (np.eye(r)[None] / l2
                    - diff_g[:, :, None] * diff_g[:, None, :] / (l2 * l2)))
            dk_g = dk_g.transpose(1, 0, 2).reshape(r, ng * r)       # (r, ng*r)
            dk = np.concatenate([dk_f, dk_g], axis=1)               # (r, N)
        else:
            dk = dk_f

        dmean_du = dk @ self._alpha                                 # (r, m)
        dvar_du = -2.0 * (dk @ v)                                   # (r,)
        dsig_du = dvar_du / (2.0 * sigma) if sigma > 1e-12 else np.zeros(r)

        inv_zs = 1.0 / self._zs
        return mean, sigma, dmean_du * inv_zs[:, None], dsig_du * inv_zs


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, clipped at 0 for stability."""
    d2 = (np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :]
          - 2.0 * A @ B.T)
    return np.maximum(d2, 0.0)


def _chol_with_jitter(K: np.ndarray, reg: float, ls: float):
    """Cholesky of K with block-scaled nugget; escalate jitter on failure."""
    N = len(K)
    base = np.maximum(np.diag(K), 1.0)   # value diag = 1, gradient diag = 1/ls^2
    jitter = reg
    for _ in range(4):
        try:
            return cho_factor(K + np.diag(jitter * base), lower=True)
        except np.linalg.LinAlgError:
            jitter *= 100.0
    return None


# --------------------------------------------------------------------------- #
# Active subspace
# --------------------------------------------------------------------------- #
def active_subspace(gradients: np.ndarray, max_dim: int, rng: np.random.Generator
                    ) -> np.ndarray:
    """Orthonormal basis ``W (d, r)`` of the dominant gradient subspace.

    Classic active-subspace construction (Constantine): the top right-singular
    vectors of the stacked gradient matrix, i.e. the leading eigenvectors of
    the empirical second-moment matrix ``E[g g^T]``. Degenerate (all-zero)
    gradients fall back to a random orthonormal basis.
    """
    G = np.atleast_2d(np.asarray(gradients, float))
    d = G.shape[1]
    if d <= max_dim:
        return np.eye(d)
    norms = np.linalg.norm(G, axis=1)
    G = G[norms > 1e-14]
    if len(G) == 0:
        Q, _ = np.linalg.qr(rng.standard_normal((d, max_dim)))
        return Q
    _, s, Vt = np.linalg.svd(G, full_matrices=False)
    rank = int(np.sum(s > 1e-12 * s[0]))
    r = max(1, min(max_dim, rank))
    W = Vt[:r].T                                   # (d, r)
    if r < max_dim:
        # pad with random orthogonal complement directions to keep some
        # off-subspace expressiveness early on (few gradients collected yet)
        extra = rng.standard_normal((d, max_dim - r))
        extra -= W @ (W.T @ extra)
        Q, _ = np.linalg.qr(extra)
        W = np.concatenate([W, Q[:, : max_dim - r]], axis=1)
    return W


# --------------------------------------------------------------------------- #
# Batch acquisition
# --------------------------------------------------------------------------- #
def _acquisition_value_grad(
    xi: np.ndarray,
    gek: GradientEnhancedGP,
    W: np.ndarray,
    kappa: float,
    batch: List[np.ndarray],
    cfg: GESBOConfig,
    cons_shift: np.ndarray,
    rep_h2: float,
) -> Tuple[float, np.ndarray]:
    """Penalized LCB acquisition and its analytic gradient w.r.t. full-space xi.

    ``A(xi) = m0(z) - kappa * sigma(z) + rho * sum max(0, m_c(z) + shift_c)^2
              + w * sum_b exp(-||xi - xi_b||^2 / (2 h^2))``

    with ``z = W^T xi``; means are in standardized units so the terms are
    commensurate. ``cons_shift[c] = ym_c / ys_c`` recenters the standardized
    constraint mean so that the hinge activates exactly at raw ``c > 0``;
    ``rep_h2`` is the squared repulsion bandwidth (set from the trust-region
    box size by :func:`propose_batch`).
    """
    z = W.T @ xi
    mean, sigma, dmean, dsig = gek.predict_std_single(z)
    a = float(mean[0]) - kappa * sigma
    da_dz = dmean[:, 0] - kappa * dsig
    # smooth hinge on surrogate constraints (standardized units)
    for c in range(1, gek._m):
        t = float(mean[c]) + cons_shift[c - 1]
        if t > 0.0:
            a += cfg.constraint_penalty_acq * t * t
            da_dz = da_dz + 2.0 * cfg.constraint_penalty_acq * t * dmean[:, c]
    grad = W @ da_dz
    # repulsion from already-selected batch members (diversity)
    if batch:
        for xb in batch:
            dxb = xi - xb
            e = np.exp(-float(dxb @ dxb) / (2 * rep_h2))
            a += cfg.repulsion_weight * e
            grad = grad - cfg.repulsion_weight * e * dxb / rep_h2
    return a, grad


def propose_batch(
    gek: GradientEnhancedGP,
    W: np.ndarray,
    center: np.ndarray,
    best: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    cfg: GESBOConfig,
    rng: np.random.Generator,
    cons_shift: np.ndarray,
) -> List[np.ndarray]:
    """Select ``batch_size`` candidate points inside the trust-region box.

    Point 1 uses ``kappa = 0`` (pure penalized-exploitation); the following
    points climb a geometric ``kappa`` ladder with the repulsion term keeping
    them apart. Near-duplicate candidates are replaced by random points, which
    also injects off-subspace exploration in the reduced-dimension regime.
    """
    q = cfg.batch_size
    d = len(center)
    kappas = [0.0] + [cfg.kappa_base * cfg.kappa_growth ** j for j in range(q - 1)]
    # repulsion bandwidth ~ a quarter of the trust-region box diagonal
    rep_h2 = max(0.25 * float(np.linalg.norm(hi - lo)), 1e-9) ** 2
    batch: List[np.ndarray] = []
    for j in range(q):
        starts = [np.clip(center, lo, hi), np.clip(best, lo, hi)]
        starts += [lo + (hi - lo) * rng.random(d) for _ in range(cfg.n_restarts)]
        cand, cand_val = None, np.inf
        for x0 in starts:
            res = minimize(
                _acquisition_value_grad,
                x0,
                args=(gek, W, kappas[j], batch, cfg, cons_shift, rep_h2),
                jac=True,
                method="L-BFGS-B",
                bounds=list(zip(lo, hi)),
                options={"maxiter": cfg.acq_maxiter},
            )
            if res.fun < cand_val:
                cand, cand_val = np.clip(res.x, lo, hi), float(res.fun)
        # de-duplicate: fall back to a random trust-region point
        if cand is None or any(np.linalg.norm(cand - b) < 1e-8 for b in batch):
            cand = lo + (hi - lo) * rng.random(d)
        batch.append(cand)
    return batch


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
class GESBOptimizer:
    """Trust-region-managed gradient-enhanced surrogate-based optimizer.

    Parameters
    ----------
    evaluate
        Callable ``x -> (f, grad_f, cons, jac_cons)`` in raw coordinates;
        inequality constraints follow the ``c(x) <= 0`` convention. For
        unconstrained problems return ``cons`` of shape ``(0,)`` and
        ``jac_cons`` of shape ``(0, d)``.
    lower_bounds, upper_bounds
        Box bounds of the design space (raw units).
    config
        :class:`GESBOConfig`; ``None`` uses the defaults.
    """

    def __init__(
        self,
        evaluate: EvaluateFn,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        config: Optional[GESBOConfig] = None,
        on_iteration: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.evaluate = evaluate
        self.lb = np.asarray(lower_bounds, float)
        self.ub = np.asarray(upper_bounds, float)
        self.scale = np.maximum(self.ub - self.lb, 1e-30)
        self.cfg = config or GESBOConfig()
        self.on_iteration = on_iteration
        self._rng = np.random.default_rng(self.cfg.seed)
        # sample store (normalized coordinates)
        self.X: List[np.ndarray] = []       # xi in [0,1]^d
        self.F: List[float] = []
        self.Gf: List[np.ndarray] = []      # df/dxi
        self.C: List[np.ndarray] = []       # (m,)
        self.Jc: List[np.ndarray] = []      # (m, d), d(cons)/dxi
        self.n_evals = 0

    # ------------------------------------------------------------- sampling
    def _eval(self, xi: np.ndarray) -> int:
        """Evaluate the true model at normalized ``xi``; return the sample index."""
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
        """Best sample: min objective among feasible, else min violation."""
        feas = [i for i in range(len(self.X))
                if self._violation(i) <= self.cfg.constraint_tol]
        if feas:
            return min(feas, key=lambda i: self.F[i])
        return min(range(len(self.X)), key=self._violation)

    # ------------------------------------------------------------ surrogate
    def _fit_surrogate(self, center: np.ndarray):
        """Fit the multi-output GEK on a window of samples nearest the center."""
        cfg = self.cfg
        Xa = np.asarray(self.X)
        n = len(Xa)
        order = np.argsort(np.linalg.norm(Xa - center[None, :], axis=1))
        keep = order[: min(cfg.max_points, n)]
        # always keep the incumbent best in the training set
        ib = self._best_index()
        if ib not in keep:
            keep = np.concatenate([keep[:-1], [ib]])
        keep = np.array(sorted(set(int(i) for i in keep)))

        m = len(self.C[0])
        Y = np.column_stack(
            [np.asarray(self.F)[keep]]
            + [np.array([self.C[i][c] for i in keep]) for c in range(m)]
        ) if m else np.asarray(self.F)[keep][:, None]

        # gradient-carrying subset: nearest max_grad_points of the window
        gsel = keep[np.argsort(np.linalg.norm(Xa[keep] - center[None, :], axis=1))]
        gsel = gsel[: min(cfg.max_grad_points, len(gsel))]

        W = (active_subspace(np.asarray([self.Gf[i] for i in gsel]),
                             cfg.max_latent_dim, self._rng)
             if cfg.use_gradients else
             active_subspace(np.zeros((1, Xa.shape[1])), cfg.max_latent_dim,
                             self._rng))
        Z = Xa[keep] @ W

        grad_idx = np.searchsorted(keep, np.sort(gsel))
        G = np.stack([
            np.column_stack([W.T @ self.Gf[i]]
                            + [W.T @ self.Jc[i][c] for c in range(m)])
            for i in np.sort(gsel)
        ]) if cfg.use_gradients else None                    # (ng, r, m+1)

        gek = GradientEnhancedGP(
            regularization=cfg.regularization,
            lengthscale_grid=cfg.lengthscale_grid,
            use_gradients=cfg.use_gradients,
        ).fit(Z, Y, G, grad_idx)
        # raw-zero crossing of each standardized constraint mean
        cons_shift = (gek._ym[1:] / gek._ys[1:]) if m else np.empty(0)
        return gek, W, cons_shift

    # ----------------------------------------------------------------- run
    def run(self, x0: np.ndarray) -> GESBOResult:
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

        for it in range(cfg.max_outer_iter):
            if self.n_evals >= cfg.max_evals:
                status = "evaluation budget exhausted"
                break
            if delta < cfg.tr_min:
                status = "trust region collapsed"
                break

            gek, W, cons_shift = self._fit_surrogate(center)
            lo = np.maximum(0.0, center - delta)
            hi = np.minimum(1.0, center + delta)
            best_i = self._best_index()
            batch = propose_batch(
                gek, W, center, self.X[best_i], lo, hi, cfg, self._rng, cons_shift
            )

            # ---- evaluate the batch (independent points: parallelizable) ----
            new_idx = []
            for xi in batch:
                if self.n_evals >= cfg.max_evals:
                    break
                new_idx.append(self._eval(xi))
            if not new_idx:
                status = "evaluation budget exhausted"
                break

            # ---- trust-region ratio test on the l1 merit ----
            cand_i = min(new_idx, key=self._merit)
            actual = self._merit(center_i) - self._merit(cand_i)
            pred = self._predicted_merit_drop(gek, W, cons_shift,
                                              self.X[center_i], self.X[cand_i])
            rho = actual / max(pred, 1e-14) if pred > 0 else (
                np.inf if actual > 0 else -np.inf)

            step = float(np.max(np.abs(self.X[cand_i] - center)))
            if actual > 0 and rho >= cfg.eta_accept:
                center, center_i = self.X[cand_i].copy(), cand_i
                if rho >= cfg.eta_expand and step >= 0.8 * delta:
                    delta = min(delta * cfg.tr_expand, cfg.tr_max)
            else:
                delta *= cfg.tr_shrink

            stall = stall + 1 if actual <= cfg.ftol_abs else 0
            rec = {
                "iter": it, "n_evals": self.n_evals, "delta": delta,
                "batch_indices": new_idx, "rho": float(rho),
                "merit_center": self._merit(center_i),
                "best_f": self.F[self._best_index()],
            }
            history.append(rec)
            if self.on_iteration is not None:
                self.on_iteration(rec)
            LOGGER.info(
                "GE-SBO iter %d: evals=%d f_best=%.6e merit=%.6e delta=%.3e rho=%.2f",
                it, self.n_evals, rec["best_f"], rec["merit_center"], delta, rho,
            )
            if stall >= cfg.stall_limit and delta <= cfg.tr_min * 10:
                status = "stalled"
                break

        ib = self._best_index()
        return GESBOResult(
            x_opt=self.lb + self.scale * self.X[ib],
            f_opt=self.F[ib],
            constraints=self.C[ib].copy(),
            is_feasible=self._violation(ib) <= cfg.constraint_tol,
            n_evals=self.n_evals,
            n_iter=len(history),
            status=status,
            history=history,
        )

    def _predicted_merit_drop(self, gek, W, cons_shift, xi_from, xi_to) -> float:
        """Surrogate-predicted merit reduction between two normalized points."""
        mean, _ = gek.predict(np.asarray([xi_from @ W, xi_to @ W]))
        def merit(row):
            viol = float(np.sum(np.maximum(0.0, row[1:]))) if len(row) > 1 else 0.0
            return float(row[0]) + self.cfg.penalty * viol
        return merit(mean[0]) - merit(mean[1])


def gesbo_minimize(
    evaluate: EvaluateFn,
    x0: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    config: Optional[GESBOConfig] = None,
    on_iteration: Optional[Callable[[dict], None]] = None,
) -> GESBOResult:
    """Functional entry point: minimize ``f(x)`` s.t. ``c(x) <= 0``, box bounds."""
    return GESBOptimizer(
        evaluate, lower_bounds, upper_bounds, config, on_iteration
    ).run(x0)
