# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Reduced-space sequential optimization: subspace frame, driver, GEK proposer.

At every iterate a low-dimensional orthonormal frame is built from the
information any SCP method has anyway (SESOP-style subspace optimization):

* ``e1 = -grad f / ||grad f||``          steepest descent;
* ``e2 =  orth(grad c_agg)``             aggregated-constraint restoration;
* ``e3 =  orth(previous step)``          momentum — with e1 this span contains
  the conjugate-gradient step, curing steepest-descent zig-zag;
* ``e4 =  orth(g_k - g_{k-1})``          secant — the span then also contains
  the memory-1 quasi-Newton direction.

Degenerate directions are replaced by random orthonormal fill so the frame
always has ``subspace_dim`` columns. The next iterate is searched inside a
trust region on that subspace, ``x_next = x + delta * E @ alpha`` with
``alpha in [-1, 1]^r``, by a *proposer*:

* :class:`GEKProposer` — a few true evaluations on the subspace, one
  gradient-enhanced kriging model of objective + aggregated constraint over
  ``alpha`` (exact projected directional derivatives), candidate set solved
  feasibility-first — the classical reduced-space GEK sub-optimization;
* the transformer proposer of :mod:`scp_uno.rs_transformer` — predicts
  ``alpha`` (and a trust-radius multiplier) from the iteration history at
  zero inner-evaluation cost.

The driver provides l1-merit acceptance with an adaptive penalty,
radius adaptation (proposer-supplied multiplier when available, otherwise
boundary-expand / interior-shrink), restarts from the incumbent best, and
feasibility-first result bookkeeping. These are *local-descent* methods for
multimodal problems — no optimality claim.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.stats import qmc

from scp_uno.gesbo_core import EvaluateFn, GradientEnhancedGP

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constraint aggregation and subspace construction
# --------------------------------------------------------------------------- #
def ks_aggregate(c: np.ndarray, J: np.ndarray, rho: float = 50.0
                 ) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """KS (log-sum-exp) aggregation of ``c(x) <= 0`` constraints and gradient."""
    if len(c) == 0:
        return None, None
    if len(c) == 1:
        return float(c[0]), J[0].copy()
    m = float(np.max(c))
    w = np.exp(rho * (c - m))
    s = float(np.sum(w))
    return m + np.log(s) / rho, (w / s) @ J


def build_subspace(
    grad_f: np.ndarray,
    directions: List[Optional[np.ndarray]],
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Orthonormal frame ``E (d, dim)``: steepest descent first, then the given
    directions Gram-Schmidt-orthogonalized in order, random fill for the rest."""
    d = len(grad_f)
    n_real = min(dim, d)        # at most d orthonormal directions exist in R^d
    n1 = float(np.linalg.norm(grad_f))
    cols = [(-grad_f / n1) if n1 > 1e-300 else _random_unit(d, rng)]

    def try_add(v) -> None:
        if v is None or len(cols) >= n_real:
            return
        w = np.asarray(v, float).copy()
        nv = float(np.linalg.norm(w))
        if nv <= 1e-300:
            return
        for e in cols:
            w -= (w @ e) * e
        n2 = float(np.linalg.norm(w))
        if n2 > 1e-10 * nv:
            cols.append(w / n2)

    for v in directions:
        try_add(v)
    while len(cols) < n_real:
        try_add(_random_unit(d, rng))
    E = np.column_stack(cols)
    if E.shape[1] < dim:        # d < dim: pad with inert zero columns
        E = np.concatenate([E, np.zeros((d, dim - E.shape[1]))], axis=1)
    return E


# backward-compatible 2D helper (kept for tests / external use)
def build_frame(grad_f, grad_c, prev_dir, rng):
    E = build_subspace(np.asarray(grad_f, float), [grad_c, prev_dir], 2, rng)
    return E[:, 0], E[:, 1]


def _random_unit(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- #
# Configuration / result / shared state
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class ReducedSpaceConfig:
    max_evals: int = 200
    subspace_dim: int = 4          # r: steepest descent, constraint, momentum, secant
    delta_init: float = 0.1        # trust radius in the normalized design box
    delta_min: float = 1e-6
    delta_max: float = 0.5
    shrink: float = 0.5            # radius factor on rejected steps
    expand: float = 1.6            # radius factor on accepted boundary steps
    interior_shrink: float = 0.7   # radius factor on accepted small interior steps
    boundary_frac: float = 0.9
    interior_frac: float = 0.3
    gamma_max: float = 2.5         # range of a proposer-supplied radius multiplier
    penalty: float = 10.0          # merit safety factor over multiplier estimate
    constraint_tol: float = 1e-6
    ks_rho: float = 50.0
    stall_limit: int = 10
    n_backtracks: int = 2          # step halvings tried when a proposal is rejected
    n_resets: int = 6              # trust-region restarts from the incumbent best
    seed: int = 0
    # GEK proposer
    n_inner: int = 5               # true evaluations spent on the subspace per iter
    n_candidates: int = 4096       # surrogate candidate points for the subspace solve


@dataclasses.dataclass
class ReducedSpaceResult:
    x_opt: np.ndarray
    f_opt: float
    constraints: np.ndarray
    is_feasible: bool
    n_evals: int
    n_iter: int
    status: str
    history: List[dict]


class Sample:
    """One true evaluation: raw arrays plus the aggregated constraint."""

    __slots__ = ("x", "f", "g", "c", "J", "c_agg", "g_agg")

    def __init__(self, x, f, g, c, J, ks_rho):
        self.x, self.f, self.g, self.c, self.J = x, f, g, c, J
        self.c_agg, self.g_agg = ks_aggregate(c, J, ks_rho)


class IterationContext:
    """Everything a proposer may use for the current iteration."""

    def __init__(self, driver: "ReducedSpaceDriver", center: Sample,
                 E: np.ndarray, delta: float):
        self.driver = driver
        self.center = center
        self.E = E                                  # (d, r) orthonormal
        self.delta = delta
        self.constrained = center.c_agg is not None

    @property
    def dim(self) -> int:
        return self.E.shape[1]

    def x_of(self, alpha: np.ndarray) -> np.ndarray:
        x = self.center.x + self.delta * (self.E @ np.asarray(alpha, float))
        return np.clip(x, self.driver.lb, self.driver.ub)

    def eval_subspace(self, alpha: np.ndarray) -> Optional[Sample]:
        """Evaluate the true model on the subspace (None if budget exhausted)."""
        if self.driver.n_evals >= self.driver.cfg.max_evals:
            return None
        return self.driver._eval(self.x_of(alpha))

    def project(self, s: Sample) -> Tuple[np.ndarray, float]:
        """In-subspace coordinates (r,) and out-of-subspace distance of s."""
        p = s.x - self.center.x
        a = (self.E.T @ p) / self.delta
        r = float(np.linalg.norm(p - self.delta * (self.E @ a)))
        return a, r / self.delta

    def gradk(self, s: Sample, which: str) -> np.ndarray:
        g = s.g if which == "f" else s.g_agg
        return self.E.T @ g


# --------------------------------------------------------------------------- #
# Proposers
# --------------------------------------------------------------------------- #
class GEKProposer:
    """Solve the subspace subproblem with a gradient-enhanced kriging model.

    Spends ``n_inner`` true evaluations at a fixed descent-biased pattern in
    the trust cube, fits one GEK over ``alpha`` for the objective and (if
    present) the aggregated constraint — with exact projected directional
    derivatives as gradient observations — and returns the feasibility-first
    best of a Sobol candidate set on the surrogate.
    """

    def propose(self, ctx: IterationContext
                ) -> Tuple[np.ndarray, Optional[float]]:
        cfg = ctx.driver.cfg
        r = ctx.dim
        pattern = _inner_pattern(r)
        samples = [ctx.center]
        for a in pattern[: max(1, cfg.n_inner)]:
            s = ctx.eval_subspace(a)
            if s is None:
                break
            samples.append(s)
        if len(samples) < 2:                    # budget gone: plain descent step
            return np.eye(r)[0], None

        Z = np.array([ctx.project(s)[0] for s in samples])
        m = 1 + (1 if ctx.constrained else 0)
        Y = np.empty((len(samples), m))
        G = np.empty((len(samples), r, m))
        for i, s in enumerate(samples):
            Y[i, 0] = s.f
            G[i, :, 0] = ctx.gradk(s, "f") * ctx.delta
            if ctx.constrained:
                Y[i, 1] = s.c_agg
                G[i, :, 1] = ctx.gradk(s, "c") * ctx.delta
        gek = GradientEnhancedGP().fit(Z, Y, G)

        P = _candidate_set(r, cfg.n_candidates, ctx.driver.rng)
        mean, _ = gek.predict(P)
        alpha = _feasibility_first_pick(P, mean[:, 0],
                                        mean[:, 1] if ctx.constrained else None)
        return alpha, None


def _inner_pattern(r: int) -> List[np.ndarray]:
    """Descent-biased evaluation pattern in the trust cube [-1, 1]^r."""
    pts = [np.zeros(r) for _ in range(2 * r)]
    pts[0] = np.zeros(r)
    pts[0][0] = 0.8                               # along steepest descent
    k = 1
    for j in range(1, r):
        p = np.zeros(r)
        p[0], p[j] = 0.3, 0.6
        pts[k] = p
        k += 1
    for j in range(1, r):
        p = np.zeros(r)
        p[0], p[j] = 0.3, -0.6
        pts[k] = p
        k += 1
    p = np.zeros(r)
    p[0] = -0.4                                   # uphill probe (curvature)
    pts[k] = p
    return pts[: k + 1]


def _candidate_set(r: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sobol candidates in [-1, 1]^r plus the axes and the origin."""
    sob = qmc.Sobol(d=r, scramble=True, seed=int(rng.integers(1 << 31)))
    P = 2.0 * sob.random(n) - 1.0
    axes = np.concatenate([np.eye(r), -np.eye(r), 0.5 * np.eye(r),
                           np.zeros((1, r))])
    return np.concatenate([P, axes])


def _feasibility_first_pick(P, f_hat, c_hat) -> np.ndarray:
    """Best candidate: min f among predicted-feasible, else min violation."""
    if c_hat is not None:
        feas = c_hat <= 0.0
        idx = (np.flatnonzero(feas)[np.argmin(f_hat[feas])] if np.any(feas)
               else int(np.argmin(c_hat)))
    else:
        idx = int(np.argmin(f_hat))
    return np.asarray(P[idx], float)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
class ReducedSpaceDriver:
    """Trust-region outer loop shared by the GEK and transformer proposers.

    A proposer returns ``(alpha, gamma)``: the step coordinates and an
    optional trust-radius multiplier (``None`` -> the driver's own
    boundary-expand / interior-shrink rule is applied on acceptance).
    """

    def __init__(
        self,
        evaluate: EvaluateFn,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        proposer,
        config: Optional[ReducedSpaceConfig] = None,
        on_iteration: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.evaluate = evaluate
        self.lb = np.asarray(lower_bounds, float)
        self.ub = np.asarray(upper_bounds, float)
        self.proposer = proposer
        self.cfg = config or ReducedSpaceConfig()
        self.on_iteration = on_iteration
        self.rng = np.random.default_rng(self.cfg.seed)
        self.samples: List[Sample] = []
        self.n_evals = 0
        self._mu = 0.0

    # ------------------------------------------------------------- sampling
    def _eval(self, x: np.ndarray) -> Sample:
        f, g, c, J = self.evaluate(x)
        s = Sample(
            np.asarray(x, float).copy(), float(f),
            np.asarray(g, float).flatten(),
            np.asarray(c, float).flatten(),
            np.asarray(J, float).reshape(-1, len(x)),
            self.cfg.ks_rho,
        )
        self.samples.append(s)
        self.n_evals += 1
        return s

    def _merit(self, s: Sample) -> float:
        viol = float(np.sum(np.maximum(0.0, s.c))) if len(s.c) else 0.0
        return s.f + self._mu * viol

    def _update_mu(self, s: Sample) -> None:
        if len(s.J):
            s0 = float(np.mean(np.abs(s.g)))
            s1 = float(np.mean(np.abs(s.J)))
            self._mu = max(self._mu, self.cfg.penalty * s0 / max(s1, 1e-300))

    def _violation(self, s: Sample) -> float:
        return float(np.max(s.c)) if len(s.c) else -np.inf

    def _best(self) -> Sample:
        feas = [s for s in self.samples
                if self._violation(s) <= self.cfg.constraint_tol]
        if feas:
            return min(feas, key=lambda s: s.f)
        return min(self.samples, key=self._violation)

    # ----------------------------------------------------------------- run
    def run(self, x0: np.ndarray) -> ReducedSpaceResult:
        cfg = self.cfg
        delta = cfg.delta_init
        history: List[dict] = []
        status = "evaluation budget exhausted"
        center = self._eval(np.clip(np.asarray(x0, float), self.lb, self.ub))
        steps_hist: List[np.ndarray] = []      # most recent first
        secants_hist: List[np.ndarray] = []
        stall = 0
        it = 0
        resets = cfg.n_resets

        def _restart(reason: str) -> bool:
            nonlocal delta, stall, center, resets
            if resets <= 0:
                return False
            resets -= 1
            # progressively smaller fresh radius: a deterministic proposer would
            # otherwise reproduce the same rejected step and re-stall immediately
            delta = cfg.delta_init * 0.5 ** (cfg.n_resets - resets - 1)
            stall = 0
            steps_hist.clear()
            secants_hist.clear()
            center = self._best()
            LOGGER.info("reduced-space restart (%s): %d left, delta=%.3g",
                        reason, resets, delta)
            return True

        while self.n_evals < cfg.max_evals:
            if delta < cfg.delta_min and not _restart("radius collapsed"):
                status = "trust region collapsed"
                break
            self._update_mu(center)
            # direction pool: constraint, then interleaved momentum/secant history
            pool: List[Optional[np.ndarray]] = [center.g_agg]
            for s_v, y_v in zip(steps_hist, secants_hist):
                pool.extend([s_v, y_v])
            E = build_subspace(center.g, pool, cfg.subspace_dim, self.rng)
            ctx = IterationContext(self, center, E, delta)

            alpha, gamma = self.proposer.propose(ctx)
            alpha = np.clip(np.asarray(alpha, float), -1.0, 1.0)
            cand = ctx.eval_subspace(alpha)
            if cand is None:
                break

            it += 1
            # backtracking: a rejected proposal often has the right direction
            # but too large a magnitude — halve it before counting a stall
            tol = 1e-12 * (1.0 + abs(self._merit(center)))
            for k in range(1, cfg.n_backtracks + 1):
                if self._merit(cand) <= self._merit(center) + tol:
                    break
                shorter = ctx.eval_subspace(alpha * 0.5 ** k)
                if shorter is None:
                    break
                if self._merit(shorter) < self._merit(cand):
                    cand = shorter
                    alpha = alpha * 0.5 ** k
                    gamma = None              # radius follows the interior rule

            if self._merit(cand) <= self._merit(center) + 1e-12 * (
                    1.0 + abs(self._merit(center))):
                step = cand.x - center.x
                if np.linalg.norm(step) > 1e-300:
                    steps_hist.insert(0, step)
                    secants_hist.insert(0, cand.g - center.g)
                    del steps_hist[3:], secants_hist[3:]
                center = cand
                if gamma is not None:
                    delta = float(np.clip(gamma * delta, cfg.delta_min,
                                          cfg.delta_max))
                else:
                    a_inf = float(np.max(np.abs(alpha)))
                    if a_inf >= cfg.boundary_frac:
                        delta = min(delta * cfg.expand, cfg.delta_max)
                    elif a_inf <= cfg.interior_frac:
                        delta = max(delta * cfg.interior_shrink, cfg.delta_min)
                stall = 0
            else:
                delta *= cfg.shrink
                stall += 1
                if stall >= cfg.stall_limit and not _restart("stalled"):
                    status = "stalled"
                    break

            rec = {"iter": it, "n_evals": self.n_evals, "delta": delta,
                   "alpha": alpha.copy(),
                   "merit_center": self._merit(center),
                   "best_f": self._best().f}
            history.append(rec)
            if self.on_iteration is not None:
                self.on_iteration(rec)

        best = self._best()
        return ReducedSpaceResult(
            x_opt=best.x.copy(), f_opt=best.f, constraints=best.c.copy(),
            is_feasible=self._violation(best) <= cfg.constraint_tol,
            n_evals=self.n_evals, n_iter=it, status=status, history=history,
        )


def gek2d_minimize(
    evaluate: EvaluateFn,
    x0: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    config: Optional[ReducedSpaceConfig] = None,
    on_iteration: Optional[Callable[[dict], None]] = None,
) -> ReducedSpaceResult:
    """Reduced-space SCP with the GEK sub-optimization (functional API).

    The name is kept from the original two-direction formulation; the default
    subspace now has four directions (descent, constraint, momentum, secant).
    """
    return ReducedSpaceDriver(
        evaluate, lower_bounds, upper_bounds, GEKProposer(), config,
        on_iteration,
    ).run(x0)
