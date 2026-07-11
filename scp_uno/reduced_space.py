# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Reduced-space (2D) sequential optimization: frame, driver and GEK proposer.

At every iterate the algorithm builds a **two-dimensional coordinate system**
from the information any SCP method has anyway:

* ``e1 = -grad f / ||grad f||``            (steepest-descent direction), and
* ``e2 =  orth(grad c_agg, e1)``           (the aggregated-constraint gradient
  orthogonalized w.r.t. the objective gradient; for unconstrained problems the
  previous accepted step, orthogonalized, acts as a momentum direction).

The next iterate is searched *inside a trust region on that plane*:
``x_next = x + delta * (alpha e1 + beta e2)`` with ``(alpha, beta)`` in
``[-1, 1]^2``. How ``(alpha, beta)`` is chosen is delegated to a *proposer*:

* :class:`GEKProposer` — spends a few true evaluations on the plane, fits a
  gradient-enhanced kriging model of the objective and aggregated constraint
  over ``(alpha, beta)`` (exact projected directional derivatives), and picks
  the best-predicted feasible point on the surrogate — the classical
  reduced-space GEK sub-optimization;
* the transformer proposer in :mod:`scp_uno.rs_transformer` — predicts
  ``(alpha, beta)`` directly from the iteration history at **zero** inner
  evaluation cost.

Because everything the proposers see lives in the 2D frame, the approach is
independent of the number of design variables by construction. The driver
provides l1-merit acceptance with an adaptive penalty, trust-region radius
adaptation and feasibility-first result bookkeeping. It aims at *efficiently
reaching a local minimum* of a multimodal problem — no global claims.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

from scp_uno.gesbo_core import EvaluateFn, GradientEnhancedGP

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constraint aggregation and frame construction
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


def build_frame(
    grad_f: np.ndarray,
    grad_c: Optional[np.ndarray],
    prev_dir: Optional[np.ndarray],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormal 2D frame ``(e1, e2)`` of the reduced space.

    ``e1`` is the steepest-descent direction; ``e2`` is the aggregated
    constraint gradient orthogonalized w.r.t. ``e1`` (falling back to the
    previous accepted step, then to a random direction, when degenerate).
    """
    d = len(grad_f)
    n1 = float(np.linalg.norm(grad_f))
    e1 = -grad_f / n1 if n1 > 1e-300 else _random_unit(d, rng)

    for v in (grad_c, prev_dir):
        if v is None:
            continue
        w = np.asarray(v, float) - (np.asarray(v, float) @ e1) * e1
        n2 = float(np.linalg.norm(w))
        if n2 > 1e-10 * max(np.linalg.norm(v), 1.0):
            return e1, w / n2
    while True:
        w = _random_unit(d, rng)
        w = w - (w @ e1) * e1
        n2 = float(np.linalg.norm(w))
        if n2 > 1e-8:
            return e1, w / n2


def _random_unit(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- #
# Configuration / result / shared state
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class ReducedSpaceConfig:
    max_evals: int = 200
    delta_init: float = 0.1        # trust radius in the normalized design box
    delta_min: float = 1e-6
    delta_max: float = 0.5
    shrink: float = 0.5
    expand: float = 1.6
    boundary_frac: float = 0.9     # step at TR boundary => try expanding
    penalty: float = 10.0          # merit safety factor over multiplier estimate
    constraint_tol: float = 1e-6
    ks_rho: float = 50.0
    stall_limit: int = 10
    n_resets: int = 3          # trust-region restarts from the incumbent best
    seed: int = 0
    # GEK proposer
    n_inner: int = 3               # true evaluations spent on the plane per iter
    grid: int = 41                 # surrogate grid resolution for the 2D solve


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
                 e1: np.ndarray, e2: np.ndarray, delta: float):
        self.driver = driver
        self.center = center
        self.e1, self.e2, self.delta = e1, e2, delta
        self.constrained = center.c_agg is not None

    def x_of(self, alpha: float, beta: float) -> np.ndarray:
        x = self.center.x + self.delta * (alpha * self.e1 + beta * self.e2)
        return np.clip(x, self.driver.lb, self.driver.ub)

    def eval_plane(self, alpha: float, beta: float) -> Optional[Sample]:
        """Evaluate the true model on the plane (None if budget exhausted)."""
        if self.driver.n_evals >= self.driver.cfg.max_evals:
            return None
        return self.driver._eval(self.x_of(alpha, beta))

    def project(self, s: Sample) -> Tuple[float, float, float]:
        """In-plane coordinates (alpha, beta) and out-of-plane distance of s."""
        p = s.x - self.center.x
        a = float(p @ self.e1) / self.delta
        b = float(p @ self.e2) / self.delta
        r = float(np.linalg.norm(p - self.delta * (a * self.e1 + b * self.e2)))
        return a, b, r / self.delta

    def grad2d(self, s: Sample, which: str) -> Tuple[float, float]:
        g = s.g if which == "f" else s.g_agg
        return float(g @ self.e1), float(g @ self.e2)


# --------------------------------------------------------------------------- #
# Proposers
# --------------------------------------------------------------------------- #
class GEKProposer:
    """Solve the 2D subproblem with a gradient-enhanced kriging surrogate.

    Spends ``n_inner`` true evaluations at a fixed descent-biased pattern in
    the trust square, fits one GEK over ``(alpha, beta)`` for the objective
    and (if present) the aggregated constraint — with the *exact* projected
    directional derivatives as gradient observations — then returns the
    feasibility-first best point of the surrogate on a grid.
    """

    PATTERN = ((0.8, 0.0), (0.4, 0.7), (0.4, -0.7), (-0.5, 0.0), (0.0, 0.8))

    def propose(self, ctx: IterationContext) -> Tuple[float, float]:
        cfg = ctx.driver.cfg
        samples = [ctx.center]
        for a, b in self.PATTERN[: max(1, cfg.n_inner)]:
            s = ctx.eval_plane(a, b)
            if s is None:
                break
            samples.append(s)
        if len(samples) < 2:
            return (1.0, 0.0)      # budget gone: plain descent step

        Z = np.array([ctx.project(s)[:2] for s in samples])
        m = 1 + (1 if ctx.constrained else 0)
        Y = np.empty((len(samples), m))
        G = np.empty((len(samples), 2, m))
        for i, s in enumerate(samples):
            Y[i, 0] = s.f
            G[i, :, 0] = ctx.grad2d(s, "f")
            G[i, :, 0] = np.asarray(G[i, :, 0]) * ctx.delta
            if ctx.constrained:
                Y[i, 1] = s.c_agg
                G[i, :, 1] = np.asarray(ctx.grad2d(s, "c")) * ctx.delta
        gek = GradientEnhancedGP().fit(Z, Y, G)

        n = cfg.grid
        ax = np.linspace(-1.0, 1.0, n)
        A, B = np.meshgrid(ax, ax, indexing="ij")
        P = np.column_stack([A.ravel(), B.ravel()])
        mean, _ = gek.predict(P)
        return _feasibility_first_pick(P, mean[:, 0],
                                       mean[:, 1] if ctx.constrained else None)


def _feasibility_first_pick(P, f_hat, c_hat) -> Tuple[float, float]:
    """Best candidate: min f among predicted-feasible, else min violation."""
    if c_hat is not None:
        feas = c_hat <= 0.0
        idx = (np.flatnonzero(feas)[np.argmin(f_hat[feas])] if np.any(feas)
               else int(np.argmin(c_hat)))
    else:
        idx = int(np.argmin(f_hat))
    return float(P[idx, 0]), float(P[idx, 1])


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
class ReducedSpaceDriver:
    """Trust-region outer loop shared by the GEK and transformer proposers."""

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
        prev_dir: Optional[np.ndarray] = None
        stall = 0
        it = 0
        resets = cfg.n_resets

        def _restart(reason: str) -> bool:
            """Restart from the incumbent best with a fresh radius."""
            nonlocal delta, stall, center, prev_dir, resets
            if resets <= 0:
                return False
            resets -= 1
            delta = cfg.delta_init
            stall = 0
            prev_dir = None
            center = self._best()
            LOGGER.info("reduced-space restart (%s): %d left", reason, resets)
            return True

        while self.n_evals < cfg.max_evals:
            if delta < cfg.delta_min and not _restart("radius collapsed"):
                status = "trust region collapsed"
                break
            self._update_mu(center)
            e1, e2 = build_frame(center.g, center.g_agg, prev_dir, self.rng)
            ctx = IterationContext(self, center, e1, e2, delta)

            alpha, beta = self.proposer.propose(ctx)
            cand = ctx.eval_plane(alpha, beta)
            if cand is None:
                break

            it += 1
            if self._merit(cand) <= self._merit(center) + 1e-12 * (
                    1.0 + abs(self._merit(center))):
                step = cand.x - center.x
                if np.linalg.norm(step) > 1e-300:
                    prev_dir = step
                center = cand
                if max(abs(alpha), abs(beta)) >= cfg.boundary_frac:
                    delta = min(delta * cfg.expand, cfg.delta_max)
                stall = 0
            else:
                delta *= cfg.shrink
                stall += 1
                if stall >= cfg.stall_limit and not _restart("stalled"):
                    status = "stalled"
                    break

            rec = {"iter": it, "n_evals": self.n_evals, "delta": delta,
                   "alpha": alpha, "beta": beta,
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
    """Reduced-space SCP with the GEK 2D sub-optimization (functional API)."""
    return ReducedSpaceDriver(
        evaluate, lower_bounds, upper_bounds, GEKProposer(), config,
        on_iteration,
    ).run(x0)
