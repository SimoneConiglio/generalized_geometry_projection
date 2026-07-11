# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Compact NumPy MMA (single inequality constraint) used as a *teacher*.

The TRANSFORMER_OPT policy is trained to imitate MMA's step map (see
:mod:`scp_uno.transformer_opt_core`). GGP problems have exactly one
inequality constraint (volume), for which the MMA subproblem is separable
and its dual is one-dimensional — solvable to machine precision by
bisection on the multiplier. This gives a fast, dependency-free teacher
whose steps match classical MMA (Svanberg 1987) with the standard
oscillation-driven asymptote update.

Also provides synthetic *task families with known optima* used to roll the
teacher out during training:

* ``toy_simp``  — ``min sum(k_i / (x_i + eps))  s.t.  v.x <= V`` — the
  classic reciprocal/volume structure of compliance topology optimization
  (monotone decreasing objective pressing against the volume cap);
* ``quad_lin``  — anisotropic quadratic with one linear constraint
  (KKT-analytic optimum);
* ``valley``    — rotated Rosenbrock-like curved valley, unconstrained.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Asymptote state (classical mmasub update rule)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class AsymptoteTracker:
    """Per-variable asymptote half-widths with the oscillation update.

    ``width_i`` shrinks (``decr``) when variable ``i`` oscillated over the
    last two steps and grows (``incr``) when it moved consistently, clamped
    to ``[asy_min, asy_max] * range`` — exactly the classical rule (and the
    coefficients of the short-cantilever preset by default).
    """

    lb: np.ndarray
    ub: np.ndarray
    asy_init: float = 0.01
    asy_min: float = 0.0001
    asy_max: float = 0.01
    incr: float = 1.2
    decr: float = 0.4

    def __post_init__(self):
        self.range = np.maximum(self.ub - self.lb, 1e-30)
        self.width = self.asy_init * self.range.copy()
        self._hist: list = []          # last three accepted iterates

    def update(self, x: np.ndarray) -> None:
        """Register a new (accepted) iterate and adapt the widths."""
        h = self._hist
        if len(h) >= 2:
            prod = (x - h[-1]) * (h[-1] - h[-2])
            factor = np.where(prod > 0, self.incr, np.where(prod < 0, self.decr, 1.0))
            self.width = np.clip(
                self.width * factor, self.asy_min * self.range, self.asy_max * self.range
            )
        h.append(x.copy())
        if len(h) > 3:
            h.pop(0)

    def oscillation(self) -> np.ndarray:
        """Sign of the last two accepted steps' product per variable:
        -1 oscillating, +1 moving consistently, 0 unknown/stationary."""
        h = self._hist
        if len(h) < 3:
            return np.zeros(len(self.lb))
        return np.sign((h[-1] - h[-2]) * (h[-2] - h[-3]))

    def last_step(self) -> np.ndarray:
        """Most recent accepted step, in fractions of the variable range."""
        h = self._hist
        if len(h) < 2:
            return np.zeros(len(self.lb))
        return (h[-1] - h[-2]) / self.range


# --------------------------------------------------------------------------- #
# One MMA step (single inequality constraint, dual bisection)
# --------------------------------------------------------------------------- #
def mma_step(
    x: np.ndarray,
    grad_f: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    width: np.ndarray,
    move: float,
    c_val: Optional[float] = None,
    grad_c: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the MMA step ``dx`` from ``x`` (classical, d=0 formulation).

    ``c_val``/``grad_c`` describe one inequality constraint ``c(x) <= 0``;
    pass ``None`` for unconstrained problems. ``width`` are the current
    asymptote half-widths, ``move`` the move limit (fraction of range).
    """
    rng = np.maximum(ub - lb, 1e-30)
    L = x - width
    U = x + width
    alpha = np.maximum(lb, np.maximum(x - move * rng, L + 0.1 * width))
    beta = np.minimum(ub, np.minimum(x + move * rng, U - 0.1 * width))

    def pq(grad):
        gp = np.maximum(grad, 0.0)
        gm = np.maximum(-grad, 0.0)
        # classical regularization keeps both terms slightly positive
        p = (U - x) ** 2 * (1.001 * gp + 0.001 * gm + 1e-9 / rng)
        q = (x - L) ** 2 * (0.001 * gp + 1.001 * gm + 1e-9 / rng)
        return p, q

    p0, q0 = pq(np.asarray(grad_f, float))
    constrained = c_val is not None and grad_c is not None
    if constrained:
        p1, q1 = pq(np.asarray(grad_c, float))
        r1 = float(c_val) - float(np.sum(p1 / (U - x) + q1 / (x - L)))
    else:
        p1 = q1 = None

    def primal(lam: float) -> np.ndarray:
        P = p0 + (lam * p1 if constrained else 0.0)
        Q = q0 + (lam * q1 if constrained else 0.0)
        sp, sq = np.sqrt(P), np.sqrt(Q)
        y = (L * sp + U * sq) / np.maximum(sp + sq, 1e-300)
        return np.clip(y, alpha, beta)

    if not constrained:
        return primal(0.0) - x

    def h(lam: float) -> float:
        y = primal(lam)
        return r1 + float(np.sum(p1 / (U - y) + q1 / (y - L)))

    if h(0.0) <= 0.0:
        return primal(0.0) - x
    lo, hi = 0.0, 1.0
    for _ in range(80):
        if h(hi) < 0.0:
            break
        lo, hi = hi, hi * 10.0
    else:  # constraint cannot be satisfied inside the move box: max effort
        return primal(hi) - x
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if h(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return primal(hi) - x


# --------------------------------------------------------------------------- #
# Synthetic constrained tasks with known optima
# --------------------------------------------------------------------------- #
class TeacherTask:
    """Differentiable task on [0, 1]^d with (at most) one inequality
    constraint and a known optimum ``x_star``."""

    def __init__(self, kind: str, d: int, rng: np.random.Generator):
        self.kind = kind
        self.d = d
        self.m = 0
        if kind == "toy_simp":
            self.m = 1
            self.kappa = 10 ** rng.uniform(-1.0, 1.0, d)
            self.eps = 0.1
            self.v = rng.uniform(0.5, 1.5, d)
            theta = rng.uniform(0.2, 0.6)
            self.V = theta * float(np.sum(self.v))       # volume cap (active)
            self.scale = 10 ** rng.uniform(-1, 2)
            self.x_star = self._simp_optimum()
        elif kind in ("quad_lin", "quad_free"):
            self.m = 1 if kind == "quad_lin" else 0
            k = rng.integers(2, min(d, 12) + 1)
            B = rng.standard_normal((d, k)) / np.sqrt(d)
            self.A = B @ B.T + 10 ** rng.uniform(-2.5, -1) * np.eye(d)
            self.scale = 10 ** rng.uniform(-1, 2)
            if kind == "quad_free":
                self.xu = rng.uniform(0.15, 0.85, d)
                self.x_star = self.xu.copy()
            else:
                for _ in range(20):
                    self.xu = rng.uniform(0.2, 0.8, d)   # unconstrained min
                    self.a = rng.uniform(0.5, 1.5, d)
                    # boundary between the start region and xu half the time
                    off = rng.uniform(-0.1, 0.2) * np.sum(self.a)
                    self.b = float(self.a @ self.xu) - off
                    x_star = self._quad_optimum()
                    if np.all(x_star > 0.02) and np.all(x_star < 0.98):
                        break
                self.x_star = x_star
        elif kind == "valley":
            self.R, _ = np.linalg.qr(rng.standard_normal((d, d)))
            self.x_star = rng.uniform(0.25, 0.75, d)
            self.scale = 10 ** rng.uniform(-2, 0)
        else:
            raise ValueError(kind)

    # ---- optima -----------------------------------------------------------
    def _simp_optimum(self) -> np.ndarray:
        """KKT: k_i/(x+eps)^2 = lam v_i on the active volume constraint."""
        def x_of(lam):
            return np.clip(np.sqrt(self.kappa / (lam * self.v)) - self.eps, 0.0, 1.0)

        lo, hi = 1e-12, 1e12
        for _ in range(200):
            mid = np.sqrt(lo * hi)
            if float(self.v @ x_of(mid)) > self.V:
                lo = mid          # too much volume: raise the price
            else:
                hi = mid
        return x_of(np.sqrt(lo * hi))

    def _quad_optimum(self) -> np.ndarray:
        if float(self.a @ self.xu) <= self.b:
            return self.xu.copy()
        Ainv_a = np.linalg.solve(self.A, self.a)
        lam = (self.a @ self.xu - self.b) / max(self.a @ Ainv_a, 1e-30)
        return self.xu - lam * Ainv_a

    # ---- evaluation ---------------------------------------------------------
    def evaluate(self, x: np.ndarray):
        """Return ``(f, grad_f, c, grad_c)`` (``c/grad_c`` are None if m=0)."""
        if self.kind == "toy_simp":
            f = self.scale * float(np.sum(self.kappa / (x + self.eps)))
            g = -self.scale * self.kappa / (x + self.eps) ** 2
            c = float(self.v @ x / self.V - 1.0)
            return f, g, c, self.v / self.V
        if self.kind in ("quad_lin", "quad_free"):
            e = x - self.xu
            f = self.scale * float(e @ self.A @ e)
            g = self.scale * 2.0 * self.A @ e
            if self.kind == "quad_free":
                return f, g, None, None
            c = float(self.a @ x - self.b) / max(np.linalg.norm(self.a), 1e-30)
            return f, g, c, self.a / max(np.linalg.norm(self.a), 1e-30)
        y = self.R @ (x - self.x_star)
        a, b = y[:-1], y[1:]
        t = b - a * a
        f = self.scale * float(100 * t @ t + a @ a)
        gy = np.zeros(self.d)
        gy[:-1] = -400 * t * a + 2 * a
        gy[1:] += 200 * t
        return f, self.scale * self.R.T @ gy, None, None


def sample_teacher_task(rng: np.random.Generator, dims=(16, 32, 64, 108, 160)
                        ) -> TeacherTask:
    kind = rng.choice(["toy_simp", "quad_lin", "quad_free", "valley"],
                      p=[0.4, 0.25, 0.2, 0.15])
    return TeacherTask(kind, int(rng.choice(dims)), rng)
