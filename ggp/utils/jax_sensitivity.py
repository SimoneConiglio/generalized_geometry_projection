# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""JAX autodiff core for constraint sensitivities (stress).

The messy, *nonlinear* part of a stress constraint — element stress recovery, the von
Mises norm, the stress relaxation, and the aggregation — is written here as pure JAX and
differentiated with ``jax.grad``/``vjp``, so no von-Mises/aggregation derivative is ever
hand-coded. The *linear* algebra (the primal ``Ku=f`` and the single adjoint ``Kλ=∂g/∂u``)
stays in deterministic scipy in the physics discipline; this module only supplies the two
partials the discipline needs:

    G(ρ, u),   ∂G/∂ρ |_explicit   (u held fixed),   ∂G/∂u   (ρ held fixed)

and the discipline forms the total ``dG/dρ_e = ∂G/∂ρ_e − λ_eᵀ(dE_drho_e·ke_ref)u_e``.

Stress model (default ``kind="verbart"`` — the **canonical unified aggregation-relaxation** of
Verbart, Langelaar & van Keulen 2017, SMO 55:663-679, adopted by the Coniglio thesis's Eulerian
chapter):
  * solid-material (unit-E) stress  ``σ̂_e = se_ref · u_e``            (NOT scaled by density)
  * reformulated local constraint   ``g_e = ρ_e (σvm(σ̂_e)/σ_lim − 1) ≤ 0``  — the density
                                     multiplies the *constraint*, making the constraint set
                                     design-independent (void elements trivially feasible),
                                     which is what makes singular optima accessible
  * lower-bound KS aggregation      ``G = (1/P) ln( (1/N) Σ_e exp(P g_e) ) ≤ 0``
    One parameter ``P`` controls both aggregation sharpness and the induced relaxation;
    P → ∞ recovers ``max_e g_e ≤ 0``, i.e. the original local constraints.
A legacy qp-style path (``σ_e = ρ_e^q σ̂_e`` + p-norm/p-mean/KS of the stress ratios) remains
selectable via ``kind in ("pnorm", "pmean", "ks")``.
"""
from __future__ import annotations

import numpy as np

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    _HAVE_JAX = True
except Exception:                                    # pragma: no cover - jax optional
    _HAVE_JAX = False


def von_mises(sig, dim: int):
    """von Mises stress from Voigt stress rows ``sig`` (…, n_stress). JAX or NumPy."""
    xp = jnp if _HAVE_JAX and not isinstance(sig, np.ndarray) else np
    if dim == 2:                                     # plane stress: σxx, σyy, τxy
        sxx, syy, txy = sig[..., 0], sig[..., 1], sig[..., 2]
        return xp.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * txy**2 + 1e-30)
    sxx, syy, szz = sig[..., 0], sig[..., 1], sig[..., 2]
    txy, tyz, txz = sig[..., 3], sig[..., 4], sig[..., 5]
    return xp.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
                   + 3.0 * (txy**2 + tyz**2 + txz**2) + 1e-30)


def _aggregate(s, P: float, kind: str):
    """Aggregate the (non-negative) stress ratios ``s`` to a single scalar.

    ``pmean``/``ks`` are *lower-bound* aggregations (≤ true max) → they relax the feasible
    domain (the "unified" relaxation); ``pnorm`` is the classic upper-bound P-norm.
    """
    if kind == "pnorm":
        return (jnp.sum(s**P))**(1.0 / P)
    if kind == "pmean":
        return (jnp.mean(s**P))**(1.0 / P)
    # KS (lower bound, mean form)
    m = jnp.max(s)
    return m + jnp.log(jnp.mean(jnp.exp(P * (s - m)))) / P


def _stress_ratio(rho, u, cell_dofs, se_ref, sigma_lim, q, dim):
    """Per-element relaxed von Mises stress ratio ``s_e = σvm(ρ_e^q σ0_e)/σ_lim``.

    With ``q = 0`` this is the solid-material (unit-E) stress ratio used by the
    Verbart formulation.
    """
    U = u[cell_dofs]                     # (n_elem, dofs_per_cell)
    sig0 = U @ se_ref.T                  # (n_elem, n_stress) — unit-E stress
    sig = (rho**q)[:, None] * sig0       # relaxed stress
    return von_mises(sig, dim) / sigma_lim


def _agg_constraint(rho, u, cell_dofs, se_ref, sigma_lim, q, dim, P, kind, active):
    if kind == "verbart":
        # Unified aggregation-relaxation (Verbart 2017): solid-material stress ratio,
        # density multiplies the reformulated local constraint, lower-bound KS over g_e.
        # Excluded elements (active=0, e.g. the load-introduction singularity) enter
        # with g_e = 0 — exactly the trivially-feasible role void elements already play.
        s = _stress_ratio(rho, u, cell_dofs, se_ref, sigma_lim, 0.0, dim)
        g = (rho * active) * (s - 1.0)
        m = jnp.max(P * g)               # log-sum-exp shift (g_e is O(1))
        return (m + jnp.log(jnp.mean(jnp.exp(P * g - m)))) / P
    s = _stress_ratio(rho, u, cell_dofs, se_ref, sigma_lim, q, dim) * active
    return _aggregate(s, P, kind) - 1.0


class StressConstraintKernel:
    """Jitted evaluator of the aggregated stress constraint value and its partials."""

    def __init__(self, cell_dofs, se_ref, sigma_lim, q=0.5, P=8.0, kind="verbart", dim=2,
                 active_mask=None):
        if not _HAVE_JAX:
            raise ImportError("jax is required for the stress constraint kernel")
        self.cell_dofs = jnp.asarray(np.asarray(cell_dofs), dtype=jnp.int32)
        self.se_ref = jnp.asarray(np.asarray(se_ref, dtype=float))
        self.sigma_lim = float(sigma_lim)
        self.q = float(q)
        self.P = float(P)
        self.kind = kind
        self.dim = int(dim)
        # Elements the stress measure applies to. Excluding the load-introduction
        # vicinity is standard for point loads: the FE stress there is singular
        # (grows unboundedly with mesh refinement), so no finite allowable can hold it.
        n_elem = int(self.cell_dofs.shape[0])
        if active_mask is None:
            self.active_mask = jnp.ones(n_elem)
        else:
            self.active_mask = jnp.asarray(np.asarray(active_mask, dtype=float))

        # The allowable is a *runtime* argument so an adaptive scheme (Le-Norato 2010)
        # can move it between iterations without retracing: jit treats the scalar as a
        # weak-typed traced value.
        def G(rho, u, sigma_allow):
            return _agg_constraint(rho, u, self.cell_dofs, self.se_ref,
                                   sigma_allow, self.q, self.dim, self.P, self.kind,
                                   self.active_mask)

        self._G = jax.jit(G)
        self._dG_drho = jax.jit(jax.grad(G, argnums=0))   # explicit ρ-partial (u fixed)
        self._dG_du = jax.jit(jax.grad(G, argnums=1))     # ∂G/∂u (ρ fixed)

    def value(self, rho, u, sigma_allow=None) -> float:
        sa = self.sigma_lim if sigma_allow is None else float(sigma_allow)
        return float(self._G(jnp.asarray(rho), jnp.asarray(u), sa))

    def value_and_grads(self, rho, u, sigma_allow=None):
        """Return ``(G, ∂G/∂ρ_explicit, ∂G/∂u)`` as NumPy arrays.

        ``sigma_allow`` (default: the nominal ``sigma_lim``) is the effective allowable the
        constraint is evaluated against; under the adaptive scheme it is frozen within an
        iteration, so it is a plain constant here (no derivative wrt it is needed).
        """
        sa = self.sigma_lim if sigma_allow is None else float(sigma_allow)
        r = jnp.asarray(rho); uu = jnp.asarray(u)
        g = float(self._G(r, uu, sa))
        dgr = np.asarray(self._dG_drho(r, uu, sa))
        dgu = np.asarray(self._dG_du(r, uu, sa))
        return g, dgr, dgu

    def elem_stress_ratio(self, rho, u):
        """Per-element von Mises stress ratio ``σvm/σ_lim`` (for post-processing).

        Under the Verbart formulation the stress itself is unrelaxed (solid-material),
        so the ratio is reported with ``q = 0`` there.
        """
        q = 0.0 if self.kind == "verbart" else self.q
        s = _stress_ratio(jnp.asarray(rho), jnp.asarray(u), self.cell_dofs, self.se_ref,
                          self.sigma_lim, q, self.dim)
        return np.asarray(s)

    def max_true_stress(self, rho, u, rho_solid: float = 0.5) -> float:
        """True (unrelaxed, solid-material) max von Mises stress over material elements.

        Only *active* elements with ``ρ_e >= rho_solid`` count — the physically meaningful
        maximum the Le-Norato adaptive allowable drives to ``sigma_lim`` (excluded
        load-introduction elements are outside the stress measure by construction).
        Returns 0.0 when no element qualifies (e.g. a near-void design early in a run)
        so callers can skip the adaptive update.
        """
        s = _stress_ratio(jnp.asarray(rho), jnp.asarray(u), self.cell_dofs, self.se_ref,
                          1.0, 0.0, self.dim)          # unit allowable, q=0 -> raw σvm
        s = np.asarray(s)
        mask = (np.asarray(rho) >= rho_solid) & (np.asarray(self.active_mask) > 0.5)
        return float(s[mask].max()) if np.any(mask) else 0.0
