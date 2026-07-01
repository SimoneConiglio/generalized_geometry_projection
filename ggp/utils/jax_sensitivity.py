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

Stress model (documented default — **Verbart-style unified aggregation-relaxation**):
  * element (unit-E) Voigt stress   ``σ0_e = se_ref · u_e``
  * relaxation                      ``σ_e = ρ_e^q · σ0_e``          (q < p removes the
                                     stress singularity; void ⇒ stress → 0)
  * local stress ratio              ``s_e = σvm(σ_e) / σ_lim``
  * aggregated constraint           ``G = Π_P(s) − 1``              (lower-bound aggregation
                                     ⇒ relaxes the feasible set; one parameter P controls
                                     both aggregation sharpness and relaxation)
The exact per-element `g_e`/relaxation of the Coniglio thesis can be dropped in by editing
`_stress_ratio` / `_aggregate` only; everything downstream (autodiff, adjoint) is unchanged.
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
    """Per-element relaxed von Mises stress ratio ``s_e = σvm(ρ_e^q σ0_e)/σ_lim``."""
    U = u[cell_dofs]                     # (n_elem, dofs_per_cell)
    sig0 = U @ se_ref.T                  # (n_elem, n_stress) — unit-E stress
    sig = (rho**q)[:, None] * sig0       # relaxed stress
    return von_mises(sig, dim) / sigma_lim


def _agg_constraint(rho, u, cell_dofs, se_ref, sigma_lim, q, dim, P, kind):
    s = _stress_ratio(rho, u, cell_dofs, se_ref, sigma_lim, q, dim)
    return _aggregate(s, P, kind) - 1.0


class StressConstraintKernel:
    """Jitted evaluator of the aggregated stress constraint value and its partials."""

    def __init__(self, cell_dofs, se_ref, sigma_lim, q=0.5, P=8.0, kind="pmean", dim=2):
        if not _HAVE_JAX:
            raise ImportError("jax is required for the stress constraint kernel")
        self.cell_dofs = jnp.asarray(np.asarray(cell_dofs), dtype=jnp.int32)
        self.se_ref = jnp.asarray(np.asarray(se_ref, dtype=float))
        self.sigma_lim = float(sigma_lim)
        self.q = float(q)
        self.P = float(P)
        self.kind = kind
        self.dim = int(dim)

        def G(rho, u):
            return _agg_constraint(rho, u, self.cell_dofs, self.se_ref,
                                   self.sigma_lim, self.q, self.dim, self.P, self.kind)

        self._G = jax.jit(G)
        self._dG_drho = jax.jit(jax.grad(G, argnums=0))   # explicit ρ-partial (u fixed)
        self._dG_du = jax.jit(jax.grad(G, argnums=1))     # ∂G/∂u (ρ fixed)

    def value(self, rho, u) -> float:
        return float(self._G(jnp.asarray(rho), jnp.asarray(u)))

    def value_and_grads(self, rho, u):
        """Return ``(G, ∂G/∂ρ_explicit, ∂G/∂u)`` as NumPy arrays."""
        r = jnp.asarray(rho); uu = jnp.asarray(u)
        g = float(self._G(r, uu))
        dgr = np.asarray(self._dG_drho(r, uu))
        dgu = np.asarray(self._dG_du(r, uu))
        return g, dgr, dgu

    def elem_stress_ratio(self, rho, u):
        """Per-element von Mises stress ratio ``σvm/σ_lim`` (for post-processing)."""
        s = _stress_ratio(jnp.asarray(rho), jnp.asarray(u), self.cell_dofs, self.se_ref,
                          self.sigma_lim, self.q, self.dim)
        return np.asarray(s)
