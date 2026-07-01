# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""JAX capsule (bar) characteristic for the 3-D node-based truss.

The 3-D bar characteristic is defined **directly from the two segment endpoints** as the
GP-smoothed distance to the segment, rather than through a ``(θ, φ)`` orientation. That
avoids the azimuth singularity of ``θ = atan2(dy, dx)`` for axis-aligned (e.g. vertical)
bars — which are common in a lattice — so ``jax.grad`` yields finite, correct endpoint
sensitivities everywhere. The smoothing is byte-for-byte the same GP kernel used by the
NumPy 3-D free mapper (``compute_local_characteristic_3d_free_with_grad_np``), so 3-D truss
bars are consistent with Free3D bars.
"""
from __future__ import annotations

import numpy as np

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    _HAVE_JAX = True
except Exception:                                    # pragma: no cover
    _HAVE_JAX = False

_DELTA_MIN = 1e-6
_EPS = 1e-7


def _capsule_W(p, P1, P2, h, r_gp):
    """GP-smoothed characteristic of the capsule with axis segment [P1, P2], radius h/2."""
    d = P2 - P1
    dd = jnp.dot(d, d) + _EPS
    t = jnp.clip(jnp.dot(p - P1, d) / dd, 0.0, 1.0)     # nearest point param on segment
    closest = P1 + t * d
    upsi = jnp.sqrt(jnp.sum((p - closest) ** 2) + _EPS)  # distance to the segment
    zeta = upsi - h / 2.0
    z = jnp.clip(zeta / r_gp, -1.0 + _EPS, 1.0 - _EPS)
    delta = (1.0 / jnp.pi) * (jnp.arccos(z) - z * jnp.sqrt(1.0 - z ** 2))
    return jnp.where(zeta < -r_gp, 1.0,
                     jnp.where(zeta > r_gp, 0.0, _DELTA_MIN + (1.0 - _DELTA_MIN) * delta))


if _HAVE_JAX:
    _grad_capsule = jax.grad(_capsule_W, argnums=(1, 2, 3))   # d/dP1, d/dP2, d/dh

    @jax.jit
    def _capsule_batch(pts, P1, P2, h, r_gp):
        W = jax.vmap(lambda p: _capsule_W(p, P1, P2, h, r_gp))(pts)
        dP1, dP2, dh = jax.vmap(lambda p: _grad_capsule(p, P1, P2, h, r_gp))(pts)
        return W, dP1, dP2, dh


def capsule_batch(pts, P1, P2, h, r_gp):
    """Return ``(W, dW/dP1, dW/dP2, dW/dh)`` at points ``pts`` (M,3) for one bar.

    Shapes: ``W`` (M,), ``dP1``/``dP2`` (M,3), ``dh`` (M,). NumPy in/out.
    """
    if not _HAVE_JAX:
        raise ImportError("jax is required for the 3-D truss capsule primitive")
    W, dP1, dP2, dh = _capsule_batch(
        jnp.asarray(pts), jnp.asarray(P1, dtype=float), jnp.asarray(P2, dtype=float),
        float(h), float(r_gp))
    return np.asarray(W), np.asarray(dP1), np.asarray(dP2), np.asarray(dh)
