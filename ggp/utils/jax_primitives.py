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


# --------------------------------------------------------------------------- #
# 3-D box primitive (solid analogue of the 2-D rectangle: six quintic faces)
# --------------------------------------------------------------------------- #
def _wval(z, sigma):
    """Quintic-smoothed half-plane: 1 inside (z<-σ), 0 outside (z>σ). Matches the 2-D rect."""
    val = (0.5 - (15.0 / (16.0 * sigma)) * z + (5.0 / (8.0 * sigma ** 3)) * z ** 3
           - (3.0 / (16.0 * sigma ** 5)) * z ** 5)
    return jnp.where(z < -sigma, 1.0, jnp.where(z > sigma, 0.0, val))


def _box_W(p, center, sizes, theta, phi, sigma):
    """Characteristic of an oriented box: product of six quintic-smoothed faces.

    Long axis n1 points along (θ,φ); n2 is the horizontal perpendicular
    ``(-sinθ, cosθ, 0)`` (fixes the roll deterministically, no singularity); n3 = n1×n2.
    """
    ct, st, cp, sp = jnp.cos(theta), jnp.sin(theta), jnp.cos(phi), jnp.sin(phi)
    n1 = jnp.array([cp * ct, cp * st, sp])
    n2 = jnp.array([-st, ct, 0.0])
    n3 = jnp.array([n1[1] * n2[2] - n1[2] * n2[1],
                    n1[2] * n2[0] - n1[0] * n2[2],
                    n1[0] * n2[1] - n1[1] * n2[0]])
    d = p - center
    xl, yl, zl = jnp.dot(d, n1), jnp.dot(d, n2), jnp.dot(d, n3)
    ax, ay, az = sizes[0] / 2.0, sizes[1] / 2.0, sizes[2] / 2.0
    zetas = jnp.array([-ax - xl, xl - ax, -ay - yl, yl - ay, -az - zl, zl - az])
    return jnp.prod(_wval(zetas, sigma))


if _HAVE_JAX:
    _grad_box = jax.grad(_box_W, argnums=(1, 2, 3, 4))    # d/dcenter, dsizes, dtheta, dphi

    @jax.jit
    def _box_batch(pts, center, sizes, theta, phi, sigma):
        W = jax.vmap(lambda p: _box_W(p, center, sizes, theta, phi, sigma))(pts)
        dC, dS, dT, dP = jax.vmap(
            lambda p: _grad_box(p, center, sizes, theta, phi, sigma))(pts)
        return W, dC, dS, dT, dP


def box_batch(pts, center, sizes, theta, phi, sigma):
    """``(W, dW/dcenter(M,3), dW/dsizes(M,3), dW/dtheta(M,), dW/dphi(M,))`` for one box."""
    if not _HAVE_JAX:
        raise ImportError("jax is required for the 3-D box primitive")
    W, dC, dS, dT, dP = _box_batch(
        jnp.asarray(pts), jnp.asarray(center, dtype=float), jnp.asarray(sizes, dtype=float),
        float(theta), float(phi), float(sigma))
    return np.asarray(W), np.asarray(dC), np.asarray(dS), np.asarray(dT), np.asarray(dP)
