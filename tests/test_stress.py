# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the JAX stress-sensitivity core (FEniCS-free).

Validates the von Mises norm, the aggregation lower-bound property, and — crucially —
that the JAX autodiff partials ``∂G/∂ρ`` and ``∂G/∂u`` agree with finite differences of
the *same* JAX constraint function. (The total adjoint sensitivity through the FE solve is
gated separately in test_adjoint_constraints.py, which needs FEniCS.)
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from ggp.utils.jax_sensitivity import von_mises, StressConstraintKernel


def test_von_mises_2d_known_values():
    # uniaxial σxx=S -> vm=S ; pure shear τ=S -> vm=sqrt(3)*S
    assert abs(float(von_mises(np.array([[2.0, 0.0, 0.0]]), 2)[0]) - 2.0) < 1e-9
    assert abs(float(von_mises(np.array([[0.0, 0.0, 1.0]]), 2)[0]) - np.sqrt(3.0)) < 1e-9
    # equibiaxial σxx=σyy=S -> vm=S
    assert abs(float(von_mises(np.array([[3.0, 3.0, 0.0]]), 2)[0]) - 3.0) < 1e-9


def test_von_mises_3d_known_values():
    # uniaxial in z
    assert abs(float(von_mises(np.array([[0, 0, 5.0, 0, 0, 0]]), 3)[0]) - 5.0) < 1e-9
    # pure shear τxy=S -> sqrt(3) S
    assert abs(float(von_mises(np.array([[0, 0, 0, 2.0, 0, 0]]), 3)[0])
               - np.sqrt(3.0) * 2.0) < 1e-9


def _synthetic_kernel(dim=2, n_elem=20, dpc=8, n_stress=3, seed=0, kind="pmean"):
    rng = np.random.default_rng(seed)
    n_dofs = 40
    cell_dofs = rng.integers(0, n_dofs, size=(n_elem, dpc))
    se_ref = rng.standard_normal((n_stress, dpc))
    k = StressConstraintKernel(cell_dofs, se_ref, sigma_lim=1.5, q=0.5, P=8.0,
                               kind=kind, dim=dim)
    return k, rng, n_dofs, n_elem


@pytest.mark.parametrize("kind", ["verbart", "pmean"])
def test_kernel_drho_matches_fd(kind):
    k, rng, n_dofs, n_elem = _synthetic_kernel(kind=kind)
    rho = rng.uniform(0.1, 1.0, n_elem)
    u = rng.standard_normal(n_dofs)
    _, dgr, _ = k.value_and_grads(rho, u)
    eps = 1e-6
    for e in rng.choice(n_elem, 8, replace=False):
        rp = rho.copy(); rp[e] += eps
        rm = rho.copy(); rm[e] -= eps
        fd = (k.value(rp, u) - k.value(rm, u)) / (2 * eps)
        assert abs(fd - dgr[e]) < 1e-5 + 1e-4 * abs(fd)


@pytest.mark.parametrize("kind", ["verbart", "pmean"])
def test_kernel_du_matches_fd(kind):
    k, rng, n_dofs, n_elem = _synthetic_kernel(kind=kind)
    rho = rng.uniform(0.1, 1.0, n_elem)
    u = rng.standard_normal(n_dofs)
    _, _, dgu = k.value_and_grads(rho, u)
    eps = 1e-6
    for d in rng.choice(n_dofs, 8, replace=False):
        up = u.copy(); up[d] += eps
        um = u.copy(); um[d] -= eps
        fd = (k.value(rho, up) - k.value(rho, um)) / (2 * eps)
        assert abs(fd - dgu[d]) < 1e-6 + 1e-4 * abs(fd)


def test_verbart_is_lower_bound_of_max_local_constraint():
    # G = (1/P) ln(mean exp(P g_e)) is a LOWER bound of max_e g_e (relaxes the
    # feasible set), and approaches it as P grows.
    k, rng, n_dofs, n_elem = _synthetic_kernel(kind="verbart")
    rho = rng.uniform(0.1, 1.0, n_elem)
    u = rng.standard_normal(n_dofs)
    s = k.elem_stress_ratio(rho, u)          # solid-material ratio (q=0)
    g = rho * (s - 1.0)
    G = k.value(rho, u)
    assert G <= g.max() + 1e-9
    k40 = StressConstraintKernel(np.asarray(k.cell_dofs), np.asarray(k.se_ref), 1.5,
                                 P=40.0, kind="verbart", dim=2)
    assert G - 1e-9 <= k40.value(rho, u) <= g.max() + 1e-9


# --------------------------------------------------------------------------- #
# Adaptive allowable (Le-Norato 2010)
# --------------------------------------------------------------------------- #
def test_kernel_value_with_sigma_allow_matches_rebuilt_kernel():
    # evaluating at a runtime allowable == building a kernel with that allowable nominal
    k, rng, n_dofs, n_elem = _synthetic_kernel(kind="verbart")
    rho = rng.uniform(0.1, 1.0, n_elem)
    u = rng.standard_normal(n_dofs)
    k2 = StressConstraintKernel(np.asarray(k.cell_dofs), np.asarray(k.se_ref),
                                sigma_lim=2.75, P=8.0, kind="verbart", dim=2)
    assert k.value(rho, u, sigma_allow=2.75) == pytest.approx(k2.value(rho, u), rel=1e-12)
    # default argument path is the nominal limit (bit-identical to the old behavior)
    assert k.value(rho, u) == pytest.approx(k.value(rho, u, sigma_allow=k.sigma_lim))


def test_max_true_stress_thresholds_on_density():
    k, rng, n_dofs, n_elem = _synthetic_kernel(kind="verbart")
    u = rng.standard_normal(n_dofs)
    rho = np.full(n_elem, 0.9)
    full = k.max_true_stress(rho, u, rho_solid=0.5)
    assert full > 0.0
    # knocking out the max-stress element from the "material" set lowers the max
    s = k.elem_stress_ratio(rho, u) * k.sigma_lim     # raw sigma_vm
    rho2 = rho.copy(); rho2[int(np.argmax(s))] = 0.1
    assert k.max_true_stress(rho2, u, rho_solid=0.5) < full
    # void design -> no material elements -> 0.0 sentinel
    assert k.max_true_stress(np.zeros(n_elem), u) == 0.0


def test_adaptive_update_rule_fixed_point():
    # sigma_allow <- alpha*(sigma_allow*sigma_lim/sigma_max) + (1-alpha)*sigma_allow.
    # If the design responds proportionally (sigma_max proportional to 1/margin), the
    # iteration drives sigma_max -> sigma_lim. Simulate the simplest proportional model:
    # sigma_max = kappa * sigma_allow (tightening the allowable scales the stress down).
    sigma_lim, kappa, alpha = 2.5, 1.4, 0.5
    sigma_allow = sigma_lim
    for _ in range(60):
        sigma_max = kappa * sigma_allow
        target = sigma_allow * sigma_lim / sigma_max
        sigma_allow = alpha * target + (1 - alpha) * sigma_allow
    assert kappa * sigma_allow == pytest.approx(sigma_lim, rel=1e-6)   # sigma_max -> limit
    # directionality: overshoot shrinks the allowable, undershoot grows it
    sa = 2.5
    tgt = sa * sigma_lim / 3.0     # sigma_max=3.0 > limit
    assert alpha * tgt + (1 - alpha) * sa < sa
    tgt = sa * sigma_lim / 2.0     # sigma_max=2.0 < limit
    assert alpha * tgt + (1 - alpha) * sa > sa


def test_verbart_void_elements_are_trivially_feasible():
    # rho_e = 0 kills the local constraint regardless of how high the (solid) stress
    # is there -- the design-independent reformulation that admits singular optima.
    k, rng, n_dofs, n_elem = _synthetic_kernel(kind="verbart")
    u = 10.0 * rng.standard_normal(n_dofs)   # large stresses everywhere
    rho = np.zeros(n_elem)
    assert k.value(rho, u) <= 0.0 + 1e-12    # all g_e = 0 -> G = 0
    # and the gradient wrt u vanishes (no dependence through void elements)
    _, _, dgu = k.value_and_grads(rho, u)
    np.testing.assert_allclose(dgu, 0.0, atol=1e-12)


def test_aggregation_lower_bound_property():
    # pmean and ks (lower-bound) underestimate the true max stress ratio
    k, rng, n_dofs, n_elem = _synthetic_kernel()
    rho = rng.uniform(0.1, 1.0, n_elem)
    u = rng.standard_normal(n_dofs)
    s = k.elem_stress_ratio(rho, u)
    G = k.value(rho, u)               # = aggregate(s) - 1
    assert (G + 1.0) <= s.max() + 1e-9
    # higher P -> closer to the true max
    k2 = StressConstraintKernel(k.cell_dofs, np.asarray(k.se_ref), 1.5, q=0.5, P=40.0,
                                kind="pmean", dim=2)
    assert (k2.value(rho, u) + 1.0) >= (G + 1.0) - 1e-9
