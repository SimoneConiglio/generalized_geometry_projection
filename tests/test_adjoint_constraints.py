# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Finite-difference gate for the adjoint-based constraint sensitivities.

The displacement (and later stress) constraints are *not* self-adjoint, so their design
sensitivity goes through an adjoint solve reusing the physics discipline's factorized K.
These tests verify the analytic adjoint gradient against a finite-difference of the true
perturbed FE solve — the key correctness gate for Workstreams A/B. Needs FEniCS.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dolfin")
pytest.importorskip("gemseo")


def _build(spec_path="ggp/cli/presets/short_cantilever.yaml"):
    from ggp.problem.loader import load_problem
    from ggp.discretisation.fem import FEMDiscretiser
    from ggp.geometry.io.registry import get_reader
    from ggp.gemseo_wrappers.physics_discipline import GGPPhysicsDiscipline

    spec = load_problem(spec_path)
    dom = get_reader(spec.geometries[0].type).read(spec.geometries[0])
    analysis = FEMDiscretiser().discretise(dom, spec)
    fixed = sorted(set(list(analysis.point_fixed_dofs) +
        [d for bc in analysis.bcs_applied for d in bc.get_boundary_values().keys()]))
    phys = GGPPhysicsDiscipline(
        V_u=analysis.function_spaces["u"], ke_ref=analysis.ke_ref, fixed_dofs=fixed,
        f_vec=analysis.load_vector.copy(), mesh_area=1.0, volfrac=0.4, p_penalty=3.0,
        Emin=1e-6, E0=1.0, fem_solver="direct")
    return spec, dom, analysis, phys


def test_adjoint_solve_solves_the_system():
    _, _, _, phys = _build()
    n = phys.num_elements
    rng = np.random.default_rng(1)
    rho = np.clip(0.5 + 0.2 * rng.standard_normal(n), 0.05, 1.0)
    phys._run({"rho_E": rho, "rho_V": rho})
    b = rng.standard_normal(phys.V_u.dim())
    b[phys.fixed_dofs] = 0.0
    lam = phys.adjoint_solve(b)
    # K λ should reproduce b on the free DOFs
    res = phys._K_global @ lam - b
    res[phys.fixed_dofs] = 0.0
    assert np.linalg.norm(res) / (np.linalg.norm(b) + 1e-30) < 1e-8


def test_displacement_adjoint_matches_fd():
    from ggp.optimization.pipeline import _DisplacementConstraintDiscipline

    _, dom, analysis, phys = _build()
    n = phys.num_elements
    rng = np.random.default_rng(0)
    rho = np.clip(0.5 + 0.2 * rng.standard_normal(n), 0.05, 1.0)
    phys._run({"rho_E": rho, "rho_V": rho})

    V_u = analysis.function_spaces["u"]
    dc = V_u.tabulate_dof_coordinates()
    ydofs = np.asarray(V_u.sub(1).dofmap().dofs())
    tgt = np.array([dom.metadata["Lx"], dom.metadata["Ly"] / 2.0])
    nn = int(ydofs[np.argmin(np.linalg.norm(dc[ydofs] - tgt, axis=1))])
    ell = np.zeros(V_u.dim()); ell[nn] = 1.0
    u_max = 5.0

    disp = _DisplacementConstraintDiscipline(phys, ell, u_max, n, volfrac=0.4)
    disp._run({"rho_E": rho}); disp._compute_jacobian()
    g_an = disp.jac["displacement"]["rho_E"].flatten()

    def gval(rho_in):
        phys._run({"rho_E": rho_in, "rho_V": rho_in})
        return abs(float(ell @ phys.last_u)) / u_max - 1.0

    eps = 1e-6
    elems = rng.choice(n, size=15, replace=False)
    for e in elems:
        rp = rho.copy(); rp[e] += eps
        rm = rho.copy(); rm[e] -= eps
        fd = (gval(rp) - gval(rm)) / (2 * eps)
        # rtol loose because FD on a stiff solve is noisy on small-magnitude entries;
        # atol covers those. The adjoint is exact.
        assert abs(fd - g_an[e]) < 1e-3 * abs(fd) + 5e-5, (e, fd, g_an[e])


@pytest.mark.parametrize("kind", ["verbart", "pmean"])
def test_stress_adjoint_matches_fd(kind):
    pytest.importorskip("jax")
    from ggp.optimization.pipeline import _StressConstraintDiscipline

    _, _, analysis, phys = _build()
    n = phys.num_elements
    rng = np.random.default_rng(3)
    rho = np.clip(0.5 + 0.2 * rng.standard_normal(n), 0.05, 1.0)
    phys._run({"rho_E": rho, "rho_V": rho})

    sd = _StressConstraintDiscipline(phys, analysis.se_ref, sigma_lim=2.0,
                                     num_elements=n, dim=2, q=0.5, P=8.0, kind=kind)
    sd._run({"rho_E": rho}); sd._compute_jacobian()
    g_an = sd.jac["stress"]["rho_E"].flatten()

    def Gval(rho_in):
        phys._run({"rho_E": rho_in, "rho_V": rho_in})
        return sd.kernel.value(rho_in, phys.last_u)

    eps = 1e-6
    for e in rng.choice(n, size=15, replace=False):
        rp = rho.copy(); rp[e] += eps
        rm = rho.copy(); rm[e] -= eps
        fd = (Gval(rp) - Gval(rm)) / (2 * eps)
        assert abs(fd - g_an[e]) < 3e-3 * abs(fd) + 5e-5, (e, fd, g_an[e])


def test_adaptive_stress_adjoint_matches_fd_at_frozen_allowable():
    # With the Le-Norato adaptive allowable, sigma_allow is FROZEN within an iteration,
    # so the adjoint gradient at the current sigma_allow must still match a finite
    # difference of G evaluated at that same (fixed) allowable.
    pytest.importorskip("jax")
    from ggp.optimization.pipeline import _StressConstraintDiscipline

    _, _, analysis, phys = _build()
    n = phys.num_elements
    rng = np.random.default_rng(5)
    rho = np.clip(0.5 + 0.2 * rng.standard_normal(n), 0.05, 1.0)
    phys._run({"rho_E": rho, "rho_V": rho})

    sd = _StressConstraintDiscipline(phys, analysis.se_ref, sigma_lim=2.0,
                                     num_elements=n, dim=2, P=8.0, kind="verbart",
                                     adaptive=True, alpha=0.5)
    # two runs so the allowable has actually moved off the nominal limit
    sd._run({"rho_E": rho})
    sd._run({"rho_E": rho})
    assert sd.sigma_allow != sd.sigma_lim
    sa = sd.sigma_allow
    sd._compute_jacobian()
    g_an = sd.jac["stress"]["rho_E"].flatten()

    def Gval(rho_in):
        phys._run({"rho_E": rho_in, "rho_V": rho_in})
        return sd.kernel.value(rho_in, phys.last_u, sigma_allow=sa)

    eps = 1e-6
    for e in rng.choice(n, size=10, replace=False):
        rp = rho.copy(); rp[e] += eps
        rm = rho.copy(); rm[e] -= eps
        fd = (Gval(rp) - Gval(rm)) / (2 * eps)
        assert abs(fd - g_an[e]) < 3e-3 * abs(fd) + 5e-5, (e, fd, g_an[e])


def test_stress_exclusion_mask_covers_load_region_and_voids():
    pytest.importorskip("jax")
    from ggp.optimization.pipeline import _stress_exclusion_mask

    _, _, analysis, phys = _build()
    V_dg = analysis.function_spaces["dg"]
    ec = V_dg.tabulate_dof_coordinates()
    # no exclusions requested and no voids on the plain SC -> None
    assert _stress_exclusion_mask(analysis, ec, 0.0) is None
    # load-region exclusion masks a small neighbourhood of the loaded DOF
    m = _stress_exclusion_mask(analysis, ec, 3.0)
    assert m is not None and 0 < int((m == 0).sum()) < 40
    # non-design (empty) elements are always excluded, even with radius 0
    analysis.empty_elements = [0, 1, 2]
    m2 = _stress_exclusion_mask(analysis, ec, 0.0)
    assert m2 is not None and np.all(m2[[0, 1, 2]] == 0.0) and m2.sum() == len(ec) - 3


def test_adaptive_allowable_moves_in_the_right_direction():
    pytest.importorskip("jax")
    from ggp.optimization.pipeline import _StressConstraintDiscipline

    _, _, analysis, phys = _build()
    n = phys.num_elements
    rho = np.full(n, 0.9)
    phys._run({"rho_E": rho, "rho_V": rho})

    sd = _StressConstraintDiscipline(phys, analysis.se_ref, sigma_lim=2.0,
                                     num_elements=n, dim=2, P=8.0, kind="verbart",
                                     adaptive=True, alpha=0.5)
    sd._run({"rho_E": rho})             # records sigma_max/sa_star, allowable untouched yet
    assert sd.sigma_allow == sd.sigma_lim
    smax, sa_star = sd._last_sigma_max, sd._last_sa_star
    assert smax > 0.0 and sa_star > 0.0
    sd._run({"rho_E": rho})             # applies the state-free damped update
    target = np.clip(sa_star * sd.sigma_lim / smax,
                     sd.sigma_lim / sd.max_correction, sd.sigma_lim * 1e3)
    expected = 0.5 * target + 0.5 * sd.sigma_lim
    assert sd.sigma_allow == pytest.approx(
        float(np.clip(expected, sd.sigma_lim / sd.max_correction, sd.sigma_lim * 1e3)))
    # fixed-point semantics at the FIXED design: allowable relative to the critical
    # allowable follows the margin/overshoot of the true max stress
    for _ in range(40):
        sd._run({"rho_E": rho})
    if smax > sd.sigma_lim:
        assert sd.sigma_allow < sa_star * 1.01      # overshoot -> binds/violated
    else:
        assert sd.sigma_allow > sa_star * 0.99      # margin -> feasible, removal allowed
