# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the mesh / unit-cell initial-guess package."""
import numpy as np
import pytest

from ggp.initialization import (
    DomainBox,
    Skeleton,
    encode_skeleton,
    fit_thickness_to_volfrac,
    get_pattern,
    is_pattern,
    list_patterns,
)
from ggp.initialization.legacy import make_grid_init_2d, make_grid_init_3d
from ggp.optimization.pipeline import GGPPipeline


# ── Registry ──────────────────────────────────────────────────────────────────

def test_all_four_patterns_registered():
    names = list_patterns()
    assert set(names) == {"tri2d", "quad_star", "tet3d", "hex_star"}
    assert is_pattern("tri2d") and not is_pattern("grid")


def test_unknown_pattern_raises():
    with pytest.raises(ValueError):
        get_pattern("does_not_exist")


# ── Pattern geometry ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["tri2d", "quad_star"])
def test_2d_pattern_within_domain(name):
    box = DomainBox(Lx=60.0, Ly=30.0)
    sk = get_pattern(name).generate(box, cell_size=15.0)
    assert sk.num_edges > 0
    assert sk.dim == 2
    nodes = sk.nodes
    assert (nodes[:, 0] >= -1e-9).all() and (nodes[:, 0] <= 60.0 + 1e-9).all()
    assert (nodes[:, 1] >= -1e-9).all() and (nodes[:, 1] <= 30.0 + 1e-9).all()
    assert (sk.lengths() > 0).all()


@pytest.mark.parametrize("name", ["tet3d", "hex_star"])
def test_3d_pattern_within_domain(name):
    box = DomainBox(Lx=60.0, Ly=30.0, Lz=30.0)
    sk = get_pattern(name).generate(box, cell_size=30.0)
    assert sk.num_edges > 0
    assert sk.dim == 3
    assert (sk.nodes[:, 2] <= 30.0 + 1e-9).all()
    assert (sk.lengths() > 0).all()


@pytest.mark.parametrize("name", ["tri2d", "quad_star", "tet3d", "hex_star"])
def test_finer_cell_size_gives_more_edges(name):
    Lz = None if name in ("tri2d", "quad_star") else 30.0
    box = DomainBox(Lx=60.0, Ly=30.0, Lz=Lz)
    coarse = get_pattern(name).generate(box, cell_size=30.0).num_edges
    fine = get_pattern(name).generate(box, cell_size=15.0).num_edges
    assert fine > coarse


def test_no_duplicate_edges():
    box = DomainBox(Lx=60.0, Ly=30.0)
    sk = get_pattern("tri2d").generate(box, cell_size=15.0)
    canon = np.sort(sk.edges, axis=1)
    assert len(np.unique(canon, axis=0)) == sk.num_edges


def test_filter_edges_in_void_region():
    """L-shape: edges with a midpoint in the void quadrant are dropped."""
    box = DomainBox(
        Lx=60.0, Ly=60.0,
        non_design=[(np.array([30.0, 30.0]), np.array([60.0, 60.0]))],
    )
    sk = get_pattern("tri2d").generate(box, cell_size=15.0)
    filtered = sk.filter_edges_in_domain(box)
    assert filtered.num_edges < sk.num_edges
    assert not box.in_non_design(filtered.midpoints()).any()


# ── Encoder ───────────────────────────────────────────────────────────────────

def test_encode_2d_roundtrip():
    # one horizontal edge from (10,5) to (20,5)
    sk = Skeleton(np.array([[10.0, 5.0], [20.0, 5.0]]), np.array([[0, 1]]))
    lb = np.array([-1.0, -1.0, 0.0, 1.0, -2 * np.pi, 0.0])
    ub = np.array([61.0, 31.0, 67.0, 67.0, 2 * np.pi, 1.0])
    x = encode_skeleton(sk, lb, ub, thickness=3.0, membership=1.0)
    assert (x >= 0).all() and (x <= 1).all()
    phys = lb + x * (ub - lb)
    # [Xc, Yc, L, h, Theta, Mc]
    np.testing.assert_allclose(phys, [15.0, 5.0, 10.0, 3.0, 0.0, 1.0], atol=1e-6)


def test_encode_3d_angles():
    # a +z edge: Theta arbitrary, Phi = +pi/2
    sk = Skeleton(np.array([[5.0, 5.0, 0.0], [5.0, 5.0, 12.0]]), np.array([[0, 1]]))
    lb = np.array([-1.0, -1.0, -1.0, 0.0, 1.0, -2 * np.pi, -2 * np.pi, 0.0])
    ub = np.array([61.0, 31.0, 31.0, 67.0, 67.0, 2 * np.pi, 2 * np.pi, 1.0])
    x = encode_skeleton(sk, lb, ub, thickness=2.0, membership=0.5)
    phys = lb + x * (ub - lb)
    # [Xc, Yc, Zc, L, h, Theta, Phi, Mc]
    np.testing.assert_allclose(phys[:5], [5.0, 5.0, 6.0, 12.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(phys[6], np.pi / 2, atol=1e-6)   # Phi elevation
    np.testing.assert_allclose(phys[7], 0.5, atol=1e-6)


def test_encode_rejects_size_mismatch():
    sk = Skeleton(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0, 1]]))
    lb = np.zeros(12)   # 2 components worth of bounds, but only 1 edge
    ub = np.ones(12)
    with pytest.raises(ValueError):
        encode_skeleton(sk, lb, ub, thickness=1.0, membership=1.0)


def test_fit_thickness_monotone_bisection():
    """Thickness bisection converges toward the target with a monotone proxy."""
    sk = Skeleton(np.array([[10.0, 5.0], [20.0, 5.0]]), np.array([[0, 1]]))
    lb = np.array([-1.0, -1.0, 0.0, 1.0, -2 * np.pi, 0.0])
    ub = np.array([61.0, 31.0, 67.0, 30.0, 2 * np.pi, 1.0])

    # proxy density that increases with thickness h (index 3 of the block)
    def evaluate(x):
        phys = lb + x * (ub - lb)
        return np.array([phys[3] / 30.0])

    h = fit_thickness_to_volfrac(
        sk, lb, ub, membership=1.0, evaluate=evaluate,
        target_volfrac=0.5, h_bounds=(1.0, 30.0),
    )
    assert 14.0 < h < 16.0   # target 0.5 -> h ~= 15


# ── Pattern resolution / pipeline glue (no FEniCS) ────────────────────────────

def _spec(mode="Free", lz=False, pattern=None):
    from ggp.problem.spec import (
        ProblemSpec, GeometrySpec, BoundaryCondition, Load,
        FormulationSpec, InitGuessSpec,
    )
    params = {"Lx": 60, "Ly": 30, "nx": 6, "ny": 3}
    gtype = "fenics_rectangle"
    if lz:
        params = {"Lx": 60, "Ly": 30, "Lz": 30, "nx": 6, "ny": 3, "nz": 3}
        gtype = "fenics_box"
    return ProblemSpec(
        geometries=[GeometrySpec(type=gtype, params=params)],
        boundary_conditions=[BoundaryCondition(region="left")],
        loads=[Load(region="mid_right")],
        formulation=FormulationSpec(mode=mode),
        init=InitGuessSpec(pattern=pattern),
    )


def test_default_pattern_is_dimension_aware():
    assert GGPPipeline(_spec(mode="Free"))._resolve_init_pattern(2) == ("quad_star", True)
    assert GGPPipeline(_spec(mode="3D_Free"))._resolve_init_pattern(3) == ("hex_star", True)


def test_grid_and_alm_are_not_mesh_patterns():
    assert GGPPipeline(_spec(pattern="grid"))._resolve_init_pattern(2) == ("grid", False)
    assert GGPPipeline(_spec(mode="ALM"))._resolve_init_pattern(2) == ("alm", False)


# ── Legacy grid regression ────────────────────────────────────────────────────

def test_make_init_grid_delegates_to_legacy():
    """The refactored _make_init must reproduce the legacy grid seed exactly."""
    n2 = 18 * 6  # 18 components, matching the 2-D presets
    lb2, ub2 = np.zeros(n2), np.full(n2, 60.0)
    kw = dict(Lx=60.0, Ly=30.0, lb=lb2, ub=ub2)
    np.testing.assert_array_equal(
        GGPPipeline._make_init("Free", n2, **kw), make_grid_init_2d(n2, **kw)
    )
    n3 = 20 * 8  # 20 components, matching the 3-D preset
    lb3, ub3 = np.zeros(n3), np.full(n3, 60.0)
    kw3 = dict(Lx=60.0, Ly=30.0, Lz=30.0, lb=lb3, ub=ub3)
    np.testing.assert_array_equal(
        GGPPipeline._make_init("3D_Free", n3, **kw3), make_grid_init_3d(n3, **kw3)
    )


def test_refined_density_decouples_render_from_fe_mesh():
    """refined_density re-evaluates the geometry on a finer grid (clean bars)."""
    from ggp.problem.spec import (
        ProblemSpec, GeometrySpec, BoundaryCondition, Load, FormulationSpec,
    )
    from ggp.visualization.render import refined_density

    spec = ProblemSpec(
        geometries=[GeometrySpec(type="fenics_rectangle",
                                 params={"Lx": 60, "Ly": 30, "nx": 6, "ny": 3})],
        boundary_conditions=[BoundaryCondition(region="left")],
        loads=[Load(region="mid_right")],
        formulation=FormulationSpec(mode="Free"),
    )
    x = np.full(4 * 6, 0.5)            # 4 components x 6 vars
    rho, coords = refined_density(spec, x, refine=2)
    assert coords.shape[0] == (6 * 2) * (3 * 2)   # refined nx*ny element centres
    assert rho.shape[0] == coords.shape[0]
    assert np.all((rho >= 0) & (rho <= 1))


def test_grid_init_2d_is_stable():
    """Snapshot a couple of invariants of the legacy 2-D grid seed."""
    lb, ub = np.zeros(36), np.full(36, 60.0)
    x = make_grid_init_2d(36, Lx=60.0, Ly=30.0, lb=lb, ub=ub)
    assert x.shape == (36,)
    assert (x >= 0).all() and (x <= 1).all()
    np.testing.assert_allclose(x[5::6], 0.5)   # membership Mc = 0.5
