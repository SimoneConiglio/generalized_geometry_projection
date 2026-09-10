# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the cadjoint geometry reader.

The pure-array helpers (corner-order permutation, facet naming) run everywhere.
The meshing tests need cadjoint installed and are skipped otherwise; none of
them need FEniCS.
"""
import numpy as np
import pytest

from ggp.geometry.io import get_reader, list_readers
from ggp.geometry.io.base import DomainRepresentation
from ggp.geometry.io.cadjoint_reader import (
    _axis_facets,
    hex_cells_to_dolfin_order,
)
from ggp.problem.spec import GeometrySpec

cadjoint = pytest.importorskip("cadjoint", reason="cadjoint is an optional geometry backend")


# ── Registry ──────────────────────────────────────────────────────────────────

def test_cadjoint_reader_is_registered():
    """The reader self-registers under the 'cadjoint' geometry type."""
    assert "cadjoint" in list_readers()
    assert get_reader("cadjoint").supports("cadjoint")
    assert not get_reader("cadjoint").supports("fenics_box")


# ── Corner ordering (no cadjoint, no FEniCS needed) ───────────────────────────

def test_vtk_to_ufc_permutation_is_lexicographic():
    """The permutation maps VTK corner order onto UFC's lexicographic order."""
    vtk_offsets = np.array(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    )
    # One cell whose "global indices" are the VTK slot numbers.
    permuted = hex_cells_to_dolfin_order(np.arange(8)[None, :])[0]
    reordered = vtk_offsets[permuted]
    lexicographic = np.array(sorted(map(tuple, vtk_offsets)))
    np.testing.assert_array_equal(reordered, lexicographic)


def test_hex_cells_to_dolfin_order_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"shape \(C, 8\)"):
        hex_cells_to_dolfin_order(np.zeros((3, 4)))


def test_axis_facets_names_all_six_faces():
    """A unit cube's corners are split into the six named axis extremes."""
    corners = np.array(np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], indexing="ij"))
    vertices = corners.reshape(3, -1).T
    facets = _axis_facets(vertices, tol=1e-12)
    assert set(facets) == {"left", "right", "bottom", "top", "front", "back"}
    assert all(len(nodes) == 4 for nodes in facets.values())
    assert set(facets["left"]).isdisjoint(facets["right"])


# ── Meshing ───────────────────────────────────────────────────────────────────

def _l_bracket_spec(**overrides):
    """An L-shaped bracket: a 60x60x10 slab with its top-right quadrant removed."""
    import jax.numpy as jnp

    def sdf(p):
        p = jnp.asarray(p)

        def box(pt, lo, hi):
            lo, hi = jnp.asarray(lo), jnp.asarray(hi)
            centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
            q = jnp.abs(pt - centre) - half
            return (jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
                    + jnp.minimum(jnp.max(q, axis=-1), 0.0))

        return jnp.maximum(box(p, [0, 0, 0], [60, 60, 10]),
                           -box(p, [30, 30, -1], [61, 61, 11]))

    params = {"sdf": sdf, "Lx": 60.0, "Ly": 60.0, "Lz": 10.0,
              "nx": 24, "ny": 24, "nz": 4}
    params.update(overrides)
    return GeometrySpec(type="cadjoint", params=params)


@pytest.fixture(scope="module")
def l_bracket():
    return get_reader("cadjoint").read(_l_bracket_spec())


def test_reader_returns_domain_representation(l_bracket):
    assert isinstance(l_bracket, DomainRepresentation)
    assert l_bracket.dim == 3
    assert l_bracket.cell_type == "hex"
    assert l_bracket.cells.shape[1] == 8
    assert l_bracket.vertices.shape[1] == 3


def test_domain_is_trimmed_to_the_cad_shape(l_bracket):
    """An L-shape covers three quadrants, so a quarter of the lattice is dropped."""
    meta = l_bracket.metadata
    assert meta["cells_in_lattice"] == 24 * 24 * 4
    assert meta["cells_kept"] == pytest.approx(0.75 * meta["cells_in_lattice"], rel=0.02)
    assert meta["cells_kept"] == l_bracket.cells.shape[0]


def test_no_element_centroid_lies_in_the_removed_quadrant(l_bracket):
    """The trim is geometric, not incidental: the notch is genuinely empty."""
    centroids = l_bracket.vertices[l_bracket.cells].mean(axis=1)
    in_notch = (centroids[:, 0] > 30.0) & (centroids[:, 1] > 30.0)
    assert not in_notch.any()


def test_unsnapped_cells_are_congruent(l_bracket):
    """FEMDiscretiser reuses one ke_ref for every element — that needs congruence."""
    assert l_bracket.metadata["uniform_cells"] is True
    corners = l_bracket.vertices[l_bracket.cells]
    offsets = corners - corners[:, :1, :]
    assert np.abs(offsets - offsets[:1]).max() < 1e-12


def test_cells_are_dolfin_ready(l_bracket):
    """DOLFIN requires ascending global indices within each cell."""
    ufc = hex_cells_to_dolfin_order(l_bracket.cells)
    assert np.all(np.diff(ufc, axis=1) > 0)
    assert ufc.max() < l_bracket.vertices.shape[0]


def test_metadata_carries_domain_extents(l_bracket):
    """FEMDiscretiser reads Lx/Ly/Lz from metadata to place BCs and loads."""
    meta = l_bracket.metadata
    assert (meta["Lx"], meta["Ly"], meta["Lz"]) == (60.0, 60.0, 10.0)
    assert meta["sdf"] is not None
    assert meta["snapped"] is False


def test_facets_track_the_bounding_box(l_bracket):
    facets = l_bracket.facets
    assert set(facets) == {"left", "right", "bottom", "top", "front", "back"}
    left = l_bracket.vertices[facets["left"]]
    np.testing.assert_allclose(left[:, 0], 0.0, atol=1e-9)


def test_accepts_a_cadjoint_sdf_object():
    """A cadjoint SDF node broadcasts over (..., 3) and needs no vmap wrapper."""
    from cadjoint.sdf import Box

    domain = get_reader("cadjoint").read(
        GeometrySpec(
            type="cadjoint",
            params={"sdf": Box(size=[1.0, 1.0, 1.0]), "origin": (-1.0, -1.0, -1.0),
                    "Lx": 2.0, "Ly": 2.0, "Lz": 2.0, "nx": 8, "ny": 8, "nz": 8},
        )
    )
    # Box(size=...) is a half-extent box of side 1 centred at the origin, so it
    # fills an eighth of the 2x2x2 sampling volume.
    assert domain.cells.shape[0] > 0
    centroids = domain.vertices[domain.cells].mean(axis=1)
    assert np.abs(centroids).max() <= 1.0


def test_accepts_a_per_point_functionalized_sdf():
    """A functionalize()d SDF maps one (3,) point at a time; the reader vmaps it."""
    import jax.numpy as jnp

    def per_point(p):
        # Sphere of radius 0.6 — scalar in, scalar out, no batching.
        return jnp.linalg.norm(p) - 0.6

    domain = get_reader("cadjoint").read(
        GeometrySpec(
            type="cadjoint",
            params={"sdf": per_point, "origin": (-1.0, -1.0, -1.0),
                    "Lx": 2.0, "Ly": 2.0, "Lz": 2.0, "nx": 10, "ny": 10, "nz": 10},
        )
    )
    centroids = domain.vertices[domain.cells].mean(axis=1)
    assert np.linalg.norm(centroids, axis=1).max() < 0.75


def test_missing_geometry_source_is_reported():
    with pytest.raises(ValueError, match="params\\['sdf'\\] or a path"):
        get_reader("cadjoint").read(GeometrySpec(type="cadjoint", params={}))


def test_scene_file_without_a_known_binding_is_reported(tmp_path):
    scene = tmp_path / "scene.py"
    scene.write_text("something_else = 1\n")
    with pytest.raises(ValueError, match="defines none of"):
        get_reader("cadjoint").read(GeometrySpec(type="cadjoint", path=scene))


def test_reads_an_sdf_from_a_scene_file(tmp_path):
    scene = tmp_path / "scene.py"
    scene.write_text(
        "from cadjoint.sdf import Box\n"
        "sdf = Box(size=[1.0, 1.0, 1.0])\n"
    )
    domain = get_reader("cadjoint").read(
        GeometrySpec(
            type="cadjoint",
            path=scene,
            params={"origin": (-1.0, -1.0, -1.0), "Lx": 2.0, "Ly": 2.0, "Lz": 2.0,
                    "nx": 8, "ny": 8, "nz": 8},
        )
    )
    assert domain.cells.shape[0] > 0


# ── SDF calling-convention adapter ────────────────────────────────────────────

def test_adapter_rejects_a_non_callable_source():
    from ggp.geometry.io.cadjoint_reader import _as_batched_sdf

    with pytest.raises(TypeError, match="must be an SDF or a callable"):
        _as_batched_sdf(object())


def test_adapter_falls_back_to_vmap_when_a_batched_call_raises():
    """A source that only accepts one point is vmapped, and the choice is cached."""
    import jax.numpy as jnp
    from ggp.geometry.io.cadjoint_reader import _as_batched_sdf

    calls = {"batched_attempts": 0}

    def single_point_only(p):
        if jnp.ndim(p) != 1:
            calls["batched_attempts"] += 1
            raise ValueError("this source takes one point at a time")
        return jnp.linalg.norm(p) - 1.0

    evaluate = _as_batched_sdf(single_point_only)
    points = np.zeros((5, 3))
    np.testing.assert_allclose(np.asarray(evaluate(points)), -1.0)
    np.testing.assert_allclose(np.asarray(evaluate(points)), -1.0)
    # The batched form is attempted once, then remembered as unsupported.
    assert calls["batched_attempts"] == 1


def test_adapter_reuses_a_broadcasting_source_without_retrying():
    """A source that broadcasts is called directly on every batched call."""
    import jax.numpy as jnp
    from ggp.geometry.io.cadjoint_reader import _as_batched_sdf

    def broadcasting(p):
        return jnp.linalg.norm(jnp.asarray(p), axis=-1) - 1.0

    evaluate = _as_batched_sdf(broadcasting)
    points = np.zeros((4, 3))
    for _ in range(3):
        np.testing.assert_allclose(np.asarray(evaluate(points)), -1.0)
    # Single points keep working through the same adapter.
    assert float(evaluate(np.zeros(3))) == pytest.approx(-1.0)
