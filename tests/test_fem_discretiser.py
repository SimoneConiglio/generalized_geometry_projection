# Copyright (c) 2026 Simone Coniglio
"""Tests for FEMDiscretiser and AnalysisDomain."""
import numpy as np
import pytest
import dolfin as df

from ggp.geometry.io.fenics_reader import FenicsRectangleReader
from ggp.discretisation.fem import FEMDiscretiser, AnalysisDomain
from ggp.problem.spec import (
    BoundaryCondition, ConstraintSpec, FormulationSpec,
    GeometrySpec, Load, ProblemSpec, SolverSpec,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _rect_spec(bc_kwargs, load_region="mid_right", Lx=2.0, Ly=2.0, nx=2, ny=2):
    """Build a minimal 2-D ProblemSpec with a rectangle domain."""
    return ProblemSpec(
        geometries=[GeometrySpec(
            type="fenics_rectangle",
            params={"Lx": Lx, "Ly": Ly, "nx": nx, "ny": ny},
        )],
        boundary_conditions=[BoundaryCondition(**bc_kwargs)],
        loads=[Load(region=load_region, type="point", value=[0.0, -1.0])],
        formulation=FormulationSpec(mode="Free", num_components=2),
    )


def _discretise(spec):
    reader = FenicsRectangleReader()
    domain = reader.read(spec.geometries[0])
    analysis = FEMDiscretiser().discretise(domain, spec)
    return analysis, domain


# ── AnalysisDomain ────────────────────────────────────────────────────────────

class TestAnalysisDomain:
    def test_construction(self):
        ec = np.zeros((10, 2))
        ad = AnalysisDomain(dim=2, eval_coords=ec)
        assert ad.dim == 2
        assert ad.num_eval_points == 10

    def test_metadata_default_empty(self):
        ec = np.zeros((5, 2))
        ad = AnalysisDomain(dim=2, eval_coords=ec)
        assert ad.metadata == {}


# ── FEMDiscretiser — fixed left BC ───────────────────────────────────────────

class TestFEMDiscretiserFixedLeft:
    # 2×2 mesh: nodes at x∈{0,1,2}, y∈{0,1,2}
    # mid_right target [2.0, 1.0] is a mesh node

    def setup_method(self):
        spec = _rect_spec({"region": "left", "type": "fixed"})
        self.analysis, self.domain = _discretise(spec)

    def test_returns_analysis_domain(self):
        assert isinstance(self.analysis, AnalysisDomain)

    def test_mesh_present(self):
        assert self.analysis.mesh is not None

    def test_function_spaces(self):
        assert "u" in self.analysis.function_spaces
        assert "dg" in self.analysis.function_spaces

    def test_eval_coords_shape(self):
        # 2×2 structured quad mesh → 4 elements
        assert self.analysis.eval_coords.shape == (4, 2)

    def test_bcs_applied(self):
        assert len(self.analysis.bcs_applied) >= 1
        assert all(isinstance(b, df.DirichletBC) for b in self.analysis.bcs_applied)

    def test_load_vector_nonzero(self):
        assert self.analysis.load_vector is not None
        assert np.any(self.analysis.load_vector != 0.0)

    def test_ke_ref_square_symmetric(self):
        ke = np.array(self.analysis.ke_ref)
        assert ke.ndim == 2
        assert ke.shape[0] == ke.shape[1]
        # Reference stiffness must be symmetric
        np.testing.assert_allclose(ke, ke.T, atol=1e-10)


# ── FEMDiscretiser — symmetry BC ─────────────────────────────────────────────

class TestFEMDiscretiserSymmetryBC:
    def test_symmetry_left_x_component(self):
        spec = _rect_spec({"region": "left", "type": "symmetry", "components": [0]})
        analysis, _ = _discretise(spec)
        assert len(analysis.bcs_applied) >= 1

    def test_top_bc(self):
        spec = ProblemSpec(
            geometries=[GeometrySpec(
                type="fenics_rectangle",
                params={"Lx": 2.0, "Ly": 2.0, "nx": 2, "ny": 2},
            )],
            boundary_conditions=[
                BoundaryCondition(region="left", type="fixed"),
                BoundaryCondition(region="top", type="fixed"),
            ],
            loads=[Load(region="mid_right", type="point", value=[0.0, -1.0])],
            formulation=FormulationSpec(),
        )
        analysis, _ = _discretise(spec)
        assert len(analysis.bcs_applied) >= 2

    def test_bottom_right_corner_bc(self):
        spec = ProblemSpec(
            geometries=[GeometrySpec(
                type="fenics_rectangle",
                params={"Lx": 2.0, "Ly": 2.0, "nx": 2, "ny": 2},
            )],
            boundary_conditions=[
                BoundaryCondition(region="bottom_right_corner", type="fixed", components=[1]),
            ],
            loads=[Load(region="mid_right", type="point", value=[0.0, -1.0])],
            formulation=FormulationSpec(),
        )
        analysis, _ = _discretise(spec)
        # Corner point BCs are stored in point_fixed_dofs, not bcs_applied
        assert len(analysis.point_fixed_dofs) >= 1


# ── FEMDiscretiser — load regions ─────────────────────────────────────────────

class TestFEMDiscretiserLoads:
    def test_mid_right_load_applied(self):
        spec = _rect_spec({"region": "left", "type": "fixed"}, load_region="mid_right")
        analysis, _ = _discretise(spec)
        assert np.any(analysis.load_vector != 0.0)

    def test_top_left_corner_load(self):
        # top_left_corner target = [0.0, Ly]; always a node on a rectangle mesh
        spec = ProblemSpec(
            geometries=[GeometrySpec(
                type="fenics_rectangle",
                params={"Lx": 2.0, "Ly": 2.0, "nx": 2, "ny": 2},
            )],
            boundary_conditions=[BoundaryCondition(region="left", type="fixed")],
            loads=[Load(region="top_left_corner", type="point", value=[0.0, -1.0])],
            formulation=FormulationSpec(),
        )
        analysis, _ = _discretise(spec)
        assert np.any(analysis.load_vector != 0.0)

    def test_no_load_vector_is_zero_everywhere(self):
        """With no recognised load region, load vector remains zero."""
        spec = ProblemSpec(
            geometries=[GeometrySpec(
                type="fenics_rectangle",
                params={"Lx": 2.0, "Ly": 2.0, "nx": 2, "ny": 2},
            )],
            boundary_conditions=[BoundaryCondition(region="left", type="fixed")],
            loads=[Load(region="unknown_region", type="point", value=[0.0, -1.0])],
            formulation=FormulationSpec(),
        )
        analysis, _ = _discretise(spec)
        np.testing.assert_allclose(analysis.load_vector, 0.0, atol=1e-15)

    def test_mesh_area_in_metadata(self):
        spec = _rect_spec({"region": "left", "type": "fixed"})
        analysis, domain = _discretise(spec)
        # metadata comes from the geometry reader
        assert "Lx" in domain.metadata
        assert "Ly" in domain.metadata


# ── Region resolution and unconstrained-model detection ──────────────────────

class TestBoundaryConditionValidation:
    """A BC that constrains nothing leaves the model singular.

    DOLFIN only warns about this on stderr ("Found no facets matching domain for
    boundary condition") and then happily assembles a singular stiffness matrix.
    The discretiser turns it into an error instead — the same guard that catches
    a region a trimmed CAD domain never reaches.
    """

    def test_unreachable_region_raises(self):
        spec = _rect_spec({"region": "nowhere", "type": "fixed"})
        with pytest.raises(ValueError, match="constrained no degrees of freedom"):
            _discretise(spec)

    def test_error_names_the_region_and_the_mesh_extent(self):
        spec = _rect_spec({"region": "nowhere", "type": "fixed"}, Lx=3.0, Ly=5.0)
        with pytest.raises(ValueError) as excinfo:
            _discretise(spec)
        message = str(excinfo.value)
        assert "'nowhere'" in message
        assert "x∈[0, 3]" in message and "y∈[0, 5]" in message

    def test_a_symmetry_bc_that_matches_is_accepted(self):
        """The guard counts symmetry constraints too, not just full fixings."""
        spec = _rect_spec({"region": "left", "type": "symmetry", "components": [0]})
        analysis, _ = _discretise(spec)
        assert sum(len(bc.get_boundary_values()) for bc in analysis.bcs_applied) > 0


class TestRegionsFollowTheMeshBounds:
    """Named regions resolve against the mesh's bounding box, not metadata.

    For the built-in readers the two coincide, so these are regression tests:
    the switch to mesh-derived bounds must not move any existing region.
    """

    def test_left_selects_the_minimum_x_face(self):
        spec = _rect_spec({"region": "left", "type": "fixed"}, Lx=4.0, Ly=2.0, nx=4, ny=2)
        analysis, _ = _discretise(spec)
        V_u = analysis.function_spaces["u"]
        coords = V_u.tabulate_dof_coordinates()
        fixed = list(analysis.bcs_applied[0].get_boundary_values().keys())
        np.testing.assert_allclose(coords[fixed][:, 0], 0.0, atol=1e-12)

    def test_top_selects_the_maximum_y_face(self):
        spec = _rect_spec({"region": "top", "type": "fixed"}, Lx=4.0, Ly=2.0, nx=4, ny=2)
        analysis, _ = _discretise(spec)
        coords = analysis.function_spaces["u"].tabulate_dof_coordinates()
        fixed = list(analysis.bcs_applied[0].get_boundary_values().keys())
        np.testing.assert_allclose(coords[fixed][:, 1], 2.0, atol=1e-12)

    def test_mid_right_load_lands_on_the_maximum_x_face(self):
        spec = _rect_spec({"region": "left", "type": "fixed"}, Lx=4.0, Ly=2.0, nx=4, ny=2)
        analysis, _ = _discretise(spec)
        coords = analysis.function_spaces["u"].tabulate_dof_coordinates()
        loaded = np.flatnonzero(analysis.load_vector)
        assert loaded.size > 0
        np.testing.assert_allclose(coords[loaded][:, 0], 4.0, atol=1e-12)
        np.testing.assert_allclose(coords[loaded][:, 1], 1.0, atol=1e-12)
