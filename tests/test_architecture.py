# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the problem definition layer and projection registry."""
import pytest
import os
from pathlib import Path

from ggp.problem.spec import (
    GeometrySpec,
    BoundaryCondition,
    Load,
    FormulationSpec,
    ObjectiveSpec,
    ConstraintSpec,
    SolverSpec,
    MaterialSpec,
    ProblemSpec,
)
from ggp.problem.loader import load_problem


# ── ProblemSpec construction ──────────────────────────────────────────────────

def test_problem_spec_defaults():
    """ProblemSpec can be created with minimal arguments."""
    spec = ProblemSpec(
        geometries=[GeometrySpec(type="fenics_rectangle", params={"Lx": 60, "Ly": 30, "nx": 60, "ny": 30})],
        boundary_conditions=[BoundaryCondition(region="left")],
        loads=[Load(region="mid_right")],
        formulation=FormulationSpec(),
    )
    assert spec.volfrac == 0.4
    assert spec.material.E == 1.0
    assert spec.solver.algorithm == "MMA"
    assert spec.objective.name == "compliance"
    assert len(spec.constraints) == 1
    assert spec.constraints[0].name == "volume"


def test_geometry_spec_with_path():
    """GeometrySpec supports file-based geometry sources."""
    g = GeometrySpec(type="step", role="non_design", path=Path("bracket.step"))
    assert g.path == Path("bracket.step")
    assert g.role == "non_design"


def test_formulation_spec_alm():
    """FormulationSpec for ALM mode carries layer info."""
    f = FormulationSpec(mode="ALM", num_components=50, num_layers=10, comp_per_layer=5, layer_height=3.0)
    assert f.num_layers == 10
    assert f.comp_per_layer == 5


def test_solver_spec_options():
    """SolverSpec carries algorithm-specific options."""
    s = SolverSpec(algorithm="SLP", use_line_search=True, options={"scaling": 0.0})
    assert s.use_line_search is True
    assert s.options["scaling"] == 0.0


# ── YAML Loader ──────────────────────────────────────────────────────────────

def test_load_problem_from_dict():
    """load_problem accepts a raw dictionary."""
    raw = {
        "geometries": [{"type": "fenics_rectangle", "params": {"Lx": 60, "Ly": 30, "nx": 60, "ny": 30}}],
        "boundary_conditions": [{"region": "left", "type": "fixed"}],
        "loads": [{"region": "mid_right", "value": [0.0, -1.0]}],
        "formulation": {"mode": "Free", "num_components": 18},
        "solver": {"algorithm": "MMA", "max_iter": 100},
        "volfrac": 0.3,
    }
    spec = load_problem(raw)
    assert spec.volfrac == 0.3
    assert spec.solver.max_iter == 100
    assert spec.formulation.num_components == 18
    assert spec.geometries[0].type == "fenics_rectangle"


def test_load_problem_from_yaml_preset():
    """Load a built-in preset YAML."""
    presets_dir = Path(__file__).parent.parent / "ggp" / "cli" / "presets"
    yaml_path = presets_dir / "short_cantilever.yaml"
    if not yaml_path.exists():
        pytest.skip("Preset YAML not found (running from different cwd)")
    
    spec = load_problem(yaml_path)
    assert spec.formulation.mode == "Free"
    assert spec.formulation.num_components == 18
    assert spec.volfrac == 0.4
    assert any(bc.region == "left" for bc in spec.boundary_conditions)


def test_load_problem_defaults():
    """Omitted fields get sensible defaults."""
    raw = {
        "geometries": [{"type": "box", "params": {"Lx": 10}}],
    }
    spec = load_problem(raw)
    assert spec.solver.algorithm == "MMA"
    assert spec.material.nu == 0.3
    assert len(spec.constraints) == 1


# ── Projection Registry ──────────────────────────────────────────────────────

def test_registry_register_and_retrieve():
    """register_mapper / get_mapper round-trip works."""
    from ggp.projection.registry import register_mapper, get_mapper, list_mappers
    from ggp.projection.base import ProjectionMapper
    import numpy as np

    @register_mapper("_test_dummy")
    class DummyMapper(ProjectionMapper):
        def forward(self, x_vars, eval_coords, power_E=1.0, power_V=1.0):
            n = eval_coords.shape[0]
            return np.zeros(n), np.zeros(n)
        def jacobian(self, x_vars, eval_coords, power_E=1.0, power_V=1.0):
            return {"rho_E": np.zeros((1, 1)), "rho_V": np.zeros((1, 1))}
        def num_vars_per_component(self):
            return 2
        def default_bounds(self, domain_extents, num_components):
            n = num_components * 2
            return np.zeros(n), np.ones(n)

    assert "_test_dummy" in list_mappers()
    mapper = get_mapper("_test_dummy")
    assert isinstance(mapper, DummyMapper)
    assert mapper.num_vars_per_component() == 2


def test_registry_unknown_raises():
    """get_mapper raises for unregistered modes."""
    from ggp.projection.registry import get_mapper
    with pytest.raises(ValueError, match="No mapper registered"):
        get_mapper("__nonexistent__")
