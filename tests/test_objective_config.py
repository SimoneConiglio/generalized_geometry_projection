# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Tests for the configurable objective: pick any defined response (compliance,
volume, displacement, stress) as the objective, minimize or maximize, mirroring how
any of them can already be a constraint.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ggp.problem.spec import ObjectiveSpec, ConstraintSpec, KNOWN_RESPONSES
from ggp.problem.loader import load_problem, _build_objective


# --------------------------------------------------------------------------- #
# Spec / loader — FEniCS-free
# --------------------------------------------------------------------------- #
def test_objective_spec_defaults():
    obj = ObjectiveSpec()
    assert obj.name == "compliance"
    assert obj.sense == "minimize"
    assert obj.params == {}


def test_build_objective_defaults_when_absent():
    obj = _build_objective({})
    assert obj == ObjectiveSpec(name="compliance", sense="minimize", params={})


def test_build_objective_roundtrips_sense_and_params():
    obj = _build_objective({"name": "stress", "sense": "maximize",
                            "params": {"P": 20.0, "aggregation": "verbart"}})
    assert obj.name == "stress"
    assert obj.sense == "maximize"
    assert obj.params == {"P": 20.0, "aggregation": "verbart"}


def test_build_objective_rejects_unknown_name():
    with pytest.raises(ValueError):
        _build_objective({"name": "not_a_response"})


def test_build_objective_rejects_unknown_sense():
    with pytest.raises(ValueError):
        _build_objective({"name": "compliance", "sense": "sideways"})


def test_known_responses_matches_constraint_and_objective_vocabulary():
    # every response nameable as a constraint should be nameable as an objective
    assert set(KNOWN_RESPONSES) == {"compliance", "volume", "displacement", "stress"}


def test_loader_default_objective_is_unchanged():
    spec = load_problem("ggp/cli/presets/short_cantilever.yaml")
    assert spec.objective == ObjectiveSpec(name="compliance", sense="minimize", params={})


def test_loader_reads_minvol_stress_preset():
    spec = load_problem("ggp/cli/presets/short_cantilever_minvol_stress.yaml")
    assert spec.objective.name == "volume"
    assert spec.objective.sense == "minimize"
    assert any(c.name == "stress" for c in spec.constraints)
    assert not any(c.name == "volume" for c in spec.constraints)


# --------------------------------------------------------------------------- #
# Pipeline-level — needs FEniCS/GEMSEO, tiny mesh
# --------------------------------------------------------------------------- #
pytest.importorskip("dolfin")
pytest.importorskip("gemseo")
pytest.importorskip("jax")

from ggp.optimization.pipeline import GGPPipeline  # noqa: E402


def _tiny_spec(objective=None, constraints=None, max_iter=6, num_components=6):
    spec = load_problem("ggp/cli/presets/short_cantilever_stress.yaml")
    geom = spec.geometries[0]
    params = dict(geom.params)
    params.update(Lx=12.0, Ly=6.0, nx=12, ny=6)
    geom = dataclasses.replace(geom, params=params)
    formulation = dataclasses.replace(spec.formulation, num_components=num_components)
    solver = dataclasses.replace(spec.solver, max_iter=max_iter, fem_solver="direct")
    kwargs = dict(geometries=[geom], formulation=formulation, solver=solver)
    if objective is not None:
        kwargs["objective"] = objective
    if constraints is not None:
        kwargs["constraints"] = constraints
    return dataclasses.replace(spec, **kwargs)


def test_default_objective_still_reports_compliance():
    spec = _tiny_spec()
    result = GGPPipeline(spec).run()
    assert result.objective_name == "compliance"
    assert np.isfinite(result.objective_value)


def test_minvol_stress_objective_runs_and_reports_volume():
    spec = _tiny_spec(
        objective=ObjectiveSpec(name="volume", sense="minimize"),
        constraints=[ConstraintSpec(name="stress", type="ineq", bound=0.0,
                                    params={"sigma_limit": 40.0, "P": 8.0})],
    )
    result = GGPPipeline(spec).run()
    assert result.objective_name == "volume"
    assert np.isfinite(result.objective_value)
    # volume response is (v - volfrac)/volfrac*100 -> should have decreased from the
    # volfrac-filled start (i.e. gone negative) as the optimizer removes material
    # under a stress bound instead of a fixed volume target.
    assert result.objective_value < 0.0


def test_minvol_stress_uses_less_volume_than_fixed_volfrac_compliance_run():
    # Minimizing volume under a stress bound should not need more material than the
    # fixed-volfrac compliance-objective baseline at a comparable stress bound.
    stress_params = {"sigma_limit": 40.0, "P": 8.0}
    base_spec = _tiny_spec(
        objective=ObjectiveSpec(name="compliance"),
        constraints=[ConstraintSpec(name="volume", type="ineq", bound=0.0),
                     ConstraintSpec(name="stress", type="ineq", bound=0.0,
                                    params=stress_params)],
    )
    base = GGPPipeline(base_spec).run()
    base_vol = float(np.mean(base.density_field))

    minvol_spec = _tiny_spec(
        objective=ObjectiveSpec(name="volume", sense="minimize"),
        constraints=[ConstraintSpec(name="stress", type="ineq", bound=0.0,
                                    params=stress_params)],
    )
    minvol = GGPPipeline(minvol_spec).run()
    minvol_vol = float(np.mean(minvol.density_field))

    assert minvol_vol <= base_vol + 0.05     # small slack: short run, not fully converged


def test_maximize_sense_increases_objective_relative_to_minimize():
    # Same tiny problem, opposite sense: maximizing volume should end up with MORE
    # material than minimizing it, everything else equal.
    stress_params = {"sigma_limit": 40.0, "P": 8.0}
    constraints = [ConstraintSpec(name="stress", type="ineq", bound=0.0,
                                  params=stress_params)]

    min_spec = _tiny_spec(objective=ObjectiveSpec(name="volume", sense="minimize"),
                          constraints=constraints)
    max_spec = _tiny_spec(objective=ObjectiveSpec(name="volume", sense="maximize"),
                          constraints=constraints)

    r_min = GGPPipeline(min_spec).run()
    r_max = GGPPipeline(max_spec).run()

    assert np.mean(r_max.density_field) > np.mean(r_min.density_field)


def test_stress_as_lone_objective_builds_the_discipline():
    # stress named only as the objective (no matching ConstraintSpec) must still
    # build _StressConstraintDiscipline, sourcing params from objective.params.
    spec = _tiny_spec(
        objective=ObjectiveSpec(name="stress", sense="minimize",
                                params={"sigma_limit": 40.0, "P": 8.0}),
        constraints=[ConstraintSpec(name="volume", type="ineq", bound=0.0)],
    )
    result = GGPPipeline(spec).run()
    assert result.objective_name == "stress"
    assert np.isfinite(result.objective_value)


def test_deflation_with_non_compliance_objective_raises():
    spec = _tiny_spec(objective=ObjectiveSpec(name="volume", sense="minimize"),
                      constraints=[ConstraintSpec(name="stress", type="ineq", bound=0.0,
                                                  params={"sigma_limit": 40.0})])
    root = np.full(6 * 6, 0.5)
    with pytest.raises(ValueError):
        GGPPipeline(spec, deflation={"roots": [root]}).run()
