# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Integration tests for the global-search strategies on the *real* GGP pipeline.

These exercise GGPPipeline.run() end to end (FEM discretisation, geometry +
physics disciplines, GEMSEO MDF scenario) at a tiny mesh / 1 iteration, covering
the new x0 / overrides / deflation injection paths and the real (non-mocked)
global_search code path. They require dolfin + gemseo and are skipped otherwise.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("dolfin")
pytest.importorskip("gemseo")

from ggp.problem.loader import load_problem
from ggp.optimization.pipeline import GGPPipeline
from ggp.optimization import global_search as gs

PRESET = "ggp/cli/presets/short_cantilever.yaml"


def _tiny_spec(max_iter=1, num_components=4):
    """Short-cantilever spec on a coarse mesh with a 1-iteration direct solve."""
    spec = load_problem(PRESET)
    geom = spec.geometries[0]
    params = dict(geom.params)
    params.update(Lx=12.0, Ly=6.0, nx=12, ny=6)
    geom = dataclasses.replace(geom, params=params)
    formulation = dataclasses.replace(spec.formulation, num_components=num_components)
    solver = dataclasses.replace(spec.solver, max_iter=max_iter, fem_solver="direct",
                                 options={})
    return dataclasses.replace(spec, geometries=[geom], formulation=formulation,
                               solver=solver)


@pytest.fixture(scope="module")
def tiny_spec():
    return _tiny_spec()


def test_pipeline_baseline_runs(tiny_spec):
    result = GGPPipeline(tiny_spec).run()
    assert np.isfinite(result.objective_value)
    assert result.design_variables.shape[0] == 6 * 4
    assert result.density_field is not None
    assert gs.compliance_of(result) > 0.0


def test_pipeline_x0_and_overrides(tiny_spec):
    n = 6 * 4
    x0 = np.full(n, 0.5)
    result = GGPPipeline(
        tiny_spec, x0=x0,
        overrides={"ka": 8.0, "pp": 50.0, "r_gp": 1.0, "gammac": 2.0,
                   "gammav": 1.0, "p_penalty": 1.0},
    ).run()
    assert np.isfinite(result.objective_value)


def test_pipeline_x0_wrong_size_raises(tiny_spec):
    with pytest.raises(ValueError):
        GGPPipeline(tiny_spec, x0=np.full(7, 0.5)).run()


def test_pipeline_deflation_reports_true_compliance(tiny_spec):
    base = GGPPipeline(tiny_spec).run()
    deflated = GGPPipeline(
        tiny_spec,
        deflation={"roots": [base.design_variables], "shift": 1.0, "power": 2.0},
    ).run()
    # objective_value must remain the raw log(C+1) even though the scenario
    # optimised the deflated objective.
    assert np.isfinite(deflated.objective_value)
    assert gs.compliance_of(deflated) > 0.0


def test_global_search_multi_start_real(tiny_spec):
    out = gs.multi_start(tiny_spec, n_starts=1, seed=0)
    assert len(out.attempts) == 2
    assert out.best_compliance == pytest.approx(
        min(a.compliance for a in out.attempts))


def test_global_search_continuation_real(tiny_spec):
    out = gs.continuation(tiny_spec, [{"r_gp": 1.5}, {"r_gp": 0.5}])
    assert len(out.attempts) == 2
    # reported best is the final phase
    assert out.best_compliance == pytest.approx(out.attempts[-1].compliance)
