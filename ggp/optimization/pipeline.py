# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Optimization pipeline orchestration."""
from __future__ import annotations

import time
import numpy as np

from ggp.problem.spec import ProblemSpec
from ggp.geometry.io.registry import get_reader
from ggp.discretisation.fem import FEMDiscretiser
from ggp.gemseo_wrappers.geometry_discipline import GGPGeometryDiscipline
from ggp.gemseo_wrappers.physics_discipline import GGPPhysicsDiscipline
from .results import OptimisationResult

from gemseo import create_scenario, create_design_space
from gemseo.mda.mda_chain import MDAChain


class GGPPipeline:
    """Orchestrates the entire GGP optimization process."""
    
    def __init__(self, spec: ProblemSpec):
        self.spec = spec
        
    def run(self) -> OptimisationResult:
        start_time = time.time()
        
        # 1. Geometry I/O
        domain_geom = self.spec.geometries[0]
        reader = get_reader(domain_geom.type)
        domain = reader.read(domain_geom)
        
        # 2. FEM Discretisation
        discretiser = FEMDiscretiser()
        analysis = discretiser.discretise(domain, self.spec)
        
        mesh_area = domain.metadata.get("mesh_area", 1.0)
        
        # 3. Create Disciplines
        geom_kwargs = {
            "num_layers": self.spec.formulation.num_layers,
            "comp_per_layer": self.spec.formulation.comp_per_layer,
            "layer_height": self.spec.formulation.layer_height,
            "ka": 10.0,
            "pp": 10.0,
            "method": self.spec.formulation.method,
        }
        
        geom_discipline = GGPGeometryDiscipline(
            mesh=analysis.mesh,
            num_components=self.spec.formulation.num_components,
            mode=self.spec.formulation.mode,
            **{k: v for k, v in geom_kwargs.items() if v is not None}
        )
        
        # Extract fixed dofs
        fixed_dofs = []
        for bc in analysis.bcs_applied:
            fixed_dofs.extend(bc.get_boundary_values().keys())
        fixed_dofs = sorted(list(set(fixed_dofs)))
        
        phys_discipline = GGPPhysicsDiscipline(
            V_u=analysis.function_spaces["u"],
            ke_ref=analysis.ke_ref,
            fixed_dofs=fixed_dofs,
            f_vec=analysis.load_vector,
            mesh_area=mesh_area,
            volfrac=self.spec.volfrac,
            iterative=self.spec.solver.iterative,
            p_penalty=3.0,
            Emin=1e-6,
            E0=1.0
        )
        
        # 4. Design Space
        design_space = create_design_space()
        num_vars = geom_discipline.mapper.num_vars_per_component() * self.spec.formulation.num_components
        
        if self.spec.formulation.mode in ["ALM", "2D_ALM"] and geom_kwargs.get("num_layers"):
            num_vars = 3 * geom_kwargs["comp_per_layer"] * geom_kwargs["num_layers"]
        elif self.spec.formulation.mode == "3D_ALM" and geom_kwargs.get("num_layers"):
            num_vars = 6 * geom_kwargs["comp_per_layer"] * geom_kwargs["num_layers"]
            
        x_init = np.random.rand(num_vars) * 0.1 + 0.45
        design_space.add_variable("x_vars", size=num_vars, lower_bound=0.0, upper_bound=1.0, value=x_init)
        
        # 5. MDA & Scenario
        chain = MDAChain([geom_discipline, phys_discipline])
        if hasattr(chain, 'cache'): chain.cache = None
        if hasattr(chain, 'cache_type'): chain.cache_type = chain.CacheType.NONE
        
        scenario = create_scenario(
            [chain], 
            objective_name="compliance", 
            design_space=design_space, 
            maximize_objective=False, 
            formulation_name="MDF"
        )
        scenario.add_constraint("volume", constraint_type="ineq", positive=False, value=0.0)
        
        algo_options = self.spec.solver.options.copy()
        if "max_iter" not in algo_options:
            algo_options["max_iter"] = self.spec.solver.max_iter
        if "algo_name" not in algo_options:
            algo_options["algo_name"] = self.spec.solver.algorithm
            
        scenario.execute(**algo_options)
        
        opt_problem = scenario.optimization_result
        
        result = OptimisationResult(
            problem_name="optimization_run",
            algorithm=self.spec.solver.algorithm,
            status=opt_problem.status,
            iterations=len(scenario.optimization_history.get("objective", [])),
            objective_value=opt_problem.f_opt,
            max_constraint_violation=0.0,
            design_variables=opt_problem.x_opt,
            history={
                "objective": scenario.optimization_history.get("objective", []),
            },
            execution_time_s=time.time() - start_time
        )
        
        return result
