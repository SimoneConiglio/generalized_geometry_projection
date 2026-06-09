import pytest
from dolfin import *
from dolfin_adjoint import *
import numpy as np
from samo_ggp.geometry.factory import GeometryFactory
from samo_ggp.physics.factory import PhysicsFactory
from samo_ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline, GGPPhysicsAdjointDiscipline
import gemseo
from gemseo import create_scenario
from gemseo.mda.mda_chain import MDAChain

def test_hybrid_pipeline_execution():
    """Verify the new Hybrid Modular Architecture with Log Scaling."""
    mesh = UnitSquareMesh(10, 10)
    V_u = VectorFunctionSpace(mesh, "CG", 1)
    def left_boundary(x, on_boundary): return on_boundary and near(x[0], 0.0)
    bc = [DirichletBC(V_u, Constant((0.0, 0.0)), left_boundary)]
    
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary): return near(x[0], 1.0)
    RightBoundary().mark(boundaries, 1)
    ds_load = Measure("ds", domain=mesh, subdomain_data=boundaries)
    L_rhs_vec = Constant((0.0, -1.0))
    
    solver = PhysicsFactory.create_solver("Elasticity_2D", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec)
    
    num_comp = 1
    geom_disc = GGPVectorizedGeometryDiscipline(mesh, num_comp, mode='Free')
    phys_disc = GGPPhysicsAdjointDiscipline(solver, mesh, mesh_area=1.0, volfrac=0.4)
    chain = MDAChain([geom_disc, phys_disc])
    
    x_init = np.array([0.5, 0.5, 0.2, 0.2, 0.0, 1.0])
    ds = gemseo.algos.design_space.DesignSpace()
    ds.add_variable("x_vars", size=6, lower_bound=np.zeros(6), upper_bound=np.ones(6), value=x_init)
    
    scenario = create_scenario(disciplines=[chain], objective_name="compliance", design_space=ds, formulation_name="DisciplinaryOpt")
    scenario.add_constraint("volume", "ineq", positive=False, value=0.0)
    
    # Execute 1 iteration
    scenario.execute(algo_name="MMA", max_iter=1)
    
    assert "compliance" in chain.local_data
    assert "volume" in chain.local_data
    # Compliance should be log(c+1)
    assert chain.local_data["compliance"][0] >= 0.0
