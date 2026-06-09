import pytest
import dolfin as df
import numpy as np
from dolfin_adjoint import Constant, DirichletBC, UnitSquareMesh
from samo_ggp.geometry.factory import GeometryFactory
from samo_ggp.physics.factory import PhysicsFactory
from samo_ggp.gemseo_wrappers.macro_discipline import GGPMacroDiscipline
import gemseo
from gemseo import create_scenario

def test_macro_discipline_execution():
    """Verify that the MacroDiscipline successfully executes and generates gradients."""
    mesh = UnitSquareMesh(2, 2)
    V_u = df.VectorFunctionSpace(mesh, "CG", 1)
    def left_boundary(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
    bc = [DirichletBC(V_u, Constant((0.0, 0.0)), left_boundary)]
    
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    class RightBoundary(df.SubDomain):
        def inside(self, x, on_boundary): return df.near(x[0], 1.0)
    RightBoundary().mark(boundaries, 1)
    ds_load = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    
    L_rhs_vec = Constant((0.0, -1.0))
    
    mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=1)
    solver = PhysicsFactory.create_solver("Elasticity_2D", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec)
    
    x_init = np.array([0.5, 0.5, 0.2, 0.2, 0.0, 1.0])
    lb = np.array([0.0]*6)
    ub = np.array([1.0]*6)
    
    disc = GGPMacroDiscipline(mapper, solver, x_init, lb, ub, mesh_area=1.0)
    
    # Test execution
    input_data = {"x_vars": np.array([0.5]*6)}
    disc.execute(input_data)
    
    # Assert outputs are generated
    assert "compliance" in disc.local_data
    assert "volume" in disc.local_data
    assert disc.local_data["compliance"][0] > 0.0
    
    # Test Jacobian computation via PyAdjoint
    jac = disc.linearize(input_data, compute_all_jacobians=True)
    assert "compliance" in jac
    assert "x_vars" in jac["compliance"]
    assert jac["compliance"]["x_vars"].shape == (1, 6)
