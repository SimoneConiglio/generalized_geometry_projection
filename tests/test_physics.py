import pytest
import dolfin as df
from dolfin_adjoint import Constant, DirichletBC, stop_annotating
from samo_ggp.physics.factory import PhysicsFactory
from samo_ggp.physics.elasticity_2d import LinearElasticitySolver

def setup_solver():
    mesh = df.UnitSquareMesh(4, 4)
    V_u = df.VectorFunctionSpace(mesh, "CG", 1)
    def left_boundary(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
    bc = [DirichletBC(V_u, Constant((0.0, 0.0)), left_boundary)]
    
    # Mark the right boundary for the load
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    class RightBoundary(df.SubDomain):
        def inside(self, x, on_boundary): return df.near(x[0], 1.0)
    RightBoundary().mark(boundaries, 1)
    ds_load = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    L_rhs_vec = Constant((0.0, -1.0))
    
    solver = PhysicsFactory.create_solver("Elasticity_2D", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec)
    return solver, mesh

def test_physics_factory():
    solver, _ = setup_solver()
    assert isinstance(solver, LinearElasticitySolver)

def test_solve_and_compliance():
    solver, mesh = setup_solver()
    
    V_rho = df.FunctionSpace(mesh, "CG", 1)
    rho = df.interpolate(df.Constant(0.5), V_rho)
    
    with stop_annotating():
        u = solver.solve(rho)
        assert u is not None
        
        j = solver.compute_compliance(u)
        assert float(j) > 0.0
        
        v = solver.compute_volume(rho)
        assert float(v) > 0.0
