import pytest
from dolfin import *
from dolfin_adjoint import *
from ggp.physics.factory import PhysicsFactory
from ggp.physics.elasticity import LinearElasticitySolver

def setup_solver():
    mesh = UnitSquareMesh(4, 4)
    V_u = VectorFunctionSpace(mesh, "CG", 1)
    def left_boundary(x, on_boundary): return on_boundary and near(x[0], 0.0)
    bc = [DirichletBC(V_u, Constant((0.0, 0.0)), left_boundary)]
    
    # Mark the right boundary for the load
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary): return near(x[0], 1.0)
    RightBoundary().mark(boundaries, 1)
    ds_load = Measure("ds", domain=mesh, subdomain_data=boundaries)
    L_rhs_vec = Constant((0.0, -1.0))
    
    solver = PhysicsFactory.create_solver("Elasticity_2D", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec)
    return solver, mesh

def test_physics_factory():
    solver, _ = setup_solver()
    assert isinstance(solver, LinearElasticitySolver)

def test_solve_and_compliance():
    solver, mesh = setup_solver()
    
    V_rho = FunctionSpace(mesh, "CG", 1)
    rho = interpolate(Constant(0.5), V_rho)
    
    with stop_annotating():
        u = solver.solve(rho)
        assert u is not None
        
        j = solver.compute_compliance(u)
        assert float(j) > 0.0
        
        v = solver.compute_volume(rho)
        assert float(v) > 0.0

def test_physics_3d_and_iterative():
    # 3D Mesh
    mesh = BoxMesh(Point(0,0,0), Point(1,1,1), 2, 2, 2)
    V_u = VectorFunctionSpace(mesh, "CG", 1)
    
    # BC: Fix left boundary
    def left_boundary(x, on_boundary): return on_boundary and near(x[0], 0.0)
    bc = [DirichletBC(V_u, Constant((0.0, 0.0, 0.0)), left_boundary)]
    
    # Load: force on right boundary
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary): return near(x[0], 1.0)
    RightBoundary().mark(boundaries, 1)
    ds_load = Measure("ds", domain=mesh, subdomain_data=boundaries)
    L_rhs_vec = Constant((0.0, 0.0, -1.0))
    
    # Iterative CG+AMG solver in 3D
    solver = PhysicsFactory.create_solver("Elasticity", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec, plane_stress=False, iterative=True)
    
    V_rho = FunctionSpace(mesh, "DG", 0)
    rho = interpolate(Constant(0.8), V_rho)
    
    # Solve and compute compliance
    u = solver.solve(rho)
    assert u is not None
    
    comp1 = solver.compute_compliance()  # Test default argument
    assert comp1 > 0.0
    
    ke = solver.get_unit_element_stiffness()
    assert ke is not None
