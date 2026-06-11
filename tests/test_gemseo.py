import pytest
from dolfin import *
from dolfin_adjoint import *
import numpy as np
from ggp.geometry.factory import GeometryFactory
from ggp.physics.factory import PhysicsFactory
from ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline, GGPPhysicsAdjointDiscipline, GGPPhysicsFastDiscipline
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

def test_physics_fast_discipline():
    """Verify that GGPPhysicsFastDiscipline executes and computes sensitivities."""
    mesh = RectangleMesh.create([Point(0, 0), Point(1, 1)], [5, 5], CellType.Type.quadrilateral)
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
    
    phys_disc = GGPPhysicsFastDiscipline(solver, mesh, mesh_area=1.0, volfrac=0.4)
    
    num_elements = mesh.num_cells()
    rho = np.ones(num_elements) * 0.5
    
    phys_disc.execute({"rho_E": rho, "rho_V": rho})
    
    assert "compliance" in phys_disc.local_data
    assert "volume" in phys_disc.local_data
    
    # Check Jacobian computation
    phys_disc._compute_jacobian()
    assert "compliance" in phys_disc.jac
    assert "rho_E" in phys_disc.jac["compliance"]
    assert phys_disc.jac["compliance"]["rho_E"].shape == (1, num_elements)

def test_vectorized_geometry_scaling():
    """Verify scaling of GGPVectorizedGeometryDiscipline."""
    mesh = UnitSquareMesh(5, 5)
    num_comp = 2
    lb = np.array([0.0, 0.0, 0.1, 0.1, -np.pi, 0.0] * num_comp)
    ub = np.array([1.0, 1.0, 1.0, 1.0, np.pi, 1.0] * num_comp)
    
    geom_disc = GGPVectorizedGeometryDiscipline(mesh, num_comp, mode='Free', lb=lb, ub=ub)
    
    # Scaled design variable (in [0, 1])
    x_scaled = np.ones(12) * 0.5
    
    geom_disc.execute({"x_vars": x_scaled})
    assert "rho_E" in geom_disc.local_data
    assert "rho_V" in geom_disc.local_data
    
    geom_disc._compute_jacobian()
    assert geom_disc.jac["rho_E"]["x_vars"].shape == (mesh.num_cells(), 12)

def test_alm_geometry_modes():
    """Verify 2D and 3D ALM modes in GGPVectorizedGeometryDiscipline."""
    # 2D ALM
    mesh_2d = UnitSquareMesh(5, 5)
    geom_2d = GGPVectorizedGeometryDiscipline(mesh_2d, num_components=2, mode='ALM', num_layers=2, comp_per_layer=1, layer_height=0.5)
    x_vars_2d = np.array([0.5, 0.2, 1.0] * 2)
    
    geom_2d.execute({"x_vars": x_vars_2d})
    assert "rho_E" in geom_2d.local_data
    
    # 3D ALM
    mesh_3d = BoxMesh(Point(0,0,0), Point(1,1,1), 2, 2, 2)
    geom_3d = GGPVectorizedGeometryDiscipline(mesh_3d, num_components=2, mode='3D_ALM', num_layers=2, comp_per_layer=1, layer_height=0.5)
    # [Xc, Yc, L, W, Theta, Mc] * 2
    x_vars_3d = np.array([0.5, 0.5, 0.2, 0.2, 0.0, 1.0] * 2)
    
    geom_3d.execute({"x_vars": x_vars_3d})
    assert "rho_E" in geom_3d.local_data
