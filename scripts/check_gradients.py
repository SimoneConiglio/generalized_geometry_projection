from dolfin import *
from dolfin_adjoint import *
import numpy as np
import gemseo
from ggp.geometry.factory import GeometryFactory
from ggp.physics.factory import PhysicsFactory
from ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline, GGPPhysicsFastDiscipline
import matplotlib.pyplot as plt

def check_gradients():
    L, H = 6.0, 3.0
    nelx, nely = 20, 10
    volfrac = 0.4
    num_components = 2

    mesh = RectangleMesh(Point(0, 0), Point(L, H), nelx, nely)
    V_u = VectorFunctionSpace(mesh, "CG", 1)
    
    def left_boundary(x, on_boundary): return on_boundary and near(x[0], 0.0)
    bc = [DirichletBC(V_u, Constant((0.0, 0.0)), left_boundary)]
    
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    ds_load = Measure("ds", domain=mesh, subdomain_data=boundaries)
    L_rhs_vec = Constant((0.0, -1.0))

    solver = PhysicsFactory.create_solver("Elasticity", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec, p=1.0, plane_stress=True)
    mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=num_components, method='GP')
    
    x_init = mapper.get_initial_design(L, H)
    lb = np.array([0.0, 0.0, 0.0, 1.0, -2*np.pi, 0.0] * num_components)
    ub = np.array([L, H, np.sqrt(L**2+H**2), np.sqrt(L**2+H**2), 2*np.pi, 1.0] * num_components)

    geom_disc = GGPVectorizedGeometryDiscipline(mesh, num_components, mode='Free', L_domain=L, H_domain=H, Ngp=2)
    
    print("Checking Geometry Jacobian...")
    res_geom = geom_disc.check_jacobian(
        input_data={"x_vars": x_init},
        output_names=["rho_E", "rho_V"],
        input_names=["x_vars"],
        step=1e-6,
        threshold=1e-3,
        parallel=False,
    )
    print(f"Geometry Jacobian check result: {res_geom}")

    phys_disc = GGPPhysicsFastDiscipline(solver, mesh, mesh_area=L*H, volfrac=volfrac)
    print("\nChecking Physics Jacobian...")
    res_phys = phys_disc.check_jacobian(
        input_data={"rho_E": np.ones(mesh.num_cells()) * 0.5, "rho_V": np.ones(mesh.num_cells()) * 0.5},
        output_names=["compliance", "volume"],
        input_names=["rho_E", "rho_V"],
        step=1e-6,
        threshold=1e-3,
        parallel=False,
    )
    print(f"Physics Jacobian check result: {res_phys}")

if __name__ == "__main__":
    check_gradients()

if __name__ == "__main__":
    check_gradients()
