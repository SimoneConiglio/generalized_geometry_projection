import dolfin as df
import numpy as np
from ggp.physics.factory import PhysicsFactory
from ggp.gemseo_wrappers.modular_disciplines import GGPPhysicsFastDiscipline

def check_solid_compliance():
    L, H = 60.0, 30.0
    nelx, nely = 60, 30
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(L, H), nelx, nely)
    V_u = df.VectorFunctionSpace(mesh, "CG", 1)
    
    def left_boundary(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
    bc = [df.DirichletBC(V_u, df.Constant((0.0, 0.0)), left_boundary)]
    
    # Unit load at tip
    dof_coords = V_u.tabulate_dof_coordinates()
    dists = np.linalg.norm(dof_coords - np.array([L, H/2.0]), axis=1)
    y_dofs = V_u.sub(1).dofmap().dofs()
    tip_y_dof = np.intersect1d(np.where(dists < 1e-3)[0], y_dofs)[0]
    
    f_vec = np.zeros(V_u.dim())
    f_vec[tip_y_dof] = -1.0
    
    # Dummy boundaries for solver init
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    ds_load = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    
    solver = PhysicsFactory.create_solver("Elasticity", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=df.Constant((0,0)), p=1.0, plane_stress=True)
    phys_disc = GGPPhysicsFastDiscipline(solver, mesh, mesh_area=L*H, volfrac=1.0)
    phys_disc.f_vec = f_vec
    phys_disc.f_vec[phys_disc.fixed_dofs] = 0.0
    
    # Run with standard FEniCS solver
    from dolfin_adjoint import stop_annotating
    with stop_annotating():
        rho_ufl = df.interpolate(df.Constant(1.0), df.FunctionSpace(mesh, "DG", 0))
        u_standard = solver.solve(rho_ufl)
        comp_standard = solver.compute_compliance(u_standard)
    print(f"STANDARD FEniCS Compliance: {comp_standard}")
    
    # Run with manual assembly
    rho_np = np.ones(mesh.num_cells())
    phys_disc.execute({"rho_E": rho_np, "rho_V": rho_np})
    comp_manual = np.exp(phys_disc.local_data["compliance"][0]) - 1.0
    print(f"MANUAL Assembly Compliance: {comp_manual}")
    print(f"Analytical Expected: ~32.0 (for E=1, nu=0.3, L=60, H=30, P=1)")

if __name__ == "__main__":
    check_solid_compliance()
