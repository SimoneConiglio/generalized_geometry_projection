from dolfin import *
from dolfin_adjoint import *
import numpy as np
from ggp.physics.factory import PhysicsFactory
from ggp.gemseo_wrappers.modular_disciplines import GGPPhysicsFastDiscipline

def check_compliance_comparison():
    # Use a slender beam to match EB theory better
    L, H = 60.0, 4.0
    nelx, nely = 60, 4
    mesh = RectangleMesh(Point(0, 0), Point(L, H), nelx, nely)
    V_u = VectorFunctionSpace(mesh, "CG", 1)
    
    def left_boundary(x, on_boundary): return on_boundary and near(x[0], 0.0)
    bc = [DirichletBC(V_u, Constant((0.0, 0.0)), left_boundary)]
    
    # Segment load at tip (width 1.0)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    class LoadPoint(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > L - 0.1 and abs(x[1] - H/2.0) < 0.51
    LoadPoint().mark(boundaries, 1)
    ds_load = Measure("ds", domain=mesh, subdomain_data=boundaries)
    
    # Total force = 1.0. Segment length is 1.0.
    L_rhs_vec = Constant((0.0, -1.0 / 1.0))
    
    solver = PhysicsFactory.create_solver("Elasticity", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec, p=1.0, plane_stress=True)
    phys_disc = GGPPhysicsFastDiscipline(solver, mesh, mesh_area=L*H, volfrac=1.0)
    
    # Standard FEniCS
    rho_ufl = interpolate(Constant(1.0), FunctionSpace(mesh, "DG", 0))
    u_standard = solver.solve(rho_ufl)
    comp_standard = solver.compute_compliance(u_standard)
    
    # Manual
    rho_np = np.ones(mesh.num_cells())
    phys_disc.execute({"rho_E": rho_np, "rho_V": rho_np})
    comp_manual = np.exp(phys_disc.local_data["compliance"][0]) - 1.0
    
    print(f"STANDARD FEniCS Compliance: {comp_standard}")
    print(f"MANUAL Assembly Compliance: {comp_manual}")
    
    # Analytical: C = 4 * (L/H)^3 = 4 * (60/4)^3 = 4 * (15)^3 = 4 * 3375 = 13500
    print(f"Analytical Slender (EB): {4 * (L/H)**3}")

if __name__ == "__main__":
    check_compliance_comparison()
