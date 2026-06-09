import dolfin as df
from dolfin_adjoint import *
import numpy as np
import gemseo
from gemseo import create_scenario
from gemseo.mda.mda_chain import MDAChain
from samo_ggp.geometry.factory import GeometryFactory
from samo_ggp.physics.factory import PhysicsFactory
from samo_ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline, GGPPhysicsAdjointDiscipline
import os

def run_3d_cantilever(max_iter=50):
    # Domain: 60 x 30 x 30
    L, H, D = 60.0, 30.0, 30.0
    nx, ny, nz = 30, 15, 15 # Lower res for desktop profiling
    volfrac = 0.2
    num_components = 40

    # Mesh and Spaces
    mesh = BoxMesh(df.Point(0, 0, 0), df.Point(L, H, D), nx, ny, nz)
    V_u = df.VectorFunctionSpace(mesh, "CG", 1)
    
    # BCs: Left face fixed (x=0)
    def left_face(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
    bc = [DirichletBC(V_u, Constant((0.0, 0.0, 0.0)), left_face)]
    
    # Load: Bottom center of right face
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    class LoadArea(df.SubDomain):
        def inside(self, x, on_boundary): 
            return on_boundary and df.near(x[0], L) and df.near(x[1], 0.0, 2.0) and df.near(x[2], D/2, 5.0)
    LoadArea().mark(boundaries, 1)
    ds_load = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    L_rhs_vec = Constant((0.0, -1.0, 0.0))

    # Solver (Scalable CG + AMG)
    solver = PhysicsFactory.create_solver("Elasticity", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec, iterative=True)
    mapper = GeometryFactory.create_mapper("3D_Free", mesh=mesh, num_components=num_components)
    x_init = mapper.get_initial_design(L, H, D)
    
    # Bounds for [X, Y, Z, L, h, Theta, Phi, Mc]
    lb = np.array([0.0, 0.0, 0.0, 1.0, 1.0, -np.pi, -np.pi, 0.0] * num_components)
    ub = np.array([L, H, D, L, H, np.pi, np.pi, 1.0] * num_components)

    # Hybrid Disciplines
    geom_disc = GGPVectorizedGeometryDiscipline(mesh, num_components, mode='Free')
    phys_disc = GGPPhysicsAdjointDiscipline(solver, mesh, mesh_area=L*H*D, volfrac=volfrac)
    chain = MDAChain([geom_disc, phys_disc])
    
    design_space = gemseo.algos.design_space.DesignSpace()
    design_space.add_variable("x_vars", size=len(x_init), lower_bound=lb, upper_bound=ub, value=x_init)
    
    scenario = create_scenario(disciplines=[chain], objective_name="compliance", design_space=design_space, formulation_name="DisciplinaryOpt")
    scenario.add_constraint("volume", "ineq", positive=False, value=0.0)
    
    print(f">>> Starting 3D Optimization ({nx}x{ny}x{nz} mesh, {num_components} components)...")
    scenario.execute(algo_name="MMA", max_iter=max_iter)

    # Post-processing
    print("Post-processing 3D result...")
    os.makedirs("results", exist_ok=True)
    opt_x = scenario.optimization_result.x_opt
    with stop_annotating():
        V_rho = df.FunctionSpace(mesh, "DG", 0)
        rho_opt_arr = geom_disc._map_logic(opt_x, power=1.0)
        rho_opt = df.Function(V_rho)
        rho_opt.vector()[:] = rho_opt_arr
        df.File("results/cantilever_3d_optimized.pvd") << rho_opt
        print("3D results saved to results/cantilever_3d_optimized.pvd (Open in ParaView).")

if __name__ == "__main__":
    run_3d_cantilever()
