import dolfin as df
from dolfin_adjoint import *
import numpy as np
import gemseo
from gemseo import create_scenario
from gemseo.mda.mda_chain import MDAChain
from samo_ggp.geometry.factory import GeometryFactory
from samo_ggp.physics.factory import PhysicsFactory
from samo_ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline, GGPPhysicsAdjointDiscipline
import matplotlib.pyplot as plt
import os

def run_short_cantilever(max_iter=50):
    L, H = 60.0, 30.0
    nelx, nely = 60, 30
    volfrac = 0.4
    num_components = 18

    mesh = RectangleMesh(df.Point(0, 0), df.Point(L, H), nelx, nely)
    V_u = df.VectorFunctionSpace(mesh, "CG", 1)
    def left_boundary(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
    bc = [DirichletBC(V_u, Constant((0.0, 0.0)), left_boundary)]
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    class MiddleRightArea(df.SubDomain):
        def inside(self, x, on_boundary): return df.near(x[0], L, 1.0) and df.near(x[1], H/2.0, 2.0)
    MiddleRightArea().mark(boundaries, 1)
    ds_load = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    L_rhs_vec = Constant((0.0, -1.0))

    solver = PhysicsFactory.create_solver("Elasticity_2D", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec)
    mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=num_components, method='GP')
    x_init = mapper.get_initial_design(L, H)
    lb = np.array([0.0, 0.0, 0.0, 1.0, -2*np.pi, 0.0] * num_components)
    ub = np.array([L, H, L*1.5, H, 2*np.pi, 1.0] * num_components)

    geom_disc = GGPVectorizedGeometryDiscipline(mesh, num_components, mode='Free', L_domain=L, H_domain=H)
    phys_disc = GGPPhysicsAdjointDiscipline(solver, mesh, mesh_area=L*H, volfrac=volfrac)
    chain = MDAChain([geom_disc, phys_disc])
    
    design_space = gemseo.algos.design_space.DesignSpace()
    design_space.add_variable("x_vars", size=len(x_init), lower_bound=lb, upper_bound=ub, value=x_init)
    scenario = create_scenario(disciplines=[chain], objective_name="compliance", design_space=design_space, formulation_name="DisciplinaryOpt")
    scenario.add_constraint("volume", "ineq", positive=False, value=0.0)
    
    scenario.execute(algo_name="MMA", max_iter=max_iter, max_optimization_step=0.1)

    # --- Post-Processing ---
    print("Post-processing optimal design...")
    os.makedirs("results", exist_ok=True)
    opt_x = scenario.optimization_result.x_opt
    
    with stop_annotating():
        V_rho = df.FunctionSpace(mesh, "DG", 0)
        # Physical Density
        rho_opt_arr = geom_disc._map_logic(opt_x, power=1.0)
        rho_opt = df.Function(V_rho)
        rho_opt.vector()[:] = rho_opt_arr
        
        df.File("results/short_cantilever_optimized.pvd") << rho_opt
        
        # Dual Plot: Density and Component Layout
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        p = df.plot(rho_opt, cmap="gray_r")
        plt.colorbar(p, label="Density $\\rho$")
        plt.title("Optimized GGP Topology: Short Cantilever")
        plt.savefig("results/short_cantilever_optimized.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Results saved.")

if __name__ == "__main__":
    run_short_cantilever()
