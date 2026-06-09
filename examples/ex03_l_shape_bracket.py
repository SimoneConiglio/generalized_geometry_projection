import dolfin as df
from dolfin_adjoint import Constant, DirichletBC, RectangleMesh, stop_annotating
import numpy as np
import gemseo
from gemseo import create_scenario
from samo_ggp.geometry.factory import GeometryFactory
from samo_ggp.physics.factory import PhysicsFactory
from samo_ggp.gemseo_wrappers.macro_discipline import GGPMacroDiscipline
import matplotlib.pyplot as plt
import os

def run_l_shape_bracket(max_iter=50):
    # Model a simple rectangle and carve out the top right via properties later if needed
    # For a true L-Shape, one would use an L-shaped mesh or penalize the void region
    L, H = 100.0, 100.0
    nelx, nely = 100, 100
    volfrac = 0.4
    num_components = 18

    mesh = RectangleMesh(df.Point(0, 0), df.Point(L, H), nelx, nely)
    V_u = df.VectorFunctionSpace(mesh, "CG", 1)
    
    # Boundary Conditions: Top edge fixed
    def top_boundary(x, on_boundary): return on_boundary and df.near(x[1], H)
    bc = [DirichletBC(V_u, Constant((0.0, 0.0)), top_boundary)]
    
    # Load: Bottom right corner pointing down
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    class BottomRightArea(df.SubDomain):
        def inside(self, x, on_boundary): return df.near(x[0], L, 2.0) and df.near(x[1], 0.0, 2.0)
    BottomRightArea().mark(boundaries, 1)
    ds_load = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    L_rhs_vec = Constant((0.0, -1.0))

    # Initialization
    mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=num_components, method='GP')
    solver = PhysicsFactory.create_solver("Elasticity_2D", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec)
    x_init = mapper.get_initial_design(L, H)
    
    lb = np.array([0.0, 0.0, 0.0, 1.0, -2*np.pi, 0.0] * num_components)
    ub = np.array([L, H, L*1.5, H, 2*np.pi, 1.0] * num_components)

    disc = GGPMacroDiscipline(mapper, solver, x_init, lb, ub, mesh_area=L*H, name="GGP_L_Shape")
    design_space = gemseo.algos.design_space.DesignSpace()
    design_space.add_variable("x_vars", size=len(x_init), lower_bound=lb, upper_bound=ub, value=x_init)
    
    scenario = create_scenario(disciplines=[disc], objective_name="compliance", design_space=design_space, formulation_name="DisciplinaryOpt")
    scenario.add_constraint("volume", "ineq", positive=False, value=volfrac)
    
    scenario.execute(algo_name="MMA", max_iter=max_iter, max_optimization_step=0.1)

    # --- Post-Processing ---
    print("Post-processing optimal design...")
    os.makedirs("results", exist_ok=True)
    
    opt_x = scenario.optimization_result.x_opt
    for c, v in zip(disc.controls_objs, opt_x):
        c.assign(float(v))
        
    with stop_annotating():
        V_rho = df.FunctionSpace(mesh, "CG", 1)
        rho_opt_ufl = mapper.map_to_density(disc.controls_objs)
        rho_opt = df.project(rho_opt_ufl, V_rho)
        rho_opt.rename("Density", "Optimized GGP Density")
        
        df.File("results/l_shape_bracket_optimized.pvd") << rho_opt
        
        plt.figure(figsize=(7, 7))
        p = df.plot(rho_opt, cmap="Blues")
        plt.colorbar(p, label="Density $\\rho$", shrink=0.7)
        plt.title("Optimized GGP Topology: L-Shape Bracket")
        plt.savefig("results/l_shape_bracket_optimized.png", dpi=300, bbox_inches="tight")
        print("Results saved to 'results/' directory.")


if __name__ == "__main__":
    run_l_shape_bracket()
