import dolfin as df
from dolfin_adjoint import *
import numpy as np
import gemseo
from gemseo import create_scenario
from gemseo.mda.mda_chain import MDAChain
from ggp.geometry.factory import GeometryFactory
from ggp.physics.factory import PhysicsFactory
from ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline, GGPPhysicsAdjointDiscipline
from ggp.utils.alm_utils import create_alm_overhang_constraints
import matplotlib.pyplot as plt
import os

def run_alm_cantilever(max_iter=50):
    L, H = 60.0, 30.0
    nelx, nely = 60, 30
    volfrac = 0.3
    
    num_layers = 30
    comp_per_layer = 1
    layer_height = H / num_layers
    alpha_deg = 45.0
    
    # Mesh and Spaces
    mesh = RectangleMesh(df.Point(0, 0), df.Point(L, H), nelx, nely)
    V_u = df.VectorFunctionSpace(mesh, "CG", 1)
    def left_boundary(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
    bc = [DirichletBC(V_u, Constant((0.0, 0.0)), left_boundary)]
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    class BottomRight(df.SubDomain):
        def inside(self, x, on_boundary): return df.near(x[0], L) and df.near(x[1], 0.0, 1.0)
    BottomRight().mark(boundaries, 1)
    ds_load = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    L_rhs_vec = Constant((0.0, -1.0))

    # Solver
    solver = PhysicsFactory.create_solver("Elasticity", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec, p=3.0)
    mapper = GeometryFactory.create_mapper("2D_ALM", mesh=mesh, num_layers=num_layers, 
                                          components_per_layer=comp_per_layer, layer_height=layer_height)
    x_init = mapper.get_initial_design(L, H, extended=True)
    
    lb_comp = [0.0, 1.0, 0.0, 0.0] * mapper.num_components
    ub_comp = [L, L, 1.0, layer_height] * mapper.num_components
    
    # Add bounds for y0, theta0
    lb_global = [-H/2, -np.pi/4]
    ub_global = [H/2, np.pi/4]
    
    lb = np.array(lb_comp + lb_global)
    ub = np.array(ub_comp + ub_global)

    # --- Hybrid Modular Architecture ---
    geom_disc = GGPVectorizedGeometryDiscipline(
        mesh, mapper.num_components, mode='ALM', 
        num_layers=num_layers, comp_per_layer=comp_per_layer, layer_height=layer_height
    )
    phys_disc = GGPPhysicsAdjointDiscipline(solver, mesh, mesh_area=L*H, volfrac=volfrac)
    
    from gemseo.core.discipline.discipline import Discipline
    class ALMConstraintsDiscipline(Discipline):
        def __init__(self, A, b):
            super().__init__(name="ALM_Constraints")
            self.A, self.b = A, b
            self.input_grammar.update_from_names(["x_vars"])
            self.output_grammar.update_from_names(["overhang_cons"])
        def _run(self, input_data=None):
            if input_data is not None: self.local_data.update(input_data)
            x = self.local_data["x_vars"].flatten()
            self.local_data["overhang_cons"] = self.A @ x - self.b
        def _compute_jacobian(self, inputs=None, outputs=None):
            self.jac = {"overhang_cons": {"x_vars": self.A}}

    A_over, b_over = create_alm_overhang_constraints(num_layers, comp_per_layer, layer_height, alpha_deg, extended=True)
    cons_disc = ALMConstraintsDiscipline(A_over, b_over)
    
    chain = MDAChain([geom_disc, phys_disc])
    design_space = gemseo.algos.design_space.DesignSpace()
    design_space.add_variable("x_vars", size=len(x_init), lower_bound=lb, upper_bound=ub, value=x_init)
    
    scenario = create_scenario(disciplines=[chain, cons_disc], objective_name="compliance", 
                               design_space=design_space, formulation_name="DisciplinaryOpt")
    scenario.add_constraint("volume", "ineq", positive=False, value=0.0)
    scenario.add_constraint("overhang_cons", "ineq", positive=False, value=0.0)
    
    # MMA with conservative move limit (0.01) matching MATLAB
    scenario.execute(algo_name="MMA", max_iter=max_iter, max_optimization_step=0.01)

    # --- Post-Processing ---
    print("Post-processing ALM result...")
    os.makedirs("results", exist_ok=True)
    opt_x = scenario.optimization_result.x_opt
    
    with stop_annotating():
        V_rho = df.FunctionSpace(mesh, "DG", 0)
        rho_opt_arr = geom_disc._map_logic(opt_x, power=1.0)
        rho_opt = df.Function(V_rho)
        rho_opt.vector()[:] = rho_opt_arr
        
        df.File("results/alm_cantilever_optimized.pvd") << rho_opt
        plt.figure(figsize=(10, 5))
        p = df.plot(rho_opt, cmap="gray_r")
        plt.colorbar(p, label="Density $\\rho$")
        plt.title(f"ALM Optimized Topology (Overhang {alpha_deg} deg)")
        plt.savefig("results/alm_cantilever_optimized.png", dpi=300)
        plt.close()
        print("Results saved.")

if __name__ == "__main__":
    run_alm_cantilever()
