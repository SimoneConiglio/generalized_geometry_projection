import dolfin as df
from dolfin_adjoint import *
import numpy as np
import gemseo
from gemseo import create_scenario
from gemseo.mda.mda_chain import MDAChain
from ggp.geometry.factory import GeometryFactory
from ggp.physics.factory import PhysicsFactory
from ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline, GGPPhysicsAdjointDiscipline
from ggp.utils.alm_utils import create_alm_3d_overhang_constraints, compute_alm_3d_overhang_jacobian
from ggp.utils.validation import StepByStepValidator
import matplotlib.pyplot as plt
import os

def run_3d_alm_cantilever(max_iter=50, validate=False):
    # Domain: 60 x 30 x 30
    L, H, D = 60.0, 30.0, 30.0
    nx, ny, nz = 30, 15, 15 
    volfrac = 0.2
    
    num_layers = 15
    comp_per_layer = 1
    layer_height = D / num_layers # Layers stacked along Z
    alpha_deg = 45.0

    # Mesh and Spaces
    mesh = BoxMesh(df.Point(0, 0, 0), df.Point(L, H, D), nx, ny, nz)
    V_u = df.VectorFunctionSpace(mesh, "CG", 1)
    
    # BCs: Left face fixed
    def left_face(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
    bc = [DirichletBC(V_u, Constant((0.0, 0.0, 0.0)), left_face)]
    
    # Load: Center of right face
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    class LoadArea(df.SubDomain):
        def inside(self, x, on_boundary): 
            return on_boundary and df.near(x[0], L) and df.near(x[1], H/2, 2.0) and df.near(x[2], D/2, 2.0)
    LoadArea().mark(boundaries, 1)
    ds_load = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    L_rhs_vec = Constant((0.0, -1.0, 0.0))

    solver = PhysicsFactory.create_solver("Elasticity", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec, iterative=True, p=3.0)
    
    num_comp = num_layers * comp_per_layer
    # Initial design: [Xc, Yc, L, W, Theta, Mc]
    x_init = []
    for layer in range(num_layers):
        for i in range(comp_per_layer):
            x_init.extend([L/2, H/2, L/2, H/2, 0.0, 0.5])
    x_init = np.array(x_init)
    
    lb = np.array([0.0, 0.0, 1.0, 1.0, -np.pi, 0.0] * num_comp)
    ub = np.array([L, H, L, H, np.pi, 1.0] * num_comp)

    # Hybrid Disciplines
    geom_disc = GGPVectorizedGeometryDiscipline(
        mesh, num_comp, mode='3D_ALM', num_layers=num_layers, 
        comp_per_layer=comp_per_layer, layer_height=layer_height
    )
    phys_disc = GGPPhysicsAdjointDiscipline(solver, mesh, mesh_area=L*H*D, volfrac=volfrac)
    
    # 3D Nonlinear Overhang Discipline
    from gemseo.core.discipline.discipline import Discipline
    class ALM3DNonlinearConstraintsDiscipline(Discipline):
        def __init__(self, nl, cl, lh, alpha):
            super().__init__(name="ALM_3D_Constraints")
            self.nl, self.cl, self.lh, self.alpha = nl, cl, lh, alpha
            self.input_grammar.update_from_names(["x_vars"])
            self.output_grammar.update_from_names(["overhang_3d"])
        def _run(self, input_data=None):
            if input_data is not None: self.local_data.update(input_data)
            x = self.local_data["x_vars"].flatten()
            self.local_data["overhang_3d"] = create_alm_3d_overhang_constraints(self.nl, self.cl, self.lh, self.alpha, x)
        def _compute_jacobian(self, inputs=None, outputs=None):
            x = self.local_data["x_vars"].flatten()
            self.jac = {"overhang_3d": {"x_vars": compute_alm_3d_overhang_jacobian(self.nl, self.cl, self.lh, self.alpha, x)}}

    cons_disc = ALM3DNonlinearConstraintsDiscipline(num_layers, comp_per_layer, layer_height, alpha_deg)
    chain = MDAChain([geom_disc, phys_disc])
    
    design_space = gemseo.algos.design_space.DesignSpace()
    design_space.add_variable("x_vars", size=len(x_init), lower_bound=lb, upper_bound=ub, value=x_init)
    
    scenario = create_scenario(disciplines=[chain, cons_disc], objective_name="compliance", design_space=design_space, formulation_name="DisciplinaryOpt")
    scenario.add_constraint("volume", "ineq", positive=False, value=0.0)
    scenario.add_constraint("overhang_3d", "ineq", positive=False, value=0.0)
    
    print(f">>> Starting 3D ALM Optimization ({nx}x{ny}x{nz} mesh, {num_layers} layers)...")
    scenario.execute(algo_name="MMA", max_iter=max_iter, max_optimization_step=0.01)

    # Post-processing
    print("Post-processing 3D ALM result...")
    os.makedirs("results", exist_ok=True)
    opt_x = scenario.optimization_result.x_opt
    with stop_annotating():
        V_rho = df.FunctionSpace(mesh, "DG", 0)
        rho_opt_arr = geom_disc._map_logic(opt_x, power=1.0)
        rho_opt = df.Function(V_rho)
        rho_opt.vector()[:] = rho_opt_arr
        df.File("results/cantilever_3d_alm_optimized.pvd") << rho_opt
        
        plt.figure(figsize=(10, 7))
        df.plot(rho_opt, cmap="gray_r")
        plt.title(f"3D ALM Optimized Topology ({alpha_deg} deg overhang)")
        plt.savefig("results/cantilever_3d_alm_optimized.png", dpi=300)
        print("3D results saved.")

if __name__ == "__main__":
    run_3d_alm_cantilever()
