import dolfin as df
from dolfin_adjoint import *
import numpy as np
import gemseo
from gemseo import create_scenario
from gemseo.mda.mda_chain import MDAChain
from ggp.geometry.factory import GeometryFactory
from ggp.physics.factory import PhysicsFactory
from ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline, GGPPhysicsAdjointDiscipline
from ggp.utils.validation import StepByStepValidator
import matplotlib.pyplot as plt
import os

def run_mbb_beam(max_iter=50, validate=False):
    L, H = 150.0, 50.0
    nelx, nely = 150, 50
    volfrac = 0.5
    num_components = 24

    mesh = RectangleMesh(df.Point(0, 0), df.Point(L, H), nelx, nely)
    V_u = df.VectorFunctionSpace(mesh, "CG", 1)
    def left_symmetry(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
    def bottom_right_support(x, on_boundary): return on_boundary and df.near(x[0], L, 1.0) and df.near(x[1], 0.0, 1.0)
    bc = [
        DirichletBC(V_u.sub(0), Constant(0.0), left_symmetry),
        DirichletBC(V_u.sub(1), Constant(0.0), bottom_right_support, method="pointwise")
    ]
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    class TopLeftArea(df.SubDomain):
        def inside(self, x, on_boundary): return df.near(x[0], 0.0, 1.0) and df.near(x[1], H, 1.0)
    TopLeftArea().mark(boundaries, 1)
    ds_load = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    L_rhs_vec = Constant((0.0, -1.0))

    solver = PhysicsFactory.create_solver("Elasticity", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec, p=3.0)
    mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=num_components, method='GP')
    x_init = mapper.get_initial_design(L, H)
    lb = np.array([0.0, 0.0, 0.0, 1.0, -2*np.pi, 0.0] * num_components)
    ub = np.array([L, H, L*1.5, H, 2*np.pi, 1.0] * num_components)

    # Validator
    validator = StepByStepValidator(output_dir="validation_mbb") if validate else None

    # --- Hybrid Modular Architecture ---
    geom_disc = GGPVectorizedGeometryDiscipline(mesh, num_components, mode='Free', L_domain=L, H_domain=H)
    phys_disc = GGPPhysicsAdjointDiscipline(solver, mesh, mesh_area=L*H, volfrac=volfrac, validator=validator)
    chain = MDAChain([geom_disc, phys_disc])
    
    design_space = gemseo.algos.design_space.DesignSpace()
    design_space.add_variable("x_vars", size=len(x_init), lower_bound=lb, upper_bound=ub, value=x_init)
    scenario = create_scenario(disciplines=[chain], objective_name="compliance", design_space=design_space, formulation_name="DisciplinaryOpt")
    scenario.add_constraint("volume", "ineq", positive=False, value=0.0)
    
    if validate:
        def validation_callback(index):
            data = chain.local_data
            validator.export_iteration(design_space.get_current_value(), data["rho_E"], data["rho_V"],
                                       data["compliance"], data["volume"], phys_disc.dj_drhoE, phys_disc.dv_drhoV)
        scenario.add_callback(validation_callback)

    scenario.execute(algo_name="MMA", max_iter=max_iter, max_optimization_step=0.01)

    # --- Post-Processing ---
    print("Post-processing optimal design...")
    os.makedirs("results", exist_ok=True)
    opt_x = scenario.optimization_result.x_opt
    
    with stop_annotating():
        V_rho = df.FunctionSpace(mesh, "DG", 0)
        rho_opt_arr = geom_disc._map_logic(opt_x, power=1.0)
        rho_opt = df.Function(V_rho)
        rho_opt.vector()[:] = rho_opt_arr
        
        df.File("results/mbb_beam_optimized.pvd") << rho_opt
        plt.figure(figsize=(10, 4))
        p = df.plot(rho_opt, cmap="gray_r")
        plt.colorbar(p, label="Density $\\rho$")
        plt.title("Optimized GGP Topology: MBB Beam (Half Model)")
        plt.savefig("results/mbb_beam_optimized.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Results saved.")

if __name__ == "__main__":
    run_mbb_beam()
