import dolfin as df
from dolfin_adjoint import *
import numpy as np
import gemseo
from gemseo import create_scenario
from gemseo.mda.mda_chain import MDAChain
from ggp.geometry.factory import GeometryFactory
from ggp.physics.factory import PhysicsFactory
from ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline, GGPPhysicsFastDiscipline
import os

def run_short_cantilever_repro(max_iter=100):
    L, H = 60.0, 30.0
    nelx, nely = 60, 30
    volfrac = 0.4
    num_components = 18

    mesh = RectangleMesh(df.Point(0, 0), df.Point(L, H), nelx, nely)
    V_u = df.VectorFunctionSpace(mesh, "CG", 1)
    
    # Boundary conditions: fixed on the left
    def left_boundary(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
    bc = [DirichletBC(V_u, Constant((0.0, 0.0)), left_boundary)]
    
    # Load: Point Load -1.0 at (L, H/2) in Y-direction
    # Direct DOF manipulation for robustness
    dof_coords = V_u.tabulate_dof_coordinates()
    # We want x=L, y=H/2, and the Y-component (usually odd index or check via sub(1))
    # Let's find all DOFs at (L, H/2)
    dists = np.linalg.norm(dof_coords - np.array([L, H/2.0]), axis=1)
    candidate_dofs = np.where(dists < 1e-3)[0]
    
    # Identify which one is the Y-component
    y_dofs = V_u.sub(1).dofmap().dofs()
    tip_y_dof = np.intersect1d(candidate_dofs, y_dofs)[0]
    print(f"Tip Y-DOF identified: {tip_y_dof} at {dof_coords[tip_y_dof]}")
    
    f_vec = np.zeros(V_u.dim())
    f_vec[tip_y_dof] = -1.0
    
    # Dummy boundaries for solver init
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    ds_load = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    L_rhs_vec = Constant((0.0, 0.0))

    # Solver with Plane Stress
    solver = PhysicsFactory.create_solver("Elasticity", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec, p=1.0, plane_stress=True)
    
    # Geometry Mapper
    mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=num_components, method='GP')
    x_init = mapper.get_initial_design(L, H)
    
    # Bounds matching MATLAB
    lb = np.array([0.0, 0.0, 0.0, 1.0, -2*np.pi, 0.0] * num_components)
    ub = np.array([L, H, np.sqrt(L**2+H**2), np.sqrt(L**2+H**2), 2*np.pi, 1.0] * num_components)

    geom_disc = GGPVectorizedGeometryDiscipline(mesh, num_components, mode='Free', L_domain=L, H_domain=H, Ngp=4)
    # Use Fast Discipline
    phys_disc = GGPPhysicsFastDiscipline(solver, mesh, mesh_area=L*H, volfrac=volfrac)
    # Manually override the load vector in the fast discipline
    phys_disc.f_vec = f_vec
    phys_disc.f_vec[phys_disc.fixed_dofs] = 0.0 # Ensure BCs
    
    chain = MDAChain([geom_disc, phys_disc])
    
    design_space = gemseo.algos.design_space.DesignSpace()
    design_space.add_variable("x_vars", size=len(x_init), lower_bound=lb, upper_bound=ub, value=x_init)
    
    scenario = create_scenario(disciplines=[chain], objective_name="compliance", design_space=design_space, formulation_name="DisciplinaryOpt")
    scenario.add_constraint("volume", "ineq", positive=False, value=0.0)
    
    # MMA with move limit 0.1 as per paper recommendation for faster convergence
    scenario.execute(algo_name="MMA", max_iter=max_iter, max_optimization_step=0.1)

    opt_x = scenario.optimization_result.x_opt
    final_compliance_log = scenario.optimization_result.f_opt
    final_compliance = np.exp(final_compliance_log) - 1.0
    final_volume = phys_disc.local_data["volume"][0]
    
    print(f"Final Compliance: {final_compliance}")
    print(f"Final Volume Constraint: {final_volume}")
    
    # Visualization
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    
    # Get mesh coordinates and cells
    coords = mesh.coordinates()
    cells = mesh.cells()
    rho_opt = geom_disc.local_data["rho_V"]
    
    # Each triangle has one density value. For a smooth plot, we can average at nodes or just use tripcolor
    plt.figure(figsize=(10, 5))
    plt.tripcolor(coords[:, 0], coords[:, 1], cells, facecolors=rho_opt, cmap='gray_r')
    plt.colorbar()
    plt.title(f"Optimized Topology (C={final_compliance:.2f}, V={final_volume:.2f}%)")
    plt.axis('equal')
    plt.savefig("results/short_cantilever_fast.png")
    print("Optimization finished. Topology saved to results/short_cantilever_fast.png")
    
    return final_compliance

if __name__ == "__main__":
    run_short_cantilever_repro()
