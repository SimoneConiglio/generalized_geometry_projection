import dolfin as df
from dolfin_adjoint import Constant, DirichletBC, RectangleMesh
import numpy as np
import gemseo
from gemseo import create_scenario
from samo_ggp.geometry.factory import GeometryFactory
from samo_ggp.physics.factory import PhysicsFactory
from samo_ggp.gemseo_wrappers.macro_discipline import GGPMacroDiscipline
from samo_ggp.gemseo_wrappers.modular_disciplines import GGPGeometryDiscipline, GGPPhysicsDiscipline
from gemseo.mda.mda_chain import MDAChain
from samo_ggp.utils.profiling import profile_me, BenchmarkLogger
import os

# Create results directory
os.makedirs("results_benchmark", exist_ok=True)

# Setup Logger
logger = BenchmarkLogger()

# Common Parameters (Short Cantilever)
L, H = 50.0, 50.0
nelx, nely = 50, 50
volfrac = 0.4
num_components = 10

# Mesh and Spaces
mesh = RectangleMesh(df.Point(0, 0), df.Point(L, H), nelx, nely)
V_u = df.VectorFunctionSpace(mesh, "CG", 1)
boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
def left_boundary(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
class MiddleRightArea(df.SubDomain):
    def inside(self, x, on_boundary): return df.near(x[0], L, 2.0) and df.near(x[1], H/2.0, 5.0)
MiddleRightArea().mark(boundaries, 1)
bc = [DirichletBC(V_u, Constant((0.0, 0.0)), left_boundary)]
ds_load = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
L_rhs_vec = Constant((0.0, -1.0))

# Initialize Mapper and Solver
mapper = GeometryFactory.create_mapper("2D_Free", mesh=mesh, num_components=num_components)
solver = PhysicsFactory.create_solver("Elasticity_2D", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=L_rhs_vec)
x_init = mapper.get_initial_design(L, H)
lb = np.array([0.0, 0.0, 5.0, 2.0, -np.pi] * num_components)
ub = np.array([L, H, L, H/2, np.pi] * num_components)

@profile_me(output_file="results_benchmark/macro_discipline.prof")
def run_macro_benchmark():
    print("\n--- Running Macro-Discipline Benchmark ---")
    disc = GGPMacroDiscipline(mapper, solver, x_init, lb, ub, mesh_area=L*H)
    design_space = gemseo.algos.design_space.DesignSpace()
    design_space.add_variable("x_vars", size=len(x_init), lower_bound=lb, upper_bound=ub, value=x_init)
    
    scenario = create_scenario(disciplines=[disc], objective_name="compliance", design_space=design_space, formulation_name="DisciplinaryOpt")
    scenario.add_constraint("volume", "ineq", positive=False, value=volfrac)
    scenario.set_differentiation_method("user")
    
    import time
    start = time.perf_counter()
    scenario.execute(algo_name="MMA", max_iter=5, max_optimization_step=0.1)
    duration = time.perf_counter() - start
    logger.log_result("Macro-Discipline", duration, iterations=5)

@profile_me(output_file="results_benchmark/modular_chain.prof")
def run_modular_benchmark():
    print("\n--- Running Modular Sub-Discipline Chain Benchmark ---")
    geom_disc = GGPGeometryDiscipline(mapper, x_init)
    phys_disc = GGPPhysicsDiscipline(solver, mesh_area=L*H)
    
    chain = MDAChain([geom_disc, phys_disc])
    
    design_space = gemseo.algos.design_space.DesignSpace()
    design_space.add_variable("x_vars", size=len(x_init), lower_bound=lb, upper_bound=ub, value=x_init)
    
    scenario = create_scenario(disciplines=[chain], objective_name="compliance", design_space=design_space, formulation_name="DisciplinaryOpt")
    scenario.add_constraint("volume", "ineq", positive=False, value=volfrac)
    
    # Let GEMSEO handle the chain rule (user mode means we must provide analytical, but we haven't implemented it in the modular disciplines)
    # We will use finite differences on the chain for this test to show overhead, 
    # as implementing cross-discipline FEniCS-adjoint tracking requires a custom GEMSEO Adapter.
    scenario.set_differentiation_method("complex_step") 
    
    import time
    start = time.perf_counter()
    scenario.execute(algo_name="MMA", max_iter=5, max_optimization_step=0.1)
    duration = time.perf_counter() - start
    logger.log_result("Modular Chain (Complex Step)", duration, iterations=5)

if __name__ == "__main__":
    run_macro_benchmark()
    # run_modular_benchmark() # Disabling temporarily because finite difference on 50 variables will be extremely slow
    logger.print_summary()
