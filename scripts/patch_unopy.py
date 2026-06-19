import re

with open("Main_ggp_unopy.py", "r") as f:
    content = f.read()

unopy_wrapper_class = """
import sys
sys.path.append("/mnt/d/samo_project/Uno/build")
import unopy
import math

class UnopyGGPWrapper:
    def __init__(self, chain, cons_disc=None):
        self.chain = chain
        self.cons_disc = cons_disc
        self.last_x = None
        self.last_obj = None
        self.last_grad_obj = None
        self.last_cons = None
        self.last_jac_cons = None

    def evaluate(self, x):
        x_arr = np.array(x)
        if self.last_x is not None and np.array_equal(x_arr, self.last_x):
            return
            
        res = self.chain.execute({"x_vars": x_arr})
        self.chain.linearize({"x_vars": x_arr})
        
        self.last_obj = float(res["compliance"][0])
        self.last_grad_obj = self.chain.jac["compliance"]["x_vars"][0, :]
        
        cons_list = []
        jac_list = []
        
        # volume constraint
        cons_list.append(float(res["volume"][0]))
        jac_list.append(self.chain.jac["volume"]["x_vars"][0, :])
        
        if self.cons_disc is not None:
            res_cons = self.cons_disc.execute({"x_vars": x_arr})
            self.cons_disc.linearize({"x_vars": x_arr})
            overhang = res_cons["overhang_cons"]
            jac_overhang = self.cons_disc.jac["overhang_cons"]["x_vars"]
            for i in range(len(overhang)):
                cons_list.append(float(overhang[i]))
                jac_list.append(jac_overhang[i, :])
                
        self.last_cons = np.array(cons_list)
        self.last_jac_cons = np.vstack(jac_list)
        self.last_x = np.copy(x_arr)

    def objective(self, x):
        self.evaluate(x)
        return self.last_obj

    def constraints(self, x, constraint_values):
        self.evaluate(x)
        constraint_values[:] = self.last_cons.tolist()

    def objective_gradient(self, x, gradient):
        self.evaluate(x)
        gradient[:] = self.last_grad_obj.tolist()

    def jacobian(self, x, jacobian_values):
        self.evaluate(x)
        jacobian_values[:] = self.last_jac_cons.flatten().tolist()
"""

# Insert wrapper before run_main_ggp
content = content.replace("def run_main_ggp(", unopy_wrapper_class + "\n\ndef run_main_ggp(")

# Now replace the gemseo scenario part
old_scenario_part = """        design_space = create_design_space()
        design_space.add_variable("x_vars", size=len(current_x), lower_bound=0.0, upper_bound=1.0, value=current_x)
        
        scenario = create_scenario(
            disciplines_to_add, "compliance", design_space, maximize_objective=False, formulation_name="MDF"
        )
        scenario.add_constraint("volume", constraint_type="ineq", positive=False, value=0.0)
        if mode == "ALM":
            scenario.add_constraint("overhang_cons", constraint_type="ineq", positive=False, value=0.0)
        
        algo_options = {
            "algo_name": "MMA",
            "max_iter": max_iter,
            "max_optimization_step": 0.01,
            "penalty_formulation": "linear",
            "ftol_rel": 1e-12,
            "xtol_rel": 1e-12,
            "ftol_abs": 1e-12,
            "xtol_abs": 1e-12,
            "max_inner_iterations": 500,
        }
        scenario.execute(**algo_options)
        
        current_x = scenario.design_space.get_current_value().copy()"""

new_unopy_part = """        # Unopy integration
        wrapper = UnopyGGPWrapper(chain, cons_disc if mode == "ALM" else None)
        
        number_variables = len(current_x)
        variables_lower_bounds = [0.0] * number_variables
        variables_upper_bounds = [1.0] * number_variables
        
        number_constraints = 1
        if mode == "ALM":
            number_constraints += len(cons_disc.b)
            
        constraints_lower_bounds = [-math.inf] * number_constraints
        constraints_upper_bounds = [0.0] * number_constraints
        
        number_jacobian_nonzeros = number_constraints * number_variables
        jacobian_row_indices = [c for c in range(number_constraints) for v in range(number_variables)]
        jacobian_column_indices = [v for c in range(number_constraints) for v in range(number_variables)]
        
        model = unopy.Model(unopy.PROBLEM_NONLINEAR, number_variables, unopy.ZERO_BASED_INDEXING)
        model.set_variables_lower_bounds(variables_lower_bounds)
        model.set_variables_upper_bounds(variables_upper_bounds)
        model.set_objective(unopy.MINIMIZE, wrapper.objective, wrapper.objective_gradient)
        model.set_constraints(number_constraints, wrapper.constraints, constraints_lower_bounds, constraints_upper_bounds,
                              number_jacobian_nonzeros, jacobian_row_indices, jacobian_column_indices, wrapper.jacobian)
        model.set_initial_primal_iterate(current_x.tolist())
        
        uno_solver = unopy.UnoSolver()
        uno_solver.set_preset("sqp")
        uno_solver.set_option("subproblem", solver_name)
        # BQPD needs to be used for QP solver if subproblem is not SCP? But MMA/CONLIN are SCP subproblems so they solve themselves.
        
        # Max iter
        # uno_solver.set_option("max_iterations", str(max_iter))
        
        print(f"Solving with Uno {unopy.current_uno_version()} using {solver_name}...")
        result = uno_solver.optimize(model)
        print("Optimization status:", result.optimization_status)
        current_x = np.array(result.primal_solution)"""

content = content.replace(old_scenario_part, new_unopy_part)

# Add solver_name arg to run_main_ggp
content = content.replace("def run_main_ggp(bc_type=\"Short_Cantilever\", max_iter=50, mode=\"Free\"):", "def run_main_ggp(bc_type=\"Short_Cantilever\", max_iter=50, mode=\"Free\", solver_name=\"MMA\"):")
content = content.replace("parser.add_argument(\"--mode\"", "parser.add_argument(\"--solver\", type=str, default=\"MMA\", choices=[\"MMA\", \"CONLIN\", \"HiGHS\"], help=\"Subproblem solver in Uno.\")\n    parser.add_argument(\"--mode\"")
content = content.replace("mode=args.mode)", "mode=args.mode, solver_name=args.solver)")

with open("Main_ggp_unopy.py", "w") as f:
    f.write(content)
