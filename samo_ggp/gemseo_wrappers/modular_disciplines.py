import numpy as np
from dolfin import *
from dolfin_adjoint import *
from gemseo.core.discipline.discipline import Discipline
from ..utils.vectorized_mapping import compute_local_characteristic_np, smooth_saturation_np

class GGPVectorizedGeometryDiscipline(Discipline):
    """
    Highly optimized GGP Geometry Discipline using NumPy vectorization.
    Supports both Free (6-var) and ALM (3-var) mapping strategies.
    """
    def __init__(self, mesh, num_components, mode='Free', L_domain=60.0, H_domain=30.0, 
                 num_layers=None, comp_per_layer=None, layer_height=None,
                 method='GP', r_gp=0.5, ka=10.0, pp=100.0, name="GGP_Geometry"):
        super().__init__(name=name)
        self.num_components = num_components
        self.mode = mode
        self.method = method
        self.r_gp = r_gp
        self.ka = ka
        self.pp = pp
        
        # ALM specific
        self.num_layers = num_layers
        self.comp_per_layer = comp_per_layer
        self.layer_height = layer_height
        
        # Correct way to get centroids for DG0 in FEniCS 2019:
        V_dg = FunctionSpace(mesh, "DG", 0)
        midpoints = V_dg.tabulate_dof_coordinates()
        self.X_mesh, self.Y_mesh = midpoints[:, 0], midpoints[:, 1]
        self.num_elements = len(self.X_mesh)
        
        # Saturation constants
        self.xt = 1.0 + 1.0/ka * np.log((1.0 + (num_components - 1.0)*np.exp(-ka))/num_components)
        self.s0 = -np.log(np.exp(-pp) + 1.0 / (np.exp(0.0) + 1.0)) / pp
        
        # IO Grammars
        self.input_grammar.update_from_names(["x_vars"])
        self.output_grammar.update_from_names(["density"])

    def _map_logic(self, x_vars):
        char_functions = []
        
        if self.mode == 'Free':
            params = x_vars.reshape(self.num_components, 6)
            for i in range(self.num_components):
                p = params[i]
                W = compute_local_characteristic_np(self.X_mesh, self.Y_mesh, p[0], p[1], p[2], p[3], p[4], self.r_gp, method=self.method)
                char_functions.append(W * p[5])
        else:
            # ALM Mode
            params = x_vars.reshape(self.num_components, 3)
            h_fixed = self.layer_height
            theta_fixed = 0.0
            for layer in range(self.num_layers):
                y_fixed = (layer + 0.5) * self.layer_height
                for i in range(self.comp_per_layer):
                    idx = layer * self.comp_per_layer + i
                    p = params[idx]
                    W = compute_local_characteristic_np(self.X_mesh, self.Y_mesh, p[0], y_fixed, p[1], h_fixed, theta_fixed, self.r_gp, method=self.method)
                    char_functions.append(W * p[2])
        
        sum_exp = np.mean(np.exp(self.ka * np.array(char_functions)), axis=0)
        ks_val = (1.0 / self.ka) * np.log(sum_exp)
        return smooth_saturation_np(ks_val, self.ka, self.pp, self.xt, self.s0)

    def _run(self, input_data=None):
        if input_data is not None: self.local_data.update(input_data)
        x_vars = self.local_data["x_vars"].flatten()
        self.local_data["density"] = self._map_logic(x_vars)

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        x = self.local_data["x_vars"].flatten()
        eps = 1e-8
        jac = np.zeros((self.num_elements, len(x)))
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            rho_plus = self._map_logic(x_plus)
            x_minus = x.copy()
            x_minus[i] -= eps
            rho_minus = self._map_logic(x_minus)
            jac[:, i] = (rho_plus - rho_minus) / (2 * eps)
        self.jac = {"density": {"x_vars": jac}}

class GGPPhysicsAdjointDiscipline(Discipline):
    """
    Optimized Physics Discipline with automatic objective scaling.
    """
    def __init__(self, solver, mesh, mesh_area, name="GGP_Physics"):
        super().__init__(name=name)
        self.solver = solver
        self.mesh = mesh
        self.mesh_area = mesh_area
        self.V_dg = FunctionSpace(mesh, "DG", 0)
        self.scale_obj = 1.0
        self.iter = 0
        
        self.input_grammar.update_from_names(["density"])
        self.output_grammar.update_from_names(["compliance", "volume"])

    def _run(self, input_data=None):
        if input_data is not None: self.local_data.update(input_data)
        rho_arr = self.local_data["density"].flatten()
        
        get_working_tape().clear_tape()
        rho_adj = Function(self.V_dg)
        rho_adj.vector()[:] = rho_arr
        rho_tracked = interpolate(rho_adj, self.V_dg)
        
        u = self.solver.solve(rho_tracked)
        j_val = float(self.solver.compute_compliance(u))
        v_val = float(self.solver.compute_volume(rho_tracked))
        
        # Scaling to 1.0 at first iteration
        if self.iter == 0:
            self.scale_obj = 1.0 / j_val
            
        self.local_data["compliance"] = np.array([j_val * self.scale_obj])
        self.local_data["volume"] = np.array([v_val / self.mesh_area])
        
        m = Control(rho_tracked)
        self.dj_drho = compute_gradient(j_val, m).vector().get_local() * self.scale_obj
        self.dv_drho = compute_gradient(v_val, m).vector().get_local() / self.mesh_area
        self.iter += 1

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        self.jac = {
            "compliance": {"density": self.dj_drho.reshape(1, -1)},
            "volume": {"density": self.dv_drho.reshape(1, -1)}
        }
