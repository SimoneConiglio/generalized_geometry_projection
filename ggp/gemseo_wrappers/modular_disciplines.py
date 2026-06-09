import numpy as np
from gemseo.core.discipline.discipline import Discipline
from ..utils.vectorized_mapping import compute_local_characteristic_np, smooth_saturation_np
import dolfin as df
from dolfin_adjoint import *

class GGPVectorizedGeometryDiscipline(Discipline):
    """
    Highly optimized GGP Geometry Discipline using NumPy vectorization.
    Supports both Free (2D: 6-var, 3D: 8-var) and ALM mapping strategies.
    """
    def __init__(self, mesh, num_components, mode='Free', method='GP', r_gp=0.5, 
                 ka=10.0, pp=100.0, gammac=3.0, gammav=1.0, name="GGP_Geometry",
                 num_layers=None, comp_per_layer=None, layer_height=None):
        super().__init__(name=name)
        self.num_components = num_components
        self.mode = mode
        self.method = method
        self.r_gp = r_gp
        self.ka = ka
        self.pp = pp
        self.gammac = gammac
        self.gammav = gammav
        self.dim = mesh.geometry().dim()
        
        # ALM specific
        self.num_layers = num_layers
        self.comp_per_layer = comp_per_layer
        self.layer_height = layer_height
        
        # Pre-calculate element centroids
        V_dg = df.FunctionSpace(mesh, "DG", 0)
        midpoints = V_dg.tabulate_dof_coordinates()
        self.X_mesh, self.Y_mesh = midpoints[:, 0], midpoints[:, 1]
        self.Z_mesh = midpoints[:, 2] if self.dim == 3 else None
        self.num_elements = len(self.X_mesh)
        
        # Saturation constants
        self.xt = 1.0 + 1.0/ka * np.log((1.0 + (num_components - 1.0)*np.exp(-ka))/num_components)
        self.s0 = -np.log(np.exp(-pp) + 1.0 / (np.exp(0.0) + 1.0)) / pp
        
        # IO Grammars
        self.input_grammar.update_from_names(["x_vars"])
        self.output_grammar.update_from_names(["rho_E", "rho_V"])

    def _map_logic(self, x_vars, power):
        char_functions = []
        
        if self.mode == 'Free':
            if self.dim == 2:
                params = x_vars.reshape(self.num_components, 6)
                for i in range(self.num_components):
                    p = params[i]
                    W = compute_local_characteristic_np(self.X_mesh, self.Y_mesh, p[0], p[1], p[2], p[3], p[4], self.r_gp, method=self.method)
                    char_functions.append(W * (p[5]**power))
            else:
                # 3D: [Xc, Yc, Zc, L, h, Theta, Phi, Mc] (8 variables)
                params = x_vars.reshape(self.num_components, 8)
                for i in range(self.num_components):
                    p = params[i]
                    W = compute_local_characteristic_np(
                        self.X_mesh, self.Y_mesh, p[0], p[1], p[3], p[4], p[5], self.r_gp, 
                        method=self.method, Z_mesh=self.Z_mesh, Z=p[2], P=p[6]
                    )
                    char_functions.append(W * (p[7]**power))
        else:
            # ALM Mode (2D only for now)
            params = x_vars.reshape(self.num_components, 3)
            h_fixed = self.layer_height
            theta_fixed = 0.0
            for layer in range(self.num_layers):
                y_fixed = (layer + 0.5) * self.layer_height
                for i in range(self.comp_per_layer):
                    idx = layer * self.comp_per_layer + i
                    p = params[idx]
                    W = compute_local_characteristic_np(self.X_mesh, self.Y_mesh, p[0], y_fixed, p[1], h_fixed, theta_fixed, self.r_gp, method=self.method)
                    char_functions.append(W * (p[2]**power))
        
        sum_exp = np.mean(np.exp(self.ka * np.array(char_functions)), axis=0)
        ks_val = (1.0 / self.ka) * np.log(sum_exp)
        return smooth_saturation_np(ks_val, self.ka, self.pp, self.xt, self.s0)

    def _run(self, input_data=None):
        if input_data is not None: self.local_data.update(input_data)
        x_vars = self.local_data["x_vars"].flatten()
        self.local_data["rho_E"] = self._map_logic(x_vars, self.gammac)
        self.local_data["rho_V"] = self._map_logic(x_vars, self.gammav)

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        x = self.local_data["x_vars"].flatten()
        eps = 1e-8
        jac_E = np.zeros((self.num_elements, len(x)))
        jac_V = np.zeros((self.num_elements, len(x)))
        for i in range(len(x)):
            x_plus = x.copy(); x_plus[i] += eps
            x_minus = x.copy(); x_minus[i] -= eps
            jac_E[:, i] = (self._map_logic(x_plus, self.gammac) - self._map_logic(x_minus, self.gammac)) / (2 * eps)
            jac_V[:, i] = (self._map_logic(x_plus, self.gammav) - self._map_logic(x_minus, self.gammav)) / (2 * eps)
        self.jac = {"rho_E": {"x_vars": jac_E}, "rho_V": {"x_vars": jac_V}}

class GGPPhysicsAdjointDiscipline(Discipline):
    """
    Optimized Physics Discipline with Log Scaling for Compliance.
    Supports 2D and 3D.
    """
    def __init__(self, solver, mesh, mesh_area, volfrac, name="GGP_Physics"):
        super().__init__(name=name)
        self.solver = solver
        self.mesh = mesh
        self.mesh_area = mesh_area
        self.volfrac = volfrac
        self.V_dg = df.FunctionSpace(mesh, "DG", 0)
        
        self.input_grammar.update_from_names(["rho_E", "rho_V"])
        self.output_grammar.update_from_names(["compliance", "volume"])

    def _run(self, input_data=None):
        if input_data is not None: self.local_data.update(input_data)
        rho_E_arr = self.local_data["rho_E"].flatten()
        rho_V_arr = self.local_data["rho_V"].flatten()
        
        get_working_tape().clear_tape()
        
        rho_E_adj = df.Function(self.V_dg)
        rho_E_adj.vector()[:] = rho_E_arr
        rho_E_tracked = interpolate(rho_E_adj, self.V_dg)
        
        rho_V_adj = df.Function(self.V_dg)
        rho_V_adj.vector()[:] = rho_V_arr
        rho_V_tracked = interpolate(rho_V_adj, self.V_dg)
        
        u = self.solver.solve(rho_E_tracked)
        j_functional = self.solver.compute_compliance(u)
        v_functional = self.solver.compute_volume(rho_V_tracked)
        
        c_val = float(j_functional)
        v_val = float(v_functional) / self.mesh_area
        
        # LOG SCALING
        self.local_data["compliance"] = np.array([np.log(c_val + 1.0)])
        self.local_data["volume"] = np.array([(v_val - self.volfrac) / self.volfrac * 100.0])
        
        # Gradients
        self.dj_drhoE = compute_gradient(j_functional, Control(rho_E_tracked)).vector().get_local() / (c_val + 1.0)
        self.dv_drhoV = compute_gradient(v_functional, Control(rho_V_tracked)).vector().get_local() * (100.0 / (self.volfrac * self.mesh_area))

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        num_elements = len(self.dj_drhoE)
        self.jac = {
            "compliance": {"rho_E": self.dj_drhoE.reshape(1, -1), "rho_V": np.zeros((1, num_elements))},
            "volume": {"rho_E": np.zeros((1, num_elements)), "rho_V": self.dv_drhoV.reshape(1, -1)}
        }
