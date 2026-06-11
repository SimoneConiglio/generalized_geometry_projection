import numpy as np
from gemseo.core.discipline.discipline import Discipline
from ..utils.vectorized_mapping import compute_local_characteristic_np, smooth_saturation_np, compute_local_characteristic_2d_with_grad_np
import dolfin as df
from dolfin import *
from dolfin_adjoint import *
import ufl

class GGPVectorizedGeometryDiscipline(Discipline):
    """
    Highly optimized GGP Geometry Discipline using NumPy vectorization.
    Supports Free (2D: 6-var, 3D: 8-var), 2D ALM (3-var), and 3D ALM (6-var oriented brick).
    """
    def __init__(self, mesh, num_components, mode='Free', method='GP', r_gp=0.5, 
                 ka=10.0, pp=3.0, gammac=3.0, gammav=1.0, name="GGP_Geometry",
                 num_layers=None, comp_per_layer=None, layer_height=None, Ngp=2, **kwargs):
        super().__init__(name=name)
        self.num_components = num_components
        self.dim = mesh.geometry().dim()
        if mode == 'ALM':
            self.mode = '2D_ALM' if self.dim == 2 else '3D_ALM'
        else:
            self.mode = mode
        # Default to AMNA for ALM modes (paper Eq. 40-41) unless explicitly overridden
        if method == 'GP' and mode == 'ALM':
            self.method = 'AMNA'
        else:
            self.method = method
        self.r_gp = r_gp
        self.ka = ka
        self.pp = pp if pp != 3.0 else 10.0  # Use a sharper sigmoid to saturate max density to 1.0
        self.gammac = gammac
        self.gammav = gammav
        self.Ngp = Ngp
        
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
        
        # Pre-calculate Sampling Window Gauss Points (2D only for now)
        if self.dim == 2:
            pts, wts = np.polynomial.legendre.leggauss(Ngp)
            pts = pts * r_gp # Map [-1, 1] to [-R, R]
            wts = wts * r_gp
            gpcx, gpcy = np.meshgrid(pts, pts)
            self.gpc_x_rel = gpcx.flatten()
            self.gpc_y_rel = gpcy.flatten()
            self.gpc_wts = (wts[:, np.newaxis] * wts[np.newaxis, :]).flatten()
            self.gpc_wts_sum = np.sum(self.gpc_wts)
            
            # Expanded coordinates for evaluation: (num_elements, Ngp^2)
            self.X_eval = self.X_mesh[:, np.newaxis] + self.gpc_x_rel[np.newaxis, :]
            self.Y_eval = self.Y_mesh[:, np.newaxis] + self.gpc_y_rel[np.newaxis, :]
            # Flatten for vectorized mapping: (num_elements * Ngp^2,)
            self.X_eval_flat = self.X_eval.flatten()
            self.Y_eval_flat = self.Y_eval.flatten()
        else:
            # 3D: Simplified to centroid for now or can be expanded
            self.X_eval_flat = self.X_mesh
            self.Y_eval_flat = self.Y_mesh
            self.Z_eval_flat = self.Z_mesh
            self.gpc_wts = np.array([1.0])
            self.gpc_wts_sum = 1.0

        # Domain bounds (optional for initial design scaling)
        self.L_domain = kwargs.get('L_domain', self.X_mesh.max() - self.X_mesh.min())
        self.H_domain = kwargs.get('H_domain', self.Y_mesh.max() - self.Y_mesh.min())
        if self.dim == 3:
            self.D_domain = kwargs.get('D_domain', self.Z_mesh.max() - self.Z_mesh.min() if self.Z_mesh is not None else 1.0)

        # Saturation constants
        self.xt = kwargs.get('xt', 1.0 + 1.0/ka * np.log((1.0 + (num_components - 1.0)*np.exp(-ka))/num_components))
        self.s0 = -np.log(np.exp(-self.pp) + 1.0 / (np.exp(0.0) + 1.0)) / self.pp
        
        # Design variable bounds for scaling
        self.lb = kwargs.get('lb', None)
        self.ub = kwargs.get('ub', None)
        
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
                    # Evaluate at all Gauss points in sampling window
                    W_gp = compute_local_characteristic_np(self.X_eval_flat, self.Y_eval_flat, p[0], p[1], p[2], p[3], p[4], self.r_gp, method=self.method)
                    # Reshape to (num_elements, Ngp^2) and integrate
                    W_el = np.sum(W_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                    char_functions.append(W_el * (p[5]**power))
            else:
                params = x_vars.reshape(self.num_components, 8)
                for i in range(self.num_components):
                    p = params[i]
                    W = compute_local_characteristic_np(
                        self.X_mesh, self.Y_mesh, p[0], p[1], p[3], p[4], p[5], self.r_gp, 
                        method=self.method, Z_mesh=self.Z_mesh, Z=p[2], P=p[6]
                    )
                    char_functions.append(W * (p[7]**power))
        elif self.mode == '2D_ALM':
            is_extended = len(x_vars) == (self.num_components * 4 + 2)
            if is_extended:
                y0 = x_vars[-2]
                theta0 = x_vars[-1]
                params = x_vars[:-2].reshape(self.num_components, 4)
                num_vars = 4
            else:
                y0 = 0.0
                theta0 = 0.0
                params = x_vars.reshape(self.num_components, 3)
                num_vars = 3
                
            for layer in range(self.num_layers):
                y_fixed = (layer + 0.5) * self.layer_height
                for i in range(self.comp_per_layer):
                    idx = layer * self.comp_per_layer + i
                    p = params[idx]
                    Xc, width, Mc = p[0], p[1], p[2]
                    hc = p[3] if is_extended else self.layer_height
                    
                    Xc_g = Xc * np.cos(theta0) - y_fixed * np.sin(theta0)
                    Yc_g = y0 + Xc * np.sin(theta0) + y_fixed * np.cos(theta0)
                    
                    W_gp = compute_local_characteristic_np(
                        self.X_eval_flat, self.Y_eval_flat, Xc_g, Yc_g, width, hc, theta0, self.r_gp, method=self.method
                    )
                    W_el = np.sum(W_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                    char_functions.append(W_el * (Mc**power))
        elif self.mode == '3D_ALM':
            # Oriented Brick in Layered 3D
            # variables per component: [Xc, Yc, L, W, Theta, Mc]
            params = x_vars.reshape(self.num_components, 6)
            h_fixed = self.layer_height
            for layer in range(self.num_layers):
                z_fixed = (layer + 0.5) * self.layer_height
                for i in range(self.comp_per_layer):
                    idx = layer * self.comp_per_layer + i
                    p = params[idx]
                    W = compute_local_characteristic_np(
                        self.X_mesh, self.Y_mesh, p[0], p[1], p[2], h_fixed, p[4], self.r_gp, 
                        method=self.method, Z_mesh=self.Z_mesh, Z=z_fixed, W_width=p[3]
                    )
                    char_functions.append(W * (p[5]**power))
        
        sum_exp = np.mean(np.exp(self.ka * np.array(char_functions)), axis=0)
        ks_val = (1.0 / self.ka) * np.log(sum_exp)
        return smooth_saturation_np(ks_val, self.ka, self.pp, self.xt, self.s0)

    def _map_logic_with_grad(self, x_vars, power):
        char_functions = []
        char_grads = [] # List of grads per component: [gradX, gradY, gradL, gradh, gradM, gradT]
        
        if self.mode == 'Free' and self.dim == 2:
            params = x_vars.reshape(self.num_components, 6)
            for i in range(self.num_components):
                p = params[i]
                # Evaluate W and its gradients at all Gauss points
                W_gp, dWdX_gp, dWdY_gp, dWdL_gp, dWdh_gp, dWdT_gp = compute_local_characteristic_2d_with_grad_np(
                    self.X_eval_flat, self.Y_eval_flat, p[0], p[1], p[2], p[3], p[4], self.r_gp, method=self.method
                )
                
                # Integrate over sampling window
                W_el = np.sum(W_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                dWdX_el = np.sum(dWdX_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                dWdY_el = np.sum(dWdY_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                dWdL_el = np.sum(dWdL_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                dWdh_el = np.sum(dWdh_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                dWdT_el = np.sum(dWdT_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                
                V_el = W_el * (p[5]**power)
                char_functions.append(V_el)
                
                # Gradients wrt parameters of component i
                # dV/dX = dW/dX * m^p
                # dV/dM = W * p * m^(p-1)
                m_p = p[5]**power
                m_p_minus_1 = power * (p[5]**(power - 1.0)) if power > 0 else 0.0
                
                # Store gradients for component i: [dX, dY, dL, dh, dT, dM]
                # Shape: (6, num_elements)
                char_grads.append([
                    dWdX_el * m_p, dWdY_el * m_p, dWdL_el * m_p, 
                    dWdh_el * m_p, dWdT_el * m_p, W_el * m_p_minus_1
                ])
            num_vars = 6
        elif self.mode == '2D_ALM':
            is_extended = len(x_vars) == (self.num_components * 4 + 2)
            if is_extended:
                y0 = x_vars[-2]
                theta0 = x_vars[-1]
                params = x_vars[:-2].reshape(self.num_components, 4)
                num_vars = 4
                dWdy0_total = np.zeros(self.num_elements)
                dWdt0_total = np.zeros(self.num_elements)
            else:
                y0 = 0.0
                theta0 = 0.0
                params = x_vars.reshape(self.num_components, 3)
                num_vars = 3
                
            for layer in range(self.num_layers):
                y_fixed = (layer + 0.5) * self.layer_height
                for i in range(self.comp_per_layer):
                    idx = layer * self.comp_per_layer + i
                    p = params[idx]
                    Xc, width, Mc = p[0], p[1], p[2]
                    hc = p[3] if is_extended else self.layer_height
                    
                    Xc_g = Xc * np.cos(theta0) - y_fixed * np.sin(theta0)
                    Yc_g = y0 + Xc * np.sin(theta0) + y_fixed * np.cos(theta0)
                    
                    W_gp, dWdX_gp, dWdY_gp, dWdL_gp, dWdh_gp, dWdT_gp = compute_local_characteristic_2d_with_grad_np(
                        self.X_eval_flat, self.Y_eval_flat, Xc_g, Yc_g, width, hc, theta0, self.r_gp, method=self.method
                    )
                    
                    # Chain rule for gradients wrt original variables
                    an_Xc_gp = dWdX_gp * np.cos(theta0) + dWdY_gp * np.sin(theta0)
                    
                    W_el = np.sum(W_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                    dWdX_el = np.sum(an_Xc_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                    dWdL_el = np.sum(dWdL_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                    
                    V_el = W_el * (Mc**power)
                    char_functions.append(V_el)
                    
                    m_p = Mc**power
                    m_p_minus_1 = power * (Mc**(power - 1.0)) if power > 0 else 0.0
                    
                    if is_extended:
                        dWdh_el = np.sum(dWdh_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                        
                        an_y0_gp = dWdY_gp
                        an_t0_gp = dWdX_gp * (-Xc * np.sin(theta0) - y_fixed * np.cos(theta0)) + \
                                   dWdY_gp * (Xc * np.cos(theta0) - y_fixed * np.sin(theta0)) + dWdT_gp
                        
                        dWdy0_el = np.sum(an_y0_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                        dWdt0_el = np.sum(an_t0_gp.reshape(self.num_elements, -1) * self.gpc_wts, axis=1) / self.gpc_wts_sum
                        
                        char_grads.append([
                            dWdX_el * m_p, dWdL_el * m_p, W_el * m_p_minus_1, dWdh_el * m_p,
                            dWdy0_el * m_p, dWdt0_el * m_p
                        ])
                    else:
                        char_grads.append([
                            dWdX_el * m_p, dWdL_el * m_p, W_el * m_p_minus_1
                        ])
        else:
            # Fallback to finite difference or implement other modes
            return self._map_logic(x_vars, power), None

        # Aggregation
        sum_exp = np.mean(np.exp(self.ka * np.array(char_functions)), axis=0)
        ks_val = (1.0 / self.ka) * np.log(sum_exp)
        
        # dKS / dVi = (1/N * exp(ka*Vi)) / sum_exp
        dKS_dV = np.exp(self.ka * np.array(char_functions)) / (len(char_functions) * sum_exp)
        
        # Saturation
        # s = ( -log(exp(-pa) + 1/(exp(pa*xs/a)+1)) / pa - s0 ) / (1-s0)
        xs, a, pa = ks_val, self.xt, self.pp
        inner_exp = np.exp((pa * xs) / a)
        rho = (-np.log(np.exp(-pa) + 1.0 / (inner_exp + 1.0)) / pa - self.s0) / (1.0 - self.s0)
        
        # ds/dxs = (exp(pa*xs/a) * 1 / (exp(pa*xs/a)+1)^2) / (a * (exp(-pa) + 1/(exp(pa*xs/a)+1))) / (1-s0)
        ds_dxs = (inner_exp / (inner_exp + 1.0)**2) / (a * (np.exp(-pa) + 1.0/(inner_exp + 1.0))) / (1.0 - self.s0)
        
        # Total Jacobian: drho/dp_ij = (ds/dxs) * (dKS/dV_i) * (dVi/dp_ij)
        if self.mode == '2D_ALM' and len(x_vars) == (self.num_components * 4 + 2):
            total_jac = np.zeros((self.num_elements, self.num_components * num_vars + 2))
            for i in range(self.num_components):
                for j in range(num_vars):
                    total_jac[:, i*num_vars + j] = ds_dxs * dKS_dV[i] * char_grads[i][j]
                
                # Accumulate global variables gradient contributions
                total_jac[:, -2] += ds_dxs * dKS_dV[i] * char_grads[i][4] # y0
                total_jac[:, -1] += ds_dxs * dKS_dV[i] * char_grads[i][5] # theta0
        else:
            total_jac = np.zeros((self.num_elements, self.num_components * num_vars))
            for i in range(self.num_components):
                for j in range(num_vars):
                    total_jac[:, i*num_vars + j] = ds_dxs * dKS_dV[i] * char_grads[i][j]
                    
        return rho, total_jac

    def _run(self, input_data=None):
        import time
        start = time.time()
        if input_data is not None: self.local_data.update(input_data)
        x_vars = self.local_data["x_vars"].flatten()
        
        if self.lb is not None and self.ub is not None:
            x_vars_unscaled = self.lb + x_vars * (self.ub - self.lb)
            rho_E, jac_E_unscaled = self._map_logic_with_grad(x_vars_unscaled, self.gammac)
            rho_V, jac_V_unscaled = self._map_logic_with_grad(x_vars_unscaled, self.gammav)
            
            # Scale Jacobians wrt the scaled variables [0, 1]
            # Jacobian wrt x_scaled is: d_rho / d_x_scaled = (d_rho / d_x_unscaled) * (ub - lb)
            # Since self.ub and self.lb are 1D arrays of size 108, we can do element-wise multiplication
            # across the columns of the Jacobian matrix of shape (1800, 108)
            jac_E = jac_E_unscaled * (self.ub - self.lb)
            jac_V = jac_V_unscaled * (self.ub - self.lb)
        else:
            rho_E, jac_E = self._map_logic_with_grad(x_vars, self.gammac)
            rho_V, jac_V = self._map_logic_with_grad(x_vars, self.gammav)
        
        self.local_data["rho_E"] = rho_E
        self.local_data["rho_V"] = rho_V
        self.jac_E = jac_E
        self.jac_V = jac_V
        # print(f"DEBUG: Geometry _run took {time.time()-start:.4f}s")

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        import time
        start = time.time()
        self.jac = {"rho_E": {"x_vars": self.jac_E}, "rho_V": {"x_vars": self.jac_V}}
        print(f"DEBUG: Geometry Jacobian took {time.time()-start:.4f}s")

class GGPPhysicsAdjointDiscipline(Discipline):
    """
    Optimized Physics Discipline with Log Scaling for Compliance.
    Supports 2D and 3D.
    """
    def __init__(self, solver, mesh, mesh_area, volfrac, name="GGP_Physics", validator=None):
        super().__init__(name=name)
        self.solver = solver
        self.mesh = mesh
        self.mesh_area = mesh_area
        self.volfrac = volfrac
        self.V_dg = df.FunctionSpace(mesh, "DG", 0)
        self.validator = validator
        self.num_elements = mesh.num_cells()
        
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
        
        # LOG SCALING: f0 = log(c + 1)
        self.local_data["compliance"] = np.array([np.log(c_val + 1.0)])
        self.local_data["volume"] = np.array([(v_val - self.volfrac) / self.volfrac * 100.0])
        
        # Gradients
        self.dj_drhoE = compute_gradient(j_functional, Control(rho_E_tracked)).vector().get_local() / (c_val + 1.0)
        self.dv_drhoV = compute_gradient(v_functional, Control(rho_V_tracked)).vector().get_local() * (100.0 / (self.volfrac * self.mesh_area))

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        self.jac = {
            "compliance": {"rho_E": self.dj_drhoE.reshape(1, -1), "rho_V": np.zeros((1, self.num_elements))},
            "volume": {"rho_E": np.zeros((1, self.num_elements)), "rho_V": self.dv_drhoV.reshape(1, -1)}
        }

class GGPPhysicsFastDiscipline(Discipline):
    """
    High-performance Physics Discipline using Fast Assembly (petsc4py).
    Pre-computes element matrices and assembles the global matrix manually.
    Supports general adjoint sensitivities.
    """
    def __init__(self, solver, mesh, mesh_area, volfrac, name="GGP_Physics_Fast"):
        super().__init__(name=name)
        self.solver = solver
        self.mesh = mesh
        self.mesh_area = mesh_area
        self.volfrac = volfrac
        self.num_elements = mesh.num_cells()
        self.V_u = solver.V_u
        
        # Pre-compute element stiffness and DOF maps
        self.ke_ref = solver.get_unit_element_stiffness()
        self.dm = self.V_u.dofmap()
        self.cell_dofs = [self.dm.cell_dofs(i) for i in range(self.num_elements)]
        
        # Extract Fixed DOFs for BCs
        self.fixed_dofs = []
        for bc in solver.bc:
            self.fixed_dofs.extend(bc.get_boundary_values().keys())
        self.fixed_dofs = sorted(list(set(self.fixed_dofs)))
        
        # Pre-assemble Load Vector (F)
        # We use FEniCS to get the base load vector
        L = ufl.dot(solver.L_rhs_vec, TestFunction(self.V_u)) * solver.ds_load(1)
        self.f_vec = assemble(L).get_local()
        # Apply BCs to F (zero out fixed DOFs)
        self.f_vec[self.fixed_dofs] = 0.0
        
        print(f"DEBUG: Max F magnitude: {np.max(np.abs(self.f_vec))}")
        print(f"DEBUG: Ke_ref norm: {np.linalg.norm(self.ke_ref)}")
        
        self.input_grammar.update_from_names(["rho_E", "rho_V"])
        self.output_grammar.update_from_names(["compliance", "volume"])
        self.last_u = None

    def _run(self, input_data=None):
        import time
        start = time.time()
        if input_data is not None: self.local_data.update(input_data)
        rho_E = self.local_data["rho_E"].flatten()
        rho_V = self.local_data["rho_V"].flatten()
        
        # Penalize Young's Modulus
        Emin = float(self.solver.Emin)
        E0 = float(self.solver.E0)
        E_vals = Emin + rho_E * (E0 - Emin)
        
        # Assemble Global Stiffness Matrix K
        from scipy.sparse import coo_matrix
        
        dofs_per_cell = self.ke_ref.shape[0]
        rows = np.repeat(self.cell_dofs, dofs_per_cell).reshape(self.num_elements, dofs_per_cell, dofs_per_cell)
        cols = np.tile(self.cell_dofs, (1, dofs_per_cell)).reshape(self.num_elements, dofs_per_cell, dofs_per_cell)
        data = np.outer(E_vals, self.ke_ref).reshape(self.num_elements, dofs_per_cell, dofs_per_cell)
        
        K_global = coo_matrix((data.flatten(), (rows.flatten(), cols.flatten())), shape=(self.V_u.dim(), self.V_u.dim())).tocsr()
        
        # Apply Dirichlet BCs directly in CSR format to bypass slow LIL conversions
        # 1. Zero out columns corresponding to fixed DOFs
        col_mask = np.isin(K_global.indices, self.fixed_dofs)
        K_global.data[col_mask] = 0.0
        
        # 2. Zero out rows and set diagonal elements to 1.0
        for dof in self.fixed_dofs:
            dof_start = K_global.indptr[dof]
            dof_end = K_global.indptr[dof+1]
            K_global.data[dof_start:dof_end] = 0.0
            col_indices = K_global.indices[dof_start:dof_end]
            diag_idx = np.where(col_indices == dof)[0]
            if len(diag_idx) > 0:
                K_global.data[dof_start + diag_idx[0]] = 1.0
        
        # Solve K U = F
        from scipy.sparse.linalg import spsolve
        u_vec = spsolve(K_global, self.f_vec)
        self.last_u = u_vec
        
        # Compliance C = F^T U
        compliance = np.dot(self.f_vec, u_vec)
        
        # Volume V
        volume = np.sum(rho_V) * (self.mesh_area / self.num_elements) / self.mesh_area
        
        self.local_data["compliance"] = np.array([np.log(compliance + 1.0)])
        self.local_data["volume"] = np.array([(volume - self.volfrac) / self.volfrac * 100.0])
        
        # --- Gradients ---
        dE_drho = (E0 - Emin)
        grad_C = np.zeros(self.num_elements)
        for i in range(self.num_elements):
            u_e = u_vec[self.cell_dofs[i]]
            grad_C[i] = -dE_drho * np.dot(u_e, np.dot(self.ke_ref, u_e))
        
        self.dj_drhoE = grad_C / (compliance + 1.0)
        self.dv_drhoV = np.ones(self.num_elements) * (100.0 / (self.volfrac * self.num_elements))
        print(f"DEBUG: Physics _run took {time.time()-start:.4f}s")

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        import time
        start = time.time()
        self.jac = {
            "compliance": {"rho_E": self.dj_drhoE.reshape(1, -1), "rho_V": np.zeros((1, self.num_elements))},
            "volume": {"rho_E": np.zeros((1, self.num_elements)), "rho_V": self.dv_drhoV.reshape(1, -1)}
        }
        # print(f"DEBUG: Physics Jacobian took {time.time()-start:.4f}s")
