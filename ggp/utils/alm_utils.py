import numpy as np

def create_alm_overhang_constraints(num_layers, comp_per_layer, layer_height, alpha_deg, extended=False, x_vars=None):
    """
    Computes the linear overhang constraints for 2D ALM.
    If extended=True and x_vars is provided, returns nonlinear constraints and jacobian considering support inclination theta0.
    Returns:
        If x_vars is None: A, b (linear constraints A*x - b <= 0)
        If x_vars is provided: cons, jac (nonlinear constraints cons <= 0)
    """
    if extended and x_vars is not None:
        theta0 = x_vars[-1]
        alpha = np.deg2rad(alpha_deg)
        
        delta_plus = layer_height * np.tan(alpha + theta0)
        delta_minus = layer_height * np.tan(alpha - theta0)
        
        dd_plus = layer_height / (np.cos(alpha + theta0)**2)
        dd_minus = -layer_height / (np.cos(alpha - theta0)**2)
        
        num_comp = num_layers * comp_per_layer
        num_vars_per_comp = 4
        num_vars = num_comp * num_vars_per_comp + 2
        num_interfaces = num_layers - 1
        num_cons = num_interfaces * comp_per_layer * 2
        
        cons = np.zeros(num_cons)
        jac = np.zeros((num_cons, num_vars))
        
        row = 0
        for layer in range(num_interfaces):
            for k in range(comp_per_layer):
                idx_k = layer * comp_per_layer + k
                idx_kp1 = (layer + 1) * comp_per_layer + k
                
                xc_k, l_k = x_vars[idx_k*4], x_vars[idx_k*4+1]
                xc_kp1, l_kp1 = x_vars[idx_kp1*4], x_vars[idx_kp1*4+1]
                
                # Constraint 1: Xc_{i+1} - Xc_i + 0.5*(L_{i+1} - L_i) - delta_plus <= 0
                cons[row] = (xc_kp1 - xc_k) + 0.5*(l_kp1 - l_k) - delta_plus
                jac[row, idx_kp1*4] = 1.0
                jac[row, idx_k*4] = -1.0
                jac[row, idx_kp1*4+1] = 0.5
                jac[row, idx_k*4+1] = -0.5
                jac[row, -1] = -dd_plus
                row += 1
                
                # Constraint 2: -Xc_{i+1} + Xc_i + 0.5*(L_{i+1} - L_i) - delta_minus <= 0
                cons[row] = -(xc_kp1 - xc_k) + 0.5*(l_kp1 - l_k) - delta_minus
                jac[row, idx_kp1*4] = -1.0
                jac[row, idx_k*4] = 1.0
                jac[row, idx_kp1*4+1] = 0.5
                jac[row, idx_k*4+1] = -0.5
                jac[row, -1] = -dd_minus
                row += 1
                
        return cons, jac
        
    delta = layer_height * np.tan(np.deg2rad(alpha_deg))
    num_comp = num_layers * comp_per_layer
    num_vars_per_comp = 4 if extended else 3
    num_vars = num_comp * num_vars_per_comp + (2 if extended else 0)
    num_interfaces = num_layers - 1
    num_cons = num_interfaces * comp_per_layer * 2
    
    A = np.zeros((num_cons, num_vars))
    b = np.ones(num_cons) * delta
    
    row = 0
    for layer in range(num_interfaces):
        for k in range(comp_per_layer):
            idx_k = layer * comp_per_layer + k
            idx_kp1 = (layer + 1) * comp_per_layer + k
            
            # Constraint 1: Xc_{k+1} - Xc_k + 0.5*L_{k+1} - 0.5*L_k <= delta
            A[row, num_vars_per_comp * idx_k] = -1.0
            A[row, num_vars_per_comp * idx_k + 1] = -0.5
            A[row, num_vars_per_comp * idx_kp1] = 1.0
            A[row, num_vars_per_comp * idx_kp1 + 1] = 0.5
            row += 1
            
            # Constraint 2: -Xc_{k+1} + Xc_k + 0.5*L_{k+1} - 0.5*L_k <= delta
            A[row, num_vars_per_comp * idx_k] = 1.0
            A[row, num_vars_per_comp * idx_k + 1] = -0.5
            A[row, num_vars_per_comp * idx_kp1] = -1.0
            A[row, num_vars_per_comp * idx_kp1 + 1] = 0.5
            row += 1
            
    return A, b

def create_alm_3d_overhang_constraints(num_layers, comp_per_layer, layer_height, alpha_deg, x_vars):
    """
    Computes the 16 nonlinear overhang constraints per component.
    Each component in layer k+1 must be supported by the component in layer k.
    x_vars: array of [Xc, Yc, L, W, Theta, Mc] for all components.
    """
    num_comp = num_layers * comp_per_layer
    num_vars = num_comp * 6
    delta = layer_height * np.tan(np.deg2rad(alpha_deg))
    params = x_vars.reshape(num_comp, 6)
    
    constraints = []
    
    for layer in range(num_layers - 1):
        for k in range(comp_per_layer):
            idx_k = layer * comp_per_layer + k
            idx_kp1 = (layer + 1) * comp_per_layer + k
            
            p_k = params[idx_k]     # [Xc_k, Yc_k, L_k, W_k, T_k, M_k]
            p_kp1 = params[idx_kp1] # [Xc_kp1, Yc_kp1, L_kp1, W_kp1, T_kp1, M_kp1]
            
            # 1. Compute global coords of the 4 vertices in layer k+1
            L1, W1, T1 = p_kp1[2], p_kp1[3], p_kp1[4]
            c1, s1 = np.cos(T1), np.sin(T1)
            
            # Local vertex offsets
            offsets = np.array([
                [ L1/2,  W1/2],
                [-L1/2,  W1/2],
                [-L1/2, -W1/2],
                [ L1/2, -W1/2]
            ])
            
            # Rotation matrix for T1
            R1 = np.array([[c1, -s1], [s1, c1]])
            v_global = p_kp1[:2] + offsets @ R1.T # (4, 2)
            
            # 2. Transform to local frame of layer k
            Xc0, Yc0, L0, W0, T0 = p_k[0], p_k[1], p_k[2], p_k[3], p_k[4]
            c0, s0 = np.cos(T0), np.sin(T0)
            R0_inv = np.array([[c0, s0], [-s0, c0]])
            
            v_local = (v_global - p_k[:2]) @ R0_inv.T # (4, 2)
            
            # 3. Formulate inequalities: |x_loc| - (L0/2 + delta) <= 0
            # We split into 4 inequalities per vertex (16 total)
            for i in range(4):
                # x-coord bounds
                constraints.append(v_local[i, 0] - (L0/2 + delta))
                constraints.append(-v_local[i, 0] - (L0/2 + delta))
                # y-coord bounds
                constraints.append(v_local[i, 1] - (W0/2 + delta))
                constraints.append(-v_local[i, 1] - (W0/2 + delta))
                
    return np.array(constraints)

def compute_alm_3d_overhang_jacobian(num_layers, comp_per_layer, layer_height, alpha_deg, x_vars):
    """
    Finite difference Jacobian for the nonlinear 3D overhang constraints.
    Given the small number of variables (6 per component) and local dependencies,
    this is efficient.
    """
    eps = 1e-7
    f0 = create_alm_3d_overhang_constraints(num_layers, comp_per_layer, layer_height, alpha_deg, x_vars)
    num_cons = len(f0)
    num_vars = len(x_vars)
    jac = np.zeros((num_cons, num_vars))
    
    # Each constraint only depends on 12 variables (layer k and layer k+1)
    # We can optimize this by only perturbing the relevant variables
    for layer in range(num_layers - 1):
        for k in range(comp_per_layer):
            idx_k = (layer * comp_per_layer + k) * 6
            idx_kp1 = ((layer + 1) * comp_per_layer + k) * 6
            
            relevant_indices = list(range(idx_k, idx_k + 6)) + list(range(idx_kp1, idx_kp1 + 6))
            
            # Find which rows in the constraints array correspond to this component pair
            # Total components pairs: (N-1)*K
            # Constraints per pair: 16
            row_start = (layer * comp_per_layer + k) * 16
            
            for v_idx in relevant_indices:
                x_plus = x_vars.copy()
                x_plus[v_idx] += eps
                f_plus = create_alm_3d_overhang_constraints(num_layers, comp_per_layer, layer_height, alpha_deg, x_plus)
                jac[row_start : row_start + 16, v_idx] = (f_plus[row_start : row_start + 16] - f0[row_start : row_start + 16]) / eps
                
    return jac
