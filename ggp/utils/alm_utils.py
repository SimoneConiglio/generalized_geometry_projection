import numpy as np


def _xc_idx(layer, comp, num_layers):
    """Global variable index for Xc_{layer, comp} in the F-order interleaved layout.

    The mapper reshapes Xc with order='F' to (nY, np_val), so the flat index of
    element [layer, comp] is comp*nY + layer, giving global x_vars index
    2*(comp*num_layers + layer).
    """
    return 2 * (comp * num_layers + layer)


def _l_idx(layer, comp, num_layers):
    """Global variable index for L_{layer, comp} in the F-order interleaved layout."""
    return 2 * (comp * num_layers + layer) + 1


def create_alm_overhang_constraints(num_layers, comp_per_layer, layer_height, alpha_deg,
                                    extended=False, x_vars=None):
    """Linear overhang constraints for 2-D ALM (interleaved variable layout).

    Variable layout expected:
      [Xc_0_0, L_0_0, Xc_0_1, L_0_1, ..., Xc_{nY-1}_{np-1}, L_{nY-1}_{np-1},
       h_0..h_{np-1},  Mc_0..Mc_{np-1},  y0, theta0]

    Constraint (per layer-interface per component):
      Right overhang: Xc_{k+1,j} + L_{k+1,j}/2 - Xc_{k,j} - L_{k,j}/2 <= delta
      Left  overhang: Xc_{k,j}   - L_{k,j}/2   - Xc_{k+1,j} + L_{k+1,j}/2 <= delta
    where delta = layer_height * tan(alpha_deg).

    If x_vars is provided the constraint is evaluated at that point (for nonlinear
    theta0 adjustment); otherwise pure linear A, b is returned.
    """
    nY = num_layers
    np_val = comp_per_layer
    n_xl = 2 * nY * np_val          # size of [Xc, L] block
    n_tot = n_xl + 2 * np_val + 2   # total vars (h, Mc, y0, theta0)
    num_interfaces = nY - 1
    num_cons = num_interfaces * np_val * 2

    if x_vars is not None and len(x_vars) >= n_tot:
        # Nonlinear version: adjust delta for theta0
        theta0 = float(x_vars[n_xl + 2*np_val + 1])
        alpha  = np.deg2rad(alpha_deg)
        delta_r = layer_height * np.tan(np.clip(alpha + theta0, -np.pi/2 + 1e-6, np.pi/2 - 1e-6))
        delta_l = layer_height * np.tan(np.clip(alpha - theta0, -np.pi/2 + 1e-6, np.pi/2 - 1e-6))
        dd_r_dt = layer_height / np.cos(np.clip(alpha + theta0, -np.pi/2 + 1e-6, np.pi/2 - 1e-6))**2
        dd_l_dt = -layer_height / np.cos(np.clip(alpha - theta0, -np.pi/2 + 1e-6, np.pi/2 - 1e-6))**2
    else:
        delta = layer_height * np.tan(np.deg2rad(alpha_deg))
        delta_r = delta_l = delta
        dd_r_dt = dd_l_dt = 0.0

    A = np.zeros((num_cons, n_tot))
    b = np.zeros(num_cons)

    row = 0
    for layer in range(num_interfaces):
        for j in range(np_val):
            ix_k   = _xc_idx(layer,     j, nY)
            il_k   = _l_idx(layer,      j, nY)
            ix_k1  = _xc_idx(layer + 1, j, nY)
            il_k1  = _l_idx(layer + 1,  j, nY)

            # Constraint 1 (right overhang): Xc_{k+1} + L_{k+1}/2 - Xc_k - L_k/2 <= delta_r
            A[row, ix_k1] =  1.0
            A[row, il_k1] =  0.5
            A[row, ix_k]  = -1.0
            A[row, il_k]  = -0.5
            if x_vars is not None:
                A[row, n_xl + 2*np_val + 1] = -dd_r_dt
            b[row] = delta_r
            row += 1

            # Constraint 2 (left overhang): Xc_k - L_k/2 - Xc_{k+1} + L_{k+1}/2 <= delta_l
            A[row, ix_k]  =  1.0
            A[row, il_k]  = -0.5
            A[row, ix_k1] = -1.0
            A[row, il_k1] =  0.5
            if x_vars is not None:
                A[row, n_xl + 2*np_val + 1] = -dd_l_dt
            b[row] = delta_l
            row += 1

    return A, b


def create_bridge_length_constraints(num_layers, comp_per_layer, bridge_length):
    """Linear bridge-length constraints for 2-D ALM.

    For each layer k > 0 and each component j, the horizontal extent of the
    upper layer relative to the base-layer centre must not exceed bridge_length:
      Xc_{k,j} + L_{k,j}/2 - Xc_{0,j}  <= BL   (right edge vs base centre)
      Xc_{0,j} - Xc_{k,j} + L_{k,j}/2  <= BL   (base centre vs left edge)

    Returns A (num_cons, n_tot), b (num_cons,) for A @ x <= b.
    """
    nY = num_layers
    np_val = comp_per_layer
    n_xl   = 2 * nY * np_val
    n_tot  = n_xl + 2 * np_val + 2
    BL     = bridge_length

    num_cons = (nY - 1) * np_val * 2
    A = np.zeros((num_cons, n_tot))
    b = np.full(num_cons, BL)

    row = 0
    for layer in range(1, nY):
        for j in range(np_val):
            ix_base = _xc_idx(0,     j, nY)
            ix_k    = _xc_idx(layer, j, nY)
            il_k    = _l_idx(layer,  j, nY)

            # Right: Xc_k + L_k/2 - Xc_0 <= BL
            A[row, ix_k]    =  1.0
            A[row, il_k]    =  0.5
            A[row, ix_base] = -1.0
            row += 1

            # Left: Xc_0 - Xc_k + L_k/2 <= BL
            A[row, ix_base] =  1.0
            A[row, ix_k]    = -1.0
            A[row, il_k]    =  0.5
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
