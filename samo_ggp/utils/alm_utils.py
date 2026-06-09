import numpy as np

def create_alm_overhang_constraints(num_layers, comp_per_layer, layer_height, alpha_deg):
    """
    Creates linear overhang constraints for ALM.
    Variables per layer: [xc, width, mc]
    alpha_deg: max overhang angle from vertical (e.g. 45)
    Returns: A (matrix), b (vector) such that Ax <= b
    """
    num_components = num_layers * comp_per_layer
    num_vars = num_components * 3
    dx_over = layer_height * np.tan(np.deg2rad(alpha_deg))
    
    A = []
    b = []
    
    # Constraints between adjacent layers
    for layer in range(num_layers - 1):
        # We assume 1 component per layer for simplicity in this formulation
        # If comp_per_layer > 1, we need to decide which pairs are constrained.
        # Here we constrain component k in layer i with component k in layer i+1.
        for k in range(comp_per_layer):
            idx_i = (layer * comp_per_layer + k) * 3
            idx_next = ((layer + 1) * comp_per_layer + k) * 3
            
            # Left edge: - (xc_next - 0.5*w_next) + (xc_i - 0.5*w_i) <= dx_over
            # => -xc_next + 0.5*w_next + xc_i - 0.5*w_i <= dx_over
            row_l = np.zeros(num_vars)
            row_l[idx_next] = -1.0
            row_l[idx_next+1] = 0.5
            row_l[idx_i] = 1.0
            row_l[idx_i+1] = -0.5
            A.append(row_l)
            b.append(dx_over)
            
            # Right edge: (xc_next + 0.5*w_next) - (xc_i + 0.5*w_i) <= dx_over
            # => xc_next + 0.5*w_next - xc_i - 0.5*w_i <= dx_over
            row_r = np.zeros(num_vars)
            row_r[idx_next] = 1.0
            row_r[idx_next+1] = 0.5
            row_r[idx_i] = -1.0
            row_r[idx_i+1] = -0.5
            A.append(row_r)
            b.append(dx_over)
            
    return np.array(A), np.array(b)
