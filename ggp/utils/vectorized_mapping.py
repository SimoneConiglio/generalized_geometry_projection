import numpy as np

def smooth_saturation_np(x, ka, pp, xt, s0):
    inner_exp = np.exp((pp * x) / xt)
    rho = (-np.log(np.exp(-pp) + 1.0 / (inner_exp + 1.0)) / pp - s0) / (1.0 - s0)
    return rho

def compute_local_characteristic_2d_with_grad_np(X_mesh, Y_mesh, X, Y, L, h, T, r_gp, method='GP', eps_safe=1e-7):
    # Transform to local frame
    dx = X_mesh - X
    dy = Y_mesh - Y
    cos_t, sin_t = np.cos(T), np.sin(T)
    
    x_loc = dx * cos_t + dy * sin_t
    y_loc = -dx * sin_t + dy * cos_t
    
    # Distance to skeleton [-L/2, L/2]
    dx_skel = np.maximum(0, np.abs(x_loc) - L/2.0)
    upsi = np.sqrt(dx_skel**2 + y_loc**2 + eps_safe)
    
    # Gradients of upsi wrt local coords
    dupsi_dxloc = (dx_skel * np.sign(x_loc)) / upsi
    dupsi_dyloc = y_loc / upsi
    
    # Gradients of local coords wrt parameters
    dxloc_dX = -cos_t
    dxloc_dY = -sin_t
    dxloc_dT = -dx * sin_t + dy * cos_t # y_loc
    
    dyloc_dX = sin_t
    dyloc_dY = -cos_t
    dyloc_dT = -dx * cos_t - dy * sin_t # -x_loc
    
    # dupsi/dL (from dx_skel = max(0, |x_loc| - L/2))
    dupsi_dL = np.where(np.abs(x_loc) > L/2.0, -0.5 * dx_skel / upsi, 0.0)
    
    # Combine for dupsi/dparams
    dupsi_dX = dupsi_dxloc * dxloc_dX + dupsi_dyloc * dyloc_dX
    dupsi_dY = dupsi_dxloc * dxloc_dY + dupsi_dyloc * dyloc_dY
    dupsi_dT = dupsi_dxloc * dxloc_dT + dupsi_dyloc * dyloc_dT
    
    # Mapping W(upsi)
    if method == 'GP':
        deltamin = 1e-6
        zetavar = upsi - h / 2.0
        z_clipped = np.clip(zetavar / r_gp, -1.0 + eps_safe, 1.0 - eps_safe)
        W_raw = (1.0/np.pi * (np.arccos(z_clipped) - z_clipped * np.sqrt(1.0 - z_clipped**2)))
        deltaiel = np.where(zetavar < -r_gp, 1.0, np.where(zetavar > r_gp, 0.0, W_raw))
        W = deltamin + (1.0 - deltamin) * deltaiel
        
        # dW/dupsi = (dW/dz) * (dz/dupsi)
        # dW/dz = -2/pi * sqrt(1-z^2)
        dW_dz = -2.0/np.pi * np.sqrt(np.maximum(0, 1.0 - z_clipped**2))
        dz_dupsi = 1.0 / r_gp
        dz_dh = -0.5 / r_gp
        
        mask = (zetavar >= -r_gp) & (zetavar <= r_gp)
        dW_dupsi = np.where(mask, (1.0 - deltamin) * dW_dz * dz_dupsi, 0.0)
        dW_dh = np.where(mask, (1.0 - deltamin) * dW_dz * dz_dh, 0.0)
    else:
        # Placeholder for MNA
        W = np.zeros_like(upsi)
        dW_dupsi = np.zeros_like(upsi)
        dW_dh = np.zeros_like(upsi)

    return W, dW_dupsi * dupsi_dX, dW_dupsi * dupsi_dY, dW_dupsi * dupsi_dL, dW_dh, dW_dupsi * dupsi_dT

def compute_local_characteristic_np(X_mesh, Y_mesh, X, Y, L, h, T, r_gp, method='GP', eps_safe=1e-7, eps_mna=1.0, Z_mesh=None, Z=0.0, P=0.0, W_width=None):
    """
    Supports:
    - 2D Free (Round bar)
    - 3D Free (Cylinder/Capsule)
    - 3D ALM (Oriented Brick)
    """
    dx = X_mesh - X
    dy = Y_mesh - Y
    
    if Z_mesh is not None:
        dz = Z_mesh - Z
        if W_width is not None:
            # 3D ALM mode (Oriented Brick)
            cos_t, sin_t = np.cos(T), np.sin(T)
            x_loc = dx * cos_t + dy * sin_t
            y_loc = -dx * sin_t + dy * cos_t
            z_loc = dz
            
            dx_b = np.maximum(np.abs(x_loc) - L/2.0, 0.0)
            dy_b = np.maximum(np.abs(y_loc) - W_width/2.0, 0.0)
            dz_b = np.maximum(np.abs(z_loc) - h/2.0, 0.0)
            
            upsi = np.sqrt(dx_b**2 + dy_b**2 + dz_b**2 + eps_safe)
        else:
            # 3D Free mode (Cylinder/Capsule)
            nx = np.cos(P) * np.cos(T)
            ny = np.cos(P) * np.sin(T)
            nz = np.sin(P)
            proj = dx*nx + dy*ny + dz*nz
            t = np.clip(proj, -L/2.0, L/2.0)
            dist2 = (dx - t*nx)**2 + (dy - t*ny)**2 + (dz - t*nz)**2
            upsi = np.sqrt(np.maximum(eps_safe, dist2))
    else:
        # 2D Mode
        # Use the same skeletons as above for consistency
        dx = X_mesh - X
        dy = Y_mesh - Y
        cos_t, sin_t = np.cos(T), np.sin(T)
        x_loc = dx * cos_t + dy * sin_t
        y_loc = -dx * sin_t + dy * cos_t
        dx_skel = np.maximum(0, np.abs(x_loc) - L/2.0)
        upsi = np.sqrt(dx_skel**2 + y_loc**2 + eps_safe)
    
    # Mapping
    if method == 'GP':
        deltamin = 1e-6
        zetavar = upsi - h / 2.0
        z_clipped = np.clip(zetavar / r_gp, -1.0 + eps_safe, 1.0 - eps_safe)
        delta_iel = (1.0/np.pi * (np.arccos(z_clipped) - z_clipped * np.sqrt(1.0 - z_clipped**2)))
        deltaiel = np.where(zetavar < -r_gp, 1.0, 
                            np.where(zetavar > r_gp, 0.0, delta_iel))
        return deltamin + (1.0 - deltamin) * deltaiel
    else:
        l, u = h / 2.0 - eps_mna / 2.0, h / 2.0 + eps_mna / 2.0
        d_v = -(eps_mna**3)
        return np.where(upsi <= l, 1.0,
                        np.where(upsi <= u,
                                 (-2.0 / d_v) * upsi**3 + (3.0 * h / d_v) * upsi**2 + 
                                 (-6.0 * l * u / d_v) * upsi + (u * (-u**2 + 3.0 * l * u) / d_v),
                                 0.0))
