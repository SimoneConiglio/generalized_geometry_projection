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
        return W, dW_dupsi * dupsi_dX, dW_dupsi * dupsi_dY, dW_dupsi * dupsi_dL, dW_dh, dW_dupsi * dupsi_dT
    elif method == 'ALM_GP':
        deltamin = 1e-6
        dx = X_mesh - X
        dy = Y_mesh - Y
        cos_t, sin_t = np.cos(T), np.sin(T)
        x_loc = dx * cos_t + dy * sin_t
        y_loc = -dx * sin_t + dy * cos_t
        
        zetavar = np.abs(x_loc) - L/2.0
        z_clipped = np.clip(zetavar / r_gp, -1.0 + eps_safe, 1.0 - eps_safe)
        W_raw = (1.0/np.pi * (np.arccos(z_clipped) - z_clipped * np.sqrt(1.0 - z_clipped**2)))
        deltaiel = np.where(zetavar < -r_gp, 1.0, np.where(zetavar > r_gp, 0.0, W_raw))
        
        mask_y = np.abs(y_loc) <= (h/2.0 + 1e-6)
        W = np.where(mask_y, deltamin + (1.0 - deltamin) * deltaiel, 0.0)
        
        # dW/dzetavar
        dW_dz = -2.0/np.pi * np.sqrt(np.maximum(0, 1.0 - z_clipped**2))
        dz_dzetavar = 1.0 / r_gp
        
        mask_x = (zetavar >= -r_gp) & (zetavar <= r_gp)
        dW_dzetavar = np.where(mask_x & mask_y, (1.0 - deltamin) * dW_dz * dz_dzetavar, 0.0)
        
        dzetavar_dxloc = np.sign(x_loc)
        dzetavar_dL = -0.5
        
        dxloc_dX = -cos_t
        dxloc_dY = -sin_t
        dxloc_dT = -dx*sin_t + dy*cos_t
        
        dW_dX = dW_dzetavar * dzetavar_dxloc * dxloc_dX
        dW_dY = dW_dzetavar * dzetavar_dxloc * dxloc_dY
        dW_dL = dW_dzetavar * dzetavar_dL
        dW_dh = np.zeros_like(W)
        dW_dT = dW_dzetavar * dzetavar_dxloc * dxloc_dT
        
        return W, dW_dX, dW_dY, dW_dL, dW_dh, dW_dT
    elif method == 'AMNA':
        sigma = r_gp
        
        # zetas
        zeta1 = -L/2.0 - x_loc
        zeta2 = x_loc - L/2.0
        zeta3 = y_loc - h/2.0
        zeta4 = -h/2.0 - y_loc
        
        zetas = [zeta1, zeta2, zeta3, zeta4]
        
        def W_val(z):
            val = 0.5 - (15.0 / (16.0 * sigma)) * z + (5.0 / (8.0 * sigma**3)) * (z**3) - (3.0 / (16.0 * sigma**5)) * (z**5)
            return np.where(z < -sigma, 1.0, np.where(z > sigma, 0.0, val))
            
        def dW_dz_val(z):
            val = - (15.0 / (16.0 * sigma)) * (1.0 - (z / sigma)**2)**2
            return np.where((z >= -sigma) & (z <= sigma), val, 0.0)
            
        W_i = [W_val(z) for z in zetas]
        dW_dzeta_i = [dW_dz_val(z) for z in zetas]
        
        W = W_i[0] * W_i[1] * W_i[2] * W_i[3]
        
        W_prod_except = []
        for i in range(4):
            prod_val = 1.0
            for k in range(4):
                if k != i:
                    prod_val = prod_val * W_i[k]
            W_prod_except.append(prod_val)
            
        # Derivatives of zetas wrt local coords and parameters
        dzeta_dX = [cos_t, -cos_t, sin_t, -sin_t]
        dzeta_dY = [sin_t, -sin_t, -cos_t, cos_t]
        dzeta_dL = [-0.5, -0.5, 0.0, 0.0]
        dzeta_dh = [0.0, 0.0, -0.5, -0.5]
        dzeta_dT = [-y_loc, y_loc, -x_loc, x_loc]
        
        dW_dX = sum(dW_dzeta_i[i] * dzeta_dX[i] * W_prod_except[i] for i in range(4))
        dW_dY = sum(dW_dzeta_i[i] * dzeta_dY[i] * W_prod_except[i] for i in range(4))
        dW_dL = sum(dW_dzeta_i[i] * dzeta_dL[i] * W_prod_except[i] for i in range(4))
        dW_dh = sum(dW_dzeta_i[i] * dzeta_dh[i] * W_prod_except[i] for i in range(4))
        dW_dT = sum(dW_dzeta_i[i] * dzeta_dT[i] * W_prod_except[i] for i in range(4))
        
        return W, dW_dX, dW_dY, dW_dL, dW_dh, dW_dT
    else:
        # Placeholder for other modes
        W = np.zeros_like(upsi)
        return W, np.zeros_like(upsi), np.zeros_like(upsi), np.zeros_like(upsi), np.zeros_like(upsi), np.zeros_like(upsi)

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
    elif method == 'ALM_GP':
        deltamin = 1e-6
        dx = X_mesh - X
        dy = Y_mesh - Y
        cos_t, sin_t = np.cos(T), np.sin(T)
        x_loc = dx * cos_t + dy * sin_t
        y_loc = -dx * sin_t + dy * cos_t
        
        zetavar = np.abs(x_loc) - L/2.0
        z_clipped = np.clip(zetavar / r_gp, -1.0 + eps_safe, 1.0 - eps_safe)
        W_raw = (1.0/np.pi * (np.arccos(z_clipped) - z_clipped * np.sqrt(1.0 - z_clipped**2)))
        deltaiel = np.where(zetavar < -r_gp, 1.0, np.where(zetavar > r_gp, 0.0, W_raw))
        
        mask_y = np.abs(y_loc) <= (h/2.0 + 1e-6)
        return np.where(mask_y, deltamin + (1.0 - deltamin) * deltaiel, 0.0)
    elif method == 'AMNA' and Z_mesh is None:
        sigma = r_gp
        zeta1 = -L/2.0 - x_loc
        zeta2 = x_loc - L/2.0
        zeta3 = y_loc - h/2.0
        zeta4 = -h/2.0 - y_loc

        def W_val(z):
            val = 0.5 - (15.0 / (16.0 * sigma)) * z + (5.0 / (8.0 * sigma**3)) * (z**3) - (3.0 / (16.0 * sigma**5)) * (z**5)
            return np.where(z < -sigma, 1.0, np.where(z > sigma, 0.0, val))

        return W_val(zeta1) * W_val(zeta2) * W_val(zeta3) * W_val(zeta4)
    else:
        l, u = h / 2.0 - eps_mna / 2.0, h / 2.0 + eps_mna / 2.0
        d_v = -(eps_mna**3)
        return np.where(upsi <= l, 1.0,
                        np.where(upsi <= u,
                                 (-2.0 / d_v) * upsi**3 + (3.0 * h / d_v) * upsi**2 + 
                                 (-6.0 * l * u / d_v) * upsi + (u * (-u**2 + 3.0 * l * u) / d_v),
                                 0.0))


import scipy.sparse as sp
from ggp.utils.Aggregation_Pi import Aggregation_Pi

def compute_continuous_ALM_characteristic_np(x, y, Xc, p, np_val, nY, Yk, Ly):
    """MNA ALM characteristic function with interleaved variable layout.

    Variable layout in Xc:
      [Xc_0_0, L_0_0, Xc_0_1, L_0_1, ..., Xc_{nY-1}_{np-1}, L_{nY-1}_{np-1},
       h_0, ..., h_{np-1},  Mc_0, ..., Mc_{np-1}]
    (h = normalised total height per printed component; Mc not used here)

    Parameters
    ----------
    x, y   : eval coordinates, 1-D arrays of length n_eval
    Xc     : design variable vector (interleaved layout above)
    p      : parameter dict (unused legacy arg; method always MNA)
    np_val : number of printed components (columns)
    nY     : number of layers
    Yk     : (nY, np_val) fixed y-centres of each component/layer
    Ly     : physical domain height (h normalised → cutoff = Ly * h)

    Returns
    -------
    W        : (np_val, n_eval)       characteristic function per component
    dW_dXc   : (np_val, nY, n_eval)  gradient wrt x-centres per layer
    dW_dL    : (np_val, nY, n_eval)  gradient wrt widths per layer
    dW_dh    : (np_val, n_eval)      gradient wrt height cutoffs
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n_eval = len(x)

    # Extract interleaved [Xc, L] block then h
    Xk = Xc[0:2*np_val*nY:2].reshape((nY, np_val), order='F')  # Xk[k,j]
    Lk = Xc[1:2*np_val*nY:2].reshape((nY, np_val), order='F')  # Lk[k,j]
    h  = Xc[2*np_val*nY : 2*np_val*nY + np_val]                 # h[j]

    Yk = np.asarray(Yk).reshape((nY, np_val), order='F') if np.asarray(Yk).ndim == 1 else np.asarray(Yk)

    # Layer index for each eval point (clamped to [0, nY-2])
    if nY > 1:
        layer_sp = float(Yk[1, 0] - Yk[0, 0]) if Yk[1, 0] != Yk[0, 0] else (Ly / nY)
        idy = np.clip(np.floor(y / layer_sp).astype(int), 0, nY - 2)
    else:
        idy = np.zeros(n_eval, dtype=int)
    idy_flat = idy  # keep as alias used below

    # Physical height cutoffs
    b = Ly * h  # shape (np_val,)

    # MNA quintic characteristic function
    gt = 3.0

    def w(t):
        inside = (t >= -gt) & (t <= gt)
        return (inside * (0.5 - (15.0/(16*gt))*t + (5.0/(8*gt**3))*t**3
                          - (3.0/(16*gt**5))*t**5)
                + (t < -gt).astype(float))

    def dw(t):
        inside = (t >= -gt) & (t <= gt)
        return inside * (-15.0/(16*gt) + 3*(5.0/(8*gt**3))*t**2
                         - 5*(3.0/(16*gt**5))*t**4)
    
    # Per-component MNA characteristic function and exact gradients
    W      = np.zeros((np_val, n_eval))
    dW_dXc = np.zeros((np_val, nY, n_eval))
    dW_dL  = np.zeros((np_val, nY, n_eval))
    dW_dh  = np.zeros((np_val, n_eval))

    k_ev = idy_flat                           # current layer index per eval point
    k_nx = np.minimum(idy_flat + 1, nY - 1)  # next layer (clamped)

    for j in range(np_val):
        xk_j  = Xk[k_ev, j]
        lk_j  = Lk[k_ev, j]
        yk_j  = Yk[k_ev, j]
        xk1_j = Xk[k_nx, j]
        lk1_j = Lk[k_nx, j]
        yk1_j = Yk[k_nx, j]

        dy = np.where(np.abs(yk1_j - yk_j) > 1e-10, yk1_j - yk_j, 1e-10)
        yr = np.clip((y - yk_j) / dy, 0.0, 1.0)

        ly = (xk_j - lk_j/2) + yr * ((xk1_j - lk1_j/2) - (xk_j - lk_j/2))
        uy = (xk_j + lk_j/2) + yr * ((xk1_j + lk1_j/2) - (xk_j + lk_j/2))

        tv1 = -x + ly         # positive inside left boundary
        tv2 = x - uy          # negative inside right boundary
        tv3 = y - b[j]        # negative below height cutoff

        wt1 = w(tv1); wt2 = w(tv2); wt3 = w(tv3)
        W[j, :] = wt1 * wt2 * wt3

        dWt1 = dw(tv1) * wt2 * wt3
        dWt2 = wt1 * dw(tv2) * wt3
        dWt3 = wt1 * wt2 * dw(tv3)

        dW_dh[j, :] = dWt3 * (-Ly)

        # Gradient wrt Xc and L, scattered to current layer (k_ev) and next (k_nx)
        # dtv1/dXc_k = (1-yr), dtv2/dXc_k = -(1-yr)  [dly/dXc_k = 1-yr; duy/dXc_k = 1-yr]
        # dtv1/dL_k  = -(1-yr)/2, dtv2/dL_k = -(1-yr)/2
        dXc_k  = (dWt1 - dWt2) * (1 - yr)
        dXc_k1 = (dWt1 - dWt2) * yr
        dL_k   = -(dWt1 + dWt2) * (1 - yr) / 2.0
        dL_k1  = -(dWt1 + dWt2) * yr / 2.0

        for k in range(nY):
            mk  = (k_ev == k)
            mnx = (k_nx == k)
            dW_dXc[j, k, :] += np.where(mk,  dXc_k,  0.0)
            dW_dXc[j, k, :] += np.where(mnx, dXc_k1, 0.0)
            dW_dL[j, k, :]  += np.where(mk,  dL_k,   0.0)
            dW_dL[j, k, :]  += np.where(mnx, dL_k1,  0.0)

    return W, dW_dXc, dW_dL, dW_dh
