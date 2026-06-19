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
        zeta5 = y_loc - h/2.0
        
        def W_val(z):
            val = 0.5 - (15.0 / (16.0 * sigma)) * z + (5.0 / (8.0 * sigma**3)) * (z**3) - (3.0 / (16.0 * sigma**5)) * (z**5)
            return np.where(z < -sigma, 1.0, np.where(z > sigma, 0.0, val))
            
        return W_val(zeta1) * W_val(zeta2) * W_val(zeta3) * W_val(zeta4) * W_val(zeta5)
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

def compute_continuous_ALM_characteristic_np(x, y, Xc, p, np_val, nY, Yk, nely):
    # Ensure column vectors
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)
    
    Xk = Xc[0:np_val*nY]
    Lk = Xc[np_val*nY:2*np_val*nY]
    h = Xc[2*np_val*nY:2*np_val*nY+nY]
    
    Xk = np.reshape(Xk, (nY, np_val), order='F')
    Lk = np.reshape(Lk, (nY, np_val), order='F')
    Yk = np.reshape(Yk, (nY, np_val), order='F')
    
    idy = np.ceil(y / np.max(Yk) * (nY - 1)).astype(int) - 1
    idy[idy < 0] = 0
    idy[idy >= nY - 1] = nY - 2
    idy_flat = idy.flatten()
    
    # xk & dxk_dXk
    xk = Xk[:-1, :]
    dxk_dXk = np.zeros((xk.shape[0], Xk.shape[0]))
    dxk_dXk[:, :-1] = np.eye(xk.shape[0])
    dxk_dXk = sp.csc_matrix(dxk_dXk)
    
    # xk+1 & dxk+1_dXk
    xk1 = Xk[1:, :]
    dxk1_dXk = np.zeros((xk1.shape[0], Xk.shape[0]))
    dxk1_dXk[:, 1:] = np.eye(xk1.shape[0])
    dxk1_dXk = sp.csc_matrix(dxk1_dXk)
    
    # yk & yk+1
    yk = Yk[:-1, :]
    yk1 = Yk[1:, :]
    
    # lk & dlk_dLk
    lk = Lk[:-1, :]
    dlk_dLk = np.zeros((lk.shape[0], Lk.shape[0]))
    dlk_dLk[:, :-1] = np.eye(lk.shape[0])
    dlk_dLk = sp.csc_matrix(dlk_dLk)
    
    # 1/lk & 1/dlk_dLk
    d11lk_dLk = np.zeros((lk.shape[0], Lk.shape[0])) #%%
    d11lk_dLk[:, :-1] = - (1.0 / lk[:, 0])**2 * np.eye(lk.shape[0])
    d11lk_dLk = sp.csc_matrix(d11lk_dLk)
    
    # lk+1 & dlk+1_dLk
    lk1 = Lk[1:, :]
    dlk1_dLk = np.zeros((lk.shape[0], Lk.shape[0]))
    dlk1_dLk[:, 1:] = np.eye(lk1.shape[0])
    dlk1_dLk = sp.csc_matrix(dlk1_dLk)
    
    if p['method'] == 'MMC':
        alp = p['alp']
        epsi = p['epsi']
        bet = p['bet']
        
        rep_y = np.tile(y, (1, np_val))
        temp_val = 1 - ((xk[idy_flat, :] - rep_y) / (lk[idy_flat, :] / 2))**(2*alp)
        chi0 = np.reshape(temp_val, (-1, len(idy_flat)), order='F') #
        
        temp_dchi0_X = (2*alp) * ((xk[idy_flat, :] - rep_y) / lk[idy_flat, :] * 2)**(2*alp-1)
        temp_dchi0_X = np.tile(temp_dchi0_X, (nY, 1))
        temp_dxk_dXk = np.tile(dxk_dXk[idy_flat, :].toarray(), (1, np_val))
        temp_lk = np.tile(lk[idy_flat, :] / 2, (nY, 1))
        dchi0_dXk = - np.reshape(temp_dchi0_X, (len(idy_flat), -1), order='F') * (temp_dxk_dXk / np.reshape(temp_lk, (len(idy_flat), -1), order='F'))
        
        temp_dchi0_L = (2*alp) * ((xk[idy_flat, :] - rep_y) / lk[idy_flat, :] * 2)**(2*alp-1) * ((xk[idy_flat, :] - rep_y) * 2)
        temp_dchi0_L = np.tile(temp_dchi0_L, (nY, 1))
        temp_d11lk = np.tile(d11lk_dLk[idy_flat, :].toarray(), (1, np_val))
        dchi0_dLk = - np.reshape(temp_dchi0_L, (len(idy_flat), -1), order='F') * temp_d11lk
        
        chi, dchi = Aggregation_Pi(chi0, p)
        
        dchi_dXk = dchi0_dXk * np.tile(dchi.T, (1, nY))
        dchi_dLk = dchi0_dLk * np.tile(dchi.T, (1, nY))
        
        b = nely * h
        tv3 = rep_y - np.tile(b.flatten(), (y.shape[0], 1))
        dh_h = np.ones((h.shape[0], 1))
        db_h = nely * np.ones((h.shape[0], 1))
        dtv3_h = - np.tile(db_h.flatten(), (y.shape[0], 1))
        #
        chi[chi <= -1e6] = -1e6
        
        def w(x):
            return (x > epsi) + ((x <= epsi) & (x >= -epsi)) * (3.0/4.0*(1-bet)*(x/epsi - x**3/3.0/epsi**3) + (1+bet)/2.0) + (x < -epsi)*bet
            
        def dw(x):
            return ((x <= epsi) & (x >= -epsi)) * ((3.0/4.0*(1-bet)*(1.0/epsi)) - (3.0/4.0*(1-bet)*(x**2/epsi**3)))
            
        W = np.tile(w(chi), (np_val, 1))
        dW_dX = np.tile(np.tile(dw(chi), (np_val, 1)), (nY, 1)) * dchi_dXk.T
        dW_dL = np.tile(np.tile(dw(chi), (np_val, 1)), (nY, 1)) * dchi_dLk.T
        
        dW_dh = np.array([1, 1, 1, 1, 1])
        dW_dh = np.tile(dW_dh, (len(idy_flat), 1)).T
        
    elif p['method'] == 'GP':
        deltamin = p['deltamin']
        r = p['r']
        
        rep_y = np.tile(y, (1, np_val))
        d1 = xk[idy_flat, :] - (rep_y + lk[idy_flat, :] / 2.0)
        d2 = -xk[idy_flat, :] + (rep_y - lk[idy_flat, :] / 2.0)
        q = 10.0
        zetavar = (1.0/q * np.log(0.5 * (np.exp(q*d1) + np.exp(q*d2))))
        
        exp_d1 = np.tile(np.exp(q*d1.T), (nY, 1))
        exp_d2 = np.tile(np.exp(q*d2.T), (nY, 1))
        q_dxk = q * np.tile(dxk_dXk[idy_flat, :].toarray(), (1, np_val))
        
        dzetavar_Xk_numinateur = 0.5 * (exp_d1 * q_dxk.T - exp_d2 * q_dxk.T)
        dzetavar_Xk_denominateur = 0.5 * (exp_d1 + exp_d2)
        dzetavar_Xk = 1.0/q * (dzetavar_Xk_numinateur / dzetavar_Xk_denominateur)
        
        q_dlk = q * np.tile(-dlk_dLk[idy_flat, :].toarray() / 2.0, (1, np_val))
        dzetavar_Lk_numinateur = 0.5 * (exp_d1 * q_dlk.T - exp_d2 * q_dlk.T)
        dzetavar_Lk_denominateur = 0.5 * (exp_d1 + exp_d2)
        dzetavar_Lk = 1.0/q * (dzetavar_Lk_numinateur / dzetavar_Lk_denominateur)
        
        z_r = np.clip(zetavar/r, -1.0, 1.0)
        deltaiel = (1.0/np.pi/r**2 * (r**2 * np.arccos(z_r) - zetavar * np.sqrt(np.maximum(0, r**2 - zetavar**2)))) * (np.abs(zetavar) <= r) + (zetavar < -r)
        ddetlaiel_dzetavar = (-2.0 * np.sqrt(np.maximum(0, r**2 - zetavar**2)) / np.pi / r**2) * (np.abs(zetavar) <= r)
        W = (deltamin + (1.0 - deltamin) * deltaiel).T
        dW_ddeltaiel = (1.0 - deltamin)
        
        dW_dX = dW_ddeltaiel * np.tile(ddetlaiel_dzetavar.T, (nY, 1)) * dzetavar_Xk
        dW_dL = dW_ddeltaiel * np.tile(ddetlaiel_dzetavar.T, (nY, 1)) * dzetavar_Lk
        dW_dh = np.zeros((h.shape[0], W.shape[1] if W.ndim > 1 else 1)) # Add dummy return
        
    elif p['method'] == 'MNA':
        rep_y = np.tile(y, (1, np_val))
        # ly & uy
        ly = xk[idy_flat, :] - lk[idy_flat, :] / 2.0 + (rep_y - yk[idy_flat, :]) / (yk1[idy_flat, :] - yk[idy_flat, :]) * ((xk1[idy_flat, :] - lk1[idy_flat, :] / 2.0) - (xk[idy_flat, :] - lk[idy_flat, :] / 2.0))
        uy = xk[idy_flat, :] + lk[idy_flat, :] / 2.0 + (rep_y - yk[idy_flat, :]) / (yk1[idy_flat, :] - yk[idy_flat, :]) * ((xk1[idy_flat, :] + lk1[idy_flat, :] / 2.0) - (xk[idy_flat, :] + lk[idy_flat, :] / 2.0))
        
        # derivé ly et uy
        y_ratio = (rep_y - yk[idy_flat, :]) / (yk1[idy_flat, :] - yk[idy_flat, :])
        y_ratio_rep = np.reshape(np.tile(y_ratio, (nY, 1)), (len(idy_flat), -1), order='F')
        
        dxk_rep = np.tile(dxk_dXk[idy_flat, :].toarray(), (1, np_val))
        dxk1_rep = np.tile((dxk1_dXk[idy_flat, :] - dxk_dXk[idy_flat, :]).toarray(), (1, np_val))
        dly_Xk = dxk_rep + y_ratio_rep * dxk1_rep
        duy_Xk = dxk_rep + y_ratio_rep * dxk1_rep
        
        dlk_rep = np.tile(-dlk_dLk[idy_flat, :].toarray(), (1, np_val)) / 2.0
        dlk1_rep = np.tile((-dlk1_dLk[idy_flat, :].toarray() / 2.0 + dlk_dLk[idy_flat, :].toarray() / 2.0), (1, np_val))
        dly_Lk = dlk_rep + y_ratio_rep * dlk1_rep
        
        dlk_pos_rep = np.tile(dlk_dLk[idy_flat, :].toarray() / 2.0, (1, np_val))
        dlk1_pos_rep = np.tile((dlk1_dLk[idy_flat, :].toarray() / 2.0 - dlk_dLk[idy_flat, :].toarray() / 2.0), (1, np_val))
        duy_Lk = dlk_pos_rep + y_ratio_rep * dlk1_pos_rep
        
        gt = 3.0
        def w(x):
            return (0.5 - (15.0/(16.0*gt))*x + (5.0/(8.0*gt**3))*x**3 - (3.0/(16.0*gt**5))*x**5) * ((x >= -gt) & (x <= gt)) + (x < -gt)
            
        def dw(x):
            return (-15.0/(16.0*gt) + 3.0*(5.0/(8.0*gt**3))*x**2 - 5.0*(3.0/(16.0*gt**5))*x**4) * ((x >= -gt) & (x <= gt))
            
        rep_x = np.tile(x, (1, np_val))
        tv1 = -rep_x + ly
        tv2 = rep_x - uy
        b = nely * h
        tv3 = rep_y - np.tile(b.flatten(), (y.shape[0], 1))
        
        W = w(tv1).T * w(tv2).T * w(tv3).T
        
        # derivé tv1 et tv2
        dtv1_Xk = +dly_Xk
        dtv2_Xk = -duy_Xk
        dtv1_Lk = +dly_Lk
        dtv2_Lk = -duy_Lk
        
        #
        dh_h = np.ones((h.shape[0], 1))
        db_h = nely * np.ones((h.shape[0], 1))
        dtv3_h = - np.tile(db_h.flatten(), (y.shape[0], 1))
        
        #
        dw_tv1_w_tv2 = np.reshape(np.tile(dw(tv1) * w(tv2), (nY, 1)), (len(idy_flat), -1), order='F')
        dw_tv2_w_tv1 = np.reshape(np.tile(dw(tv2) * w(tv1), (nY, 1)), (len(idy_flat), -1), order='F')
        w_tv3_rep = np.reshape(np.tile(w(tv3), (nY, 1)), (len(idy_flat), -1), order='F')
        
        dW_dX = ((dw_tv1_w_tv2 * dtv1_Xk + dw_tv2_w_tv1 * dtv2_Xk) * w_tv3_rep).T
        dW_dL = ((dw_tv1_w_tv2 * dtv1_Lk + dw_tv2_w_tv1 * dtv2_Lk) * w_tv3_rep).T
        
        dW_dh = ((np.reshape(dw(tv3), (len(idy_flat), -1), order='F') * dtv3_h) * np.reshape(w(tv1) * w(tv2), (len(idy_flat), -1), order='F')).T
        
    return W, dW_dX, dW_dL, dW_dh
