import numpy as np

def grad_3d_free_neg(dx, dy, dz, L, T, P, eps_safe):
    x0 = (1/2)*np.sin(P)
    x1 = L*x0
    x2 = dz + x1
    x3 = np.sin(T)
    x4 = np.cos(P)
    x5 = (1/2)*x4
    x6 = L*x5
    x7 = x3*x6
    x8 = dy + x7
    x9 = np.cos(T)
    x10 = x6*x9
    x11 = dx + x10
    x12 = np.sqrt(eps_safe + x11**2 + x2**2 + x8**2)
    x13 = x12**(-1.0)
    x14 = x3*x8
    x15 = x11*x9
    upsi = x12
    dX = -x11*x13
    dY = -x13*x8
    dZ = -x13*x2
    dL = x13*(x0*x2 + x14*x5 + x15*x5)
    dT = x13*(x10*x8 - x11*x7)
    dP = x13*((1/2)*L*x2*x4 - x1*x14 - x1*x15)
    return upsi, dX, dY, dZ, dL, dT, dP

def grad_3d_free_pos(dx, dy, dz, L, T, P, eps_safe):
    x0 = (1/2)*np.sin(P)
    x1 = L*x0
    x2 = dz - x1
    x3 = np.sin(T)
    x4 = np.cos(P)
    x5 = (1/2)*x4
    x6 = L*x5
    x7 = dy - x3*x6
    x8 = np.cos(T)
    x9 = x6*x8
    x10 = dx - x9
    x11 = np.sqrt(eps_safe + x10**2 + x2**2 + x7**2)
    x12 = x11**(-1.0)
    x13 = x3*x7
    x14 = x10*x8
    upsi = x11
    dX = -x10*x12
    dY = -x12*x7
    dZ = -x12*x2
    dL = x12*(-x0*x2 - x13*x5 - x14*x5)
    dT = x12*((1/2)*L*x10*x3*x4 - x7*x9)
    dP = x12*(x1*x13 + x1*x14 - x2*x6)
    return upsi, dX, dY, dZ, dL, dT, dP

def grad_3d_free_mid(dx, dy, dz, L, T, P, eps_safe):
    x0 = np.sin(P)
    x1 = np.cos(P)
    x2 = np.cos(T)
    x3 = x1*x2
    x4 = np.sin(T)
    x5 = x1*x4
    x6 = dx*x3 + dy*x5 + dz*x0
    x7 = dz - x0*x6
    x8 = x3*x6
    x9 = dx - x8
    x10 = dy - x5*x6
    x11 = np.sqrt(eps_safe + x10**2 + x7**2 + x9**2)
    x12 = x11**(-1.0)
    x13 = x0*x7
    x14 = x1**2
    x15 = x14*x2*x4
    x16 = 2*x14
    x17 = -dx*x5 + dy*x1*x2
    x18 = 2*x17
    x19 = (1/2)*x9
    x20 = (1/2)*x10
    x21 = -2*dx*x0*x2 - 2*dy*x0*x4 + 2*dz*x1
    upsi = x11
    dX = -x12*(-x10*x15 - x13*x3 + (1/2)*x9*(-x16*x2**2 + 2))
    dY = -x12*((1/2)*x10*(-x16*x4**2 + 2) - x13*x5 - x15*x9)
    dZ = -x12*(-x0*x10*x5 - x0*x3*x9 + (1/2)*x7*(2 - 2*x0**2))
    dL = np.zeros_like(dx)
    dT = x12*(-x13*x17 + x19*(2*x1*x4*x6 - x18*x3) + x20*(-x18*x5 - 2*x8))
    dP = x12*(x19*(2*x0*x2*x6 - x21*x3) + x20*(2*x0*x4*x6 - x21*x5) + (1/2)*x7*(-x0*x21 - 2*x1*x6))
    return upsi, dX, dY, dZ, dL, dT, dP

def compute_local_characteristic_3d_free_with_grad_np(X_mesh, Y_mesh, Z_mesh, Xc, Yc, Zc, L, h, T, P, r_gp, method='GP', eps_safe=1e-7):
    dx = X_mesh - Xc
    dy = Y_mesh - Yc
    dz = Z_mesh - Zc
    
    nx = np.cos(P) * np.cos(T)
    ny = np.cos(P) * np.sin(T)
    nz = np.sin(P)
    
    proj = dx*nx + dy*ny + dz*nz
    
    mask_neg = proj < -L/2.0
    mask_pos = proj > L/2.0
    mask_mid = ~(mask_neg | mask_pos)
    
    upsi = np.zeros_like(dx)
    dX = np.zeros_like(dx)
    dY = np.zeros_like(dx)
    dZ = np.zeros_like(dx)
    dL = np.zeros_like(dx)
    dT = np.zeros_like(dx)
    dP = np.zeros_like(dx)
    
    if np.any(mask_neg):
        u, dx_, dy_, dz_, dl_, dt_, dp_ = grad_3d_free_neg(dx[mask_neg], dy[mask_neg], dz[mask_neg], L, T, P, eps_safe)
        upsi[mask_neg] = u
        dX[mask_neg] = dx_
        dY[mask_neg] = dy_
        dZ[mask_neg] = dz_
        dL[mask_neg] = dl_
        dT[mask_neg] = dt_
        dP[mask_neg] = dp_
        
    if np.any(mask_pos):
        u, dx_, dy_, dz_, dl_, dt_, dp_ = grad_3d_free_pos(dx[mask_pos], dy[mask_pos], dz[mask_pos], L, T, P, eps_safe)
        upsi[mask_pos] = u
        dX[mask_pos] = dx_
        dY[mask_pos] = dy_
        dZ[mask_pos] = dz_
        dL[mask_pos] = dl_
        dT[mask_pos] = dt_
        dP[mask_pos] = dp_

    if np.any(mask_mid):
        u, dx_, dy_, dz_, dl_, dt_, dp_ = grad_3d_free_mid(dx[mask_mid], dy[mask_mid], dz[mask_mid], L, T, P, eps_safe)
        upsi[mask_mid] = u
        dX[mask_mid] = dx_
        dY[mask_mid] = dy_
        dZ[mask_mid] = dz_
        dL[mask_mid] = dl_
        dT[mask_mid] = dt_
        dP[mask_mid] = dp_
        
    if method == 'GP':
        deltamin = 1e-6
        zetavar = upsi - h / 2.0
        dzeta_dh = -0.5
        
        z_clipped = np.clip(zetavar / r_gp, -1.0 + eps_safe, 1.0 - eps_safe)
        
        delta_iel = (1.0/np.pi * (np.arccos(z_clipped) - z_clipped * np.sqrt(1.0 - z_clipped**2)))
        ddelta_dzeta = (-2.0 * np.sqrt(1.0 - z_clipped**2)) / (np.pi * r_gp)
        
        W = np.where(zetavar < -r_gp, 1.0, np.where(zetavar > r_gp, 0.0, deltamin + (1.0 - deltamin) * delta_iel))
        
        dW_dzeta = np.where((zetavar >= -r_gp) & (zetavar <= r_gp), (1.0 - deltamin) * ddelta_dzeta, 0.0)
        
        dWdX = dW_dzeta * dX
        dWdY = dW_dzeta * dY
        dWdZ = dW_dzeta * dZ
        dWdL = dW_dzeta * dL
        dWdh = dW_dzeta * dzeta_dh
        dWdT = dW_dzeta * dT
        dWdP = dW_dzeta * dP
        
        return W, dWdX, dWdY, dWdZ, dWdL, dWdh, dWdT, dWdP
    else:
        raise NotImplementedError("Only GP method is supported for 3D Free analytical gradients.")

def compute_local_characteristic_3d_alm_with_grad_np(X_mesh, Y_mesh, Z_mesh, Xc, Yc, Zc, L, W_width, h, T, r_gp, method='GP', eps_safe=1e-7):
    dx = X_mesh - Xc
    dy = Y_mesh - Yc
    dz = Z_mesh - Zc
    
    cos_t = np.cos(T)
    sin_t = np.sin(T)
    
    x_loc = dx * cos_t + dy * sin_t
    y_loc = -dx * sin_t + dy * cos_t
    z_loc = dz
    
    dx_b = x_loc - L/2.0
    dy_b = y_loc - W_width/2.0
    dz_b = z_loc - h/2.0
    
    hx = np.maximum(dx_b, 0.0)
    hy = np.maximum(dy_b, 0.0)
    hz = np.maximum(dz_b, 0.0)
    
    upsi = np.sqrt(hx**2 + hy**2 + hz**2 + eps_safe)
    inv_upsi = 1.0 / upsi
    
    dhx_dXc = -cos_t * (dx_b > 0)
    dhy_dXc = sin_t * (dy_b > 0)
    dX = (hx * dhx_dXc + hy * dhy_dXc) * inv_upsi
    
    dhx_dYc = -sin_t * (dx_b > 0)
    dhy_dYc = -cos_t * (dy_b > 0)
    dY = (hx * dhx_dYc + hy * dhy_dYc) * inv_upsi
    
    dZ = (hz * -1.0 * (dz_b > 0)) * inv_upsi
    dL = (hx * -0.5 * (dx_b > 0)) * inv_upsi
    dW_w = (hy * -0.5 * (dy_b > 0)) * inv_upsi
    dh = (hz * -0.5 * (dz_b > 0)) * inv_upsi
    
    dhx_dT = y_loc * (dx_b > 0)
    dhy_dT = -x_loc * (dy_b > 0)
    dT = (hx * dhx_dT + hy * dhy_dT) * inv_upsi
    
    if method == 'GP':
        deltamin = 1e-6
        # Note: in 3D ALM we often define zetavar using a specific projection.
        # But based on the un-gradient version in vectorized_mapping:
        zetavar = upsi - 1e-8 # Or usually we use upsi - h/2, but h is already in upsi for 3D ALM?
        # Let's check compute_local_characteristic_np for 3D ALM!
        # wait, in 3D ALM, upsi IS the distance to the brick! 
        # So zetavar = upsi - r_gp ? No, let's look at vectorized_mapping.py!
        pass
    
    return upsi, dX, dY, dZ, dL, dW_w, dh, dT
