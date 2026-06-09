import numpy as np

def smooth_saturation_np(x, ka, pp, xt, s0):
    inner_exp = np.exp((pp * x) / xt)
    rho = (-np.log(np.exp(-pp) + 1.0 / (inner_exp + 1.0)) / pp - s0) / (1.0 - s0)
    return rho

def compute_local_characteristic_np(X_mesh, Y_mesh, X, Y, L, h, T, r_gp, method='GP', eps_safe=1e-7, eps_mna=1.0, Z_mesh=None, Z=0.0, P=0.0):
    """
    Supports 2D and 3D round-ended bar characteristic functions.
    3D primitive is a cylinder with hemispherical ends.
    T: rotation in XY plane.
    P: pitch rotation (elevation).
    """
    dx = X_mesh - X
    dy = Y_mesh - Y
    
    if Z_mesh is not None:
        dz = Z_mesh - Z
        # 3D Rotation logic
        # Direction vector n based on angles T (theta) and P (phi)
        nx = np.cos(P) * np.cos(T)
        ny = np.cos(P) * np.sin(T)
        nz = np.sin(P)
        
        # Projection of p=(dx,dy,dz) onto n
        proj = dx*nx + dy*ny + dz*nz
        
        # Distance squared to the line segment
        # Segment goes from -L/2*n to +L/2*n
        t = np.clip(proj, -L/2.0, L/2.0)
        dist2 = (dx - t*nx)**2 + (dy - t*ny)**2 + (dz - t*nz)**2
        upsi = np.sqrt(np.maximum(eps_safe, dist2))
    else:
        # 2D Logic
        r2 = dx**2 + dy**2 + eps_safe
        r = np.sqrt(r2)
        phi = np.arctan2(dy, dx + eps_safe) - T
        abs_cos = np.sqrt(np.cos(phi)**2 + eps_safe)
        abs_sin = np.sqrt(np.sin(phi)**2 + eps_safe)
        
        upsi = np.where(
            r * abs_cos >= L / 2.0,
            np.sqrt(np.maximum(eps_safe, r2 + L**2 / 4.0 - r * L * abs_cos)),
            r * abs_sin
        )
    
    if method == 'GP':
        zetavar = upsi - h / 2.0
        z_clipped = np.clip(zetavar / r_gp, -1.0 + eps_safe, 1.0 - eps_safe)
        delta_iel = (1.0/np.pi * (np.arccos(z_clipped) - z_clipped * np.sqrt(1.0 - z_clipped**2)))
        return np.where(zetavar < -r_gp, 1.0, 
                        np.where(zetavar > r_gp, 0.0, delta_iel))
    else:
        l, u = h / 2.0 - eps_mna / 2.0, h / 2.0 + eps_mna / 2.0
        d_v = -(eps_mna**3)
        return np.where(upsi <= l, 1.0,
                        np.where(upsi <= u,
                                 (-2.0 / d_v) * upsi**3 + (3.0 * h / d_v) * upsi**2 + 
                                 (-6.0 * l * u / d_v) * upsi + (u * (-u**2 + 3.0 * l * u) / d_v),
                                 0.0))
