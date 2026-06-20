import numpy as np
import pytest
from ggp.utils.vectorized_mapping_3d import compute_local_characteristic_3d_free_with_grad_np, compute_local_characteristic_3d_alm_with_grad_np

def fd_gradient_free(X_mesh, Y_mesh, Z_mesh, params, r_gp, method='GP', eps=1e-5):
    # params: [Xc, Yc, Zc, L, h, T, P]
    W_base, *_ = compute_local_characteristic_3d_free_with_grad_np(
        X_mesh, Y_mesh, Z_mesh, *params, r_gp, method=method
    )
    
    grads = []
    for i in range(len(params)):
        p_plus = list(params)
        p_plus[i] += eps
        W_plus, *_ = compute_local_characteristic_3d_free_with_grad_np(
            X_mesh, Y_mesh, Z_mesh, *p_plus, r_gp, method=method
        )
        
        p_minus = list(params)
        p_minus[i] -= eps
        W_minus, *_ = compute_local_characteristic_3d_free_with_grad_np(
            X_mesh, Y_mesh, Z_mesh, *p_minus, r_gp, method=method
        )
        
        grad_fd = (W_plus - W_minus) / (2 * eps)
        grads.append(grad_fd)
    
    return grads

def test_3d_free_analytical_gradients():
    # Setup a small 3D mesh grid
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z = np.linspace(-1, 1, 10)
    X, Y, Z = np.meshgrid(x, y, z)
    X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
    
    params = [0.1, -0.2, 0.3, 1.2, 0.5, np.pi/4, np.pi/6] # Xc, Yc, Zc, L, h, T, P
    r_gp = 0.2
    
    W_an, dX, dY, dZ, dL, dh, dT, dP = compute_local_characteristic_3d_free_with_grad_np(
        X_flat, Y_flat, Z_flat, *params, r_gp
    )
    
    grads_an = [dX, dY, dZ, dL, dh, dT, dP]
    grads_fd = fd_gradient_free(X_flat, Y_flat, Z_flat, params, r_gp)
    
    for i, name in enumerate(["X", "Y", "Z", "L", "h", "T", "P"]):
        err = np.abs(grads_an[i] - grads_fd[i]).max()
        assert err < 1e-4, f"Gradient mismatch for {name}: {err}"

def fd_gradient_alm(X_mesh, Y_mesh, Z_mesh, params, r_gp, method='GP', eps=1e-5):
    # params: [Xc, Yc, Zc, L, W, h, T]
    W_base, *_ = compute_local_characteristic_3d_alm_with_grad_np(
        X_mesh, Y_mesh, Z_mesh, *params, r_gp, method=method
    )
    
    grads = []
    for i in range(len(params)):
        p_plus = list(params)
        p_plus[i] += eps
        W_plus, *_ = compute_local_characteristic_3d_alm_with_grad_np(
            X_mesh, Y_mesh, Z_mesh, *p_plus, r_gp, method=method
        )
        
        p_minus = list(params)
        p_minus[i] -= eps
        W_minus, *_ = compute_local_characteristic_3d_alm_with_grad_np(
            X_mesh, Y_mesh, Z_mesh, *p_minus, r_gp, method=method
        )
        
        grad_fd = (W_plus - W_minus) / (2 * eps)
        grads.append(grad_fd)
    
    return grads

def test_3d_alm_analytical_gradients():
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z = np.linspace(-1, 1, 10)
    X, Y, Z = np.meshgrid(x, y, z)
    X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()
    
    params = [0.1, -0.2, 0.3, 1.2, 0.8, 0.5, np.pi/4] # Xc, Yc, Zc, L, W, h, T
    r_gp = 0.2
    
    W_an, dX, dY, dZ, dL, dW, dh, dT = compute_local_characteristic_3d_alm_with_grad_np(
        X_flat, Y_flat, Z_flat, *params, r_gp
    )
    
    grads_an = [dX, dY, dZ, dL, dW, dh, dT]
    grads_fd = fd_gradient_alm(X_flat, Y_flat, Z_flat, params, r_gp)
    
    for i, name in enumerate(["X", "Y", "Z", "L", "W", "h", "T"]):
        err = np.abs(grads_an[i] - grads_fd[i]).max()
        assert err < 1e-4, f"Gradient mismatch for {name}: {err}"
