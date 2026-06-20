import numpy as np
import pytest
from ggp.utils.vectorized_mapping import (
    compute_continuous_ALM_characteristic_np,
    compute_local_characteristic_np
)

def test_compute_continuous_ALM_characteristic_np():
    # Dummy data for compute_continuous_ALM_characteristic_np
    x_mesh = np.array([0.0, 1.0])
    y_mesh = np.array([0.0, 1.0])
    
    np_val = 1
    nY = 3
    # Xc contains Xk (nY*np), Lk (nY*np), h (nY)
    Xc = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    p = 10.0
    Yk = np.array([0.1, 0.5, 0.9])
    nely = 10
    
    W, grad_X, grad_L, grad_h = compute_continuous_ALM_characteristic_np(
        x_mesh, y_mesh, Xc, p, np_val, nY, Yk, nely
    )
    assert W.shape[1] == nY

def test_compute_local_characteristic_np_3d():
    x_mesh = np.array([0.0, 1.0])
    y_mesh = np.array([0.0, 1.0])
    z_mesh = np.array([0.0, 1.0])
    
    W, grad_W = compute_local_characteristic_np(
        X_mesh=x_mesh, Y_mesh=y_mesh, X=0.5, Y=0.5, L=0.2, h=0.2, T=0.0, r_gp=0.1,
        method='GP', Z_mesh=z_mesh, Z=0.5, P=0.0, W_width=0.2
    )
    assert W.shape[0] == 2

