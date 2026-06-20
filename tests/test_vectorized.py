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
    p = {'method': 'GP', 'deltamin': 1e-6, 'r': 0.1, 'alp': 4, 'epsi': 1e-4, 'bet': 2}
    Yk = np.array([0.1, 0.5, 0.9])
    nely = 10
    
    W, grad_X, grad_L, grad_h = compute_continuous_ALM_characteristic_np(
        x_mesh, y_mesh, Xc, p, np_val, nY, Yk, nely
    )
    assert len(W) > 0

def test_compute_local_characteristic_np_3d():
    x_mesh = np.array([0.0, 1.0])
    y_mesh = np.array([0.0, 1.0])
    z_mesh = np.array([0.0, 1.0])
    
    W = compute_local_characteristic_np(
        X_mesh=x_mesh, Y_mesh=y_mesh, X=0.5, Y=0.5, L=0.2, h=0.2, T=0.0, r_gp=0.1,
        method='GP', Z_mesh=z_mesh, Z=0.5, P=0.0, W_width=0.2
    )
    assert len(W) > 0

def test_smooth_saturation_np():
    from ggp.utils.vectorized_mapping import smooth_saturation_np
    res = smooth_saturation_np(np.array([0.5]), 1.0, 1.0, 1.0, 0.1)
    assert res.shape == (1,)

def test_compute_local_characteristic_2d_with_grad_np():
    from ggp.utils.vectorized_mapping import compute_local_characteristic_2d_with_grad_np
    x_mesh = np.array([0.0, 1.0])
    y_mesh = np.array([0.0, 1.0])
    
    methods = ['GP', 'ALM_GP', 'AMNA']
    for m in methods:
        res = compute_local_characteristic_2d_with_grad_np(
            X_mesh=x_mesh, Y_mesh=y_mesh, X=0.5, Y=0.5, L=0.2, h=0.2, T=0.0, r_gp=0.1, method=m
        )
        assert len(res) == 6
        assert res[0].shape == (2,)

def test_compute_local_characteristic_np_methods():
    x_mesh = np.array([0.0, 1.0])
    y_mesh = np.array([0.0, 1.0])
    
    methods = ['GP', 'ALM_GP', 'AMNA', 'MNA']
    for m in methods:
        W = compute_local_characteristic_np(
            X_mesh=x_mesh, Y_mesh=y_mesh, X=0.5, Y=0.5, L=0.2, h=0.2, T=0.0, r_gp=0.1, method=m
        )
        assert W.shape == (2,)

def test_compute_continuous_ALM_characteristic_np_MMC():
    x_mesh = np.array([0.0, 1.0])
    y_mesh = np.array([0.0, 1.0])
    np_val = 1
    nY = 3
    Xc = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    p = {'method': 'MMC', 'alp': 4, 'epsi': 1e-4, 'bet': 2, 'ka': 100, 'aggregation': 'KS'}
    Yk = np.array([0.1, 0.5, 0.9])
    nely = 10
    
    W, grad_X, grad_L, grad_h = compute_continuous_ALM_characteristic_np(
        x_mesh, y_mesh, Xc, p, np_val, nY, Yk, nely
    )
    assert len(W) > 0

