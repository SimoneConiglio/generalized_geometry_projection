import numpy as np
from ggp.utils.vectorized_mapping import compute_continuous_ALM_characteristic_np

nY = 30
np_val = 4

Xk = np.random.rand(nY * np_val)
Lk = np.random.rand(nY * np_val)
h = np.random.rand(nY)
Xc = np.concatenate([Xk, Lk, h])

x = np.random.rand(7200, 1)
y = np.random.rand(7200, 1)
Yk = np.random.rand(nY, np_val)
Yk = Yk.flatten(order='F')

p = {'method': 'GGP', 'gammac': 3.0}

W, dW_dX, dW_dL, dW_dh = compute_continuous_ALM_characteristic_np(x, y, Xc, p, np_val, nY, Yk, 30)
print(W.shape)
