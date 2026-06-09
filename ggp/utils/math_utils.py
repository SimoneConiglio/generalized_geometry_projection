import numpy as np

def smooth_saturation(x, kappa_s, xt=1.0):
    """
    Regularized saturation function to aggregate densities safely.
    Matches Eq 71 from Bhat et al.
    """
    sb_x = - (1.0 / kappa_s) * np.log(np.exp(-kappa_s) + 1.0 / (1.0 + np.exp(kappa_s * x / xt)))
    sb_0 = - (1.0 / kappa_s) * np.log(np.exp(-kappa_s) + 1.0 / (1.0 + np.exp(0.0)))
    return (sb_x - sb_0) / (1.0 - sb_0)

def ks_aggregation(densities, kappa, axis=0):
    """
    Kreisselmeier-Steinhauser (KS) aggregation function.
    """
    n = densities.shape[axis]
    return (1.0 / kappa) * np.log(np.mean(np.exp(kappa * densities), axis=axis))
