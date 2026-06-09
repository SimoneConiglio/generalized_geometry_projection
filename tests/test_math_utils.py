import numpy as np
import pytest
from ggp.utils.math_utils import smooth_saturation, ks_aggregation

def test_smooth_saturation():
    """Test the regularized saturation bounding."""
    kappa_s = 10.0
    # Values inside range
    x_in = 0.5
    res_in = smooth_saturation(x_in, kappa_s)
    assert 0.0 <= res_in <= 1.0
    
    # Values outside range (should saturate near 1)
    x_high = 5.0
    res_high = smooth_saturation(x_high, kappa_s)
    assert np.isclose(res_high, 1.0, atol=1e-2)

def test_ks_aggregation():
    """Test Kreisselmeier-Steinhauser aggregation."""
    densities = np.array([0.1, 0.5, 0.9])
    kappa = 10.0
    res = ks_aggregation(densities, kappa)
    
    # KS with mean() approximates the max, but can be slightly lower due to the 1/N factor.
    # The true maximum is bounded by res + ln(N)/kappa
    assert res >= np.max(densities) - np.log(len(densities))/kappa
    assert res <= np.max(densities) + np.log(len(densities))/kappa
