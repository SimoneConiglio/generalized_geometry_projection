import pytest
import numpy as np
import os
from ggp.utils.validation import StepByStepValidator

def test_validator_export(tmpdir):
    validator = StepByStepValidator(output_dir=str(tmpdir))
    
    x_vars = np.array([0.5, 0.5])
    rho_E = np.array([1.0, 1.0])
    rho_V = np.array([0.5, 0.5])
    compliance = 10.0
    volume = 0.5
    gradients_J = np.array([-1.0, -1.0])
    gradients_V = np.array([0.1, 0.1])
    
    validator.export_iteration(x_vars, rho_E, rho_V, compliance, volume, gradients_J, gradients_V)
    
    expected_file = os.path.join(str(tmpdir), "iter_000.mat")
    assert os.path.exists(expected_file)
    
def test_validator_compare():
    val_py = np.array([1.0, 2.0, 3.0])
    val_mat = np.array([1.0, 2.000000001, 3.0])
    
    # Should pass comparison
    assert StepByStepValidator.compare_numpy(val_py, val_mat, "test_arr", tol=1e-6)
    
    # Should fail comparison
    assert not StepByStepValidator.compare_numpy(val_py, val_mat, "test_arr", tol=1e-10)
