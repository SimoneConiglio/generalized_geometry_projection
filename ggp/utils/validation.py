import numpy as np
import scipy.io as sio
import os

class StepByStepValidator:
    """
    Utility to export intermediate data to .mat files for step-by-step
    comparison with the MATLAB GGP implementation.
    """
    def __init__(self, output_dir="validation_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.iter = 0

    def export_iteration(self, x_vars, rho_E, rho_V, compliance, volume, gradients_J, gradients_V):
        """Dumps all data for a single iteration."""
        filename = os.path.join(self.output_dir, f"iter_{self.iter:03d}.mat")
        data = {
            "x": x_vars,
            "rho_E": rho_E,
            "rho_V": rho_V,
            "f0": compliance,
            "f1": volume,
            "df0": gradients_J,
            "df1": gradients_V
        }
        sio.savemat(filename, data)
        print(f"[Validator] Exported {filename}")
        self.iter += 1

    @staticmethod
    def compare_numpy(val_python, val_matlab, name, tol=1e-8):
        """Helper to compare arrays and print differences."""
        diff = np.abs(val_python - val_matlab)
        max_diff = np.max(diff)
        print(f"Comparison of {name}: Max Diff = {max_diff:.2e}")
        return max_diff < tol
