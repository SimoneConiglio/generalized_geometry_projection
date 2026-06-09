from abc import ABC, abstractmethod

class BaseSolver(ABC):
    """
    Abstract Base Class for Physics Solvers in GGP.
    Handles the forward FEA and adjoint sensitivity analysis.
    """
    @abstractmethod
    def solve(self, density_field):
        """
        Solves the forward physics problem (e.g., linear elasticity).
        """
        pass

    @abstractmethod
    def compute_compliance(self, u):
        """
        Computes the objective function (compliance).
        """
        pass
