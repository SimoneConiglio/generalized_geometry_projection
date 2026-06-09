from abc import ABC, abstractmethod
import numpy as np

class BaseMapper(ABC):
    """
    Abstract Base Class for Geometry Mapping in GGP.
    Maps design variables to a density field.
    """
    @abstractmethod
    def map_to_density(self, design_variables):
        """
        Maps the primitive parameters to the element densities.
        """
        pass

    @abstractmethod
    def get_initial_design(self):
        """
        Returns the initial guess for the design variables.
        """
        pass
