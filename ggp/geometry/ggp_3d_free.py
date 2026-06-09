import numpy as np
from .base_mapper import BaseMapper

class GGP3DMapper(BaseMapper):
    """
    3D Generalized Geometry Projection Mapper (Initial Design only).
    Mapping is handled by GGPVectorizedGeometryDiscipline.
    """
    def __init__(self, mesh, num_components):
        self.mesh = mesh
        self.num_components = num_components

    def map_to_density(self, design_variables):
        # Implementation in GGPVectorizedGeometryDiscipline
        pass

    def get_initial_design(self, L, H, D):
        """
        L: Length (X), H: Height (Y), D: Depth (Z)
        Generates a 3D crossed-grid of components.
        8 variables: [X, Y, Z, L, h, Theta, Phi, Mc]
        """
        # Determine number of points along each axis for a uniform grid
        # nc^3 ~ num_components
        nc = int(np.ceil(np.power(self.num_components, 1/3)))
        if nc < 2: nc = 2
        
        x = np.linspace(L/4, 3*L/4, nc)
        y = np.linspace(H/4, 3*H/4, nc)
        z = np.linspace(D/4, 3*D/4, nc)
        xx, yy, zz = np.meshgrid(x, y, z)
        
        Xc, Yc, Zc = xx.flatten(), yy.flatten(), zz.flatten()
        
        # Trim if we have too many
        if len(Xc) > self.num_components:
            Xc, Yc, Zc = Xc[:self.num_components], Yc[:self.num_components], Zc[:self.num_components]
        
        num_act = len(Xc)
        Lc = (L/nc) * np.ones(num_act)
        hc = (H/nc) * np.ones(num_act)
        Tc = np.zeros(num_act)
        Pc = np.zeros(num_act)
        Mc = 0.5 * np.ones(num_act)
        
        # Pad if we have too few
        if num_act < self.num_components:
            diff = self.num_components - num_act
            Xc = np.append(Xc, [L/2]*diff)
            Yc = np.append(Yc, [H/2]*diff)
            Zc = np.append(Zc, [D/2]*diff)
            Lc = np.append(Lc, [L/nc]*diff)
            hc = np.append(hc, [H/nc]*diff)
            Tc = np.append(Tc, [0]*diff)
            Pc = np.append(Pc, [0]*diff)
            Mc = np.append(Mc, [0.5]*diff)
            
        x_init = np.zeros(self.num_components * 8)
        x_init[0::8], x_init[1::8], x_init[2::8] = Xc, Yc, Zc
        x_init[3::8], x_init[4::8] = Lc, hc
        x_init[5::8], x_init[6::8], x_init[7::8] = Tc, Pc, Mc
        
        return x_init
