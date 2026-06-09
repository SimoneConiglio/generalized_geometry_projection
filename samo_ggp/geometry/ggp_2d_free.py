from dolfin import *
from dolfin_adjoint import *
import ufl
import numpy as np
from ..geometry.base_mapper import BaseMapper

class GGP2DMapper(BaseMapper):
    """
    2D Generalized Geometry Projection Mapper.
    Implementation of smooth density mapping from primitives to a FEniCS mesh.
    """
    def __init__(self, mesh, num_components, ka=10.0, pp=100.0, eps_safe=1e-7, eps_mna=3.0):
        self.mesh = mesh
        self.num_components = num_components
        
        self.ka = Constant(ka)
        self.pp = Constant(pp)
        self.eps_safe = Constant(eps_safe)
        self.eps_mna = Constant(eps_mna)
        self.x_c = SpatialCoordinate(mesh)
        
        # Calculate saturation constants
        xt_v = 1.0 + 1.0/ka * np.log((1.0 + (num_components - 1.0)*np.exp(-ka))/num_components)
        s0_v = -np.log(np.exp(-pp) + 1.0 / (np.exp(0.0) + 1.0)) / pp
        self.xt = Constant(xt_v)
        self.s0 = Constant(s0_v)

    def _compute_local_density(self, X, Y, L, h, T):
        dx, dy = self.x_c[0] - X, self.x_c[1] - Y
        r2 = dx**2 + dy**2 + self.eps_safe
        r = ufl.sqrt(r2)
        phi = ufl.atan_2(dy, dx + self.eps_safe) - T
        
        # Smoothing absolute values for gradients
        abs_cos = ufl.sqrt(ufl.cos(phi)**2 + self.eps_safe)
        abs_sin = ufl.sqrt(ufl.sin(phi)**2 + self.eps_safe)
        
        upsi = ufl.conditional(
            r * abs_cos >= L / 2.0, 
            ufl.sqrt(ufl.max_value(self.eps_safe, r2 + L**2 / 4.0 - r * L * abs_cos)), 
            r * abs_sin
        )
        
        l, u = h / 2.0 - self.eps_mna / 2.0, h / 2.0 + self.eps_mna / 2.0
        d_v = -(self.eps_mna**3)
        
        # Regularized Heaviside mapping
        return ufl.conditional(
            upsi <= l, Constant(1.0), 
            ufl.conditional(
                upsi <= u, 
                (Constant(-2.0) / d_v) * upsi**3 + (Constant(3.0) * h / d_v) * upsi**2 + 
                (Constant(-6.0) * l * u / d_v) * upsi + (u * (-u**2 + Constant(3.0) * l * u) / d_v), 
                Constant(0.0)
            )
        )

    def map_to_density(self, ctrls):
        # Set quadrature degree to prevent overflow in complex UFL graphs
        parameters["form_compiler"]["quadrature_degree"] = 4
        
        densities = [self._compute_local_density(ctrls[i*5], ctrls[i*5+1], ctrls[i*5+2], ctrls[i*5+3], ctrls[i*5+4]) 
                     for i in range(self.num_components)]
        
        # Saturated KS Aggregation
        sum_exp = sum([ufl.exp(self.ka * d) for d in densities])
        ks_val = (1.0 / self.ka) * ufl.ln(sum_exp / self.num_components)
        inner_exp = ufl.exp((self.pp * ks_val) / self.xt)
        
        rho = (-ufl.ln(ufl.exp(-self.pp) + 1.0 / (inner_exp + 1.0)) / self.pp - self.s0) / (1.0 - self.s0)
        return rho

    def get_initial_design(self, L_domain, H_domain):
        x_init = []
        for i in range(self.num_components):
            x_init.extend([L_domain/2.0, (i+0.5)*H_domain/self.num_components, L_domain*0.5, H_domain/self.num_components, 0.1])
        return np.array(x_init)
