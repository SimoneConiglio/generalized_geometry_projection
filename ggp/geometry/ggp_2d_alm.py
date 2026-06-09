from dolfin import *
from dolfin_adjoint import *
import ufl
import numpy as np
from .base_mapper import BaseMapper

class GGP2DALMMapper(BaseMapper):
    """
    ALM-specific GGP Mapper.
    Components are restricted to horizontal layers with fixed height and zero rotation.
    Design variables per component: [Xc, width, Mc]
    """
    def __init__(self, mesh, num_layers, components_per_layer, layer_height, 
                 method='GP', ka=10.0, pp=100.0, eps_safe=1e-7, r_gp=0.5, deltamin=1e-6):
        self.mesh = mesh
        self.num_layers = num_layers
        self.comp_per_layer = components_per_layer
        self.num_components = num_layers * components_per_layer
        self.layer_height = layer_height
        self.method = method
        
        self.ka = Constant(ka)
        self.pp = Constant(pp)
        self.eps_safe = Constant(eps_safe)
        self.r_gp = Constant(r_gp)
        self.deltamin = Constant(deltamin)
        self.x_c = SpatialCoordinate(mesh)
        
        # Saturation constants
        xt_v = 1.0 + 1.0/ka * np.log((1.0 + (self.num_components - 1.0)*np.exp(-ka))/self.num_components)
        s0_v = -np.log(np.exp(-pp) + 1.0 / (np.exp(0.0) + 1.0)) / pp
        self.xt = Constant(xt_v)
        self.s0 = Constant(s0_v)

    def _compute_local_characteristic(self, X, Y, L, h, T):
        dx, dy = self.x_c[0] - X, self.x_c[1] - Y
        r2 = dx**2 + dy**2 + self.eps_safe
        r = ufl.sqrt(r2)
        phi = ufl.atan_2(dy, dx + self.eps_safe) - T
        
        abs_cos = ufl.sqrt(ufl.cos(phi)**2 + self.eps_safe)
        abs_sin = ufl.sqrt(ufl.sin(phi)**2 + self.eps_safe)
        
        upsi = ufl.conditional(
            r * abs_cos >= L / 2.0, 
            ufl.sqrt(ufl.max_value(self.eps_safe, r2 + L**2 / 4.0 - r * L * abs_cos)), 
            r * abs_sin
        )
        
        # We always use GP for ALM as it's the paper standard
        zetavar = upsi - h / 2.0
        z_clipped = ufl.max_value(-1.0 + self.eps_safe, ufl.min_value(1.0 - self.eps_safe, zetavar / self.r_gp))
        delta_iel = (1.0/np.pi * (ufl.acos(z_clipped) - z_clipped * ufl.sqrt(1.0 - z_clipped**2)))
        return ufl.conditional(zetavar < -self.r_gp, 1.0, 
                               ufl.conditional(zetavar > self.r_gp, 0.0, delta_iel))

    def map_to_density(self, ctrls):
        """
        ctrls: flattened array of [Xc, width, Mc] for each component.
        Length: num_components * 3
        """
        parameters["form_compiler"]["quadrature_degree"] = 4
        char_functions = []
        
        h_fixed = Constant(self.layer_height)
        theta_fixed = Constant(0.0)
        
        for layer in range(self.num_layers):
            y_fixed = Constant((layer + 0.5) * self.layer_height)
            for i in range(self.comp_per_layer):
                idx = (layer * self.comp_per_layer + i) * 3
                # variables: Xc=ctrls[idx], width=ctrls[idx+1], Mc=ctrls[idx+2]
                W_i = self._compute_local_characteristic(ctrls[idx], y_fixed, ctrls[idx+1], h_fixed, theta_fixed)
                char_functions.append(W_i * ctrls[idx+2])
        
        sum_exp = sum([ufl.exp(self.ka * d) for d in char_functions])
        ks_val = (1.0 / self.ka) * ufl.ln(sum_exp / self.num_components)
        
        inner_exp = ufl.exp((self.pp * ks_val) / self.xt)
        rho = (-ufl.ln(ufl.exp(-self.pp) + 1.0 / (inner_exp + 1.0)) / self.pp - self.s0) / (1.0 - self.s0)
        return rho

    def get_initial_design(self, L_domain, H_domain):
        x_init = []
        for layer in range(self.num_layers):
            for i in range(self.comp_per_layer):
                xc = (i + 1) * L_domain / (self.comp_per_layer + 1)
                width = L_domain / (self.comp_per_layer)
                mc = 0.5
                x_init.extend([xc, width, mc])
        return np.array(x_init)
