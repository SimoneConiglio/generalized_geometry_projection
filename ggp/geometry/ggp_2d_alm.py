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
                 method='AMNA', ka=10.0, pp=100.0, eps_safe=1e-7, r_gp=0.5, deltamin=1e-6):
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
        if self.method == 'AMNA':
            dx = self.x_c[0] - X
            dy = self.x_c[1] - Y
            cos_t, sin_t = ufl.cos(T), ufl.sin(T)
            x_loc = dx * cos_t + dy * sin_t
            y_loc = -dx * sin_t + dy * cos_t
            
            zeta1 = -L/2.0 - x_loc
            zeta2 = x_loc - L/2.0
            zeta3 = y_loc - h/2.0
            zeta4 = -h/2.0 - y_loc
            zeta5 = y_loc - h/2.0
            
            sigma = self.r_gp
            
            def smooth_heaviside_ufl(zeta, sig):
                val = 0.5 - (15.0 / (16.0 * sig)) * zeta + (5.0 / (8.0 * sig**3)) * (zeta**3) - (3.0 / (16.0 * sig**5)) * (zeta**5)
                return ufl.conditional(zeta < -sig, 1.0, ufl.conditional(zeta > sig, 0.0, val))
                
            W1 = smooth_heaviside_ufl(zeta1, sigma)
            W2 = smooth_heaviside_ufl(zeta2, sigma)
            W3 = smooth_heaviside_ufl(zeta3, sigma)
            W4 = smooth_heaviside_ufl(zeta4, sigma)
            W5 = smooth_heaviside_ufl(zeta5, sigma)
            
            return W1 * W2 * W3 * W4 * W5
        else:
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
            
            zetavar = upsi - h / 2.0
            z_clipped = ufl.max_value(-1.0 + self.eps_safe, ufl.min_value(1.0 - self.eps_safe, zetavar / self.r_gp))
            delta_iel = (1.0/np.pi * (ufl.acos(z_clipped) - z_clipped * ufl.sqrt(1.0 - z_clipped**2)))
            return ufl.conditional(zetavar < -self.r_gp, 1.0, 
                                   ufl.conditional(zetavar > self.r_gp, 0.0, delta_iel))

    def map_to_density(self, ctrls):
        """
        ctrls: flattened array of variables.
        If length is num_components * 3: uses [Xc, width, Mc] with y0=0, theta0=0
        If length is num_components * 4 + 2: uses [Xc, width, Mc, hc] + [y0, theta0]
        """
        parameters["form_compiler"]["quadrature_degree"] = 4
        char_functions = []
        
        is_extended = len(ctrls) == (self.num_components * 4 + 2)
        if is_extended:
            y0 = ctrls[-2]
            theta0 = ctrls[-1]
            num_vars = 4
        else:
            y0 = Constant(0.0)
            theta0 = Constant(0.0)
            num_vars = 3
        
        for layer in range(self.num_layers):
            y_fixed = (layer + 0.5) * self.layer_height
            for i in range(self.comp_per_layer):
                idx = (layer * self.comp_per_layer + i) * num_vars
                Xc = ctrls[idx]
                width = ctrls[idx+1]
                Mc = ctrls[idx+2]
                
                if is_extended:
                    hc = ctrls[idx+3]
                else:
                    hc = Constant(self.layer_height)
                
                # Transform to global rotated coordinates
                Xc_g = Xc * ufl.cos(theta0) - y_fixed * ufl.sin(theta0)
                Yc_g = y0 + Xc * ufl.sin(theta0) + y_fixed * ufl.cos(theta0)
                
                W_i = self._compute_local_characteristic(Xc_g, Yc_g, width, hc, theta0)
                char_functions.append(W_i * Mc)
        
        sum_exp = sum([ufl.exp(self.ka * d) for d in char_functions])
        ks_val = (1.0 / self.ka) * ufl.ln(sum_exp / self.num_components)
        
        inner_exp = ufl.exp((self.pp * ks_val) / self.xt)
        rho = (-ufl.ln(ufl.exp(-self.pp) + 1.0 / (inner_exp + 1.0)) / self.pp - self.s0) / (1.0 - self.s0)
        return rho

    def get_initial_design(self, L_domain, H_domain, extended=False):
        x_init = []
        for layer in range(self.num_layers):
            for i in range(self.comp_per_layer):
                xc = (i + 1) * L_domain / (self.comp_per_layer + 1)
                width = L_domain / (self.comp_per_layer)
                mc = 0.5
                if extended:
                    x_init.extend([xc, width, mc, self.layer_height])
                else:
                    x_init.extend([xc, width, mc])
        
        if extended:
            x_init.extend([0.0, 0.0]) # y0, theta0
            
        return np.array(x_init)
