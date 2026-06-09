from dolfin import *
from dolfin_adjoint import *
import ufl
import numpy as np
from ..geometry.base_mapper import BaseMapper

class GGP2DMapper(BaseMapper):
    """
    2D Generalized Geometry Projection Mapper.
    Implementation aligned with GGP-Matlab (Coniglio et al. 2020).
    """
    def __init__(self, mesh, num_components, method='GP', ka=10.0, pp=100.0, 
                 eps_safe=1e-7, eps_mna=1.0, r_gp=0.5, deltamin=1e-6):
        self.mesh = mesh
        self.num_components = num_components
        self.method = method
        
        self.ka = Constant(ka)
        self.pp = Constant(pp)
        self.eps_safe = Constant(eps_safe)
        self.eps_mna = Constant(eps_mna) # p.sigma in MATLAB
        self.r_gp = Constant(r_gp)       # p.r in MATLAB
        self.deltamin = Constant(deltamin)
        self.x_c = SpatialCoordinate(mesh)
        
        # Calculate saturation constants (Eq. 71-72)
        xt_v = 1.0 + 1.0/ka * np.log((1.0 + (num_components - 1.0)*np.exp(-ka))/num_components)
        s0_v = -np.log(np.exp(-pp) + 1.0 / (np.exp(0.0) + 1.0)) / pp
        self.xt = Constant(xt_v)
        self.s0 = Constant(s0_v)

    def _compute_local_characteristic(self, X, Y, L, h, T):
        """Computes W_i(x,y) for a single component."""
        dx, dy = self.x_c[0] - X, self.x_c[1] - Y
        r2 = dx**2 + dy**2 + self.eps_safe
        r = ufl.sqrt(r2)
        
        # Local coordinate system rotation
        phi = ufl.atan_2(dy, dx + self.eps_safe) - T
        
        # Distance function upsi for round-ended bar
        abs_cos = ufl.sqrt(ufl.cos(phi)**2 + self.eps_safe)
        abs_sin = ufl.sqrt(ufl.sin(phi)**2 + self.eps_safe)
        
        upsi = ufl.conditional(
            r * abs_cos >= L / 2.0, 
            ufl.sqrt(ufl.max_value(self.eps_safe, r2 + L**2 / 4.0 - r * L * abs_cos)), 
            r * abs_sin
        )
        
        if self.method == 'GP':
            # Geometry Projection: circular window integral (Section 2.2)
            zetavar = upsi - h / 2.0
            # Piecewise smooth approximation of the circular area
            # Using ufl.acos and ufl.sqrt for the analytical circular projection
            # To avoid numerical issues at bounds, we clip zetavar/r_gp
            z_clipped = ufl.max_value(-1.0 + self.eps_safe, ufl.min_value(1.0 - self.eps_safe, zetavar / self.r_gp))
            
            delta_iel = (1.0/np.pi * (ufl.acos(z_clipped) - z_clipped * ufl.sqrt(1.0 - z_clipped**2)))
            # Conditional for outside the transition radius
            delta_final = ufl.conditional(zetavar < -self.r_gp, 1.0, 
                                          ufl.conditional(zetavar > self.r_gp, 0.0, delta_iel))
            
            return self.deltamin + (1.0 - self.deltamin) * delta_final

        else:
            # Moving Node Approach: Polynomial Regularization (Section 2.3)
            l, u = h / 2.0 - self.eps_mna / 2.0, h / 2.0 + self.eps_mna / 2.0
            d_v = -(self.eps_mna**3)
            
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
        """Maps all components to global density field."""
        # Set quadrature degree
        parameters["form_compiler"]["quadrature_degree"] = 4
        
        # Each component has 6 variables in MATLAB: Xc, Yc, L, h, T, M
        char_functions = []
        for i in range(self.num_components):
            idx = i * 6
            W_i = self._compute_local_characteristic(ctrls[idx], ctrls[idx+1], ctrls[idx+2], ctrls[idx+3], ctrls[idx+4])
            # Multiply by mass variable Mc (ctrls[idx+5])
            char_functions.append(W_i * ctrls[idx+5])
        
        # Saturated KS Aggregation (Lower Bound version 'KSl')
        sum_exp = sum([ufl.exp(self.ka * d) for d in char_functions])
        ks_val = (1.0 / self.ka) * ufl.ln(sum_exp / self.num_components)
        
        # Smooth Saturation
        inner_exp = ufl.exp((self.pp * ks_val) / self.xt)
        rho = (-ufl.ln(ufl.exp(-self.pp) + 1.0 / (inner_exp + 1.0)) / self.pp - self.s0) / (1.0 - self.s0)
        
        return rho

    def get_initial_design(self, L_domain, H_domain):
        """Generates the crossed grid of components matching MATLAB initialization."""
        # Match GGP-Matlab crossed grid initialization
        ncx = max(1, int(np.sqrt(self.num_components / 2 * (L_domain / H_domain))))
        ncy = max(1, int((self.num_components / 2) / ncx))
        
        xp = np.linspace(0, L_domain, ncx + 2)
        yp = np.linspace(0, H_domain, ncy + 2)
        xx, yy = np.meshgrid(xp, yp)
        
        Xc = np.tile(xx.flatten(), 2)
        Yc = np.tile(yy.flatten(), 2)
        
        # Ensure we have exactly num_components
        if len(Xc) > self.num_components:
            Xc, Yc = Xc[:self.num_components], Yc[:self.num_components]
        elif len(Xc) < self.num_components:
            Xc = np.append(Xc, [L_domain/2] * (self.num_components - len(Xc)))
            Yc = np.append(Yc, [H_domain/2] * (self.num_components - len(Yc)))
            
        Lc = 2 * np.sqrt((L_domain / (ncx + 2))**2 + (H_domain / (ncy + 2))**2) * np.ones(self.num_components)
        
        # Alternating angles for crossed grid
        base_angle = np.arctan2(H_domain / ncy, L_domain / ncx)
        half_len = self.num_components // 2
        Tc = np.zeros(self.num_components)
        Tc[:half_len] = base_angle
        Tc[half_len:] = -base_angle
        
        hc = 2.0 * np.ones(self.num_components) # p.minh
        Mc = 0.5 * np.ones(self.num_components) # initial_d
        
        x_init = np.zeros(self.num_components * 6)
        x_init[0::6] = Xc
        x_init[1::6] = Yc
        x_init[2::6] = Lc
        x_init[3::6] = hc
        x_init[4::6] = Tc
        x_init[5::6] = Mc
        
        return x_init
