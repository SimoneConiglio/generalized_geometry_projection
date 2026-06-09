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
        self.eps_mna = Constant(eps_mna) 
        self.r_gp = Constant(r_gp)       
        self.deltamin = Constant(deltamin)
        self.x_c = SpatialCoordinate(mesh)
        
        # Calculate saturation constants (Eq. 71-72)
        xt_v = 1.0 + 1.0/ka * np.log((1.0 + (num_components - 1.0)*np.exp(-ka))/num_components)
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
        
        if self.method == 'GP':
            zetavar = upsi - h / 2.0
            z_clipped = ufl.max_value(-1.0 + self.eps_safe, ufl.min_value(1.0 - self.eps_safe, zetavar / self.r_gp))
            delta_iel = (1.0/np.pi * (ufl.acos(z_clipped) - z_clipped * ufl.sqrt(1.0 - z_clipped**2)))
            delta_final = ufl.conditional(zetavar < -self.r_gp, 1.0, 
                                          ufl.conditional(zetavar > self.r_gp, 0.0, delta_iel))
            return self.deltamin + (1.0 - self.deltamin) * delta_final
        else:
            l, u = h / 2.0 - self.eps_mna / 2.0, h / 2.0 + self.eps_mna / 2.0
            d_v = -(self.eps_mna**3)
            return ufl.conditional(upsi <= l, Constant(1.0), 
                ufl.conditional(upsi <= u, 
                    (Constant(-2.0) / d_v) * upsi**3 + (Constant(3.0) * h / d_v) * upsi**2 + 
                    (Constant(-6.0) * l * u / d_v) * upsi + (u * (-u**2 + Constant(3.0) * l * u) / d_v), 
                    Constant(0.0)))

    def map_to_density(self, ctrls):
        parameters["form_compiler"]["quadrature_degree"] = 4
        char_functions = []
        for i in range(self.num_components):
            idx = i * 6
            W_i = self._compute_local_characteristic(ctrls[idx], ctrls[idx+1], ctrls[idx+2], ctrls[idx+3], ctrls[idx+4])
            char_functions.append(W_i * ctrls[idx+5])
        sum_exp = sum([ufl.exp(self.ka * d) for d in char_functions])
        ks_val = (1.0 / self.ka) * ufl.ln(sum_exp / self.num_components)
        inner_exp = ufl.exp((self.pp * ks_val) / self.xt)
        rho = (-ufl.ln(ufl.exp(-self.pp) + 1.0 / (inner_exp + 1.0)) / self.pp - self.s0) / (1.0 - self.s0)
        return rho

    def get_initial_design(self, L_domain, H_domain):
        # EXACT GGP-MATLAB INITIALIZATION
        nc_half = self.num_components // 2
        ncx = int(np.round(np.sqrt(nc_half * L_domain / H_domain)))
        ncy = int(np.round(nc_half / ncx))
        
        xp = np.linspace(0, L_domain, ncx + 2)
        yp = np.linspace(0, H_domain, ncy + 2)
        xx, yy = np.meshgrid(xp[1:-1], yp[1:-1])
        
        xc_base = xx.flatten()
        yc_base = yy.flatten()
        
        # Crossed grid (X-shape)
        Xc = np.concatenate([xc_base, xc_base])
        Yc = np.concatenate([yc_base, yc_base])
        
        Tc = np.zeros(len(Xc))
        angle = np.pi / 4.0
        Tc[:len(xc_base)] = angle
        Tc[len(xc_base):] = -angle
        
        L_init = 2 * np.sqrt((L_domain / (ncx + 2))**2 + (H_domain / (ncy + 2))**2)
        Lc = L_init * np.ones(len(Xc))
        hc = 2.0 * np.ones(len(Xc))
        Mc = 0.5 * np.ones(len(Xc))
        
        # Match exact num_components
        if len(Xc) > self.num_components:
            Xc, Yc, Lc, hc, Tc, Mc = Xc[:self.num_components], Yc[:self.num_components], Lc[:self.num_components], hc[:self.num_components], Tc[:self.num_components], Mc[:self.num_components]
        elif len(Xc) < self.num_components:
            diff = self.num_components - len(Xc)
            Xc = np.append(Xc, [L_domain/2]*diff)
            Yc = np.append(Yc, [H_domain/2]*diff)
            Lc = np.append(Lc, [L_init]*diff)
            hc = np.append(hc, [2.0]*diff)
            Tc = np.append(Tc, [0.0]*diff)
            Mc = np.append(Mc, [0.5]*diff)
            
        x_init = np.zeros(self.num_components * 6)
        x_init[0::6], x_init[1::6], x_init[2::6] = Xc, Yc, Lc
        x_init[3::6], x_init[4::6], x_init[5::6] = hc, Tc, Mc
        return x_init
