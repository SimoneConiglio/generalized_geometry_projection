import numpy as np
from dolfin import SpatialCoordinate, Constant
import ufl
from .base_mapper import BaseMapper

class GGP3DMapper(BaseMapper):
    """
    3D Generalized Geometry Projection Mapper.
    Provides FEniCS auto-differentiated UFL expressions for 3D Free elements.
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
        
        xt_v = 1.0 + 1.0/ka * np.log((1.0 + (num_components - 1.0)*np.exp(-ka))/num_components)
        s0_v = -np.log(np.exp(-pp) + 1.0 / (np.exp(0.0) + 1.0)) / pp
        self.xt = Constant(xt_v)
        self.s0 = Constant(s0_v)

    def _compute_local_characteristic(self, X, Y, Z, L, h, T, P):
        dx, dy, dz = self.x_c[0] - X, self.x_c[1] - Y, self.x_c[2] - Z
        
        nx = ufl.cos(P) * ufl.cos(T)
        ny = ufl.cos(P) * ufl.sin(T)
        nz = ufl.sin(P)
        
        proj = dx*nx + dy*ny + dz*nz
        t_val = ufl.conditional(ufl.lt(proj, -L/2.0), -L/2.0, 
                    ufl.conditional(ufl.gt(proj, L/2.0), L/2.0, proj))
        
        dist2 = (dx - t_val*nx)**2 + (dy - t_val*ny)**2 + (dz - t_val*nz)**2
        upsi = ufl.sqrt(ufl.max_value(self.eps_safe, dist2))
        
        if self.method == 'GP':
            zetavar = upsi - h / 2.0
            z_clipped = ufl.max_value(-1.0 + self.eps_safe, ufl.min_value(1.0 - self.eps_safe, zetavar / self.r_gp))
            delta_iel = (1.0/np.pi * (ufl.acos(z_clipped) - z_clipped * ufl.sqrt(1.0 - z_clipped**2)))
            delta_final = ufl.conditional(zetavar < -self.r_gp, 1.0, 
                                          ufl.conditional(zetavar > self.r_gp, 0.0, delta_iel))
            return self.deltamin + (1.0 - self.deltamin) * delta_final
        else:
            # Fallback to GP for unsupported method in 3D
            raise NotImplementedError("Only GP is currently supported for 3D UFL mapping.")

    def map_to_density(self, design_variables):
        """
        Maps design variables to a UFL expression for density in 3D.
        Expects design_variables as [Xc, Yc, Zc, L, h, Theta, Phi, Mc] * num_components.
        """
        ks_sum = 0
        for i in range(self.num_components):
            start = i * 8
            W = self._compute_local_characteristic(
                design_variables[start], design_variables[start+1], design_variables[start+2],
                design_variables[start+3], design_variables[start+4], design_variables[start+5],
                design_variables[start+6]
            )
            m_comp = design_variables[start+7]
            ks_sum += ufl.exp(self.ka * W * m_comp)
            
        ks_val = (1.0 / self.ka) * ufl.ln(ks_sum / self.num_components)
        
        inner_exp = ufl.exp(self.pp * ks_val / self.xt)
        rho_ufl = (-ufl.ln(ufl.exp(-self.pp) + 1.0 / (inner_exp + 1.0)) / self.pp - self.s0) / (1.0 - self.s0)
        return rho_ufl

    def get_initial_design(self, L, H, D):
        nc = int(np.ceil(np.power(self.num_components, 1/3)))
        if nc < 2: nc = 2
        
        x = np.linspace(L/4, 3*L/4, nc)
        y = np.linspace(H/4, 3*H/4, nc)
        z = np.linspace(D/4, 3*D/4, nc)
        xx, yy, zz = np.meshgrid(x, y, z)
        
        Xc, Yc, Zc = xx.flatten(), yy.flatten(), zz.flatten()
        
        if len(Xc) > self.num_components:
            Xc, Yc, Zc = Xc[:self.num_components], Yc[:self.num_components], Zc[:self.num_components]
        
        num_act = len(Xc)
        Lc = (L/nc) * np.ones(num_act)
        hc = (H/nc) * np.ones(num_act)
        Tc = np.zeros(num_act)
        Pc = np.zeros(num_act)
        Mc = 0.5 * np.ones(num_act)
        
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
