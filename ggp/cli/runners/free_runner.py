import dolfin as df
from dolfin_adjoint import *
import numpy as np
import gemseo
from gemseo import create_scenario
from gemseo.mda.mda_chain import MDAChain
from ggp.geometry.factory import GeometryFactory
from ggp.physics.factory import PhysicsFactory
from ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline, GGPPhysicsFastDiscipline
import matplotlib.pyplot as plt
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "GGP-Matlab"))
from mmasub import mmasub

def norato_bar(xi, eta, L, h):
    """
    Computes the boundary points of Norato components for visualization.
    """
    xi2 = xi**2
    eta2 = eta**2
    denom = xi2 + eta2
    denom_safe = np.where(denom == 0, 1.0, denom)
    ratio = xi2 / denom_safe
    limit_ratio = L**2 / (h**2 + L**2)
    cond1 = (ratio >= limit_ratio)
    
    term1_a = (L / 2.0) * np.sqrt(ratio)
    sqrt_arg = h**2 / 4.0 - (L**2 / 4.0) * (eta2 / denom_safe)
    sqrt_arg = np.clip(sqrt_arg, 0.0, None)
    term1_b = np.sqrt(sqrt_arg)
    term1 = term1_a + term1_b
    
    term2 = (h / 2.0) * np.sqrt(1.0 + xi2 / (eta**2 + (eta == 0).astype(float)))
    d_part = np.where(cond1, term1, term2)
    non_zero = (xi != 0) | (eta != 0)
    d = np.where(non_zero, d_part, (np.sqrt(2.0) / 2.0) * h)
    return d

class GGPHybridPhysicsDiscipline(GGPPhysicsFastDiscipline):
    """
    Subclass of GGPPhysicsFastDiscipline adding support for passive/empty elements and post-processing plots.
    """
    def __init__(self, solver, mesh, mesh_area, volfrac, emptyelts=None, L=60.0, H=30.0, bc_type="Short_Cantilever", max_iter=50):
        super().__init__(solver, mesh, mesh_area, volfrac)
        self.emptyelts = emptyelts if emptyelts is not None else []
        self.L = L
        self.H = H
        self.bc_type = bc_type
        self.max_iter = max_iter
        
        # Pointwise BC fix for MBB:
        if self.bc_type == "MBB":
            dof_coords = self.V_u.tabulate_dof_coordinates()
            y_dofs = self.V_u.sub(1).dofmap().dofs()
            br_dists = np.linalg.norm(dof_coords - np.array([self.L, 0.0]), axis=1)
            br_node_dofs = np.where(br_dists < 1e-3)[0]
            br_y_dof = np.intersect1d(br_node_dofs, y_dofs)[0]
            self.fixed_dofs.append(br_y_dof)
            self.fixed_dofs = sorted(list(set(self.fixed_dofs)))
        self.geom_disc = None
        self.cvec = []
        self.vvec = []
        self.plot_rate = 10
        
        R_val = 0.5
        Ngp_val = 2
        rs = f"{R_val:3.2f}".replace('.', '_')
        
        nelx = int(L) if bc_type == "L-shape" else 60
        nely = int(H) if bc_type == "L-shape" else 30
        self.folder_name = f"Optimization_history_{bc_type}GPnelx_{nelx}nely_{nely}_R_{rs}_Ngp_{Ngp_val}_SC_change"
        self.image_prefix = f"{bc_type}GPnelx_{nelx}nely_{nely}_R_{rs}_Ngp_{Ngp_val}"
        
        os.makedirs(self.folder_name, exist_ok=True)
        self.Path = self.folder_name + "/"
        
        self.tt = np.arange(0, 2.0 * np.pi + 0.005, 0.005)
        self.cc_angle = np.cos(self.tt)
        self.ss_angle = np.sin(self.tt)

    def _run(self, input_data=None):
        if input_data is not None:
            self.local_data.update(input_data)
        
        rho_E = self.local_data["rho_E"].flatten().copy()
        rho_V = self.local_data["rho_V"].flatten().copy()
        
        E0 = float(self.solver.E0)
        E_vals = rho_E * E0
        
        if len(self.emptyelts) > 0:
            rho_E[self.emptyelts] = 0.0
            rho_V[self.emptyelts] = 0.0
            Emin = float(self.solver.Emin)
            E_vals[self.emptyelts] = Emin
        
        # Assemble global stiffness
        from scipy.sparse import coo_matrix
        dofs_per_cell = self.ke_ref.shape[0]
        rows = np.repeat(self.cell_dofs, dofs_per_cell).reshape(self.num_elements, dofs_per_cell, dofs_per_cell)
        cols = np.tile(self.cell_dofs, (1, dofs_per_cell)).reshape(self.num_elements, dofs_per_cell, dofs_per_cell)
        data = np.outer(E_vals, self.ke_ref).reshape(self.num_elements, dofs_per_cell, dofs_per_cell)
        K_global = coo_matrix((data.flatten(), (rows.flatten(), cols.flatten())), shape=(self.V_u.dim(), self.V_u.dim())).tocsr()
        
        # BCs
        col_mask = np.isin(K_global.indices, self.fixed_dofs)
        K_global.data[col_mask] = 0.0
        for dof in self.fixed_dofs:
            dof_start = K_global.indptr[dof]
            dof_end = K_global.indptr[dof+1]
            K_global.data[dof_start:dof_end] = 0.0
            col_indices = K_global.indices[dof_start:dof_end]
            diag_idx = np.where(col_indices == dof)[0]
            if len(diag_idx) > 0:
                K_global.data[dof_start + diag_idx[0]] = 1.0
                
        # Solve
        from scipy.sparse.linalg import spsolve
        u_vec = spsolve(K_global, self.f_vec)
        self.last_u = u_vec
        
        compliance = np.dot(self.f_vec, u_vec)
        volume = np.sum(rho_V) * (self.mesh_area / self.num_elements) / self.mesh_area
        
        self.local_data["compliance"] = np.array([np.log(compliance + 1.0)])
        self.local_data["volume"] = np.array([(volume - self.volfrac) / self.volfrac * 100.0])
        
        # Gradients
        dE_drho = E0
        grad_C = np.zeros(self.num_elements)
        for i in range(self.num_elements):
            u_e = u_vec[self.cell_dofs[i]]
            grad_C[i] = -dE_drho * np.dot(u_e, np.dot(self.ke_ref, u_e))
        
        self.dj_drhoE = grad_C / (compliance + 1.0)
        self.dv_drhoV = np.ones(self.num_elements) * (100.0 / (self.volfrac * self.num_elements))
        if len(self.emptyelts) > 0:
            self.dj_drhoE[self.emptyelts] = 0.0
            self.dv_drhoV[self.emptyelts] = 0.0
            
        # Callback logic inside the discipline loop
        c_val = compliance
        v_val = volume
        self.cvec.append(c_val)
        self.vvec.append(v_val)
        
        outit = len(self.cvec)
        print(f" It.:{outit:5d} Obj.:{c_val:4.3e} Vol.:{v_val:7.3f}")
        
        if outit % self.plot_rate == 0 or outit == self.max_iter:
            if self.geom_disc is not None and "x_vars" in self.geom_disc.local_data:
                x_vars = self.geom_disc.local_data["x_vars"].flatten()
                if self.geom_disc.lb is not None and self.geom_disc.ub is not None:
                    x_vars = self.geom_disc.lb + x_vars * (self.geom_disc.ub - self.geom_disc.lb)
                
                # 1. Convergence plot
                plt.figure(3)
                plt.clf()
                plt.subplot(2, 1, 1)
                plt.plot(np.arange(1, outit + 1), self.cvec, 'bo-', label='C')
                plt.scatter(outit, c_val, color='k')
                plt.grid(True)
                plt.xlabel('iter')
                plt.ylabel('C')
                plt.title(f"C = {c_val:.2f} at iteration {outit}")
                
                plt.subplot(2, 1, 2)
                plt.plot(np.arange(1, outit + 1), np.array(self.vvec) * 100, 'ro-', label='V')
                plt.scatter(outit, v_val * 100, color='k')
                plt.grid(True)
                plt.xlabel('iter')
                plt.ylabel('V [%]')
                plt.title(f"V = {v_val*100:.2f}% at iteration {outit}")
                plt.tight_layout()
                plt.savefig(f"{self.Path}{self.image_prefix}convergence.png")
                plt.close()
                
                # 2. Plot Densities
                plt.figure(1)
                plt.clf()
                nelx = int(self.L) if self.bc_type == "L-shape" else 60
                nely = int(self.H) if self.bc_type == "L-shape" else 30
                dx = self.L / nelx
                dy = self.H / nely
                cols = np.round((self.geom_disc.X_mesh - dx/2.0) / dx).astype(int)
                rows = np.round((self.geom_disc.Y_mesh - dy/2.0) / dy).astype(int)
                xPhys = np.zeros((nely, nelx))
                xPhys[rows, cols] = rho_V
                
                plt.imshow(1.0 - xPhys, cmap='gray', origin='lower', extent=[0.0, self.L, 0.0, self.H])
                plt.colorbar()
                plt.axis('equal')
                plt.axis('off')
                plt.title(f"Density at iteration {outit}")
                plt.savefig(f"{self.Path}density_{outit-1:03d}.png")
                plt.close()
                
                # 3. Component Plot
                plt.figure(2)
                plt.clf()
                Xc = x_vars[0::6]
                Yc = x_vars[1::6]
                Lc = x_vars[2::6]
                hc = x_vars[3::6]
                Tc = x_vars[4::6]
                Mc = x_vars[5::6]
                
                num_comps = len(Xc)
                cc_angle_mat = np.tile(self.cc_angle, (num_comps, 1))
                ss_angle_mat = np.tile(self.ss_angle, (num_comps, 1))
                C0 = np.tile(np.cos(Tc).reshape(-1, 1), (1, len(self.tt)))
                S0 = np.tile(np.sin(Tc).reshape(-1, 1), (1, len(self.tt)))
                xxx = np.tile(Xc.reshape(-1, 1), (1, len(self.tt))) + cc_angle_mat
                yyy = np.tile(Yc.reshape(-1, 1), (1, len(self.tt))) + ss_angle_mat
                xi = C0 * (xxx - Xc.reshape(-1, 1)) + S0 * (yyy - Yc.reshape(-1, 1))
                Eta = -S0 * (xxx - Xc.reshape(-1, 1)) + C0 * (yyy - Yc.reshape(-1, 1))
                Lc_mat = np.tile(Lc.reshape(-1, 1), (1, len(self.tt)))
                hc_mat = np.tile(hc.reshape(-1, 1), (1, len(self.tt)))
                dd = norato_bar(xi, Eta, Lc_mat, hc_mat)
                
                xn = Xc.reshape(-1, 1) + dd * cc_angle_mat
                yn = Yc.reshape(-1, 1) + dd * ss_angle_mat
                
                tolshow = 0.1
                Shown_compo = np.where(Mc > tolshow)[0]
                
                plt.xlim(0.0, self.L)
                plt.ylim(0.0, self.H)
                
                from matplotlib.patches import Polygon as MPolygon
                if self.bc_type == 'L-shape':
                    clip_poly = MPolygon([
                        (0.0, 0.0),
                        (self.L, 0.0),
                        (self.L, self.H / 2.0),
                        (self.L / 2.0, self.H / 2.0),
                        (self.L / 2.0, self.H),
                        (0.0, self.H)
                    ], transform=plt.gca().transData)
                    plt.fill([self.L/2.0, self.L, self.L, self.L/2.0],
                             [self.H/2.0, self.H/2.0, self.H, self.H], 'w', zorder=10)
                else:
                    clip_poly = MPolygon([
                        (0.0, 0.0),
                        (self.L, 0.0),
                        (self.L, self.H),
                        (0.0, self.H)
                    ], transform=plt.gca().transData)
                    
                plt.gca().add_patch(clip_poly)
                clip_poly.set_visible(False)
                
                cmap_jet = plt.get_cmap('jet')
                for idx in Shown_compo:
                    color = cmap_jet(Mc[idx])
                    polys = plt.fill(xn[idx, :], yn[idx, :], facecolor=color, alpha=0.5)
                    for poly in polys:
                        poly.set_clip_path(clip_poly)
                        
                plt.axis('equal')
                plt.axis('off')
                plt.savefig(f"{self.Path}component_{outit-1:03d}.png")
                plt.close()

def run_main_ggp(bc_type="Short_Cantilever", max_iter=50, mode="Free", algorithm="MMA", use_line_search=False, L_opt=None, H_opt=None, nelx_opt=None, nely_opt=None, volfrac_opt=0.4):
    print(f"\n=================================================================")
    print(f"   Running Main GGP (GEMSEO + FEniCS + petsc4py) | BC: {bc_type} | Mode: {mode} | Algo: {algorithm} | Line Search: {use_line_search}")
    print(f"=================================================================\n")
    
    # Mesh definition
    if bc_type == "L-shape":
        L = L_opt if L_opt is not None else 60.0
        H = H_opt if H_opt is not None else 60.0
        nelx = nelx_opt if nelx_opt is not None else 60
        nely = nely_opt if nely_opt is not None else 60
    else:
        L = L_opt if L_opt is not None else 60.0
        H = H_opt if H_opt is not None else 30.0
        nelx = nelx_opt if nelx_opt is not None else 60
        nely = nely_opt if nely_opt is not None else 30
        
    volfrac = volfrac_opt
    
    mesh = df.RectangleMesh.create([df.Point(0, 0), df.Point(L, H)], [nelx, nely], df.CellType.Type.quadrilateral)
    V_u = df.VectorFunctionSpace(mesh, "CG", 1)
    V_dg = df.FunctionSpace(mesh, "DG", 0)
    
    # Load and fixed DOFs identification
    f_vec = np.zeros(V_u.dim())
    bc = []
    emptyelts = []
    
    dof_coords = V_u.tabulate_dof_coordinates()
    y_dofs = V_u.sub(1).dofmap().dofs()
    
    if bc_type == "Short_Cantilever":
        def left_boundary(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
        bc = [DirichletBC(V_u, Constant((0.0, 0.0)), left_boundary)]
        dists = np.linalg.norm(dof_coords - np.array([L, H/2.0]), axis=1)
        tip_y_dof = np.intersect1d(np.where(dists < 1e-3)[0], y_dofs)[0]
        f_vec[tip_y_dof] = -1.0
    elif bc_type == "MBB":
        def left_sym(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
        def right_supp(x, on_boundary): return on_boundary and df.near(x[0], L) and df.near(x[1], 0.0)
        bc = [
            DirichletBC(V_u.sub(0), Constant(0.0), left_sym),
            DirichletBC(V_u.sub(1), Constant(0.0), right_supp)
        ]
        dists = np.linalg.norm(dof_coords - np.array([0.0, H]), axis=1)
        tip_y_dof = np.intersect1d(np.where(dists < 1e-3)[0], y_dofs)[0]
        f_vec[tip_y_dof] = -1.0
    elif bc_type == "L-shape":
        def top_boundary(x, on_boundary): return on_boundary and df.near(x[1], H)
        bc = [DirichletBC(V_u, Constant((0.0, 0.0)), top_boundary)]
        dists = np.linalg.norm(dof_coords - np.array([L, H/2.0]), axis=1)
        tip_y_dof = np.intersect1d(np.where(dists < 1e-3)[0], y_dofs)[0]
        f_vec[tip_y_dof] = -1.0
        
        centers_x = dof_coords[::2, 0]
        centers_y = dof_coords[::2, 1]
        emptyelts_mask = (centers_x > L/2.0) & (centers_y > H/2.0)
        # Convert mask to element indices (assuming regular numbering, approx)
        x_centers = np.linspace(0.5, L-0.5, nelx)
        y_centers = np.linspace(0.5, H-0.5, nely)
        XX, YY = np.meshgrid(x_centers, y_centers)
        elts_mask = (XX > L/2.0) & (YY > H/2.0)
        emptyelts = np.where(elts_mask.flatten(order='F'))[0].tolist()
        volfrac = 0.4
        
    ds_load = df.Measure("ds", domain=mesh)
    
    # Physics Solver creation
    solver = PhysicsFactory.create_solver("Elasticity", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=Constant((0.0, 0.0)), p=3.0, plane_stress=True)
    
    # Exact GGP-MATLAB Initialization to replicate baseline coordinates
    if mode == "Free":
        ncx, ncy = 1, 1
        num_components = 2*(ncx + 2)*(ncy + 2)
        # Nodal coordinates (0-based)
        Xx_idx, Yy_idx = np.where(np.ones((1 + nely, 1 + nelx)).T)
        Xx = Xx_idx.astype(float)
        Yy = (nely - Yy_idx).astype(float)
        xp = np.linspace(np.min(Xx), np.max(Xx), ncx + 2)
        yp = np.linspace(np.min(Yy), np.max(Yy), ncy + 2)
        xx, yy = np.meshgrid(xp, yp)
        xx_flat = xx.flatten(order='F')
        yy_flat = yy.flatten(order='F')
        
        Xc_init = np.tile(xx_flat, 2)
        Yc_init = np.tile(yy_flat, 2)
        Lc_init = 2.0 * np.sqrt((nelx / (ncx + 2))**2 + (nely / (ncy + 2))**2) * np.ones_like(Xc_init)
        
        angle = np.arctan2(nely / ncy, nelx / ncx)
        half_len = len(Xc_init) // 2
        Tc_init = angle * np.concatenate([np.ones(half_len), -np.ones(half_len)])
        hc_init = 2.0 * np.ones_like(Xc_init)
        initial_d = 0.5 
        Mc_init = initial_d * np.ones_like(Xc_init)
        
        x_init = np.column_stack([Xc_init, Yc_init, Lc_init, hc_init, Tc_init, Mc_init]).flatten()
        
        # Build upper and lower bounds
        Xl = (np.min(Xx) - 1.0) * np.ones_like(Xc_init)
        Xu = (np.max(Xx) + 1.0) * np.ones_like(Xc_init)
        Yl = (np.min(Yy) - 1.0) * np.ones_like(Xc_init)
        Yu = (np.max(Yy) + 1.0) * np.ones_like(Xc_init)
        Ll = np.zeros_like(Xc_init)
        Lu = np.sqrt(nelx**2 + nely**2) * np.ones_like(Xc_init)
        minh =1
        hl = minh * np.ones_like(Xc_init)
        hu = np.sqrt(nelx**2 + nely**2) * np.ones_like(Xc_init)
        Tl = -2.0 * np.pi * np.ones_like(Xc_init)
        Tu = 2.0 * np.pi * np.ones_like(Xc_init)
        Ml = np.zeros_like(Xc_init)
        Mu = np.ones_like(Xc_init)
        
        lb = np.column_stack([Xl, Yl, Ll, hl, Tl, Ml]).flatten()
        ub = np.column_stack([Xu, Yu, Lu, hu, Tu, Mu]).flatten()
        
        kwargs = {}
    elif mode == "ALM":
        num_layers = 10
        comp_per_layer = 5
        layer_height = H / num_layers
        num_components = num_layers * comp_per_layer
        
        x_init = np.zeros(num_components * 3)
        lb = np.zeros_like(x_init)
        ub = np.zeros_like(x_init)
        
        x_c = np.linspace(L/(2*comp_per_layer), L - L/(2*comp_per_layer), comp_per_layer)
        Xc_init = np.tile(x_c, num_layers)
        
        x_init[0::3] = Xc_init
        x_init[1::3] = 10.0  # Width
        x_init[2::3] = 0.5   # Mc
        
        lb[0::3] = -1.0
        ub[0::3] = L + 1.0
        lb[1::3] = 0.0
        ub[1::3] = np.sqrt(L**2 + H**2)
        lb[2::3] = 0.0
        ub[2::3] = 1.0
        
        kwargs = {'num_layers': num_layers, 'comp_per_layer': comp_per_layer, 'layer_height': layer_height}
    
    
    phys_disc = GGPHybridPhysicsDiscipline(solver, mesh, mesh_area=L*H, volfrac=volfrac, emptyelts=emptyelts, L=L, H=H, bc_type=bc_type, max_iter=max_iter)
    
    phys_disc.f_vec = f_vec
    phys_disc.f_vec[phys_disc.fixed_dofs] = 0.0
    
    # GEMSEO Design Space
    from gemseo import create_scenario, create_design_space
    from gemseo.core.discipline.discipline import Discipline
    from ggp.utils.alm_utils import create_alm_overhang_constraints
    
    current_x = (x_init - lb) / (ub - lb)
    r_gp_values = [1.5, 1.0, 0.5] if mode != 'Free' else [0.5]
    
    class OverhangDiscipline(Discipline):
        def __init__(self, num_layers, comp_per_layer, layer_height, alpha_deg, lb, ub):
            super().__init__("OverhangDiscipline")
            self.input_grammar.update_from_names(["x_vars"])
            self.output_grammar.update_from_names(["overhang_cons"])
            self.default_inputs = {"x_vars": np.zeros(num_layers * comp_per_layer * 3)}
            self.A, self.b = create_alm_overhang_constraints(num_layers, comp_per_layer, layer_height, alpha_deg, extended=False)
            self.lb = lb
            self.ub = ub
            
        def _run(self, input_data=None):
            if input_data is not None:
                self.local_data.update(input_data)
            x = self.local_data["x_vars"]
            x_unscaled = self.lb + x * (self.ub - self.lb)
            cons = self.A @ x_unscaled - self.b
            self.local_data["overhang_cons"] = cons
            
        def _compute_jacobian(self, inputs=None, outputs=None):
            jac = self.A * (self.ub - self.lb)
            self.jac = {"overhang_cons": {"x_vars": np.atleast_2d(jac)}}
    
    for step_idx, r in enumerate(r_gp_values):
        print(f"\n{'='*50}")
        print(f"   Starting Optimization Phase {step_idx+1}/{len(r_gp_values)} | r_gp = {r}")
        print(f"{'='*50}")
        
        geom_disc = GGPVectorizedGeometryDiscipline(mesh, num_components, mode=mode, L_domain=L, H_domain=H, Ngp=2, r_gp=r, pp=100.0, lb=lb, ub=ub, **kwargs)
        phys_disc.geom_disc = geom_disc
        chain = MDAChain([geom_disc, phys_disc])
        
        # Deactivate gradient history storage on the chain
        if hasattr(chain, 'cache'): chain.cache = None
        if hasattr(chain, 'cache_type'): chain.cache_type = Discipline.CacheType.NONE
        
        disciplines_to_add = [chain]
        if mode == "ALM":
            cons_disc = OverhangDiscipline(num_layers, comp_per_layer, layer_height, 45.0, lb, ub)
            if hasattr(cons_disc, 'cache'): cons_disc.cache = None
            if hasattr(cons_disc, 'cache_type'): cons_disc.cache_type = Discipline.CacheType.NONE
            disciplines_to_add.append(cons_disc)
            
        design_space = create_design_space()
        design_space.add_variable("x_vars", size=len(current_x), lower_bound=0.0, upper_bound=1.0, value=current_x)
        
        scenario = create_scenario(
            disciplines_to_add, "compliance", design_space, maximize_objective=False, formulation_name="MDF"
        )
        scenario.add_constraint("volume", constraint_type="ineq", positive=False, value=0.0)
        if mode == "ALM":
            scenario.add_constraint("overhang_cons", constraint_type="ineq", positive=False, value=0.0)
        
        algo_options = {
            "algo_name": algorithm,
            "max_iter": max_iter,
            "use_line_search": use_line_search,
            "max_optimization_step": 0.05,
            "max_asymptote_distance":0.05,
            "initial_asymptotes_distance": 0.05,
            "min_asymptote_distance":0.001,
            "use_penalty_formulation": True,
            "ftol_rel": 1e-12,
            "xtol_rel": 1e-12,
            "ftol_abs": 1e-12,
            "xtol_abs": 1e-12,
            "max_inner_iterations": 500,
        }
        scenario.execute(**algo_options)
        
        current_x = scenario.design_space.get_current_value().copy()
        
    # Final values printing
    if len(phys_disc.cvec) > 0:
        final_compliance = phys_disc.cvec[-1]
        final_volume = phys_disc.vvec[-1] * 100.0
        print(f"\nFinal Optimized Compliance: {final_compliance:.4f}")
        print(f"Final Volume Constraint: {final_volume:.4f}%")
        print(f"\nOptimization finished successfully! Outputs saved in directory: {phys_disc.Path}")
    else:
        print("Optimization did not run any evaluations.")

