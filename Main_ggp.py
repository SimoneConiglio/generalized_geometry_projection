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

def run_main_ggp(bc_type="Short_Cantilever", max_iter=50, nelx=60, nely=30):
    print(f"\n=================================================================")
    print(f"   Running Main GGP (GEMSEO + FEniCS + petsc4py) | BC: {bc_type}")
    print(f"=================================================================\n")
    
    # Mesh definition
    if bc_type == "L-shape":
        L, H = 60.0, 60.0
        nelx, nely = 60, 60
    else:
        L, H = 60.0, 30.0
        nelx, nely = 60, 30
        
    volfrac = 0.4
    num_components = 18
    
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
        def left_symmetry(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
        def bottom_right_support(x, on_boundary): return on_boundary and df.near(x[0], L, 1.0) and df.near(x[1], 0.0, 1.0)
        bc = [
            DirichletBC(V_u.sub(0), Constant(0.0), left_symmetry),
            DirichletBC(V_u.sub(1), Constant(0.0), bottom_right_support, method="pointwise")
        ]
        dists = np.linalg.norm(dof_coords - np.array([0.0, H]), axis=1)
        top_y_dof = np.intersect1d(np.where(dists < 1e-3)[0], y_dofs)[0]
        f_vec[top_y_dof] = -1.0
        
    elif bc_type == "L-shape":
        def top_boundary(x, on_boundary): return on_boundary and df.near(x[1], H)
        bc = [DirichletBC(V_u, Constant((0.0, 0.0)), top_boundary)]
        dists = np.linalg.norm(dof_coords - np.array([L, H/2.0]), axis=1)
        mid_y_dof = np.intersect1d(np.where(dists < 1e-3)[0], y_dofs)[0]
        f_vec[mid_y_dof] = -1.0
        
        # Passive elements: top right quadrant
        midpoints = V_dg.tabulate_dof_coordinates()
        xc_mid, yc_mid = midpoints[:, 0], midpoints[:, 1]
        emptyelts = np.where((xc_mid >= L/2.0 - 1e-5) & (yc_mid >= H/2.0 - 1e-5))[0]
        
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    ds_load = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    
    solver = PhysicsFactory.create_solver("Elasticity", V_u=V_u, bc=bc, ds_load=ds_load, L_rhs_vec=Constant((0.0, 0.0)), p=3.0, plane_stress=True)
    
    # Exact GGP-MATLAB Initialization to replicate baseline coordinates
    ncx = 1
    ncy = 1
    xp = np.linspace(0.0, L, ncx + 2)
    yp = np.linspace(0.0, H, ncy + 2)
    xx, yy = np.meshgrid(xp, yp)
    xx_flat = xx.flatten(order='F')
    yy_flat = yy.flatten(order='F')
    
    Xc_init = np.tile(xx_flat, 2)
    Yc_init = np.tile(yy_flat, 2)
    Lc_init = 2.0 * np.sqrt((L / (ncx + 2))**2 + (H / (ncy + 2))**2) * np.ones_like(Xc_init)
    
    angle = np.arctan2(H / ncy, L / ncx)
    half_len = len(Xc_init) // 2
    Tc_init = angle * np.concatenate([np.ones(half_len), -np.ones(half_len)])
    hc_init = 2.0 * np.ones_like(Xc_init)
    Mc_init = 0.5 * np.ones_like(Xc_init)
    
    x_init = np.zeros(num_components * 6)
    x_init[0::6], x_init[1::6], x_init[2::6] = Xc_init, Yc_init, Lc_init
    x_init[3::6], x_init[4::6], x_init[5::6] = hc_init, Tc_init, Mc_init
    
    # Setup bounds exactly matching baseline
    Xl = (0.0 - 1.0) * np.ones_like(Xc_init)
    Xu = (L + 1.0) * np.ones_like(Xc_init)
    Yl = (0.0 - 1.0) * np.ones_like(Xc_init)
    Yu = (H + 1.0) * np.ones_like(Xc_init)
    Ll = np.zeros_like(Xc_init)
    Lu = np.sqrt(L**2 + H**2) * np.ones_like(Xc_init)
    hl = 1.0 * np.ones_like(Xc_init)
    hu = np.sqrt(L**2 + H**2) * np.ones_like(Xc_init)
    Tl = -2.0 * np.pi * np.ones_like(Xc_init)
    Tu = 2.0 * np.pi * np.ones_like(Xc_init)
    Ml = np.zeros_like(Xc_init)
    Mu = np.ones_like(Xc_init)
    
    lb = np.zeros(num_components * 6)
    lb[0::6], lb[1::6], lb[2::6] = Xl, Yl, Ll
    lb[3::6], lb[4::6], lb[5::6] = hl, Tl, Ml
    
    ub = np.zeros(num_components * 6)
    ub[0::6], ub[1::6], ub[2::6] = Xu, Yu, Lu
    ub[3::6], ub[4::6], ub[5::6] = hu, Tu, Mu
    
    geom_disc = GGPVectorizedGeometryDiscipline(mesh, num_components, mode='Free', L_domain=L, H_domain=H, Ngp=2, lb=lb, ub=ub)
    phys_disc = GGPHybridPhysicsDiscipline(solver, mesh, mesh_area=L*H, volfrac=volfrac, emptyelts=emptyelts, L=L, H=H, bc_type=bc_type, max_iter=max_iter)
    phys_disc.geom_disc = geom_disc
    
    phys_disc.f_vec = f_vec
    phys_disc.f_vec[phys_disc.fixed_dofs] = 0.0
    
    chain = MDAChain([geom_disc, phys_disc])
    
    # Scale design variable vector to [0, 1] for MMA optimizer matching Matlab behavior
    X = (x_init - lb) / (ub - lb)
    
    # MMA initialization
    m_mma = 1
    n_mma = len(X)
    eeen = np.ones(n_mma)
    eeem = np.ones(m_mma)
    zeron = np.zeros(n_mma)
    zerom = np.zeros(m_mma)
    xval = X.copy()
    xold1 = xval.copy()
    xold2 = xval.copy()
    xmin = zeron.copy()
    xmax = eeen.copy()
    low = xmin.copy()
    upp = xmax.copy()
    C = 1000.0 * eeem
    d = 0.0 * eeem
    a0 = 1.0
    a = zerom.copy()
    
    outit = 0
    print("\nStarting optimization loop...")
    while outit < max_iter:
        outit += 1
        
        # 1. Execute Geometry mapping
        geom_disc.execute({"x_vars": xval})
        rho_E = geom_disc.local_data["rho_E"]
        rho_V = geom_disc.local_data["rho_V"]
        
        # 2. Execute Physics solver
        phys_disc.execute({"rho_E": rho_E, "rho_V": rho_V})
        # Compliance and volume
        c_val = np.exp(phys_disc.local_data["compliance"][0]) - 1.0
        v_val = (phys_disc.local_data["volume"][0] / 100.0) * volfrac + volfrac
        
        # 3. Gradients calculation via GEMSEO linearization
        jac = geom_disc.linearize(input_data={"x_vars": xval}, compute_all_jacobians=True)
        jac_E = jac["rho_E"]["x_vars"]
        jac_V = jac["rho_V"]["x_vars"]
        
        df0dx = phys_disc.dj_drhoE @ jac_E
        dfdx = phys_disc.dv_drhoV @ jac_V
        
        # 4. MMA step
        f0val = np.log(c_val + 1.0)
        fval = np.array([(v_val - volfrac) / volfrac * 100.0])
        
        X, ymma, zmma, lam, xsi, eta, mu, zet, S, low, upp = mmasub(
            m_mma, n_mma, outit, xval, xmin, xmax, xold1, xold2,
            f0val, df0dx, fval, dfdx, low, upp, a0, a, C, d
        )
        
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = X.copy()
        
    # Final values printing
    if len(phys_disc.cvec) > 0:
        final_compliance = phys_disc.cvec[-1]
        final_volume = phys_disc.vvec[-1] * 100.0
        print(f"\nFinal Optimized Compliance: {final_compliance:.4f}")
        print(f"Final Volume Constraint: {final_volume:.4f}%")
        print(f"\nOptimization finished successfully! Outputs saved in directory: {phys_disc.Path}")
    else:
        print("Optimization did not run any evaluations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main GGP Optimization using GEMSEO, FEniCS, and petsc4py fast disciplines.")
    parser.add_argument("--bc", type=str, default="Short_Cantilever", choices=["Short_Cantilever", "MBB", "L-shape"], help="Boundary condition type.")
    parser.add_argument("--max_iter", type=int, default=50, help="Maximum number of optimization iterations.")
    args = parser.parse_args()
    
    run_main_ggp(bc_type=args.bc, max_iter=args.max_iter)
