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

# Import unopy
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Uno/build"))
import unopy

def norato_bar(xi, eta, L, h):
    xi2 = xi**2; eta2 = eta**2; denom = xi2 + eta2
    denom_safe = np.where(denom == 0, 1.0, denom)
    ratio = xi2 / denom_safe; limit_ratio = L**2 / (h**2 + L**2); cond1 = (ratio >= limit_ratio)
    term1 = (L/2.0)*np.sqrt(ratio) + np.sqrt(np.clip(h**2/4.0 - (L**2/4.0)*(eta2/denom_safe), 0.0, None))
    term2 = (h/2.0)*np.sqrt(1.0 + xi2/(eta**2 + (eta==0).astype(float)))
    d_part = np.where(cond1, term1, term2)
    return np.where((xi != 0)|(eta != 0), d_part, (np.sqrt(2.0)/2.0)*h)

class GGPHybridPhysicsDiscipline(GGPPhysicsFastDiscipline):
    def __init__(self, solver, mesh, mesh_area, volfrac, emptyelts, L, H, bc_type, max_iter, preset):
        super().__init__(solver, mesh, mesh_area, volfrac)
        self.emptyelts = emptyelts or []; self.L, self.H, self.bc_type, self.max_iter = L, H, bc_type, max_iter
        if self.bc_type == "MBB":
            dof_coords = self.V_u.tabulate_dof_coordinates(); y_dofs = self.V_u.sub(1).dofmap().dofs()
            br_dists = np.linalg.norm(dof_coords - np.array([self.L, 0.0]), axis=1)
            self.fixed_dofs.append(np.intersect1d(np.where(br_dists < 1e-3)[0], y_dofs)[0])
            self.fixed_dofs = sorted(list(set(self.fixed_dofs)))
        self.cvec, self.vvec, self.plot_rate = [], [], 100
        self.Path = f"Optimization_history_{bc_type}_{preset}/"
        os.makedirs(self.Path, exist_ok=True)
        self.tt = np.arange(0, 2.*np.pi+0.005, 0.005); self.cc_angle, self.ss_angle = np.cos(self.tt), np.sin(self.tt)

    def _run(self, input_data=None):
        if input_data: self.local_data.update(input_data)
        rho_E, rho_V = self.local_data["rho_E"].flatten().copy(), self.local_data["rho_V"].flatten().copy()
        E0, Emin = float(self.solver.E0), float(self.solver.Emin)
        E_vals = rho_E * E0
        if self.emptyelts: rho_E[self.emptyelts] = 0.0; rho_V[self.emptyelts] = 0.0; E_vals[self.emptyelts] = Emin
        from scipy.sparse import coo_matrix; dpc = self.ke_ref.shape[0]
        rows = np.repeat(self.cell_dofs, dpc).reshape(self.num_elements, dpc, dpc)
        cols = np.tile(self.cell_dofs, (1, dpc)).reshape(self.num_elements, dpc, dpc)
        data = np.outer(E_vals, self.ke_ref).reshape(self.num_elements, dpc, dpc)
        K = coo_matrix((data.flatten(), (rows.flatten(), cols.flatten())), shape=(self.V_u.dim(), self.V_u.dim())).tocsr()
        for d in self.fixed_dofs:
            K.data[K.indptr[d]:K.indptr[d+1]] = 0.0
            di = np.where(K.indices[K.indptr[d]:K.indptr[d+1]] == d)[0]
            if len(di)>0: K.data[K.indptr[d]+di[0]] = 1.0
        from scipy.sparse.linalg import spsolve; u = spsolve(K, self.f_vec); compliance = np.dot(self.f_vec, u)
        volume = np.sum(rho_V)*(self.mesh_area/self.num_elements)/self.mesh_area
        # EXACT BASELINE SCALING
        self.local_data["compliance"], self.local_data["volume"] = np.array([np.log(compliance+1.0)]), np.array([(volume-self.volfrac)/self.volfrac*100.0])
        grad_C = np.zeros(self.num_elements)
        for i in range(self.num_elements): grad_C[i] = -E0*np.dot(u[self.cell_dofs[i]], np.dot(self.ke_ref, u[self.cell_dofs[i]]))
        self.dj_drhoE, self.dv_drhoV = grad_C/(compliance+1.0), np.ones(self.num_elements)*(100.0/(self.volfrac*self.num_elements))
        if self.emptyelts: self.dj_drhoE[self.emptyelts] = 0.0; self.dv_drhoV[self.emptyelts] = 0.0
        self.cvec.append(compliance); self.vvec.append(volume); outit = len(self.cvec)
        if outit % self.plot_rate == 0 or outit == self.max_iter:
            if self.geom_disc is not None and "x_vars" in self.geom_disc.local_data:
                plt.figure(3); plt.clf(); plt.subplot(2,1,1); plt.plot(self.cvec, 'bo-'); plt.subplot(2,1,2); plt.plot(np.array(self.vvec)*100, 'ro-'); plt.savefig(f"{self.Path}convergence.png"); plt.close()

def run_main_ggp(bc_type, max_iter, mode, solver_name, preset, hessian, radius, mechanism, max_step):
    print(f"--- Running GGP | Solver: {solver_name} | Preset: {preset} ---")
    if bc_type == "L-shape": L, H, nelx, nely = 60.0, 60.0, 60, 60
    else: L, H, nelx, nely = 60.0, 30.0, 60, 30
    volfrac = 0.4
    mesh = df.RectangleMesh.create([df.Point(0, 0), df.Point(L, H)], [nelx, nely], df.CellType.Type.quadrilateral)
    V_u = df.VectorFunctionSpace(mesh, "CG", 1); f_vec = np.zeros(V_u.dim()); bc = []; emptyelts = []
    dof_coords = V_u.tabulate_dof_coordinates(); y_dofs = V_u.sub(1).dofmap().dofs()
    if bc_type == "Short_Cantilever":
        bc = [DirichletBC(V_u, Constant((0.0, 0.0)), "on_boundary && near(x[0], 0.0)")]
        tip = np.intersect1d(np.where(np.linalg.norm(dof_coords - np.array([L, H/2.0]), axis=1) < 1e-3)[0], y_dofs)[0]; f_vec[tip] = -1.0
    solver = PhysicsFactory.create_solver("Elasticity", V_u=V_u, bc=bc, ds_load=df.Measure("ds", domain=mesh), L_rhs_vec=Constant((0.0, 0.0)), p=3.0, plane_stress=True)
    if mode == "Free":
        ncx, ncy = 1, 1; num_components = 2*(ncx+2)*(ncy+2)
        xp = np.linspace(0, nelx, ncx+2); yp = np.linspace(0, nely, ncy+2); xx, yy = np.meshgrid(xp, yp)
        xx_f, yy_f = xx.flatten(order='F'), yy.flatten(order='F')
        xi = np.tile(xx_f, 2); yi = np.tile(yy_f, 2)
        li = 2.0*np.sqrt((nelx/3)**2+(nely/3)**2)*np.ones_like(xi); hi = 2.0*np.ones_like(xi); mi = 0.5*np.ones_like(xi)
        ai = np.arctan2(nely, nelx)*np.concatenate([np.ones(len(xi)//2), -np.ones(len(xi)//2)])
        x_init = np.column_stack([xi, yi, li, hi, ai, mi]).flatten()
        # EXACT BASELINE BOUNDS
        xl_b = (np.min(xx_f)-1.)*np.ones_like(xi); xu_b = (np.max(xx_f)+1.)*np.ones_like(xi)
        yl_b = (np.min(yy_f)-1.)*np.ones_like(xi); yu_b = (np.max(yy_f)+1.)*np.ones_like(xi)
        ll_b = np.zeros_like(xi); lu_b = np.sqrt(nelx**2+nely**2)*np.ones_like(xi)
        hl_b = np.ones_like(xi); hu_b = np.sqrt(nelx**2+nely**2)*np.ones_like(xi)
        al_b = -2.*np.pi*np.ones_like(xi); au_b = 2.*np.pi*np.ones_like(xi)
        ml_b = np.zeros_like(xi); mu_b = np.ones_like(xi)
        lb = np.column_stack([xl_b, yl_b, ll_b, hl_b, al_b, ml_b]).flatten()
        ub = np.column_stack([xu_b, yu_b, lu_b, hu_b, au_b, mu_b]).flatten()
    
    phys_disc = GGPHybridPhysicsDiscipline(solver, mesh, L*H, volfrac, emptyelts, L, H, bc_type, max_iter, preset)
    phys_disc.f_vec = f_vec; phys_disc.f_vec[phys_disc.fixed_dofs] = 0.0
    geom_disc = GGPVectorizedGeometryDiscipline(mesh, len(xi), "Free", L_domain=L, H_domain=H, Ngp=2, r_gp=0.5, pp=100.0)
    phys_disc.geom_disc = geom_disc
    chain = MDAChain([geom_disc, phys_disc])

    # Help GEMSEO guess input sizes
    chain.default_inputs = {"x_vars": x_init}
    geom_disc.default_inputs = {"x_vars": x_init}
    phys_disc.default_inputs = {"rho_E": np.zeros(mesh.num_cells()), "rho_V": np.zeros(mesh.num_cells())}

    
    def obj_f(x): x_o = lb + np.array(x)*(ub-lb); chain.execute({"x_vars": x_o}); return chain.local_data["compliance"][0]
    def obj_g(x, out): 
        x_o = lb + np.array(x)*(ub-lb); chain.linearize({"x_vars": x_o}, compute_all_jacobians=True); g = chain.jac["compliance"]["x_vars"].flatten()*(ub-lb)
        if max_step > 0:
            gn = np.linalg.norm(g); 
            if gn > max_step: g *= (max_step / gn)
        out[:] = g
    def con_f(x, out): x_o = lb + np.array(x)*(ub-lb); chain.execute({"x_vars": x_o}); out[0] = chain.local_data["volume"][0]
    def jac_f(x, out): 
        x_o = lb + np.array(x)*(ub-lb); chain.linearize({"x_vars": x_o}, compute_all_jacobians=True); g = chain.jac["volume"]["x_vars"].flatten()*(ub-lb)
        if max_step > 0:
            gn = np.linalg.norm(g);
            if gn > max_step: g *= (max_step / gn)
        out[:] = g
    
    nv = len(lb); model = unopy.Model("NLP", nv, unopy.ZERO_BASED_INDEXING)
    model.set_variables_lower_bounds([0.0]*nv); model.set_variables_upper_bounds([1.0]*nv)
    model.set_objective(unopy.MINIMIZE, obj_f, obj_g)
    model.set_constraints(1, con_f, [-1e10], [0.0], nv, [0]*nv, list(range(nv)), jac_f)
    model.set_initial_primal_iterate(((x_init-lb)/(ub-lb)).tolist())
    
    uno = unopy.UnoSolver(); uno.set_preset(preset); uno.set_option("hessian_model", hessian); uno.set_option("subproblem_solver", solver_name)
    uno.set_option("globalization_mechanism", mechanism)
    if mechanism == "TR": uno.set_option("TR_radius", radius)
    uno.set_option("max_iterations", max_iter)
    if "mma" in solver_name.lower(): uno.set_option("mma_external_move_limit", 0.05)
    
    print(f"Solving with Uno {unopy.current_uno_version()}..."); result = uno.optimize(model)
    print("Status:", result.optimization_status)
    if len(phys_disc.cvec) > 0: print(f"Final Compliance: {phys_disc.cvec[-1]:.4f}, Volume: {phys_disc.vvec[-1]*100.0:.4f}%")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--bc", default="Short_Cantilever"); p.add_argument("--max_iter", type=int, default=300)
    p.add_argument("--solver", default="MMA"); p.add_argument("--preset", default="filtersmma"); p.add_argument("--mode", default="Free")
    p.add_argument("--hessian", default="identity"); p.add_argument("--radius", type=float, default=10.0); p.add_argument("--mechanism", default="LS"); p.add_argument("--max_step", type=float, default=0.0)
    args = p.parse_args(); run_main_ggp(args.bc, args.max_iter, args.mode, args.solver, args.preset, args.hessian, args.radius, args.mechanism, args.max_step)
