import dolfin as df
from dolfin_adjoint import *
import numpy as np
import gemseo
from gemseo.mda.mda_chain import MDAChain
from ggp.geometry.factory import GeometryFactory
from ggp.physics.factory import PhysicsFactory
from ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline, GGPPhysicsFastDiscipline
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Import the CONLIN SCP solver
from scp_uno.conlin import CONLIN


class GGPHybridPhysicsDiscipline(GGPPhysicsFastDiscipline):
    def __init__(self, solver, mesh, mesh_area, volfrac, emptyelts=None, L=60.0, H=30.0,
                 bc_type="Short_Cantilever", max_iter=50, plot_rate=10):
        super().__init__(solver, mesh, mesh_area, volfrac)
        self.emptyelts = emptyelts or []
        self.L, self.H, self.bc_type, self.max_iter = L, H, bc_type, max_iter
        self.plot_rate = plot_rate
        self.Path = f"Optimization_history_{bc_type}_SCP_CONLIN_UNO/"
        os.makedirs(self.Path, exist_ok=True)
        self.cvec, self.vvec = [], []
        self.geom_disc = None  # set by caller after construction

    def _run(self, input_data=None):
        if input_data:
            self.local_data.update(input_data)
        rho_E = self.local_data["rho_E"].flatten().copy()
        rho_V = self.local_data["rho_V"].flatten().copy()
        E0 = float(self.solver.E0)
        E_vals = rho_E * E0
        if self.emptyelts:
            rho_E[self.emptyelts] = 0.0
            rho_V[self.emptyelts] = 0.0
            E_vals[self.emptyelts] = float(self.solver.Emin)
        from scipy.sparse import coo_matrix
        dpc = self.ke_ref.shape[0]
        rows = np.repeat(self.cell_dofs, dpc).reshape(self.num_elements, dpc, dpc)
        cols = np.tile(self.cell_dofs, (1, dpc)).reshape(self.num_elements, dpc, dpc)
        data = np.outer(E_vals, self.ke_ref).reshape(self.num_elements, dpc, dpc)
        K = coo_matrix(
            (data.flatten(), (rows.flatten(), cols.flatten())),
            shape=(self.V_u.dim(), self.V_u.dim()),
        ).tocsr()
        for d in self.fixed_dofs:
            K.data[K.indptr[d]:K.indptr[d + 1]] = 0.0
            di = np.where(K.indices[K.indptr[d]:K.indptr[d + 1]] == d)[0]
            if len(di) > 0:
                K.data[K.indptr[d] + di[0]] = 1.0
        from scipy.sparse.linalg import spsolve
        u = spsolve(K, self.f_vec)
        compliance = np.dot(self.f_vec, u)
        volume = np.sum(rho_V) * (self.mesh_area / self.num_elements) / self.mesh_area
        self.local_data["compliance"] = np.array([np.log(compliance + 1.0)])
        self.local_data["volume"] = np.array([(volume - self.volfrac) / self.volfrac * 100.0])
        grad_C = np.zeros(self.num_elements)
        for i in range(self.num_elements):
            grad_C[i] = -E0 * np.dot(u[self.cell_dofs[i]], np.dot(self.ke_ref, u[self.cell_dofs[i]]))
        self.dj_drhoE = grad_C / (compliance + 1.0)
        self.dv_drhoV = np.ones(self.num_elements) * (100.0 / (self.volfrac * self.num_elements))
        if self.emptyelts:
            self.dj_drhoE[self.emptyelts] = 0.0
            self.dv_drhoV[self.emptyelts] = 0.0
        self.cvec.append(compliance)
        self.vvec.append(volume)
        outit = len(self.cvec)
        print(f" It.:{outit:5d} Obj.:{compliance:4.3e} Vol.:{volume:7.3f}")

        if outit % self.plot_rate == 0 or outit == self.max_iter:
            self._save_plots(outit, compliance, volume, rho_V)

    def _save_plots(self, outit, compliance, volume, rho_V):
        """Save convergence, density, and component plots."""
        # 1. Convergence
        plt.figure(3)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(1, outit + 1), self.cvec, "bo-")
        plt.grid(True)
        plt.xlabel("iter")
        plt.ylabel("C")
        plt.title(f"C = {compliance:.2f} at iteration {outit}")
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(1, outit + 1), np.array(self.vvec) * 100, "ro-")
        plt.grid(True)
        plt.xlabel("iter")
        plt.ylabel("V [%]")
        plt.title(f"V = {volume * 100:.2f}% at iteration {outit}")
        plt.tight_layout()
        plt.savefig(f"{self.Path}convergence.png")
        plt.close()

        # 2. Density field
        if self.geom_disc is not None:
            nelx, nely = 60, 30
            dx, dy = self.L / nelx, self.H / nely
            X_mesh = self.geom_disc.X_mesh
            Y_mesh = self.geom_disc.Y_mesh
            cols = np.round((X_mesh - dx / 2.0) / dx).astype(int)
            rows_idx = np.round((Y_mesh - dy / 2.0) / dy).astype(int)
            xPhys = np.zeros((nely, nelx))
            xPhys[rows_idx, cols] = rho_V
            plt.figure(1)
            plt.clf()
            plt.imshow(1.0 - xPhys, cmap="gray", origin="lower", extent=[0.0, self.L, 0.0, self.H])
            plt.colorbar()
            plt.axis("equal")
            plt.axis("off")
            plt.title(f"Density — iter {outit}")
            plt.savefig(f"{self.Path}density_{outit:03d}.png")
            plt.close()

            # 3. Component outlines
            x_vars = self.geom_disc.local_data["x_vars"].flatten()
            if self.geom_disc.lb is not None and self.geom_disc.ub is not None:
                x_vars = self.geom_disc.lb + x_vars * (self.geom_disc.ub - self.geom_disc.lb)
            Xc = x_vars[0::6]; Yc = x_vars[1::6]
            Lc = x_vars[2::6]; hc = x_vars[3::6]
            Tc = x_vars[4::6]; Mc = x_vars[5::6]
            tt = np.linspace(0, 2 * np.pi, 60)
            cc = np.cos(tt); ss = np.sin(tt)
            plt.figure(2)
            plt.clf()
            plt.xlim(0.0, self.L)
            plt.ylim(0.0, self.H)
            cmap_jet = plt.get_cmap("jet")
            for idx in np.where(Mc > 0.1)[0]:
                C0, S0 = np.cos(Tc[idx]), np.sin(Tc[idx])
                xi = Lc[idx] / 2 * cc
                eta = hc[idx] / 2 * ss
                xb = Xc[idx] + C0 * xi - S0 * eta
                yb = Yc[idx] + S0 * xi + C0 * eta
                plt.fill(xb, yb, facecolor=cmap_jet(Mc[idx]), alpha=0.5)
            plt.axis("equal")
            plt.axis("off")
            plt.title(f"Components — iter {outit}")
            plt.savefig(f"{self.Path}component_{outit:03d}.png")
            plt.close()


def run_ggp_conlin(bc_type="Short_Cantilever", max_iter=300):
    L, H = 60.0, 30.0
    nelx, nely = 60, 30
    volfrac = 0.4

    mesh = df.RectangleMesh.create(
        [df.Point(0, 0), df.Point(L, H)], [nelx, nely], df.CellType.Type.quadrilateral
    )
    V_u = df.VectorFunctionSpace(mesh, "CG", 1)
    f_vec = np.zeros(V_u.dim())
    dof_coords = V_u.tabulate_dof_coordinates()
    y_dofs = V_u.sub(1).dofmap().dofs()
    bc = [DirichletBC(V_u, Constant((0.0, 0.0)), "on_boundary && near(x[0], 0.0)")]
    tip = np.intersect1d(
        np.where(np.linalg.norm(dof_coords - np.array([L, H / 2.0]), axis=1) < 1e-3)[0], y_dofs
    )[0]
    f_vec[tip] = -1.0
    solver = PhysicsFactory.create_solver(
        "Elasticity",
        V_u=V_u,
        bc=bc,
        ds_load=df.Measure("ds", domain=mesh),
        L_rhs_vec=Constant((0.0, 0.0)),
        p=3.0,
        plane_stress=True,
    )

    # Grid initialisation (same as MMA script)
    ncx, ncy = 1, 1
    num_components = 2 * (ncx + 2) * (ncy + 2)
    xp = np.linspace(0.0, nelx, ncx + 2)
    yp = np.linspace(0.0, nely, ncy + 2)
    xx, yy = np.meshgrid(xp, yp)
    xx_flat, yy_flat = xx.flatten(order="F"), yy.flatten(order="F")
    Xc_init = np.tile(xx_flat, 2)
    Yc_init = np.tile(yy_flat, 2)
    Lc_init = 2.0 * np.sqrt((nelx / (ncx + 2)) ** 2 + (nely / (ncy + 2)) ** 2) * np.ones_like(Xc_init)
    angle = np.arctan2(nely / ncy, nelx / ncx)
    Tc_init = angle * np.concatenate([np.ones(len(Xc_init) // 2), -np.ones(len(Xc_init) // 2)])
    hc_init = 2.0 * np.ones_like(Xc_init)
    Mc_init = 0.5 * np.ones_like(Xc_init)
    x_init = np.column_stack([Xc_init, Yc_init, Lc_init, hc_init, Tc_init, Mc_init]).flatten()
    lb = np.column_stack([
        (np.min(xx_flat) - 1.0) * np.ones_like(Xc_init),
        (np.min(yy_flat) - 1.0) * np.ones_like(Xc_init),
        np.zeros_like(Xc_init),
        np.ones_like(Xc_init),
        -2.0 * np.pi * np.ones_like(Xc_init),
        np.zeros_like(Xc_init),
    ]).flatten()
    ub = np.column_stack([
        (np.max(xx_flat) + 1.0) * np.ones_like(Xc_init),
        (np.max(yy_flat) + 1.0) * np.ones_like(Xc_init),
        np.sqrt(nelx ** 2 + nely ** 2) * np.ones_like(Xc_init),
        np.sqrt(nelx ** 2 + nely ** 2) * np.ones_like(Xc_init),
        2.0 * np.pi * np.ones_like(Xc_init),
        np.ones_like(Xc_init),
    ]).flatten()

    phys_disc = GGPHybridPhysicsDiscipline(solver, mesh, L * H, volfrac, None, L, H, bc_type, max_iter)
    phys_disc.f_vec = f_vec
    phys_disc.f_vec[phys_disc.fixed_dofs] = 0.0
    geom_disc = GGPVectorizedGeometryDiscipline(
        mesh, len(Xc_init), "Free", L_domain=L, H_domain=H, Ngp=2, r_gp=0.5, pp=100.0, lb=lb, ub=ub
    )
    phys_disc.geom_disc = geom_disc
    chain = MDAChain([geom_disc, phys_disc])

    from gemseo import create_design_space
    ds = create_design_space()
    ds.add_variable("x_vars", size=len(lb), lower_bound=0.0, upper_bound=1.0, value=((x_init - lb) / (ub - lb)))

    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdo_functions.mdo_function import MDOFunction

    def obj_func_call(x):
        chain.execute({"x_vars": x})
        return chain.local_data["compliance"]

    def obj_jac_call(x):
        return chain.linearize({"x_vars": x}, compute_all_jacobians=True)["compliance"]["x_vars"]

    def vol_func_call(x):
        chain.execute({"x_vars": x})
        return chain.local_data["volume"]

    def vol_jac_call(x):
        return chain.linearize({"x_vars": x}, compute_all_jacobians=True)["volume"]["x_vars"]

    obj_func = MDOFunction(obj_func_call, "compliance", jac=obj_jac_call, input_names=["x_vars"])
    vol_func = MDOFunction(vol_func_call, "volume", jac=vol_jac_call, input_names=["x_vars"])
    problem = OptimizationProblem(ds)
    problem.objective = obj_func
    problem.add_constraint(vol_func, constraint_type="ineq")
    problem.lagrange_multipliers_computation = False

    # Patch out KKT residual computation (not meaningful for our SCP outer loop)
    import gemseo.algos.stop_criteria as sc
    import gemseo.algos.opt.base_optimization_library as bol
    dummy_kkt = lambda *args, **kwargs: 1.0
    sc.kkt_residual_computation = dummy_kkt
    bol.kkt_residual_computation = dummy_kkt

    # ---- CONLIN options ----
    # CONLIN uses a reciprocal/linear mixed approximation for each subproblem.
    # Inner solver: scipy SLSQP (same inner as the MMA baseline for a fair comparison).
    conlin_algo = CONLIN()
    options = {
        "max_iter": max_iter,
        "inner_library": "scipy",
        "inner_solver": "SLSQP",
        "max_optimization_step": 0.02,   # move limit (same as MMA run)
        "ftol_rel": 1e-12,
        "xtol_rel": 1e-12,
        "ftol_abs": 1e-12,
        "xtol_abs": 1e-12,
        "kkt_tol_abs": -1.0,
        "kkt_tol_rel": -1.0,
    }

    print("\nStarting SCP-CONLIN Optimization...")
    conlin_algo.execute(problem, **options)

    if len(phys_disc.cvec) > 0:
        print(f"\nFinal Compliance: {phys_disc.cvec[-1]:.4f}, Volume: {phys_disc.vvec[-1] * 100.0:.4f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GGP Short Cantilever — CONLIN subsolver")
    parser.add_argument("--max_iter", type=int, default=300, help="Maximum number of outer SCP iterations")
    args = parser.parse_args()
    run_ggp_conlin(max_iter=args.max_iter)
