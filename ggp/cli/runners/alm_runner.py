"""
Alternating Optimization Strategy for ALM Topology Optimization using GEMSEO Augmented Lagrangian.
Outer Loop: Solves the PDE to find displacement u^k.
Inner Loop: Uses GEMSEO Augmented_Lagrangian_order_1 to solve the geometric problem
            maximizing strain energy subject to constraints.
"""
import dolfin as df
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt

import gemseo
from gemseo import create_scenario, create_design_space
from gemseo.core.discipline.discipline import Discipline

from ggp.geometry.factory import GeometryFactory
from ggp.physics.factory import PhysicsFactory
from ggp.gemseo_wrappers.modular_disciplines import GGPVectorizedGeometryDiscipline
from ggp.utils.alm_utils import create_alm_overhang_constraints

gemseo.configure()

class AlternatingLowerDiscipline(Discipline):
    def __init__(self, geom_disc, E_e_array, dE_drho, C_offset, num_elements, mesh_area, volfrac, 
                 num_layers, comp_per_layer, layer_height, alpha_deg, lb, ub):
        super().__init__()
        self.geom_disc = geom_disc
        self.E_e_array = E_e_array
        self.dE_drho = dE_drho
        self.C_offset = C_offset
        self.num_elements = num_elements
        self.mesh_area = mesh_area
        self.volfrac = volfrac
        self.num_layers = num_layers
        self.comp_per_layer = comp_per_layer
        self.layer_height = layer_height
        self.alpha_deg = alpha_deg
        self.lb = lb
        self.ub = ub
        
        self.input_grammar.update_from_names(["x_vars"])
        self.output_grammar.update_from_names(["proxy_obj", "vol_cons", "overhang_cons"])
        self.default_inputs = {"x_vars": np.ones(len(lb)) * 0.5}

    def _run(self, input_data=None):
        xval = self.local_data["x_vars"].flatten()
        self.geom_disc.execute({"x_vars": xval})
        rho_E = self.geom_disc.local_data["rho_E"].flatten()
        rho_V = self.geom_disc.local_data["rho_V"].flatten()
        
        # Objective: Proxy compliance
        proxy = self.C_offset - np.sum(rho_E * self.dE_drho * self.E_e_array)
        proxy_log = np.log(proxy + 1.0)
        
        # Volume constraint: (v_val - volfrac) / volfrac * 100 <= 0
        v_val = np.sum(rho_V) * (self.mesh_area / self.num_elements) / self.mesh_area
        vol_cons = (v_val - self.volfrac) / self.volfrac * 100.0
        
        # Overhang constraints
        x_unscaled = self.lb + xval * (self.ub - self.lb)
        alpha = np.deg2rad(self.alpha_deg)
        np_val = self.comp_per_layer
        nY = self.num_layers
        
        Xk = x_unscaled[0:np_val*nY]
        Lk = x_unscaled[np_val*nY:2*np_val*nY]
        hk = x_unscaled[2*np_val*nY:2*np_val*nY+nY]
        
        num_cons = (nY - 1) * np_val * 2
        cons = np.zeros(num_cons)
        
        row = 0
        for layer in range(nY - 1):
            for k in range(np_val):
                idx = layer * np_val + k
                idx_next = (layer + 1) * np_val + k
                
                delta = hk[layer] * np.tan(alpha)
                
                # Cons 1: X_next - X + 0.5*(L_next - L) - delta <= 0
                cons[row] = Xk[idx_next] - Xk[idx] + 0.5*(Lk[idx_next] - Lk[idx]) - delta
                row += 1
                
                # Cons 2: -X_next + X + 0.5*(L_next - L) - delta <= 0
                cons[row] = -(Xk[idx_next] - Xk[idx]) + 0.5*(Lk[idx_next] - Lk[idx]) - delta
                row += 1
        
        self.local_data["proxy_obj"] = np.array([proxy_log])
        self.local_data["vol_cons"] = np.array([vol_cons])
        self.local_data["overhang_cons"] = cons

    def _compute_jacobian(self, input_data=None, output_data=None):
        xval = self.local_data["x_vars"].flatten()
        
        # Linearize geometry mapping
        jac_geom = self.geom_disc.linearize(input_data={"x_vars": xval}, compute_all_jacobians=True)
        jac_E = jac_geom["rho_E"]["x_vars"]
        jac_V = jac_geom["rho_V"]["x_vars"]
        
        # Objective Jacobian
        rho_E = self.geom_disc.local_data["rho_E"].flatten()
        proxy = self.C_offset - np.sum(rho_E * self.dE_drho * self.E_e_array)
        dj_drhoE = - (self.dE_drho * self.E_e_array)
        df0dx = (dj_drhoE @ jac_E) / (proxy + 1.0)
        
        # Volume Jacobian
        dv_drhoV = np.ones(self.num_elements) * (100.0 / (self.volfrac * self.num_elements))
        dv_dx = dv_drhoV @ jac_V
        
        # Overhang Jacobian
        x_unscaled = self.lb + xval * (self.ub - self.lb)
        alpha = np.deg2rad(self.alpha_deg)
        np_val = self.comp_per_layer
        nY = self.num_layers
        
        num_vars = len(x_unscaled)
        num_cons = (nY - 1) * np_val * 2
        jac_unscaled = np.zeros((num_cons, num_vars))
        
        row = 0
        for layer in range(nY - 1):
            for k in range(np_val):
                idx = layer * np_val + k
                idx_next = (layer + 1) * np_val + k
                
                # cons 1
                jac_unscaled[row, idx_next] = 1.0
                jac_unscaled[row, idx] = -1.0
                jac_unscaled[row, np_val*nY + idx_next] = 0.5
                jac_unscaled[row, np_val*nY + idx] = -0.5
                jac_unscaled[row, 2*np_val*nY + layer] = -np.tan(alpha)
                row += 1
                
                # cons 2
                jac_unscaled[row, idx_next] = -1.0
                jac_unscaled[row, idx] = 1.0
                jac_unscaled[row, np_val*nY + idx_next] = 0.5
                jac_unscaled[row, np_val*nY + idx] = -0.5
                jac_unscaled[row, 2*np_val*nY + layer] = -np.tan(alpha)
                row += 1
        jac_cons = jac_unscaled * (self.ub - self.lb)
        
        self.jac = {
            "proxy_obj": {"x_vars": np.atleast_2d(df0dx)},
            "vol_cons": {"x_vars": np.atleast_2d(dv_dx)},
            "overhang_cons": {"x_vars": jac_cons}
        }


def run_alternating_alm_gemseo(max_outer=30, max_inner=10, alpha_deg=45.0):
    print(f"\n=================================================================")
    print(f"   Alternating Opt | GEMSEO Augmented Lagrangian Order 1")
    print(f"   Outer iterations: {max_outer} | Max AL Inner iterations: {max_inner}")
    print(f"=================================================================\n")
    
    L, H = 60.0, 30.0
    nelx, nely = 60, 30
    volfrac = 0.3
    
    num_layers = 30
    comp_per_layer = 4
    layer_height = H / num_layers
    
    mesh = df.RectangleMesh.create([df.Point(0, 0), df.Point(L, H)], [nelx, nely], df.CellType.Type.quadrilateral)
    V_u = df.VectorFunctionSpace(mesh, "CG", 1)
    
    # Load and fixed DOFs
    f_vec = np.zeros(V_u.dim())
    def left_boundary(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
    bc = [df.DirichletBC(V_u, df.Constant((0.0, 0.0)), left_boundary)]
    
    dof_coords = V_u.tabulate_dof_coordinates()
    y_dofs = V_u.sub(1).dofmap().dofs()
    dists = np.linalg.norm(dof_coords - np.array([L, 0.0]), axis=1)
    br_y_dof = np.intersect1d(np.where(dists < 1e-3)[0], y_dofs)[0]
    f_vec[br_y_dof] = -1.0
    
    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    ds_load = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    
    solver = PhysicsFactory.create_solver("Elasticity", V_u=V_u, bc=bc, ds_load=ds_load, 
                                          L_rhs_vec=df.Constant((0.0, 0.0)), p=3.0, plane_stress=True)
    
    ke_ref = solver.get_unit_element_stiffness()
    num_elements = mesh.num_cells()
    cell_dofs = []
    dofmap = V_u.dofmap()
    for cell in df.cells(mesh):
        cell_dofs.append(dofmap.cell_dofs(cell.index()))
    cell_dofs = np.array(cell_dofs)
    
    fixed_dofs = []
    for cond in bc:
        dofs = cond.get_boundary_values().keys()
        fixed_dofs.extend(list(dofs))
    fixed_dofs = np.array(list(set(fixed_dofs)))
    
    mapper = GeometryFactory.create_mapper("2D_ALM", mesh=mesh, num_layers=num_layers, 
                                          components_per_layer=comp_per_layer, layer_height=layer_height)
    x_init = mapper.get_initial_design(L, H, extended=True, initial_mc=1.0, initial_volfrac=volfrac)
    
    np_val = comp_per_layer
    nY = num_layers
    
    # [x, L, h, m]
    xLB = 0.0 * np.ones(np_val * nY)
    xUB = L * np.ones(np_val * nY)
    
    LLB = 0.0 * np.ones(np_val * nY)
    LUB = L * np.ones(np_val * nY)
    
    hLB = 0.5 * layer_height * np.ones(nY)
    hUB = 2.0 * layer_height * np.ones(nY)
    
    mLB = 0.0 * np.ones(np_val)
    mUB = 1.0 * np.ones(np_val)
    
    lb = np.concatenate([xLB, LLB, hLB, mLB])
    ub = np.concatenate([xUB, LUB, hUB, mUB])
    
    geom_disc = GGPVectorizedGeometryDiscipline(
        mesh, len(lb), mode='ALM',
        num_layers=num_layers, comp_per_layer=comp_per_layer, layer_height=layer_height,
        pp=10.0, r_gp=0.5, lb=lb, ub=ub
    )
    
    folder_name = "Optimization_history_ALM_Alternating"
    os.makedirs(folder_name, exist_ok=True)
    
    # Initialize design
    X = (x_init - lb) / (ub - lb)
    n_mma = len(X)
    xval = X.copy()
    
    mesh_area = L * H
    E0 = float(solver.E0)
    Emin = float(solver.Emin)
    dE_drho = E0 - Emin
    
    cvec = []
    vvec = []
    
    # Pre-allocate array to hold E_e_array values so discipline can read it
    current_E_e_array = np.zeros(num_elements)
    
    # Create the discipline
    inner_disc = AlternatingLowerDiscipline(
        geom_disc, current_E_e_array, dE_drho, 0.0, num_elements, mesh_area, volfrac,
        num_layers, comp_per_layer, layer_height, alpha_deg, lb, ub
    )
    
    # Setup GEMSEO Design Space for inner loop
    design_space = create_design_space()
    design_space.add_variable("x_vars", size=n_mma, lower_bound=0.0, upper_bound=1.0, value=xval)
    
    # Setup Scenario
    scenario = create_scenario(
        inner_disc, "proxy_obj", design_space, maximize_objective=False, formulation_name="DisciplinaryOpt"
    )
    scenario.add_constraint("vol_cons", constraint_type="ineq", positive=False)
    scenario.add_constraint("overhang_cons", constraint_type="ineq", positive=False)
    
    for outer_it in range(1, max_outer + 1):
        # 1. Forward geometry mapping to get true densities
        geom_disc.execute({"x_vars": xval})
        rho_E = geom_disc.local_data["rho_E"].flatten()
        rho_V = geom_disc.local_data["rho_V"].flatten()
        
        # 2. Solve PDE once
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import spsolve
        
        E_vals = Emin + rho_E * dE_drho
        dofs_per_cell = ke_ref.shape[0]
        rows = np.repeat(cell_dofs, dofs_per_cell).reshape(num_elements, dofs_per_cell, dofs_per_cell)
        cols = np.tile(cell_dofs, (1, dofs_per_cell)).reshape(num_elements, dofs_per_cell, dofs_per_cell)
        data = np.outer(E_vals, ke_ref).reshape(num_elements, dofs_per_cell, dofs_per_cell)
        K_global = coo_matrix((data.flatten(), (rows.flatten(), cols.flatten())), shape=(V_u.dim(), V_u.dim())).tocsr()
        
        col_mask = np.isin(K_global.indices, fixed_dofs)
        K_global.data[col_mask] = 0.0
        for dof in fixed_dofs:
            s, e = K_global.indptr[dof], K_global.indptr[dof+1]
            K_global.data[s:e] = 0.0
            col_idx = K_global.indices[s:e]
            diag = np.where(col_idx == dof)[0]
            if len(diag) > 0: K_global.data[s + diag[0]] = 1.0
            
        f_vec_bc = f_vec.copy()
        f_vec_bc[fixed_dofs] = 0.0
        u_vec = spsolve(K_global, f_vec_bc)
        
        true_compliance = np.dot(f_vec_bc, u_vec)
        true_volume = np.sum(rho_V) * (mesh_area / num_elements) / mesh_area
        
        cvec.append(true_compliance)
        vvec.append(true_volume)
        print(f"\nOuter It {outer_it:3d} | Compliance: {true_compliance:.3e} | Volume: {true_volume*100:.2f}%")
        
        # 3. Compute element strain energies
        for i in range(num_elements):
            u_e = u_vec[cell_dofs[i]]
            current_E_e_array[i] = np.dot(u_e, np.dot(ke_ref, u_e))
        
        # Update C_offset in the discipline to ensure proxy_obj stays positive
        inner_disc.C_offset = 2.0 * true_compliance
        
        # 4. Inner Loop via GEMSEO Augmented_Lagrangian_order_1
        print(f"   Running GEMSEO Inner Loop AL...")
        
        # Reset the current design inside the design space
        design_space.set_current_value(xval)
        
        # Execute the optimization with AL
        scenario.execute(
            algo_name="Augmented_Lagrangian_order_1",
            max_iter=max_inner,
            sub_algorithm_name="L-BFGS-B"
        )
        
        # Update xval with the best from this inner loop
        xval = scenario.design_space.get_current_value().copy()
        
        # 5. Plotting
        plt.figure(3)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(1, outer_it + 1), cvec, 'bo-', label='C')
        plt.grid(True)
        plt.ylabel('Compliance')
        plt.title(f"Outer Iteration {outer_it}")
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(1, outer_it + 1), np.array(vvec) * 100, 'ro-', label='V')
        plt.grid(True)
        plt.ylabel('Volume [%]')
        plt.tight_layout()
        plt.savefig(f"{folder_name}/convergence.png")
        plt.close()
        
        plt.figure(1)
        plt.clf()
        cols = np.round((geom_disc.X_mesh - (L/nelx)/2.0) / (L/nelx)).astype(int)
        rows = np.round((geom_disc.Y_mesh - (H/nely)/2.0) / (H/nely)).astype(int)
        xPhys = np.zeros((nely, nelx))
        xPhys[rows, cols] = rho_V
        plt.imshow(1.0 - xPhys, cmap='gray', origin='lower', extent=[0.0, L, 0.0, H], vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f"Density - Outer Iter {outer_it}")
        plt.savefig(f"{folder_name}/density_{outer_it:03d}.png")
        plt.close()

