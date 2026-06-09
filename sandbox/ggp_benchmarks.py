import dolfin as df
from dolfin_adjoint import *
import numpy as np
import ufl
import gemseo
from gemseo import create_scenario
from gemseo.core.discipline.discipline import Discipline
import sys

# --- Benchmark Selector ---
CASE = "MBB" # Default. Change to "Short_Cantilever" or "L-shape"
if len(sys.argv) > 1:
    CASE = sys.argv[1]

print(f"--- Running GGP Benchmark: {CASE} ---")

# --- 1. Parameters ---
if CASE == "MBB":
    L_domain, H_domain = 150.0, 50.0
    nelx, nely = 150, 50
    volfrac = 0.4
elif CASE == "Short_Cantilever":
    L_domain, H_domain = 50.0, 50.0
    nelx, nely = 50, 50
    volfrac = 0.4
elif CASE == "L-shape":
    L_domain, H_domain = 100.0, 100.0
    nelx, nely = 100, 100
    volfrac = 0.4
else:
    raise ValueError("Invalid CASE")

ka, pp = 10.0, 100.0
num_components = 10

# --- 2. Mesh and Boundaries ---
mesh = RectangleMesh(df.Point(0, 0), df.Point(L_domain, H_domain), nelx, nely)
V_u = df.VectorFunctionSpace(mesh, "CG", 1)
V_d = df.FunctionSpace(mesh, "CG", 1)

boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

def left_boundary(x, on_boundary): return on_boundary and df.near(x[0], 0.0)
def top_boundary(x, on_boundary): return on_boundary and df.near(x[1], H_domain)

if CASE == "MBB":
    def bottom_right_point(x, on_boundary): return df.near(x[0], L_domain) and df.near(x[1], 0.0)
    bc = [
        DirichletBC(V_u.sub(0), Constant(0.0), left_boundary), # Symmetry u_x = 0
        DirichletBC(V_u.sub(1), Constant(0.0), bottom_right_point, method="pointwise")
    ]
    class TopLeftArea(df.SubDomain):
        def inside(self, x, on_boundary): return df.near(x[0], 0.0, 2.0) and df.near(x[1], H_domain, 2.0)
    TopLeftArea().mark(boundaries, 1)
    L_rhs_vec = Constant((0.0, -1.0))

elif CASE == "Short_Cantilever":
    bc = [DirichletBC(V_u, Constant((0.0, 0.0)), left_boundary)]
    class MiddleRightArea(df.SubDomain):
        def inside(self, x, on_boundary): return df.near(x[0], L_domain, 2.0) and df.near(x[1], H_domain/2.0, 5.0)
    MiddleRightArea().mark(boundaries, 1)
    L_rhs_vec = Constant((0.0, -1.0))

elif CASE == "L-shape":
    bc = [DirichletBC(V_u, Constant((0.0, 0.0)), top_boundary)]
    class BottomRightArea(df.SubDomain):
        def inside(self, x, on_boundary): return df.near(x[0], L_domain, 2.0) and df.near(x[1], H_domain/2.0, 5.0)
    BottomRightArea().mark(boundaries, 1)
    L_rhs_vec = Constant((0.0, -1.0))

ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

# --- 3. Design Variables ---
controls_objs = [Constant(0.0) for _ in range(num_components * 5)]
lb_arr = np.array([0.0, 0.0, 5.0, 2.0, -np.pi] * num_components)
ub_arr = np.array([L_domain, H_domain, L_domain, H_domain/2, np.pi] * num_components)
x_init = []
for i in range(num_components):
    x_init.extend([L_domain/2.0, (i+0.5)*H_domain/num_components, L_domain*0.5, H_domain/num_components, 0.1])
x_init = np.array(x_init)
for c, v in zip(controls_objs, x_init): c.assign(v)

# --- 4. GGP Logic ---
x_c = df.SpatialCoordinate(mesh)
df.parameters["form_compiler"]["quadrature_degree"] = 2
eps_safe = Constant(1e-7)

def get_rho(ctrls):
    def compute_local(X, Y, L, h, T):
        dx, dy = x_c[0]-X, x_c[1]-Y
        r2 = dx**2 + dy**2 + eps_safe
        r = ufl.sqrt(r2)
        phi = ufl.atan_2(dy, dx + eps_safe) - T
        abs_cos = ufl.sqrt(ufl.cos(phi)**2 + eps_safe)
        abs_sin = ufl.sqrt(ufl.sin(phi)**2 + eps_safe)
        upsi = ufl.conditional(r*abs_cos >= L/Constant(2.0), ufl.sqrt(ufl.max_value(eps_safe, r2 + L**2/Constant(4.0) - r*L*abs_cos)), r*abs_sin)
        eps_mna = Constant(3.0)
        l, u = h/Constant(2.0) - eps_mna/Constant(2.0), h/Constant(2.0) + eps_mna/Constant(2.0)
        d_v = -(eps_mna**3)
        return ufl.conditional(upsi <= l, Constant(1.0), ufl.conditional(upsi <= u, (Constant(-2.0)/d_v)*upsi**3 + (Constant(3.0)*h/d_v)*upsi**2 + (Constant(-6.0)*l*u/d_v)*upsi + (u*(-u**2+Constant(3.0)*l*u)/d_v), Constant(0.0)))
    
    densities = [compute_local(ctrls[i*5], ctrls[i*5+1], ctrls[i*5+2], ctrls[i*5+3], ctrls[i*5+4]) for i in range(num_components)]
    if CASE == "L-shape":
        mask = ufl.conditional(ufl.And(x_c[0] >= L_domain/2.0, x_c[1] >= H_domain/2.0), Constant(0.0), Constant(1.0))
        densities = [d * mask for d in densities]

    xt_v = 1.0 + 1.0/ka * np.log((1.0 + (num_components - 1.0)*np.exp(-ka))/num_components)
    s0_v = -np.log(np.exp(-pp) + 1.0 / (np.exp(0.0) + 1.0)) / pp
    xt, s0, pp_c, ka_c = Constant(xt_v), Constant(s0_v), Constant(pp), Constant(ka)
    sum_exp = sum([ufl.exp(ka_c * d) for d in densities])
    ks_val = (Constant(1.0)/ka_c) * ufl.ln(sum_exp / Constant(num_components))
    inner_exp = ufl.exp((pp_c * ks_val) / xt)
    return (-ufl.ln(Constant(np.exp(-pp)) + Constant(1.0) / (inner_exp + Constant(1.0))) / pp_c - s0) / (Constant(1.0) - s0)

# --- 5. Discipline ---
class GGPDiscipline(Discipline):
    def __init__(self):
        super().__init__(name=f"GGP_{CASE}")
        self.input_grammar.update_from_names(["x_vars"])
        self.output_grammar.update_from_names(["compliance", "volume"])
        self.default_inputs = {"x_vars": x_init}
        self.iter, self.file = 0, df.File(f"results_{CASE}/density.pvd")
        self.last_x, self.scale_obj = None, 1.0
        self.move_limit = 0.1

    def _run(self, **kwargs):
        x_raw = self.local_data["x_vars"].flatten()
        if self.last_x is not None:
            x_norm, last_norm = (x_raw-lb_arr)/(ub_arr-lb_arr), (self.last_x-lb_arr)/(ub_arr-lb_arr)
            x_array = (last_norm + np.clip(x_norm-last_norm, -self.move_limit, self.move_limit)) * (ub_arr-lb_arr) + lb_arr
        else: x_array = x_raw

        if self.last_x is not None and np.allclose(self.last_x, x_array, atol=1e-8): return

        tape = get_working_tape(); tape.clear_tape()
        for c, v in zip(controls_objs, x_array): c.assign(float(v))
        rho = get_rho(controls_objs)
        rho_safe = ufl.max_value(Constant(0.0), rho)
        E = Constant(1e-6) + rho_safe**3.0 * (1.0 - 1e-6)
        mu, lmbda = E / 2.6, E*0.3 / (1.3 * 0.4)
        def eps_f(u): return 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        def sig_f(u): return lmbda*ufl.tr(eps_f(u))*ufl.Identity(2) + 2.0*mu*eps_f(u)
        u_t, v_t = df.TrialFunction(V_u), df.TestFunction(V_u)
        a, L = ufl.inner(sig_f(u_t), eps_f(v_t))*df.dx, ufl.dot(L_rhs_vec, v_t)*ds(1)
        u_sol = Function(V_u)
        solve(a == L, u_sol, bc, solver_parameters={"linear_solver": "mumps"})
        J_val = assemble(ufl.action(L, u_sol))
        V_val = assemble(rho * df.dx)
        if self.iter == 0: self.scale_obj = 1.0 / float(J_val)
        m_ctrls = [Control(c) for c in controls_objs]
        dj_raw = np.array([float(g) for g in compute_gradient(J_val, m_ctrls)])
        dv_raw = np.array([float(g) for g in compute_gradient(V_val, m_ctrls)]) / (L_domain * H_domain)
        self.dj, self.dv = np.nan_to_num(dj_raw) * self.scale_obj, np.nan_to_num(dv_raw)
        self.j_out, self.v_out = float(J_val), float(V_val) / (L_domain * H_domain)
        self.last_x = x_array.copy()
        print(f"Iter {self.iter:03d} | Compliance: {self.j_out:.4f} | Vol: {self.v_out:.4f}")
        if self.iter % 5 == 0:
            r_viz = project(rho, V_d); r_viz.rename("Density", "Density"); self.file << (r_viz, self.iter)
        self.iter += 1
        self.local_data["compliance"] = np.array([self.j_out * self.scale_obj])
        self.local_data["volume"] = np.array([self.v_out])

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        self.jac = {"compliance": {"x_vars": self.dj.reshape(1,-1)}, "volume": {"x_vars": self.dv.reshape(1,-1)}}

# --- Run ---
disc = GGPDiscipline()
design_space = gemseo.algos.design_space.DesignSpace()
design_space.add_variable("x_vars", size=len(x_init), lower_bound=lb_arr, upper_bound=ub_arr, value=x_init)
scenario = create_scenario(disciplines=[disc], formulation_name="DisciplinaryOpt", objective_name="compliance", design_space=design_space, maximize_objective=False)
scenario.add_constraint("volume", constraint_type="ineq", positive=False, value=volfrac)
scenario.set_differentiation_method("user")
scenario.execute(algo_name="MMA", max_iter=50, max_optimization_step=0.1, normalize_design_space=True)
print(f"Benchmark {CASE} finished.")
