from dolfin import *
from dolfin_adjoint import *
import ufl
from ..physics.base_solver import BaseSolver

class LinearElasticitySolver(BaseSolver):
    """
    FEniCS Linear Elasticity Solver with Adjoint support.
    Scalable version supporting iterative solvers and AMG preconditioning.
    """
    def __init__(self, V_u, bc, ds_load, L_rhs_vec, p=3.0, Emin=1e-6, E0=1.0, iterative=False, plane_stress=True):
        self.V_u = V_u
        self.bc = bc
        self.ds_load = ds_load
        self.L_rhs_vec = L_rhs_vec
        self.p = Constant(p)
        self.Emin = Constant(Emin)
        self.E0 = Constant(E0)
        self.iterative = iterative
        self.plane_stress = plane_stress
        self.dim = V_u.mesh().geometry().dim()

    def solve(self, rho):
        # SIMP Penalization
        rho_safe = ufl.max_value(Constant(0.0), rho)
        E = self.Emin + rho_safe**self.p * (self.E0 - self.Emin)
        
        # Material properties (nu=0.3)
        mu = E / 2.6
        if self.plane_stress and self.dim == 2:
            lmbda = E * 0.3 / (1.0 - 0.3**2)
        else:
            lmbda = E * 0.3 / (1.3 * (1.0 - 2.0*0.3))
        
        def eps_f(u): return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        def sig_f(u): return lmbda * ufl.tr(eps_f(u)) * ufl.Identity(self.dim) + 2.0 * mu * eps_f(u)
        
        u_trial, v_test = TrialFunction(self.V_u), TestFunction(self.V_u)
        a = ufl.inner(sig_f(u_trial), eps_f(v_test)) * dx
        L = ufl.dot(self.L_rhs_vec, v_test) * self.ds_load(1)
        
        u_sol = Function(self.V_u)
        
        if self.iterative:
            # Scalable Iterative Solver (CG + AMG)
            # Standard choice for symmetric positive definite elasticity
            solver_params = {
                "linear_solver": "cg",
                "preconditioner": "hypre_amg" if has_krylov_solver_preconditioner("hypre_amg") else "gamg",
                "krylov_solver": {
                    "relative_tolerance": 1e-8,
                    "maximum_iterations": 2000,
                    "monitor_convergence": False
                }
            }
        else:
            # Direct Solver
            solver_params = {"linear_solver": "mumps"}
            
        solve(a == L, u_sol, self.bc, solver_parameters=solver_params)
        
        # Store for objective computation
        self.last_L = L
        self.last_u = u_sol
        return u_sol

    def compute_compliance(self, u=None):
        if u is None:
            u = self.last_u
        return assemble(ufl.action(self.last_L, u))

    def compute_volume(self, rho):
        return assemble(rho * dx)

    def get_unit_element_stiffness(self):
        """
        Computes the stiffness matrix of a single element for E=1.0.
        Assumes all elements in the mesh are identical (structured grid).
        """
        # E=1.0
        mu = 1.0 / 2.6
        if self.plane_stress and self.dim == 2:
            lmbda = 0.3 / (1.0 - 0.3**2)
        else:
            lmbda = 0.3 / (1.3 * (1.0 - 2.0*0.3))
        
        def eps_f(u): return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        def sig_f(u): return lmbda * ufl.tr(eps_f(u)) * ufl.Identity(self.dim) + 2.0 * mu * eps_f(u)
        
        u_trial, v_test = TrialFunction(self.V_u), TestFunction(self.V_u)
        a = ufl.inner(sig_f(u_trial), eps_f(v_test)) * dx
        
        # Assemble for the first cell
        cell = next(cells(self.V_u.mesh()))
        ke = assemble_local(a, cell)
        return ke
