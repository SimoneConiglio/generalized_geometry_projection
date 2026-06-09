import numpy as np
from dolfin import *
import dolfin_adjoint
from dolfin_adjoint import *
from gemseo.core.discipline.discipline import Discipline

class GGPMacroDiscipline(Discipline):
    """
    Monolithic GGP Discipline (Macro-Discipline).
    Wraps the entire mapping and physics solve into a single GEMSEO unit.
    Optimized for dolfin-adjoint tape management.
    """
    def __init__(self, mapper, solver, x_init, lb, ub, mesh_area, name="GGP_Macro"):
        super().__init__(name=name)
        self.mapper = mapper
        self.solver = solver
        self.mesh_area = mesh_area
        
        # IO Grammars
        self.input_grammar.update_from_names(["x_vars"])
        self.output_grammar.update_from_names(["compliance", "volume"])
        self.default_inputs = {"x_vars": x_init}
        
        # Internal state
        self.lb, self.ub = lb, ub
        self.x_vals = x_init
        self.last_x = None
        self.scale_obj = 1.0
        self.iter = 0

    def _run(self, input_data=None):
        if input_data is not None:
            self.local_data.update(input_data)
        x_vars = self.local_data["x_vars"].flatten()
        
        # 1. Clear tape BEFORE graph construction
        tape = get_working_tape()
        tape.clear_tape()
        
        # 2. Build graph with FRESH AdjConstants in every iteration
        # This is the safest way to ensure they are tracked in the new tape.
        controls_objs = [dolfin_adjoint.Constant(float(v)) for v in x_vars]
            
        rho = self.mapper.map_to_density(controls_objs)
        u = self.solver.solve(rho)
        
        # 3. Ensure these are tape-tracked functionals
        j_functional = self.solver.compute_compliance(u)
        v_functional = self.solver.compute_volume(rho)
        
        # 4. Extract gradients
        m_ctrls = [Control(c) for c in controls_objs]
        dj_raw = np.array([float(g) for g in compute_gradient(j_functional, m_ctrls)])
        dv_raw = np.array([float(g) for g in compute_gradient(v_functional, m_ctrls)]) / self.mesh_area
        
        # Normalization and scaling
        if self.iter == 0:
            self.scale_obj = 1.0 / float(j_functional)
            
        self.dj = np.nan_to_num(dj_raw) * self.scale_obj
        self.dv = np.nan_to_num(dv_raw)
        
        # Store results
        self.local_data["compliance"] = np.array([float(j_functional) * self.scale_obj])
        self.local_data["volume"] = np.array([float(v_functional) / self.mesh_area])
        
        self.iter += 1

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        self.jac = {
            "compliance": {"x_vars": self.dj.reshape(1, -1)},
            "volume": {"x_vars": self.dv.reshape(1, -1)}
        }
