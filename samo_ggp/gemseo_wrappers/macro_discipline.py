import numpy as np
import dolfin as df
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
        self.controls_objs = [df.Constant(v) for v in x_init]
        self.last_x = None
        self.scale_obj = 1.0
        self.iter = 0

    def _run(self):
        x_vars = self.local_data["x_vars"].flatten()
        
        # Clear tape for new iteration
        tape = get_working_tape()
        tape.clear_tape()
        
        # Assign constants and build graph
        for c, v in zip(self.controls_objs, x_vars):
            c.assign(float(v))
            
        # Adjoint-tracked graph
        rho = self.mapper.map_to_density(self.controls_objs)
        u = self.solver.solve(rho)
        
        j_val = self.solver.compute_compliance(u)
        v_val = self.solver.compute_volume(rho)
        
        # Extract gradients
        m_ctrls = [Control(c) for c in self.controls_objs]
        dj_raw = np.array([float(g) for g in compute_gradient(j_val, m_ctrls)])
        dv_raw = np.array([float(g) for g in compute_gradient(v_val, m_ctrls)]) / self.mesh_area
        
        # Normalization and scaling
        if self.iter == 0:
            self.scale_obj = 1.0 / float(j_val)
            
        self.dj = np.nan_to_num(dj_raw) * self.scale_obj
        self.dv = np.nan_to_num(dv_raw)
        
        # Store results
        self.local_data["compliance"] = np.array([float(j_val) * self.scale_obj])
        self.local_data["volume"] = np.array([float(v_val) / self.mesh_area])
        
        self.iter += 1

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        self.jac = {
            "compliance": {"x_vars": self.dj.reshape(1, -1)},
            "volume": {"x_vars": self.dv.reshape(1, -1)}
        }
