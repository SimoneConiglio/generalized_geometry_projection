import numpy as np
from gemseo.core.discipline.discipline import Discipline

class GGPGeometryDiscipline(Discipline):
    """
    GEMSEO Discipline for GGP Geometry Mapping.
    Input: x_vars (primitive parameters)
    Output: density (field)
    """
    def __init__(self, mapper, x_init, name="GGP_Geometry"):
        super().__init__(name=name)
        self.mapper = mapper
        
        # IO Grammars
        self.input_grammar.update_from_names(["x_vars"])
        self.output_grammar.update_from_names(["density"])
        self.default_inputs = {"x_vars": x_init}
        
        # Internal state for derivatives
        self.ctrls = [mapper.eps_safe for _ in range(len(x_init))] # Placeholder type

    def _run(self):
        x_vars = self.local_data["x_vars"].flatten()
        # This discipline is purely for the forward mapping in the GEMSEO chain.
        # However, for dolfin-adjoint to work across disciplines, 
        # we need to be careful about how the tape is shared.
        
        # In the modular approach, we just return the density as a numpy-compatible object 
        # or a FEniCS Function if using a specific adapter.
        # For simplicity in this benchmark, we'll keep the FEniCS objects.
        rho = self.mapper.map_to_density(x_vars) 
        self.local_data["density"] = rho 

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        # Derivatives will be handled by dolfin-adjoint internally or via GEMSEO chain rule
        pass

class GGPPhysicsDiscipline(Discipline):
    """
    GEMSEO Discipline for GGP Physics Solve.
    Input: density
    Output: compliance, volume
    """
    def __init__(self, solver, mesh_area, name="GGP_Physics"):
        super().__init__(name=name)
        self.solver = solver
        self.mesh_area = mesh_area
        
        # IO Grammars
        self.input_grammar.update_from_names(["density"])
        self.output_grammar.update_from_names(["compliance", "volume"])

    def _run(self):
        rho = self.local_data["density"]
        u = self.solver.solve(rho)
        j_val = self.solver.compute_compliance(u)
        v_val = self.solver.compute_volume(rho)
        
        self.local_data["compliance"] = np.array([float(j_val)])
        self.local_data["volume"] = np.array([float(v_val) / self.mesh_area])

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        pass
