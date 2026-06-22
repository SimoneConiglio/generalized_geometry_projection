# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Optimization pipeline orchestration."""
from __future__ import annotations

import time
import numpy as np

from ggp.problem.spec import ProblemSpec
from ggp.geometry.io.registry import get_reader
from ggp.discretisation.fem import FEMDiscretiser
from ggp.gemseo_wrappers.geometry_discipline import GGPGeometryDiscipline
from ggp.gemseo_wrappers.physics_discipline import GGPPhysicsDiscipline
from .results import OptimisationResult

from gemseo import create_scenario, create_design_space
from gemseo.mda.mda_chain import MDAChain
from gemseo.core.discipline.discipline import Discipline


class _OverhangDiscipline(Discipline):
    """Overhang + optional bridge-length constraint discipline for ALM 2-D mode.

    Evaluates  A * x_unscaled - b <= 0  (linear, possibly including theta0 correction).
    """

    def __init__(self, num_layers, comp_per_layer, layer_height, alpha_deg, lb, ub,
                 bridge_length=None):
        super().__init__("OverhangDiscipline")
        from ggp.utils.alm_utils import (create_alm_overhang_constraints,
                                          create_bridge_length_constraints)
        A_oh, b_oh = create_alm_overhang_constraints(
            num_layers, comp_per_layer, layer_height, alpha_deg
        )
        if bridge_length is not None and bridge_length > 0:
            A_bl, b_bl = create_bridge_length_constraints(
                num_layers, comp_per_layer, bridge_length
            )
            self.A     = np.vstack([A_oh, A_bl])
            self.b_rhs = np.concatenate([b_oh, b_bl])
        else:
            self.A     = A_oh
            self.b_rhs = b_oh

        self.lb = lb
        self.ub = ub
        n = lb.shape[0]
        self.input_grammar.update_from_names(["x_vars"])
        self.output_grammar.update_from_names(["overhang"])
        self.default_inputs = {"x_vars": np.full(n, 0.5)}
        if hasattr(self, "cache"):
            self.cache = None
        if hasattr(self, "cache_type"):
            self.cache_type = Discipline.CacheType.NONE

    def _run(self, input_data=None):
        if input_data is not None:
            self.local_data.update(input_data)
        x = self.local_data["x_vars"].flatten()
        x_unscaled = self.lb + x * (self.ub - self.lb)
        self.local_data["overhang"] = self.A @ x_unscaled - self.b_rhs

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        scale = self.ub - self.lb
        self.jac = {"overhang": {"x_vars": self.A * scale[np.newaxis, :]}}


class GGPPipeline:
    """Orchestrates the entire GGP optimization process."""

    def __init__(self, spec: ProblemSpec):
        self.spec = spec

    @staticmethod
    def _make_init(mode: str, num_vars: int, **kwargs) -> np.ndarray:
        """Return a normalized [0,1] initial design vector appropriate for *mode*.

        The key insight is that thin initial bars (small h) let MMA explore
        freely, while thick blobs saturate KS and freeze gradients.
        """
        n = num_vars
        rng = np.random.default_rng(42)  # fixed seed for reproducibility

        if mode in ("Free", "2D_Free"):
            # 6 vars per component: [Xc, Yc, L, h, Theta, Mc]
            # Matches Matlab reference initialization: spread on a grid, h=minh (norm=0),
            # L = (domain_width / n_comp / 2) spread, Mc = initial_d = 0.5
            x = np.empty(n)
            nc = n // 6
            x[0::6] = rng.uniform(0.1, 0.9, nc)   # Xc: spread across domain
            x[1::6] = rng.uniform(0.1, 0.9, nc)   # Yc: spread across domain
            x[2::6] = 0.25                          # L: medium length
            x[3::6] = 0.0                           # h: minimum thickness (h=minh, norm=0)
            x[4::6] = rng.uniform(0.4, 0.6, nc)   # Theta: near-zero angle
            x[5::6] = 0.50                          # Mc: initial_d=0.5
            return x

        if mode in ("ALM", "2D_ALM"):
            # Interleaved layout: [Xc_0_0, L_0_0, ..., h_0..h_{np-1}, Mc_0..Mc_{np-1}, y0, theta0]
            # n = 2*nY*np_val + 2*np_val + 2
            # Passed via kwargs: np_val, nY
            np_val = kwargs.get("np_val", 1)
            nY     = kwargs.get("nY", 1)
            n_xl   = 2 * nY * np_val
            x = np.empty(n)
            # [Xc, L] interleaved: spread Xc uniformly, thin L
            comp_positions_norm = np.tile(np.linspace(0.1, 0.9, max(np_val, 1)), nY)
            x[0:n_xl:2] = comp_positions_norm[:nY * np_val]  # Xc normalised
            x[1:n_xl:2] = 0.05                                # L: thin
            # h: mid-range (0.6 normalised → actual ~0.2+0.6*0.8 in default_bounds)
            x[n_xl       : n_xl + np_val] = 0.8
            # Mc: medium
            x[n_xl + np_val : n_xl + 2*np_val] = 0.50
            # y0, theta0: neutral
            if n >= n_xl + 2*np_val + 2:
                x[n_xl + 2*np_val]     = 0.5  # y0 at mid-range (normalised)
                x[n_xl + 2*np_val + 1] = 0.5  # theta0 at mid-range (0 rotation)
            return x

        if mode == "3D_Free":
            # 8 vars per component: [Xc, Yc, Zc, L, h, Theta, Phi, Mc]
            x = np.empty(n)
            nc = n // 8
            x[0::8] = rng.uniform(0.1, 0.9, nc)        # Xc
            x[1::8] = rng.uniform(0.1, 0.9, nc)        # Yc
            x[2::8] = rng.uniform(0.1, 0.9, nc)        # Zc
            x[3::8] = 0.25                              # L
            x[4::8] = 0.02                              # h (thin)
            x[5::8] = rng.uniform(0.4, 0.6, nc)        # Theta
            x[6::8] = rng.uniform(0.4, 0.6, nc)        # Phi
            x[7::8] = 0.50                              # Mc
            return x

        # Fallback for other modes
        return rng.uniform(0.4, 0.6, n)

    def run(self) -> OptimisationResult:
        start_time = time.time()

        # 1. Geometry I/O
        domain_geom = self.spec.geometries[0]
        reader = get_reader(domain_geom.type)
        domain = reader.read(domain_geom)

        # 2. FEM Discretisation
        discretiser = FEMDiscretiser()
        analysis = discretiser.discretise(domain, self.spec)

        Lx = domain.metadata.get("Lx", 1.0)
        Ly = domain.metadata.get("Ly", 1.0)
        Lz = domain.metadata.get("Lz", None)
        mesh_area = Lx * Ly * (Lz if Lz is not None else 1.0)

        # 3. Geometry Discipline
        geom_kwargs = {
            "num_layers": self.spec.formulation.num_layers,
            "comp_per_layer": self.spec.formulation.comp_per_layer,
            "layer_height": self.spec.formulation.layer_height,
            "ka": 10.0,
            "pp": 10.0,
            "method": self.spec.formulation.method,
            "r_gp": self.spec.formulation.r_gp,
        }

        geom_discipline = GGPGeometryDiscipline(
            mesh=analysis.mesh,
            num_components=self.spec.formulation.num_components,
            mode=self.spec.formulation.mode,
            **{k: v for k, v in geom_kwargs.items() if v is not None}
        )

        # 4. Physics Discipline
        fixed_dofs = list(analysis.point_fixed_dofs)
        for bc in analysis.bcs_applied:
            fixed_dofs.extend(bc.get_boundary_values().keys())
        fixed_dofs = sorted(list(set(fixed_dofs)))

        phys_discipline = GGPPhysicsDiscipline(
            V_u=analysis.function_spaces["u"],
            ke_ref=analysis.ke_ref,
            fixed_dofs=fixed_dofs,
            f_vec=analysis.load_vector,
            mesh_area=mesh_area,
            volfrac=self.spec.volfrac,
            iterative=self.spec.solver.iterative,
            p_penalty=3.0,
            Emin=1e-6,
            E0=1.0
        )

        # 5. Design Space
        design_space = create_design_space()
        mode = self.spec.formulation.mode

        if mode in ["ALM", "2D_ALM"] and self.spec.formulation.num_layers:
            _np_val = self.spec.formulation.comp_per_layer
            _nY     = self.spec.formulation.num_layers
            num_vars = 2 * _nY * _np_val + 2 * _np_val + 2  # [Xc,L]+h+Mc+[y0,theta0]
        elif mode == "3D_ALM" and self.spec.formulation.num_layers:
            num_vars = (
                6
                * self.spec.formulation.comp_per_layer
                * self.spec.formulation.num_layers
            )
        else:
            num_vars = (
                geom_discipline.mapper.num_vars_per_component()
                * self.spec.formulation.num_components
            )

        _init_kwargs = {}
        if mode in ["ALM", "2D_ALM"] and self.spec.formulation.num_layers:
            _init_kwargs = {
                "np_val": self.spec.formulation.comp_per_layer,
                "nY":     self.spec.formulation.num_layers,
            }
        x_init = self._make_init(mode, num_vars, **_init_kwargs)
        design_space.add_variable(
            "x_vars", size=num_vars, lower_bound=0.0, upper_bound=1.0, value=x_init
        )

        # 6. MDA Chain
        chain = MDAChain([geom_discipline, phys_discipline])
        if hasattr(chain, "cache"):
            chain.cache = None
        if hasattr(chain, "cache_type"):
            chain.cache_type = chain.CacheType.NONE

        # 7. Collect additional disciplines (overhang constraint)
        extra_disciplines: list[Discipline] = []
        has_overhang = any(c.name == "overhang" for c in self.spec.constraints)
        if has_overhang and mode in ["ALM", "2D_ALM"] and self.spec.formulation.num_layers:
            overhang_spec = next(c for c in self.spec.constraints if c.name == "overhang")
            alpha_deg = overhang_spec.params.get("alpha_deg", 45.0)
            bridge_len = overhang_spec.params.get("bridge_length", None)
            oh_disc = _OverhangDiscipline(
                num_layers=self.spec.formulation.num_layers,
                comp_per_layer=self.spec.formulation.comp_per_layer,
                layer_height=self.spec.formulation.layer_height,
                alpha_deg=alpha_deg,
                lb=geom_discipline.lb,
                ub=geom_discipline.ub,
                bridge_length=bridge_len,
            )
            extra_disciplines.append(oh_disc)

        # 8. Scenario
        all_disciplines = [chain] + extra_disciplines
        scenario = create_scenario(
            all_disciplines,
            objective_name="compliance",
            design_space=design_space,
            maximize_objective=False,
            formulation_name="MDF"
        )

        for c in self.spec.constraints:
            if c.name == "volume":
                scenario.add_constraint(
                    "volume", constraint_type="ineq", positive=False, value=0.0
                )
            elif c.name == "overhang" and extra_disciplines:
                scenario.add_constraint(
                    "overhang", constraint_type="ineq", positive=False, value=0.0
                )

        algo_options = self.spec.solver.options.copy()
        if "max_iter" not in algo_options:
            algo_options["max_iter"] = self.spec.solver.max_iter
        if "algo_name" not in algo_options:
            algo_options["algo_name"] = self.spec.solver.algorithm

        scenario.execute(**algo_options)

        opt_result = scenario.optimization_result

        # 9. Capture final density field at the optimal point
        x_opt = opt_result.x_opt
        try:
            geom_discipline.execute({"x_vars": x_opt})
            density_field = np.asarray(geom_discipline.local_data["rho_E"]).flatten().copy()
            if analysis.empty_elements:
                density_field[analysis.empty_elements] = 0.0
        except Exception:
            density_field = None

        n_iter = len(scenario.formulation.optimization_problem.database)

        result = OptimisationResult(
            problem_name="optimization_run",
            algorithm=self.spec.solver.algorithm,
            status=opt_result.status,
            iterations=n_iter,
            objective_value=float(opt_result.f_opt),
            max_constraint_violation=0.0,
            design_variables=x_opt,
            history={},
            execution_time_s=time.time() - start_time,
            density_field=density_field,
            eval_coords=geom_discipline.eval_coords.copy(),
            dim=domain.dim,
        )

        return result
