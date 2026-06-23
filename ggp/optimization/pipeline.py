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
        # Normalize each constraint by its rhs (delta or BL > 0) so values are O(1).
        # This matches the Matlab ALM_constraint.m form g = (A x)/b - 1, which is
        # what makes KS aggregation with rho=40 numerically well-behaved (raw
        # physical residuals span tens of units → exp(40*res) overflows).
        self.local_data["overhang"] = (self.A @ x_unscaled) / self.b_rhs - 1.0

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        scale = self.ub - self.lb
        self.jac = {
            "overhang": {
                "x_vars": (self.A * scale[np.newaxis, :]) / self.b_rhs[:, np.newaxis]
            }
        }


class GGPPipeline:
    """Orchestrates the entire GGP optimization process."""

    def __init__(self, spec: ProblemSpec):
        self.spec = spec

    @staticmethod
    def _make_init(mode: str, num_vars: int, **kwargs) -> np.ndarray:
        """Return a normalized [0,1] initial design vector for the legacy seeds.

        Delegates to the verbatim grid / ALM initial guesses in
        :mod:`ggp.initialization.legacy`.  Mesh / unit-cell patterns are handled
        separately in :meth:`run` via the initialization registry.
        """
        from ggp.initialization.legacy import (
            make_alm_init,
            make_grid_init_2d,
            make_grid_init_3d,
        )

        n = num_vars
        if mode in ("Free", "2D_Free"):
            return make_grid_init_2d(n, **kwargs)
        if mode in ("ALM", "2D_ALM"):
            return make_alm_init(n, **kwargs)
        if mode == "3D_Free":
            return make_grid_init_3d(n, **kwargs)
        # Fallback for other modes
        return np.random.default_rng(42).uniform(0.4, 0.6, n)

    def _domain_box(self, Lx: float, Ly: float, Lz):
        """Build a :class:`DomainBox` (extents + non-design void regions)."""
        from ggp.initialization import DomainBox

        non_design = []
        for g in self.spec.geometries:
            if g.role == "non_design" and g.type == "box":
                origin = np.asarray(g.params.get("origin", [0.0, 0.0]), dtype=float)
                far = [Lx, Ly] if Lz is None else [Lx, Ly, Lz]
                extent = np.asarray(g.params.get("extent", far), dtype=float)
                non_design.append((origin, extent))
        return DomainBox(Lx=Lx, Ly=Ly, Lz=Lz, non_design=non_design)

    def _resolve_init_pattern(self, dim: int):
        """Resolve the configured init pattern to (name, is_mesh_pattern).

        ALM always uses its legacy staircase.  For Free modes the default is the
        dimension-aware mesh pattern (``tri2d`` in 2-D, ``tet3d`` in 3-D);
        ``grid`` selects the legacy crossed-bar seed.
        """
        from ggp.initialization import is_pattern, list_patterns

        if "ALM" in (self.spec.formulation.mode or ""):
            return ("alm", False)
        name = self.spec.init.pattern
        if name is None:
            name = "tri2d" if dim == 2 else "tet3d"
        if name == "grid":
            return ("grid", False)
        if is_pattern(name):
            return (name, True)
        raise ValueError(
            f"Unknown init pattern '{name}'. Use 'grid' or one of {list_patterns()}."
        )

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

        # 2b. Initial-guess skeleton (mesh / unit-cell patterns).
        # For a mesh pattern the tessellation edge count drives num_components, so
        # the skeleton must be built before the geometry discipline.  ALM and the
        # legacy "grid" pattern keep the spec's num_components.
        dim = 3 if Lz is not None else 2
        box = self._domain_box(Lx, Ly, Lz)
        init_pattern, init_is_mesh = self._resolve_init_pattern(dim)
        init_skeleton = None
        num_components = self.spec.formulation.num_components
        if init_is_mesh:
            from ggp.initialization import get_pattern

            cell_size = self.spec.init.cell_size
            if cell_size is None:
                # dimension-aware default: half the smallest extent in 2-D
                # (matches the canonical ~15 for a 60x30 domain), the smallest
                # extent in 3-D (3-D edge density is much higher).
                cell_size = min(box.extents) / 2.0 if dim == 2 else min(box.extents)
            init_skeleton = get_pattern(init_pattern).generate(
                box, cell_size=cell_size, **self.spec.init.params
            ).filter_edges_in_domain(box)
            num_components = init_skeleton.num_edges
            if num_components != self.spec.formulation.num_components:
                print(
                    f"  [init] pattern '{init_pattern}' (cell_size={cell_size:g}) "
                    f"-> {num_components} components "
                    f"(overriding num_components={self.spec.formulation.num_components})"
                )

        # 3. Geometry Discipline
        # Use pp=100 to match Matlab smooth_sat.m (sharper binary saturation)
        geom_kwargs = {
            "num_layers": self.spec.formulation.num_layers,
            "comp_per_layer": self.spec.formulation.comp_per_layer,
            "layer_height": self.spec.formulation.layer_height,
            "ka": 10.0,
            "pp": 100.0,
            "method": self.spec.formulation.method,
            "r_gp": self.spec.formulation.r_gp,
        }

        geom_discipline = GGPGeometryDiscipline(
            mesh=analysis.mesh,
            num_components=num_components,
            mode=self.spec.formulation.mode,
            **{k: v for k, v in geom_kwargs.items() if v is not None}
        )

        # 4. Physics Discipline
        fixed_dofs = list(analysis.point_fixed_dofs)
        for bc in analysis.bcs_applied:
            fixed_dofs.extend(bc.get_boundary_values().keys())
        fixed_dofs = sorted(list(set(fixed_dofs)))

        # GP method: linear stiffness (p=1, no SIMP on top of KS saturation)
        # AMNA/MNA methods: SIMP with p=3
        method = self.spec.formulation.method or "GP"
        p_penalty = 1.0 if method == "GP" else 3.0
        # Emin: the reference GP branch (model_updateM.py) computes E = rho*E0 with
        # NO Emin floor (void E -> ~0 via the smooth-saturation residual). GGP-Topo's
        # SIMP form adds Emin, which spuriously raises void E and, because the initial
        # design is mostly void, shifts compliance ~1.5% and steers the optimizer to a
        # different (asymmetric) optimum. So the GP method always runs with Emin=0;
        # every other method keeps the 1e-6 floor.
        #
        # Emin=0 leaves genuinely material-free DOFs (e.g. the interior of a forced
        # non-design void region, as in the L-shape bracket) with an all-zero
        # stiffness row, which would make the global matrix singular. That is handled
        # in the FE solve by pinning those isolated DOFs to zero, NOT by floating the
        # whole void on a fake Emin stiffness, so the load-carrying physics is exact.
        e_min = 0.0 if method == "GP" else 1e-6

        phys_discipline = GGPPhysicsDiscipline(
            V_u=analysis.function_spaces["u"],
            ke_ref=analysis.ke_ref,
            fixed_dofs=fixed_dofs,
            f_vec=analysis.load_vector,
            mesh_area=mesh_area,
            volfrac=self.spec.volfrac,
            iterative=self.spec.solver.iterative,
            p_penalty=p_penalty,
            Emin=e_min,
            E0=1.0,
            empty_elements=analysis.empty_elements if analysis.empty_elements else None,
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
                * num_components
            )

        _init_kwargs = {}
        if mode in ["ALM", "2D_ALM"] and self.spec.formulation.num_layers:
            _oh = next((c for c in self.spec.constraints if c.name == "overhang"), None)
            _alpha = _oh.params.get("alpha_deg", 45.0) if _oh else 45.0
            _init_kwargs = {
                "np_val":       self.spec.formulation.comp_per_layer,
                "nY":           self.spec.formulation.num_layers,
                "layer_height": self.spec.formulation.layer_height,
                "alpha_deg":    _alpha,
                "Lx":           Lx,
            }
        elif mode in ("Free", "2D_Free"):
            # Pass domain extents and mapper bounds for Matlab-style grid init
            _init_kwargs = {
                "Lx": Lx,
                "Ly": Ly,
                "lb": geom_discipline.lb,
                "ub": geom_discipline.ub,
            }
            # Pass non-design origin for L-shape aware initialization
            for geom in self.spec.geometries:
                if geom.role == "non_design" and geom.type == "box":
                    _init_kwargs["non_design_origin"] = geom.params.get("origin", None)
                    break
        elif mode == "3D_Free":
            _init_kwargs = {
                "Lx": Lx,
                "Ly": Ly,
                "Lz": Lz if Lz is not None else 30.0,
                "lb": geom_discipline.lb,
                "ub": geom_discipline.ub,
            }
        if init_is_mesh:
            # Mesh / unit-cell pattern: encode one bar per skeleton edge, fitting
            # the bar thickness so the seeded design starts near the target volume
            # fraction (avoids the near-empty legacy 3-D seed).
            from ggp.initialization import encode_skeleton, fit_thickness_to_volfrac

            lb = np.asarray(geom_discipline.lb)
            ub = np.asarray(geom_discipline.ub)
            membership = self.spec.init.membership
            thickness = self.spec.init.thickness
            if thickness is None and self.spec.init.fit_volfrac:
                def _evaluate(xv):
                    geom_discipline.execute({"x_vars": xv})
                    return np.asarray(geom_discipline.local_data["rho_E"]).flatten()

                h_idx = 3 if dim == 2 else 4  # mapper 'h' (thickness) variable
                thickness = fit_thickness_to_volfrac(
                    init_skeleton, lb, ub,
                    membership=membership,
                    evaluate=_evaluate,
                    target_volfrac=self.spec.volfrac,
                    h_bounds=(float(lb[h_idx]), float(ub[h_idx])),
                )
            elif thickness is None:
                thickness = 2.0
            x_init = encode_skeleton(
                init_skeleton, lb, ub, thickness=thickness, membership=membership
            )
        else:
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

        # NOTE on overhang aggregation: the Matlab reference (ALM_constraint.m)
        # aggregates all overhang sub-constraints into a single KS scalar (ka=40).
        # GEMSEO offers this natively via
        #   problem.constraints.aggregate(idx, method="upper_bound_KS", rho=ka)
        # but upper_bound_KS is conservative (over-estimates the max), which shrinks
        # the feasible polytope and raises compliance (~262 vs ~195). We therefore
        # keep the (nY-1)*np*2 individual linear inequalities, which give MMA the
        # exact feasible region. The constraint values are still normalised to O(1)
        # in _OverhangDiscipline, matching the Matlab g = (A x)/b - 1 form.

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
