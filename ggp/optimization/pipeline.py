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


class _DeflatedObjectiveDiscipline(Discipline):
    """Deflated-objective discipline used by the deflation global-search strategy.

    Implements the Farrell/Papadopoulos/Surowiec deflation operator: given a set
    of previously found minima (``roots`` in normalised [0,1] design space), the
    raw objective ``J(x) = log(C+1)`` is multiplied by a deflation factor

        M(x) = shift + sum_i  1 / ||x - x_i||^power

    so that ``J_def(x) = J(x) * M(x)``.  M(x) -> +inf as x approaches any known
    root, which makes the deflated objective repel the optimiser from minima it
    has already discovered and steers it towards distinct ones.

    Inputs : ``compliance`` (== log(C+1), the physics output) and ``x_vars``.
    Output : ``compliance_deflated`` (the new scenario objective).
    """

    def __init__(self, roots, num_vars, shift=1.0, power=2.0, eps=1e-6):
        super().__init__("DeflatedObjective")
        self.roots = [np.asarray(r, dtype=float).flatten() for r in roots]
        self.shift = float(shift)
        self.power = float(power)
        self.eps = float(eps)
        self.input_grammar.update_from_names(["compliance", "x_vars"])
        self.output_grammar.update_from_names(["compliance_deflated"])
        self.default_inputs = {
            "compliance": np.array([1.0]),
            "x_vars": np.full(num_vars, 0.5),
        }
        if hasattr(self, "cache"):
            self.cache = None
        if hasattr(self, "cache_type"):
            self.cache_type = Discipline.CacheType.NONE

    def _factor_and_grad(self, x):
        """Return (M, dM/dx) for the deflation factor at design point *x*."""
        m = self.shift
        grad = np.zeros_like(x)
        for r in self.roots:
            d = x - r
            dist2 = float(d @ d) + self.eps ** 2
            dist = np.sqrt(dist2)
            # term = dist^-power ; d(term)/dx = -power * dist^(-power-2) * d
            term = dist ** (-self.power)
            m += term
            grad += -self.power * dist ** (-self.power - 2.0) * d
        return m, grad

    def _run(self, input_data=None):
        if input_data is not None:
            self.local_data.update(input_data)
        j = float(np.asarray(self.local_data["compliance"]).flatten()[0])
        x = np.asarray(self.local_data["x_vars"]).flatten()
        m, _ = self._factor_and_grad(x)
        self.local_data["compliance_deflated"] = np.array([j * m])

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        j = float(np.asarray(self.local_data["compliance"]).flatten()[0])
        x = np.asarray(self.local_data["x_vars"]).flatten()
        m, dm = self._factor_and_grad(x)
        # d(J*M)/dJ = M ; d(J*M)/dx = J * dM/dx
        self.jac = {
            "compliance_deflated": {
                "compliance": np.array([[m]]),
                "x_vars": (j * dm)[np.newaxis, :],
            }
        }


class GGPPipeline:
    """Orchestrates the entire GGP optimization process."""

    def __init__(self, spec: ProblemSpec, x0=None, overrides=None, deflation=None):
        """Orchestrate a single GGP optimisation run.

        Parameters
        ----------
        spec : ProblemSpec
            The frozen problem specification (single source of truth).
        x0 : np.ndarray, optional
            Normalised [0,1] initial design vector. When given it overrides the
            deterministic :meth:`_make_init` starting point, enabling warm-starts,
            random restarts and basin-hopping perturbations.
        overrides : dict, optional
            Per-run overrides for the GGP sharpness / penalisation knobs. Recognised
            keys: ``ka``, ``pp``, ``r_gp``, ``gammac``, ``gammav`` (geometry) and
            ``p_penalty``, ``Emin`` (physics). Used by the continuation strategy.
        deflation : dict, optional
            When present, the raw objective is deflated to repel known minima.
            Keys: ``roots`` (list of normalised [0,1] design vectors), ``shift``
            (default 1.0), ``power`` (default 2.0).
        """
        self.spec = spec
        self.x0 = None if x0 is None else np.asarray(x0, dtype=float).flatten()
        self.overrides = dict(overrides) if overrides else {}
        self.deflation = dict(deflation) if deflation else None

    @staticmethod
    def _make_init(mode: str, num_vars: int, **kwargs) -> np.ndarray:
        """Return a normalized [0,1] initial design vector appropriate for *mode*.

        For Free 2D, replicates the Matlab GGP_main.m initialization:
        paired crossed bars on a regular 3x3 grid at ±atan2(Ly,Lx) angle,
        L = 2*sqrt((Lx/3)^2+(Ly/3)^2), h=2, Mc=0.5.
        """
        n = num_vars

        if mode in ("Free", "2D_Free"):
            # Matlab-style grid initialization (GGP_main.m lines 159-168)
            # ncx=1, ncy=1 → 3×3 grid of (ncx+2)×(ncy+2) positions
            Lx = kwargs.get("Lx", 60.0)
            Ly = kwargs.get("Ly", 30.0)
            lb = kwargs.get("lb", None)
            ub = kwargs.get("ub", None)
            nc = n // 6
            ncx, ncy = 1, 1
            # Standard Matlab-style grid (same for rectangular and L-shape domains).
            # For the L-shape, only 1 of 9 positions is in the non-design region —
            # the other 8 (including corners like (0,Ly) at ±theta) provide structural
            # connectivity across both arms.  The empty-element override in physics
            # handles the non-design region correctly.
            xp = np.linspace(0.0, Lx, ncx + 2)
            yp = np.linspace(0.0, Ly, ncy + 2)
            xx, yy = np.meshgrid(xp, yp)
            grid_X = xx.flatten()
            grid_Y = yy.flatten()
            half = nc // 2
            theta = np.arctan2(Ly / ncy, Lx / ncx)
            Lc = 2.0 * np.sqrt((Lx / (ncx + 2)) ** 2 + (Ly / (ncy + 2)) ** 2)

            # Paired bars: first nc//2 at +theta, remaining at -theta
            n_grid = len(grid_X)
            idx_pos = np.arange(half) % n_grid
            idx_neg = np.arange(nc - half) % n_grid
            Xc = np.concatenate([grid_X[idx_pos], grid_X[idx_neg]])
            Yc = np.concatenate([grid_Y[idx_pos], grid_Y[idx_neg]])
            Tc = np.concatenate([theta * np.ones(half), -theta * np.ones(nc - half)])

            hc = 2.0   # initial h just above minh=1
            Mc = 0.5   # initial_d

            # Normalize to [0,1] using mapper bounds
            x = np.empty(n)
            if lb is not None and ub is not None:
                x[0::6] = np.clip((Xc - lb[0::6]) / (ub[0::6] - lb[0::6]), 0.0, 1.0)
                x[1::6] = np.clip((Yc - lb[1::6]) / (ub[1::6] - lb[1::6]), 0.0, 1.0)
                x[2::6] = np.clip((Lc - lb[2::6]) / (ub[2::6] - lb[2::6]), 0.0, 1.0)
                x[3::6] = np.clip((hc - lb[3::6]) / (ub[3::6] - lb[3::6]), 0.0, 1.0)
                x[4::6] = np.clip((Tc - lb[4::6]) / (ub[4::6] - lb[4::6]), 0.0, 1.0)
            else:
                # Fallback: rough normalized values
                x[0::6] = Xc / (Lx + 2)
                x[1::6] = Yc / (Ly + 2)
                x[2::6] = Lc / np.sqrt(Lx**2 + Ly**2)
                x[3::6] = 0.015
                x[4::6] = 0.5
            x[5::6] = Mc
            return x

        if mode in ("ALM", "2D_ALM"):
            # Interleaved layout: [Xc_0_0, L_0_0, ..., h_0..h_{np-1}, Mc_0..Mc_{np-1}, y0, theta0]
            np_val     = kwargs.get("np_val", 1)
            nY         = kwargs.get("nY", 1)
            layer_h    = kwargs.get("layer_height", 3.0)
            alpha_deg  = kwargs.get("alpha_deg", 45.0)
            n_xl       = 2 * nY * np_val
            x = np.empty(n)

            # Staircase initialization: each column is a maximum-overhang ascending
            # staircase so that the rightmost column reaches x=Lx at mid-height
            # (the load layer), giving non-zero gradient from iteration 1.
            # Physical Xc bounds: lb=-1, ub=Lx+1 (range = Lx+2).
            Lx         = kwargs.get("Lx", 60.0)
            xc_range   = Lx + 2.0                          # ub - lb = (Lx+1) - (-1)
            delta_norm = np.tan(np.deg2rad(alpha_deg)) * layer_h / xc_range
            load_layer  = nY // 2
            P_bot_right = (Lx + 1.0) / xc_range - load_layer * delta_norm
            P_bot_left  = (0.0 + 1.0) / xc_range
            P_bottom = np.linspace(P_bot_left, P_bot_right, np_val)

            # Assign Xc[k, j] = P_bottom[j] + k * delta_norm (clamped to [0,1])
            # F-order: x_vars index of Xc[k,j] = 2*(j*nY + k)
            for j in range(np_val):
                for k in range(nY):
                    x[2*(j*nY + k)]     = float(np.clip(P_bottom[j] + k*delta_norm, 0.0, 1.0))
                    x[2*(j*nY + k) + 1] = 0.333   # L normalized ≈ 6 physical

            x[n_xl       : n_xl + np_val] = 1.0   # h = 1 (full height)
            x[n_xl + np_val : n_xl + 2*np_val] = 0.50  # Mc = 0.5
            if n >= n_xl + 2*np_val + 2:
                x[n_xl + 2*np_val]     = 0.5    # y0 = 0
                # theta0 = 0 (normalised 0.5 of the [-pi/2, pi/2] range). theta0=0 is
                # the optimal build orientation for the cantilever: jointly-optimized
                # compliance was tested at theta0 = 0 / -56 / -90 deg -> C = 86 / 108 /
                # 131, i.e. it worsens monotonically as the plane rotates away from 0
                # (vertical layers let the bars form the efficient horizontal load path;
                # beam-axis layering at +/-90 is less efficient).
                x[n_xl + 2*np_val + 1] = 0.5    # theta0 = 0
            return x

        if mode == "3D_Free":
            # 8 vars per component: [Xc, Yc, Zc, L, h, Theta, Phi, Mc]
            # Grid init: 3D cross-bar pairs on a regular grid (analogous to 2D)
            Lx = kwargs.get("Lx", 60.0)
            Ly = kwargs.get("Ly", 30.0)
            Lz = kwargs.get("Lz", 30.0)
            lb = kwargs.get("lb", None)
            ub = kwargs.get("ub", None)
            nc = n // 8

            # Build 3-D grid: sample independently along each axis so the first
            # nc//2 positions are spread across x, y, z simultaneously (diagonal sweep).
            half = nc // 2
            grid_X = np.linspace(0.0, Lx, half + 1)[:-1]
            grid_Y = np.linspace(0.0, Ly, half + 1)[:-1]
            grid_Z = np.linspace(0.0, Lz, half + 1)[:-1]

            # Diagonal bar length spanning ~1/half of each axis
            Lc = 2.0 * np.sqrt((Lx / half) ** 2 + (Ly / half) ** 2 + (Lz / half) ** 2)
            theta = np.arctan2(Ly / half, Lx / half)
            phi = np.arctan2(Lz / half, np.sqrt((Lx / half) ** 2 + (Ly / half) ** 2))

            Xc = np.concatenate([grid_X, grid_X[: nc - half]])
            Yc = np.concatenate([grid_Y, grid_Y[: nc - half]])
            Zc = np.concatenate([grid_Z, grid_Z[: nc - half]])
            Tc = np.concatenate([theta * np.ones(half), -theta * np.ones(nc - half)])
            Pc = np.concatenate([phi * np.ones(half), -phi * np.ones(nc - half)])

            hc = 2.0
            Mc = 0.5

            x = np.empty(n)
            if lb is not None and ub is not None:
                x[0::8] = np.clip((Xc - lb[0::8]) / (ub[0::8] - lb[0::8]), 0.0, 1.0)
                x[1::8] = np.clip((Yc - lb[1::8]) / (ub[1::8] - lb[1::8]), 0.0, 1.0)
                x[2::8] = np.clip((Zc - lb[2::8]) / (ub[2::8] - lb[2::8]), 0.0, 1.0)
                x[3::8] = np.clip((Lc - lb[3::8]) / (ub[3::8] - lb[3::8]), 0.0, 1.0)
                x[4::8] = np.clip((hc - lb[4::8]) / (ub[4::8] - lb[4::8]), 0.0, 1.0)
                x[5::8] = np.clip((Tc - lb[5::8]) / (ub[5::8] - lb[5::8]), 0.0, 1.0)
                x[6::8] = np.clip((Pc - lb[6::8]) / (ub[6::8] - lb[6::8]), 0.0, 1.0)
            else:
                x[0::8] = Xc / Lx
                x[1::8] = Yc / Ly
                x[2::8] = Zc / Lz
                x[3::8] = Lc / np.sqrt(Lx**2 + Ly**2 + Lz**2)
                x[4::8] = 0.02
                x[5::8] = 0.5
                x[6::8] = 0.5
            x[7::8] = Mc
            return x

        # Fallback for other modes
        return np.random.default_rng(42).uniform(0.4, 0.6, n)

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
        # Apply per-run sharpness overrides (continuation strategy).
        for _k in ("ka", "pp", "r_gp", "gammac", "gammav"):
            if _k in self.overrides and self.overrides[_k] is not None:
                geom_kwargs[_k] = self.overrides[_k]

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

        # GP method: linear stiffness (p=1, no SIMP on top of KS saturation)
        # AMNA/MNA methods: SIMP with p=3
        method = self.spec.formulation.method or "GP"
        p_penalty = 1.0 if method == "GP" else 3.0
        # Emin: the reference Free GP branch (model_updateM.py) computes E = rho*E0
        # with NO Emin floor (void E -> ~0 via the smooth-saturation residual).
        # GGP-Topo's SIMP form adds Emin, which spuriously raises void E and, because
        # the initial design is mostly void, shifts compliance ~1.5% and steers the
        # optimizer to a different (asymmetric) optimum. Match the reference: no Emin
        # floor for the Free GP case. NB: ALM modes also default to method="GP" but
        # their reference uses MNA *with* Emin (and Emin=0 makes the ALM FE singular
        # -> NaN), so the no-floor case is restricted to non-ALM modes.
        _is_alm = "ALM" in (self.spec.formulation.mode or "")
        e_min = 0.0 if (method == "GP" and not _is_alm) else 1e-6

        # Per-run penalisation overrides (continuation strategy).
        if self.overrides.get("p_penalty") is not None:
            p_penalty = self.overrides["p_penalty"]
        if self.overrides.get("Emin") is not None:
            e_min = self.overrides["Emin"]

        phys_discipline = GGPPhysicsDiscipline(
            V_u=analysis.function_spaces["u"],
            ke_ref=analysis.ke_ref,
            fixed_dofs=fixed_dofs,
            f_vec=analysis.load_vector,
            mesh_area=mesh_area,
            volfrac=self.spec.volfrac,
            iterative=self.spec.solver.iterative,
            fem_solver=self.spec.solver.fem_solver,
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
                * self.spec.formulation.num_components
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
        if self.x0 is not None:
            if self.x0.shape[0] != num_vars:
                raise ValueError(
                    f"x0 has length {self.x0.shape[0]} but the problem expects "
                    f"{num_vars} design variables."
                )
            x_init = np.clip(self.x0, 0.0, 1.0)
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

        # 7b. Deflation: repel known minima by deflating the objective.
        objective_name = "compliance"
        if self.deflation and self.deflation.get("roots"):
            defl_disc = _DeflatedObjectiveDiscipline(
                roots=self.deflation["roots"],
                num_vars=num_vars,
                shift=self.deflation.get("shift", 1.0),
                power=self.deflation.get("power", 2.0),
            )
            extra_disciplines.append(defl_disc)
            objective_name = "compliance_deflated"

        # 8. Scenario
        all_disciplines = [chain] + extra_disciplines
        scenario = create_scenario(
            all_disciplines,
            objective_name=objective_name,
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

        # objective_value is always the *raw* log(C+1), so it is comparable across
        # strategies. Under deflation the scenario objective is the deflated value,
        # so recompute the true compliance at the optimum from the geom+phys chain.
        if objective_name != "compliance":
            try:
                chain.execute({"x_vars": x_opt})
                objective_value = float(
                    np.asarray(chain.local_data["compliance"]).flatten()[0]
                )
            except Exception:
                objective_value = float("nan")
        else:
            f_opt = opt_result.f_opt
            objective_value = float(f_opt) if f_opt is not None else float("nan")

        result = OptimisationResult(
            problem_name="optimization_run",
            algorithm=self.spec.solver.algorithm,
            status=opt_result.status,
            iterations=n_iter,
            objective_value=objective_value,
            max_constraint_violation=0.0,
            design_variables=x_opt,
            history={},
            execution_time_s=time.time() - start_time,
            density_field=density_field,
            eval_coords=geom_discipline.eval_coords.copy(),
            dim=domain.dim,
        )

        return result
