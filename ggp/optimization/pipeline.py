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


class _SymmetryExpander(Discipline):
    """Enforce top-bottom (y) symmetry by expanding ``num_free`` design components
    into ``2*num_free`` geometry components: the free components plus their mirror
    images about ``y = Ly/2``.

    With the Free-2D bounds (Y symmetric about Ly/2, Theta symmetric about 0) the
    mirror of a component ``[x, y, l, h, th, m]`` in the *normalised* [0,1] space is
    exactly ``[x, 1-y, l, h, 1-th, m]`` -- a linear map with a constant Jacobian, so
    GEMSEO composes the gradients and the geometry/physics code is untouched. The
    optimiser sees only ``x_free`` (half the variables); every design is symmetric by
    construction, which removes the entire family of symmetry-broken local minima.
    """

    _MIR_A = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])   # multiplier (y, theta flip)
    _MIR_B = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])     # offset      (y->1-y, th->1-th)

    def __init__(self, num_free: int, vpc: int = 6):
        super().__init__("SymmetryExpander")
        self.num_free = num_free
        self.vpc = vpc
        n_free = num_free * vpc
        self.input_grammar.update_from_names(["x_free"])
        self.output_grammar.update_from_names(["x_vars"])
        self.default_inputs = {"x_free": np.full(n_free, 0.5)}
        # Constant Jacobian d(x_vars)/d(x_free) = [[I];[diag(mir_a)]].
        eye = np.eye(n_free)
        diag = np.diag(np.tile(self._MIR_A, num_free))
        self._jac = np.vstack([eye, diag])
        if hasattr(self, "cache"):
            self.cache = None
        if hasattr(self, "cache_type"):
            self.cache_type = Discipline.CacheType.NONE

    def expand(self, x_free: np.ndarray) -> np.ndarray:
        xf = np.asarray(x_free, float).reshape(self.num_free, self.vpc)
        mir = xf * self._MIR_A + self._MIR_B
        return np.concatenate([xf.flatten(), mir.flatten()])

    def _run(self, input_data=None):
        if input_data is not None:
            self.local_data.update(input_data)
        self.local_data["x_vars"] = self.expand(self.local_data["x_free"].flatten())

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        self.jac = {"x_vars": {"x_free": self._jac}}


def _auto_grid(nc: int, Lx: float, Ly: float) -> tuple[int, int]:
    """Factor *nc* into (gnx, gny) whose aspect ratio best matches Lx/Ly, so the
    tiles ``(Lx/gnx) x (Ly/gny)`` are as square as possible."""
    ratio = Lx / Ly
    best = (nc, 1)
    best_err = float("inf")
    for gny in range(1, nc + 1):
        if nc % gny:
            continue
        gnx = nc // gny
        err = abs((gnx / gny) - ratio)
        if err < best_err:
            best_err, best = err, (gnx, gny)
    return best


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


class _DisplacementConstraintDiscipline(Discipline):
    """Point/DOF displacement constraint ``|ℓᵀu| / u_max - 1 <= 0``.

    Not self-adjoint, so the sensitivity needs one adjoint solve reusing the physics
    discipline's cached, BC-eliminated ``K``:  with ``K λ = ∂g/∂u`` and ``∂g/∂u = sign·ℓ/u_max``,
    the total design sensitivity is ``dg/dρ_e = -λ_eᵀ (dE_drho_e · ke_ref) u_e`` (there is no
    explicit ρ-dependence — displacement enters only through ``u``).
    """

    def __init__(self, phys_discipline, ell, u_max, num_elements, volfrac=0.5,
                 name="DisplacementConstraint"):
        super().__init__(name=name)
        self.phys = phys_discipline
        self.ell = np.asarray(ell, dtype=float).flatten()   # DOF selector (n_dofs,)
        self.u_max = float(u_max)
        self.num_elements = num_elements
        self._sign = 1.0
        self.input_grammar.update_from_names(["rho_E"])
        self.output_grammar.update_from_names(["displacement"])
        self.default_inputs = {"rho_E": np.full(num_elements, volfrac)}
        if hasattr(self, "cache"):
            self.cache = None
        if hasattr(self, "cache_type"):
            self.cache_type = Discipline.CacheType.NONE

    def _run(self, input_data=None):
        if input_data is not None:
            self.local_data.update(input_data)
        u = self.phys.last_u
        val = float(self.ell @ u)
        self._sign = 1.0 if val >= 0.0 else -1.0
        self.local_data["displacement"] = np.array([abs(val) / self.u_max - 1.0])

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        dg_du = self._sign * self.ell / self.u_max
        lam = self.phys.adjoint_solve(dg_du)
        dg_drho = self.phys.dstiffness_sensitivity(lam)     # -λ_eᵀ (dE_drho ke_ref) u_e
        self.jac = {"displacement": {"rho_E": dg_drho.reshape(1, -1)}}


def _stress_exclusion_mask(analysis, eval_coords, exclude_radius):
    """Element mask excluding non-stress-measured elements.

    Two exclusions, both required for a meaningful stress measure:
    * the **load-introduction vicinity** (within ``exclude_radius`` of any loaded DOF):
      a point load's FE stress is singular (grows without bound under mesh refinement),
      so no finite allowable can hold it — standard practice in stress-constrained TO
      (Le et al. 2010 spread or exclude the load region for the same reason);
    * **non-design (empty) elements**: the physics discipline forces ``rho -> 0`` there,
      but the geometry's raw ``rho_E`` (which this discipline consumes) can be ~1 when a
      component overlaps the void region — combined with the ``Emin`` stiffness the
      recovered "solid stress" is enormous and fictitious.

    Returns ``None`` (no mask) when neither exclusion applies.
    """
    active = np.ones(eval_coords.shape[0])
    if exclude_radius > 0.0:
        loaded = np.nonzero(np.asarray(analysis.load_vector).flatten())[0]
        if len(loaded):
            dof_coords = analysis.function_spaces["u"].tabulate_dof_coordinates()
            pts = dof_coords[loaded]
            d = np.linalg.norm(eval_coords[:, None, :] - pts[None, :, :], axis=2).min(axis=1)
            active[d <= exclude_radius] = 0.0
    if analysis.empty_elements:
        active[np.asarray(analysis.empty_elements, dtype=int)] = 0.0
    return active if np.any(active == 0.0) else None


class _StressConstraintDiscipline(Discipline):
    """Aggregated von Mises stress constraint (unified aggregation-relaxation).

    The nonlinear algebra (stress recovery, von Mises, relaxation, aggregation) and its
    partials ``∂G/∂ρ|explicit`` and ``∂G/∂u`` come from the JAX kernel
    (:class:`ggp.utils.jax_sensitivity.StressConstraintKernel`); the implicit-``u`` part is
    one adjoint solve reusing the physics discipline's factorized ``K``:
        ``dG/dρ_e = ∂G/∂ρ_e − λ_eᵀ (dE_drho_e · ke_ref) u_e``,  ``K λ = ∂G/∂u``.

    Adaptive normalization (Le, Norato, Bruns, Ha & Tortorelli 2010): aggregation is a
    lower bound of the true max stress, so a converged ``G <= 0`` design can still
    violate ``σ_max <= σ_lim`` pointwise (or, with a fixed loose allowable, end
    over-conservative). With ``adaptive=True`` the emitted constraint is Le's form,
    ``c · S(x) / σ_lim - 1 <= 0``, with the *current-design* aggregate stress measure

        S(x) = σ_a*(x)   — the critical allowable, i.e. the root of ``G(x, σ_a*) = 0``
                           (the g-space analogue of σ_PN; found by bisection)

    and the **lagged, O(1) geometric ratio** ``c = σ_max/σ_a*`` from the previous
    iterate (``σ_max`` = ρ-weighted max von Mises, Verbart-consistent). Because the
    absolute stress scale enters through ``S(x)`` *instantly*, a design whose stress
    explodes within one MMA step is labeled infeasible immediately — only the mildly
    varying ratio is lagged (this is what makes Le's normalization immune to the
    phantom-feasible iterates an allowable-with-lag form produces under mass
    minimization). At a fixed design the output equals ``σ_max/σ_lim - 1`` exactly;
    fixed point: ``σ_max = σ_lim``. Sensitivity of ``σ_a*(x)`` via the implicit
    function theorem: ``dσ_a*/dρ = -(∂G/∂ρ)/(∂G/∂σ_a)`` at ``(x, σ_a*)``, with the
    ``u``-path handled by the same single adjoint solve. When the bisection bracket is
    clipped (no interior root), the raw ``G`` at the clipped allowable is emitted so
    the optimizer keeps a nonzero gradient.
    """

    def __init__(self, phys_discipline, se_ref, sigma_lim, num_elements, dim,
                 q=0.5, P=8.0, kind="verbart", volfrac=0.5,
                 adaptive=False, alpha=0.5, rho_solid=0.5, max_correction=4.0,
                 active_mask=None, name="StressConstraint"):
        super().__init__(name=name)
        from ggp.utils.jax_sensitivity import StressConstraintKernel
        self.phys = phys_discipline
        cell_dofs = np.asarray([np.asarray(phys_discipline.cell_dofs[e])
                                for e in range(num_elements)], dtype=np.int64)
        self.kernel = StressConstraintKernel(cell_dofs, se_ref, sigma_lim,
                                             q=q, P=P, kind=kind, dim=dim,
                                             active_mask=active_mask)
        self.num_elements = num_elements
        self._rho = None
        self.adaptive = bool(adaptive)
        self.alpha = float(alpha)
        self.rho_solid = float(rho_solid)
        # Safety clamp: sigma_allow stays in [sigma_lim/max_correction,
        # sigma_lim*max_correction] so an unachievable target (e.g. a stress raiser the
        # optimizer cannot remove) can never ratchet the allowable to zero and blow up
        # the constraint.
        self.max_correction = float(max_correction)
        self.sigma_lim = float(sigma_lim)
        self.sigma_allow = float(sigma_lim)     # allowable the raw G is evaluated at
        self._c = None                          # lagged Le ratio sigma_max/sigma_a*
        self._c_next = None                     # ratio recorded at the current design
        self._mode = "raw"                      # 'le' (adaptive interior) or 'raw'
        self._sa_star = None                    # current critical allowable (adaptive)
        self._prev_rho = None                   # design of the last _run (idempotency)
        self._cached_out = None
        # The Verbart weight must be the MASS density rho_V, not the penalized
        # stiffness density rho_E: with rho_E ~ Mc^gammac (gammac=3) a gray design
        # cuts the stress weight cubically while its mass falls only linearly, so
        # mass-minimization can park in a feasible zero-stiffness "gray haze". With
        # the rho_V weight the haze is infeasible (sigma_hat ~ 1/rho^3 vs weight
        # rho^1 -> g ~ 1/rho^2 grows), as in the SIMP-literature formulation.
        # rho_E stays an input for the implicit u-path (adjoint through K(rho_E)).
        self.input_grammar.update_from_names(["rho_E", "rho_V"])
        self.output_grammar.update_from_names(["stress"])
        self.default_inputs = {"rho_E": np.full(num_elements, volfrac),
                               "rho_V": np.full(num_elements, volfrac)}
        if hasattr(self, "cache"):
            self.cache = None
        if hasattr(self, "cache_type"):
            self.cache_type = Discipline.CacheType.NONE

    def _run(self, input_data=None):
        if input_data is not None:
            self.local_data.update(input_data)
        # rho_V is the Verbart weight (mass density); rho_E enters only implicitly
        # through the displacement u = u(K(rho_E)), handled by the adjoint.
        self._rho = self.local_data["rho_V"].flatten()
        u = self.phys.last_u

        if not self.adaptive:
            self._mode = "raw"
            G = self.kernel.value(self._rho, u, sigma_allow=self.sigma_allow)
            self.local_data["stress"] = np.array([G])
            return

        # IDEMPOTENT per design: GEMSEO's chain evaluates the discipline more than once
        # per optimizer iteration (value pass + linearization pass). Advancing the
        # lagged ratio on every call would make the constraint's value and gradient
        # correspond to *different* constraints within one MMA iteration — the update
        # must advance only when the design actually changes (Le's update is
        # per-design-iteration, not per-evaluation).
        if self._prev_rho is not None and np.array_equal(self._rho, self._prev_rho):
            self.local_data["stress"] = np.array([self._cached_out])
            return
        self._prev_rho = self._rho.copy()

        lo = self.sigma_lim / self.max_correction
        hi = self.sigma_lim * 1e3
        sa_star = self.kernel.critical_allowable(self._rho, u, lo=lo, hi=hi)
        # rho-weighted (Verbart-consistent) max — NOT thresholded on density: a
        # thresholded max is blind to a load path fading through intermediate
        # densities, which lets mass-minimization collapse the design "feasibly".
        wmax = self.kernel.weighted_max_stress(self._rho, u)
        interior = (sa_star > lo * 1.0001) and (sa_star < hi * 0.9999) and wmax > 0.0

        # Advance the lag: the ratio recorded at the previous design becomes current.
        if self._c_next is not None:
            self._c = self._c_next

        if interior:
            # Le's normalized constraint c·S(x)/σ_lim - 1 with S(x) = σ_a*(x) fresh at
            # the current design and c the LAGGED (previous-design) O(1) ratio. First
            # evaluation has no lag available -> use the current ratio (output then
            # equals σ_max/σ_lim - 1 exactly).
            c = self._c if self._c is not None else wmax / sa_star
            self._mode = "le"
            self._sa_star = sa_star
            self._c_used = c
            self.sigma_allow = sa_star          # introspection: current critical allowable
            out = c * sa_star / self.sigma_lim - 1.0
            self._c_next = wmax / sa_star       # ratio of THIS design, for the next one
        else:
            # No interior root (all-margin or fully violated design): emit the raw
            # aggregate at the clipped allowable so a usable gradient survives.
            self._mode = "raw"
            self.sigma_allow = float(np.clip(sa_star, lo, hi))
            out = self.kernel.value(self._rho, u, sigma_allow=self.sigma_allow)
        self._cached_out = float(out)
        self.local_data["stress"] = np.array([out])

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        u = self.phys.last_u
        if self._mode == "le":
            # d(out)/dρ = (c/σ_lim)·dσ_a*/dρ with, by the implicit function theorem at
            # G(x, σ_a*) = 0:  dσ_a*/d(·) = -(dG/d(·)) / (∂G/∂σ_a).
            _, dG_drho_expl, dG_du = self.kernel.value_and_grads(
                self._rho, u, sigma_allow=self._sa_star)
            G_sa = self.kernel.dG_dsigma_allow(self._rho, u, self._sa_star)
            lam = self.phys.adjoint_solve(dG_du)
            scale = -self._c_used / (self.sigma_lim * G_sa)
            d_rhoV = scale * dG_drho_expl                              # explicit weight path
            d_rhoE = scale * self.phys.dstiffness_sensitivity(lam)     # implicit u-path
        else:
            _, dG_drho_expl, dG_du = self.kernel.value_and_grads(
                self._rho, u, sigma_allow=self.sigma_allow)
            lam = self.phys.adjoint_solve(dG_du)
            d_rhoV = dG_drho_expl
            d_rhoE = self.phys.dstiffness_sensitivity(lam)
        self.jac = {"stress": {"rho_E": d_rhoE.reshape(1, -1),
                               "rho_V": d_rhoV.reshape(1, -1)}}


class _ShiftedObjectiveDiscipline(Discipline):
    """Affine-rescale a response used as the objective: ``(base + shift) * scale``.

    An affine map changes neither the minimizer nor the gradient direction — it exists
    for two gemseo-mma reasons:
    * ``shift``: the stopping test divides by the *signed* initial objective
      (``change_relative_f = change_f / f_ref``), so a response that starts negative
      (e.g. the volume response below the reference fill) stops MMA at iteration 1.
      The shift keeps the objective positive.
    * ``scale``: MMA weighs constraint violation against the objective via a fixed
      penalty (c = 1000); a ×100-scaled response (volume in percent) leaves only ~10×
      authority for feasibility, and MMA will happily descend the objective while
      infeasible. Scaling the objective to O(1) restores the usual ~1000× feasibility
      priority. (E.g. shift=200, scale=0.01 turns the volume response into v + 1.)

    The pipeline inverts the map when reporting (``f_opt/scale - shift``).
    """

    def __init__(self, base_name: str, shift: float, scale: float = 1.0):
        super().__init__(f"Shifted_{base_name}")
        self.base_name = base_name
        self.out_name = f"{base_name}_shifted"
        self.shift = float(shift)
        self.scale = float(scale)
        self.input_grammar.update_from_names([base_name])
        self.output_grammar.update_from_names([self.out_name])
        self.default_inputs = {base_name: np.zeros(1)}
        if hasattr(self, "cache"):
            self.cache = None
        if hasattr(self, "cache_type"):
            self.cache_type = Discipline.CacheType.NONE

    def _run(self, input_data=None):
        if input_data is not None:
            self.local_data.update(input_data)
        val = np.asarray(self.local_data[self.base_name]).flatten()
        self.local_data[self.out_name] = (val + self.shift) * self.scale

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        self.jac = {self.out_name: {self.base_name: self.scale * np.eye(1)}}


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

    def __init__(self, roots, num_vars, shift=1.0, power=2.0, tau=1e-2, offset=0.0):
        super().__init__("DeflatedObjective")
        self.roots = [np.asarray(r, dtype=float).flatten() for r in roots]
        self.shift = float(shift)
        self.power = float(power)
        # tau floors the repeller at 1/tau (default 100), keeping it bounded.
        self.tau = float(tau)
        # offset == 0      -> pure deflation, objective (J)   * M(x)
        # offset == f_star -> tunneling,      objective (J-f*)* M(x)  (Levy-Montalvo
        #   style): below the incumbent f* the shifted objective goes negative, so the
        #   optimiser is driven to *better* basins while M(x) repels the known ones.
        self.offset = float(offset)
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
        """Return (M, dM/dx) for the deflation factor at design point *x*.

        Uses a **bounded** repeller ``term = 1 / (dist^power + tau)`` rather than the
        bare pole ``dist^-power``: the bare pole reaches ~1e12 at a root (with
        eps=1e-6, power=2), which makes the deflated/tunnelling objective and its
        gradient explode when the sharp MMA phase steps near a known minimum, and
        the subproblem bails out after a few iterations. The ``tau`` floor caps the
        repeller at ``1/tau`` and keeps the gradient finite and smooth (-> 0 at the
        root), so the optimiser is steered away from known minima without blowing up.
        """
        m = self.shift
        grad = np.zeros_like(x)
        p = self.power
        for r in self.roots:
            d = x - r
            dist2 = float(d @ d)
            g = dist2 ** (p / 2.0)                      # = ||x-r||^power
            den = g + self.tau
            m += 1.0 / den
            # d(1/den)/dx = -(1/den^2) * dg/dx ;  dg/dx = power * dist2^(p/2 - 1) * d
            grad += -(p * dist2 ** (p / 2.0 - 1.0) / den ** 2) * d if dist2 > 0 else np.zeros_like(x)
        return m, grad

    def _run(self, input_data=None):
        if input_data is not None:
            self.local_data.update(input_data)
        j = float(np.asarray(self.local_data["compliance"]).flatten()[0])
        x = np.asarray(self.local_data["x_vars"]).flatten()
        m, _ = self._factor_and_grad(x)
        self.local_data["compliance_deflated"] = np.array([(j - self.offset) * m])

    def _compute_jacobian(self, inputs=None, outputs=None, **kwargs):
        j = float(np.asarray(self.local_data["compliance"]).flatten()[0])
        x = np.asarray(self.local_data["x_vars"]).flatten()
        m, dm = self._factor_and_grad(x)
        # d((J-off)*M)/dJ = M ; d((J-off)*M)/dx = (J-off) * dM/dx
        j = j - self.offset
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

        if mode in ("Free", "2D_Free") and kwargs.get("init") == "grid":
            # Grid-fill initialisation: tile the domain with grid_nx x grid_ny
            # rectangles (squares when Lx/grid_nx == Ly/grid_ny), each at density
            # = volfrac, so the design starts as a coarse but *feasible, uniform*
            # block layout -- a far more informative guess than thin crossed bars.
            Lx = kwargs.get("Lx", 60.0)
            Ly = kwargs.get("Ly", 30.0)
            lb = kwargs.get("lb", None)
            ub = kwargs.get("ub", None)
            nc = n // 6
            gnx = kwargs.get("grid_nx") or _auto_grid(nc, Lx, Ly)[0]
            gny = kwargs.get("grid_ny") or _auto_grid(nc, Lx, Ly)[1]
            if gnx * gny != nc:
                raise ValueError(
                    f"grid_nx*grid_ny ({gnx}*{gny}) must equal num_components ({nc})")
            vf = kwargs.get("volfrac", 0.5)
            ii, jj = np.meshgrid(np.arange(gnx), np.arange(gny))   # (gny, gnx)
            Xc = ((ii + 0.5) * Lx / gnx).flatten()                 # row-major: j*gnx+i
            Yc = ((jj + 0.5) * Ly / gny).flatten()
            Lc = np.full(nc, Lx / gnx)        # rectangle width
            hc = np.full(nc, Ly / gny)        # rectangle height
            Tc = np.zeros(nc)                 # axis-aligned
            x = np.empty(n)
            x[0::6] = np.clip((Xc - lb[0::6]) / (ub[0::6] - lb[0::6]), 0.0, 1.0)
            x[1::6] = np.clip((Yc - lb[1::6]) / (ub[1::6] - lb[1::6]), 0.0, 1.0)
            x[2::6] = np.clip((Lc - lb[2::6]) / (ub[2::6] - lb[2::6]), 0.0, 1.0)
            x[3::6] = np.clip((hc - lb[3::6]) / (ub[3::6] - lb[3::6]), 0.0, 1.0)
            x[4::6] = np.clip((Tc - lb[4::6]) / (ub[4::6] - lb[4::6]), 0.0, 1.0)
            x[5::6] = vf
            return x

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

        if mode in ("3D_Box", "Box3D"):
            # Grid-fill: tile the domain with nc oriented boxes (axis-aligned start),
            # each sized to a lattice cell, Mc = volfrac. The optimiser then moves,
            # resizes, rotates and removes them.
            Lx = kwargs.get("Lx", 60.0); Ly = kwargs.get("Ly", 30.0); Lz = kwargs.get("Lz", 30.0)
            lb = kwargs.get("lb"); ub = kwargs.get("ub"); vf = kwargs.get("volfrac", 0.3)
            nc = n // 9
            c = int(np.ceil(nc ** (1.0 / 3.0)))
            xs = (np.arange(c) + 0.5) * Lx / c
            ys = (np.arange(c) + 0.5) * Ly / c
            zs = (np.arange(c) + 0.5) * Lz / c
            XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
            Xc = XX.ravel()[:nc]; Yc = YY.ravel()[:nc]; Zc = ZZ.ravel()[:nc]
            sx, sy, sz = Lx / c, Ly / c, Lz / c
            x = np.empty(n)
            x[0::9] = np.clip((Xc - lb[0::9]) / (ub[0::9] - lb[0::9]), 0.0, 1.0)
            x[1::9] = np.clip((Yc - lb[1::9]) / (ub[1::9] - lb[1::9]), 0.0, 1.0)
            x[2::9] = np.clip((Zc - lb[2::9]) / (ub[2::9] - lb[2::9]), 0.0, 1.0)
            x[3::9] = np.clip((sx - lb[3::9]) / (ub[3::9] - lb[3::9]), 0.0, 1.0)
            x[4::9] = np.clip((sy - lb[4::9]) / (ub[4::9] - lb[4::9]), 0.0, 1.0)
            x[5::9] = np.clip((sz - lb[5::9]) / (ub[5::9] - lb[5::9]), 0.0, 1.0)
            x[6::9] = np.clip((0.0 - lb[6::9]) / (ub[6::9] - lb[6::9]), 0.0, 1.0)  # theta
            x[7::9] = np.clip((0.0 - lb[7::9]) / (ub[7::9] - lb[7::9]), 0.0, 1.0)  # phi
            x[8::9] = vf
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

        # Truss mode: build the node lattice + radius-based bar connectivity once,
        # up-front, so the geometry discipline (KS aggregates over *bars*) and the
        # design space (2 vars/node + 2 vars/bar) both get the right sizes.
        mode = self.spec.formulation.mode
        truss_nodes = truss_bars = None
        num_components_eff = self.spec.formulation.num_components
        if mode in ("Truss", "2D_Truss"):
            from ggp.projection.truss_2d import build_ground_structure
            gnx = self.spec.formulation.grid_nx or 4
            gny = self.spec.formulation.grid_ny or 3
            truss_nodes, truss_bars = build_ground_structure(
                gnx, gny, Lx, Ly, self.spec.formulation.truss_radius
            )
            num_components_eff = len(truss_bars)
        elif mode in ("Truss3D", "3D_Truss"):
            from ggp.projection.truss_3d import build_ground_structure_3d
            gnx = self.spec.formulation.grid_nx or 3
            gny = self.spec.formulation.grid_ny or 2
            gnz = self.spec.formulation.grid_nz or 2
            truss_nodes, truss_bars = build_ground_structure_3d(
                gnx, gny, gnz, Lx, Ly, Lz if Lz is not None else 30.0,
                self.spec.formulation.truss_radius,
            )
            num_components_eff = len(truss_bars)

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
            "Ngp": self.spec.formulation.Ngp,
            "min_thickness": self.spec.formulation.min_thickness,
            "fix_mc": self.spec.formulation.fix_mc or None,
        }
        # Apply per-run sharpness overrides (continuation strategy).
        for _k in ("ka", "pp", "r_gp", "gammac", "gammav"):
            if _k in self.overrides and self.overrides[_k] is not None:
                geom_kwargs[_k] = self.overrides[_k]

        # Truss mode: hand the connectivity to the mapper via kwargs.
        if truss_nodes is not None:
            geom_kwargs["nodes"] = truss_nodes
            geom_kwargs["bars"] = truss_bars

        geom_discipline = GGPGeometryDiscipline(
            mesh=analysis.mesh,
            num_components=num_components_eff,
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
        # 'rect' (rectangle primitive) uses linear stiffness like 'GP', but keeps a
        # small Emin floor below because rectangle densities can be exactly 0
        # (the AMNA characteristic saturates to 0 outside the block) -> Emin=0 would
        # make the FE system singular in fully-void regions.
        p_penalty = 1.0 if method in ("GP", "rect", "box") else 3.0
        # Emin: the reference Free GP branch (model_updateM.py) computes E = rho*E0
        # with NO Emin floor (void E -> ~0 via the smooth-saturation residual).
        # GGP-Topo's SIMP form adds Emin, which spuriously raises void E and, because
        # the initial design is mostly void, shifts compliance ~1.5% and steers the
        # optimizer to a different (asymmetric) optimum. Match the reference: no Emin
        # floor for the Free GP case. NB: ALM modes also default to method="GP" but
        # their reference uses MNA *with* Emin (and Emin=0 makes the ALM FE singular
        # -> NaN), so the no-floor case is restricted to non-ALM modes.
        _is_alm = "ALM" in (self.spec.formulation.mode or "")
        # Truss ground structures leave large fully-void regions (bars are thin lines
        # in the domain), so a zero Emin makes K singular; keep a small floor for them.
        _is_truss = "Truss" in (self.spec.formulation.mode or "")
        e_min = 0.0 if (method == "GP" and not _is_alm and not _is_truss) else 1e-6

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

        if mode in ("Truss", "2D_Truss", "Truss3D", "3D_Truss"):
            # dim vars per node + 2 vars per bar (h, Mc); the mapper's bounds array
            # already has the right total length (2N+2B in 2D, 3N+2B in 3D).
            num_vars = len(geom_discipline.lb)
        elif mode in ["ALM", "2D_ALM"] and self.spec.formulation.num_layers:
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

        # Top-bottom symmetry: optimiser controls only the free half of the
        # components; a _SymmetryExpander appends their mirror images.
        _vpc = geom_discipline.mapper.num_vars_per_component()
        sym = (self.spec.formulation.symmetry == "y" and mode in ("Free", "2D_Free"))
        sym_expander = None
        if sym:
            num_free = self.spec.formulation.num_components // 2
            num_vars = _vpc * num_free
            sym_expander = _SymmetryExpander(num_free, vpc=_vpc)

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
                "Ly": (Ly / 2.0) if sym else Ly,   # free components live in the bottom half
                "lb": geom_discipline.lb[:num_vars] if sym else geom_discipline.lb,
                "ub": geom_discipline.ub[:num_vars] if sym else geom_discipline.ub,
                "init": self.spec.formulation.init,
                "grid_nx": self.spec.formulation.grid_nx,
                "grid_ny": self.spec.formulation.grid_ny,
                "volfrac": self.spec.volfrac,
            }
            # Pass non-design origin for L-shape aware initialization
            for geom in self.spec.geometries:
                if geom.role == "non_design" and geom.type == "box":
                    _init_kwargs["non_design_origin"] = geom.params.get("origin", None)
                    break
        elif mode in ("3D_Free", "3D_Box", "Box3D"):
            _init_kwargs = {
                "Lx": Lx,
                "Ly": Ly,
                "Lz": Lz if Lz is not None else 30.0,
                "lb": geom_discipline.lb,
                "ub": geom_discipline.ub,
                "volfrac": self.spec.volfrac,
            }
        if self.x0 is not None:
            if self.x0.shape[0] != num_vars:
                raise ValueError(
                    f"x0 has length {self.x0.shape[0]} but the problem expects "
                    f"{num_vars} design variables."
                )
            x_init = np.clip(self.x0, 0.0, 1.0)
        elif mode in ("Truss", "2D_Truss", "Truss3D", "3D_Truss"):
            # Start: every node on its lattice site, every ground-structure bar
            # present with a modest thickness and Mc = volfrac (the optimizer then
            # moves nodes, fattens load-paths and removes idle bars via h/Mc).
            N, B = len(truss_nodes), len(truss_bars)
            nd = truss_nodes.shape[1]                    # 2 (planar) or 3 (spatial)
            lb, ub = geom_discipline.lb, geom_discipline.ub
            x_un = np.empty(num_vars)
            x_un[: nd * N] = truss_nodes.ravel()
            x_un[nd * N : nd * N + B] = max(self.spec.formulation.min_thickness or 1.0, 2.0)
            x_un[nd * N + B :] = self.spec.volfrac
            x_init = np.clip((x_un - lb) / (ub - lb), 0.0, 1.0)
        else:
            x_init = self._make_init(mode, num_vars, **_init_kwargs)
        # Under symmetry the design variable is the free half ("x_free"); the
        # expander produces the full "x_vars" the geometry discipline consumes.
        design_var = "x_free" if sym else "x_vars"
        design_space.add_variable(
            design_var, size=num_vars, lower_bound=0.0, upper_bound=1.0, value=x_init
        )

        # 6. MDA Chain
        chain_disciplines = ([sym_expander] if sym else []) + [geom_discipline, phys_discipline]
        chain = MDAChain(chain_disciplines)
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

        # A response's discipline is built whenever it's needed as a CONSTRAINT or as the
        # OBJECTIVE; when only the latter, its params come from spec.objective.params
        # (mirrors ConstraintSpec.params) instead of a ConstraintSpec.
        needed_responses = {c.name for c in self.spec.constraints} | {self.spec.objective.name}

        # 7a2. Displacement constraint/objective: bound or optimize |u| at a tracked DOF.
        disp_spec = next((c for c in self.spec.constraints if c.name == "displacement"), None)
        if "displacement" in needed_responses:
            dp = disp_spec.params if disp_spec is not None else self.spec.objective.params
            V_u = analysis.function_spaces["u"]
            dof_coords = V_u.tabulate_dof_coordinates()
            axis = int(dp.get("axis", 1))
            u_max = float(dp.get("u_max", 1.0))
            sub_dofs_axis = np.asarray(V_u.sub(axis).dofmap().dofs())
            pt = dp.get("point", None)
            if pt is None:                                  # default: the loaded (mid-right) point
                pt = [Lx, Ly / 2.0] + ([(Lz or 30.0) / 2.0] if analysis.dim == 3 else [])
            target = np.asarray(pt[:analysis.dim], dtype=float)
            nearest = int(sub_dofs_axis[np.argmin(
                np.linalg.norm(dof_coords[sub_dofs_axis] - target, axis=1))])
            ell = np.zeros(V_u.dim()); ell[nearest] = 1.0
            disp_disc = _DisplacementConstraintDiscipline(
                phys_discipline, ell, u_max,
                num_elements=phys_discipline.num_elements, volfrac=self.spec.volfrac,
            )
            extra_disciplines.append(disp_disc)

        # 7a3. Stress constraint/objective: aggregated von Mises (adjoint via JAX kernel).
        stress_spec = next((c for c in self.spec.constraints if c.name == "stress"), None)
        if "stress" in needed_responses:
            if analysis.se_ref is None:
                raise ValueError("stress constraint/objective requires se_ref from the discretiser")
            sp = stress_spec.params if stress_spec is not None else self.spec.objective.params
            stress_disc = _StressConstraintDiscipline(
                phys_discipline, analysis.se_ref,
                sigma_lim=float(sp.get("sigma_limit", 1.0)),
                num_elements=phys_discipline.num_elements, dim=analysis.dim,
                q=float(sp.get("q", 0.5)), P=float(sp.get("P", sp.get("ka", 8.0))),
                kind=sp.get("aggregation", "verbart"), volfrac=self.spec.volfrac,
                adaptive=bool(sp.get("adaptive", False)),
                alpha=float(sp.get("alpha", 0.5)),
                rho_solid=float(sp.get("rho_solid", 0.5)),
                max_correction=float(sp.get("max_correction", 4.0)),
                active_mask=_stress_exclusion_mask(
                    analysis, geom_discipline.eval_coords,
                    float(sp.get("exclude_radius", 0.0))),
            )
            extra_disciplines.append(stress_disc)

        # 7b. Deflation: repel known minima by deflating the objective. Deflation wraps the
        # "compliance" discipline output specifically, so it only composes with a compliance
        # objective; other objectives would need _DeflatedObjectiveDiscipline generalized.
        objective_name = self.spec.objective.name
        deflation_active = bool(self.deflation and self.deflation.get("roots"))
        if deflation_active:
            if objective_name != "compliance":
                raise ValueError(
                    "Deflation currently only supports a 'compliance' objective "
                    f"(got '{objective_name}')."
                )
            defl_disc = _DeflatedObjectiveDiscipline(
                roots=self.deflation["roots"],
                num_vars=num_vars,
                shift=self.deflation.get("shift", 1.0),
                power=self.deflation.get("power", 2.0),
                offset=self.deflation.get("offset", 0.0),
            )
            extra_disciplines.append(defl_disc)
            objective_name = "compliance_deflated"

        # 7c. Optional affine objective rescale (objective.params.shift/scale): keeps a
        # negative-valued response positive for gemseo-mma's relative-change stopping
        # test, and normalizes a large-scaled response (volume in percent) to O(1) so
        # MMA's fixed constraint penalty retains its usual feasibility authority.
        obj_shift = float(self.spec.objective.params.get("shift", 0.0))
        obj_scale = float(self.spec.objective.params.get("scale", 1.0))
        if (obj_shift != 0.0 or obj_scale != 1.0) and not deflation_active:
            shift_disc = _ShiftedObjectiveDiscipline(objective_name, obj_shift, obj_scale)
            extra_disciplines.append(shift_disc)
            objective_name = shift_disc.out_name

        # 8. Scenario
        all_disciplines = [chain] + extra_disciplines
        scenario = create_scenario(
            all_disciplines,
            objective_name=objective_name,
            design_space=design_space,
            maximize_objective=(self.spec.objective.sense == "maximize"),
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
            elif c.name == "displacement":
                scenario.add_constraint(
                    "displacement", constraint_type="ineq", positive=False, value=0.0
                )
            elif c.name == "stress":
                scenario.add_constraint(
                    "stress", constraint_type="ineq", positive=False, value=0.0
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
        # Under symmetry x_opt is the free half -> expand to the full geometry vector.
        x_opt_full = sym_expander.expand(x_opt) if sym else x_opt
        try:
            geom_discipline.execute({"x_vars": x_opt_full})
            density_field = np.asarray(geom_discipline.local_data["rho_E"]).flatten().copy()
            if analysis.empty_elements:
                density_field[analysis.empty_elements] = 0.0
        except Exception:
            density_field = None

        n_iter = len(scenario.formulation.optimization_problem.database)

        # Under deflation the scenario's objective is the *deflated* value, so recompute
        # the true "compliance" at the optimum from the geom+phys chain. Otherwise
        # opt_result.f_opt is already the raw, correctly-signed value of whatever
        # response is the objective (GEMSEO un-negates internally for maximize).
        if deflation_active:
            try:
                chain.execute({design_var: x_opt})
                objective_value = float(
                    np.asarray(chain.local_data["compliance"]).flatten()[0]
                )
            except Exception:
                objective_value = float("nan")
        else:
            f_opt = opt_result.f_opt
            objective_value = float(f_opt) if f_opt is not None else float("nan")
            # report the raw response value (invert the affine objective rescale)
            if obj_scale != 1.0:
                objective_value = objective_value / obj_scale
            objective_value -= obj_shift

        result = OptimisationResult(
            problem_name="optimization_run",
            algorithm=self.spec.solver.algorithm,
            status=opt_result.status,
            iterations=n_iter,
            objective_value=objective_value,
            objective_name=self.spec.objective.name,
            max_constraint_violation=0.0,
            design_variables=x_opt,
            history={},
            execution_time_s=time.time() - start_time,
            density_field=density_field,
            eval_coords=geom_discipline.eval_coords.copy(),
            dim=domain.dim,
        )

        return result
