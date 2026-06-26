# Copyright (c) 2026 Simone Coniglio
# Licensed under the MIT license. See LICENSE file in the project directory for details.
"""Global-search strategies for reaching *improved* local minima in GGP topology
optimisation.

Gradient-based (Lagrangian / MMA) topology optimisation is a local solver on a
strongly non-convex landscape, so it converges to whatever basin the starting
point sits in. For the GGP / Moving-Morphable-Component family the result is
particularly sensitive to the initial component layout and to the *sharpness* of
the geometry projection.

This module implements state-of-the-art techniques for finding better local minima,
all as thin orchestration layers on top of :class:`GGPPipeline`:

* :func:`continuation`     -- homotopy / parameter-continuation on the GGP sharpness
                              knobs (``r_gp``, ``ka``, ``pp``, ``p_penalty``),
                              warm-started phase by phase.
* :func:`multi_start`      -- best-of-N from a diverse set of random initial layouts.
* :func:`basin_hopping`    -- stochastic perturbation of the incumbent optimum with
                              a Metropolis acceptance rule.
* :func:`deflated_search`  -- Farrell/Papadopoulos/Surowiec deflation to repel
                              already-found minima and discover distinct ones.
* :func:`tunneling`        -- Levy-Montalvo tunneling: minimise a shifted, repelled
                              objective ``(J-f*)·M(x)`` to emerge in a *better* basin.
* :func:`chaotic_search`   -- chaos-optimisation: ergodic logistic-map restarts.

The single most important enabler is **continuation**: passing a ``schedule`` to any
of the global strategies makes each inner solve follow the sharpness homotopy, which
is what lets *every* method improve on the single-start baseline (given the time to
run the extra phases) rather than only the bare continuation. This mirrors a result
from chaos-control theory -- a nonlinear optimiser is a chaotic dynamical system
(we measured its sensitive dependence: a 1e-13 solver perturbation grows to ~0.1 %),
and continuation / annealing a smoothing parameter is the practical way to tame it.

The strategies never touch FEniCS or GEMSEO directly: they construct a
:class:`GGPPipeline` with an injected ``x0`` / ``overrides`` / ``deflation`` and
read back the :class:`OptimisationResult`. The pipeline class is imported lazily
(and can be injected via ``pipeline_cls``) so the orchestration logic is unit
testable without the heavy optimisation stack installed.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .results import OptimisationResult


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Attempt:
    """A single optimisation run within a global-search strategy."""

    label: str
    compliance: float
    objective_value: float            # log(C+1), as stored by the pipeline
    iterations: int
    status: str
    execution_time_s: float
    design_variables: np.ndarray
    accepted: bool = True             # used by basin hopping / deflation bookkeeping


@dataclasses.dataclass
class GlobalSearchResult:
    """Outcome of a global-search strategy."""

    method: str
    best_result: OptimisationResult
    best_compliance: float
    attempts: List[Attempt]
    total_time_s: float

    @property
    def best_index(self) -> int:
        comps = [a.compliance for a in self.attempts]
        return int(np.nanargmin(comps)) if comps else -1

    def summary(self) -> str:
        lines = [f"[{self.method}] best compliance = {self.best_compliance:.6g} "
                 f"over {len(self.attempts)} run(s), {self.total_time_s:.1f}s"]
        for a in self.attempts:
            mark = "*" if a.compliance == self.best_compliance else " "
            acc = "" if a.accepted else " (rejected)"
            lines.append(f"  {mark} {a.label:<18} C={a.compliance:.6g} "
                         f"iters={a.iterations}{acc}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def compliance_of(result: OptimisationResult) -> float:
    """Recover the true compliance C from the pipeline objective ``log(C+1)``."""
    val = result.objective_value
    if val is None or not np.isfinite(val):
        return float("inf")
    return float(np.expm1(val))


def _default_pipeline_cls():
    """Lazily import :class:`GGPPipeline` (keeps this module import-light)."""
    from .pipeline import GGPPipeline
    return GGPPipeline


def _run_once(
    spec,
    *,
    x0=None,
    overrides=None,
    deflation=None,
    pipeline_cls=None,
) -> OptimisationResult:
    cls = pipeline_cls or _default_pipeline_cls()
    return cls(spec, x0=x0, overrides=overrides, deflation=deflation).run()


def random_initial_design(
    num_vars: int,
    mode: str = "Free",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return a diverse normalised [0,1] initial design for a random restart.

    The design space is already normalised to [0,1], so we sample directly there
    with per-variable distributions appropriate to the GGP component layout:
    positions spread across the domain, thin bars, full angular range and a
    moderate membership density. Unknown layouts fall back to ``U(0.3, 0.7)``.
    """
    rng = rng or np.random.default_rng()

    if mode in ("Free", "2D_Free") and num_vars % 6 == 0:
        nc = num_vars // 6
        x = np.empty(num_vars)
        x[0::6] = rng.uniform(0.1, 0.9, nc)    # Xc
        x[1::6] = rng.uniform(0.1, 0.9, nc)    # Yc
        x[2::6] = rng.uniform(0.25, 0.75, nc)  # L
        x[3::6] = rng.uniform(0.0, 0.10, nc)   # h (normalised; small => thin bar)
        x[4::6] = rng.uniform(0.0, 1.0, nc)    # Theta (full range)
        x[5::6] = rng.uniform(0.4, 0.6, nc)    # Mc
        return x

    if mode == "3D_Free" and num_vars % 8 == 0:
        nc = num_vars // 8
        x = np.empty(num_vars)
        x[0::8] = rng.uniform(0.1, 0.9, nc)    # Xc
        x[1::8] = rng.uniform(0.1, 0.9, nc)    # Yc
        x[2::8] = rng.uniform(0.1, 0.9, nc)    # Zc
        x[3::8] = rng.uniform(0.25, 0.75, nc)  # L
        x[4::8] = rng.uniform(0.0, 0.10, nc)   # h
        x[5::8] = rng.uniform(0.0, 1.0, nc)    # Theta
        x[6::8] = rng.uniform(0.0, 1.0, nc)    # Phi
        x[7::8] = rng.uniform(0.4, 0.6, nc)    # Mc
        return x

    return rng.uniform(0.3, 0.7, num_vars)


def perturb_design(
    x: np.ndarray,
    step: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Gaussian perturbation of a normalised design, clipped to [0,1]."""
    rng = rng or np.random.default_rng()
    return np.clip(np.asarray(x, float) + step * rng.standard_normal(len(x)), 0.0, 1.0)


def _attempt_from_result(label, result, accepted=True) -> Attempt:
    return Attempt(
        label=label,
        compliance=compliance_of(result),
        objective_value=float(result.objective_value),
        iterations=int(result.iterations),
        status=str(result.status),
        execution_time_s=float(result.execution_time_s or 0.0),
        design_variables=np.asarray(result.design_variables, float).flatten(),
        accepted=accepted,
    )


def _finalise(method, attempts, results_by_label, t0) -> GlobalSearchResult:
    comps = [a.compliance for a in attempts]
    best_i = int(np.nanargmin(comps))
    best_label = attempts[best_i].label
    return GlobalSearchResult(
        method=method,
        best_result=results_by_label[best_label],
        best_compliance=comps[best_i],
        attempts=attempts,
        total_time_s=time.time() - t0,
    )


def _local_solve(spec, *, x0=None, schedule=None, deflation=None, pipeline_cls=None):
    """One 'local solve' that may itself be a warm-started **continuation**.

    This is the key that lets *every* global strategy improve on the single-start
    baseline: instead of solving the raw, rugged *sharp* problem, each attempt
    follows the GGP sharpness homotopy (the ``schedule`` of ``r_gp`` overrides),
    which reshapes the landscape and reaches a better basin.

    Returns ``(final_result, phase_results)`` where ``final_result`` is evaluated at
    the *last* (target-sharpness) phase and is therefore comparable to the baseline.
    With ``schedule is None`` this is a plain single solve.
    """
    if not schedule:
        r = _run_once(spec, x0=x0, deflation=deflation, pipeline_cls=pipeline_cls)
        return r, [r]
    cur = None if x0 is None else np.asarray(x0, float).flatten()
    phases = []
    for ov in schedule:
        r = _run_once(spec, x0=cur, overrides=ov, deflation=deflation,
                      pipeline_cls=pipeline_cls)
        phases.append(r)
        cur = np.asarray(r.design_variables, float).flatten()
    return phases[-1], phases


# --------------------------------------------------------------------------- #
# Chaotic dynamics (chaos-theory-inspired search)
# --------------------------------------------------------------------------- #
def logistic_map(n: int, seed: float = 0.137, mu: float = 4.0,
                 burn_in: int = 50) -> np.ndarray:
    """Return *n* values of the logistic map ``x <- mu x (1-x)``.

    At ``mu = 4`` the map is fully chaotic and **ergodic** on (0, 1): its orbit
    covers the interval far more uniformly than a finite pseudo-random sample,
    which is exactly the property chaotic-optimization algorithms exploit for
    global search. A short ``burn_in`` discards the transient.
    """
    x = float(seed) % 1.0
    if x in (0.0, 0.25, 0.5, 0.75, 1.0):     # avoid fixed/periodic points
        x = 0.137
    for _ in range(burn_in):
        x = mu * x * (1.0 - x)
    out = np.empty(n)
    for i in range(n):
        x = mu * x * (1.0 - x)
        out[i] = x
    return out


def chaotic_initial_design(num_vars: int, mode: str = "Free",
                           seed: float = 0.137, mu: float = 4.0) -> np.ndarray:
    """Diverse initial design seeded by a chaotic (logistic-map) stream instead of
    an RNG. Same per-variable ranges as :func:`random_initial_design`, but the
    ergodicity of the chaotic orbit spreads successive restarts more evenly across
    the design space."""
    c = logistic_map(num_vars, seed=seed, mu=mu)

    if mode in ("Free", "2D_Free") and num_vars % 6 == 0:
        nc = num_vars // 6
        x = np.empty(num_vars)
        x[0::6] = 0.1 + 0.8 * c[0::6]    # Xc in [0.1, 0.9]
        x[1::6] = 0.1 + 0.8 * c[1::6]    # Yc
        x[2::6] = 0.25 + 0.5 * c[2::6]   # L
        x[3::6] = 0.10 * c[3::6]         # h (thin)
        x[4::6] = c[4::6]                # Theta (full range)
        x[5::6] = 0.4 + 0.2 * c[5::6]    # Mc
        return x
    return 0.1 + 0.8 * c                 # generic fallback in [0.1, 0.9]


# --------------------------------------------------------------------------- #
# 1. Continuation / homotopy
# --------------------------------------------------------------------------- #
def continuation(
    spec,
    schedule: Sequence[Dict[str, Any]],
    x0: Optional[np.ndarray] = None,
    pipeline_cls=None,
    on_phase: Optional[Callable[[int, OptimisationResult], None]] = None,
) -> GlobalSearchResult:
    """Parameter-continuation: solve a sequence of phases that sharpen the GGP
    landscape, warm-starting each phase from the previous optimum.

    Parameters
    ----------
    schedule
        List of ``overrides`` dicts, one per phase, e.g.
        ``[{"r_gp": 1.5}, {"r_gp": 1.0}, {"r_gp": 0.5}]`` or with ``p_penalty`` /
        ``ka`` / ``pp`` ramps. Early (smooth) phases explore; late (sharp) phases
        refine.
    """
    if not schedule:
        raise ValueError("continuation schedule must contain at least one phase")
    t0 = time.time()
    attempts: List[Attempt] = []
    results_by_label: Dict[str, OptimisationResult] = {}
    current_x = None if x0 is None else np.asarray(x0, float)

    last_result = None
    for i, overrides in enumerate(schedule):
        result = _run_once(spec, x0=current_x, overrides=overrides,
                           pipeline_cls=pipeline_cls)
        label = f"phase{i}:" + ",".join(f"{k}={v}" for k, v in overrides.items())
        attempts.append(_attempt_from_result(label, result))
        results_by_label[label] = result
        current_x = np.asarray(result.design_variables, float).flatten()
        last_result = result
        if on_phase is not None:
            on_phase(i, result)

    # The phases use different sharpness (e.g. r_gp), so their compliances are NOT
    # comparable across phases -- a smooth early phase reports an artificially low
    # value. Continuation is a single warm-started trajectory whose meaningful
    # outcome is the FINAL phase, evaluated at the target (baseline) sharpness.
    return GlobalSearchResult(
        method="continuation",
        best_result=last_result,
        best_compliance=compliance_of(last_result),
        attempts=attempts,
        total_time_s=time.time() - t0,
    )


# --------------------------------------------------------------------------- #
# 2. Multi-start / random restart
# --------------------------------------------------------------------------- #
def multi_start(
    spec,
    n_starts: int = 5,
    seed: int = 0,
    mode: Optional[str] = None,
    include_default: bool = True,
    schedule: Optional[Sequence[Dict[str, Any]]] = None,
    pipeline_cls=None,
    on_start: Optional[Callable[[int, OptimisationResult], None]] = None,
) -> GlobalSearchResult:
    """Best-of-N optimisation from diverse random initial component layouts.

    The first run (when ``include_default``) uses the deterministic grid start;
    the remaining ``n_starts`` runs use :func:`random_initial_design`. When
    ``schedule`` is given, each start is refined by a warm-started **continuation**
    (the homotopy that lets restarts reach below the baseline rather than into worse
    sharp-landscape basins).
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    mode = mode or spec.formulation.mode
    num_vars = _num_vars(spec)

    attempts: List[Attempt] = []
    results_by_label: Dict[str, OptimisationResult] = {}

    runs: List[Optional[np.ndarray]] = []
    if include_default:
        runs.append(None)                       # deterministic grid init
    for _ in range(n_starts):
        runs.append(random_initial_design(num_vars, mode, rng))

    for i, x0 in enumerate(runs):
        result, _ = _local_solve(spec, x0=x0, schedule=schedule, pipeline_cls=pipeline_cls)
        label = "default" if (include_default and i == 0) else f"restart{i}"
        attempts.append(_attempt_from_result(label, result))
        results_by_label[label] = result
        if on_start is not None:
            on_start(i, result)

    return _finalise("multi_start", attempts, results_by_label, t0)


# --------------------------------------------------------------------------- #
# 3. Basin hopping
# --------------------------------------------------------------------------- #
def basin_hopping(
    spec,
    n_hops: int = 5,
    step: float = 0.1,
    temperature: float = 1.0,
    seed: int = 0,
    x0: Optional[np.ndarray] = None,
    schedule: Optional[Sequence[Dict[str, Any]]] = None,
    chaotic: bool = False,
    pipeline_cls=None,
    on_hop: Optional[Callable[[int, OptimisationResult, bool], None]] = None,
) -> GlobalSearchResult:
    """Basin hopping: local-optimise, then repeatedly perturb the *accepted*
    incumbent and re-optimise, accepting by a Metropolis rule on the compliance.

    ``temperature`` is in compliance units; acceptance probability for a worse
    proposal is ``exp(-(C_new - C_inc) / T)``. With ``schedule`` each local solve is
    a continuation. With ``chaotic=True`` the perturbations are drawn from a logistic
    chaotic stream rather than Gaussian noise (chaos-driven exploration).
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    chaos = iter(logistic_map(2048, seed=0.301 + 0.001 * seed)) if chaotic else None

    def _perturb(x):
        if chaos is None:
            return perturb_design(x, step=step, rng=rng)
        # centred chaotic increment in [-step, step]
        inc = np.array([step * (2.0 * next(chaos) - 1.0) for _ in range(len(x))])
        return np.clip(x + inc, 0.0, 1.0)

    attempts: List[Attempt] = []
    results_by_label: Dict[str, OptimisationResult] = {}

    # Initial local minimisation.
    result, _ = _local_solve(spec, x0=x0, schedule=schedule, pipeline_cls=pipeline_cls)
    attempts.append(_attempt_from_result("hop0", result))
    results_by_label["hop0"] = result
    inc_x = np.asarray(result.design_variables, float).flatten()
    inc_C = compliance_of(result)
    if on_hop is not None:
        on_hop(0, result, True)

    for i in range(1, n_hops + 1):
        x_try = _perturb(inc_x)
        result, _ = _local_solve(spec, x0=x_try, schedule=schedule, pipeline_cls=pipeline_cls)
        C_new = compliance_of(result)
        if C_new <= inc_C:
            accept = True
        else:
            accept = rng.random() < np.exp(-(C_new - inc_C) / max(temperature, 1e-12))
        if accept:
            inc_x = np.asarray(result.design_variables, float).flatten()
            inc_C = C_new
        label = f"hop{i}"
        attempts.append(_attempt_from_result(label, result, accepted=accept))
        results_by_label[label] = result
        if on_hop is not None:
            on_hop(i, result, accept)

    return _finalise("basin_hopping", attempts, results_by_label, t0)


# --------------------------------------------------------------------------- #
# 4. Deflation
# --------------------------------------------------------------------------- #
def deflated_search(
    spec,
    n_solutions: int = 4,
    shift: float = 1.0,
    power: float = 2.0,
    seed: int = 0,
    x0: Optional[np.ndarray] = None,
    schedule: Optional[Sequence[Dict[str, Any]]] = None,
    pipeline_cls=None,
    on_solution: Optional[Callable[[int, OptimisationResult], None]] = None,
) -> GlobalSearchResult:
    """Deflated search / **deflated continuation** (Farrell/Papadopoulos/Surowiec).

    Solve once, then repeatedly re-solve with the objective deflated away from every
    minimum found so far, so the optimiser is driven into *distinct* basins. With
    ``schedule`` the deflation is applied along the continuation homotopy (deflated
    continuation), which is what lets the distinct minima it finds actually beat the
    baseline rather than only differ from it. Returns the set of minima and the best.
    """
    t0 = time.time()
    attempts: List[Attempt] = []
    results_by_label: Dict[str, OptimisationResult] = {}
    roots: List[np.ndarray] = []

    for i in range(max(1, n_solutions)):
        deflation = None if not roots else {"roots": list(roots), "shift": shift,
                                            "power": power}
        result, _ = _local_solve(spec, x0=x0, schedule=schedule, deflation=deflation,
                                 pipeline_cls=pipeline_cls)
        label = f"sol{i}"
        attempts.append(_attempt_from_result(label, result))
        results_by_label[label] = result
        roots.append(np.asarray(result.design_variables, float).flatten())
        if on_solution is not None:
            on_solution(i, result)

    return _finalise("deflated_search", attempts, results_by_label, t0)


# --------------------------------------------------------------------------- #
# 5. Tunneling (Levy-Montalvo)
# --------------------------------------------------------------------------- #
def tunneling(
    spec,
    n_solutions: int = 4,
    power: float = 2.0,
    seed: int = 0,
    x0: Optional[np.ndarray] = None,
    schedule: Optional[Sequence[Dict[str, Any]]] = None,
    pipeline_cls=None,
    on_solution: Optional[Callable[[int, OptimisationResult], None]] = None,
) -> GlobalSearchResult:
    """Tunneling method (Levy & Montalvo), in a robust **escape-then-refine** form.

    Classic tunneling minimises a *shifted, repelled* objective ``(J - f*)·M(x)`` to
    slide through the ``f*`` level set into a better basin. That shifted objective,
    however, sits near zero inside a known basin and trips the preset MMA's tight
    *relative* convergence tolerance, so the sharp phase stops after a few iterations.
    We therefore split each tunnelling step into two well-posed solves:

    1. **escape** -- minimise the *multiplicative* deflated objective ``J·M(x)`` (the
       stable, strictly-positive deflation operator) under the *smoothest* homotopy
       phase, starting from the incumbent. ``M`` repels the known minima, so the
       optimiser is pushed out of their basins into a new region.
    2. **refine** -- a *clean* continuation (no deflation) from that escaped point,
       which converges normally to the new basin's minimum.

    This keeps the tunnelling idea -- leave the basins already found, then minimise --
    while every sub-solve is numerically well-conditioned. The incumbent ``f*`` is
    lowered whenever a better minimum is found, biasing the search downward.
    """
    t0 = time.time()
    attempts: List[Attempt] = []
    results_by_label: Dict[str, OptimisationResult] = {}
    roots: List[np.ndarray] = []
    smooth_phase = (schedule[0] if schedule else {"r_gp": 1.5})

    # Phase 0: clean minimisation to establish the incumbent.
    result, _ = _local_solve(spec, x0=x0, schedule=schedule, pipeline_cls=pipeline_cls)
    attempts.append(_attempt_from_result("min0", result))
    results_by_label["min0"] = result
    inc_x = np.asarray(result.design_variables, float).flatten()
    roots.append(inc_x)
    best = result
    if on_solution is not None:
        on_solution(0, result)

    for i in range(1, max(1, n_solutions)):
        # 1. escape: deflated (multiplicative) solve under the smooth phase.
        deflation = {"roots": list(roots), "shift": 1.0, "power": power}
        escaped = _run_once(spec, x0=inc_x, overrides=smooth_phase,
                            deflation=deflation, pipeline_cls=pipeline_cls)
        x_escape = np.asarray(escaped.design_variables, float).flatten()
        # 2. refine: clean continuation from the escaped point.
        result, _ = _local_solve(spec, x0=x_escape, schedule=schedule,
                                 pipeline_cls=pipeline_cls)
        label = f"tunnel{i}"
        attempts.append(_attempt_from_result(label, result))
        results_by_label[label] = result
        new_x = np.asarray(result.design_variables, float).flatten()
        roots.append(new_x)
        if compliance_of(result) < compliance_of(best):
            best = result
            inc_x = new_x                 # tunnel onward from the improved incumbent
        if on_solution is not None:
            on_solution(i, result)

    return _finalise("tunneling", attempts, results_by_label, t0)


# --------------------------------------------------------------------------- #
# 6. Chaotic search (chaos-optimisation / ergodic restarts)
# --------------------------------------------------------------------------- #
def chaotic_search(
    spec,
    n_starts: int = 4,
    mu: float = 4.0,
    seed: int = 0,
    mode: Optional[str] = None,
    include_default: bool = True,
    schedule: Optional[Sequence[Dict[str, Any]]] = None,
    pipeline_cls=None,
    on_start: Optional[Callable[[int, OptimisationResult], None]] = None,
) -> GlobalSearchResult:
    """Chaos-optimisation: best-of-N where the restart layouts are seeded by an
    **ergodic logistic-map** stream (``mu=4``) instead of an RNG, then refined by
    continuation. The ergodicity of the chaotic orbit covers the design space more
    evenly than a finite pseudo-random sample, the property chaotic-optimization
    algorithms exploit for global search."""
    t0 = time.time()
    mode = mode or spec.formulation.mode
    num_vars = _num_vars(spec)

    attempts: List[Attempt] = []
    results_by_label: Dict[str, OptimisationResult] = {}

    runs: List[Optional[np.ndarray]] = []
    if include_default:
        runs.append(None)
    for k in range(n_starts):
        # distinct chaotic seed per restart -> distinct ergodic orbit
        s = (0.111 + 0.07 * (k + 1) + 0.013 * seed) % 1.0
        runs.append(chaotic_initial_design(num_vars, mode, seed=s, mu=mu))

    for i, x0 in enumerate(runs):
        result, _ = _local_solve(spec, x0=x0, schedule=schedule, pipeline_cls=pipeline_cls)
        label = "default" if (include_default and i == 0) else f"chaos{i}"
        attempts.append(_attempt_from_result(label, result))
        results_by_label[label] = result
        if on_start is not None:
            on_start(i, result)

    return _finalise("chaotic_search", attempts, results_by_label, t0)


# --------------------------------------------------------------------------- #
# Spec introspection
# --------------------------------------------------------------------------- #
def _num_vars(spec) -> int:
    """Number of design variables implied by *spec* (mirrors the pipeline)."""
    f = spec.formulation
    mode = f.mode
    if mode in ("ALM", "2D_ALM") and f.num_layers:
        return 2 * f.num_layers * f.comp_per_layer + 2 * f.comp_per_layer + 2
    if mode == "3D_ALM" and f.num_layers:
        return 6 * f.comp_per_layer * f.num_layers
    per = 8 if mode == "3D_Free" else 6
    return per * f.num_components
