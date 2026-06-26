# Reaching improved local minima in GGP topology optimization

> A state-of-the-art review of techniques for escaping poor local minima in
> Lagrangian / gradient-based topology optimization, and their implementation and
> benchmarking inside **GGP-Topo** on the Short Cantilever 2D (SC 2D) use case.

## 1. The problem

Density- and component-based topology optimization (TO) minimizes a compliance-type
objective subject to a volume constraint. The discretized problem is **strongly
non-convex**: the material-interpolation penalization (SIMP `p`, GGP membership
`Mc`) and the geometry projection create many stationary points. Gradient-based
"Lagrangian" solvers — MMA, SLP, CONLIN, the optimality-criteria update, augmented
Lagrangian — are all **local**: they descend into whatever basin contains the
starting point and certify only a KKT point, never global optimality.

For the **Generalized Geometry Projection (GGP)** family — and its relatives MMC
(Moving Morphable Components), MMV, and geometry projection of bars/plates — this is
*especially* acute:

* **Initial-layout sensitivity.** The optimizer rearranges a fixed number of
  explicit components (bars). The final topology is largely determined by where the
  bars start and how they are allowed to merge/split, so two reasonable initial
  layouts routinely converge to visibly different structures with different
  compliances (Guo et al. 2014; Zhang et al. 2016).
* **Landscape sharpness.** The KS aggregation parameter `ka`, the smooth-saturation
  steepness `pp`, the sampling-window radius `r_gp`, and the SIMP exponent `p`
  control how rugged the landscape is. Sharp settings give crisp, manufacturable
  designs but a rugged landscape full of local traps; smooth settings give a benign
  landscape but blurry, intermediate-density designs.

In GGP-Topo this manifests concretely: `GGPPipeline.run()` performs **one
deterministic start** with **fixed** sharpness (`ka=10`, `pp=100`, `r_gp` from the
spec, `p_penalty` fixed by the formulation method) and no restart/continuation/
global-search layer. This document reviews how the literature attacks this, then
implements and benchmarks the most promising techniques.

## 2. State of the art

### 2.1 Continuation / homotopy (most widely used)
Solve a sequence of related, progressively harder problems, warm-starting each from
the previous optimum, so the optimizer is guided through a benign landscape before
the rugged one is imposed:

* **SIMP penalization continuation** — ramp `p` from 1 (convex-ish, gray) to 3–5
  (crisp). Standard practice (Bendsøe & Sigmund 2003; Sigmund & Maute 2013, *Topology
  optimization approaches*, SMO).
* **Heaviside / projection β-continuation** — gradually sharpen the
  projection/threshold to remove gray and control length scale (Guest et al. 2004;
  Wang, Lazarov & Sigmund 2011). The MMC/MMV literature uses the analogous sharpening
  of the component characteristic function (Zhang, Yuan, Zhang & Guo 2016).
* **Filter-radius and mesh continuation** — start coarse/heavily filtered, refine.

The GGP analogue is **sharpness continuation**: ramp `r_gp` (sampling window),
`ka` (aggregation), `pp` (saturation) and/or `p_penalty` from smooth to sharp.
GGP-Topo's own legacy `free_runner.py` carried an unused `r_gp = [1.5, 1.0, 0.5]`
schedule — exactly this idea.

### 2.2 Multi-start / random restart
Run the local solver from many diverse initial designs and keep the best (best-of-N).
Embarrassingly parallel and a strong baseline for any non-convex problem. For
MMC/GGP it directly targets initial-layout sensitivity: randomizing component
positions, angles and lengths samples distinct basins (Guo et al. 2014; Zhang et al.
2016). The cost is N local solves; the payoff is robust improvement and an empirical
picture of how rugged the landscape actually is.

### 2.3 Stochastic perturbation: basin hopping & simulated annealing
Hybridize local descent with stochastic global moves. **Basin hopping** (Wales &
Doye 1997) alternates a local minimization with a random perturbation of the
*incumbent* optimum, accepting the new basin by a **Metropolis** rule
`exp(-ΔC/T)`. **Simulated annealing** anneals an acceptance temperature. These
explore the *neighborhood of good designs* rather than restarting blindly, and tend
to find improved minima with fewer solves than naive multi-start once a decent
incumbent exists.

### 2.4 Deflation / deflated continuation (recent, principled)
Farrell, Papadopoulos & Surowiec (2021, *Computing multiple solutions of topology
optimization problems*, SIAM J. Sci. Comput.; see also Papadopoulos, Farrell &
Surowiec 2021) systematically compute **distinct** solutions by **deflation**: once a
minimizer `x_i` is found, the objective (or the residual) is multiplied by a
deflation operator

```
M(x) = shift + Σ_i 1 / ‖x − x_i‖^power
```

that becomes singular at each known root, so a re-solve from the *same* start is
repelled from previously found minima and driven into a new basin. Combined with
continuation ("deflated continuation"), this traces disconnected branches of
solutions and reliably surfaces better minima than single-start. It is the most
principled answer to "find me a *different, better* optimum."

### 2.5 Other approaches (lower priority here)
* **Population metaheuristics** (GA, PSO, differential evolution), usually hybridized
  with a gradient local search. Powerful but very expensive at TO scale; best for
  low-dimensional parameterizations.
* **Topological-derivative / bubble methods** — nucleate holes using the topological
  sensitivity (Sokołowski & Żochowski 1999; Allaire et al.). Natural for density
  fields, less so for a fixed set of explicit GGP components.
* **Tunneling / filled-function methods** — analytically deform the objective to
  "tunnel" past a known minimum; related in spirit to deflation.
* **Optimizer-robustness knobs** — GCMMA, adaptive move limits / asymptotes
  (Svanberg 1987, 2002). These reduce premature stagnation but do not, by themselves,
  change the basin the solver lands in.

## 3. What we implemented

All four high-value techniques are implemented as thin orchestration layers over
`GGPPipeline`, in `ggp/optimization/global_search.py`. Two backward-compatible hooks
were added to `GGPPipeline` (`ggp/optimization/pipeline.py`):

* `x0` — inject a normalized `[0,1]` initial design (warm-start / restart / perturbation);
* `overrides` — per-run sharpness/penalization overrides (`ka`, `pp`, `r_gp`,
  `gammac`, `gammav`, `p_penalty`, `Emin`);
* `deflation` — append a `_DeflatedObjectiveDiscipline` implementing `J·M(x)` with
  analytic Jacobian, switching the scenario objective to the deflated one while still
  reporting the *true* compliance at the optimum.

| Technique | Function | Knobs |
|---|---|---|
| Continuation / homotopy | `continuation(spec, schedule)` | warm-started `r_gp`/`ka`/`pp`/`p_penalty` ramp |
| Multi-start | `multi_start(spec, n_starts, seed)` | best-of-N random layouts via `random_initial_design` |
| Basin hopping | `basin_hopping(spec, n_hops, step, temperature)` | Metropolis on compliance |
| Deflation | `deflated_search(spec, n_solutions, shift, power)` | deflation operator over found roots |

Each returns a `GlobalSearchResult` (best `OptimisationResult`, all attempts with
their compliances, wall-clock). The strategies are exposed on the CLI
(`ggp search --strategy ...`) and benchmarked by `benchmarks/sc2d_local_minima.py`.

## 4. Benchmark: Short Cantilever 2D

**Problem.** 60×30 domain (2:1), left edge fixed, unit downward point load at
mid-right, 18 free GP bars, 40% volume fraction, MMA. Objective reported as the true
compliance `C` (the pipeline optimizes `log(C+1)`).

**Protocol.** Single-start **baseline** vs. the four strategies. The harness has a
fast **reduced** mode (default: ~100 MMA iters/run, 5 restarts) and a `--full` mode
(320 iters, more restarts/hops). Run:

```bash
python benchmarks/sc2d_local_minima.py            # reduced (default)
python benchmarks/sc2d_local_minima.py --full     # publication fidelity
```

### 4.1 Results

<!-- BENCHMARK_RESULTS_START -->
_Populated by `benchmarks/sc2d_local_minima.py` (see
`benchmarks/sc2d_local_minima_results.md`)._
<!-- BENCHMARK_RESULTS_END -->

**Figures** (written to `docs/_static/`):

* `sc2d_local_minima_spread.png` — compliance of every individual run per strategy,
  with the per-strategy best and the baseline reference line. The vertical spread
  *is* the local-minima problem made visible.
* `sc2d_local_minima_<method>.png` — the best design found by each strategy.

### 4.2 Reading the results
* **Baseline** is the reference single local minimum.
* **Multi-start** and **deflation** are expected to give the largest improvement,
  because they sample *distinct* basins rather than refining one.
* **Continuation** improves robustness/crispness at low extra cost (a handful of
  warm-started phases) and often matches the baseline basin from a cleaner path.
* **Basin hopping** improves on the incumbent when a good one is cheap to reach.
* Any strategy's best compliance should be **≤** the baseline (it always includes a
  baseline-equivalent run), so the method can never do worse than single-start.

## 5. Reproducing & extending

* Run a single strategy: `ggp search --preset short_cantilever --strategy continuation`.
* Add a new continuation schedule: pass a list of `overrides` dicts to
  `global_search.continuation`.
* New strategies plug in the same way (construct `GGPPipeline(spec, x0=..., overrides=...,
  deflation=...)`, read back `OptimisationResult`); the orchestration is FEniCS-free and
  unit-tested in `tests/test_global_search.py`.

## References
1. M. P. Bendsøe, O. Sigmund. *Topology Optimization: Theory, Methods, and
   Applications.* Springer, 2003.
2. O. Sigmund, K. Maute. *Topology optimization approaches.* Struct. Multidiscip.
   Optim. 48(6):1031–1055, 2013.
3. J. K. Guest, J. H. Prévost, T. Belytschko. *Achieving minimum length scale … using
   nodal design variables and projection functions.* IJNME 61(2):238–254, 2004.
4. F. Wang, B. S. Lazarov, O. Sigmund. *On projection methods, convergence and robust
   formulations in topology optimization.* SMO 43(6):767–784, 2011.
5. X. Guo, W. Zhang, W. Zhong. *Doing topology optimization explicitly and
   geometrically — a new moving morphable components based framework.* J. Appl. Mech.
   81(8), 2014.
6. W. Zhang, J. Yuan, J. Zhang, X. Guo. *A new topology optimization approach based on
   Moving Morphable Components (MMC) and the ersatz material model.* SMO 53:1243–1260,
   2016.
7. S. Coniglio, J. Morlier, C. Gogu, R. Amargier. *Generalized Geometry Projection: A
   unified approach for geometric feature based topology optimization.* Arch. Comput.
   Methods Eng. 27:1573–1610, 2020.
8. D. P. Wales, J. P. K. Doye. *Global optimization by basin-hopping …* J. Phys. Chem.
   A 101(28):5111–5116, 1997.
9. I. P. A. Papadopoulos, P. E. Farrell, T. M. Surowiec. *Computing multiple solutions
   of topology optimization problems.* SIAM J. Sci. Comput. 43(3):A1555–A1582, 2021.
10. K. Svanberg. *The method of moving asymptotes — a new method for structural
    optimization.* IJNME 24(2):359–373, 1987.
