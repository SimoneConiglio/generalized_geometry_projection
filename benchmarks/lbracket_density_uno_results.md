# Uno on the density-based L-bracket with local stress constraints

Mass minimization, one relaxed von Mises constraint per element (no aggregation):
n design variables = m constraints, dense m x n Jacobian. Start: full material.
`uno s/it` is Uno's own time per iteration (FE + sensitivity callbacks excluded);
`FE s/jac` is the cost of one dense adjoint Jacobian (n back-substitutions).
SQP/IPM presets minimize n*V (the O(1/n) volume gradient otherwise makes the
identity-initialised L-BFGS step O(1/n)); `filterslp` minimizes V.

Regenerate with `python -m benchmarks.lbracket_density_uno --report`.

| n = m | Jacobian nnz | Preset | Uno options | Status | Iter. | Volume | max g | Wall (s) | Uno (s) | uno s/it | FE+sens (s) | FE s/jac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 9.0e+04 | filterslp | - | SUCCESS | 53 | 0.3136 | +1.1e-07 | 7 | 4 | 0.085 | 2 | 0.017 |
| 300 | 9.0e+04 | filtersqp | - | SUCCESS | 33 | 0.3243 | +1.1e-13 | 2 | 1 | 0.041 | 1 | 0.022 |
| 300 | 9.0e+04 | funnelsqp | - | SUCCESS | 31 | 0.3149 | +3.5e-13 | 3 | 2 | 0.076 | 1 | 0.021 |
| 300 | 9.0e+04 | ipopt | - | SUCCESS | 91 | 0.3008 | +9.7e-09 | 5 | 3 | 0.031 | 3 | 0.017 |
| 1200 | 1.4e+06 | filterslp | - | SUCCESS | 59 | 0.3795 | +1.4e-07 | 430 | 370 | 6.265 | 60 | 0.488 |
| 1200 | 1.4e+06 | filtersqp | - | SUCCESS | 80 | 0.3775 | +1.1e-12 | 105 | 61 | 0.764 | 43 | 0.478 |
| 1200 | 1.4e+06 | funnelsqp | - | SUCCESS | 149 | 0.3962 | +1.2e-12 | 259 | 148 | 0.992 | 110 | 0.486 |
| 1200 | 1.4e+06 | ipopt | - | SUCCESS | 133 | 0.3821 | +3.0e-08 | 327 | 202 | 1.516 | 124 | 0.502 |
| 2700 | 7.3e+06 | filterslp | - | SUCCESS | 78 | 0.4129 | +4.1e-07 | 2256 | 1733 | 22.222 | 511 | 3.136 |
| 2700 | 7.3e+06 | filtersqp | {"TR_radius": 0.05, "TR_increase_factor": 1.0} | ITERATION_LIMIT | 300 | 0.4035 | +8.1e-05 | 2381 | 1365 | 4.549 | 996 | 3.308 |
| 2700 | 7.3e+06 | ipopt | - | SUCCESS | 152 | 0.4057 | +5.7e-08 | 3527 | 2911 | 19.151 | 603 | 3.465 |

## Findings (n = m = 300, 1 200, 2 700)

* All presets reach feasible KKT points with the same truss-like topology up to
  n = m = 1 200; at 2 700 `ipopt` and `filterslp` still converge (59 and 38 min).
* Uno's own cost per iteration grows ~n^2 for `filtersqp`/`funnelsqp` (BQPD
  warm-started active-set QP) and ~n^3 for `ipopt` (factorization of a KKT system
  filled in by the dense m x n Jacobian) and `filterslp`: 19-22 s/it at n = 2 700,
  i.e. Uno, not the FE model, dominates the run time from n ~ 1 000 on.
* `filtersqp` with Uno's default trust region (radius 10) jumps to a hugely
  infeasible point at n = 2 700 and its feasibility-restoration QP (BQPD) did not
  return in > 1 h. A capped radius (0.05, as for the GGP cantilever) avoids it:
  4.5 s/it, but it needs > 300 iterations (V = 0.4035, max g = 8e-5 at the cap).
* The optimal volume grows with refinement (0.30 -> 0.41): the re-entrant corner
  stress is singular, so a finer mesh needs more material around it.
* Extrapolating, n = m ~ 10^4 costs ~2-20 min per iteration and ~1 GB per dense
  Jacobian copy: local constraints without aggregation are out of reach for these
  dense-Jacobian NLP methods beyond a few thousand elements.
